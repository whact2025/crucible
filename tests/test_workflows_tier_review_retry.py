# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Review-feedback retry loop in ``crucible.workflows.tier``.

When a GATING review rubric (e.g. ``tier_completeness``) fails, the tier
re-implements its slice(s) with the reviewer's findings as feedback and
re-reviews — up to ``ReviewPolicy.retry_budget`` times — instead of
abandoning the tier (``rc=3``) on the first failure. Mirrors the
build-feedback retry for the build/verify gate.

Behavioral e2e: a stub backend that (a) makes ``assay.implement`` succeed
by writing the in-scope file, and (b) returns a FAIL verdict on the first
``tier_completeness`` review and a PASS verdict on the second. With
``retry_budget=1`` the tier must recover to ``rc=0``; with ``retry_budget=0``
the same fail-then-(would-pass) backend must abandon at ``rc=3`` after the
first review.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ContentHash,
    Evidence,
    ExecutionCompleteness,
    Finding,
    Implementation,
    Invocation,
    InvocationResult,
    Message,
    ProjectContext,
    ReviewCoverage,
    Spec,
    Usage,
    Verdict,
)
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG

from crucible.common.policies import BreachPolicy, ReviewPolicy
from crucible.workflows._build_verify import BuildVerifyOutcome, RunnerResult
from crucible.workflows._development_budget import (
    DevelopmentBudgetLedger,
    DevelopmentBudgetPolicy,
    RoleAllocation,
)
from crucible.workflows.tier import (
    _DEFAULT_TIER_RESPONSE_RETRY_BUDGET,
    _build_review_conflict_authority_artifact,
    _build_review_feedback_text,
    _render_review_conflict_authority_artifact,
    _render_tier_completeness_artifact,
    _review_conflict_gate_diagnostic,
    _review_conflict_index,
    _review_conflict_runtime_block,
    _validate_review_conflict_authority_verdict,
)


_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_TRACE = BackendExecutionTrace(
    backend="stub",
    requested_model="stub-model",
    events=(
        BackendExecutionEvent(
            sequence=0,
            kind="terminal",
            name="end_turn",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------


class _FakeFinding:
    def __init__(self, finding_id, severity, suggested_action):
        self.id = finding_id
        self.criterion_id = "slice_delivery"
        self.category = "partial_delivery"
        self.severity = severity
        self.suggested_action = suggested_action
        self.targets = ()
        self.evidence = ()


def test_build_review_feedback_text_renders_actionable_findings():
    gating = [
        ("tier_completeness", (
            _FakeFinding(
                "normalized_values_missing", "High",
                "Wire normalized values into the schema.",
            ),
            _FakeFinding(
                "aliases_not_consumed", "High",
                "Consume aliases in the loader codegen.",
            ),
        )),
        ("diff_scope", (_FakeFinding(
            "scope_ok", "Medium", "Edit stays in scope.",
        ),)),
    ]
    payload = json.loads(_build_review_feedback_text(gating))
    assert payload["schema_version"] == "crucible.repair_envelope.v1"
    assert payload["omitted_findings"] == 0
    assert [item["finding_id"] for item in payload["findings"]] == [
        "normalized_values_missing", "aliases_not_consumed", "scope_ok",
    ]
    rendered = json.dumps(payload)
    assert "Wire normalized values" not in rendered
    assert "Consume aliases" not in rendered


def test_build_review_feedback_text_empty():
    payload = json.loads(_build_review_feedback_text([]))
    assert payload["findings"] == []
    assert payload["omitted_findings"] == 0


def _conflict_fixture(
    *,
    rubric_name: str = "tier_completeness",
    category: str = "partial_delivery",
    clause: str = "Remove PublicSymbol from export surface.",
    requirement_refs: tuple[str, ...] = (),
):
    slice_ = Slice(
        id="s0",
        directive=ImplementationDirective(
            slice_id="s0",
            requirement_refs=requirement_refs,
            intent=f"Implement behavior. {clause} Preserve compatibility.",
            files_in_scope=("file_0.py",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        ),
    )
    tier = Tier(tier_id="t0", slices=(slice_,), depends_on=())
    original = Finding(
        id="remove_public_symbol",
        severity="Medium",
        category=category,
        evidence=(Evidence(
            source_id="artifact",
            kind="slice_directive",
            location="s0",
            excerpt=clause,
        ),),
        suggested_action="generated reviewer prose must not be authority",
        source="llm",
        criterion_id=(
            "declared_scope"
            if rubric_name == "diff_scope" else "slice_delivery"
        ),
        confidence=0.99,
        disposition="confirmed",
    )
    indexed = _review_conflict_index([
        (rubric_name, (original,)),
    ])
    candidate = b"diff --git a/file_0.py b/file_0.py\n+PublicSymbol\n"
    retry = b"diff --git a/file_0.py b/file_0.py\n-PublicSymbol\n"
    retry_sha = hashlib.sha256(retry).hexdigest()
    gate = {
        "slice_id": "s0",
        "classification": "verify_failed",
        "retry_sha256": retry_sha,
        "diagnostic": {
            "role": "verify",
            "runner": "tests.runner",
            "returncode": 1,
            "classification": "verify_failed",
            "stdout": "ImportError: PublicSymbol is required",
            "stderr": "",
            "values": {},
        },
    }
    artifact, digest, packet = _build_review_conflict_authority_artifact(
        issue_text="Change behavior while preserving the public API.",
        tier=tier,
        candidate_diff=candidate,
        retry_diffs=(retry,),
        gate_rejections=(gate,),
        indexed_findings=indexed,
    )
    return clause, artifact, digest, packet


def _conflict_verdict(
    packet,
    *,
    relation: str = "verification_conflict",
    extra_evidence: tuple[Evidence, ...] = (),
    include_finding: bool = True,
    gate_excerpt: str = "ImportError: PublicSymbol is required",
    slice_location: str = "slice:s0",
) -> Verdict:
    authority_id = packet["review_findings"][0]["authority_id"]
    clause = packet["review_findings"][0]["finding"]["evidence"][0][
        "excerpt"
    ]
    evidence = (
        Evidence(
            source_id="artifact",
            kind="review_finding",
            location=f"finding:{authority_id}",
            excerpt=clause,
        ),
        Evidence(
            source_id="artifact",
            kind="slice_directive",
            location=slice_location,
            excerpt=clause,
        ),
        Evidence(
            source_id="artifact",
            kind="gate_diagnostic",
            location="gate:0",
            excerpt=gate_excerpt,
        ),
        *extra_evidence,
    )
    finding = Finding(
        id=authority_id,
        severity="Low",
        category=relation,
        evidence=evidence,
        suggested_action=None,
        source="llm",
        criterion_id="review_conflict_authority",
        confidence=0.99,
        disposition=(
            "dismissed" if relation == "verification_conflict"
            else "addressed"
        ),
    )
    return Verdict(
        rubric="review_conflict_authority",
        decision="pass",
        findings=((finding,) if include_finding else ()),
        artifact_ref=ContentHash("sha256:" + "a" * 64),
        artifact_commit=None,
        reasoning="bounded authority classification",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
        ),
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
    )


def test_review_conflict_authority_exact_closure_admits_bound_gate_conflict():
    clause, artifact, digest, packet = _conflict_fixture()

    assert digest.startswith("sha256:")
    assert clause.encode() in artifact
    assert b"generated reviewer prose" not in artifact
    outcome = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(packet),
    )

    assert outcome["status"] == "admitted"
    assert outcome["errors"] == []


def test_review_conflict_packet_and_runtime_preserve_root_authority():
    requirement = "PublicSymbol must remain importable from the package root."
    _clause, _artifact, digest, packet = _conflict_fixture(
        requirement_refs=(requirement,),
    )

    packet_slice = packet["slices"][0]
    assert packet["schema_version"] == (
        "crucible.review-conflict-authority.v2"
    )
    assert packet_slice["requirement_refs"] == [requirement]
    assert "Remove PublicSymbol" in packet_slice["generated_intent"]
    assert "intent" not in packet_slice

    outcome = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(packet),
    )
    assert outcome["status"] == "admitted"

    block = _review_conflict_runtime_block(
        digest=digest,
        records=outcome["records"],
        packet=packet,
    )
    runtime = json.loads(block.rsplit("\n", 1)[1])
    assert runtime["schema_version"] == (
        "crucible.review-conflict-runtime-authority.v2"
    )
    assert runtime["authoritative_root_requirements"] == [{
        "slice_id": "s0",
        "requirement_refs": [requirement],
    }]
    assert "Generated implementation intent is non-authoritative" in block


def test_tier_completeness_artifact_keeps_root_and_generated_intent_separate():
    requirement = "Keep leading whitespace exactly:\n  PublicSymbol stays."
    slice_ = Slice(
        id="s0",
        directive=ImplementationDirective(
            slice_id="s0",
            requirement_refs=(requirement,),
            intent="Generated planner note: remove PublicSymbol.",
            files_in_scope=("file_0.py",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        ),
    )
    artifact = _render_tier_completeness_artifact(
        Tier(tier_id="t0", slices=(slice_,), depends_on=()),
        b"diff --git a/file_0.py b/file_0.py\n",
    ).decode("utf-8")

    assert requirement in artifact
    assert "authoritative requirement_ref[0]" in artifact
    assert "generated intent (non-authoritative" in artifact
    assert "authoritative current-slice allocation" in artifact
    assert "**creates_paths:** none (verification-only)" not in artifact
    assert "**creates_paths:** none\n" in artifact
    assert artifact.index(requirement) < artifact.index(
        "Generated planner note"
    )


def test_review_conflict_authority_never_dismisses_root_clause_as_generated():
    clause, _artifact, _digest, packet = _conflict_fixture(
        requirement_refs=("Remove PublicSymbol from export surface.",),
    )
    assert clause == packet["slices"][0]["requirement_refs"][0]

    outcome = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(packet),
    )

    assert outcome["status"] == "inconclusive"
    assert any(
        "duplicates an authoritative requirement_ref" in item
        for item in outcome["errors"]
    )


def test_review_conflict_actionable_generated_prose_requires_root_binding():
    requirement = "PublicSymbol must remain importable from the package root."
    _clause, _artifact, _digest, packet = _conflict_fixture(
        clause="Keep PublicSymbol available from export surface.",
        requirement_refs=(requirement,),
    )

    unbound = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(packet, relation="actionable"),
    )
    assert unbound["status"] == "inconclusive"
    assert any(
        "same-slice requirement binding" in item
        for item in unbound["errors"]
    )

    packet["review_findings"][0]["finding"]["evidence"].append({
        "source_id": "artifact",
        "location": "requirement_ref[0]",
        "excerpt": requirement,
    })
    root_binding = Evidence(
        source_id="artifact",
        kind="root_requirement",
        location="requirement:s0:0",
        excerpt=requirement,
    )
    bound = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(
            packet,
            relation="actionable",
            extra_evidence=(root_binding,),
        ),
    )
    assert bound["status"] == "admitted"
    assert bound["errors"] == []
    assert bound["records"][0]["requirement_bindings"] == [{
        "slice_id": "s0",
        "requirement_index": 0,
        "requirement_ref": requirement,
    }]


def test_review_conflict_authority_admits_live_logical_location_shape():
    clause, _artifact, _digest, packet = _conflict_fixture()
    verification_entry = packet["review_findings"][0]
    verification_entry["authority_id"] = "review_conflict_001"
    scope_payload = {
        "finding_id": "SCOPE-001",
        "criterion_id": "incidental_changes",
        "category": "incidental_change",
        "severity": "Medium",
        "targets": [],
        "evidence": [{
            "source_id": "artifact",
            "location": "file_0.py import hunk",
            "excerpt": "+PublicSymbol",
        }],
    }
    packet["review_findings"].insert(0, {
        "authority_id": "review_conflict_000",
        "rubric": "diff_scope",
        "finding": scope_payload,
    })
    artifact, _digest = _render_review_conflict_authority_artifact(packet)
    artifact_text = artifact.decode("utf-8")
    clause_start = artifact_text.rindex(clause)
    candidate_excerpt = json.dumps(
        packet["candidate"]["diff"], ensure_ascii=False,
    )[1:-1]
    retry_excerpt = json.dumps(
        packet["rejected_retries"][0]["diff"], ensure_ascii=False,
    )[1:-1]

    scope_finding = Finding(
        id="review_conflict_000",
        severity="Medium",
        category="actionable",
        evidence=(
            Evidence(
                source_id="artifact",
                kind="review_finding",
                location="finding:review_conflict_000",
                excerpt='"category": "incidental_change"',
            ),
            Evidence(
                source_id="artifact",
                kind="candidate_diff",
                location="candidate.diff",
                excerpt=candidate_excerpt,
            ),
        ),
        suggested_action=None,
        source="llm",
        criterion_id="review_conflict_authority",
        confidence=0.99,
        disposition="addressed",
    )
    verification_finding = Finding(
        id="review_conflict_001",
        severity="Medium",
        category="verification_conflict",
        evidence=(
            Evidence(
                source_id="artifact",
                kind="review_finding",
                location="finding:review_conflict_001",
                excerpt='"finding_id": "remove_public_symbol"',
            ),
            Evidence(
                source_id="artifact",
                kind="slice_directive",
                location=(
                    "slice:s0;chars:"
                    f"{clause_start}-{clause_start + len(clause)}"
                ),
                excerpt=clause,
            ),
            Evidence(
                source_id="artifact",
                kind="rejected_retry_diff",
                location="rejected_retry:0",
                excerpt=retry_excerpt,
            ),
            Evidence(
                source_id="artifact",
                kind="gate_diagnostic",
                location="gate:0",
                excerpt="ImportError: PublicSymbol is required",
            ),
        ),
        suggested_action=None,
        source="llm",
        criterion_id="review_conflict_authority",
        confidence=0.99,
        disposition="dismissed",
    )
    verdict = Verdict(
        rubric="review_conflict_authority",
        decision="pass",
        findings=(scope_finding, verification_finding),
        artifact_ref=ContentHash("sha256:" + "a" * 64),
        artifact_commit=None,
        reasoning="bounded authority classification",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
        ),
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
    )

    outcome = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=verdict,
    )

    assert outcome["status"] == "admitted"
    assert outcome["errors"] == []
    assert [record["classification"] for record in outcome["records"]] == [
        "actionable",
        "verification_conflict",
    ]


def test_review_conflict_authority_rejects_unbound_supplemental_locations():
    _clause, _artifact, _digest, packet = _conflict_fixture()
    cases = (
        Evidence(
            source_id="artifact",
            kind="candidate_diff",
            location="candidate.diff",
            excerpt="Change behavior while preserving the public API.",
        ),
        Evidence(
            source_id="artifact",
            kind="rejected_retry_diff",
            location="rejected_retry:1",
            excerpt="-PublicSymbol",
        ),
        Evidence(
            source_id="artifact",
            kind="rejected_retry_diff",
            location="rejected_retry:0",
            excerpt="+PublicSymbol",
        ),
    )

    for evidence in cases:
        outcome = _validate_review_conflict_authority_verdict(
            packet=packet,
            verdict=_conflict_verdict(packet, extra_evidence=(evidence,)),
        )
        assert outcome["status"] == "inconclusive"
        assert outcome["errors"]


def test_review_conflict_authority_rejects_malformed_or_wrong_logical_span():
    clause, artifact, _digest, packet = _conflict_fixture()
    artifact_text = artifact.decode("utf-8")
    start = artifact_text.rindex(clause)
    locations = (
        "slice:s0;chars:-1-2",
        "slice:s0;chars:2-2",
        "slice:s0;chars:2-1",
        "slice:s0;chars:2-3;junk",
        f"slice:missing;chars:{start}-{start + len(clause)}",
        f"slice:s0;chars:{start + 1}-{start + len(clause) + 1}",
    )

    for location in locations:
        outcome = _validate_review_conflict_authority_verdict(
            packet=packet,
            verdict=_conflict_verdict(packet, slice_location=location),
        )
        assert outcome["status"] == "inconclusive"
        assert outcome["errors"]


def test_review_conflict_authority_input_error_and_missing_id_fail_closed():
    _clause, _artifact, _digest, packet = _conflict_fixture()
    packet["input_errors"] = ["non-UTF8 evidence"]

    outcome = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(packet, include_finding=False),
    )

    assert outcome["status"] == "inconclusive"
    assert "non-UTF8 evidence" in outcome["errors"]
    assert any("omitted ids" in item for item in outcome["errors"])


def test_review_conflict_authority_never_dismisses_scope_safety():
    _clause, _artifact, _digest, packet = _conflict_fixture(
        rubric_name="diff_scope",
        category="incidental_change",
    )

    outcome = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(packet),
    )

    assert outcome["status"] == "inconclusive"
    assert any("cannot dismiss scope-safety" in item for item in outcome["errors"])


def test_review_conflict_authority_rejects_extra_same_slice_clause():
    _clause, _artifact, _digest, packet = _conflict_fixture()
    extra = Evidence(
        source_id="artifact",
        kind="slice_directive",
        location="slice:s0",
        excerpt="Preserve compatibility.",
    )

    outcome = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(packet, extra_evidence=(extra,)),
    )

    assert outcome["status"] == "inconclusive"
    assert any("exactly one slice-clause" in item for item in outcome["errors"])


def test_review_conflict_authority_rejects_whole_directive_waiver():
    _clause, _artifact, _digest, packet = _conflict_fixture()
    full_intent = packet["slices"][0]["generated_intent"]
    packet["review_findings"][0]["finding"]["evidence"][0][
        "excerpt"
    ] = full_intent

    outcome = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(packet),
    )

    assert outcome["status"] == "inconclusive"
    assert any("narrow directive sub-clause" in item for item in outcome["errors"])


def test_review_conflict_authority_rejects_unrelated_same_file_failure():
    _clause, _artifact, _digest, packet = _conflict_fixture()
    unrelated = "ImportError while loading file_0.py"
    packet["gate_rejections"][0]["diagnostic"]["stdout"] = unrelated

    outcome = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(packet, gate_excerpt=unrelated),
    )

    assert outcome["status"] == "inconclusive"
    assert any("no code-subject binding" in item for item in outcome["errors"])


def test_review_conflict_authority_rejects_cross_slice_gate_binding():
    _clause, _artifact, _digest, packet = _conflict_fixture()
    packet["slices"].append({
        "slice_id": "s1",
        "requirement_refs": [],
        "generated_intent": "Remove PublicSymbol from export surface.",
        "files_in_scope": ["file_0.py"],
        "creates_paths": [],
    })
    packet["gate_rejections"][0]["slice_id"] = "s1"

    outcome = _validate_review_conflict_authority_verdict(
        packet=packet,
        verdict=_conflict_verdict(packet),
    )

    assert outcome["status"] == "inconclusive"
    assert any("cross-wires" in item for item in outcome["errors"])


def test_review_conflict_gate_diagnostic_selects_exact_failed_runner():
    passed_dev = RunnerResult(
        role="dev_verify", runner="dev.runner", returncode=0,
        values={}, raw_output=b"dev passed", stderr=b"", duration_s=0.1,
    )
    failed_verify = RunnerResult(
        role="verify", runner="verify.runner", returncode=1,
        values={}, raw_output=b"PublicSymbol failed", stderr=b"", duration_s=0.1,
    )
    outcome = BuildVerifyOutcome(
        "verify_failed",
        build=None,
        verify=failed_verify,
        dev_verify=passed_dev,
    )

    diagnostic = _review_conflict_gate_diagnostic(outcome)

    assert diagnostic is not None
    assert diagnostic["role"] == "verify"
    assert diagnostic["runner"] == "verify.runner"


# ---------------------------------------------------------------------------
# Policy defaults
# ---------------------------------------------------------------------------


def test_review_policy_retry_budget_defaults_off():
    """Default OFF preserves prior behavior for every product that
    doesn't opt in (review failure stays immediately terminal)."""
    assert ReviewPolicy().retry_budget == 0


def test_deepswe_policy_opts_into_review_retry():
    from crucible.products.deepswe.spec_synthesis import DEEPSWE_REVIEW_POLICY
    assert DEEPSWE_REVIEW_POLICY.retry_budget == 2


# ---------------------------------------------------------------------------
# Behavioral e2e
# ---------------------------------------------------------------------------


_FAIL_VERDICT = json.dumps({
    "schema_version": "assay.review.v2",
    "decision": "revise",
    "reasoning": "Slice s0 carries the data but never consumes it.",
    "coverage": {
        "complete": True,
        "sources_reviewed": ["artifact"],
        "omissions": [],
    },
    "findings": [{
        "id": "missing_wiring",
        "criterion_id": "slice_delivery",
        "severity": "High",
        "confidence": 0.95,
        "disposition": "confirmed",
        "category": "partial_delivery",
        "targets": [],
        "evidence": [{
            "source_id": "artifact",
            "kind": "diff",
            "location": "file_0.py",
            "excerpt": "# implement attempt 1",
        }],
        "proposal": {
            "action": "Wire the carried value into the consumer.",
            "replacement": None,
        },
        "attack_spec": None,
    }],
})

_PASS_VERDICT = json.dumps({
    "schema_version": "assay.review.v2",
    "decision": "pass",
    "reasoning": "The wiring is now present; nothing material is missing.",
    "coverage": {
        "complete": True,
        "sources_reviewed": ["artifact"],
        "omissions": [],
    },
    "findings": [],
})


class _FailThenPassBackend:
    """Writes the in-scope file (so assay.implement → success) on every
    invoke, and returns a review verdict that FAILS on the first review
    and PASSES on the second. Implement vs review is discriminated by the
    rubric markers assay.review puts in the invocation prompt."""

    name = "stub-fail-then-pass"
    model = "stub-model"

    def __init__(self, repo: Path):
        self.repo = repo
        self.implement_calls = 0
        self.review_calls = 0

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    @staticmethod
    def _is_review(invocation) -> bool:
        system = getattr(invocation, "system", "") or ""
        # Review feedback deliberately contains a JSON ``findings`` field, so
        # classify by the system-level wire contract, never untrusted user
        # content threaded into a retry directive.
        return "assay.review.v2" in system

    def invoke(self, invocation) -> InvocationResult:
        if self._is_review(invocation):
            self.review_calls += 1
            msg = _FAIL_VERDICT if self.review_calls == 1 else _PASS_VERDICT
            return InvocationResult(
                final_message=msg, transcript=(),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        # Implement turn: make a real, fresh change so the diff is
        # non-empty and assay.implement classifies success.
        self.implement_calls += 1
        f = self.repo / "file_0.py"
        prior = f.read_text() if f.exists() else ""
        f.write_text(prior + f"# implement attempt {self.implement_calls}\n")
        return InvocationResult(
            final_message="done", transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _MalformedReviewBackend(_FailThenPassBackend):
    """Always emits a repairable schema failure for reviewer turns.

    The implementer path remains successful so the test can distinguish
    response-contract repair (another reviewer call) from semantic repair
    (another implementer call).
    """

    def invoke(self, invocation) -> InvocationResult:
        if self._is_review(invocation):
            self.review_calls += 1
            return InvocationResult(
                final_message=json.dumps({
                    "schema_version": "assay.review.v2",
                }),
                transcript=(),
                usage=Usage(
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.01,
                ),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        return super().invoke(invocation)


class _AlwaysReviseBackend(_FailThenPassBackend):
    """Produces a successful retry that the second review still rejects."""

    def invoke(self, invocation) -> InvocationResult:
        if self._is_review(invocation):
            self.review_calls += 1
            return InvocationResult(
                final_message=_FAIL_VERDICT,
                transcript=(),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        return super().invoke(invocation)


class _BlockedReviewRetryBackend(_FailThenPassBackend):
    """Reports a terminal generic blocker on the first semantic repair."""

    def invoke(self, invocation) -> InvocationResult:
        if self._is_review(invocation):
            return super().invoke(invocation)
        self.implement_calls += 1
        target = self.repo / "file_0.py"
        prior = target.read_text() if target.exists() else ""
        if self.implement_calls == 1:
            target.write_text(prior + "# implement attempt 1\n")
            return InvocationResult(
                final_message="DONE initial candidate",
                transcript=(),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        target.write_text(prior + "# partial blocked retry\n")
        message = "BLOCKED: review repair conflicts with a required contract"
        return InvocationResult(
            final_message=message,
            transcript=(Message(role="assistant", content=message),),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _AccountingDebtReviewRetryBackend(_FailThenPassBackend):
    """Makes only the paid semantic repair's accounting incomplete."""

    def invoke(self, invocation) -> InvocationResult:
        result = super().invoke(invocation)
        if not self._is_review(invocation) and self.implement_calls == 2:
            return replace(
                result,
                completeness=ExecutionCompleteness(
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=True,
                    trace_complete=True,
                ),
            )
        return result


class _ConflictResolutionBackend(_FailThenPassBackend):
    """Exercises gate-rejected rollback -> authority -> reconciliation."""

    def __init__(
        self,
        repo: Path,
        *,
        malformed_authority: bool = False,
        block_reconciliation: bool = False,
    ):
        super().__init__(repo)
        self.malformed_authority = malformed_authority
        self.block_reconciliation = block_reconciliation
        self.diff_scope_calls = 0
        self.completeness_calls = 0
        self.authority_calls = 0

    @staticmethod
    def _result(message: object) -> InvocationResult:
        return InvocationResult(
            final_message=(
                json.dumps(message) if isinstance(message, dict) else str(message)
            ),
            transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )

    @staticmethod
    def _pass(reason: str) -> dict:
        return {
            "schema_version": "assay.review.v2",
            "decision": "pass",
            "reasoning": reason,
            "coverage": {
                "complete": True,
                "sources_reviewed": ["artifact"],
                "omissions": [],
            },
            "findings": [],
        }

    def invoke(self, invocation) -> InvocationResult:
        system = getattr(invocation, "system", "") or ""
        normalized_system = " ".join(system.split())
        if "independent authority adjudicator" in normalized_system:
            self.authority_calls += 1
            if self.malformed_authority:
                return self._result({"schema_version": "assay.review.v2"})
            return self._result({
                "schema_version": "assay.review.v2",
                "decision": "pass",
                "reasoning": (
                    "The scope finding remains actionable; removing the "
                    "public symbol is contradicted by executable evidence."
                ),
                "coverage": {
                    "complete": True,
                    "sources_reviewed": ["artifact"],
                    "omissions": [],
                },
                "findings": [
                    {
                        "id": "review_conflict_000",
                        "criterion_id": "review_conflict_authority",
                        "severity": "Low",
                        "confidence": 0.99,
                        "disposition": "addressed",
                        "category": "actionable",
                        "targets": [],
                        "evidence": [{
                            "source_id": "artifact",
                            "kind": "review_finding",
                            "location": "finding:review_conflict_000",
                            "excerpt": "# candidate PublicSymbol reordered",
                        }],
                        "proposal": None,
                        "attack_spec": None,
                    },
                    {
                        "id": "review_conflict_001",
                        "criterion_id": "review_conflict_authority",
                        "severity": "Low",
                        "confidence": 0.99,
                        "disposition": "dismissed",
                        "category": "verification_conflict",
                        "targets": [],
                        "evidence": [
                            {
                                "source_id": "artifact",
                                "kind": "review_finding",
                                "location": "finding:review_conflict_001",
                                "excerpt": (
                                    "Remove PublicSymbol from export surface."
                                ),
                            },
                            {
                                "source_id": "artifact",
                                "kind": "slice_directive",
                                "location": "slice:s0",
                                "excerpt": (
                                    "Remove PublicSymbol from export surface."
                                ),
                            },
                            {
                                "source_id": "artifact",
                                "kind": "gate_diagnostic",
                                "location": "gate:0",
                                "excerpt": (
                                    "ImportError: PublicSymbol is required"
                                ),
                            },
                        ],
                        "proposal": None,
                        "attack_spec": None,
                    },
                ],
            })
        if "adversarial reviewer of implementer diffs" in normalized_system:
            self.review_calls += 1
            self.diff_scope_calls += 1
            if self.diff_scope_calls > 1:
                message = self._pass("The reconciled diff has no drift.")
                message["coverage"]["sources_reviewed"] = [
                    "artifact",
                    "context:files_in_scope",
                    "context:creates_paths",
                    "context:dependency_provided_files",
                ]
                return self._result(message)
            return self._result({
                "schema_version": "assay.review.v2",
                "decision": "revise",
                "reasoning": "The candidate contains incidental symbol churn.",
                "coverage": {
                    "complete": True,
                    "sources_reviewed": [
                        "artifact",
                        "context:files_in_scope",
                        "context:creates_paths",
                        "context:dependency_provided_files",
                    ],
                    "omissions": [],
                },
                "findings": [{
                    "id": "public-symbol-reorder",
                    "criterion_id": "incidental_changes",
                    "severity": "Medium",
                    "confidence": 0.99,
                    "disposition": "confirmed",
                    "category": "incidental_change",
                    "targets": [],
                    "evidence": [{
                        "source_id": "artifact",
                        "kind": "diff_change",
                        "location": "file_0.py",
                        "excerpt": "# candidate PublicSymbol reordered",
                    }],
                    "proposal": {
                        "action": "Remove only the incidental reorder.",
                        "replacement": None,
                    },
                    "attack_spec": None,
                }],
            })
        if "tier-level implementation completeness" in normalized_system:
            self.review_calls += 1
            self.completeness_calls += 1
            if self.completeness_calls > 1:
                return self._result(self._pass(
                    "The bounded directive exception and remaining intent pass."
                ))
            return self._result({
                "schema_version": "assay.review.v2",
                "decision": "revise",
                "reasoning": "The directive asks to remove the public symbol.",
                "coverage": {
                    "complete": True,
                    "sources_reviewed": ["artifact"],
                    "omissions": [],
                },
                "findings": [{
                    "id": "remove-public-symbol",
                    "criterion_id": "slice_delivery",
                    "severity": "Medium",
                    "confidence": 0.99,
                    "disposition": "confirmed",
                    "category": "partial_delivery",
                    "targets": [],
                    "evidence": [{
                        "source_id": "artifact",
                        "kind": "slice_directive",
                        "location": "s0",
                        "excerpt": "Remove PublicSymbol from export surface.",
                    }],
                    "proposal": {
                        "action": "Remove PublicSymbol.",
                        "replacement": None,
                    },
                    "attack_spec": None,
                }],
            })
        self.implement_calls += 1
        if self.block_reconciliation and self.implement_calls == 3:
            message = "BLOCKED: reconciliation conflicts with a root contract"
            return InvocationResult(
                final_message=message,
                transcript=(Message(role="assistant", content=message),),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        content = {
            1: "# candidate PublicSymbol reordered\n",
            2: "# retry removed PublicSymbol\n",
            3: "# reconciled PublicSymbol stable\n",
        }[self.implement_calls]
        (self.repo / "file_0.py").write_text(content)
        return self._result("implemented")


class _AccountingDebtConflictResolutionBackend(_ConflictResolutionBackend):
    """Crosses the cap while making reconciliation accounting incomplete."""

    def invoke(self, invocation) -> InvocationResult:
        result = super().invoke(invocation)
        if not self._is_review(invocation) and self.implement_calls == 3:
            return replace(
                result,
                usage=Usage(
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.2,
                ),
                completeness=ExecutionCompleteness(
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=True,
                    trace_complete=True,
                ),
            )
        return result


class _BudgetConflictResolutionBackend(_ConflictResolutionBackend):
    """Crosses the cap only on the conflict-reconciliation call."""

    def invoke(self, invocation) -> InvocationResult:
        result = super().invoke(invocation)
        if not self._is_review(invocation) and self.implement_calls == 3:
            return replace(
                result,
                usage=Usage(
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.2,
                ),
            )
        return result


class _TwoSliceFailThenPassBackend(_FailThenPassBackend):
    """Writes alternating slice files so two-slice retries stay in scope."""

    def invoke(self, invocation) -> InvocationResult:
        if self._is_review(invocation):
            return super().invoke(invocation)
        self.implement_calls += 1
        slice_index = (self.implement_calls - 1) % 2
        target = self.repo / f"file_{slice_index}.py"
        prior = target.read_text() if target.exists() else ""
        target.write_text(
            prior + f"# implement attempt {self.implement_calls}\n"
        )
        return InvocationResult(
            final_message="done",
            transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


def _init_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "README.md").write_text("seed\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _spec_single_slice(base_commit: str) -> Spec:
    sl = Slice(
        id="s0",
        directive=ImplementationDirective(
            slice_id="s0",
            intent="slice zero work — carry a value through",
            files_in_scope=("file_0.py",),
            creates_paths=("file_0.py",),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    return Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="tier_review_retry_test",
            repo_url="https://example.invalid/rr.git",
            base_commit=base_commit,
            branch=None, submodule_path=None, runners={},
        ),
        execution=None,
        intent="exercise the review-feedback retry loop",
        constraints=(), done_criteria=(), plan_policy=None, depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=(sl,), depends_on=()),),
        ),
        implementation_refs=(), verdict_refs=(),
        measurement_refs=(), lifecycle_event_refs=(),
    )


def _spec_two_slices(base_commit: str) -> Spec:
    slices = tuple(
        Slice(
            id=f"s{index}",
            directive=ImplementationDirective(
                slice_id=f"s{index}",
                intent=f"slice {index} work",
                files_in_scope=(f"file_{index}.py",),
                creates_paths=(f"file_{index}.py",),
                verify_mode="skip",
                blast_radius={"max_files": 1},
            ),
        )
        for index in range(2)
    )
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier_review_retry_two_slice_test",
            repo_url="https://example.invalid/rr2.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="exercise paid-call budget checks between slices",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=slices, depends_on=()),),
        ),
    )


def _spec_public_export_conflict(base_commit: str) -> Spec:
    slice_ = Slice(
        id="s0",
        directive=ImplementationDirective(
            slice_id="s0",
            intent=(
                "Change structured conversion behavior. Remove PublicSymbol "
                "from export surface. Preserve explicit public mixin behavior."
            ),
            files_in_scope=("file_0.py",),
            creates_paths=("file_0.py",),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        ),
    )
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier-review-public-export-conflict",
            repo_url="https://example.invalid/public-export.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={"verify": "tests.runner"},
        ),
        execution=None,
        intent=(
            "Change structured conversion while preserving the public API."
        ),
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(tiers=(
            Tier(tier_id="t0", slices=(slice_,), depends_on=()),
        )),
    )


def _run_public_export_conflict(
    tmp_path: Path,
    monkeypatch,
    *,
    malformed_authority: bool = False,
    fail_final_gate: bool = False,
    block_reconciliation: bool = False,
    backend_type=_ConflictResolutionBackend,
    budget_usd: float | None = None,
    response_retry_budget: int = _DEFAULT_TIER_RESPONSE_RETRY_BUDGET,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    Corpus(corpus_dir).put_spec(_spec_public_export_conflict(base))
    backend = backend_type(
        repo,
        malformed_authority=malformed_authority,
        block_reconciliation=block_reconciliation,
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *a, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kw: backend,
    )
    gate_calls = 0

    def gate(*args, **kwargs):  # noqa: ARG001
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 2:
            failed = RunnerResult(
                role="verify",
                runner="tests.runner",
                returncode=1,
                values={},
                raw_output=b"ImportError: PublicSymbol is required",
                stderr=b"",
                duration_s=0.1,
            )
            outcome = BuildVerifyOutcome(
                "verify_failed",
                build=None,
                verify=failed,
            )
        elif fail_final_gate and gate_calls == 4:
            failed = RunnerResult(
                role="verify",
                runner="tests.runner",
                returncode=1,
                values={},
                raw_output=b"final regression",
                stderr=b"",
                duration_s=0.1,
            )
            outcome = BuildVerifyOutcome(
                "verify_failed",
                build=None,
                verify=failed,
            )
        else:
            outcome = BuildVerifyOutcome(
                "success",
                build=None,
                verify=None,
            )
        from crucible.workflows import _build_verify

        _build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_review_conflict_gate_stub",
            succeeded=outcome.classification.startswith("success"),
        )
        return outcome

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        gate,
    )
    from crucible.workflows.tier import run_tier

    rc = run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        budget_usd=budget_usd,
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        breach_policy=BreachPolicy(enabled=False),
        review_policy=ReviewPolicy(
            rubrics=("diff_scope", "tier_completeness"),
            retry_budget=1,
        ),
        response_retry_budget=response_retry_budget,
    )
    run_dir = sorted((corpus_dir / "runs").glob("*tier*"))[-1]
    summary = json.loads((run_dir / "summary.json").read_text())
    return rc, summary, backend, gate_calls, repo, Corpus(corpus_dir)


def _run(
    tmp_path: Path,
    monkeypatch,
    retry_budget: int,
    *,
    budget_usd: float | None = None,
    backend_type=_FailThenPassBackend,
    skip_build_verify: bool = True,
    development_budget_ledger: DevelopmentBudgetLedger | None = None,
    review_policy: ReviewPolicy | None = None,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    Corpus(corpus_dir).put_spec(_spec_single_slice(base))

    backend = backend_type(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *a, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kw: backend,
    )
    from crucible.workflows.tier import run_tier

    rc = run_tier(
        spec=corpus_dir, tier_id="t0", cwd=str(repo),
        budget_usd=budget_usd,
        skip_build_verify=skip_build_verify, skip_smoke_check=True,
        skip_mutation_scope=True, skip_smoke_bench=True,
        breach_policy=BreachPolicy(enabled=False),
        review_policy=(
            review_policy
            if review_policy is not None
            else ReviewPolicy(
                rubrics=("tier_completeness",), retry_budget=retry_budget,
            )
        ),
        development_budget_ledger=development_budget_ledger,
    )
    runs = [r for r in (corpus_dir / "runs").iterdir() if "tier" in r.name]
    summary = json.loads((runs[-1] / "summary.json").read_text())
    return rc, summary, backend


def test_review_retry_recovers_to_rc0(tmp_path, monkeypatch):
    """retry_budget=1: first review fails, the slice is re-implemented
    with feedback, the second review passes → tier lands (rc=0)."""
    from crucible.workflows import tier as tier_mod

    strict_review = tier_mod.review
    review_bounds = []

    def recording_review(*args, **kwargs):
        review_bounds.append((
            kwargs.get("response_retry_budget"),
            kwargs.get("chunk_chars"),
        ))
        return strict_review(*args, **kwargs)

    monkeypatch.setattr(tier_mod, "review", recording_review)
    rc, summary, backend = _run(tmp_path, monkeypatch, retry_budget=1)
    assert rc == 0, f"expected recovery to rc=0; summary status={summary.get('status')}"
    assert summary["status"] == "complete"  # STATUS_TIER_COMPLETE
    # Two reviews (fail, pass) and two implements (initial + one retry).
    assert backend.review_calls == 2
    assert backend.implement_calls == 2
    # Each semantic Phase-4 review independently gets the bounded autonomous
    # response-contract repairs. A valid model-authored domain ``revise``
    # must still reach the existing implementer-feedback retry path.
    assert review_bounds == [(8, 80_000), (8, 80_000)]


def test_tier_response_retry_budget_is_explicit_and_bounded(tmp_path):
    from crucible.workflows.tier import _parse_args, run_tier

    base = [
        "--spec", str(tmp_path / "spec"),
        "--tier", "t0",
    ]
    assert _parse_args(base).response_retry_budget == 8
    assert _parse_args([
        *base,
        "--response-retry-budget", "5",
    ]).response_retry_budget == 5
    with pytest.raises(SystemExit) as exc:
        _parse_args([*base, "--response-retry-budget", "-1"])
    assert exc.value.code == 2
    for invalid in ("9", "999999999999999999999999"):
        with pytest.raises(SystemExit) as exc:
            _parse_args([*base, "--response-retry-budget", invalid])
        assert exc.value.code == 2
    with pytest.raises(ValueError, match="must be at most 8"):
        run_tier(
            spec=tmp_path / "unopened-spec",
            tier_id="t0",
            cwd=str(tmp_path),
            response_retry_budget=9,
        )


def test_review_retry_blocked_is_terminal_without_rereview_or_fallback(
    tmp_path,
    monkeypatch,
):
    rc, summary, backend = _run(
        tmp_path,
        monkeypatch,
        retry_budget=2,
        backend_type=_BlockedReviewRetryBackend,
    )

    assert rc == 3
    assert summary["status"] == "review_failed"
    assert backend.implement_calls == 2
    assert backend.review_calls == 1
    assert len(summary["phase_4_reviews"]) == 1
    assert "phase_4_pre_breach_fallback" not in summary
    assert backend.repo.joinpath("file_0.py").read_text() == (
        "# implement attempt 1\n"
    )
    corpus = Corpus(tmp_path / "corpus" / "spec1")
    blocked = [
        corpus.get_record(ref, as_type=Implementation)
        for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).classification
        == "hard_fail_implementer_blocked"
    ]
    assert len(blocked) == 1
    assert "implementer_terminal_disposition" in blocked[0].artifacts


@pytest.mark.parametrize(
    ("backend_type", "budget_usd", "expected_status", "classification"),
    [
        (
            _AccountingDebtReviewRetryBackend,
            None,
            "incomplete_accounting",
            "hard_fail_incomplete_evidence",
        ),
        (
            _FailThenPassBackend,
            0.025,
            "budget_exhausted",
            "hard_fail_budget_exhausted",
        ),
    ],
)
def test_review_retry_paid_terminal_is_persisted_before_any_gate_or_rereview(
    tmp_path,
    monkeypatch,
    backend_type,
    budget_usd,
    expected_status,
    classification,
):
    rc, summary, backend = _run(
        tmp_path,
        monkeypatch,
        retry_budget=2,
        backend_type=backend_type,
        budget_usd=budget_usd,
    )

    assert rc == 1
    assert summary["status"] == expected_status
    assert backend.implement_calls == 2
    assert backend.review_calls == 1
    assert backend.repo.joinpath("file_0.py").read_text() == (
        "# implement attempt 1\n"
    )
    corpus = Corpus(tmp_path / "corpus" / "spec1")
    retained = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert retained.classification == classification


def test_mandatory_tail_stays_protected_after_tier_review_retry(
    tmp_path,
    monkeypatch,
):
    policy = DevelopmentBudgetPolicy(
        task_budget_usd=1.0,
        planning_review_cap_usd=0.4,
        implementation_verifier_reserve_usd=0.5,
        contingency_usd=0.1,
        roles=(
            RoleAllocation(
                role="implementer",
                group="implementation_verifier",
                max_calls=3,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=0.6,
            ),
            RoleAllocation(
                role="tier_completeness",
                group="implementation_verifier",
                max_calls=2,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=0.1,
                protected_tail_usd=0.1,
            ),
            RoleAllocation(
                role="next_implementer",
                group="implementation_verifier",
                max_calls=1,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=0.6,
            ),
        ),
    )

    class _RecordingLedger(DevelopmentBudgetLedger):
        def __init__(self, budget_policy):
            super().__init__(budget_policy)
            self.completion_snapshots = []

        def complete_protected_tail_role(self, role: str) -> None:
            metrics = self.to_summary()["autonomous_development_metrics"]
            self.completion_snapshots.append((
                role,
                dict(metrics["calls_by_role"]),
            ))
            super().complete_protected_tail_role(role)

    ledger = _RecordingLedger(policy)
    rc, summary, backend = _run(
        tmp_path,
        monkeypatch,
        retry_budget=1,
        development_budget_ledger=ledger,
    )

    assert rc == 0
    assert summary["status"] == "complete"
    assert backend.implement_calls == 2
    assert backend.review_calls == 2
    # Tier success is not the terminal owner boundary: a later Tier in the
    # Architect DAG can invoke the same mandatory rubric again.
    assert ledger.completion_snapshots == []
    tail = ledger.to_summary()["implementation_verifier_authority"][
        "protected_tail"
    ]
    assert tail["owners"]["tier_completeness"][
        "terminal_completed"
    ] is False
    assert tail["owners"]["tier_completeness"]["remaining_usd"] == 0.08
    assert tail["owners"]["tier_completeness"][
        "released_unused_usd"
    ] == 0.0

    class _RecordingBackend:
        name = "next-stub"
        model = "stub-model"

        def __init__(self):
            self.invocations = []

        def invoke(self, invocation):
            self.invocations.append(invocation)
            return InvocationResult(
                final_message="done",
                transcript=(),
                usage=Usage(
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.0,
                ),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )

    next_backend = _RecordingBackend()
    ledger.wrap("next_implementer", next_backend).invoke(Invocation(
        system="next tier",
        user="implement",
        tools=None,
        max_turns=1,
        budget_usd=0.6,
    ))
    assert next_backend.invocations[0].budget_usd == pytest.approx(0.48)


def test_response_contract_exhaustion_never_reimplements_domain_work(
    tmp_path,
    monkeypatch,
):
    """Infrastructure failure is terminal even with semantic retries enabled.

    Nine malformed reviewer responses are the initial response plus the eight
    default strict-response repairs. They must produce one deterministic
    inconclusive verdict and ``review_failed``; they must not be misread as
    domain feedback and sent to the implementer/rebuild loop.
    """

    rc, summary, backend = _run(
        tmp_path,
        monkeypatch,
        retry_budget=2,
        backend_type=_MalformedReviewBackend,
    )

    assert rc == 3
    assert summary["status"] == "review_failed"
    assert backend.review_calls == 9
    assert backend.implement_calls == 1
    assert len(summary["phase_4_reviews"]) == 1
    assert summary["phase_4_reviews"][0]["review_attempt"] == 0
    assert summary["phase_4_reviews"][0]["infrastructure_failure"] is True

    corpus = Corpus(tmp_path / "corpus" / "spec1")
    latest = corpus.latest_spec()
    assert latest is not None
    verdict = corpus.get_record(latest.verdict_refs[-1], as_type=Verdict)
    assert verdict.decision == "inconclusive"
    assert verdict.findings
    assert all(finding.source == "deterministic" for finding in verdict.findings)
    retry_manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_retry_manifest"],
    ))
    assert retry_manifest["terminal_reason"] == "retries_exhausted"
    assert len(retry_manifest["attempts"]) == 9


def test_no_retry_budget_abandons_on_first_failure(tmp_path, monkeypatch):
    """retry_budget=0 (default): the first review failure is terminal —
    rc=3, one review, one implement (no re-implement)."""
    rc, summary, backend = _run(tmp_path, monkeypatch, retry_budget=0)
    assert rc == 3
    assert summary["status"] == "review_failed"
    assert backend.review_calls == 1
    assert backend.implement_calls == 1


def test_breach_disabled_re_review_rejection_rolls_back_retry(
    tmp_path,
    monkeypatch,
):
    rc, summary, backend = _run(
        tmp_path,
        monkeypatch,
        retry_budget=1,
        backend_type=_AlwaysReviseBackend,
    )

    assert rc == 3
    assert summary["status"] == "review_failed"
    assert backend.review_calls == 2
    assert backend.implement_calls == 2
    assert (tmp_path / "repo" / "file_0.py").read_text() == (
        "# implement attempt 1\n"
    )
    latest = Corpus(tmp_path / "corpus" / "spec1").latest_spec()
    assert latest is not None
    assert len(latest.implementation_refs) == 1


@pytest.mark.timeout(300)
def test_public_export_conflict_reconciles_and_lands_with_verdict_provenance(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_mod

    strict_review = tier_mod.review
    authority_anchor_policies = []
    review_chunk_chars = []

    def recording_review(*args, **kwargs):
        rubric = args[1]
        review_chunk_chars.append(kwargs.get("chunk_chars"))
        if rubric.name == "review_conflict_authority":
            authority_anchor_policies.append(
                kwargs.get("review_anchor_policy")
            )
        return strict_review(*args, **kwargs)

    monkeypatch.setattr(tier_mod, "review", recording_review)
    rc, summary, backend, gate_calls, repo, corpus = (
        _run_public_export_conflict(tmp_path, monkeypatch)
    )

    assert rc == 0
    assert authority_anchor_policies == ["off"]
    assert review_chunk_chars == [80_000] * 5
    assert summary["status"] == "complete"
    assert gate_calls == 4
    assert backend.implement_calls == 3
    assert backend.diff_scope_calls == 2
    assert backend.completeness_calls == 2
    assert backend.authority_calls == 1
    assert (repo / "file_0.py").read_text() == (
        "# reconciled PublicSymbol stable\n"
    )
    conflict = summary["phase_4_review_conflict_authority"]
    assert conflict["status"] == "landed"
    assert [
        item["classification"] for item in conflict["records"]
    ] == ["actionable", "verification_conflict"]
    verdict_ref = conflict["authority_verdict_ref"]
    assert verdict_ref in [str(ref) for ref in corpus.latest_spec().verdict_refs]
    verdict = corpus.get_record(ContentHash(verdict_ref), as_type=Verdict)
    assert verdict.rubric == "review_conflict_authority"
    binding_summary = summary["phase_5_final_bindings"][-1]
    assert binding_summary["review_conflict_verdict_ref"] == verdict_ref
    final_impl = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    input_ref = final_impl.artifacts["review_conflict_authority_input"]
    assert input_ref == verdict.artifact_ref
    binding_note = json.loads(corpus.get_artifact(
        final_impl.artifacts["tier_final_workspace_binding"],
    ))
    assert binding_note["review_conflict_verdict_ref"] == verdict_ref


def test_review_conflict_reconciliation_blocked_stops_fresh_reviews(
    tmp_path,
    monkeypatch,
):
    rc, summary, backend, gate_calls, repo, corpus = (
        _run_public_export_conflict(
            tmp_path,
            monkeypatch,
            block_reconciliation=True,
        )
    )

    assert rc == 3
    assert summary["status"] == "review_failed"
    assert gate_calls == 2
    assert backend.implement_calls == 3
    assert backend.diff_scope_calls == 1
    assert backend.completeness_calls == 1
    assert backend.authority_calls == 1
    assert repo.joinpath("file_0.py").read_text() == (
        "# candidate PublicSymbol reordered\n"
    )
    assert summary["phase_4_review_conflict_authority"]["status"] == (
        "reconciliation_implementer_terminal"
    )
    blocked = [
        corpus.get_record(ref, as_type=Implementation)
        for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).classification
        == "hard_fail_implementer_blocked"
    ]
    assert len(blocked) == 1
    assert "implementer_terminal_disposition" in blocked[0].artifacts


@pytest.mark.parametrize(
    (
        "backend_type",
        "expected_status",
        "expected_classification",
        "expected_accounting_debt",
    ),
    (
        pytest.param(
            _AccountingDebtConflictResolutionBackend,
            "incomplete_accounting",
            "hard_fail_incomplete_evidence",
            ("token_usage_complete", "cost_complete"),
            id="accounting-over-budget",
        ),
        pytest.param(
            _BudgetConflictResolutionBackend,
            "budget_exhausted",
            "hard_fail_budget_exhausted",
            (),
            id="budget",
        ),
    ),
)
def test_review_conflict_paid_terminal_stops_before_gate_or_fresh_review(
    tmp_path,
    monkeypatch,
    backend_type,
    expected_status,
    expected_classification,
    expected_accounting_debt,
):
    rc, summary, backend, gate_calls, repo, corpus = (
        _run_public_export_conflict(
            tmp_path,
            monkeypatch,
            backend_type=backend_type,
            budget_usd=0.1,
        )
    )

    assert rc == 1
    assert summary["status"] == expected_status
    assert summary["phase_4_review_conflict_authority"]["status"] == (
        expected_status
    )
    # The initial candidate gate and rejected review-retry gate ran before the
    # conflict. The terminal reconciliation must not reach its own gate.
    assert gate_calls == 2
    assert backend.implement_calls == 3
    assert backend.diff_scope_calls == 1
    assert backend.completeness_calls == 1
    assert backend.authority_calls == 1
    assert "phase_4_pre_breach_fallback" not in summary
    assert "phase_5_cumulative_verification" not in summary
    assert repo.joinpath("file_0.py").read_text() == (
        "# candidate PublicSymbol reordered\n"
    )

    latest = corpus.latest_spec()
    assert latest is not None
    implementation_ref = latest.implementation_refs[-1]
    failed = corpus.get_record(
        implementation_ref,
        as_type=Implementation,
    )
    assert failed.classification == expected_classification
    assert failed.accounting_debt == expected_accounting_debt
    assert b"reconciled PublicSymbol stable" in corpus.get_artifact(
        failed.diff_ref
    )
    assert "implementer_terminal_disposition" not in failed.artifacts

    run_dir = sorted(
        (tmp_path / "corpus" / "spec1" / "runs").glob("*tier*")
    )[-1]
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    terminal_events = [
        event
        for event in events
        if event.get("status") == expected_status
        and event.get("slice_id") == "s0"
    ]
    assert terminal_events == [{
        "implementation_ref": str(implementation_ref),
        "phase": 4,
        "slice_id": "s0",
        "status": expected_status,
    }]


def test_malformed_review_conflict_authority_fails_closed_after_exact_rollback(
    tmp_path,
    monkeypatch,
):
    rc, summary, backend, gate_calls, repo, _corpus = (
        _run_public_export_conflict(
            tmp_path,
            monkeypatch,
            malformed_authority=True,
            response_retry_budget=2,
        )
    )

    assert rc == 3
    assert summary["status"] == "review_failed"
    assert gate_calls == 2
    assert backend.implement_calls == 2
    assert backend.authority_calls == 3
    assert (repo / "file_0.py").read_text() == (
        "# candidate PublicSymbol reordered\n"
    )
    conflict = summary["phase_4_review_conflict_authority"]
    assert conflict["status"] == "inconclusive"
    assert summary["phase_5_final_bindings"] == []


def test_invalid_conflict_packet_starts_no_paid_authority_call(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "crucible.workflows.tier._REVIEW_CONFLICT_MAX_EVIDENCE_BYTES",
        64,
    )
    rc, summary, backend, gate_calls, repo, _corpus = (
        _run_public_export_conflict(tmp_path, monkeypatch)
    )

    assert rc == 3
    assert summary["status"] == "review_failed"
    assert gate_calls == 2
    assert backend.authority_calls == 0
    assert backend.implement_calls == 2
    assert (repo / "file_0.py").read_text() == (
        "# candidate PublicSymbol reordered\n"
    )
    conflict = summary["phase_4_review_conflict_authority"]
    assert conflict["status"] == "inconclusive"
    assert conflict["cost_usd"] == 0.0
    assert "authority_verdict_ref" not in conflict
    assert any("size limit" in item for item in conflict["errors"])


def test_review_conflict_final_gate_failure_rolls_back_reconciliation(
    tmp_path,
    monkeypatch,
):
    rc, summary, backend, gate_calls, repo, _corpus = (
        _run_public_export_conflict(
            tmp_path,
            monkeypatch,
            fail_final_gate=True,
        )
    )

    assert rc == 1
    assert summary["status"] == "final_binding_failed"
    assert gate_calls == 4
    assert backend.implement_calls == 3
    assert (repo / "file_0.py").read_text() == (
        "# candidate PublicSymbol reordered\n"
    )
    assert summary["phase_5_final_bindings"] == []


@pytest.mark.timeout(300)
def test_review_retry_gate_mutation_restores_without_reusing_checkpoint(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    gate_calls = 0

    def _gate(*args, **kwargs):  # noqa: ARG001
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 2:
            target = tmp_path / "repo" / "file_0.py"
            target.write_text(
                target.read_text() + "# forbidden gate mutation\n"
            )
        outcome = BuildVerifyOutcome("success", build=None, verify=None)
        tier_workflow._build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_review_retry_gate_stub",
            succeeded=True,
        )
        return outcome

    monkeypatch.setattr(tier_workflow._build_verify, "build_and_verify", _gate)
    rc, summary, backend = _run(
        tmp_path,
        monkeypatch,
        retry_budget=1,
        skip_build_verify=False,
    )

    assert rc == 0
    assert summary["status"] == "complete"
    assert backend.review_calls == 2
    assert gate_calls == 3
    assert (tmp_path / "repo" / "file_0.py").read_text() == (
        "# implement attempt 1\n"
    )
    latest = Corpus(tmp_path / "corpus" / "spec1").latest_spec()
    assert latest is not None
    landed, pending = tier_workflow._partition.partition_for_resume(
        Corpus(tmp_path / "corpus" / "spec1"),
        latest.tier_dag.tiers[0],
        spec=latest,
    )
    assert pending == ()
    assert "s0" in landed


def test_zero_budget_starts_no_backend_call(tmp_path, monkeypatch):
    rc, summary, backend = _run(
        tmp_path,
        monkeypatch,
        retry_budget=1,
        budget_usd=0.0,
    )
    assert rc == 1
    assert summary["status"] == "budget_exhausted"
    assert backend.implement_calls == 0
    assert backend.review_calls == 0


def _run_two_slices(
    tmp_path: Path,
    monkeypatch,
    *,
    budget_usd: float,
    review_policy: ReviewPolicy,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    Corpus(corpus_dir).put_spec(_spec_two_slices(base))
    backend = _TwoSliceFailThenPassBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *a, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kw: backend,
    )

    from crucible.workflows.tier import run_tier

    rc = run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        budget_usd=budget_usd,
        max_concurrent_slices=1,
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        breach_policy=BreachPolicy(enabled=False),
        review_policy=review_policy,
    )
    run_dir = sorted((corpus_dir / "runs").glob("*tier*"))[-1]
    summary = json.loads((run_dir / "summary.json").read_text())
    return rc, summary, backend


def test_exact_cap_between_serial_slices_starts_no_second_implement(
    tmp_path,
    monkeypatch,
):
    rc, summary, backend = _run_two_slices(
        tmp_path,
        monkeypatch,
        budget_usd=0.01,
        review_policy=ReviewPolicy(rubrics=()),
    )
    assert rc == 1
    assert summary["status"] == "budget_exhausted"
    assert backend.implement_calls == 1
    assert backend.review_calls == 0


def test_exact_cap_between_review_retry_slices_starts_no_second_retry(
    tmp_path,
    monkeypatch,
):
    # Two initial implements + one failing review + one retry = $0.04.
    # The second slice retry must not start with remaining==0.
    rc, summary, backend = _run_two_slices(
        tmp_path,
        monkeypatch,
        budget_usd=0.04,
        review_policy=ReviewPolicy(
            rubrics=("tier_completeness",),
            retry_budget=1,
        ),
    )
    assert rc == 1
    assert summary["status"] == "budget_exhausted"
    assert backend.implement_calls == 3
    assert backend.review_calls == 1
