# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for durable deterministic and child-gate authority."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from assay import ExecutionContext, Implementation
from assay.canonical import parse, serialize
from assay.corpus.store import Corpus

from crucible.workflows._build_verify import BuildVerifyOutcome, RunnerResult
from crucible.workflows._gate_evidence import (
    CONDITIONAL_ABSENCE_ARTIFACT,
    DECLARED_TEST_EVIDENCE_PENDING_ARTIFACT,
    GateLedgerEntry,
    HANDOFF_GATE_LEDGER_ARTIFACT,
    HANDOFF_VERIFICATION_ARTIFACT,
    TIER_VERIFICATION_ARTIFACT,
    TIER_VERIFICATION_MANIFEST_ARTIFACT,
    conditional_absence_matches_implementation,
    load_handoff_gate_ledger,
    load_verification_evidence,
    persist_handoff_gate_ledger,
    persist_conditional_absence_evidence,
    persist_tier_verification_manifest,
    persist_verification_evidence,
    tier_verification_manifest_matches,
    verification_matches_implementation,
)
from crucible.workflows._outcome import TerminalClass, WorkflowOutcome


_AUTHORITY = "sha256:" + "a" * 64


def _runner(
    *,
    role: str = "verify",
    returncode: int = 0,
    runner: str | None = None,
) -> RunnerResult:
    return RunnerResult(
        role=role,
        runner=runner or f"tests.fake_{role}",
        returncode=returncode,
        values={"tests_passed": 2.0, "delta": -0.5},
        raw_output=b'{"values":{"tests_passed":2}}',
        stderr=b"warning\n",
        duration_s=1.25,
    )


def _outcome(classification: str = "success") -> BuildVerifyOutcome:
    return BuildVerifyOutcome(
        classification=classification,
        build=_runner(
            role="build",
            returncode=0 if classification == "success" else 1,
        ),
        verify=_runner(returncode=0 if classification == "success" else 1),
    )


def _execution() -> ExecutionContext:
    return ExecutionContext(
        timestamp="2026-08-05T00:00:00+00:00",
        machine="test",
        cwd="/repo",
        toolchain={},
        backend=None,
        model=None,
        recipe="handoff",
        recipe_commit=None,
        gpu_model=None,
        run_id="handoff-test",
    )


def _implementation(
    corpus: Corpus,
    *,
    diff_ref,
    evidence_ref,
    files_modified: tuple[str, ...] = ("src/a.py",),
) -> Implementation:
    return Implementation(
        slice_id="s0",
        base_commit="base",
        result_commit=None,
        diff_ref=diff_ref,
        transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
        files_modified=files_modified,
        scope_escapes=(),
        classification="success",
        execution=_execution(),
        artifacts={HANDOFF_VERIFICATION_ARTIFACT: evidence_ref},
    )


def test_conditional_absence_evidence_binds_selector_and_authorities(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    empty_ref = corpus.put_artifact(b"", kind="diffs")
    transcript_ref = corpus.put_artifact(b"[]", kind="transcripts")
    decision = Implementation(
        slice_id="decision", base_commit="base", result_commit=None,
        diff_ref=corpus.put_artifact(b"decision", kind="diffs"),
        transcript_ref=transcript_ref, files_modified=("decision.json",),
        scope_escapes=(), classification="success", execution=_execution(),
        artifacts={},
    )
    selected = replace(
        decision,
        slice_id="rust",
        diff_ref=corpus.put_artifact(b"rust", kind="diffs"),
        files_modified=("native/runtime-rust/lib.rs",),
    )
    selector = b'{"selection":{"language":"rust"}}\n'
    evidence_ref = persist_conditional_absence_evidence(
        corpus,
        tier_id="t3", slice_id="cpp",
        spec_execution_authority_hash=_AUTHORITY,
        candidate_diff_ref=empty_ref, base_commit="base", files_modified=(),
        creates_paths=("native/runtime-cpp",), selector_path="decision.json",
        selector_bytes=selector, json_pointer="/selection/language",
        expected_value="rust", decision_slice_id="decision",
        decision_implementation=decision, selected_slice_id="rust",
        selected_implementation=selected,
    )
    target = Implementation(
        slice_id="cpp", base_commit="base", result_commit=None,
        diff_ref=empty_ref, transcript_ref=transcript_ref, files_modified=(),
        scope_escapes=(), classification="soft_fail_no_changes",
        execution=_execution(),
        artifacts={CONDITIONAL_ABSENCE_ARTIFACT: evidence_ref},
    )

    kwargs = dict(
        tier_id="t3", slice_id="cpp",
        spec_execution_authority_hash=_AUTHORITY,
        creates_paths=("native/runtime-cpp",), decision_slice_id="decision",
        decision_implementation=decision, selected_slice_id="rust",
        selected_implementation=selected,
    )
    assert conditional_absence_matches_implementation(corpus, target, **kwargs)
    assert not conditional_absence_matches_implementation(
        corpus,
        target,
        **{**kwargs, "selected_implementation": replace(selected, base_commit="stale")},
    )


@pytest.mark.parametrize(
    ("selector", "pointer", "expected"),
    (
        (b"{}", "/selection/language", "rust"),
        (b'{"selection":{"language":"cpp"}}', "/selection/language", "rust"),
        (b'{"selection":{"language":["rust","cpp"]}}', "/selection/language", "rust"),
        (b'{"selection":{"language":"rust","language":"rust"}}', "/selection/language", "rust"),
    ),
)
def test_conditional_absence_evidence_rejects_missing_mismatched_or_ambiguous_selector(
    tmp_path: Path, selector: bytes, pointer: str, expected: object,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    empty_ref = corpus.put_artifact(b"", kind="diffs")
    transcript = corpus.put_artifact(b"[]", kind="transcripts")
    authority = Implementation(
        slice_id="authority", base_commit="base", result_commit=None,
        diff_ref=empty_ref, transcript_ref=transcript, files_modified=(),
        scope_escapes=(), classification="success", execution=_execution(),
        artifacts={},
    )
    with pytest.raises(ValueError):
        persist_conditional_absence_evidence(
            corpus, tier_id="t", slice_id="absent",
            spec_execution_authority_hash=_AUTHORITY,
            candidate_diff_ref=empty_ref, base_commit="base", files_modified=(),
            creates_paths=("out",), selector_path="decision.json",
            selector_bytes=selector, json_pointer=pointer, expected_value=expected,
            decision_slice_id="decision", decision_implementation=authority,
            selected_slice_id="selected", selected_implementation=authority,
        )


def _workflow_outcome(
    *,
    workflow: str,
    terminal_class: TerminalClass,
    detail_status: str,
    operation_id: str,
) -> WorkflowOutcome:
    return WorkflowOutcome(
        schema_version="crucible.workflow-outcome.v1",
        workflow=workflow,
        run_id=f"{workflow}-run",
        terminal_class=terminal_class,
        detail_status=detail_status,
        cost_usd=0.5,
        spec_version=3,
        refs=(),
        error_kind=None,
        error_message=None,
        parent_run_id="handoff-parent",
        operation_id=operation_id,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def test_verification_evidence_round_trips_and_binds_exact_candidate(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    diff_ref = corpus.put_artifact(b"diff --git a/src/a.py b/src/a.py\n", kind="diffs")
    evidence_ref = persist_verification_evidence(
        corpus,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        candidate_diff_ref=diff_ref,
        base_commit="base",
        files_modified=("src/a.py",),
        outcome=_outcome(),
        skipped_by_policy=False,
        build_timeout_s=60.0,
        verify_timeout_s=120.0,
    )
    evidence = load_verification_evidence(corpus, str(evidence_ref))
    implementation = _implementation(
        corpus,
        diff_ref=diff_ref,
        evidence_ref=evidence_ref,
    )

    assert evidence.satisfies_required_gate
    assert evidence.verify is not None
    assert corpus.get_artifact(evidence.verify.raw_output_ref) == (
        b'{"values":{"tests_passed":2}}'
    )
    assert corpus.get_artifact(evidence.verify.stderr_ref) == b"warning\n"
    assert verification_matches_implementation(
        corpus,
        implementation,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        require_gate=True,
    )
    assert not verification_matches_implementation(
        corpus,
        replace(implementation, artifacts={}),
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        require_gate=True,
    )

    for changed in (
        {"tier_id": "different"},
        {"slice_id": "different"},
        {"spec_execution_authority_hash": "sha256:" + "b" * 64},
    ):
        kwargs = {
            "tier_id": "t0",
            "slice_id": "s0",
            "spec_execution_authority_hash": _AUTHORITY,
            "require_gate": True,
            **changed,
        }
        assert not verification_matches_implementation(
            corpus,
            implementation,
            **kwargs,
        )

    wrong_diff = _implementation(
        corpus,
        diff_ref=corpus.put_artifact(b"different", kind="diffs"),
        evidence_ref=evidence_ref,
    )
    wrong_files = _implementation(
        corpus,
        diff_ref=diff_ref,
        evidence_ref=evidence_ref,
        files_modified=("src/other.py",),
    )
    assert not verification_matches_implementation(
        corpus,
        wrong_diff,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        require_gate=True,
    )
    assert not verification_matches_implementation(
        corpus,
        wrong_files,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        require_gate=True,
    )


@pytest.mark.parametrize("version", (1, 2, 3))
def test_superseded_verification_cannot_authorize_resume(tmp_path: Path, version: int) -> None:
    corpus = Corpus(tmp_path / "corpus")
    diff_ref = corpus.put_artifact(b"candidate", kind="diffs")
    ref = persist_verification_evidence(
        corpus, tier_id="t0", slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        candidate_diff_ref=diff_ref, base_commit="base",
        files_modified=("src/a.py",), outcome=_outcome(),
        skipped_by_policy=False, build_timeout_s=60.0, verify_timeout_s=120.0,
    )
    payload = parse(corpus.get_artifact(ref))
    assert payload["schema_version"] == "crucible.digest-bound-verification-evidence.v4"
    payload["schema_version"] = f"crucible.digest-bound-verification-evidence.v{version}"
    historical_ref = corpus.put_artifact(serialize(payload), kind="verification_evidence")
    with pytest.raises(ValueError, match="unsupported VerificationEvidence schema"):
        load_verification_evidence(corpus, str(historical_ref))
    implementation = _implementation(corpus, diff_ref=diff_ref, evidence_ref=historical_ref)
    assert not verification_matches_implementation(
        corpus, implementation, tier_id="t0", slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY, require_gate=True,
    )


@pytest.mark.parametrize("location, field, value", (
    ("root", "sil_finalization_required", False),
    ("root", "sil_finalization", None),
    ("build", "authority_evidence", []),
))
def test_removed_product_evidence_fields_are_rejected(
    tmp_path: Path, location: str, field: str, value: object,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    diff_ref = corpus.put_artifact(b"candidate", kind="diffs")
    ref = persist_verification_evidence(
        corpus, tier_id="t0", slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        candidate_diff_ref=diff_ref, base_commit="base",
        files_modified=("src/a.py",), outcome=_outcome(),
        skipped_by_policy=False, build_timeout_s=60.0, verify_timeout_s=120.0,
    )
    payload = parse(corpus.get_artifact(ref))
    target = payload if location == "root" else payload[location]
    target[field] = value
    forged_ref = corpus.put_artifact(serialize(payload), kind="verification_evidence")
    with pytest.raises(ValueError, match="fields differ"):
        load_verification_evidence(corpus, str(forged_ref))


@pytest.mark.parametrize(
    ("outcome", "skipped_by_policy", "classification", "disposition"),
    [
        (
            BuildVerifyOutcome("runner_missing", build=None, verify=None),
            False,
            "runner_missing",
            "unavailable",
        ),
        (None, True, "skipped_by_policy", "waived"),
    ],
)
def test_unavailable_or_waived_evidence_never_satisfies_required_gate(
    tmp_path: Path,
    outcome: BuildVerifyOutcome | None,
    skipped_by_policy: bool,
    classification: str,
    disposition: str,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    diff_ref = corpus.put_artifact(b"candidate", kind="diffs")
    ref = persist_verification_evidence(
        corpus,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        candidate_diff_ref=diff_ref,
        base_commit="base",
        files_modified=(),
        outcome=outcome,
        skipped_by_policy=skipped_by_policy,
        build_timeout_s=None,
        verify_timeout_s=None,
    )
    evidence = load_verification_evidence(corpus, str(ref))
    implementation = _implementation(
        corpus,
        diff_ref=diff_ref,
        evidence_ref=ref,
        files_modified=(),
    )

    assert evidence.classification == classification
    assert evidence.disposition == disposition
    assert not evidence.satisfies_required_gate
    assert not verification_matches_implementation(
        corpus,
        implementation,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        require_gate=True,
    )
    assert verification_matches_implementation(
        corpus,
        implementation,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        require_gate=False,
    )


def test_dev_verify_evidence_cannot_authorize_handoff_resume(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    diff_ref = corpus.put_artifact(b"candidate", kind="diffs")
    ref = persist_verification_evidence(
        corpus,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        candidate_diff_ref=diff_ref,
        base_commit="base",
        files_modified=(),
        outcome=BuildVerifyOutcome(
            "success",
            build=None,
            verify=None,
            dev_verify=_runner(role="dev_verify"),
        ),
        skipped_by_policy=False,
        build_timeout_s=60.0,
        verify_timeout_s=60.0,
        gate="dev_verify",
    )
    implementation = _implementation(
        corpus,
        diff_ref=diff_ref,
        evidence_ref=ref,
        files_modified=(),
    )

    assert not verification_matches_implementation(
        corpus,
        implementation,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        require_gate=True,
    )


def test_passing_declared_test_diagnostic_uses_a_non_authorizing_slot(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    diff_ref = corpus.put_artifact(b"candidate", kind="diffs")
    evidence_ref = persist_verification_evidence(
        corpus,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        candidate_diff_ref=diff_ref,
        base_commit="base",
        files_modified=("src/a.py",),
        outcome=_outcome(),
        skipped_by_policy=False,
        build_timeout_s=60.0,
        verify_timeout_s=60.0,
    )
    implementation = replace(
        _implementation(
            corpus,
            diff_ref=diff_ref,
            evidence_ref=evidence_ref,
        ),
        classification="hard_fail_missing_declared_test_mutation",
        artifacts={DECLARED_TEST_EVIDENCE_PENDING_ARTIFACT: evidence_ref},
    )

    evidence = load_verification_evidence(corpus, str(evidence_ref))
    assert evidence.satisfies_required_gate
    assert DECLARED_TEST_EVIDENCE_PENDING_ARTIFACT == (
        "declared_test_evidence_pending"
    )
    assert TIER_VERIFICATION_ARTIFACT not in implementation.artifacts
    assert HANDOFF_VERIFICATION_ARTIFACT not in implementation.artifacts
    assert not verification_matches_implementation(
        corpus,
        implementation,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        require_gate=True,
    )


def test_forged_success_without_required_runner_results_is_rejected(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    diff_ref = corpus.put_artifact(b"candidate", kind="diffs")

    with pytest.raises(ValueError, match="successful build"):
        persist_verification_evidence(
            corpus,
            tier_id="t0",
            slice_id="s0",
            spec_execution_authority_hash=_AUTHORITY,
            candidate_diff_ref=diff_ref,
            base_commit="base",
            files_modified=(),
            outcome=BuildVerifyOutcome("success", build=None, verify=None),
            skipped_by_policy=False,
            build_timeout_s=60.0,
            verify_timeout_s=60.0,
        )


def test_tier_manifest_binds_diffless_slice_and_exact_sibling_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    empty_slice_ref = corpus.put_artifact(b"", kind="diffs")
    sibling_ref = corpus.put_artifact(b"sibling b1", kind="diffs")
    union_ref = corpus.put_artifact(b"mixed tier union", kind="diffs")
    evidence_ref = persist_verification_evidence(
        corpus,
        tier_id="t0",
        slice_id="verify-only",
        spec_execution_authority_hash=_AUTHORITY,
        candidate_diff_ref=union_ref,
        base_commit="base",
        files_modified=("src/a.py",),
        outcome=_outcome(),
        skipped_by_policy=False,
        build_timeout_s=60.0,
        verify_timeout_s=60.0,
    )
    verification_only = replace(
        _implementation(
            corpus,
            diff_ref=empty_slice_ref,
            evidence_ref=evidence_ref,
            files_modified=(),
        ),
        slice_id="verify-only",
        artifacts={TIER_VERIFICATION_ARTIFACT: evidence_ref},
    )
    sibling = replace(
        _implementation(
            corpus,
            diff_ref=sibling_ref,
            evidence_ref=evidence_ref,
        ),
        slice_id="mutating",
        artifacts={},
    )
    with pytest.raises(ValueError, match="exact obligations"):
        persist_tier_verification_manifest(
            corpus,
            tier_id="t0",
            spec_execution_authority_hash=_AUTHORITY,
            project_base_commit="base",
            candidate_diff_ref=union_ref,
            base_commit="base",
            files_modified=("src/a.py",),
            gate="verify",
            obligation_slice_ids=("verify-only", "mutating"),
            current_implementations=(verification_only, sibling),
            dependency_implementations=(),
            verification_evidence_refs=(evidence_ref,),
        )
    manifest_ref = persist_tier_verification_manifest(
        corpus,
        tier_id="t0",
        spec_execution_authority_hash=_AUTHORITY,
        project_base_commit="base",
        candidate_diff_ref=union_ref,
        base_commit="base",
        files_modified=("src/a.py",),
        gate="verify",
        obligation_slice_ids=("verify-only",),
        current_implementations=(verification_only, sibling),
        dependency_implementations=(),
        verification_evidence_refs=(evidence_ref,),
    )
    verification_only = replace(
        verification_only,
        artifacts={
            **verification_only.artifacts,
            TIER_VERIFICATION_MANIFEST_ARTIFACT: manifest_ref,
        },
    )
    sibling = replace(
        sibling,
        artifacts={TIER_VERIFICATION_MANIFEST_ARTIFACT: manifest_ref},
    )

    assert tier_verification_manifest_matches(
        corpus,
        manifest_ref=str(manifest_ref),
        tier_id="t0",
        spec_execution_authority_hash=_AUTHORITY,
        expected_project_base_commit="base",
        current_implementations=(verification_only, sibling),
        dependency_implementations=(),
        expected_obligation_slice_ids=("verify-only",),
        require_gate=True,
    )
    evidence = load_verification_evidence(corpus, str(evidence_ref))
    assert evidence.verify is not None
    missing_log_ref = evidence.verify.raw_output_ref
    real_get_artifact = corpus.get_artifact

    def missing_runner_log(ref):
        if str(ref) == missing_log_ref:
            raise FileNotFoundError(missing_log_ref)
        return real_get_artifact(ref)

    monkeypatch.setattr(corpus, "get_artifact", missing_runner_log)
    assert not tier_verification_manifest_matches(
        corpus,
        manifest_ref=str(manifest_ref),
        tier_id="t0",
        spec_execution_authority_hash=_AUTHORITY,
        expected_project_base_commit="base",
        current_implementations=(verification_only, sibling),
        dependency_implementations=(),
        expected_obligation_slice_ids=("verify-only",),
        require_gate=True,
    )
    monkeypatch.setattr(corpus, "get_artifact", real_get_artifact)
    superseded_sibling = replace(
        sibling,
        diff_ref=corpus.put_artifact(b"sibling b2", kind="diffs"),
    )
    assert not tier_verification_manifest_matches(
        corpus,
        manifest_ref=str(manifest_ref),
        tier_id="t0",
        spec_execution_authority_hash=_AUTHORITY,
        expected_project_base_commit="base",
        current_implementations=(verification_only, superseded_sibling),
        dependency_implementations=(),
        expected_obligation_slice_ids=("verify-only",),
        require_gate=True,
    )


def test_verification_decoder_rejects_unknown_fields_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    diff_ref = corpus.put_artifact(b"candidate", kind="diffs")
    good_ref = persist_verification_evidence(
        corpus,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        candidate_diff_ref=diff_ref,
        base_commit="base",
        files_modified=(),
        outcome=_outcome(),
        skipped_by_policy=False,
        build_timeout_s=60.0,
        verify_timeout_s=60.0,
    )
    payload = dict(parse(corpus.get_artifact(good_ref)))
    payload["future_field"] = True
    unknown_ref = corpus.put_artifact(serialize(payload), kind="verification_evidence")
    noncanonical_ref = corpus.put_artifact(
        b'{"schema_version": "crucible.digest-bound-verification-evidence.v1"}',
        kind="verification_evidence",
    )

    with pytest.raises(ValueError, match="fields differ"):
        load_verification_evidence(corpus, str(unknown_ref))
    with pytest.raises(ValueError, match="canonical JSON"):
        load_verification_evidence(corpus, str(noncanonical_ref))


def test_gate_ledger_preserves_independent_typed_child_outcomes(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    candidate_ref = corpus.put_artifact(b"candidate", kind="diffs")
    breach = _workflow_outcome(
        workflow="breach",
        terminal_class=TerminalClass.BUDGET_EXHAUSTED,
        detail_status="budget_exhausted",
        operation_id="breach-s0",
    )
    review = _workflow_outcome(
        workflow="review",
        terminal_class=TerminalClass.SUCCEEDED,
        detail_status="pass",
        operation_id="review-s0",
    )
    entries = (
        GateLedgerEntry(
            gate="breach",
            candidate_diff_sha256=str(candidate_ref),
            terminal_class=breach.terminal_class.value,
            decision=breach.detail_status,
            exit_code=1,
            workflow_outcome=breach,
            verification_evidence_ref=None,
            error_kind=None,
            error_message=None,
        ),
        GateLedgerEntry(
            gate="review",
            candidate_diff_sha256=str(candidate_ref),
            terminal_class=review.terminal_class.value,
            decision=review.detail_status,
            exit_code=0,
            workflow_outcome=review,
            verification_evidence_ref=None,
            error_kind=None,
            error_message=None,
        ),
    )
    ref = persist_handoff_gate_ledger(
        corpus,
        tier_id="t0",
        slice_id="s0",
        spec_execution_authority_hash=_AUTHORITY,
        terminal_candidate_diff_ref=candidate_ref,
        entries=entries,
    )
    ledger = load_handoff_gate_ledger(corpus, str(ref))

    assert ledger.entries[0].workflow_outcome == breach
    assert ledger.entries[1].workflow_outcome == review
    assert not ledger.entries[0].succeeded
    assert ledger.entries[1].succeeded
    assert HANDOFF_GATE_LEDGER_ARTIFACT == "handoff_gate_ledger"
