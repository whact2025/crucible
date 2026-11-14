# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fixed-candidate Plan admission, routing, and publication tests."""

from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from assay import (
    Evidence,
    ExecutionCompleteness,
    Finding,
    LifecycleEvent,
    PlanOutcome,
    ProjectContext,
    ReviewCoverage,
    Rubric,
    Spec,
    Verdict,
)
from assay.corpus.store import ConcurrentSpecError, Corpus
from assay.canonical import hash_record

from crucible.workflows import plan as plan_workflow
from crucible.workflows._atomic_persistence import commit_staged_spec


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
        intent="edit the existing crate source",
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


def _candidate_payload() -> dict:
    return {
        "tiers": [
            {
                "tier_id": "t0",
                "depends_on": [],
                "slices": [
                    {
                        "id": "s0",
                        "directive": {
                            "slice_id": "s0",
                            "intent": "edit the existing crate source",
                            "requirement_refs": [
                                "edit the existing crate source"
                            ],
                            "files_in_scope": ["src/thing.py"],
                            "creates_paths": [],
                            "verify_mode": "tests",
                            "blast_radius": {
                                "max_files": 1,
                                "languages": ["python"],
                            },
                        },
                    }
                ],
            }
        ]
    }


@pytest.mark.parametrize(
    "references",
    [None, {}, "reference", [{"reference_id": "missing-fields"}]],
)
def test_diagnostic_candidate_rejects_malformed_read_only_references(
    references,
) -> None:
    payload = _candidate_payload()
    payload["tiers"][0]["slices"][0]["directive"][
        "read_only_references"
    ] = references
    with pytest.raises(ValueError, match="invalid TierDAG artifact"):
        plan_workflow._diagnostic_candidate_dag(json.dumps(payload).encode())


@pytest.mark.parametrize("field", ["extra", "bool_line", "duplicate_id"])
def test_diagnostic_candidate_enforces_typed_read_only_references(field) -> None:
    reference = {
        "reference_id": "source",
        "path": "existing.py",
        "sha256": "sha256:" + "a" * 64,
        "start_line": 1,
        "end_line": 2,
    }
    references = [reference]
    if field == "extra":
        reference["unreviewed"] = True
    elif field == "bool_line":
        reference["start_line"] = True
    else:
        references.append(dict(reference))
    payload = _candidate_payload()
    payload["tiers"][0]["slices"][0]["directive"][
        "read_only_references"
    ] = references
    with pytest.raises(ValueError, match="invalid TierDAG artifact"):
        plan_workflow._diagnostic_candidate_dag(json.dumps(payload).encode())


def test_legacy_diagnostic_candidate_keeps_empty_reference_hash() -> None:
    payload = _candidate_payload()
    legacy = plan_workflow._strict_review_candidate_dag(payload)
    diagnostic = plan_workflow._diagnostic_candidate_dag(json.dumps(payload).encode())
    assert diagnostic == legacy
    assert diagnostic.tiers[0].slices[0].directive.read_only_references == ()
    assert hash_record(diagnostic) == hash_record(legacy)


def test_review_candidate_reader_uses_closed_duplicate_free_schema(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"tiers": [], "tiers": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object member"):
        plan_workflow._read_review_candidate(candidate)

    payload = _candidate_payload()
    payload["unreviewed_extension"] = True
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown=.*unreviewed_extension"):
        plan_workflow._read_review_candidate(candidate)


def test_review_candidate_reader_rejects_invalid_dag_identities() -> None:
    duplicate_tiers = _candidate_payload()
    duplicate_tiers["tiers"].append(copy.deepcopy(duplicate_tiers["tiers"][0]))

    mismatched_slice = _candidate_payload()
    mismatched_slice["tiers"][0]["slices"][0]["directive"][
        "slice_id"
    ] = "different"

    duplicate_dependencies = _candidate_payload()
    duplicate_dependencies["tiers"][0]["depends_on"] = ["t0", "t0"]

    invalid_verify_mode = _candidate_payload()
    invalid_verify_mode["tiers"][0]["slices"][0]["directive"][
        "verify_mode"
    ] = []

    no_slices = _candidate_payload()
    no_slices["tiers"][0]["slices"] = []

    cases = (
        ({"tiers": []}, "at least one tier"),
        (no_slices, "at least one slice"),
        (duplicate_tiers, "tier_id values must be unique"),
        (mismatched_slice, "does not match directive.slice_id"),
        (duplicate_dependencies, "duplicate dependencies"),
        (invalid_verify_mode, "verify_mode is unsupported"),
    )
    for payload, message in cases:
        with pytest.raises(ValueError, match=message):
            plan_workflow._strict_review_candidate_dag(payload)


def test_review_candidate_reader_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "candidate.json"
    target.write_text(json.dumps(_candidate_payload()), encoding="utf-8")
    link = tmp_path / "candidate-link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        plan_workflow._read_review_candidate(link)


def test_review_candidate_reader_rejects_non_unicode_scalar_keys_and_values(
    tmp_path: Path,
) -> None:
    invalid_key = _candidate_payload()
    invalid_key["\ud800"] = None
    invalid_value = _candidate_payload()
    invalid_value["tiers"][0]["slices"][0]["directive"]["intent"] = "\udfff"
    candidate = tmp_path / "candidate.json"

    for payload in (invalid_key, invalid_value):
        candidate.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="non-Unicode-scalar string"):
            plan_workflow._read_review_candidate(candidate)


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"),
    reason="requires POSIX nonblocking FIFOs",
)
def test_review_candidate_reader_nonblocking_open_rejects_fifo_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(_candidate_payload()), encoding="utf-8")
    real_open = os.open

    def _swap_before_open(path, flags, *args, **kwargs):
        candidate.unlink()
        os.mkfifo(candidate)
        assert flags & os.O_NONBLOCK
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(plan_workflow.os, "open", _swap_before_open)

    with pytest.raises(ValueError, match="changed to a non-regular file"):
        plan_workflow._read_review_candidate(candidate)


def test_review_candidate_cli_handles_malformed_field_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec_dir = tmp_path / "corpus"
    Corpus(spec_dir).put_spec(_spec())
    payload = _candidate_payload()
    payload["tiers"][0]["slices"][0]["directive"]["verify_mode"] = []
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    @contextmanager
    def _unexpected_pool(*_args, **_kwargs):
        raise AssertionError("malformed candidate must not acquire a backend")
        yield  # pragma: no cover

    monkeypatch.setattr(plan_workflow, "backend_pool", _unexpected_pool)
    rc = plan_workflow.main([
        "--spec",
        str(spec_dir),
        "--cwd",
        str(workspace),
        "--review-candidate",
        str(candidate_path),
    ])

    assert rc == 2
    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "invalid_review_candidate"
    assert "verify_mode is unsupported" in summary["message"]


@pytest.mark.parametrize("malformation", ["deep_json", "surrogate_value"])
def test_review_candidate_cli_handles_adversarial_json_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec_dir = tmp_path / "corpus"
    Corpus(spec_dir).put_spec(_spec())
    candidate_path = tmp_path / "candidate.json"
    if malformation == "deep_json":
        depth = 2_000
        candidate_path.write_bytes(
            b'{"tiers":' + b"[" * depth + b"]" * depth + b"}"
        )
        expected_message = "too deeply nested"
    else:
        payload = _candidate_payload()
        payload["tiers"][0]["slices"][0]["directive"]["intent"] = "\ud800"
        candidate_path.write_text(json.dumps(payload), encoding="utf-8")
        expected_message = "non-Unicode-scalar string"

    @contextmanager
    def _unexpected_pool(*_args, **_kwargs):
        raise AssertionError("invalid candidate must not acquire a backend")
        yield  # pragma: no cover

    monkeypatch.setattr(plan_workflow, "backend_pool", _unexpected_pool)
    rc = plan_workflow.main([
        "--spec",
        str(spec_dir),
        "--cwd",
        str(workspace),
        "--review-candidate",
        str(candidate_path),
    ])

    assert rc == 2
    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "invalid_review_candidate"
    assert expected_message in summary["message"]


def test_fixed_candidate_publication_refuses_concurrent_spec_advance(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    corpus.put_spec(_spec())
    base = corpus.latest_spec()
    assert base is not None
    candidate = plan_workflow._strict_review_candidate_dag(
        _candidate_payload()
    )
    staged = replace(base, tier_dag=candidate)
    corpus.commit(lifecycle_events=(LifecycleEvent(
        kind="concurrent_update",
        reason="advance while candidate review is in flight",
        affected_versions=(1,),
        execution=plan_workflow.exec_ctx_now("test"),
    ),))

    with pytest.raises(ConcurrentSpecError, match="refusing to rebase"):
        commit_staged_spec(
            corpus,
            base=base,
            staged=staged,
            structural_fields=("tier_dag",),
            require_exact_base=True,
        )

    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.version == 1
    assert latest.tier_dag is None


def test_fixed_candidate_accounting_failure_keeps_candidate_and_review_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = plan_workflow._strict_review_candidate_dag(
        _candidate_payload()
    )

    def _accounting_failure(*_args, **_kwargs):
        raise plan_workflow._PlanAccountingIncomplete(
            ("cost_complete",),
            role="critic",
        )

    monkeypatch.setattr(plan_workflow, "plan", _accounting_failure)
    outcome = plan_workflow.run_plan(
        _spec(),
        candidate_dag=candidate,
        critic_backend=object(),
        judge_backend=object(),
        corpus=Corpus(tmp_path / "corpus"),
        critic_rubric=object(),
        judge_rubric=object(),
    )

    assert outcome.status == "incomplete_accounting"
    assert outcome.rounds_used == 1
    assert outcome.candidate_dag == candidate
    assert outcome.spec.tier_dag is None


def test_fixed_candidate_accounting_failure_retains_transformed_review_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    candidate = plan_workflow._strict_review_candidate_dag(
        _candidate_payload()
    )
    transformed_payload = _candidate_payload()
    transformed_payload["tiers"][0]["slices"][0]["directive"][
        "files_in_scope"
    ] = ["src/transformed.py"]
    transformed = plan_workflow._strict_review_candidate_dag(
        transformed_payload
    )
    reviewed_artifacts: list[bytes] = []

    def _strict_review(artifact: bytes, *_args, **kwargs) -> Verdict:
        reviewed_artifacts.append(artifact)
        artifact_ref = kwargs["corpus"].put_artifact(
            artifact,
            kind="reviewed_artifacts",
        )
        return Verdict(
            rubric="critic",
            decision="pass",
            findings=(),
            artifact_ref=artifact_ref,
            artifact_commit=None,
            reasoning="semantic pass with incomplete cost accounting",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
                artifact_chars_total=len(artifact),
                artifact_chars_reviewed=len(artifact),
            ),
            backend_name="stub",
            transcript_ref=None,
            cost_usd=0.0,
            execution=None,
            kind="critic",
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=False,
                terminal_complete=True,
                trace_complete=True,
            ),
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )

    monkeypatch.setattr(plan_workflow, "strict_review", _strict_review)
    outcome = plan_workflow.run_plan(
        _spec(),
        candidate_dag=candidate,
        critic_backend=object(),
        judge_backend=object(),
        corpus=corpus,
        critic_rubric=Rubric(
            name="critic",
            system_prompt="review",
            user_template="{artifact}",
        ),
        judge_rubric=Rubric(
            name="judge",
            system_prompt="judge",
            user_template="{artifact}",
        ),
        candidate_transform=lambda _candidate: transformed,
    )

    assert outcome.status == "incomplete_accounting"
    assert outcome.rounds_used == 1
    assert outcome.candidate_dag == transformed
    assert outcome.candidate_dag != candidate
    assert outcome.candidate_ref is not None
    assert reviewed_artifacts == [corpus.get_artifact(outcome.candidate_ref)]
    assert outcome.spec.tier_dag is None


@pytest.mark.parametrize(
    ("critic_decision", "expected_decision", "expected_event"),
    [
        ("pass", "accepted", "planned"),
        ("revise", "rejected", "plan_rejected"),
    ],
)
def test_fixed_candidate_real_plan_retains_terminal_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    critic_decision: str,
    expected_decision: str,
    expected_event: str,
) -> None:
    corpus = Corpus(tmp_path / "corpus")
    review_calls: list[str] = []

    def _strict_review(artifact: bytes, *_args, **kwargs) -> Verdict:
        role = kwargs["kind"]
        review_calls.append(role)
        artifact_ref = kwargs["corpus"].put_artifact(
            artifact,
            kind="reviewed_artifacts",
        )
        decision = critic_decision if role == "critic" else "pass"
        finding = (
            Finding(
                id="candidate_needs_revision",
                severity="High",
                category="scope",
                evidence=(),
                suggested_action="submit a corrected candidate",
                source="llm",
            ),
        ) if decision == "revise" else ()
        return Verdict(
            rubric=role,
            decision=decision,
            findings=finding,
            artifact_ref=artifact_ref,
            artifact_commit=None,
            reasoning=f"{role} {decision}",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
                artifact_chars_total=len(artifact),
                artifact_chars_reviewed=len(artifact),
            ),
            backend_name="stub",
            transcript_ref=None,
            cost_usd=0.0,
            execution=None,
            kind=role,
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=True,
                terminal_complete=True,
                trace_complete=True,
            ),
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )

    monkeypatch.setattr(plan_workflow, "strict_review", _strict_review)
    outcome = plan_workflow.run_plan(
        _spec(),
        candidate_dag=plan_workflow._strict_review_candidate_dag(
            _candidate_payload()
        ),
        critic_backend=object(),
        judge_backend=object(),
        corpus=corpus,
        critic_rubric=Rubric(
            name="critic",
            system_prompt="review",
            user_template="{artifact}",
        ),
        judge_rubric=Rubric(
            name="judge",
            system_prompt="judge",
            user_template="{artifact}",
        ),
    )

    assert review_calls == ["critic", "judge"]
    assert outcome.decision == expected_decision
    assert outcome.candidate_ref is not None
    events = [
        corpus.get_record(ref, as_type=LifecycleEvent)
        for ref in outcome.spec.lifecycle_event_refs
    ]
    assert [event.kind for event in events] == [expected_event]


def test_generated_plan_final_gate_replaces_planned_terminal_after_rebase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec_dir = tmp_path / "corpus"
    corpus = Corpus(spec_dir)
    corpus.put_spec(_spec())
    candidate = plan_workflow._strict_review_candidate_dag(
        _candidate_payload()
    )
    acquired_roles: list[str] = []

    class _Pool:
        def get(self, role: str, **_kwargs):
            acquired_roles.append(role)
            return object()

    @contextmanager
    def _pool(*_args, **_kwargs):
        yield _Pool()

    def _run_plan(spec, **kwargs):
        assert kwargs["candidate_dag"] is None
        planned_ref = corpus.put_record(
            LifecycleEvent(
                kind="planned",
                reason="provisional Assay acceptance",
                affected_versions=(spec.version + 1,),
                execution=plan_workflow.exec_ctx_now("plan"),
            ),
            kind="lifecycle_events",
        )
        candidate_ref = corpus.put_artifact(
            json.dumps(_candidate_payload()).encode("utf-8"),
            kind="plan_candidates",
        )
        corpus.commit(lifecycle_events=(LifecycleEvent(
            kind="concurrent_update",
            reason="advance before final validation publication",
            affected_versions=(spec.version + 1,),
            execution=plan_workflow.exec_ctx_now("test"),
        ),))
        return PlanOutcome(
            spec=replace(
                spec,
                tier_dag=candidate,
                lifecycle_event_refs=(
                    *spec.lifecycle_event_refs,
                    planned_ref,
                ),
            ),
            candidate_dag=candidate,
            candidate_ref=candidate_ref,
            decision="accepted",
            status="accepted",
            rounds_used=1,
            final_critic_decision="pass",
            final_judge_decision="pass",
            reason="stubbed generated-plan acceptance",
            cost_usd=0.0,
        )

    def _reject_final_gate(_spec, *, ruleset, ctx) -> Verdict:
        del ruleset, ctx
        artifact_ref = corpus.put_artifact(
            b"final validation input",
            kind="review_inputs",
        )
        return Verdict(
            rubric="plan_gate",
            decision="reject",
            findings=(Finding(
                id="final_workspace_drift",
                severity="High",
                category="other",
                evidence=(),
                suggested_action="generate and review a fresh plan",
                source="deterministic",
            ),),
            artifact_ref=artifact_ref,
            artifact_commit=None,
            reasoning="final workspace validation rejected the plan",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("tier_dag",),
                omissions=(),
            ),
            backend_name=None,
            transcript_ref=None,
            cost_usd=0.0,
            execution=None,
            kind="validate",
        )

    monkeypatch.setattr(plan_workflow, "backend_pool", _pool)
    monkeypatch.setattr(plan_workflow, "run_plan", _run_plan)
    monkeypatch.setattr(plan_workflow, "validate", _reject_final_gate)
    monkeypatch.setattr(plan_workflow, "available_rules", lambda: ())

    rc = plan_workflow.main([
        "--planner-count", "1",
        "--spec",
        str(spec_dir),
        "--cwd",
        str(workspace),
    ])

    assert rc == 1
    assert acquired_roles == ["plan_planner", "plan_critic", "plan_judge"]
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.version == 2
    assert latest.tier_dag is None
    events = [
        corpus.get_record(ref, as_type=LifecycleEvent)
        for ref in latest.lifecycle_event_refs
    ]
    assert [event.kind for event in events] == [
        "concurrent_update",
        "plan_rejected",
    ]
    assert all(event.kind != "planned" for event in events)
    assert all(
        event.kind != "plan_candidate_source_bound" for event in events
    )
    assert events[-1].affected_versions == (latest.version,)
    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "validate_blocked"
    assert summary["version"] == latest.version


@pytest.mark.parametrize(
    (
        "accepted",
        "gate_passes",
        "concurrent_advance",
        "expected_rc",
        "expected_status",
    ),
    [
        (True, True, False, 0, "accepted"),
        (True, False, False, 1, "validate_blocked"),
        (False, True, False, 2, "candidate_revision_required"),
        (True, True, True, 3, "concurrent_spec_change"),
        (False, True, True, 3, "concurrent_spec_change"),
        (True, False, True, 3, "concurrent_spec_change"),
    ],
)
@pytest.mark.parametrize("replacement", [False, True])
def test_review_candidate_cli_never_acquires_planner_and_commits_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted: bool,
    gate_passes: bool,
    concurrent_advance: bool,
    expected_rc: int,
    expected_status: str,
    replacement: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "thing.py").write_text("value = 1\n")
    spec_dir = tmp_path / "corpus"
    corpus = Corpus(spec_dir)
    base = _spec()
    if replacement:
        prior = _candidate_payload()
        prior["tiers"][0]["slices"][0]["directive"]["intent"] = "Retain the prior plan."
        base = replace(base, tier_dag=plan_workflow._strict_review_candidate_dag(prior))
    corpus.put_spec(base)
    candidate_path = tmp_path / "candidate.json"
    raw_candidate = json.dumps(
        _candidate_payload(),
        separators=(",", ":"),
    ).encode("utf-8")
    candidate_path.write_bytes(raw_candidate)
    acquired_roles: list[str] = []
    seen: dict[str, object] = {}

    class _Pool:
        def get(self, role: str, **_kwargs):
            acquired_roles.append(role)
            if role == "plan_planner":
                raise AssertionError("fixed candidate acquired plan_planner")
            return object()

    @contextmanager
    def _pool(_args, _run_dir, *, cwd=None):
        seen["backend_cwd"] = cwd
        yield _Pool()

    def _run_plan(spec, **kwargs):
        seen["planner_backend"] = kwargs["planner_backend"]
        seen["candidate_dag"] = kwargs["candidate_dag"]
        seen["repair_attempts"] = kwargs[
            "max_candidate_validation_repairs"
        ]
        dag = kwargs["candidate_dag"]
        candidate_ref = kwargs["corpus"].put_artifact(
            b"canonical candidate",
            kind="plan_candidates",
        )
        if concurrent_advance:
            kwargs["corpus"].commit(lifecycle_events=(LifecycleEvent(
                kind="concurrent_update",
                reason="advance while candidate review is in flight",
                affected_versions=(1,),
                execution=plan_workflow.exec_ctx_now("test"),
            ),))
        return PlanOutcome(
            spec=replace(spec, tier_dag=dag if accepted else None),
            candidate_dag=dag,
            candidate_ref=candidate_ref,
            decision="accepted" if accepted else "rejected",
            status="accepted" if accepted else expected_status,
            rounds_used=1,
            final_critic_decision="pass" if accepted else "revise",
            final_judge_decision="pass",
            reason="stubbed fixed-candidate adjudication",
            cost_usd=0.0,
        )

    def _validate(_spec, *, ruleset, ctx):
        del ruleset, ctx
        artifact_ref = corpus.put_artifact(
            b"validation input",
            kind="review_inputs",
        )
        return Verdict(
            rubric="plan_gate",
            decision="pass" if gate_passes else "reject",
            findings=(
                ()
                if gate_passes
                else (Finding(
                    id="final_workspace_drift",
                    severity="High",
                    category="other",
                    evidence=(Evidence(
                        kind="filesystem_check",
                        location="src/thing.py",
                        excerpt="workspace changed after review",
                    ),),
                    suggested_action="review a new candidate invocation",
                    source="deterministic",
                ),)
            ),
            artifact_ref=artifact_ref,
            artifact_commit=None,
            reasoning="deterministic validation passed",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("tier_dag",),
                omissions=(),
            ),
            backend_name=None,
            transcript_ref=None,
            cost_usd=0.0,
            execution=None,
            kind="validate",
        )

    monkeypatch.setattr(plan_workflow, "backend_pool", _pool)
    monkeypatch.setattr(plan_workflow, "run_plan", _run_plan)
    monkeypatch.setattr(plan_workflow, "validate", _validate)
    monkeypatch.setattr(plan_workflow, "available_rules", lambda: ())
    monkeypatch.setattr(
        plan_workflow,
        "plan_contract_fingerprint",
        lambda *_args, **_kwargs: "sha256:" + "a" * 64,
    )
    monkeypatch.setattr(
        plan_workflow,
        "bind_plan_contract",
        lambda _corpus, staged, *, fingerprint: staged,
    )

    rc = plan_workflow.main([
        "--spec",
        str(spec_dir),
        "--cwd",
        str(workspace),
        "--review-candidate",
        str(candidate_path),
        *(
            ["--replace-plan-digest", str(hash_record(base.tier_dag))]
            if replacement else []
        ),
    ])

    assert rc == expected_rc
    assert acquired_roles == ["plan_critic", "plan_judge"]
    assert seen["planner_backend"] is None
    assert seen["repair_attempts"] == 0
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.version == 1
    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == expected_status
    assert summary["candidate_source_mode"] == "operator_file"
    assert summary["planner_calls"] == 0
    assert summary["critic_adjudications"] == 1
    assert summary["judge_adjudications"] == 1
    assert summary.get("replaces_tier_dag_digest") == (
        hash_record(base.tier_dag) if replacement else None
    )
    assert latest.intent == base.intent
    assert latest.project_context == base.project_context
    assert latest.constraints == base.constraints
    assert latest.done_criteria == base.done_criteria
    assert corpus.get_spec(0) == base
    if concurrent_advance:
        assert latest.tier_dag == base.tier_dag
        concurrent_events = [
            corpus.get_record(ref, as_type=LifecycleEvent)
            for ref in latest.lifecycle_event_refs
        ]
        assert [event.kind for event in concurrent_events] == [
            "concurrent_update"
        ]
        return

    assert latest.tier_dag == (
        seen["candidate_dag"] if accepted and gate_passes else base.tier_dag
    )
    if replacement and not (accepted and gate_passes):
        assert summary["retained_tier_dag_digest"] == hash_record(base.tier_dag)
    events = [
        corpus.get_record(ref, as_type=LifecycleEvent)
        for ref in latest.lifecycle_event_refs
    ]
    assert [event.kind for event in events] == (
        ["plan_rejected", "plan_candidate_source_bound"]
        if accepted and not gate_passes
        else ["plan_candidate_source_bound"]
    )
    source_event = next(
        event
        for event in events
        if event.kind == "plan_candidate_source_bound"
    )
    source_payload = json.loads(
        source_event.reason.removeprefix("fixed plan candidate source: ")
    )
    assert source_event.affected_versions == (latest.version,)
    assert source_payload["planner_calls"] == 0
    assert source_payload["base_spec_digest"].startswith("sha256:")
    assert source_payload["base_spec_digest"] == hash_record(base)
    assert source_payload.get("replaces_tier_dag_digest") == (
        hash_record(base.tier_dag) if replacement else None
    )
    assert source_payload["canonical_candidate_ref"].startswith("sha256:")
    assert source_payload["decision"] == (
        "rejected" if accepted and not gate_passes else (
            "accepted" if accepted else "rejected"
        )
    )
    assert source_payload["raw_candidate_ref"].startswith("sha256:")
    assert corpus.get_artifact(source_payload["raw_candidate_ref"]) == (
        raw_candidate
    )

@pytest.mark.parametrize(
    "conflict",
    [
        ["--enable-source-tools"],
        ["--no-validate-gate"],
        ["--max-validate-replan-attempts", "1"],
        ["--planner-session-id", "existing-session"],
        ["--enable-planner-control"],
    ],
)
def test_review_candidate_cli_rejects_planner_authority_flags(
    tmp_path: Path,
    conflict: list[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        plan_workflow._parse_args([
            "--spec",
            str(tmp_path / "spec"),
            "--review-candidate",
            str(tmp_path / "candidate.json"),
            *conflict,
        ])

    assert exc_info.value.code == 2


def test_plan_cli_rejects_multiple_candidate_repairs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        plan_workflow._parse_args([
            "--spec", str(tmp_path / "spec"),
            "--max-validate-replan-attempts", "2",
        ])
    assert error.value.code == 2


def test_run_plan_rejects_multiple_candidate_repairs_before_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    def forbidden_plan(*args, **kwargs):
        pytest.fail("invalid repair allowance reached the generic planner")

    monkeypatch.setattr(plan_workflow, "plan", forbidden_plan)
    with pytest.raises(
        ValueError,
        match="max_candidate_validation_repairs must be 0 or 1",
    ):
        plan_workflow.run_plan(
            _spec(),
            planner_backend=SimpleNamespace(name="planner"),
            critic_backend=object(),
            judge_backend=None,
            corpus=object(),
            critic_rubric=object(),
            max_candidate_validation_repairs=2,
        )
