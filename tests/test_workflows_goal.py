# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""End-to-end smoke tests for ``crucible.workflows.goal``.

Stubs the Claude backend with a ``ScriptedBackend`` returning canned
Red / Blue / Judge JSON responses. Verifies:

- Red emits only Medium findings → workflow short-circuits with rc=0
  (no Blue / Judge invocation).
- Multi-round path: Red surfaces High → Blue revises → Judge proceeds → rc=0.
- Judge ``reject`` short-circuits with rc=2.
- ``--write-revised`` flag emits ``<spec>/revised_spec.yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BudgetConstraint,
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    Message,
    ProjectContext,
    Spec,
    Usage,
    Verdict,
)
from assay.corpus.store import Corpus
from ._review_v2_adapter import adapt_scripted_review_response


class ScriptedBackend:
    name = "stub"
    model = "stub-model-v1"

    def __init__(
        self,
        responses: list[str],
        *,
        accounting_debt_on_calls: set[int] | None = None,
    ):
        self._responses = list(responses)
        self._accounting_debt_on_calls = accounting_debt_on_calls or set()
        self.calls: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.calls.append(invocation)
        if not self._responses:
            raise AssertionError(
                f"ScriptedBackend exhausted on call {len(self.calls)}"
            )
        final = adapt_scripted_review_response(self._responses.pop(0), invocation)
        accounting_complete = (
            len(self.calls) not in self._accounting_debt_on_calls
        )
        return InvocationResult(
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
            usage=Usage(input_tokens=10, output_tokens=20, cost_usd=0.01),
            stop_reason="end_turn",
            completeness=ExecutionCompleteness(
                token_usage_complete=accounting_complete,
                cost_complete=accounting_complete,
                terminal_complete=True,
                trace_complete=True,
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                events=(
                    BackendExecutionEvent(
                        sequence=0,
                        kind="terminal",
                        name="end_turn",
                    ),
                ),
            ),
        )


# ----------------------------------------------------------------------
# Canned rubric responses
# ----------------------------------------------------------------------


def _red_only_medium() -> str:
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "goal is fine",
        "coverage": {"complete": True, "sources_reviewed": ["artifact"],
                     "omissions": []},
        "findings": [{
            "id": "minor_style", "criterion_id": "intent_clarity",
            "severity": "Medium", "confidence": 0.7,
            "disposition": "confirmed", "category": "ambiguity",
            "targets": [],
            "evidence": [{"source_id": "artifact", "kind": "quote",
                          "location": "intent", "excerpt": "renderer faster"}],
            "proposal": {"action": "tighten the prose", "replacement": None},
            "attack_spec": None,
        }],
    })


def _red_with_high(*ids: str, excerpt: str = "renderer faster") -> str:
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "revise",
        "reasoning": "goal has gaps",
        "coverage": {"complete": True, "sources_reviewed": ["artifact"],
                     "omissions": []},
        "findings": [
            {
                "id": rid, "criterion_id": "constraint_completeness",
                "severity": "High", "confidence": 0.9,
                "disposition": "confirmed", "category": "missing_constraint",
                "targets": [],
                "evidence": [{"source_id": "artifact",
                              "kind": "intent_excerpt", "location": "intent",
                              "excerpt": excerpt}],
                "proposal": {"action": "add a perf budget", "replacement": None},
                "attack_spec": None,
            }
            for rid in ids
        ],
    })


def _blue_revises(
    *ids: str,
    intent: str = "make the renderer faster within the declared budget",
    excerpt: str = "renderer faster",
) -> str:
    replacement = json.dumps({
        "intent": intent,
        "constraints": [{"kind": "budget", "usd_limit": 50.0,
                         "time_limit_hours": None}],
        "done_criteria": [],
    })
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "revisions proposed",
        "coverage": {"complete": True, "sources_reviewed": [
                         "artifact", *(f"prior:red:{rid}" for rid in ids),
                     ],
                     "omissions": []},
        "findings": [{
            "id": "canonical_revision",
            "criterion_id": "red_coverage",
            "severity": "Low", "confidence": 0.95,
            "disposition": "addressed", "category": "canonical_revision",
            "targets": list(ids),
            "evidence": [{"source_id": "artifact", "kind": "quote",
                          "location": "intent", "excerpt": excerpt}],
            "proposal": {"action": "apply canonical revision",
                         "replacement": replacement},
            "attack_spec": None,
        }],
    })


def _blue_invalid_revision(*ids: str) -> str:
    payload = json.loads(_blue_revises(*ids))
    replacement = json.loads(payload["findings"][0]["proposal"]["replacement"])
    replacement["intent"] = ""
    payload["findings"][0]["proposal"]["replacement"] = json.dumps(replacement)
    return json.dumps(payload)


def _judge_sources(
    ids: tuple[str, ...],
    *,
    round_index: int = 0,
    validation_ids: tuple[str, ...] = (),
    earlier_sources: tuple[str, ...] = (),
) -> list[str]:
    return [
        "artifact",
        *earlier_sources,
        *(f"prior:round_{round_index}.red:{rid}" for rid in ids),
        f"prior:round_{round_index}.blue:canonical_revision",
        *(
            f"prior:round_{round_index}.validation:{finding_id}"
            for finding_id in validation_ids
        ),
    ]


def _judge_proceed(
    *ids: str,
    validation_ids: tuple[str, ...] = (),
) -> str:
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "goal is ready",
        "coverage": {"complete": True, "sources_reviewed": _judge_sources(
                         ids, validation_ids=validation_ids,
                     ),
                     "omissions": []},
        "findings": [{
            "id": "goal_ready", "criterion_id": "intent_clarity",
            "severity": "Low", "confidence": 0.95,
            "disposition": "addressed", "category": "ready",
            "targets": [*ids, "canonical_revision", *validation_ids],
            "evidence": [{"source_id": "artifact", "kind": "quote",
                          "location": "goal", "excerpt": "budget"}],
            "proposal": {"action": "advance to plan", "replacement": None},
            "attack_spec": None,
        }],
    })


def _judge_reject(*ids: str) -> str:
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "reject",
        "reasoning": "infeasible",
        "coverage": {"complete": True,
                     "sources_reviewed": _judge_sources(ids),
                     "omissions": []},
        "findings": [{
            "id": "infeasible_goal", "criterion_id": "feasibility",
            "severity": "Critical", "confidence": 0.95,
            "disposition": "confirmed", "category": "infeasible",
            "targets": [*ids, "canonical_revision"],
            "evidence": [{"source_id": "artifact", "kind": "quote",
                          "location": "goal", "excerpt": "budget"}],
            "proposal": {"action": "re-author goal", "replacement": None},
            "attack_spec": None,
        }],
    })


def _judge_revise(
    reasoning: str,
    *ids: str,
    round_index: int = 0,
    earlier_sources: tuple[str, ...] = (),
) -> str:
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "revise",
        "reasoning": reasoning,
        "coverage": {"complete": True,
                     "sources_reviewed": _judge_sources(
                         ids,
                         round_index=round_index,
                         earlier_sources=earlier_sources,
                     ),
                     "omissions": []},
        "findings": [{
            "id": "remaining_gap", "criterion_id": "red_disposition",
            "severity": "High", "confidence": 0.9,
            "disposition": "confirmed", "category": "unaddressed_red",
            "targets": [
                *ids,
                "canonical_revision",
                *(["remaining_gap"] if round_index > 0 else []),
            ],
            "evidence": [{"source_id": "artifact", "kind": "quote",
                          "location": "goal", "excerpt": "budget"}],
            "proposal": {"action": "another round", "replacement": None},
            "attack_spec": None,
        }],
    })


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _starter_spec() -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="goal_test",
            repo_url="https://example.invalid/gt.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="make the renderer faster",
        constraints=(BudgetConstraint(usd_limit=50.0),),
        done_criteria=(),
        plan_policy=None,
    )


def _setup_corpus(tmp_path: Path) -> Corpus:
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_starter_spec())
    return corpus


def _patch_backend(monkeypatch, scripted: ScriptedBackend) -> None:
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, routing=None, log_dir=None, cwd=None, config_path=None, spec_dir=None, **kw: scripted,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_goal_short_circuits_when_red_finds_only_medium(
    tmp_path: Path, monkeypatch,
):
    """A clean Red review still receives an explicit Blue/Judge decision."""
    corpus = _setup_corpus(tmp_path)
    scripted = ScriptedBackend([
        _red_only_medium(),
        _blue_revises("minor_style"),
        _judge_proceed("minor_style"),
    ])
    _patch_backend(monkeypatch, scripted)

    from crucible.workflows.goal import main

    rc = main([
        "--spec", str(corpus.spec_dir),
        "--max-rounds", "3",
    ])
    assert rc == 0
    assert len(scripted.calls) == 3


def test_goal_full_round_proceeds(tmp_path: Path, monkeypatch):
    """Red surfaces High → Blue revises → Judge proceeds → rc=0."""
    corpus = _setup_corpus(tmp_path)
    scripted = ScriptedBackend([
        _red_with_high("missing_perf_budget"),
        _blue_revises("missing_perf_budget"),
        _judge_proceed("missing_perf_budget"),
    ])
    _patch_backend(monkeypatch, scripted)

    from crucible.workflows.goal import main

    rc = main([
        "--spec", str(corpus.spec_dir),
        "--max-rounds", "3",
    ])
    assert rc == 0
    spec = corpus.latest_spec()
    # Red + Blue + deterministic revision validation + Judge.
    assert len(spec.verdict_refs) == 4
    assert len(spec.lifecycle_event_refs) == 1
    # Evidence, approved goal, and the race-correct lifecycle marker share one
    # visibility commit; no intermediate authoritative version can leak.
    assert spec.version == 1
    evidence_commit = json.loads(
        next((corpus.spec_dir / "versions").glob("v1_*.json")).read_text()
    )
    assert evidence_commit["intent"] == (
        "make the renderer faster within the declared budget"
    )
    assert len(evidence_commit["verdict_refs"]) == 4
    assert evidence_commit["lifecycle_event_refs"] == [
        str(spec.lifecycle_event_refs[0])
    ]
    assert spec.intent == "make the renderer faster within the declared budget"
    run_dir = max((corpus.spec_dir / "runs").iterdir())
    final_goal = run_dir / "final_goal.yaml"
    assert final_goal.is_file()
    payload = yaml.safe_load(final_goal.read_text())
    assert payload["intent"] == spec.intent


def test_goal_reject_short_circuits(tmp_path: Path, monkeypatch):
    """Judge issues reject → rc=2; loop halts immediately."""
    corpus = _setup_corpus(tmp_path)
    scripted = ScriptedBackend([
        _red_with_high("infeasible_target"),
        _blue_revises("infeasible_target"),
        _judge_reject("infeasible_target"),
    ])
    _patch_backend(monkeypatch, scripted)

    from crucible.workflows.goal import main

    rc = main([
        "--spec", str(corpus.spec_dir),
        "--max-rounds", "3",
    ])
    assert rc == 2


def test_goal_max_rounds_exhausted(tmp_path: Path, monkeypatch):
    """If Judge never proceeds, workflow runs all rounds then returns rc=1."""
    corpus = _setup_corpus(tmp_path)
    scripted = ScriptedBackend([
        # Round 0: Red high, Blue revises, Judge withholds.
        _red_with_high("p0"),
        _blue_revises("p0", intent="first staged revision"),
        _judge_revise("needs another round", "p0"),
        # Round 1: same.
        _red_with_high("p0", excerpt="first staged revision"),
        _blue_revises(
            "p0",
            intent="second staged revision",
            excerpt="first staged revision",
        ),
        _judge_revise(
            "still needs work",
            "p0",
            round_index=1,
            earlier_sources=(
                "prior:round_0.red:p0",
                "prior:round_0.blue:canonical_revision",
                "prior:round_0.judge:remaining_gap",
            ),
        ),
    ])
    _patch_backend(monkeypatch, scripted)

    from crucible.workflows.goal import main

    rc = main([
        "--spec", str(corpus.spec_dir),
        "--max-rounds", "2",
    ])
    assert rc == 1
    # Revise is a draft-stage signal, not authority to publish either draft.
    assert corpus.latest_spec().intent == "make the renderer faster"
    run_dir = max((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["promotions"] == 0
    assert summary["staged_revisions"] == 2
    # Both round-1 reviewers receive the actual staged round-0 candidate.
    assert "first staged revision" in scripted.calls[3].user
    assert "first staged revision" in scripted.calls[4].user


def test_goal_judge_cannot_override_failed_deterministic_validation(
    tmp_path: Path, monkeypatch,
):
    corpus = _setup_corpus(tmp_path)
    scripted = ScriptedBackend([
        _red_with_high("invalid_revision"),
        _blue_invalid_revision("invalid_revision"),
        _judge_proceed(
            "invalid_revision",
            validation_ids=("goal_revision_gate_reject",),
        ),
    ])
    _patch_backend(monkeypatch, scripted)

    from crucible.workflows.goal import main

    rc = main([
        "--spec", str(corpus.spec_dir),
        "--max-rounds", "1",
    ])
    assert rc == 3
    assert corpus.latest_spec().intent == "make the renderer faster"
    run_dir = max((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["promotions"] == 0
    assert summary["rounds"][0]["validation_decision"] == "reject"
    assert summary["rounds"][0]["judge_decision"] == "inconclusive"


def test_goal_red_accounting_debt_stops_before_blue_and_cannot_promote(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _setup_corpus(tmp_path)
    scripted = ScriptedBackend(
        [_red_only_medium()],
        accounting_debt_on_calls={1},
    )
    _patch_backend(monkeypatch, scripted)

    from crucible.workflows.goal import main

    assert main([
        "--spec", str(corpus.spec_dir),
        "--max-rounds", "3",
    ]) == 3
    assert len(scripted.calls) == 1
    spec = corpus.latest_spec()
    assert spec is not None
    assert spec.intent == "make the renderer faster"
    verdicts = [
        corpus.get_record(ref, as_type=Verdict)
        for ref in spec.verdict_refs
    ]
    assert verdicts[-1].decision == "inconclusive"
    assert verdicts[-1].rubric == "goal_accounting"
    assert all(verdict.rubric != "goal_budget" for verdict in verdicts)
    run_dir = max((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "inconclusive"
    assert summary["promotions"] == 0
    assert summary["costs"]["total_usd"] == pytest.approx(0.01)
    assert summary["costs"]["remaining_usd"] == 0.0
    assert summary["costs"]["accounting_debt_by_role"] == {
        "goal_red": ["cost_complete", "token_usage_complete"],
    }
