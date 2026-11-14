# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed accounting boundaries for the generic review workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assay import (
    ExecutionCompleteness,
    ProjectContext,
    ReviewCoverage,
    Spec,
    Verdict,
)
from assay.corpus.store import Corpus
from assay.primitives.review import MAX_CHUNK_RESPONSE_RETRY_BUDGET


class _Backend:
    name = "stub"
    model = "stub-model"

    def close(self) -> None:
        pass


def _corpus(tmp_path: Path) -> Corpus:
    corpus = Corpus(tmp_path / "spec")
    corpus.put_spec(Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="test",
            repo_url="https://example.invalid/repo.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="review the artifact",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
    ))
    return corpus


def test_response_retry_budget_cli_is_explicit_and_nonnegative(
    tmp_path: Path,
) -> None:
    from crucible.workflows.review import _parse_args

    base = [
        "--spec", str(tmp_path / "spec"),
        "--rubric", "spec_clarity",
        "--artifact", str(tmp_path / "artifact.md"),
    ]
    assert _parse_args(base).response_retry_budget == 0
    explicit = _parse_args([*base, "--response-retry-budget", "2"])
    assert explicit.response_retry_budget == 2
    maximum = _parse_args([
        *base,
        "--response-retry-budget",
        str(MAX_CHUNK_RESPONSE_RETRY_BUDGET),
    ])
    assert maximum.response_retry_budget == MAX_CHUNK_RESPONSE_RETRY_BUDGET
    with pytest.raises(SystemExit) as exc:
        _parse_args([*base, "--response-retry-budget", "-1"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        _parse_args([
            *base,
            "--response-retry-budget",
            str(MAX_CHUNK_RESPONSE_RETRY_BUDGET + 1),
        ])
    assert exc.value.code == 2


def test_generic_review_uses_bounded_autonomous_chunk_envelope(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _corpus(tmp_path)
    artifact = tmp_path / "artifact.md"
    artifact.write_text("A complete artifact.\n", encoding="utf-8")
    pass_verdict = Verdict(
        rubric="spec_clarity",
        decision="pass",
        findings=(),
        artifact_ref=corpus.put_artifact(
            artifact.read_bytes(),
            kind="review_inputs",
            name_hint=artifact.name,
        ),
        artifact_commit=None,
        reasoning="complete",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
            artifact_chars_total=len(artifact.read_text()),
            artifact_chars_reviewed=len(artifact.read_text()),
        ),
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _Backend(),
    )
    review_kwargs: dict[str, object] = {}

    def fake_review(*args, **kwargs):
        review_kwargs.update(kwargs)
        return pass_verdict

    monkeypatch.setattr("crucible.workflows.review.review", fake_review)

    from crucible.workflows.review import main

    assert main([
        "--spec", str(corpus.spec_dir),
        "--rubric", "spec_clarity",
        "--artifact", str(artifact),
    ]) == 0
    assert review_kwargs["chunk_chars"] == 80_000


def test_generic_review_nominal_pass_with_accounting_debt_is_inconclusive(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _corpus(tmp_path)
    artifact = tmp_path / "artifact.md"
    artifact.write_text("A complete artifact.\n", encoding="utf-8")
    artifact_ref = corpus.put_artifact(
        artifact.read_bytes(),
        kind="review_inputs",
        name_hint=artifact.name,
    )
    pass_with_debt = Verdict(
        rubric="spec_clarity",
        decision="pass",
        findings=(),
        artifact_ref=artifact_ref,
        artifact_commit=None,
        reasoning="the artifact satisfies every criterion",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
            artifact_chars_total=len(artifact.read_text()),
            artifact_chars_reviewed=len(artifact.read_text()),
        ),
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.42,
        execution=None,
        input_tokens=120,
        output_tokens=30,
        execution_completeness=ExecutionCompleteness(
            token_usage_complete=False,
            cost_complete=False,
            terminal_complete=True,
            trace_complete=True,
        ),
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _Backend(),
    )
    review_kwargs: dict[str, object] = {}

    def fake_review(*args, **kwargs):
        review_kwargs.update(kwargs)
        return pass_with_debt

    monkeypatch.setattr("crucible.workflows.review.review", fake_review)

    from crucible.workflows.review import main

    assert main([
        "--spec", str(corpus.spec_dir),
        "--rubric", "spec_clarity",
        "--artifact", str(artifact),
        "--response-retry-budget", "2",
    ]) == 3
    assert review_kwargs["response_retry_budget"] == 2

    run_dir = max((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert summary["decision"] == "inconclusive"
    assert summary["passed"] is False
    assert summary["cost_usd"] == pytest.approx(0.42)
    assert summary["accounting_debt"] == [
        "token_usage_complete",
        "cost_complete",
    ]
    assert summary["costs"]["total_usd"] == pytest.approx(0.42)
    assert summary["costs"]["remaining_usd"] == 0.0
    assert summary["costs"]["accounting_debt_by_role"] == {
        "spec_clarity": ["cost_complete", "token_usage_complete"],
    }
    assert "only a lower bound" in summary["reasoning"]
    assert result["status"] == "inconclusive"
    assert result["cost_usd"] == pytest.approx(0.42)

    committed = Corpus(corpus.spec_dir)
    latest = committed.latest_spec()
    assert latest is not None
    persisted = committed.get_record(
        latest.verdict_refs[-1],
        as_type=Verdict,
    )
    assert persisted.decision == "inconclusive"
    assert persisted.cost_usd == pytest.approx(0.42)
    assert persisted.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
