# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""End-to-end integration test for crucible workflows.

Stands up a tmp git repo + Corpus, swaps `crucible.backends.ClaudeBackend`
for a `ScriptedBackend` stub (no live Claude API), and drives the
architect workflow (plan → critic → implement chain) through to a
persisted Spec version. Validates that the assay primitives + workflows
+ rubrics + backend protocol all wire together correctly.

The stub backend returns canned JSON responses per call:
  call 0 → planner emits a TierDAG JSON
  call 1 → critic emits a passing verdict
  call 2 → implementer emits "DONE" (no actual file mutation in this
           test — the implement primitive will classify as
           `soft_fail_no_changes`, which is exactly what we want to
           verify the workflow correctly reports)

Verifies:
  - The workflow runs end-to-end without exceptions.
  - The Corpus has a new Spec version with a tier_dag stamped.
  - At least one critic Verdict was persisted under verdict_refs.
  - The Implementation was persisted under implementation_refs with the
    expected classification.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    Message,
    ProjectContext,
    Spec,
    Usage,
)
from assay.corpus.store import Corpus
from ._review_v2_adapter import adapt_scripted_review_response
from assay.types.artifacts import Implementation, LifecycleEvent
from assay.types.execution import ExecutionContext
from assay.types.findings import Verdict


_DAG_JSON = json.dumps(
    {
        "tiers": [
            {
                "tier_id": "t0",
                "depends_on": [],
                "slices": [
                    {
                        "id": "s0",
                        "directive": {
                            "slice_id": "s0",
                            "intent": "create greet.py with hello() returning 'hi'",
                            "requirement_refs": [
                                "add a hello() function returning the string 'hi'"
                            ],
                            "files_in_scope": [],
                            "creates_paths": ["greet.py"],
                            "verify_mode": "tests",
                            "blast_radius": {"max_files": 1},
                        },
                    }
                ],
            }
        ]
    }
)

def _review_pass(*sources: str) -> str:
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "complete and valid",
        "coverage": {
            "complete": True,
            "sources_reviewed": list(sources),
            "omissions": [],
        },
        "findings": [],
    })


def _route_backends(monkeypatch, backends: dict[str, "ScriptedBackend"]):
    def _resolve(
        role,
        *,
        routing=None,
        log_dir=None,
        cwd=None,
        capability=None,
        native_shell=None,
        config_path=None,
        spec_dir=None,
        allow_trusted_project_instructions=True,
        feature_requirements=(),
    ):
        if role == "implementer":
            assert allow_trusted_project_instructions is False
        return backends[role]

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        _resolve,
    )


class ScriptedBackend:
    """Drop-in backend stub for the architect pipeline.

    Returns scripted final_message strings in order; raises if exhausted.
    Public ``name`` + ``model`` + ``invoke`` to satisfy ``assay.Backend``.
    """

    name = "stub"
    model = "stub-model-v1"

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[Invocation] = []
        self.attached_toolset = None

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        self.attached_toolset = toolset

    def close(self) -> None:
        pass

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.calls.append(invocation)
        if not self._responses:
            raise AssertionError(
                f"ScriptedBackend exhausted on call {len(self.calls)}"
            )
        final = adapt_scripted_review_response(self._responses.pop(0), invocation)
        return InvocationResult(
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
            usage=Usage(input_tokens=10, output_tokens=20, cost_usd=0.01),
            stop_reason="end_turn",
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


def _init_repo(path: Path) -> str:
    """Set up an empty git repo and return its HEAD sha."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@assay"], cwd=path, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "assay test"], cwd=path, check=True
    )
    (path / "README.md").write_text("seed\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _starter_spec(base_commit: str) -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="e2e_demo",
            repo_url="https://example.invalid/e2e_demo.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="add a hello() function returning the string 'hi'",
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


def test_architect_recipe_end_to_end(tmp_path: Path, monkeypatch):
    """architect: plan → implement persists tier_dag + verdict + implementation."""
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)

    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_starter_spec(base))

    # Planner, Critic, and Judge are independent identities. The tier owns a
    # fourth Implementer backend.
    backends = {
        "architect_planner": ScriptedBackend([_DAG_JSON]),
        "architect_critic": ScriptedBackend([_review_pass(
            "artifact",
            "context:spec_intent",
            "context:spec_constraints",
            "context:spec_done_criteria",
        )]),
        "architect_judge": ScriptedBackend([_review_pass(
            "artifact",
            "context:spec_intent",
            "context:spec_constraints",
            "context:spec_done_criteria",
            "context:prior_critic_findings",
            "context:prior_judge_findings",
        )]),
        "implementer": ScriptedBackend([
            "DONE — stubbed implementer.",
            "DONE — focused recovery also made no edit.",
        ]),
    }
    _route_backends(monkeypatch, backends)

    # The implementer primitive shells out to git in cwd; chdir into the
    # repo so its `git rev-parse HEAD` and `git diff HEAD` work.
    monkeypatch.chdir(repo)

    from crucible.workflows.architect import main

    rc = main(
        [
            "--planner-count", "1",
            "--spec",
            str(corpus_dir),
            "--critic-rubric",
            "plan_critic",
            "--judge-rubric",
            "plan_judge",
        ]
    )

    # The implementer returned without modifying files, then exhausted the
    # one bounded empty-diff recovery in the same way. The workflow exits 1
    # for the terminal non-success classification.
    assert rc == 1, (
        f"expected rc=1 (empty-diff failure), got {rc}; "
        "unexpected architect exit code"
    )
    assert {role: len(backend.calls) for role, backend in backends.items()} == {
        "architect_planner": 1,
        "architect_critic": 1,
        "architect_judge": 1,
        "implementer": 2,
    }
    assert [call.max_turns for call in backends["implementer"].calls] == [
        50,
        8,
    ]

    # Latest Spec has tier_dag stamped + verdict + implementation refs.
    final = corpus.latest_spec()
    assert final is not None
    assert final.tier_dag is not None
    assert len(final.tier_dag.tiers) == 1
    assert final.tier_dag.tiers[0].slices[0].id == "s0"
    assert final.tier_dag.tiers[0].slices[0].directive.requirement_refs == (
        "add a hello() function returning the string 'hi'",
    )
    assert len(final.verdict_refs) >= 1  # at least the critic verdict
    assert len(final.implementation_refs) == 1

    from crucible.rubrics import load
    from crucible.workflows._plan_provenance import (
        accepted_plan_contract_state,
        plan_contract_fingerprint,
    )

    fingerprint = plan_contract_fingerprint(
        final,
        critic_rubric=load("plan_critic"),
        judge_rubric=load("plan_judge"),
        candidate_transform=None,
        cwd=repo,
    )
    assert accepted_plan_contract_state(
        corpus,
        final,
        expected_fingerprint=fingerprint,
    ).status == "fresh"

    # Each ref is canonical and resolves to its typed persisted record.
    for ref in final.verdict_refs:
        assert ref.startswith("sha256:")
        assert isinstance(corpus.get_record(ref, as_type=Verdict), Verdict)
    for ref in final.implementation_refs:
        assert ref.startswith("sha256:")
        assert isinstance(
            corpus.get_record(ref, as_type=Implementation),
            Implementation,
        )


def test_architect_recipe_returns_2_on_planner_failure(tmp_path: Path, monkeypatch):
    """When the planner fails to produce a parseable DAG, workflow exits 2."""
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_starter_spec(base))

    # Planner returns unstructured text → no TierDAG. The default
    # PlanPolicy.max_critique_rounds=4 (restored pre-cutover value
    # — Stage 1.30b) means up to 5 planner rounds (initial + 4
    # critiques). Script that many responses so the loop reaches
    # the exhausted state cleanly without the ScriptedBackend
    # raising ``exhausted on call N`` mid-loop.
    planner = ScriptedBackend([
        "not a json object at all",
        "still not a json object after the rebuke",
        "third attempt, still garbage",
        "fourth attempt, garbage continues",
        "fifth attempt, final round, still no JSON",
    ])
    _route_backends(monkeypatch, {
        "architect_planner": planner,
        "architect_critic": ScriptedBackend([]),
        "architect_judge": ScriptedBackend([]),
    })
    monkeypatch.chdir(repo)

    from crucible.workflows.architect import main

    rc = main([
        "--planner-count", "1", "--spec", str(corpus_dir),
        "--critic-rubric", "plan_critic",
    ])
    assert rc == 2  # plan_failed


def test_architect_recipe_reuses_accepted_plan(tmp_path: Path, monkeypatch):
    """An accepted planned lifecycle reuses its DAG without a plan call."""
    from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG

    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    corpus = Corpus(corpus_dir)
    seeded = _starter_spec(base)
    corpus.put_spec(seeded)
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    Slice(
                        id="s0",
                        directive=ImplementationDirective(
                            slice_id="s0",
                            intent="i",
                            files_in_scope=("greet.py",),
                            creates_paths=(),
                            verify_mode="tests",
                            blast_radius={},
                        ),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    planned_event = LifecycleEvent(
        kind="planned",
        reason="accepted test plan",
        affected_versions=(seeded.version + 1,),
        execution=ExecutionContext(
            timestamp="2026-06-28T00:00:00+00:00",
            machine="test",
            cwd=str(repo),
            toolchain={},
            backend=None,
            model=None,
            recipe="plan",
            recipe_commit=None,
            gpu_model=None,
            run_id=None,
        ),
    )
    event_ref = corpus.put_record(planned_event, kind="lifecycle_events")
    staged = seeded.with_appended_refs(
        tier_dag=dag, lifecycle_event_refs=(event_ref,),
    )
    from crucible.rubrics import load
    from crucible.workflows._plan_provenance import (
        bind_plan_contract,
        plan_contract_fingerprint,
    )

    fingerprint = plan_contract_fingerprint(
        seeded,
        critic_rubric=load("plan_critic"),
        judge_rubric=load("plan_judge"),
        candidate_transform=None,
        cwd=repo,
    )
    corpus.put_spec(bind_plan_contract(
        corpus,
        staged,
        fingerprint=fingerprint,
    ))

    # Only implementer calls are expected — the initial no-op plus one
    # bounded empty-diff recovery, with no planner / critic.
    backends = {
        "architect_planner": ScriptedBackend([]),
        "architect_critic": ScriptedBackend([]),
        "architect_judge": ScriptedBackend([]),
        "implementer": ScriptedBackend([
            "DONE — implemented.",
            "DONE — recovery also made no edit.",
        ]),
    }
    _route_backends(monkeypatch, backends)
    monkeypatch.chdir(repo)

    from crucible.workflows.architect import main

    rc = main(
        [
            "--spec",
            str(corpus_dir),
            "--critic-rubric",
            "plan_critic",
            "--skip-validate",
        ]
    )
    assert rc == 1  # bounded recovery still produced no file mutation
    assert len(backends["architect_planner"].calls) == 0
    assert len(backends["architect_critic"].calls) == 0
    assert len(backends["architect_judge"].calls) == 0
    assert len(backends["implementer"].calls) == 2
    assert [call.max_turns for call in backends["implementer"].calls] == [
        50,
        8,
    ]


def test_architect_refuses_legacy_plan_without_explicit_replan(
    tmp_path: Path,
    monkeypatch,
):
    """An unbound accepted DAG stops before any paid planning call."""
    from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG

    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    corpus = Corpus(corpus_dir)
    seeded = _starter_spec(base)
    corpus.put_spec(seeded)
    dag = TierDAG(tiers=(Tier(
        tier_id="t0",
        depends_on=(),
        slices=(Slice(
            id="s0",
            directive=ImplementationDirective(
                slice_id="s0",
                intent="Implement the requested behavior.",
                files_in_scope=("greet.py",),
                creates_paths=(),
                verify_mode="tests",
                blast_radius={},
            ),
        ),),
    ),))
    planned_event = LifecycleEvent(
        kind="planned",
        reason="legacy accepted plan",
        affected_versions=(seeded.version + 1,),
        execution=ExecutionContext(
            timestamp="2026-06-28T00:00:00+00:00",
            machine="test",
            cwd=str(repo),
            toolchain={},
            backend=None,
            model=None,
            recipe="plan",
            recipe_commit=None,
            gpu_model=None,
            run_id=None,
        ),
    )
    event_ref = corpus.put_record(planned_event, kind="lifecycle_events")
    corpus.put_spec(seeded.with_appended_refs(
        tier_dag=dag,
        lifecycle_event_refs=(event_ref,),
    ))

    backends = {
        "architect_planner": ScriptedBackend([]),
        "architect_critic": ScriptedBackend([]),
        "architect_judge": ScriptedBackend([]),
    }
    _route_backends(monkeypatch, backends)
    monkeypatch.chdir(repo)

    from crucible.workflows.architect import main

    rc = main([
        "--spec",
        str(corpus_dir),
        "--critic-rubric",
        "plan_critic",
        "--skip-validate",
    ])

    assert rc == 2
    assert all(not backend.calls for backend in backends.values())


# ----------------------------------------------------------------------
# Unit tests for the two-fact classification helper
# ----------------------------------------------------------------------


# _effective_classification was a helper inside architect.py. After
# Area E (architect → run_tier delegation), the worst-of-two
# classification logic lives inside run_tier's per-slice cascade
# (which has its own test coverage in tests/test_workflows_tier.py).
# The four standalone helper tests previously here are obsolete.
