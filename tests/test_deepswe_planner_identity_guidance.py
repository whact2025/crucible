# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""DeepSWE advertises its existing identity gate without repairing responses."""

from dataclasses import replace
import json

import pytest
from assay import ExecutionCompleteness, InvocationResult, Message, Usage, Verdict
from assay.corpus.store import Corpus
from assay.primitives.plan import _run_planner, _tier_dag_response_schema

from crucible.backends.features import feature_profile_for_route
from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec
from crucible.rubrics import load
from crucible.workflows import plan as workflow
from crucible.workflows._plan_provenance import plan_contract_fingerprint
from crucible.workflows._plan_validation import build_pre_review_candidate_validator
from tests.test_deepswe_spec_synthesis import _instance


_INTENT = "Implement the requested feature."


def _payload(*, directive_id="s0"):
    return {
        "tiers": [
            {
                "tier_id": "t0",
                "depends_on": [],
                "slices": [
                    {
                        "id": "s0",
                        "directive": {
                            "slice_id": directive_id,
                            "intent": _INTENT,
                            "requirement_refs": [_INTENT],
                            "read_only_references": [],
                            "files_in_scope": [],
                            "creates_paths": ["feature.py"],
                            "verify_mode": "tests",
                            "blast_radius": {"max_files": 1, "languages": []},
                        },
                    }
                ],
            }
        ],
    }


class _Planner:
    def __init__(self, name, responses):
        self.name = name
        self.model = "kimi-k3.1-pro" if name == "kimi" else "fixture"
        self.features = feature_profile_for_route(
            name,
            model=self.model,
            base_url="https://api.moonshot.ai/v1",
        )
        self.responses = iter(responses)
        self.invocations = []

    def invoke(self, invocation):
        self.invocations.append(invocation)
        text = next(self.responses)
        return InvocationResult(
            final_message=text,
            transcript=(Message(role="assistant", content=text),),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
            stop_reason="end_turn",
            completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=True,
                terminal_complete=True,
                trace_complete=False,
            ),
        )


@pytest.mark.parametrize("name", ["generic", "kimi"])
@pytest.mark.parametrize("budget", [None, 12.0])
def test_real_planner_invocation_renders_identity_and_json_guidance(
    tmp_path, name, budget
):
    spec = synthesize_starter_spec(
        _instance(instruction=_INTENT), task_budget_usd=budget
    )
    backend = _Planner(name, [json.dumps(_payload())])

    dag, _, error, _ = _run_planner(
        spec,
        backend,
        Corpus(tmp_path / "corpus"),
        (),
        (),
        0,
        7,
        1.0,
        planner_guidance=spec.plan_policy.planner_guidance,
    )

    (invocation,) = backend.invocations
    rendered = " ".join(invocation.user.split())
    assert (
        "`Slice.id` and its `ImplementationDirective.slice_id` MUST be exactly equal"
        in rendered
    )
    assert "copy it into both JSON fields `id` and `directive.slice_id`" in rendered
    assert "including the top-level `tiers` array" in rendered
    assert "one complete replacement TierDAG" in rendered
    assert "never a patch, partial DAG, change log, or prose wrapper" in rendered
    assert ("Kimi planner path and exhaustion protocol" in invocation.system) is (
        name == "kimi"
    )
    assert invocation.response_schema == _tier_dag_response_schema()
    assert invocation.max_turns == 7
    assert invocation.budget_usd == 1.0
    assert error is None
    slice_ = dag.tiers[0].slices[0]
    assert slice_.id == slice_.directive.slice_id == "s0"


@pytest.mark.parametrize("budget", [None, 12.0])
def test_real_planner_invocation_renders_contract_and_capture_guidance(
    tmp_path, budget
):
    spec = synthesize_starter_spec(
        _instance(instruction=_INTENT), task_budget_usd=budget
    )
    backend = _Planner("generic", [json.dumps(_payload())])

    _, _, error, _ = _run_planner(
        spec,
        backend,
        Corpus(tmp_path / "corpus"),
        (),
        (),
        0,
        7,
        1.0,
        planner_guidance=spec.plan_policy.planner_guidance,
    )

    (invocation,) = backend.invocations
    rendered = " ".join(invocation.user.split())
    requirements, policy_heading, policy = rendered.partition(
        "## Product decomposition policy"
    )
    assert policy_heading
    capture = "Plan every new file at a path the repository does not ignore"
    contract = "Edits that form one contract belong in one slice"
    assert capture in policy
    # A compact plan has exactly one slice, so it cannot split a contract.
    assert (contract in policy) is (budget is None)
    # Guidance is product policy; the citable root requirement stays the task.
    assert capture not in requirements
    assert contract not in requirements
    assert spec.intent == _INTENT
    assert error is None


@pytest.mark.parametrize("name", ["generic", "kimi"])
@pytest.mark.parametrize("initial_failure", ["json", "identity"])
def test_guidance_does_not_repair_ids_or_expand_one_correction(
    tmp_path, monkeypatch, name, initial_failure
):
    spec = synthesize_starter_spec(_instance(instruction=_INTENT), task_budget_usd=12.0)
    corpus = Corpus(tmp_path / "corpus")
    corpus.put_spec(spec)
    # The malformed first candidate is missing the outer tiers-array bracket.
    complete = json.dumps(_payload(), separators=(",", ":"))
    initial = (
        complete[:-2] + "}"
        if initial_failure == "json"
        else json.dumps(_payload(directive_id="s0-cache"))
    )
    backend = _Planner(name, [initial, json.dumps(_payload(directive_id="s0-cache"))])

    def no_review(*_args, **_kwargs):
        pytest.fail("an invalid identity reached paid review")

    monkeypatch.setattr(workflow, "strict_review", no_review)
    outcome = workflow.run_plan(
        spec,
        planner_backend=backend,
        critic_backend=object(),
        judge_backend=object(),
        corpus=corpus,
        cwd=str(tmp_path),
        critic_rubric=load("plan_critic"),
        judge_rubric=load("plan_judge"),
        candidate_validator=build_pre_review_candidate_validator(
            spec=spec, project_root=tmp_path
        ),
        max_candidate_validation_repairs=1,
    )

    assert len(backend.invocations) == 2
    assert outcome.status == "planner_contract_failure"
    assert outcome.decision == "rejected"
    assert outcome.spec.tier_dag is None
    assert corpus.latest_spec() == spec
    assert outcome.usage.cost_usd == pytest.approx(0.02)
    slice_ = outcome.candidate_dag.tiers[0].slices[0]
    assert slice_.id == "s0"
    assert slice_.directive.slice_id == "s0-cache"
    verdicts = [
        corpus.get_record(ref, as_type=Verdict) for ref in outcome.spec.verdict_refs
    ]
    assert len(verdicts) == 2
    assert verdicts[-1].findings[0].category == "slice_identity_mismatch"
    assert all(verdict.kind == "validate" for verdict in verdicts)
    for invocation in backend.invocations:
        assert "`Slice.id`" in invocation.user
        assert "`ImplementationDirective.slice_id`" in invocation.user


@pytest.mark.parametrize("budget", [None, 12.0])
def test_product_guidance_is_bound_into_plan_fingerprint(budget):
    spec = synthesize_starter_spec(
        _instance(instruction=_INTENT), task_budget_usd=budget
    )
    legacy = replace(
        spec, plan_policy=replace(spec.plan_policy, planner_guidance="legacy guidance")
    )
    kwargs = {
        "critic_rubric": load("plan_critic"),
        "judge_rubric": load("plan_judge"),
        "candidate_transform": None,
    }
    assert plan_contract_fingerprint(spec, **kwargs) != plan_contract_fingerprint(
        legacy, **kwargs
    )
