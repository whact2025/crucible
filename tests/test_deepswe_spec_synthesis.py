# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.products.deepswe.spec_synthesis``.

Hermetic — constructs ``DeepSWEInstance`` records inline (no
filesystem touches, no docker, no network). Verifies the Spec
shape produced for each supported language + the policy constants.
"""

from __future__ import annotations

import dataclasses

from assay.types.spec import Spec

from crucible.common.policies import BreachPolicy, ReviewPolicy
from crucible.products.deepswe.dataset import DeepSWEInstance
from crucible.products.deepswe.spec_synthesis import (
    DEEPSWE_BREACH_POLICY,
    DEEPSWE_COMPACT_PLAN_MAX_BUDGET_USD,
    DEEPSWE_REVIEW_POLICY,
    resolve_deepswe_breach_policy,
    resolve_deepswe_plan_policy,
    synthesize_starter_spec,
    uses_deepswe_compact_plan_policy,
)


def _instance(
    *,
    task_id: str = "task-1",
    ext_id: str = "e1",
    repo: str = "ex/ex",
    repository_url: str | None = None,
    language: str = "python",
    category: str = "feature_request",
    instruction: str = "Body of instruction.md",
    base_commit_hash: str = "0123456789abcdef0123456789abcdef01234567",
) -> DeepSWEInstance:
    """Inline DeepSWEInstance — no file I/O. Defaults to a Python
    task; override `language` / etc. per test."""
    return DeepSWEInstance(
        task_id=task_id,
        ext_id=ext_id,
        repo=repo,
        repository_url=repository_url or f"https://github.com/{repo}",
        language=language,
        category=category,
        display_title="Title",
        display_description="Description",
        base_commit_hash=base_commit_hash,
        docker_image=f"public.ecr.aws/d3j8x8q7/swe-bench-202605:{ext_id}",
        cpus=2,
        memory_mb=8192,
        storage_mb=20480,
        build_timeout_sec=1800.0,
        allow_internet=False,
        verifier_timeout_sec=1800.0,
        agent_timeout_sec=5400.0,
        instruction=instruction,
    )


# ---------- synthesize_starter_spec happy path ----------


_VERIFIER = "crucible.products.deepswe.runners.deepswe_verifier"
_DEV_VERIFY = "crucible.products.deepswe.runners.deepswe_dev_verify"


def test_synthesize_python_instance():
    """Python task → Spec with the python runner under 'build' role
    AND the uniform deepswe verifier under 'verify' role."""
    inst = _instance(language="python", instruction="Add ABC support")
    spec = synthesize_starter_spec(inst)

    assert isinstance(spec, Spec)
    assert spec.version == 0
    assert spec.parent is None
    assert spec.root is None
    assert spec.intent == "Add ABC support"
    assert spec.constraints == ()
    assert spec.done_criteria == ()

    pc = spec.project_context
    assert pc.project == "deepswe/ex/ex"
    assert pc.repo_url == "https://github.com/ex/ex"
    assert pc.base_commit == "0123456789abcdef0123456789abcdef01234567"
    assert pc.branch is None
    assert pc.submodule_path is None
    assert pc.runners == {
        "build": "crucible.common.runners.lang.python",
        "verify": _VERIFIER,
        "dev_verify": _DEV_VERIFY,
    }


def test_synthesize_rust_instance():
    """Rust task → rust runner under build role + verifier under
    verify."""
    inst = _instance(language="rust", repo="rust-lang/example")
    spec = synthesize_starter_spec(inst)
    assert spec.project_context.runners == {
        "build": "crucible.common.runners.lang.rust",
        "verify": _VERIFIER,
        "dev_verify": _DEV_VERIFY,
    }


def test_synthesize_go_instance_verify_only():
    """Go build runner hasn't shipped (slice 3 of step 2's
    per-language work — distinct from the slice 3 of this surface
    which is the verifier wrapper that landed already). The verify
    runner is still wired."""
    inst = _instance(language="go", repo="abs-lang/abs")
    spec = synthesize_starter_spec(inst)
    assert spec.project_context.runners == {
        "verify": _VERIFIER,
        "dev_verify": _DEV_VERIFY,
    }


def test_synthesize_typescript_instance_verify_only():
    """TS build runner hasn't shipped — verify runner still wired."""
    inst = _instance(language="typescript", repo="Textualize/textual")
    spec = synthesize_starter_spec(inst)
    assert spec.project_context.runners == {
        "verify": _VERIFIER,
        "dev_verify": _DEV_VERIFY,
    }


def test_synthesize_javascript_instance_verify_only():
    """JS build runner hasn't shipped — verify runner still wired."""
    inst = _instance(language="javascript", repo="example/example")
    spec = synthesize_starter_spec(inst)
    assert spec.project_context.runners == {
        "verify": _VERIFIER,
        "dev_verify": _DEV_VERIFY,
    }


def test_synthesize_preserves_repository_url_casing():
    """``project_context.repo_url`` comes from the manifest's
    ``repository_url`` field (full URL), NOT constructed from the
    short-form ``repo``. Some DeepSWE repos preserve nonstandard
    owner casing (Textualize/textual, Fatal1ty/mashumaro); the
    full URL avoids case-folding ambiguity."""
    inst = _instance(
        repo="Textualize/textual",
        repository_url="https://github.com/Textualize/textual",
    )
    spec = synthesize_starter_spec(inst)
    assert spec.project_context.repo_url == "https://github.com/Textualize/textual"


def test_synthesize_unknown_language_verify_only():
    """Defensive: a language we've never seen returns only the
    verify entry (uniform across DeepSWE) — no spurious build
    runner. Future-proof against DeepSWE adding more languages."""
    inst = _instance(language="elixir")
    spec = synthesize_starter_spec(inst)
    assert spec.project_context.runners == {
        "verify": _VERIFIER,
        "dev_verify": _DEV_VERIFY,
    }


def test_synthesize_uses_base_commit_hash():
    """``project_context.base_commit`` comes from the instance's
    ``base_commit_hash`` field (DeepSWE's name for the same field
    SWE-bench calls ``base_commit``)."""
    inst = _instance(base_commit_hash="deadbeefcafebabe1234567890abcdef12345678")
    spec = synthesize_starter_spec(inst)
    assert (
        spec.project_context.base_commit == "deadbeefcafebabe1234567890abcdef12345678"
    )


def test_synthesize_intent_is_instruction_body():
    """The Spec's ``intent`` is the instruction.md body verbatim —
    DeepSWE prompts carry enumerated-behavior structure that the
    planner sees directly."""
    body = (
        "## Module resolution\n\n"
        "- Module names resolve as-typed.\n"
        "- ...\n\n"
        "## Implementation notes\n\n"
        "Internal symbols are flexible; public signatures must be preserved."
    )
    inst = _instance(instruction=body)
    spec = synthesize_starter_spec(inst)
    assert spec.intent == body


def test_synthesize_translates_v11_branch_commit_transport_instruction():
    body = (
        "Implement the requested alias behavior.\n\n"
        "IMPORTANT: Please work on this in a new branch from main and commit "
        "everything when you are done.\n"
    )

    spec = synthesize_starter_spec(_instance(instruction=body))

    assert spec.intent.startswith("Implement the requested alias behavior.")
    assert "new branch from main" not in spec.intent
    assert "commit everything" not in spec.intent
    assert "CRUCIBLE WORKFLOW BOUNDARY" in spec.intent
    assert "uncommitted working-tree edits" in spec.intent
    assert "do not create or switch branches and do not commit" in spec.intent


def test_synthesize_does_not_rewrite_noncanonical_commit_prose():
    body = "Implement the feature and document when callers should commit."

    spec = synthesize_starter_spec(_instance(instruction=body))

    assert spec.intent == body


def test_synthesize_plan_policy_allows_critique_convergence():
    """DeepSWE adds bounded convergence headroom without weakening vetoes."""
    inst = _instance()
    spec = synthesize_starter_spec(inst)
    assert spec.plan_policy is not None
    assert spec.plan_policy.max_critique_rounds == 6
    assert spec.plan_policy.judge_required is True
    assert spec.plan_policy.fresh_judge_reassessment_min_prior_rounds == 2
    assert spec.plan_policy.breaker_required is False
    assert spec.plan_policy.persistence_veto_threshold == 3
    assert (
        spec.plan_policy.fresh_judge_reassessment_min_prior_rounds
        < spec.plan_policy.persistence_veto_threshold
    )
    assert spec.plan_policy.judge_veto_weighted_threshold == 10.0
    assert spec.plan_policy.max_total_slices == 16
    assert spec.plan_policy.max_total_tiers == 12
    assert spec.plan_policy.inflation_compaction_retries == 1
    assert spec.plan_policy.max_verification_only_empty_scope_slices == 1
    guidance = " ".join(spec.plan_policy.planner_guidance.split())
    assert "compact and delta-focused" in guidance
    assert "preserve every already-grounded slice" in guidance
    assert "change only fields or slices justified by an active" in guidance
    assert "Merge semantically duplicate repairs" in guidance
    assert "one complete replacement TierDAG" in guidance
    assert "never a patch, partial DAG, change log, or prose wrapper" in guidance
    assert "All root-to-plan and plan-to-root coverage obligations" in guidance


def test_twelve_dollar_envelope_requires_one_complete_tier_and_slice():
    spec = synthesize_starter_spec(
        _instance(),
        task_budget_usd=DEEPSWE_COMPACT_PLAN_MAX_BUDGET_USD,
    )

    policy = spec.plan_policy
    assert policy is not None
    assert policy.max_total_tiers == 1
    assert policy.max_total_slices == 1
    assert policy.max_verification_only_empty_scope_slices == 0
    guidance = " ".join(policy.planner_guidance.split())
    assert "exactly one tier" in guidance
    assert "exactly one cohesive, source-bearing implementation slice" in guidance
    assert "complete root directive" in guidance
    assert "not permission to narrow, omit, or reinterpret" in guidance


def test_twelve_dollars_and_one_cent_preserves_feature_decomposition():
    policy = resolve_deepswe_plan_policy(12.01)

    assert uses_deepswe_compact_plan_policy(12.0) is True
    assert uses_deepswe_compact_plan_policy(12.01) is False
    assert uses_deepswe_compact_plan_policy(None) is False
    assert resolve_deepswe_plan_policy(None) == policy
    assert policy.max_total_tiers == 12
    assert policy.max_total_slices == 16
    assert policy.max_verification_only_empty_scope_slices == 1


def test_synthesize_language_is_case_insensitive_post_load():
    """Edge: an instance whose ``language`` field is mixed case
    (would be lowercased by load_dataset) still resolves to the
    right runner. Belt-and-suspenders against future callers that
    construct DeepSWEInstance manually without lowercasing."""
    inst = _instance(language="PYTHON")
    spec = synthesize_starter_spec(inst)
    assert spec.project_context.runners == {
        "build": "crucible.common.runners.lang.python",
        "verify": _VERIFIER,
        "dev_verify": _DEV_VERIFY,
    }


# ---------- policy constants ----------


def test_deepswe_breach_policy_shape():
    """Verify the DEEPSWE_BREACH_POLICY constant matches the design:
    LLM Breaker kept (impl_breaker, not swebench_impl_breaker),
    max_rounds bumped (no F2P deterministic seeding), enabled by
    default (quality mechanism)."""
    assert isinstance(DEEPSWE_BREACH_POLICY, BreachPolicy)
    assert DEEPSWE_BREACH_POLICY.enabled is True
    assert DEEPSWE_BREACH_POLICY.max_rounds == 10
    assert DEEPSWE_BREACH_POLICY.breaker_rubric == "impl_breaker"
    assert DEEPSWE_BREACH_POLICY.judge_rubric == "breach_judge"
    assert DEEPSWE_BREACH_POLICY.require_attack_authority is True
    # Explicitly NOT swebench_impl_breaker — that rubric is bonded
    # to SWE-bench's FAIL_TO_PASS shape which DeepSWE doesn't have.
    assert "swebench" not in (DEEPSWE_BREACH_POLICY.breaker_rubric or "")


def test_deepswe_review_policy_drops_spec_clarity():
    """The DEEPSWE_REVIEW_POLICY constant drops spec_clarity for
    the same reason SWE-bench's does — the spec is dataset-authored
    (datacurve-ai), not crucible-authored. Keeps diff_scope +
    tier_completeness."""
    assert isinstance(DEEPSWE_REVIEW_POLICY, ReviewPolicy)
    assert "spec_clarity" not in DEEPSWE_REVIEW_POLICY.rubrics
    assert set(DEEPSWE_REVIEW_POLICY.rubrics) == {
        "diff_scope",
        "tier_completeness",
        "api_contract_fidelity",
    }
    # api_contract_fidelity is ADVISORY (records findings, does not gate /
    # trigger retries) — the gating retry loop didn't converge and blew the
    # per-instance cap. The work-rubrics (diff_scope/tier_completeness) gate.
    assert DEEPSWE_REVIEW_POLICY.advisory_rubrics == ("api_contract_fidelity",)
    assert "diff_scope" not in DEEPSWE_REVIEW_POLICY.advisory_rubrics
    assert "tier_completeness" not in DEEPSWE_REVIEW_POLICY.advisory_rubrics


# ---------- spec-fidelity fixes (#1 system prompt, #2 API elevation,
# ---------- #3 api_contract_fidelity rubric) ----------


def test_deepswe_system_prompt_specializes_swebench_without_conflict():
    """DeepSWE overrides source-only capture and preserves API fidelity."""
    from crucible.products.deepswe import prompts as deepswe_prompts
    from crucible.products.swebench.prompts import SWEBENCH_SYSTEM_PROMPT

    prompt = deepswe_prompts.DEEPSWE_SYSTEM_PROMPT
    source_only_rule = deepswe_prompts._SWEBENCH_TEST_PATH_RULE
    assert SWEBENCH_SYSTEM_PROMPT.count(source_only_rule) == 1
    assert source_only_rule not in prompt
    assert prompt == (
        SWEBENCH_SYSTEM_PROMPT.replace(
            source_only_rule,
            deepswe_prompts._DEEPSWE_FULL_PATCH_RULE,
            1,
        )
        + deepswe_prompts._API_CONTRACT_CLAUSE
    )
    assert prompt.count("DEEPSWE PATCH CONTRACT") == 1
    assert "API CONTRACT" in prompt
    # The clause names the concrete failure shapes from the RCA.
    assert "cancel()" in prompt
    assert "metrics" in prompt
    assert "COMPLETE repository patch" in prompt
    assert "base_test.py" in prompt
    assert "test_set.py" in prompt
    assert "browser_test_runner.js" in prompt
    assert "Do not classify a path by" in prompt
    assert "provenance-bound mutation journal" in prompt
    assert "``/tests`` and ``/solution`` are hidden" in prompt
    # swebench's own prompt is untouched (no leaked clause).
    assert "API CONTRACT" not in SWEBENCH_SYSTEM_PROMPT
    assert "DEEPSWE PATCH CONTRACT" not in SWEBENCH_SYSTEM_PROMPT


def test_extract_api_contract_section_surfaces_code_blocks():
    """#2: fenced code blocks are re-emitted under a contract header."""
    from crucible.products.deepswe.spec_synthesis import (
        _extract_api_contract_section,
    )

    instruction = (
        "Add aliases.\n\nApi:\n```js\n"
        "const r = await c.initialize({ concurrency: 5 })\n"
        "console.log(r.metrics.database.duration)\n```\n\nExpected: ...\n"
    )
    sec = _extract_api_contract_section(instruction)
    assert "PUBLIC API CONTRACT" in sec
    assert "r.metrics.database.duration" in sec
    assert "c.initialize({ concurrency: 5 })" in sec


def test_extract_api_contract_section_empty_without_code_blocks():
    """#2: prose-only instructions get no appended section (intent stays
    verbatim)."""
    from crucible.products.deepswe.spec_synthesis import (
        _extract_api_contract_section,
    )

    assert _extract_api_contract_section("just prose, no fences") == ""


def test_synthesize_appends_api_contract_when_code_blocks_present():
    """#2: synthesize_starter_spec elevates the issue's code blocks into the
    intent (prefix stays the verbatim instruction)."""
    body = (
        "Add async init.\n\nApi:\n```ts\n"
        "const result = await container.initialize({ concurrency: 5 })\n"
        "console.log(result.totalDuration)\n```\n"
    )
    inst = _instance(instruction=body)
    spec = synthesize_starter_spec(inst)
    assert spec.intent.startswith(body)
    assert "PUBLIC API CONTRACT" in spec.intent
    assert "result.totalDuration" in spec.intent


def test_raw_instruction_breach_authority_validates_against_augmented_spec():
    """Issue transport stays raw while directives bind the canonical DAG."""

    import json

    from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG
    from crucible.workflows.breach import (
        _parse_breach_authority_payload,
        _validate_authority_canonical_tier,
        encode_breach_authority,
    )

    instruction = (
        "Add the public helper:\n\n"
        "```python\n"
        "def public_helper(value: str) -> str: ...\n"
        "```"
    )
    spec = synthesize_starter_spec(_instance(instruction=instruction))
    assert spec.intent != instruction
    directive = ImplementationDirective(
        slice_id="api",
        intent="implement the public helper",
        files_in_scope=("src/api.py",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1},
        requirement_refs=(instruction,),
    )
    spec = dataclasses.replace(
        spec,
        tier_dag=TierDAG(
            tiers=(
                Tier(
                    tier_id="t0",
                    slices=(Slice(id="api", directive=directive),),
                    depends_on=(),
                ),
            )
        ),
    )
    encoded, digest = encode_breach_authority(
        tier_id="t0",
        issue_text=instruction,
        slice_directives=(directive,),
    )
    authority = _parse_breach_authority_payload(
        json.loads(encoded),
        encoded=encoded,
        digest=digest,
    )

    _validate_authority_canonical_tier(authority, spec)


def test_api_contract_fidelity_rubric_loads_and_validates():
    """#3: the rubric loads, is in the policy, and the tier accepts it."""
    from crucible.rubrics import load as load_rubric
    from crucible.workflows.tier import (
        _SUPPORTED_REVIEW_RUBRICS,
        _validate_review_rubrics,
    )

    rubric = load_rubric("api_contract_fidelity")
    assert rubric is not None
    assert "api_contract_fidelity" in _SUPPORTED_REVIEW_RUBRICS
    # The policy's full rubric set must validate (no unsupported names).
    _validate_review_rubrics(DEEPSWE_REVIEW_POLICY.rubrics)
    # The rubric's user_template consumes the spec_intent + artifact the
    # tier feeds it.
    tmpl = getattr(rubric, "user_template", "") or ""
    assert "{spec_intent}" in tmpl and "{artifact}" in tmpl


def test_resolve_deepswe_breach_policy_enabled_true():
    """resolve(enabled=True) returns the canonical constant
    unchanged."""
    policy = resolve_deepswe_breach_policy(enabled=True)
    assert policy is DEEPSWE_BREACH_POLICY


def test_resolve_deepswe_breach_policy_enabled_false():
    """resolve(enabled=False) flips the enabled flag but keeps the
    other fields. Used by --no-include-breach to skip the breach
    loop without losing rubric / max_rounds configuration."""
    policy = resolve_deepswe_breach_policy(enabled=False)
    assert policy.enabled is False
    assert policy.max_rounds == DEEPSWE_BREACH_POLICY.max_rounds
    assert policy.breaker_rubric == DEEPSWE_BREACH_POLICY.breaker_rubric
    assert policy.judge_rubric == DEEPSWE_BREACH_POLICY.judge_rubric
    assert (
        policy.require_attack_authority
        is DEEPSWE_BREACH_POLICY.require_attack_authority
    )


def test_breach_policy_is_immutable():
    """BreachPolicy is a frozen dataclass — direct mutation fails.
    Confirms the constant can't be accidentally mutated by callers."""
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        DEEPSWE_BREACH_POLICY.max_rounds = 99  # type: ignore[misc]
