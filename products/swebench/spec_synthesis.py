# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Synthesize a starter ``Spec`` from a ``SwebenchInstance``.

The SWE-bench adapter (``workflows/swebench.py``) calls
``synthesize_starter_spec(inst)`` per instance to produce a Spec
that the planner can consume. No ``tier_dag`` is set — the
planner produces it.

Spec shape:

- ``project_context``: GitHub-derived repo URL + the instance's
  ``base_commit`` + per-language ``runners`` dict (from
  ``crucible.common.runners.lang.<lang>``). The runners dict feeds
  ``_build_verify``'s build / verify subprocess invocation if the
  cascade reaches it.
- ``intent``: the issue's ``problem_statement`` verbatim. The
  planner sees this in its critique loop.
- ``constraints``: per-language hints (don't modify test suite, etc).
- ``plan_policy``: a bounded, independently judged policy. Product-specific
  authoring constraints express the preference for a minimal single-issue
  plan.
- ``tier_dag``: None — produced by the planner downstream.

Implementations / verdicts / measurements / lifecycle events:
all empty tuples (the instance is fresh).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from assay import ProjectContext
from assay.types.policy import PlanPolicy
from assay.types.spec import Spec

from crucible.common.policies import BreachPolicy, ReviewPolicy
from crucible.constraints import AuthoringInstructionConstraint

if TYPE_CHECKING:
    from .dataset import SwebenchInstance


# SWE-bench Verified is Python-only today. The infer-language helper
# is forward-compatible for SWE-bench Multilingual (uses the instance's
# ``language`` field when present; falls back to "python" on Verified
# instances that don't carry the field).
def _infer_language(inst: "SwebenchInstance") -> str:
    """Return the canonical language tag for an instance.

    Reads ``inst.language`` if the dataset variant carries it
    (Multilingual). Otherwise defaults to ``"python"`` since SWE-bench
    Verified is Python-only.
    """
    lang = getattr(inst, "language", "") or "python"
    return lang.strip().lower() or "python"


# Python SWE-bench instances route to an authoring-only visible-test
# runner. It executes inside the isolated instance container and rejects
# dataset ``test_patch`` / FAIL_TO_PASS / PASS_TO_PASS fields. Hidden
# evaluation is performed once, after authoring, by the official harness.
#
# Rust + cpp keep the generic language runners — SWE-bench Verified
# is Python-only today; if/when a polyglot variant lands, those
# would get their own tuned runners with analogous logic.
_LANG_TO_RUNNER_MODULE: dict[str, str] = {
    "python": "crucible.products.swebench.runners.swebench_pytest",
    "rust": "crucible.common.runners.lang.rust",
    "cpp": "crucible.common.runners.lang.cpp",
}


def _runners_for_instance(inst: "SwebenchInstance") -> dict[str, str]:
    """Per-instance runner dict for ``ProjectContext.runners``.

    Maps the per-instance language to the language's runner module
    dotted-path. The runner module's ``main()`` dispatches on the
    ``parameters.role`` field injected by
    ``_build_verify._run_runner`` — role=="build" runs the per-language
    install command, role=="verify" runs the per-language test
    command. Bundled runners own their parameter schema and default command;
    static selection is carried by the synthesized Spec rather than process
    environment or workflow CLI state.

    The SWE-bench tier cascade uses ``_build_verify.build_and_verify`` as a
    visible-tests regression
    check between breach rounds — does the implementer's diff break
    the project's own tests? The hidden FAIL_TO_PASS / PASS_TO_PASS
    tests are the leaderboard ground truth, scored separately by the
    SWE-bench eval harness outside the cascade.
    """
    language = _infer_language(inst)
    module = _LANG_TO_RUNNER_MODULE.get(language)
    if module is None:
        # Unknown language — return empty runners and let the agent's
        # Bash tool discover the project's own conventions at runtime.
        # Safer than synthesizing wrong commands.
        return {}
    return {"dev_verify": module}


_SWEBENCH_DEFAULT_PLAN_POLICY = PlanPolicy(
    # Single round of plan critique on SWE-bench. Issues are small;
    # multiple round-trips over the planner+critic are wasted spend.
    max_critique_rounds=1,
    # Independent Judge is mandatory even when the Critic passes. This
    # prevents one review role/model from unilaterally approving its own
    # plan interpretation.
    judge_required=True,
    # No breaker at plan time (breach loop happens later in the tier
    # cascade against the implementer's diff, not against the plan).
    breaker_required=False,
    # Default veto threshold; unused at plan time but required by
    # PlanPolicy's dataclass.
    persistence_veto_threshold=3,
)


SWEBENCH_AUTHORING_POLICY_OWNER = "crucible.products.swebench"
SWEBENCH_AUTHORING_POLICY_ID = "crucible.swebench.authoring.v3"

_READ_ONLY_TEST_EVIDENCE_V1 = (
    "Repository test files are read-only authoring evidence: inspect and run "
    "existing tests through verify_mode=tests, but never include a test path "
    "in files_in_scope or creates_paths and never create a test-writing "
    "implementation slice."
)
_READ_ONLY_TEST_EVIDENCE_V2 = (
    _READ_ONLY_TEST_EVIDENCE_V1
    + " Do not name or quote test paths or individual test symbols in "
    "directive intent; refer to them only as existing repository test "
    "evidence."
)
_MINIMAL_SINGLE_ISSUE_PLAN_V1 = (
    "Default to one source-bearing slice and one tier for a single-issue "
    "SWE-bench fix. Do not add changelog, release-note, or documentation "
    "slices unless the issue explicitly requires those deliverables."
)
_EVIDENCE_BOUND_CLEANUP_V1 = (
    "Keep cleanup obligations evidence-bound. Do not declare an import "
    "unused, remove or relocate an import, or change a public re-export "
    "unless the issue explicitly requires it or repository source evidence "
    "proves the symbol has no import-time, registry, or public-API role. "
    "Omit speculative cleanup from a slice directive instead of turning it "
    "into a mandatory completion requirement."
)


def _managed_authoring_instruction(
    instruction: str,
    *,
    rule_id: str,
    revision: int,
) -> AuthoringInstructionConstraint:
    return AuthoringInstructionConstraint(
        instruction,
        policy_owner=SWEBENCH_AUTHORING_POLICY_OWNER,
        policy_rule_id=rule_id,
        policy_revision=revision,
    )


SWEBENCH_PLANNER_CONSTRAINTS = (
    _managed_authoring_instruction(
        _READ_ONLY_TEST_EVIDENCE_V2,
        rule_id="readonly_test_evidence",
        revision=2,
    ),
    _managed_authoring_instruction(
        _MINIMAL_SINGLE_ISSUE_PLAN_V1,
        rule_id="minimal_single_issue_plan",
        revision=1,
    ),
    _managed_authoring_instruction(
        _EVIDENCE_BOUND_CLEANUP_V1,
        rule_id="evidence_bound_cleanup",
        revision=1,
    ),
)

# Specs created before managed authoring instructions carried explicit
# ownership have only free-form text. Exact matching is deliberately narrow:
# an unrecognized or differently owned constraint belongs to the lineage and
# must survive a product-policy refresh. Keep every historical product string
# here so old managed rules can be replaced without guessing from keywords.
_LEGACY_SWEBENCH_AUTHORING_INSTRUCTIONS = frozenset({
    _READ_ONLY_TEST_EVIDENCE_V1,
    _READ_ONLY_TEST_EVIDENCE_V2,
    _MINIMAL_SINGLE_ISSUE_PLAN_V1,
    _EVIDENCE_BOUND_CLEANUP_V1,
})


def is_swebench_managed_authoring_constraint(constraint: object) -> bool:
    """Return whether ``constraint`` belongs to the SWE-bench product."""

    if not isinstance(constraint, AuthoringInstructionConstraint):
        return False
    if constraint.policy_owner == SWEBENCH_AUTHORING_POLICY_OWNER:
        return True
    return (
        constraint.policy_owner is None
        and constraint.instruction
        in _LEGACY_SWEBENCH_AUTHORING_INSTRUCTIONS
    )


def swebench_managed_authoring_constraints(
    constraints: tuple[object, ...],
) -> tuple[AuthoringInstructionConstraint, ...]:
    """Select managed rules, including exact legacy unowned spellings."""

    return tuple(
        constraint
        for constraint in constraints
        if is_swebench_managed_authoring_constraint(constraint)
    )


def reconcile_swebench_authoring_constraints(
    constraints: tuple[object, ...],
) -> tuple[object, ...]:
    """Replace only SWE-bench-owned rules with the current managed block.

    Unknown, operator-owned, and non-authoring constraints retain their exact
    values and relative order. The current product block occupies the first
    historical managed rule's position, or is appended when none existed.
    """

    reconciled: list[object] = []
    inserted = False
    for constraint in constraints:
        if is_swebench_managed_authoring_constraint(constraint):
            if not inserted:
                reconciled.extend(SWEBENCH_PLANNER_CONSTRAINTS)
                inserted = True
            continue
        reconciled.append(constraint)
    if not inserted:
        reconciled.extend(SWEBENCH_PLANNER_CONSTRAINTS)
    return tuple(reconciled)


# SWE-bench breach policy. One named policy that the worker passes
# through ``run_architect(breach_policy=)``.
# Current product defaults:
# - ``enabled=True``: SWE-bench's quality mechanism for resolved-rate
#   is the breach loop; cheap-smoke runs flip enabled off via
#   ``dataclasses.replace`` at the call site.
# - ``max_rounds=8`` gives the Defender a bounded convergence budget. The
#   workflow's cost cap remains an independent upper bound.
# - ``breaker_rubric / judge_rubric``: SWE-bench-specialized rubrics
#   under ``crucible/rubrics/swebench_*.yaml``. The Breaker attacks
#   alternative interpretations of the SWE-bench problem_statement
#   rather than only the diff; the Judge inherits the bundled
#   ``breach_judge`` rubric.
SWEBENCH_BREACH_POLICY = BreachPolicy(
    enabled=True,
    max_rounds=8,
    breaker_rubric="swebench_impl_breaker",
    judge_rubric="breach_judge",
    require_attack_authority=True,
)


def resolve_swebench_breach_policy(*, enabled: bool) -> BreachPolicy:
    """Return the SWE-bench breach policy with ``enabled`` set to
    the caller's value. Used by the swebench worker's parallel /
    serial paths (both honor the ``--no-include-breach`` ablation
    flag without duplicating the ``dataclasses.replace`` boilerplate
    at each call site).

    The other fields (``max_rounds``, ``breaker_rubric``,
    ``judge_rubric``) come from the SWE-bench template constant
    above — those are policy decisions tied to the SWE-bench
    benchmark contract, not per-call overrides.
    """
    if enabled:
        return SWEBENCH_BREACH_POLICY
    return dataclasses.replace(SWEBENCH_BREACH_POLICY, enabled=False)


# SWE-bench tier review set. Drops ``spec_clarity`` from the
# default. Rationale: SWE-bench specs aren't crucible-authored — the
# intent IS the dataset's verbatim ``problem_statement`` (a GitHub
# issue body, with all the hedging language real issues carry), and
# ``done_criteria=()`` is set deliberately because the completion
# contract IS the dataset's FAIL_TO_PASS / PASS_TO_PASS test set, not
# crucible-authored criteria. Running ``spec_clarity`` against this shape
# evaluates dataset authoring rather than the implementation. The remaining
# rubrics (``diff_scope``, ``tier_completeness``) review the implementation.
SWEBENCH_REVIEW_POLICY = ReviewPolicy(
    rubrics=("diff_scope", "tier_completeness"),
    # One bounded semantic repair lets the review pass reconcile an otherwise sound
    # implementation with findings raised against a Defender amendment. Tier's
    # outer mutation journal keeps this fail-closed: no retry bytes become
    # resumable authority until review, final verification, and the lifecycle
    # compare-and-swap all succeed.
    retry_budget=1,
)


def synthesize_starter_spec(inst: "SwebenchInstance") -> Spec:
    """Build a starter Spec for a SWE-bench instance.

    The Spec has no ``tier_dag`` — the planner produces it. The
    Spec's ``project_context`` carries per-language runners derived
    from the instance's language; the planner sees the issue text
    via ``intent``; the implementer phases consume both.

    The returned Spec is fresh (version=0, no parent/root, no
    implementation/verdict/measurement refs). Persisting it to a
    per-instance Corpus is the caller's responsibility.
    """
    runners = _runners_for_instance(inst)
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project=f"swebench/{inst.repo}",
            repo_url=f"https://github.com/{inst.repo}",
            base_commit=inst.base_commit,
            branch=None,
            submodule_path=None,
            runners=runners,
        ),
        execution=None,
        intent=inst.problem_statement,
        constraints=SWEBENCH_PLANNER_CONSTRAINTS,
        done_criteria=(),
        plan_policy=_SWEBENCH_DEFAULT_PLAN_POLICY,
    )


__all__ = [
    "SWEBENCH_AUTHORING_POLICY_ID",
    "SWEBENCH_AUTHORING_POLICY_OWNER",
    "SWEBENCH_BREACH_POLICY",
    "SWEBENCH_PLANNER_CONSTRAINTS",
    "SWEBENCH_REVIEW_POLICY",
    "is_swebench_managed_authoring_constraint",
    "reconcile_swebench_authoring_constraints",
    "resolve_swebench_breach_policy",
    "swebench_managed_authoring_constraints",
    "synthesize_starter_spec",
]
