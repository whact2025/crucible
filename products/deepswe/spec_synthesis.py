# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Synthesize a starter ``Spec`` from a ``DeepSWEInstance``.

The DeepSWE workflow calls
``synthesize_starter_spec(inst)`` per task to produce a Spec the
planner can consume. No ``tier_dag`` is set — the planner produces it.

Spec shape:

- ``project_context``: full GitHub repo URL (from the instance's
  ``repository_url``) + ``base_commit_hash`` + per-language
  ``runners`` dict. The runners dict feeds ``_build_verify``'s
  build / verify subprocess invocation if the cascade reaches it. Every task
  receives the hidden verifier and development-verification adapters; Python
  and Rust also receive a build runner.
- ``intent``: the ``instruction.md`` behavior verbatim, with only the
  dataset's external branch/commit transport instruction translated to
  Crucible's equivalent working-tree capture boundary. DeepSWE prompts carry
  enumerated expected behaviors + implementation notes — the planner sees
  this structure directly.
- ``constraints``: empty for now. The air-gap directive
  (``no network access``) is enforced at the kernel layer via
  ``docker run --network=none`` and surfaced in the implementer
  prompt template, not as a Spec constraint.
- ``plan_policy``: PlanPolicy with up to six critique rounds. Finite task
  envelopes at or below $12 use an explicit one-tier/one-slice shape so one
  complete planning -> implementation -> Breaker/entailment -> mandatory-review
  cycle is budget-feasible; this boundary is not a provider-price promise.
  Early acceptance still exits the loop immediately. The planner guidance
  keeps every new file at a path candidate capture can see and, for the
  standard decomposition, keeps the edits of one contract in one slice.
- ``tier_dag``: None — produced by the planner downstream.

Implementations / verdicts / measurements / lifecycle events: all
empty tuples (the instance is fresh).

"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

from assay import ProjectContext
from assay.types.policy import PlanPolicy
from assay.types.spec import Spec

from crucible.common.policies import BreachPolicy, ReviewPolicy

if TYPE_CHECKING:
    from .dataset import DeepSWEInstance


# Per-language build adapters. All languages still receive the uniform
# DeepSWE hidden verifier and development-verification adapters below.
_LANG_TO_RUNNER_MODULE: dict[str, str] = {
    "python": "crucible.common.runners.lang.python",
    "rust": "crucible.common.runners.lang.rust",
}


_DEEPSWE_VERIFIER_MODULE = "crucible.products.deepswe.runners.deepswe_verifier"

_DEEPSWE_DEV_VERIFY_MODULE = "crucible.products.deepswe.runners.deepswe_dev_verify"


# Fenced code block matcher (``` … ```), language tag optional.
_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

_EXTERNAL_GIT_WORKFLOW_SUFFIX = (
    "\n\nIMPORTANT: Please work on this in a new branch from main and commit "
    "everything when you are done.\n"
)
_CRUCIBLE_GIT_WORKFLOW_BOUNDARY = (
    "\n\nCRUCIBLE WORKFLOW BOUNDARY: Leave the complete solution as "
    "uncommitted working-tree edits. Crucible owns the isolated checkout, "
    "branch, patch capture, and commit lifecycle; do not create or switch "
    "branches and do not commit.\n"
)

_DEEPSWE_PLANNER_GUIDANCE = """
Keep planning output compact and delta-focused. On a revision, preserve every
already-grounded slice and change only fields or slices justified by an active
Critic/Judge finding or deterministic validation failure. Do not repeat
repository exploration already represented by valid paths and source excerpts
unless an active finding identifies missing or contradictory evidence. Merge
semantically duplicate repairs.

Even when the repair is narrow, emit one complete replacement TierDAG in the
required typed schema, never a patch, partial DAG, change log, or prose wrapper.
Before responding, check that every JSON object and array is closed, including
the top-level `tiers` array, even for a one-tier, one-slice plan.

For each slice, `Slice.id` and its `ImplementationDirective.slice_id` MUST be
exactly equal, character for character. Choose one stable identifier and copy
it into both JSON fields `id` and `directive.slice_id` (for example, `s0` in
both). During a correction, preserve this identity unless an active finding
identifies the identity itself as defective. A different directive identifier
is not an alias and will be rejected before implementation.

All root-to-plan and plan-to-root coverage obligations still apply.
""".strip()

# Tier executes and gates one slice at a time with a writer confined to that
# slice's declared paths, so the first part of a contract split over slices
# fails the full-suite gate where no implementer can make the paired edit.
_DEEPSWE_CONTRACT_PLANNER_GUIDANCE = """
Edits that form one contract belong in one slice. When one behavior needs
paired edits that only build and pass together (mirrored or forwarded
signatures, an interface and its implementers, a public API and its registry,
a signature and its call sites, a required field and its construction sites),
plan them as a single slice whose `files_in_scope` and `creates_paths` list
every file of the contract and whose intent names the paired sites. Do not
split them across sibling slices or across tiers: each slice is implemented
and verified on its own before the next slice starts, its implementer may
write only that slice's declared paths, and one part alone fails the project's
own build or consistency tests. This refines the general rule that dependent
work goes in later tiers: later tiers are for prerequisites that leave the
repository coherent on their own, never for the other part of a paired edit.
This bound is not permission to bundle independently landable behavior.
""".strip()

# Candidate capture stages the working tree with `git add -A`, which honors
# the repository's ignore rules; an ignored new file never reaches the patch.
_DEEPSWE_CAPTURE_PLANNER_GUIDANCE = """
Plan every new file at a path the repository does not ignore. The candidate
patch is captured the way `git add -A` sees the working tree, so a new file
whose path matches the repository's ignore rules (`.gitignore`) is never
captured and is missing at verification. Check those rules before declaring a
`creates_paths` entry in a cache, dependency, environment, generated, or
build-output location. When a new file has to live in such a location, cite
the complete root `.gitignore` and any `.gitignore` in a parent directory of
the path as read-only references (`read_only_references`), each from line 1 to
its last line, and state in the slice intent that each of these references is
the whole file, so the review can see that no pattern matches the path; a
partial excerpt cannot show that. An edit to an already tracked file is
captured wherever it lives.
""".strip()


# A finite task envelope at or below this boundary cannot reliably fund
# repeated implementation, Breach, and mandatory-review cycles for a
# multi-tier plan.  This is a workflow envelope class, not a promise about
# any provider's price.  Compact the complete task into one coherent
# increment so the bounded run pays each required phase once.
DEEPSWE_COMPACT_PLAN_MAX_BUDGET_USD = 12.0

_DEEPSWE_COMPACT_PLANNER_GUIDANCE = """
This run has a low-budget workflow envelope. Emit exactly one tier containing
exactly one cohesive, source-bearing implementation slice. The single slice
must preserve the complete root directive: combine tightly coupled schema,
runtime, test, and documentation work when necessary rather than deferring any
required behavior to a later tier. This compaction is an execution-budget
constraint, not permission to narrow, omit, or reinterpret the task.
""".strip()


def _extract_api_contract_section(instruction: str) -> str:
    """Re-surface the issue's fenced code blocks as an explicit API contract.

    DeepSWE instructions state the exact public API in code blocks
    (e.g. ``result.metrics.database.duration``), but those blocks sit in
    prose and the planner paraphrases them into slice intents ("per-
    registration metrics") — losing the literal shape. Re-emitting the
    verbatim blocks under a "match exactly" header raises their salience for
    BOTH the planner and the implementer, complementing the system-prompt
    API-contract clause (``products/deepswe/prompts.py``).

    Returns "" when the instruction carries no code blocks (nothing to
    elevate) so the intent is unchanged for prose-only tasks. The blocks are
    a copy of text already in the instruction — additive, never a rewrite.
    """
    blocks = [m.group(1).strip() for m in _CODE_BLOCK_RE.finditer(instruction)]
    blocks = [b for b in blocks if b]
    if not blocks:
        return ""
    joined = "\n\n".join(f"```\n{b}\n```" for b in blocks)
    return (
        "\n\n## PUBLIC API CONTRACT (verbatim from the issue)\n\n"
        "The code below is copied from the issue above. The hidden verifier "
        "asserts this exact public surface — implement these names, object "
        "shapes, and signatures EXACTLY. Do not rename, restructure (e.g. a "
        "name-keyed record vs an array), or add/remove parameters. Where a "
        "slice directive paraphrases the API, this literal surface wins.\n\n" + joined
    )


def _translate_external_git_workflow(instruction: str) -> str:
    """Translate DeepSWE v1.1's host workflow into Crucible's equivalent.

    The v1.1 corpus appends one exact branch-and-commit instruction to every
    task. Crucible deliberately forbids model-authored Git metadata mutations:
    it owns an isolated checkout, captures the complete patch, and durably
    commits workflow evidence itself. Preserve every task-behavior byte while
    replacing only that exact trailing transport instruction.
    """

    if not instruction.endswith(_EXTERNAL_GIT_WORKFLOW_SUFFIX):
        return instruction
    return (
        instruction.removesuffix(_EXTERNAL_GIT_WORKFLOW_SUFFIX)
        + _CRUCIBLE_GIT_WORKFLOW_BOUNDARY
    )


def _runners_for_instance(inst: "DeepSWEInstance") -> dict[str, str]:
    """Per-instance runner dict for ``ProjectContext.runners``.

    Three roles:

    - ``verify``: the dataset's hidden-test gate (``deepswe_verifier``,
      runs ``bash /tests/test.sh`` in the VERIFIER container). Uniform
      across languages. The final cascade gate.
    - ``dev_verify``: the project-native test runner
      (``deepswe_dev_verify``, runs the project's own tests inside the
      AUTHORING container). Per-round breach gate + author-time signal.
      Uniform module; dispatches on ``parameters.language`` internally.
    - ``build``: the per-language build runner, when one exists for the
      instance's language.

    Unknown / not-yet-shipped build languages return verify +
    dev_verify but no build entry (the cascade's Bash tool discovers
    the project's own build conventions at runtime rather than crucible
    synthesizing wrong commands).
    """
    language = (inst.language or "").strip().lower()
    runners: dict[str, str] = {
        "verify": _DEEPSWE_VERIFIER_MODULE,
        "dev_verify": _DEEPSWE_DEV_VERIFY_MODULE,
    }
    module = _LANG_TO_RUNNER_MODULE.get(language)
    if module is not None:
        runners["build"] = module
    return runners


_DEEPSWE_DEFAULT_PLAN_POLICY = PlanPolicy(
    # Give the initial candidate six bounded revision opportunities. The
    # final-canary C route exhausted the former five-candidate ceiling while
    # retaining planning dollars, wall time, and all implementation authority.
    # Critic/Judge consensus still exits immediately and every hard,
    # persistence, plan-size inflation, and weighted-severity veto remains
    # terminal, so this is headroom for convergence rather than seven
    # mandatory paid calls.
    max_critique_rounds=6,
    # Feature plans are independently adjudicated; Critic approval alone is
    # not a terminal gate.
    judge_required=True,
    # Judge feedback is first delivered to the Planner as an active repair
    # frontier.  When the revised candidate then earns a Critic pass, require
    # the Judge to perform a fresh current-state audit without anchoring on its
    # own previous verdict.  Acceptance still requires that independent Judge
    # to pass, and all deterministic/persistence/weighted gates remain active.
    # Two completed Judge rounds put the one-shot reassessment immediately
    # before the third-round persistence veto could make a stable generated
    # lineage terminal.  Setting this to the veto threshold itself would make
    # the escape hatch unreachable for the exact self-anchoring case.
    fresh_judge_reassessment_min_prior_rounds=2,
    breaker_required=False,
    persistence_veto_threshold=3,
    # DeepSWE feature plans need more seams than small bug-fix plans, but the
    # authoring worker still has a finite context/turn budget.  These product
    # bounds are included in every planner prompt and enforced after canonical
    # parsing.  The first excess gets one explicit compaction repair; a second
    # excess fails closed.
    max_total_slices=16,
    max_total_tiers=12,
    inflation_compaction_retries=1,
    # Permit one terminal test/build-only checkpoint with no mutation scope.
    # Mutation-shaped empty slices and additional checkpoints are rejected as
    # filler before the plan becomes executable.
    max_verification_only_empty_scope_slices=1,
    planner_guidance="\n\n".join((
        _DEEPSWE_PLANNER_GUIDANCE,
        _DEEPSWE_CONTRACT_PLANNER_GUIDANCE,
        _DEEPSWE_CAPTURE_PLANNER_GUIDANCE,
    )),
)

_DEEPSWE_COMPACT_PLAN_POLICY = dataclasses.replace(
    _DEEPSWE_DEFAULT_PLAN_POLICY,
    max_total_slices=1,
    max_total_tiers=1,
    max_verification_only_empty_scope_slices=0,
    # One slice cannot split a contract, so that note is left out. The
    # captured-path note applies to every plan; the compact bound stays last.
    planner_guidance="\n\n".join((
        _DEEPSWE_PLANNER_GUIDANCE,
        _DEEPSWE_CAPTURE_PLANNER_GUIDANCE,
        _DEEPSWE_COMPACT_PLANNER_GUIDANCE,
    )),
)


def uses_deepswe_compact_plan_policy(
    task_budget_usd: float | None,
) -> bool:
    """Return whether one paid task belongs to the compact envelope class."""

    if task_budget_usd is None:
        return False
    budget = float(task_budget_usd)
    return 0.0 < budget <= DEEPSWE_COMPACT_PLAN_MAX_BUDGET_USD


def resolve_deepswe_plan_policy(
    task_budget_usd: float | None,
) -> PlanPolicy:
    """Resolve the plan shape before any planner authority is dispatched."""

    if uses_deepswe_compact_plan_policy(task_budget_usd):
        return _DEEPSWE_COMPACT_PLAN_POLICY
    return _DEEPSWE_DEFAULT_PLAN_POLICY


# DeepSWE breach policy. Same overall shape as SWE-bench's policy
# but with the deterministic option-B fixed-tests Breaker dropped
# (no FAIL_TO_PASS table in DeepSWE's dataset entries to seed it)
# and ``max_rounds`` bumped to compensate.
#
# - ``enabled=True``: DeepSWE has the same hardening posture as
#   SWE-bench — the breach loop is the load-bearing quality
#   mechanism. Cheap-smoke runs flip ``enabled=False`` via
#   ``resolve_deepswe_breach_policy``.
# - ``max_rounds=10``: without a deterministic FAIL_TO_PASS seed, the Breaker
#   has a wider search space and the Defender receives additional convergence
#   headroom.
# - ``breaker_rubric="impl_breaker"`` (generic, NOT
#   ``swebench_impl_breaker``): DeepSWE prompts are structured
#   prose, not a verbatim GitHub issue body. The generic
#   ``impl_breaker`` rubric attacks alternative interpretations of
#   the spec without SWE-bench-specific framing.
# - ``judge_rubric="breach_judge"``: bundled rubric, same as
#   SWE-bench's.
DEEPSWE_BREACH_POLICY = BreachPolicy(
    enabled=True,
    max_rounds=10,
    breaker_rubric="impl_breaker",
    judge_rubric="breach_judge",
    # DeepSWE's generic Breaker synthesizes attacks from prose rather than a
    # deterministic FAIL_TO_PASS list.  Treat those attacks as allegations:
    # bind them to the exact issue + accepted slice directives and admit them
    # independently before any test or production mutation.
    require_attack_authority=True,
)


def resolve_deepswe_breach_policy(*, enabled: bool) -> BreachPolicy:
    """Return the DeepSWE breach policy with ``enabled`` set to the
    caller's value. Used by the deepswe worker's parallel / serial
    paths (both honor the ``--no-include-breach`` ablation flag
    without duplicating the ``dataclasses.replace`` boilerplate at
    each call site).

    The other fields (``max_rounds``, ``breaker_rubric``,
    ``judge_rubric``) come from the DEEPSWE_BREACH_POLICY constant
    above — those are policy decisions tied to the DeepSWE
    benchmark contract, not per-call overrides.
    """
    if enabled:
        return DEEPSWE_BREACH_POLICY
    return dataclasses.replace(DEEPSWE_BREACH_POLICY, enabled=False)


# DeepSWE tier review set. Drops ``spec_clarity`` from the
# default for the same reason SWE-bench does: the DeepSWE spec IS
# the dataset's ``instruction.md`` body (datacurve-ai-authored, not
# crucible-authored), and ``done_criteria=()`` is set deliberately
# because the completion contract IS the binary verifier reward,
# not crucible-authored criteria. Running ``spec_clarity`` against
# this shape produces guaranteed false negatives that would block
# correct diffs from clearing review.
#
# The remaining rubrics (``diff_scope``, ``tier_completeness``)
# review the implementation, which IS the right surface here.
#
DEEPSWE_REVIEW_POLICY = ReviewPolicy(
    # ``api_contract_fidelity`` flags deviations in names, shapes, and
    # signatures. It is deviation-only, so absence is handled by completeness.
    rubrics=("diff_scope", "tier_completeness", "api_contract_fidelity"),
    # API-fidelity findings are retained as diagnostic evidence without
    # triggering another implementation round. Diff scope and completeness
    # remain gating.
    advisory_rubrics=("api_contract_fidelity",),
    # Allow two repairs for gating findings before abandoning the tier.
    retry_budget=2,
)


def synthesize_starter_spec(
    inst: "DeepSWEInstance",
    *,
    task_budget_usd: float | None = None,
) -> Spec:
    """Build a starter Spec for a DeepSWE task.

    The Spec has no ``tier_dag`` — the planner produces it. The
    Spec's ``project_context`` carries per-language runners derived
    from the instance's language; the planner sees the instruction
    body via ``intent``; the implementer phases consume both.

    The returned Spec is fresh (version=0, no parent/root, no
    implementation/verdict/measurement refs). Persisting it to a
    per-instance Corpus is the caller's responsibility.

    Note: ``project_context.repo_url`` uses the manifest's
    ``repository_url`` field (full https URL) rather than
    constructing it from the short-form ``repo``. Some DeepSWE
    repos preserve nonstandard owner casing (e.g. ``Textualize/textual``,
    ``Fatal1ty/mashumaro``); the full URL avoids any case-folding
    that a constructed URL might apply.
    """
    runners = _runners_for_instance(inst)
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project=f"deepswe/{inst.repo}",
            repo_url=inst.repository_url,
            base_commit=inst.base_commit_hash,
            branch=None,
            submodule_path=None,
            runners=runners,
        ),
        execution=None,
        # Append an elevated, verbatim copy of the issue's API code blocks
        # so the planner's slice paraphrases can't silently override the
        # literal public surface the hidden verifier asserts.
        intent=(
            _translate_external_git_workflow(inst.instruction)
            + _extract_api_contract_section(inst.instruction)
        ),
        constraints=(),
        done_criteria=(),
        plan_policy=resolve_deepswe_plan_policy(task_budget_usd),
    )


__all__ = [
    "DEEPSWE_COMPACT_PLAN_MAX_BUDGET_USD",
    "DEEPSWE_BREACH_POLICY",
    "DEEPSWE_REVIEW_POLICY",
    "resolve_deepswe_plan_policy",
    "resolve_deepswe_breach_policy",
    "synthesize_starter_spec",
    "uses_deepswe_compact_plan_policy",
]
