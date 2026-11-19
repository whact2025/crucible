# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""DeepSWE workflow worker — per-task cascade orchestration.

Drives the implement / breach / review loop against the
datacurve-ai/deep-swe corpus. For each filtered task, the worker
runs an isolated multi-workspace container model:

1. Populate independent candidate, baseline, replay, and grading workspaces
   from the same immutable image and base commit. Start clean-baseline and
   authoring containers without hidden mounts; the baseline is used only for
   differential public verification.
2. ``start_authoring_container`` — run with
   ``--network=none``, bind-mount host workspace at ``/app`` (so
   the implementer/defender's host-side edits flow into the
   in-container ``dev_verify`` gate) and host logs dir at ``/logs``.
   This container mounts NEITHER ``/tests`` NOR ``/solution`` — the
   hidden dataset tests and the gold solution are structurally
   absent during authoring.
3. ``synthesize_starter_spec(inst)`` → Spec, persist to per-task
   Corpus.
4. ``run_architect(extra_parameters={"deepswe_metadata": {...}},
   checkpoint_hook=DeepSWEArchitectCheckpointHook(...),
   breach_policy=DEEPSWE_BREACH_POLICY,
   review_policy=DEEPSWE_REVIEW_POLICY)``. Runner metadata lets tier slice +
   breach rounds select the
   ``dev_verify`` gate (per-round adversarial tests run inside the
   authoring container) and drop all host SDK builtins so the
   agents work exclusively through the container-bash MCP tool. The separate
   product-owned checkpoint hook records bounded mutation settlement without
   exposing DeepSWE semantics to Architect.
5. After Tier restores pipeline-owned breach artifacts, capture the complete
   content-addressed repository patch and replay its exact bytes onto the
   proof-only checkout populated from the immutable task image.
6. Publish the byte-exact captured candidate as ``/logs/artifacts/model.patch``,
   then ``start_verifier_container`` (+ ``/tests:ro``, still NO
   ``/solution``) against the separate fresh grading checkout. The dataset
   grader applies ``model.patch`` there exactly once before running
   ``deepswe_verifier`` ONCE for the authoritative reward (v1.0
   ``reward.txt`` or v1.1 ``reward.json``). The retained ``model.patch`` must
   still byte-match the captured candidate after the verifier stops. This is
   the only place the hidden tests run.
7. Prove every container stopped before trusting its outputs, then tear down
   all workspaces and mounts.

The workflow supports serial cascade mode (``--implement``) plus ``--list``
and ``--dry-run`` validation modes.

The per-task Corpus is stable across attempts, but every attempt gets a
fresh workspace populated from the image. Existing lineage therefore forces
``restart=True`` so landed slices are replayed rather than incorrectly
skipped against a clean checkout.

"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from assay import (
    LifecycleEvent,
    Measurement,
    Spec,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus
from assay.types.spec import ProjectContext

from crucible.backends.config import (
    bind_routing_snapshot,
    load_routing,
    routing_feature_fingerprint,
    routing_fingerprint,
    routing_source_fingerprint,
)
from crucible.backends.pricing import load_pricing, pricing_for_model
from crucible.common.cost import CostAccumulator, PRICING_EVIDENCE_SCHEMA
from crucible.common.benchmark_runtime import (
    ImmutableImageError,
    TaskLeaseBusy,
    canonical_fingerprint,
    file_digest,
    file_set_fingerprint,
    image_verification_policy_identity,
    managed_container_labels,
    managed_container_name,
    task_execution_lease,
    verify_immutable_image,
)
from crucible.common.docker_workflow_settlement import (
    DockerSettlementRegistrationClient,
)
from crucible.common.storage_quota import (
    DirectoryGrowthQuota,
    StorageQuotaExceeded,
)
from crucible.products.deepswe.dataset import (
    DeepSWEInstance,
    DeepSWEInstanceError,
    DeepSWETestsTreeIdentity,
    filter_instances,
    fingerprint_task_tests_tree,
    load_dataset,
    load_snapshot_test_inventory,
    load_task_test_inventory,
    snapshot_task_tests_tree,
)
from crucible.products.deepswe.dev_verify_overrides import (
    resolve_dev_verify_command,
)
from crucible.products.deepswe.instance_container import (
    InstanceContainerSetupError,
    RustToolchainUnavailable,
    capture_model_patch_from_container,
    check_disk_headroom,
    docker_available,
    is_full_commit_id,
    populate_workspace_from_image,
    pull_image,
    reconcile_orphans,
    start_authoring_container,
    start_patch_render_container,
    start_verifier_container,
    stop_instance_container_strict,
    teardown_instance_container,
)
from crucible.products.deepswe.mutation_settlement import (
    DEEPSWE_CHECKPOINT_EXTENSION_KEY,
    DEEPSWE_CHECKPOINT_EXTENSION_SCHEMA,
    DEEPSWE_MUTATION_SETTLEMENT_SCHEMA,
    DeepSWEArchitectCheckpointHook,
)
from crucible.workflows._deepswe_baseline_preflight import (
    DevelopmentPreflightRestorationReceipt,
    build_development_preflight_parameters,
    run_development_verifier_preflight,
)
from crucible.products.deepswe.spec_synthesis import (
    DEEPSWE_BREACH_POLICY,
    DEEPSWE_COMPACT_PLAN_MAX_BUDGET_USD,
    DEEPSWE_REVIEW_POLICY,
    resolve_deepswe_breach_policy,
    resolve_deepswe_plan_policy,
    synthesize_starter_spec,
    uses_deepswe_compact_plan_policy,
)
# DeepSWE shares SWE-bench's issue-fixing implementer contract (run
# tests, iterate, write-early), extended with an API-contract fidelity
# clause (DeepSWE instructions state the exact public API the hidden
# verifier asserts; the cascade's slice paraphrases must not override it).
# Opts out of assay's don't-run-tests default.
from crucible.products.deepswe.prompts import DEEPSWE_SYSTEM_PROMPT
from crucible.products.deepswe.pristine_verifier import (
    CandidateApplyReport,
    CandidateApplyStatus,
    CandidateCaptureStatus,
    DeepSWECandidate,
    PristineReplayReport,
    PristineReplayStatus,
    VerifierPatchReport,
    VerifierPatchStatus,
    candidate_from_image_patch,
    encode_candidate_envelope,
    prepare_candidate_replay,
    prove_image_replay,
    validate_candidate,
    verify_hidden_model_patch,
)
from crucible.products.deepswe.patch_limits import (
    configured_patch_byte_limit,
    validate_patch_byte_limit,
)
from crucible.products.deepswe.secure_artifacts import (
    RewardCounts,
    RewardTestInventory,
    read_reward_artifact,
)
from crucible.workflows._args import (
    add_backend_limit_args,
    nonnegative_budget_arg,
    positive_budget_arg,
    positive_int_arg,
    validate_backend_limit_args,
)
from crucible.workflows import _outcome
from crucible.workflows._backend_recipe import backend_recipe_identity
from crucible.workflows._development_budget import (
    DevelopmentBudgetPolicy,
    RoleAllocation,
)
from crucible.workflows._plan_competition import (
    DEFAULT_PLANNER_COUNT,
    planner_count_arg,
    validate_competition_admission,
)
from crucible.workflows._benchmark_planning import (
    BENCHMARK_PLANNING_BUDGET_SHARE,
    BenchmarkPlanningPolicy,
    benchmark_planning_identity,
    planning_recipe_identity,
)
from crucible.workflows._run import (
    exec_ctx_now,
    open_run_with_capture,
    read_assay_version,
    write_atomic_bytes,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def load_e12_arm_set_for_deepswe():
    """Load DeepSWE's exact, registry-derived E12 A/B/C route checkpoint."""

    from ._abc_evaluation import load_e12_arm_set

    return load_e12_arm_set(
        Path(__file__).resolve().parents[1]
        / "products"
        / "deepswe"
        / "e12_arms.yaml"
    )

_EMPTY_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()
DEFAULT_PLANNER_TOOL_CALL_LIMIT = 32
DEFAULT_PLANNER_TOOL_RESULT_BYTES = 262_144
DEFAULT_PLANNER_TOOL_CALL_TRANCHES = (24, 4, 4)
_PLANNER_SEMANTIC_CANDIDATE_LIMIT = (
    resolve_deepswe_plan_policy(DEEPSWE_COMPACT_PLAN_MAX_BUDGET_USD).max_critique_rounds
    + 1
)
_PLANNER_CONDITIONAL_REPAIR_LIMIT = 2
_PLANNER_PHYSICAL_CALL_LIMIT = (
    _PLANNER_SEMANTIC_CANDIDATE_LIMIT
    + _PLANNER_CONDITIONAL_REPAIR_LIMIT
)
_KIMI_PLANNER_RESPONSE_MAX_TOKENS = 16_384
_PLANNER_CUMULATIVE_OUTPUT_TOKENS = (
    _KIMI_PLANNER_RESPONSE_MAX_TOKENS * _PLANNER_PHYSICAL_CALL_LIMIT
)
_REVIEW_SEMANTIC_CANDIDATE_LIMIT = _PLANNER_SEMANTIC_CANDIDATE_LIMIT
_REVIEW_RESPONSE_ATTEMPTS_PER_CANDIDATE = 3
_REVIEW_PHYSICAL_CALL_LIMIT = (
    _REVIEW_SEMANTIC_CANDIDATE_LIMIT
    * _REVIEW_RESPONSE_ATTEMPTS_PER_CANDIDATE
)
_KIMI_REVIEW_RESPONSE_MAX_TOKENS = 32_768
_REVIEW_CUMULATIVE_OUTPUT_TOKENS = (
    _KIMI_REVIEW_RESPONSE_MAX_TOKENS * _REVIEW_PHYSICAL_CALL_LIMIT
)


def _deepswe_development_budget_policy(
    task_budget_usd: float,
    *,
    planner_count: int = DEFAULT_PLANNER_COUNT,
) -> DevelopmentBudgetPolicy:
    """Freeze DeepSWE's shared 40/50/10 role authority for one task."""

    validate_competition_admission(planner_count, task_budget_usd)
    planner_calls = (
        _PLANNER_PHYSICAL_CALL_LIMIT if planner_count == 1 else planner_count
    )
    critic_calls = (
        _REVIEW_PHYSICAL_CALL_LIMIT
        if planner_count == 1
        else max(_REVIEW_PHYSICAL_CALL_LIMIT, (planner_count + 1) * 5)
    )
    judge_calls = (
        _REVIEW_PHYSICAL_CALL_LIMIT
        if planner_count == 1
        else max(_REVIEW_PHYSICAL_CALL_LIMIT, (planner_count + 1) * 6 + 4)
    )
    planning_roles = {
        # Independent drafts make one generation call each. Explicit single-
        # track runs retain their bounded semantic/correction loop. The source
        # and dollar authorities remain shared across the complete task.
        "architect_planner": (
            planner_calls,
            max(
                _PLANNER_CUMULATIVE_OUTPUT_TOKENS,
                planner_calls * _KIMI_PLANNER_RESPONSE_MAX_TOKENS,
            ),
            3_600.0,
        ),
        # Each draft and optional synthesis receives independent review.
        # Comparison is a separate Judge call. Every review owns the same
        # two bounded response repairs. Physical ceilings additionally cover
        # output-length and GLM target-ledger/decision repairs: five Critic
        # attempts, six Judge attempts, and four comparison attempts. Preserve
        # previously declared output and wall authority while the USD cap remains
        # authoritative; private reasoning is billable output on Kimi.
        "architect_critic": (
            critic_calls,
            max(
                _REVIEW_CUMULATIVE_OUTPUT_TOKENS,
                critic_calls * _KIMI_REVIEW_RESPONSE_MAX_TOKENS,
            ),
            1_800.0,
        ),
        "architect_judge": (
            judge_calls,
            max(
                _REVIEW_CUMULATIVE_OUTPUT_TOKENS,
                judge_calls * _KIMI_REVIEW_RESPONSE_MAX_TOKENS,
            ),
            1_800.0,
        ),
    }
    development_roles = {
        "implementer": (12, 160_000, 10_800.0),
        "impl_breaker": (10, 96_000, 7_200.0),
        "breach_defender": (10, 128_000, 10_800.0),
        "breach_judge": (10, 96_000, 7_200.0),
        "diff_scope": (3, 32_000, 1_800.0),
        "tier_completeness": (3, 32_000, 1_800.0),
        "api_contract_fidelity": (3, 32_000, 1_800.0),
        "review_conflict_authority": (3, 32_000, 1_800.0),
    }
    planning_cap = float(task_budget_usd) * BENCHMARK_PLANNING_BUDGET_SHARE
    development_reserve = float(task_budget_usd) * 0.50
    contingency = float(task_budget_usd) * 0.10
    development_authority = development_reserve + contingency
    # Preserve the full contingency tail across implementation and breach.
    # Compact tasks reserve a shared Test Author / production Defender tranche
    # so an exact repair dispatch remains reachable after attack authoring.
    # Larger tasks retain the established terminal-review allocation; they do
    # not need the final-only nested-budget closure policy to create capacity.
    # These are protected minimums, not reviewer micro-ceilings: an owner may
    # borrow otherwise-uncommitted implementation/verifier authority while
    # every sibling's outstanding minimum remains unavailable.
    mandatory_tail_minimums = (
        {
            "impl_breaker": float(task_budget_usd) * 0.05,
            "breach_defender": float(task_budget_usd) * 0.03,
            "diff_scope": float(task_budget_usd) * 0.01,
            "tier_completeness": float(task_budget_usd) * 0.01,
        }
        if uses_deepswe_compact_plan_policy(task_budget_usd)
        else {
            "impl_breaker": float(task_budget_usd) * 0.05,
            "diff_scope": float(task_budget_usd) * 0.025,
            "tier_completeness": float(task_budget_usd) * 0.025,
        }
    )
    allocations = tuple(
        RoleAllocation(
            role=role,
            group="planning_review",
            max_calls=limits[0],
            max_output_tokens=limits[1],
            max_wall_seconds=limits[2],
            max_cost_usd=planning_cap,
        )
        for role, limits in sorted(planning_roles.items())
    ) + tuple(
        RoleAllocation(
            role=role,
            group="implementation_verifier",
            max_calls=limits[0],
            max_output_tokens=limits[1],
            max_wall_seconds=limits[2],
            # Implementer is the only development role that may need the
            # planning authority released after implementation admission.
            # The ledger's active group cap and outstanding protected tails
            # remain the tighter per-call authority; this role ceiling merely
            # avoids stranding an otherwise valid one-way transfer.
            max_cost_usd=(
                float(task_budget_usd)
                if role == "implementer"
                else development_authority
            ),
            protected_tail_usd=mandatory_tail_minimums.get(role, 0.0),
        )
        for role, limits in sorted(development_roles.items())
    )
    return DevelopmentBudgetPolicy(
        task_budget_usd=float(task_budget_usd),
        planning_review_cap_usd=planning_cap,
        implementation_verifier_reserve_usd=development_reserve,
        contingency_usd=contingency,
        roles=allocations,
        release_unused_planning_authority=True,
    )


def _deepswe_planning_envelope(
    task_budget_usd: float | None,
    *,
    planner_count: int = DEFAULT_PLANNER_COUNT,
) -> dict[str, Any]:
    """Return the complete budget-derived planning policy identity."""

    compact = uses_deepswe_compact_plan_policy(task_budget_usd)
    policy = resolve_deepswe_plan_policy(task_budget_usd)
    return {
        **benchmark_planning_identity(
            planner_count,
            planning_budget_share=(
                None
                if planner_count == 1 and task_budget_usd is None
                else BENCHMARK_PLANNING_BUDGET_SHARE
            ),
        ),
        "class": (
            "compact_one_tier_one_slice"
            if compact else "standard_feature_decomposition"
        ),
        "compact_max_budget_usd": DEEPSWE_COMPACT_PLAN_MAX_BUDGET_USD,
        "critic_rubric": "plan_critic" if compact else "deepswe_plan_critic",
        "max_total_tiers": policy.max_total_tiers,
        "max_total_slices": policy.max_total_slices,
    }


class _HiddenOraclePersistenceError(RuntimeError):
    """A one-shot verifier result could not reach durable authority."""


class DeepSWEVerifierStatus(StrEnum):
    """Typed infrastructure/domain state for the one-shot hidden verifier."""

    NOT_RUN = "not_run"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    TIMEOUT = "timeout"
    COMMAND_FAILED = "command_failed"
    REWARD_MISSING = "reward_missing"
    REWARD_INVALID = "reward_invalid"
    REWARD_MISMATCH = "reward_mismatch"
    PATCH_MISSING = "patch_missing"
    PATCH_MISMATCH = "patch_mismatch"
    PATCH_UNREADABLE = "patch_unreadable"


def _patch_byte_limit_arg(raw: str) -> int:
    try:
        return validate_patch_byte_limit(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="crucible-workflow-deepswe")
    p.add_argument(
        "--spec", required=True, type=Path,
        help="Corpus-shaped spec directory. Per-task Corpora land "
             "under <spec>/instances/<task_id>/corpus/; run artifacts "
             "(summary.json) land under <spec>/runs/<ts>_deepswe_*/.",
    )
    p.add_argument(
        "--dataset", required=True, type=Path,
        help="Path to the deep-swe clone's tasks/manifest.json. Fetch "
             "out-of-band: `git clone "
             "https://github.com/datacurve-ai/deep-swe.git`.",
    )
    # Filters.
    p.add_argument(
        "--task-id", action="append", default=[],
        help="Whitelist by task_id; repeatable. Empty (default) = "
             "no task-id filter.",
    )
    p.add_argument(
        "--repo", action="append", default=[],
        help="Whitelist by repo (owner/name short form, e.g. "
             "``abs-lang/abs``); repeatable.",
    )
    p.add_argument(
        "--language", action="append", default=[],
        help="Whitelist by language (typescript / python / go / rust / "
             "javascript); case-insensitive; repeatable.",
    )
    p.add_argument(
        "--category", action="append", default=[],
        help="Whitelist by category (feature_request / bugfix / "
             "enhancement); repeatable.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap the post-filter task count (default: no cap).",
    )
    # Mode flags (mutually exclusive).
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--list", action="store_true", dest="list_only",
        help="Print the filtered task list (task_id\\trepo\\tlanguage"
             "\\tcategory) and exit. No cascade work.",
    )
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Print the per-task plan (one line per task) without "
             "running any cascade work. Useful for validating filters "
             "+ counts before kicking a sweep.",
    )
    mode.add_argument(
        "--implement", action="store_true",
        help="Run the cascade per task (plan → tier → breach → review "
             "→ verify) in a serial task loop.",
    )
    # No single-shot baseline mode is exposed; all implementation runs use the
    # cascade so CLI matrices cannot select a nonfunctional execution path.
    # Cascade knobs.
    p.add_argument(
        "--backend-config", type=Path, default=None,
        help="Path to a backend routing YAML overriding discovery.",
    )
    p.add_argument(
        "--budget-usd", type=nonnegative_budget_arg, default=None,
        help="Aggregate USD budget across the whole sweep. Workers "
             "halt the next task launch when cumulative cost reaches "
             "this floor.",
    )
    p.add_argument(
        "--budget-usd-per-instance", type=positive_budget_arg, default=None,
        help="Per-task USD cap applied independently to each cascade.",
    )
    p.add_argument(
        "--max-turns", type=positive_int_arg, default=90,
        help="Per-task max architect turns (planner + implementer). "
             "Default 90 gives the planner headroom to finish "
             "investigating and emit a TierDAG before the cap.",
    )
    p.add_argument(
        "--planner-count", type=planner_count_arg, default=DEFAULT_PLANNER_COUNT,
        help="Independent plans per task (default: 3), sharing that task's "
             "planning and total USD authority. Use 1 explicitly for the "
             "single-track refinement or diagnostic-cost mode.",
    )
    add_backend_limit_args(p)
    p.add_argument(
        "--planner-tool-call-limit",
        type=positive_int_arg,
        default=DEFAULT_PLANNER_TOOL_CALL_LIMIT,
        help="Hard per-task ceiling on planner source-tool dispatches "
             f"(default: {DEFAULT_PLANNER_TOOL_CALL_LIMIT}).",
    )
    p.add_argument(
        "--planner-tool-result-bytes",
        type=positive_int_arg,
        default=DEFAULT_PLANNER_TOOL_RESULT_BYTES,
        help="Hard per-task cumulative UTF-8 ceiling on successful planner "
             "source-tool payload bytes; workflow control/error annotations "
             f"are excluded (default: {DEFAULT_PLANNER_TOOL_RESULT_BYTES}).",
    )
    p.add_argument(
        "--max-patch-bytes",
        type=_patch_byte_limit_arg,
        default=configured_patch_byte_limit(),
        help="Absolute candidate/model.patch byte cap (default: 64 MiB; "
             "bounded above by 512 MiB).",
    )
    p.add_argument(
        "--no-include-breach", action="store_true",
        help="Disable the breach loop for this run. Default ON via "
             "DEEPSWE_BREACH_POLICY.enabled=True.",
    )
    p.add_argument(
        "--replan", action="store_true",
        help="Regenerate each task's plan. Accepted plans are otherwise "
             "reused only when their saved execution and planning policy "
             "match this run. Completed task checkpoints still require --rerun.",
    )
    p.add_argument(
        "--rerun", action="store_true",
        help="Re-execute every selected task even when its canonical "
             "checkpoint is reusable.",
    )
    p.add_argument(
        "--rerun-status", action="append", default=[], choices=(
            "candidate_failed", "cascade_error", "pristine_replay_failed",
            "resolved", "setup_failed", "storage_quota_exceeded",
            "toolchain_unavailable",
            "unresolved", "verification_failed", "verification_incomplete",
        ),
        help="Re-execute checkpoints with this status; repeatable. "
             "Pre-oracle infrastructure statuses retry by default; hidden "
             "oracle statuses require this flag or --rerun.",
    )
    p.add_argument(
        "--lifecycle-out", type=Path, default=None,
        help="If set, checkpoint one durable lifecycle record per task "
             "(`{task_id, ext_id, status, rc, wall_seconds, ...}`) "
             "in JSONL format. Repeated runs repair a torn tail, skip "
             "already-recorded task identities, and never add duplicate "
             "rows. This is a LIFECYCLE report, not a "
             "leaderboard predictions JSONL — model.patch capture "
             "from /logs/artifacts/model.patch is a separate "
             "concern not exposed by this flag.",
    )
    # Disk-headroom override.
    p.add_argument(
        "--skip-disk-check", action="store_true",
        help="Skip the free-disk floor check. Equivalent to setting "
             "CRUCIBLE_DEEPSWE_SKIP_DISK_CHECK=1.",
    )
    args = p.parse_args(argv)
    validate_backend_limit_args(p, args)
    if args.diagnostic_cost_policy is not None and args.planner_count != 1:
        p.error("diagnostic cost policy requires explicit --planner-count 1")
    if args.implement and args.planner_count > 1:
        assigned = _resolve_child_budget(
            aggregate_budget=args.budget_usd,
            spent_usd=0.0,
            per_instance_budget=args.budget_usd_per_instance,
        )
        # A zero aggregate sweep cap is an intentional no-launch terminal.
        # Every competition actually admitted still needs positive authority.
        if assigned != 0.0:
            try:
                validate_competition_admission(args.planner_count, assigned)
            except ValueError as exc:
                p.error(str(exc))
    if args.diagnostic_cost_policy is not None and (args.rerun or args.rerun_status):
        p.error("diagnostic upper-bound pilot requires a fresh task, not a rerun")
    # Establish trusted path roots once. Downstream secure dirfd/openat reads
    # reject relative roots rather than re-resolving attacker-controlled paths.
    for field in ("spec", "dataset", "backend_config", "lifecycle_out"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, Path(value).expanduser().resolve())
    return args


def _load_and_filter(args: argparse.Namespace) -> tuple[DeepSWEInstance, ...]:
    """Load the dataset + apply CLI filters. Returns the filtered
    tuple. Raises on dataset-load failure; explicit filter
    mismatches return an empty tuple (caller handles)."""
    instances = load_dataset(args.dataset)
    return filter_instances(
        instances,
        task_ids=args.task_id or None,
        repos=args.repo or None,
        languages=args.language or None,
        categories=args.category or None,
        limit=args.limit,
    )


def _open_or_create_corpus(spec_dir: Path) -> Corpus:
    """Open the workspace-level Corpus, materializing a minimal
    Spec if none exists (so per-task chains have a valid version=0
    to anchor onto). Mirrors the swebench worker's behavior.

    Uses ``Corpus.put_spec`` (not ``put_record``) because the
    cascade reads ``latest_spec`` via the versions/ index, not by
    listing the artifacts/specs/ dir. ``put_spec`` writes BOTH
    the content-hashed artifact AND the versions/v<N>_<hash>.json
    pointer; ``put_record(spec, kind='specs')`` only writes the
    former and leaves the cascade saying "no spec manifest found
    under .../versions".
    """
    spec_dir.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(spec_dir)
    if corpus.latest_spec() is None:
        synthetic = Spec(
            version=0, parent=None, root=None, schema_version=1,
            project_context=ProjectContext(
                project="deepswe", repo_url="https://deep-swe/",
                base_commit="0" * 40,
                branch=None, submodule_path=None, runners={},
            ),
            execution=None,
            intent="DeepSWE workflow root spec (auto-synthesized).",
            constraints=(),
            done_criteria=(),
            plan_policy=None,
        )
        corpus.put_spec(synthetic)
    return corpus


def _list_mode(filtered: tuple[DeepSWEInstance, ...]) -> int:
    for inst in filtered:
        print(f"{inst.task_id}\t{inst.repo}\t{inst.language}\t{inst.category}")
    return 0


def _dry_run_mode(filtered: tuple[DeepSWEInstance, ...]) -> int:
    """Per-task plan summary: task_id, image, base_commit, language,
    category, instruction byte size. No actual cascade work — useful
    for filter validation before a sweep."""
    print(f"# DeepSWE dry-run plan: {len(filtered)} tasks")
    for inst in filtered:
        print(
            f"{inst.task_id}\t{inst.docker_image}\t"
            f"{inst.base_commit_hash[:12]}\t{inst.language}\t"
            f"{inst.category}\t"
            f"instruction={len(inst.instruction)}B"
        )
    return 0


def _resolve_child_budget(
    per_instance_budget: float | None,
    aggregate_budget: float | None,
    spent_usd: float,
) -> float | None:
    """Return the tighter per-task cap after prior aggregate spend."""

    remaining = (
        None
        if aggregate_budget is None
        else max(0.0, aggregate_budget - spent_usd)
    )
    if per_instance_budget is None:
        return remaining
    if remaining is None:
        return per_instance_budget
    return min(per_instance_budget, remaining)


def _build_run_architect_kwargs(
    *,
    args: argparse.Namespace,
    inst: DeepSWEInstance,
    instance_spec_dir: Path,
    container_id: str,
    instance_log_dir: Path,
    host_workspace: Path,
    baseline_container_id: str | None = None,
    restart: bool = False,
    budget_usd: float | None = None,
    execution_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Compose the kwargs run_architect needs for one DeepSWE task.

    deepswe_metadata is plumbed through ``extra_parameters`` so:

    - ``deepswe_dev_verify`` (the per-round breach gate) can read the
      task's ``language`` to pick the project-native test command and
      docker-exec into the authoring container.
    - ``deepswe_verifier`` (the final dataset gate) can docker-exec
      into the verifier container and read the v1.0 ``reward.txt`` or v1.1
      ``reward.json`` artifact from the right host logs dir.

    Mutation settlement is deliberately not conveyed through that opaque
    runner mapping. ``checkpoint_hook`` is a typed product-owned adapter, so
    Architect never inspects DeepSWE metadata or proof semantics.

    ``cwd`` is set to ``host_workspace`` (NOT ``instance_spec_dir``):
    the host workspace is the ``/app``-mirrored git repo, so
    ``assay.implement``'s ``git diff`` capture sees the implementer's
    actual edits and the planner's read-only source tools (rooted at
    ``repo_root = cwd``) inspect the code rather than the Corpus folder.
    ``spec`` stays the Corpus dir so ``latest_spec`` resolves.
    This mirrors SWE-bench, which passes ``spec=instance_corpus_dir``
    + ``cwd=str(workspace)`` as two distinct dirs.
    """
    if baseline_container_id is not None:
        # The ordinary cascade and the standalone quality launcher share this
        # exact closed projection.  Keep the legacy no-baseline helper shape
        # below for tests and callers that do not run differential verification.
        deepswe_metadata = build_development_preflight_parameters(
            instance=inst,
            candidate_container_id=container_id,
            baseline_container_id=baseline_container_id,
            logs_dir=instance_log_dir,
            max_patch_bytes=args.max_patch_bytes,
        )["deepswe_metadata"]
    else:
        deepswe_metadata = {
            "container_id": container_id,
            "logs_dir": str(instance_log_dir),
            # Dispatch key for deepswe_dev_verify's per-language test
            # command. Threaded here because the DeepSWE starter spec
            # carries no LanguagesConstraint, so _build_verify's
            # languages-injection path never fires.
            "language": inst.language,
            # Carry the task_id + ext_id for traceability in lifecycle
            # events and Measurement raw payloads.
            "task_id": inst.task_id,
            "ext_id": inst.ext_id,
            # Bounds every Tier/Architect settlement capture before persistence.
            "max_patch_bytes": args.max_patch_bytes,
        }
        # Preserve the pre-existing no-baseline helper contract.
        dev_verify_command = resolve_dev_verify_command(
            task_id=inst.task_id, repo=inst.repo,
        )
        if dev_verify_command:
            deepswe_metadata["dev_verify_command"] = dev_verify_command
    breach_policy = resolve_deepswe_breach_policy(
        enabled=not args.no_include_breach,
    )
    effective_task_budget = (
        args.budget_usd_per_instance
        if budget_usd is None
        else budget_usd
    )
    planning_envelope = _deepswe_planning_envelope(
        effective_task_budget, planner_count=args.planner_count,
    )
    return dict(
        spec=instance_spec_dir,
        cwd=str(host_workspace),
        backend_config=args.backend_config,
        budget_usd=effective_task_budget,
        backend_limit_mode=args.backend_limit_mode,
        backend_limit_deadline_seconds=(
            args.backend_limit_deadline_seconds
        ),
        diagnostic_cost_policy=getattr(args, "diagnostic_cost_policy", None),
        development_budget_policy=(
            _deepswe_development_budget_policy(
                effective_task_budget, planner_count=args.planner_count,
            )
            if effective_task_budget is not None
            and effective_task_budget > 0
            else None
        ),
        max_turns=args.max_turns,
        planner_count=args.planner_count,
        planning_policy=BenchmarkPlanningPolicy(
            planner_count=args.planner_count,
            planning_budget_share=planning_envelope["planning_budget_share"],
            execution_fingerprint=execution_fingerprint,
        ),
        planner_tool_call_limit=args.planner_tool_call_limit,
        planner_tool_result_bytes=args.planner_tool_result_bytes,
        planner_tool_call_tranches=(
            (args.planner_tool_call_limit,)
            if args.planner_count > 1
            else (
                DEFAULT_PLANNER_TOOL_CALL_TRANCHES
                if args.planner_tool_call_limit
                == sum(DEFAULT_PLANNER_TOOL_CALL_TRANCHES)
                else None
            )
        ),
        include_review=True,
        # The ordinary feature-work critic enforces decomposed increments.
        # Under the explicitly compact task envelope, the persisted
        # PlanPolicy instead requires one complete tier/slice, so use the
        # generic Critic: asking the feature critic to split the only allowed
        # slice would make the policy internally contradictory.
        critic_rubric=str(planning_envelope["critic_rubric"]),
        # Benchmark planning must ground proposed paths and symbols in the
        # task checkout. Architect attaches only build_source_tools under its
        # read_only capability; SDK-native/write tools remain unavailable.
        enable_source_tools=True,
        replan=args.replan,
        # The host workspace is populated fresh from the image on every
        # attempt. Existing Corpus implementations belong to an earlier
        # workspace and therefore must be replayed, never resume-skipped.
        restart=restart,
        # This capability is intentionally programmatic-only. The worker has
        # just repopulated the bind-mounted workspace from the immutable image,
        # so old later-tier Corpus records are audit lineage, not live bytes.
        fresh_lineage=restart,
        breach_policy=breach_policy,
        review_policy=DEEPSWE_REVIEW_POLICY,
        breach_issue_text=inst.instruction,
        extra_parameters={"deepswe_metadata": deepswe_metadata},
        # Architect observes only generic checkpoint coordinates. DeepSWE owns
        # mutation-journal semantics, patch bounds, and the stronger clean
        # zero-tier proof through this opaque product hook.
        checkpoint_hook=DeepSWEArchitectCheckpointHook(
            task_id=inst.task_id,
            ext_id=inst.ext_id,
            expected_base_commit=inst.base_commit_hash,
            max_patch_bytes=args.max_patch_bytes,
        ),
        impl_system_prompt=DEEPSWE_SYSTEM_PROMPT,
        # The authoring container bind-mounts a SINGLE host workspace at
        # /app for the whole instance, so every landed slice's writes
        # (plus breach-round mutations) persist on disk across tiers
        # within a run. Tell the cascade not to re-apply dependency diffs
        # between tiers — the workspace already is the accumulated state;
        # re-applying double-applies and hard-fails (the tier-handoff bug
        # the adaptix pilot surfaced 2026-06-13).
        #
        # Within this invocation the bind-mounted workspace persists across
        # tiers. Across invocations ``restart=True`` replays all slices into
        # the newly populated workspace (see above).
        persistent_workspace=True,
    )


def _run_dev_verify_preflight(
    *,
    spec: Spec,
    cwd: Path,
    baseline_cwd: Path,
    log_dir: Path,
    journal_root: Path,
    extra_parameters: dict[str, Any],
    restoration_callback: (
        Callable[[DevelopmentPreflightRestorationReceipt], None] | None
    ) = None,
):
    """Compatibility seam for the extracted provider-free preflight."""

    return run_development_verifier_preflight(
        spec=spec,
        candidate_workspace=cwd,
        baseline_workspace=baseline_cwd,
        log_dir=log_dir,
        journal_root=journal_root,
        extra_parameters=extra_parameters,
        restoration_callback=restoration_callback,
    )


def _gate_created_paths_summary(
    restorations: list[DevelopmentPreflightRestorationReceipt],
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Return the candidate's gate-created paths and their summary entries.

    The summary lists at most as many paths as a gate run can be bound to
    and escapes a name that is not valid UTF-8, so it stays bounded and every
    JSON writer can encode it; ``gate_created_path_count`` is the full count.
    """

    from crucible.products.deepswe.runners import (  # noqa: PLC0415
        deepswe_dev_verify,
    )

    # The candidate checkout is the first workspace of a receipt.
    paths = (
        restorations[-1].workspaces[0].gate_created_paths
        if restorations
        else ()
    )
    return paths, {
        "gate_created_paths": [
            path.encode("utf-8", "backslashreplace").decode("utf-8")
            for path in paths[: deepswe_dev_verify.MAX_GATE_CREATED_PATHS]
        ],
        "gate_created_path_count": len(paths),
        "gate_created_paths_bound": False,
    }


def _clean_infrastructure_provenance_from_runner(
    runner: Any,
) -> dict[str, object] | None:
    """Extract only bounded clean-probe evidence from trusted runner stdout."""

    from crucible.products.deepswe.runners import (  # noqa: PLC0415
        deepswe_dev_verify,
    )

    try:
        payload = json.loads(runner.raw_output.decode("utf-8"))
    except (
        AttributeError,
        RecursionError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None
    raw = (
        payload.get(deepswe_dev_verify.CLEAN_INFRASTRUCTURE_PROVENANCE_KEY)
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "language",
        "command_sha256",
        "flags",
    }:
        return None
    flags = raw.get("flags")
    if (
        raw.get("schema_version")
        != deepswe_dev_verify.CLEAN_INFRASTRUCTURE_PROVENANCE_SCHEMA
        or not isinstance(raw.get("language"), str)
        or not isinstance(raw.get("command_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", raw["command_sha256"]) is None
        or not isinstance(flags, dict)
        or set(flags)
        != set(deepswe_dev_verify.PROVENANCE_INFRASTRUCTURE_FLAGS)
        or any(type(value) is not bool for value in flags.values())
    ):
        return None
    return {
        "schema_version": raw["schema_version"],
        "language": raw["language"],
        "command_sha256": raw["command_sha256"],
        "flags": dict(flags),
    }


_VERIFIER_OBSERVABILITY_FIELDS = (
    "command_returncode",
    "baseline_test_returncode",
    "new_test_returncode",
    "diagnostics_persisted",
    "verifier_stdout_bytes",
    "verifier_stderr_bytes",
    "verifier_stdout_truncated",
    "verifier_stderr_truncated",
)

_VERIFIER_BOOLEAN_KPI_FIELDS = frozenset({
    "diagnostics_persisted",
    "verifier_stdout_truncated",
    "verifier_stderr_truncated",
})


def _optional_kpi_bool(value: Any) -> bool | None:
    if value in (0, 0.0, False):
        return False
    if value in (1, 1.0, True):
        return True
    return None


def _run_dataset_verify(
    inst: DeepSWEInstance,
    *,
    container_id: str,
    instance_log_dir: Path,
) -> dict[str, Any]:
    """Run the authoritative dataset gate ONCE against a verifier container.

    The returned verdict retains the legacy ``verify_returncode`` and ``rc``
    fields while also carrying bounded command diagnostics and explicit
    baseline/new-test return codes from the runner.

    This is the only place the hidden ``/tests/test.sh`` runs. It is
    invoked AFTER the cascade — the cascade itself uses the
    ``dev_verify`` gate against the authoring container (no
    ``/tests``), so the dataset verdict here is independent of any
    per-round adversarial tests the Defender authored.

    Drives ``deepswe_verifier.main`` directly via in-memory stdio
    rather than the runner subprocess plumbing: the worker already
    holds the live verifier ``container_id`` + ``logs_dir``, so the
    BenchConfig is trivial to construct and we avoid re-resolving
    runners.
    """
    import io  # noqa: PLC0415

    from crucible.products.deepswe.runners import (  # noqa: PLC0415
        deepswe_verifier,
    )

    bench_config = json.dumps({
        "parameters": {
            "role": "verify",
            "deepswe_metadata": {
                "container_id": container_id,
                "logs_dir": str(instance_log_dir),
                "defer_reward_read": True,
            },
        },
    })
    out = io.StringIO()
    rc = deepswe_verifier.main(io.StringIO(bench_config), out)
    values: dict[str, Any] = {}
    try:
        payload = json.loads(out.getvalue() or "{}")
        values = payload.get("values", {}) if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        values = {}
    command_returncode = values.get("command_returncode")
    legacy_returncode = values.get("verify_returncode")
    if command_returncode is None:
        command_returncode = legacy_returncode
    if legacy_returncode is None:
        legacy_returncode = command_returncode
    verdict: dict[str, Any] = {
        "rc": int(rc),
        "reward": None,
        "resolved": None,
        "command_returncode": command_returncode,
        "verify_returncode": legacy_returncode,
        "reward_pending": bool(values.get("reward_pending")),
        "reward_valid": False,
        "reward_state": "pending_stop_validation",
    }
    for field in _VERIFIER_OBSERVABILITY_FIELDS:
        if field == "command_returncode":
            continue
        value = values.get(field)
        if field in _VERIFIER_BOOLEAN_KPI_FIELDS:
            value = _optional_kpi_bool(value)
        verdict[field] = value
    verdict["verifier_status"] = _dataset_verifier_status(verdict).value
    return verdict


def _dataset_verifier_status(
    verdict: dict[str, Any],
) -> DeepSWEVerifierStatus:
    """Classify hidden verifier domain and infrastructure separately."""

    reward_state = verdict.get("reward_state")
    if reward_state == "missing":
        return DeepSWEVerifierStatus.REWARD_MISSING
    if reward_state == "mismatch":
        return DeepSWEVerifierStatus.REWARD_MISMATCH
    if reward_state in {"invalid", "unreadable"}:
        return DeepSWEVerifierStatus.REWARD_INVALID
    command_returncode = verdict.get("command_returncode")
    if command_returncode is None:
        command_returncode = verdict.get("verify_returncode")
    try:
        command_rc = int(float(command_returncode))
    except (TypeError, ValueError):
        return DeepSWEVerifierStatus.COMMAND_FAILED
    if command_rc == 124:
        return DeepSWEVerifierStatus.TIMEOUT
    if command_rc != 0:
        return DeepSWEVerifierStatus.COMMAND_FAILED
    reward_valid = verdict.get(
        "reward_valid",
        verdict.get("reward") in (0, 0.0, 1, 1.0),
    )
    if (
        reward_valid
        and verdict.get("reward") == 1.0
        and verdict.get("resolved") == 1.0
        and int(verdict.get("rc", 1)) == 0
    ):
        return DeepSWEVerifierStatus.RESOLVED
    if (
        reward_valid
        and verdict.get("reward") == 0.0
        and verdict.get("resolved") == 0.0
    ):
        return DeepSWEVerifierStatus.UNRESOLVED
    return DeepSWEVerifierStatus.COMMAND_FAILED


def _validate_stopped_verifier_reward(
    verdict: dict[str, Any],
    *,
    instance_log_dir: Path,
    test_inventory: RewardTestInventory | None,
) -> dict[str, Any]:
    """Re-read reward evidence after strict verifier quiescence.

    The runner deliberately does not open the reward path. The workflow trusts
    only this descriptor-relative, bounded read after Docker has proven the
    verifier stopped, and derives both reward KPIs from that exact token.

    ``test_inventory`` is the whitelist the private tests snapshot declares,
    or ``None`` when it declares none. The artifact must be the kind that
    snapshot's grader writes: a ``reward.json`` whose totals are the whitelist
    sizes when one is declared, a count-less ``reward.txt`` token otherwise.
    Any other artifact is a ``mismatch`` and carries no reward.
    """

    validated = dict(verdict)
    reward_read = read_reward_artifact(
        instance_log_dir,
        inventory=test_inventory,
    )
    reward_state = reward_read.state
    reward_error = reward_read.error
    if reward_state == "valid" and (
        (reward_read.counts is None) != (test_inventory is None)
    ):
        reward_state = "mismatch"
        reward_error = (
            "reward.json cannot be bound to a tests/config.json whitelist"
            if test_inventory is None
            else "reward.txt cannot grade a task whose tests declare a whitelist"
        )
    reward_valid = reward_state == "valid"
    validated["raw_reward"] = reward_read.token if reward_valid else ""
    validated["reward_pending"] = False
    validated["reward"] = reward_read.reward if reward_valid else None
    validated["resolved"] = reward_read.reward if reward_valid else None
    validated["reward_counts"] = (
        reward_read.counts.to_payload()
        if reward_valid and reward_read.counts is not None
        else None
    )
    # The reader's reasons name contract fields only, never file content.
    validated["reward_error"] = (
        None if reward_valid else (reward_error or reward_state)[:500]
    )
    validated["reward_state"] = reward_state
    validated["reward_valid"] = reward_valid
    validated["verifier_status"] = _dataset_verifier_status(validated).value
    return validated


def _dataset_verdict_status(verdict: dict[str, Any]) -> str:
    """Map the hidden verifier result to the authoritative task status."""
    verifier_status = _dataset_verifier_status(verdict)
    if verifier_status == DeepSWEVerifierStatus.RESOLVED:
        return "resolved"
    if verifier_status == DeepSWEVerifierStatus.UNRESOLVED:
        return "unresolved"
    return "verification_failed"


def _persist_candidate_blob(corpus: Corpus, candidate: DeepSWECandidate) -> str:
    validate_candidate(candidate)
    stored = str(corpus.put_artifact(
        candidate.patch,
        kind="diffs",
        name_hint="deepswe_candidate.diff",
    ))
    if stored != candidate.diff_ref:
        raise ValueError(
            "DeepSWE candidate artifact store returned unexpected digest"
        )
    return stored


def _candidate_measurement(
    corpus: Corpus,
    inst: DeepSWEInstance,
    candidate: DeepSWECandidate,
    *,
    checkpoint: dict[str, Any],
    run_dir: Path,
    pristine_replay: PristineReplayReport | None,
    verifier_patch: VerifierPatchReport | None,
) -> Measurement:
    _persist_candidate_blob(corpus, candidate)
    raw_ref = corpus.put_artifact(
        encode_candidate_envelope(
            candidate,
            task_id=inst.task_id,
            ext_id=inst.ext_id,
            checkpoint=checkpoint,
            pristine_replay=pristine_replay,
            verifier_patch=verifier_patch,
        ),
        kind="bench_raw",
        name_hint=f"deepswe_{inst.task_id}_candidate.json",
    )
    return Measurement(
        bench_id=f"deepswe/{inst.task_id}/candidate",
        commit=inst.base_commit_hash,
        config_hash=f"{run_dir.name}:{inst.ext_id}:candidate",
        values={
            "capture_ok": (
                0.0
                if candidate.status == CandidateCaptureStatus.FAILED else 1.0
            ),
            "patch_bytes": float(candidate.diff_bytes),
            "raw_diff_bytes": float(candidate.raw_diff_bytes),
        },
        raw_output_ref=raw_ref,
        execution=exec_ctx_now("deepswe_candidate", run_id=run_dir.name),
    )


def _persist_candidate_only_outcome(
    corpus: Corpus,
    inst: DeepSWEInstance,
    candidate: DeepSWECandidate,
    *,
    status: str,
    run_dir: Path,
    checkpoint_record: dict[str, Any],
    pristine_replay: PristineReplayReport | None = None,
) -> dict[str, Any]:
    """Publish salvaged bytes without claiming a hidden-oracle verdict."""

    checkpoint = _validate_lifecycle_record(
        dict(checkpoint_record), line_number=1,
    )
    if (
        checkpoint["task_id"] != inst.task_id
        or checkpoint["ext_id"] != inst.ext_id
        or checkpoint["status"] != status
    ):
        raise ValueError("candidate checkpoint identity/status does not match task")
    measurement = _candidate_measurement(
        corpus,
        inst,
        candidate,
        checkpoint=checkpoint,
        run_dir=run_dir,
        pristine_replay=pristine_replay,
        verifier_patch=None,
    )
    execution = exec_ctx_now("deepswe_candidate", run_id=run_dir.name)
    committed = corpus.commit(
        measurements=(measurement,),
        lifecycle_event_factory=lambda committed_version: LifecycleEvent(
            kind="abandoned",
            reason=(
                "DeepSWE candidate retained without hidden evaluation "
                f"status={status} capture={candidate.status.value}"
            ),
            affected_versions=(committed_version,),
            execution=execution,
        ),
    )
    return {
        "candidate_measurement_ref": str(committed.refs.measurement_refs[0]),
        "candidate_lifecycle_ref": str(
            committed.refs.lifecycle_event_refs[0]
        ),
        "instance_spec_version": committed.spec.version,
    }


def _persist_hidden_invocation_marker(
    corpus: Corpus,
    inst: DeepSWEInstance,
    candidate: DeepSWECandidate,
    *,
    run_dir: Path,
    checkpoint_record: dict[str, Any],
    pristine_replay: PristineReplayReport,
    tests_snapshot: DeepSWETestsTreeIdentity,
) -> dict[str, Any]:
    """Durably consume one-shot oracle authority before invoking it.

    The candidate Measurement and invocation marker share one Corpus commit.
    A crash at any later point can therefore distinguish "safe to invoke"
    from "oracle may already have run" without trusting lifecycle JSONL.
    """

    validate_candidate(candidate)
    if (
        pristine_replay.status != PristineReplayStatus.PASSED
        or pristine_replay.candidate_sha256 != candidate.sha256
        or pristine_replay.observed_sha256 != candidate.sha256
    ):
        raise ValueError(
            "hidden invocation marker requires an exact pristine replay"
        )
    checkpoint = _validate_lifecycle_record(
        dict(checkpoint_record),
        line_number=1,
    )
    if (
        checkpoint["task_id"] != inst.task_id
        or checkpoint["ext_id"] != inst.ext_id
        or checkpoint["status"] != "verification_incomplete"
    ):
        raise ValueError("hidden invocation checkpoint identity/status mismatch")
    snapshot_payload = tests_snapshot.to_payload()
    if (
        checkpoint.get("tests_snapshot") != snapshot_payload
        or checkpoint.get("tests_source_identity") != snapshot_payload
    ):
        raise ValueError(
            "hidden invocation checkpoint is not bound to tests snapshot"
        )
    candidate_measurement = _candidate_measurement(
        corpus,
        inst,
        candidate,
        checkpoint=checkpoint,
        run_dir=run_dir,
        pristine_replay=pristine_replay,
        verifier_patch=None,
    )
    raw_ref = corpus.put_artifact(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "deepswe_hidden_invocation",
                "state": "started",
                "task_id": inst.task_id,
                "ext_id": inst.ext_id,
                "candidate": candidate.to_payload(),
                "pristine_replay": pristine_replay.to_payload(),
                "tests_snapshot": snapshot_payload,
                "checkpoint": checkpoint,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8"),
        kind="bench_raw",
        name_hint=f"deepswe_{inst.task_id}_hidden_invocation.json",
    )
    execution = exec_ctx_now(
        "deepswe_hidden_invocation",
        run_id=run_dir.name,
    )
    invocation_measurement = Measurement(
        bench_id=f"deepswe/{inst.task_id}/hidden_invocation",
        commit=inst.base_commit_hash,
        config_hash=f"{run_dir.name}:{inst.ext_id}:hidden_invocation",
        values={
            "invocation_committed": 1.0,
            "patch_bytes": float(candidate.diff_bytes),
            "tests_bytes": float(tests_snapshot.total_bytes),
        },
        raw_output_ref=raw_ref,
        execution=execution,
    )
    committed = corpus.commit(
        measurements=(candidate_measurement, invocation_measurement),
        lifecycle_event_factory=lambda committed_version: LifecycleEvent(
            kind="abandoned",
            reason=(
                "DeepSWE hidden verifier invocation authority consumed; "
                "final verdict pending"
            ),
            affected_versions=(committed_version,),
            execution=execution,
        ),
    )
    return {
        "candidate_measurement_ref": str(committed.refs.measurement_refs[0]),
        "hidden_invocation_measurement_ref": str(
            committed.refs.measurement_refs[1]
        ),
        "hidden_invocation_lifecycle_ref": str(
            committed.refs.lifecycle_event_refs[0]
        ),
        "instance_spec_version": committed.spec.version,
    }


def _persist_dataset_outcome(
    corpus: Corpus,
    inst: DeepSWEInstance,
    verdict: dict[str, Any],
    *,
    status: str,
    run_dir: Path,
    checkpoint_record: dict[str, Any],
    candidate: DeepSWECandidate,
    pristine_replay: PristineReplayReport,
    verifier_patch: VerifierPatchReport,
) -> dict[str, Any]:
    """Attach the hidden verifier oracle result to the task lineage.

    The raw verifier payload, its ``Measurement``, and the terminal lifecycle
    share one Spec visibility point. ``unresolved`` and malformed / failed
    verifier results deliberately publish ``abandoned`` so an earlier
    cascade-level ``landed`` event cannot masquerade as benchmark success.
    """

    validate_candidate(candidate)
    candidate_bound = (
        candidate.status != CandidateCaptureStatus.FAILED
        and pristine_replay.status == PristineReplayStatus.PASSED
        and pristine_replay.candidate_sha256 == candidate.sha256
        and pristine_replay.observed_sha256 == candidate.sha256
        and verifier_patch.status == VerifierPatchStatus.MATCHED
        and verifier_patch.candidate_sha256 == candidate.sha256
        and verifier_patch.observed_sha256 == candidate.sha256
        and verifier_patch.observed_bytes == candidate.diff_bytes
    )
    if status in {"resolved", "unresolved"} and not candidate_bound:
        raise ValueError(
            "hidden verifier outcome is not bound to an exact pristine candidate"
        )

    checkpoint = _validate_lifecycle_record(
        dict(checkpoint_record),
        line_number=1,
    )
    if (
        checkpoint["task_id"] != inst.task_id
        or checkpoint["ext_id"] != inst.ext_id
        or checkpoint["status"] != status
    ):
        raise ValueError(
            "hidden verifier checkpoint identity/status does not match task"
        )
    raw_ref = corpus.put_artifact(
        json.dumps(
            {
                "schema_version": 3,
                "kind": "deepswe_hidden_verifier",
                "task_id": inst.task_id,
                "ext_id": inst.ext_id,
                "status": status,
                "verifier": verdict,
                "candidate": candidate.to_payload(),
                "pristine_replay": pristine_replay.to_payload(),
                "verifier_patch": verifier_patch.to_payload(),
                "tests_snapshot": checkpoint.get("tests_snapshot"),
                # This is the resume authority if the process dies after the
                # Corpus commit but before lifecycle.jsonl publication.
                "checkpoint": checkpoint,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ).encode("utf-8"),
        kind="bench_raw",
        name_hint=f"deepswe_{inst.task_id}_hidden_verifier.json",
    )
    reward_valid = verdict.get(
        "reward_valid",
        verdict.get("reward") in (0, 0.0, 1, 1.0),
    )
    values: dict[str, float] = {
        "reward_valid": 1.0 if reward_valid else 0.0,
    }
    for key in (
        "rc",
        "reward",
        "resolved",
        "verify_returncode",
        *_VERIFIER_OBSERVABILITY_FIELDS,
    ):
        try:
            values[key] = float(verdict[key])
        except (KeyError, TypeError, ValueError):
            # The raw artifact preserves absent/malformed values exactly.
            pass
    reward_counts = verdict.get("reward_counts")
    if reward_counts is not None:
        # Present only beside a valid v1.1 reward; the same exact payload is
        # in the raw artifact and in the checkpoint committed with it.
        values.update(
            {
                name: float(count)
                for name, count in RewardCounts.from_payload(
                    reward_counts
                ).to_payload().items()
            }
        )

    execution = exec_ctx_now(
        "deepswe_hidden_verifier",
        run_id=run_dir.name,
    )
    measurement = Measurement(
        bench_id=f"deepswe/{inst.task_id}/hidden_verifier",
        commit=inst.base_commit_hash,
        config_hash=f"{run_dir.name}:{inst.ext_id}",
        values=values,
        raw_output_ref=raw_ref,
        execution=execution,
    )
    candidate_measurement = _candidate_measurement(
        corpus,
        inst,
        candidate,
        checkpoint=checkpoint,
        run_dir=run_dir,
        pristine_replay=pristine_replay,
        verifier_patch=verifier_patch,
    )
    diagnostic = checkpoint.get("diagnostic_only") is True
    if diagnostic:
        values.update({"diagnostic_only": 1.0, "rankable": 0.0, "promotable": 0.0})
        # Measurement freezes its values at construction; persist the same
        # restriction beside the real oracle reward, not only in a report.
        measurement = dataclasses.replace(measurement, values=values)
    lifecycle_kind = (
        "diagnostic_completed" if diagnostic and status == "resolved"
        else "landed" if status == "resolved" else "abandoned"
    )
    committed = corpus.commit(
        # Candidate evidence and the one-shot hidden oracle become visible in
        # one Spec version. A crash cannot leave a resumable reward detached
        # from the exact bytes the verifier graded.
        measurements=(candidate_measurement, measurement),
        lifecycle_event_factory=lambda committed_version: LifecycleEvent(
            kind=lifecycle_kind,
            reason=(
                "authoritative DeepSWE hidden verifier "
                f"status={status} reward={verdict.get('reward')} "
                f"resolved={verdict.get('resolved')}"
            ),
            affected_versions=(committed_version,),
            execution=execution,
        ),
    )
    candidate_measurement_ref = committed.refs.measurement_refs[0]
    measurement_ref = committed.refs.measurement_refs[1]
    lifecycle_ref = committed.refs.lifecycle_event_refs[0]
    return {
        "hidden_verifier_measurement_ref": str(measurement_ref),
        "candidate_measurement_ref": str(candidate_measurement_ref),
        "hidden_verifier_lifecycle_ref": str(lifecycle_ref),
        "instance_spec_version": committed.spec.version,
    }


class _CostShim:
    """Minimal duck-typed ``usage`` for ``CostAccumulator.record`` — carries
    a precomputed cost (and optional tokens) with no backend object."""

    __slots__ = (
        "cost_usd", "input_tokens", "output_tokens",
        "cache_creation_tokens", "cache_read_tokens",
    )

    def __init__(
        self,
        cost_usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        self.cost_usd = cost_usd
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_tokens = cache_creation_tokens
        self.cache_read_tokens = cache_read_tokens


def _architect_operation_id(task_id: str) -> str:
    """Return a path-safe, stable saga operation id for one task."""

    identity = canonical_fingerprint({
        "workflow": "deepswe",
        "task_id": task_id,
    })
    return f"architect-{identity.removeprefix('sha256:')}"


def _architect_run_cost(
    parent_run_dir: Path | str,
    operation_id: str,
) -> tuple[dict[str, Any] | None, _outcome.WorkflowOutcome | None, Path | None]:
    """Return cost data and identity for the saga-bound Architect child.

    ``run_architect`` keeps its ``CostAccumulator`` private and returns only
    an rc, persisting the rollup to its own run ``summary.json``. The deepswe
    worker reads it back so per-task spend feeds the sweep accumulator (and
    the ``--budget-usd`` halt) instead of recording $0. The parent saga is the
    sole child identity authority; unrelated, concurrent, or partial run
    folders are never candidates.
    """

    try:
        outcome = _outcome.child_outcome(parent_run_dir, operation_id)
        run_dir = _outcome.child_run_dir(parent_run_dir, operation_id)
    except ValueError:
        return None, None, None
    if outcome.workflow != "architect":
        return None, None, None
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, outcome, run_dir
    cost_source = summary
    if not isinstance(cost_source.get("costs"), dict):
        partial = summary.get("partial_summary")
        if isinstance(partial, dict):
            cost_source = partial
    costs = cost_source.get("costs")
    return (
        costs if isinstance(costs, dict) else None,
        outcome,
        run_dir,
    )


def _architect_run_summary(run_dir: Path | None) -> dict[str, Any] | None:
    """Return the Architect's semantic summary, including exception partials."""

    if run_dir is None:
        return None
    try:
        raw = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    partial = raw.get("partial_summary")
    if isinstance(partial, dict) and not isinstance(raw.get("tier_results"), list):
        return partial
    return raw


_TIER_SETTLEMENT_SCHEMA = "crucible.tier-mutation-settlement.v1"


def _validate_tier_mutation_settlement(
    value: Any,
    *,
    inst: DeepSWEInstance,
    host_workspace: Path,
    tier_id: str,
    operation_id: str,
    tier_run_dir: Path | None,
    max_patch_bytes: int,
) -> dict[str, Any]:
    """Validate one positive, content-bound Tier settlement attestation."""

    if not isinstance(value, dict):
        raise ValueError("Tier mutation settlement attestation is missing")
    attestation = dict(value)
    digest = attestation.pop("attestation_sha256", None)
    if (
        digest != canonical_fingerprint(attestation)
        or attestation.get("schema_version") != _TIER_SETTLEMENT_SCHEMA
        or attestation.get("settled") is not True
        or attestation.get("tier_id") != tier_id
        or attestation.get("operation_id") != operation_id
        or attestation.get("task_id") != inst.task_id
        or attestation.get("ext_id") != inst.ext_id
        or Path(str(attestation.get("workspace", ""))).resolve()
        != host_workspace.resolve()
        or attestation.get("base_commit") != inst.base_commit_hash
        or attestation.get("max_patch_bytes") != max_patch_bytes
        or attestation.get("remaining_journal_manifests") != []
        or attestation.get("registry_clear") is not True
    ):
        raise ValueError("Tier mutation settlement identity/proof is invalid")
    recovery = attestation.get("journal_recovery")
    if not isinstance(recovery, dict) or recovery.get("failures") != []:
        raise ValueError("Tier mutation settlement recovery is incomplete")
    patch_sha = attestation.get("workspace_patch_sha256")
    patch_bytes = attestation.get("workspace_patch_bytes")
    files_modified = attestation.get("workspace_files_modified")
    if (
        not isinstance(patch_sha, str)
        or len(patch_sha) != 64
        or any(char not in "0123456789abcdef" for char in patch_sha)
        or isinstance(patch_bytes, bool)
        or not isinstance(patch_bytes, int)
        or patch_bytes < 0
        or not isinstance(files_modified, list)
        or not all(isinstance(path, str) for path in files_modified)
        or not isinstance(attestation.get("spec_version"), int)
        or not isinstance(attestation.get("spec_hash"), str)
    ):
        raise ValueError("Tier mutation settlement workspace binding is invalid")
    artifacts = attestation.get("tier_run_artifacts")
    if tier_run_dir is None:
        if attestation.get("tier_run_id") is not None or artifacts != {}:
            raise ValueError("final mutation settlement has unexpected child run")
    else:
        run_dir = tier_run_dir.resolve()
        if (
            attestation.get("tier_run_id") != run_dir.name
            or not isinstance(artifacts, dict)
        ):
            raise ValueError("Tier mutation settlement child identity is invalid")
        expected_artifacts = {
            name: file_digest(run_dir / name)
            for name in ("summary.json", "result.json", "outcome.json")
        }
        if artifacts != expected_artifacts:
            raise ValueError("Tier mutation settlement child artifacts changed")
    return value


def _require_architect_mutation_settlement(
    architect_run_dir: Path | None,
    *,
    inst: DeepSWEInstance,
    host_workspace: Path,
    max_patch_bytes: int,
) -> dict[str, Any]:
    """Require complete settlement proofs from the saga-bound Architect run."""

    if architect_run_dir is None:
        raise ValueError("saga-bound Architect run directory is unavailable")
    try:
        raw = json.loads(
            (architect_run_dir / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError("Architect settlement summary is unreadable") from exc
    if not isinstance(raw, dict):
        raise ValueError("Architect settlement summary is not an object")
    summary = raw
    if not isinstance(summary.get("tier_results"), list):
        partial = summary.get("partial_summary")
        if isinstance(partial, dict):
            summary = partial
    tier_results = summary.get("tier_results")
    tiers_executed = summary.get("tiers_executed")
    if (
        not isinstance(tier_results, list)
        or isinstance(tiers_executed, bool)
        or not isinstance(tiers_executed, int)
        or tiers_executed < 0
        or tiers_executed != len(tier_results)
    ):
        raise ValueError("Architect tier settlement inventory is incomplete")
    # The namespace is an authoritative contract advertisement whether it is
    # attached to the terminal exception envelope or to its selected partial
    # summary.  Never let a malformed outer new-format record downgrade into
    # otherwise-valid legacy fields nested in ``partial_summary``.
    extension_present = (
        "checkpoint_extensions" in raw
        or "checkpoint_extensions" in summary
    )
    if extension_present:
        extensions = summary.get("checkpoint_extensions")
        if not isinstance(extensions, dict):
            raise ValueError(
                "Architect checkpoint extension namespace is invalid"
            )
        extension = extensions.get(DEEPSWE_CHECKPOINT_EXTENSION_KEY)
        if (
            not isinstance(extension, dict)
            or extension.get("schema_version")
            != DEEPSWE_CHECKPOINT_EXTENSION_SCHEMA
        ):
            raise ValueError(
                "Architect summary lacks the required DeepSWE checkpoint proof"
            )
        aggregate = extension.get("mutation_settlement")
        aggregate_count = (
            extension.get("mutation_settlement", {}).get("tiers_executed")
            if isinstance(extension.get("mutation_settlement"), dict)
            else None
        )
        if (
            not isinstance(aggregate, dict)
            or aggregate.get("schema_version")
            != DEEPSWE_MUTATION_SETTLEMENT_SCHEMA
            or isinstance(aggregate_count, bool)
            or not isinstance(aggregate_count, int)
            or aggregate_count != tiers_executed
            or not isinstance(aggregate.get("tier_attestations"), list)
            or len(aggregate["tier_attestations"]) != tiers_executed
        ):
            raise ValueError(
                "DeepSWE Architect checkpoint settlement is incomplete"
            )
        tier_attestations = aggregate["tier_attestations"]
        final_attestation = aggregate.get("final_attestation")
    else:
        # Backward-read only. Once Architect advertises the generic extension
        # namespace, malformed or missing DeepSWE data must never downgrade to
        # legacy scattered fields.
        tier_attestations = [
            result.get("mutation_journal_settlement")
            if isinstance(result, dict) else None
            for result in tier_results
        ]
        final_attestation = summary.get("mutation_journal_settlement")

    validated: list[dict[str, Any]] = []
    for index, (result, attestation) in enumerate(
        zip(tier_results, tier_attestations, strict=True)
    ):
        if not isinstance(result, dict):
            raise ValueError("Architect tier settlement entry is invalid")
        tier_id = result.get("tier_id")
        operation_id = result.get("operation_id")
        run_dir_raw = result.get("run_dir")
        if (
            not isinstance(tier_id, str)
            or operation_id != f"tier-{index}"
            or not isinstance(run_dir_raw, str)
        ):
            raise ValueError("Architect tier settlement provenance is invalid")
        validated.append(_validate_tier_mutation_settlement(
            attestation,
            inst=inst,
            host_workspace=host_workspace,
            tier_id=tier_id,
            operation_id=operation_id,
            tier_run_dir=Path(run_dir_raw),
            max_patch_bytes=max_patch_bytes,
        ))
    final = _validate_tier_mutation_settlement(
        final_attestation,
        inst=inst,
        host_workspace=host_workspace,
        tier_id="__architect_final__",
        operation_id="architect-final",
        tier_run_dir=None,
        max_patch_bytes=max_patch_bytes,
    )
    if not validated and (
        final.get("workspace_patch_sha256") != _EMPTY_PATCH_SHA256
        or final.get("workspace_patch_bytes") != 0
        or final.get("workspace_files_modified") != []
    ):
        raise ValueError(
            "zero-tier Architect settlement does not prove a clean workspace"
        )
    return {
        "schema_version": "crucible.deepswe.mutation-settlement.v1",
        "tiers_executed": len(validated),
        "tier_attestations": validated,
        "final_attestation": final,
    }


def _establish_capture_mutation_settlement(
    corpus: Corpus,
    architect_run_dir: Path | None,
    *,
    inst: DeepSWEInstance,
    host_workspace: Path,
    max_patch_bytes: int,
) -> dict[str, Any]:
    """Re-audit journals after quiescence and bind Architect to capture."""

    architect_settlement = _require_architect_mutation_settlement(
        architect_run_dir,
        inst=inst,
        host_workspace=host_workspace,
        max_patch_bytes=max_patch_bytes,
    )
    from crucible.workflows.tier import (  # noqa: PLC0415
        attest_mutation_journal_settlement,
    )

    capture_attestation = attest_mutation_journal_settlement(
        corpus=corpus,
        cwd=host_workspace,
        tier_id="__deepswe_capture__",
        operation_id="deepswe-capture",
        task_id=inst.task_id,
        ext_id=inst.ext_id,
        expected_base_commit=inst.base_commit_hash,
        max_patch_bytes=max_patch_bytes,
    )
    capture_attestation = _validate_tier_mutation_settlement(
        capture_attestation,
        inst=inst,
        host_workspace=host_workspace,
        tier_id="__deepswe_capture__",
        operation_id="deepswe-capture",
        tier_run_dir=None,
        max_patch_bytes=max_patch_bytes,
    )
    architect_final = architect_settlement["final_attestation"]
    if (
        architect_final["workspace_patch_sha256"]
        != capture_attestation["workspace_patch_sha256"]
        or architect_final["workspace_patch_bytes"]
        != capture_attestation["workspace_patch_bytes"]
        or architect_final["workspace_files_modified"]
        != capture_attestation["workspace_files_modified"]
        or architect_final["spec_version"]
        != capture_attestation["spec_version"]
        or architect_final["spec_hash"] != capture_attestation["spec_hash"]
    ):
        raise ValueError("workspace changed after Architect journal settlement")
    return {
        **architect_settlement,
        "capture_attestation": capture_attestation,
    }


_PRICING_USAGE_FIELDS = {
    "input_tokens", "output_tokens", "cache_creation_tokens",
    "cache_read_tokens", "cost_usd",
}


def _pricing_summary_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Architect pricing {field} must be a non-negative int")
    return value


def _pricing_summary_cost(value: Any, *, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"Architect pricing {field} must be numeric")
    amount = float(value)
    if not math.isfinite(amount) or amount < 0.0:
        raise ValueError(
            f"Architect pricing {field} must be finite and non-negative"
        )
    return amount


def _pricing_evidence_from_summary(
    value: Any, *, field: str,
) -> UsagePricingEvidence:
    if (
        type(value) is not dict
        or set(value) != {"segments", "complete"}
        or type(value["segments"]) is not list
        or type(value["complete"]) is not bool
    ):
        raise ValueError(f"Architect pricing {field} is invalid")
    segments: list[UsagePricingSegment] = []
    for index, raw in enumerate(value["segments"]):
        name = f"{field}.segments[{index}]"
        if type(raw) is not dict or set(raw) != _PRICING_USAGE_FIELDS:
            raise ValueError(f"Architect pricing {name} is invalid")
        segments.append(UsagePricingSegment(
            input_tokens=_pricing_summary_int(
                raw["input_tokens"], field=f"{name}.input_tokens",
            ),
            output_tokens=_pricing_summary_int(
                raw["output_tokens"], field=f"{name}.output_tokens",
            ),
            cache_creation_tokens=_pricing_summary_int(
                raw["cache_creation_tokens"],
                field=f"{name}.cache_creation_tokens",
            ),
            cache_read_tokens=_pricing_summary_int(
                raw["cache_read_tokens"],
                field=f"{name}.cache_read_tokens",
            ),
            cost_usd=_pricing_summary_cost(
                raw["cost_usd"], field=f"{name}.cost_usd",
            ),
        ))
    return UsagePricingEvidence(
        segments=tuple(segments), complete=value["complete"],
    )


def _architect_pricing_calls(
    arch_costs: dict[str, Any],
) -> tuple[
    list[tuple[str, str, str | None, _CostShim, UsagePricingEvidence]],
    bool,
]:
    """Decode the physical-call ledger before mutating the parent total."""

    if arch_costs.get("pricing_evidence_schema") != PRICING_EVIDENCE_SCHEMA:
        raise ValueError("Architect pricing evidence schema is invalid")
    raw_calls = arch_costs.get("pricing_evidence_by_call")
    if type(raw_calls) is not list:
        raise ValueError("Architect pricing_evidence_by_call is invalid")
    calls: list[
        tuple[str, str, str | None, _CostShim, UsagePricingEvidence]
    ] = []
    for index, raw in enumerate(raw_calls):
        name = f"pricing_evidence_by_call[{index}]"
        if type(raw) is not dict or set(raw) != {
            "role", "backend", "model", "usage", "pricing_evidence",
        }:
            raise ValueError(f"Architect pricing {name} is invalid")
        role, backend, model = raw["role"], raw["backend"], raw["model"]
        if (
            type(role) is not str or not role
            or type(backend) is not str or not backend
            or (
                model is not None
                and (type(model) is not str or not model)
            )
        ):
            raise ValueError(f"Architect pricing {name} identity is invalid")
        usage_raw = raw["usage"]
        if (
            type(usage_raw) is not dict
            or set(usage_raw) != _PRICING_USAGE_FIELDS
        ):
            raise ValueError(f"Architect pricing {name}.usage is invalid")
        usage = _CostShim(
            _pricing_summary_cost(
                usage_raw["cost_usd"], field=f"{name}.usage.cost_usd",
            ),
            _pricing_summary_int(
                usage_raw["input_tokens"], field=f"{name}.usage.input_tokens",
            ),
            _pricing_summary_int(
                usage_raw["output_tokens"],
                field=f"{name}.usage.output_tokens",
            ),
            _pricing_summary_int(
                usage_raw["cache_creation_tokens"],
                field=f"{name}.usage.cache_creation_tokens",
            ),
            _pricing_summary_int(
                usage_raw["cache_read_tokens"],
                field=f"{name}.usage.cache_read_tokens",
            ),
        )
        if (
            usage.cache_creation_tokens + usage.cache_read_tokens
            > usage.input_tokens
        ):
            raise ValueError(f"Architect pricing {name} cache usage is invalid")
        evidence = _pricing_evidence_from_summary(
            raw["pricing_evidence"], field=f"{name}.pricing_evidence",
        )
        if evidence.complete and not evidence.reconciles(usage):
            raise ValueError(
                f"Architect pricing {name} complete evidence does not reconcile"
            )
        calls.append((role, backend, model, usage, evidence))

    totals = (
        ("input_tokens", "total_input_tokens"),
        ("output_tokens", "total_output_tokens"),
        ("cache_creation_tokens", "total_cache_creation_tokens"),
        ("cache_read_tokens", "total_cache_read_tokens"),
    )
    if any(
        sum(getattr(call[3], usage_field) for call in calls)
        != _pricing_summary_int(
            arch_costs.get(summary_field), field=summary_field,
        )
        for usage_field, summary_field in totals
    ) or not math.isclose(
        math.fsum(call[3].cost_usd for call in calls),
        _pricing_summary_cost(arch_costs.get("total_usd"), field="total_usd"),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Architect pricing call ledger does not reconcile aggregate usage"
        )
    aggregate = _pricing_evidence_from_summary(
        arch_costs.get("pricing_evidence"), field="pricing_evidence",
    )
    segments = tuple(
        segment for call in calls for segment in call[4].segments
    )
    if aggregate.segments != segments:
        raise ValueError(
            "Architect aggregate pricing segments do not match call order"
        )
    calls_complete = all(
        evidence.complete and evidence.reconciles(usage)
        for _, _, _, usage, evidence in calls
    )
    if aggregate.complete and not calls_complete:
        raise ValueError(
            "Architect aggregate pricing completeness does not reconcile calls"
        )
    pricing_rollup_incomplete = arch_costs.get(
        "pricing_rollup_incomplete"
    )
    if (
        type(pricing_rollup_incomplete) is not bool
        or pricing_rollup_incomplete is aggregate.complete
        or type(arch_costs.get("rollup_incomplete")) is not bool
        or (
            pricing_rollup_incomplete
            and arch_costs["rollup_incomplete"] is not True
        )
    ):
        raise ValueError(
            "Architect pricing completeness flags do not reconcile evidence"
        )

    expected_by_role: dict[str, dict[str, Any]] = {}
    expected_by_backend: dict[str, list[float]] = {}
    expected_by_model: dict[str, list[float]] = {}
    for role, backend, model, usage, _evidence in calls:
        slot = expected_by_role.setdefault(role, {
            "costs": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "calls": 0,
            "backends": set(),
            "models": set(),
        })
        slot["costs"].append(usage.cost_usd)
        slot["input_tokens"] += usage.input_tokens
        slot["output_tokens"] += usage.output_tokens
        slot["cache_creation_tokens"] += usage.cache_creation_tokens
        slot["cache_read_tokens"] += usage.cache_read_tokens
        slot["calls"] += 1
        slot["backends"].add(backend)
        if model is not None:
            slot["models"].add(model)
        expected_by_backend.setdefault(backend, []).append(usage.cost_usd)
        if model is not None:
            expected_by_model.setdefault(model, []).append(usage.cost_usd)

    raw_by_role = arch_costs.get("by_role")
    if type(raw_by_role) is not dict or set(raw_by_role) != set(expected_by_role):
        raise ValueError("Architect pricing by_role projection is invalid")
    for role, expected in expected_by_role.items():
        raw_slot = raw_by_role[role]
        cache_present = bool(
            expected["cache_creation_tokens"]
            or expected["cache_read_tokens"]
        )
        expected_fields = {
            "cost_usd", "input_tokens", "output_tokens", "calls",
            "backends", "models",
        }
        if cache_present:
            expected_fields.update({
                "cache_creation_tokens", "cache_read_tokens",
            })
        if type(raw_slot) is not dict or set(raw_slot) != expected_fields:
            raise ValueError(
                f"Architect pricing by_role[{role!r}] projection is invalid"
            )
        for field in ("input_tokens", "output_tokens", "calls"):
            if _pricing_summary_int(
                raw_slot[field], field=f"by_role[{role!r}].{field}",
            ) != expected[field]:
                raise ValueError(
                    f"Architect pricing by_role[{role!r}] projection "
                    "does not reconcile"
                )
        if cache_present:
            for field in ("cache_creation_tokens", "cache_read_tokens"):
                if _pricing_summary_int(
                    raw_slot[field], field=f"by_role[{role!r}].{field}",
                ) != expected[field]:
                    raise ValueError(
                        f"Architect pricing by_role[{role!r}] projection "
                        "does not reconcile"
                    )
        if not math.isclose(
            _pricing_summary_cost(
                raw_slot["cost_usd"],
                field=f"by_role[{role!r}].cost_usd",
            ),
            math.fsum(expected["costs"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Architect pricing by_role[{role!r}] projection "
                "does not reconcile"
            )
        if (
            raw_slot["backends"] != sorted(expected["backends"])
            or raw_slot["models"] != sorted(expected["models"])
        ):
            raise ValueError(
                f"Architect pricing by_role[{role!r}] routes do not reconcile"
            )

    for field, expected in (
        ("by_backend", expected_by_backend),
        ("by_model", expected_by_model),
    ):
        raw_projection = arch_costs.get(field)
        if (
            type(raw_projection) is not dict
            or set(raw_projection) != set(expected)
        ):
            raise ValueError(f"Architect pricing {field} projection is invalid")
        for key, costs in expected.items():
            if not math.isclose(
                _pricing_summary_cost(
                    raw_projection[key], field=f"{field}[{key!r}]",
                ),
                math.fsum(costs),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Architect pricing {field} projection does not reconcile"
                )
    # Any incomplete child aggregate makes the parent ledger incomplete. This
    # includes both an unavailable corpus ref with exact retained calls and an
    # individual provider call whose physical pricing evidence is incomplete.
    return calls, not aggregate.complete


def _cost_summary_has_pricing_extension(costs: dict[str, Any]) -> bool:
    return any(field in costs for field in (
        "pricing_evidence_schema",
        "total_cache_creation_tokens", "total_cache_read_tokens",
        "pricing_evidence", "pricing_evidence_by_call",
    ))


def _rollup_architect_cost(
    cost: CostAccumulator, arch_costs: dict[str, Any], *, task_id: str
) -> float:
    """Replay an Architect run's physical calls or its legacy role totals."""

    if _cost_summary_has_pricing_extension(arch_costs):
        calls, force_incomplete = _architect_pricing_calls(arch_costs)
        total = _pricing_summary_cost(
            arch_costs.get("total_usd"), field="total_usd",
        )
        for role, backend, model, usage, evidence in calls:
            cost.record(
                role=role,
                usage=usage,
                backend_name=backend,
                model=model,
                pricing_evidence=evidence,
            )
        if force_incomplete:
            cost.note_unloaded_ref()
        return total

    total = _pricing_summary_cost(
        arch_costs.get("total_usd"), field="total_usd",
    )

    by_role = arch_costs.get("by_role") or {}
    attributed = 0.0
    attributed_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
    }
    if by_role:
        for role, slot in by_role.items():
            backends = slot.get("backends") or ["aggregate"]
            models = slot.get("models") or [None]
            usage = _CostShim(
                float(slot.get("cost_usd", 0.0) or 0.0),
                int(slot.get("input_tokens", 0) or 0),
                int(slot.get("output_tokens", 0) or 0),
                int(slot.get("cache_creation_tokens", 0) or 0),
                int(slot.get("cache_read_tokens", 0) or 0),
            )
            cost.record(
                role=role,
                usage=usage,
                backend_name=backends[0],
                model=models[0],
            )
            attributed += usage.cost_usd
            for field in attributed_usage:
                attributed_usage[field] += getattr(usage, field)
    else:
        cost.record(
            role=f"deepswe:{task_id}",
            usage=_CostShim(
                total,
                int(arch_costs.get("total_input_tokens", 0) or 0),
                int(arch_costs.get("total_output_tokens", 0) or 0),
                int(arch_costs.get("total_cache_creation_tokens", 0) or 0),
                int(arch_costs.get("total_cache_read_tokens", 0) or 0),
            ),
            backend_name="aggregate",
            model=None,
        )
        return total

    remainder = total - attributed
    remainder_usage = {
        field: (
            int(arch_costs.get(f"total_{field}", 0) or 0) - value
            if f"total_{field}" in arch_costs else 0
        )
        for field, value in attributed_usage.items()
    }
    if remainder < -1e-6 or any(value < 0 for value in remainder_usage.values()):
        raise ValueError("Architect aggregate usage is below role attribution")
    remainder = max(0.0, remainder)
    if remainder > 1e-9 or any(remainder_usage.values()):
        cost.record(
            role=f"deepswe:{task_id}:unattributed",
            usage=_CostShim(
                remainder,
                remainder_usage["input_tokens"],
                remainder_usage["output_tokens"],
                remainder_usage["cache_creation_tokens"],
                remainder_usage["cache_read_tokens"],
            ),
            backend_name="architect",
            model=None,
        )
    return total


def _prepare_task_corpus(
    corpus: Corpus,
    starter: Spec,
    *,
    task_id: str,
) -> bool:
    """Initialize a task Corpus or validate its stable lineage.

    Returns whether a prior lineage exists. Task ids are stable directory
    keys, so silently accepting changed repository/instruction content would
    reuse an unrelated plan. Fail closed and require a new spec root.
    """
    latest = corpus.latest_spec()
    if latest is None:
        corpus.put_spec(starter)
        return False
    expected = starter.project_context
    actual = latest.project_context
    if not (
        actual.project == expected.project
        and actual.repo_url == expected.repo_url
        and actual.base_commit == expected.base_commit
        and latest.intent == starter.intent
        and latest.plan_policy == starter.plan_policy
    ):
        raise ValueError(
            f"DeepSWE task {task_id!r} conflicts with existing Corpus "
            "lineage; use a new spec root for changed task content or "
            "planning-envelope policy"
        )
    return True


def _finish_storage_quota(
    quota: DirectoryGrowthQuota,
    summary: dict[str, Any],
    *,
    kill_if_already_exceeded=None,
) -> None:
    """Stop a quota monitor and retain its peak even when it raises."""

    if quota.exceeded and kill_if_already_exceeded is not None:
        # The monitor may trip during a container transition when no active id
        # is available to its one-shot callback. Kill the newly active sandbox
        # before surfacing the authoritative quota error.
        kill_if_already_exceeded()
    try:
        quota.__exit__(None, None, None)
    finally:
        summary["storage_peak_growth_bytes"] = quota.peak_growth_bytes


def _append_secondary_failure(
    summary: dict[str, Any],
    *,
    phase: str,
    kind: str,
    error: str,
) -> None:
    """Retain a later failure without replacing the initiating cause."""

    entry = {
        "phase": phase,
        "kind": kind,
        "error": error[:2000],
    }
    failures = summary.setdefault("secondary_failures", [])
    if entry not in failures:
        failures.append(entry)


def _preserve_architect_primary(
    summary: dict[str, Any],
    *,
    phase: str,
    kind: str,
    error: str,
) -> bool:
    """Record post-Architect infrastructure fallout as secondary.

    This applies only before hidden-verifier authority is consumed. Once the
    oracle starts, its one-shot terminal state remains the control-flow
    authority even when an earlier Architect attempt also failed.
    """

    architect_phase = summary.get("architect_failure_phase")
    architect_kind = summary.get("architect_failure_kind")
    rc = summary.get("rc")
    if (
        not isinstance(architect_phase, str)
        or not architect_phase
        or not isinstance(architect_kind, str)
        or not architect_kind
        or isinstance(rc, bool)
        or not isinstance(rc, int)
        or rc == 0
        or summary.get("hidden_invocation_measurement_ref") is not None
    ):
        return False
    summary["status"] = "candidate_failed"
    summary["failure_phase"] = architect_phase
    summary["failure_kind"] = architect_kind
    _append_secondary_failure(
        summary,
        phase=phase,
        kind=kind,
        error=error,
    )
    return True


def _run_task_cascade(
    inst: DeepSWEInstance,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    cost: CostAccumulator,
    child_budget_usd: float | None = None,
) -> dict[str, Any]:
    """Run one DeepSWE task end-to-end with isolated execution capabilities.
    Returns a per-task summary dict for the run-level summary.json.

    Container lifecycle:

    1. AUTHORING container (``/app`` + ``/logs``, no ``/tests``, no
       ``/solution``) — the cascade body (Implementer, Defender,
       per-round ``dev_verify``) runs against this. Hidden tests are
       structurally absent.
    2. BASELINE container — runs the same public verification command against
       an independently populated clean checkout so pre-existing failures are
       distinguished from candidate regressions.
    3. SOURCE RENDER container — after authoring is stopped, renders the
       candidate with image Git from the authoring checkout, with no hidden
       mounts and a streaming byte cap.
    4. VERIFIER replay checkout — independently populated, receives those
       exact bytes, and is rendered again by a fresh no-tests image container.
       Only byte identity between the two image renderings issues replay proof.
    5. GRADING checkout — independently populated from the same immutable
       image and base commit, but never receives the candidate on the host.
    6. VERIFIER container (``/app`` + ``/logs`` + private ``/tests:ro``
       snapshot) — starts against the fresh grading checkout so the dataset
       grader applies ``model.patch`` exactly once. Run exactly once for the
       dataset reward and prove its retained patch equals the candidate.

    Failures inside this function are caught + logged + reported
    in the returned dict; they do NOT propagate to the worker loop
    (so one task's failure can't block the rest of the sweep).
    """
    run_dir = Path(run_dir).expanduser().resolve()
    # Lazy import — run_architect pulls a lot of optional surface.
    from crucible.workflows.architect import run_architect  # noqa: PLC0415

    instance_spec_dir = (
        run_dir.parent.parent / "instances" / inst.task_id / "corpus"
    )
    task_run_dir = run_dir / "instances" / inst.task_id
    task_log_root = task_run_dir / "logs"
    authoring_log_dir = task_log_root / "authoring"
    baseline_log_dir = task_log_root / "baseline"
    # This directory is never mounted into a model-accessible container.
    # Stale authoring output therefore cannot preseed reward.txt/model.patch.
    instance_log_dir = task_log_root / "hidden-verifier"
    private_root = task_run_dir / "private"
    workspace_root = run_dir / "workspaces" / inst.task_id
    authoring_log_dir.mkdir(parents=True, exist_ok=True)
    baseline_log_dir.mkdir(parents=True, exist_ok=True)
    instance_log_dir.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    instance_spec_dir.mkdir(parents=True, exist_ok=True)

    assigned_task_budget = (
        args.budget_usd_per_instance
        if child_budget_usd is None else child_budget_usd
    )
    planning_envelope = _deepswe_planning_envelope(
        assigned_task_budget, planner_count=args.planner_count,
    )
    summary: dict[str, Any] = {
        "task_id": inst.task_id,
        "ext_id": inst.ext_id,
        "repo": inst.repo,
        "language": inst.language,
        "category": inst.category,
        "status": "pending",
        "container_id": None,
        "baseline_container_id": None,
        "verifier_container_id": None,
        "render_container_id": None,
        "render_container_ids": [],
        "rc": None,
        "reward": None,
        "resolved": None,
        "reward_counts": None,
        "reward_error": None,
        "verification_rc": None,
        "verify_returncode": None,
        "command_returncode": None,
        "baseline_test_returncode": None,
        "new_test_returncode": None,
        "diagnostics_persisted": None,
        "verifier_stdout_bytes": None,
        "verifier_stderr_bytes": None,
        "verifier_stdout_truncated": None,
        "verifier_stderr_truncated": None,
        "hidden_verifier_measurement_ref": None,
        "hidden_verifier_lifecycle_ref": None,
        "hidden_invocation_measurement_ref": None,
        "hidden_invocation_lifecycle_ref": None,
        "candidate_measurement_ref": None,
        "instance_spec_version": None,
        "cost_usd": 0.0,
        "budget_usd": assigned_task_budget,
        "planning_envelope": planning_envelope,
        "breach_artifacts_purged": [],
        "mutation_journal_settlement": None,
        "authoring_status": "pending",
        "candidate_capture_status": CandidateCaptureStatus.FAILED.value,
        "candidate": None,
        "pristine_replay": None,
        "render_replay": None,
        "verifier_status": DeepSWEVerifierStatus.NOT_RUN.value,
        "verifier_patch_preload": None,
        "verifier_patch": None,
        "failure_phase": None,
        "failure_kind": None,
        "architect_detail_status": None,
        "architect_decision": None,
        "architect_terminal_class": None,
        "architect_failure_phase": None,
        "architect_failure_kind": None,
        "architect_tiers_executed": None,
        "dev_verify_preflight": None,
        "zero_tier_settlement_proven": False,
        "secondary_failures": [],
        "cost_incomplete": False,
        "cleanup_status": "pending",
        "cleanup_errors": [],
        "storage_limit_bytes": inst.storage_mb * 1024 * 1024,
        "storage_peak_growth_bytes": 0,
        "wall_seconds": 0.0,
        "image_digest": inst.docker_image,
        "base_commit": None,
        "max_patch_bytes": args.max_patch_bytes,
        "tests_snapshot": None,
        "tests_source_identity": None,
        "test_inventory": None,
    }

    started = time.monotonic()
    if getattr(args, "diagnostic_cost_policy", None) is not None:
        summary.update({
            "diagnostic_cost_policy": args.diagnostic_cost_policy,
            "diagnostic_only": True,
            "rankable": False,
            "promotable": False,
        })
    authoring_id: str | None = None
    baseline_id: str | None = None
    render_id: str | None = None
    verifier_id: str | None = None
    active_container_id: str | None = None
    storage_quota: DirectoryGrowthQuota | None = None
    candidate: DeepSWECandidate | None = None
    pristine_replay: PristineReplayReport | None = None
    verifier_patch: VerifierPatchReport | None = None
    tests_identity: DeepSWETestsTreeIdentity | None = None
    try:
        validate_competition_admission(args.planner_count, assigned_task_budget)
        immutable_image = pull_image(inst.docker_image)
        inst = dataclasses.replace(inst, docker_image=immutable_image)
        summary["image_digest"] = immutable_image
        tests_identity = fingerprint_task_tests_tree(args.dataset, inst.task_id)
        summary["tests_source_identity"] = tests_identity.to_payload()
        # Fail fast, before any container or provider spend, on a whitelist
        # the grade could never be bound to. The live tree is not authority:
        # the inventory that binds the reward is read from the private
        # snapshot immediately before the hidden verifier.
        load_task_test_inventory(args.dataset, inst.task_id)

        # Workspaces are deliberately outside the directory bind-mounted at
        # /logs. Otherwise the model can reach sibling baseline/pristine
        # checkouts (including writable .git metadata) through /logs and
        # bypass the authoring container's /app/.git protection.
        # The workspace MUST be populated from the image's baked
        # /app before the bind-mount masks it — without this, the
        # implementer (and the verifier's
        # ``git reset --soft $PIER_MODEL_BASE_COMMIT``) start
        # against an empty directory and reward=0 uniformly.
        host_workspace = workspace_root / "candidate"
        host_workspace.mkdir(parents=True, exist_ok=True)
        container_scope = str(Path(args.spec).resolve())
        docker_settlement = DockerSettlementRegistrationClient.from_environment(
            required=False,
        )

        def _settlement_labels(*, phase: str) -> dict[str, str]:
            labels = managed_container_labels(
                scope=container_scope,
                task_id=inst.task_id,
                phase=phase,
            )
            if docker_settlement is None:
                return labels
            return docker_settlement.labels_for(labels)

        register_container: Callable[[str], None] | None = None
        if docker_settlement is not None:

            def _register_settlement_container(reference: str) -> None:
                docker_settlement.register_container(reference)

            register_container = _register_settlement_container
        # The dataset pins a few tasks by an abbreviated base commit. The
        # first population resolves the pin, once, to the full object id of
        # the image checkout. From here the two roles stay apart:
        # ``inst.base_commit_hash`` remains the dataset pin and the task
        # identity of checkpoints and Measurements, while ``base_commit`` /
        # ``checkout_inst`` carry the full id to every comparison with Git.
        base_commit = populate_workspace_from_image(
            inst.docker_image,
            host_workspace,
            base_commit=inst.base_commit_hash,
            labels=_settlement_labels(phase="workspace-population"),
            register_container=register_container,
        )
        if not is_full_commit_id(base_commit):
            raise InstanceContainerSetupError(
                "deepswe: workspace population did not resolve a full base "
                f"commit id for task {inst.task_id!r}"
            )
        checkout_inst = dataclasses.replace(inst, base_commit_hash=base_commit)
        summary["base_commit"] = base_commit
        # Per-task Corpus + starter Spec persistence. Existing lineage is
        # retained for audit, while ``restart`` below guarantees its landed
        # slices are replayed into this run's fresh workspace. The Spec's
        # project context carries the full id that Tier compares with HEAD.
        instance_corpus = Corpus(instance_spec_dir)
        had_lineage = _prepare_task_corpus(
            instance_corpus,
            synthesize_starter_spec(
                checkout_inst,
                task_budget_usd=summary["budget_usd"],
            ),
            task_id=inst.task_id,
        )
        baseline_workspace = workspace_root / "baseline"
        baseline_workspace.mkdir(parents=True, exist_ok=True)
        populate_workspace_from_image(
            inst.docker_image,
            baseline_workspace,
            base_commit=base_commit,
            labels=_settlement_labels(phase="baseline-workspace-population"),
            register_container=register_container,
        )
        verifier_workspace = workspace_root / "verifier"
        verifier_workspace.mkdir(parents=True, exist_ok=True)
        populate_workspace_from_image(
            inst.docker_image,
            verifier_workspace,
            base_commit=base_commit,
            labels=_settlement_labels(phase="verifier-workspace-population"),
            register_container=register_container,
        )
        grading_workspace = workspace_root / "grading"
        grading_workspace.mkdir(parents=True, exist_ok=True)
        populate_workspace_from_image(
            inst.docker_image,
            grading_workspace,
            base_commit=base_commit,
            labels=_settlement_labels(phase="grading-workspace-population"),
            register_container=register_container,
        )
        summary["checkpoint_fingerprint"] = _deepswe_checkpoint_fingerprint(
            inst,
            args=args,
            child_budget_usd=summary["budget_usd"],
            tests_identity=tests_identity,
        )

        def _kill_active_container() -> None:
            for container_id in {
                candidate_id
                for candidate_id in (active_container_id, baseline_id, render_id)
                if candidate_id is not None
            }:
                subprocess.run(
                    ["docker", "kill", container_id],
                    cwd=str(host_workspace),
                    capture_output=True,
                    text=True,
                    check=False,
                )

        # ``storage_mb`` is a growth allowance over the image-populated
        # workspace baseline. Monitor every writable bind mount across the
        # complete authoring + verifier lifetime; Docker writable-layer
        # storage flags do not cover those mounts.
        storage_quota = DirectoryGrowthQuota(
            (
                workspace_root,
                task_log_root,
                private_root,
            ),
            limit_bytes=inst.storage_mb * 1024 * 1024,
            on_exceeded=_kill_active_container,
            require_kernel=True,
        )
        storage_quota.__enter__()

        # Phase A — AUTHORING container. No /tests, no /solution:
        # the cascade's dev_verify gate runs the Defender's
        # adversarial tests (discovered from the /app worktree),
        # never the hidden dataset suite.
        baseline_id = start_authoring_container(
            inst,
            host_logs_dir=baseline_log_dir,
            host_workspace=baseline_workspace,
            container_name=managed_container_name(
                prefix="deepswe",
                scope=container_scope,
                task_id=inst.task_id,
                phase="baseline",
            ),
            labels=_settlement_labels(phase="baseline"),
            register_container=register_container,
        )
        summary["baseline_container_id"] = baseline_id
        authoring_id = start_authoring_container(
            inst,
            host_logs_dir=authoring_log_dir,
            host_workspace=host_workspace,
            container_name=managed_container_name(
                prefix="deepswe",
                scope=container_scope,
                task_id=inst.task_id,
                phase="author",
            ),
            labels=_settlement_labels(phase="authoring"),
            register_container=register_container,
        )
        summary["container_id"] = authoring_id
        active_container_id = authoring_id
        if storage_quota.exceeded:
            exceeded_quota = storage_quota
            storage_quota = None
            _finish_storage_quota(
                exceeded_quota,
                summary,
                kill_if_already_exceeded=_kill_active_container,
            )

        kwargs = _build_run_architect_kwargs(
            args=args, inst=checkout_inst,
            instance_spec_dir=instance_spec_dir,
            container_id=authoring_id,
            baseline_container_id=baseline_id,
            instance_log_dir=authoring_log_dir,
            host_workspace=host_workspace,
            restart=had_lineage,
            budget_usd=child_budget_usd,
            execution_fingerprint=summary["checkpoint_fingerprint"],
        )
        preflight_log_dir = task_run_dir / "dev-verify-preflight"
        preflight_restorations: list[DevelopmentPreflightRestorationReceipt] = []
        preflight = _run_dev_verify_preflight(
            spec=instance_corpus.latest_spec(),
            cwd=host_workspace,
            baseline_cwd=baseline_workspace,
            log_dir=preflight_log_dir,
            journal_root=(
                instance_spec_dir
                / "runs"
                / f"{run_dir.name}-dev-verify-preflight"
                / "final_gate_mutations"
            ),
            extra_parameters=kwargs["extra_parameters"],
            restoration_callback=preflight_restorations.append,
        )
        preflight_runner = preflight.dev_verify
        clean_infrastructure_provenance = (
            _clean_infrastructure_provenance_from_runner(preflight_runner)
            if preflight_runner is not None
            else None
        )
        gate_created_paths, gate_created_summary = _gate_created_paths_summary(
            preflight_restorations
        )
        summary["dev_verify_preflight"] = {
            "classification": preflight.classification,
            "returncode": (
                preflight_runner.returncode
                if preflight_runner is not None else None
            ),
            "values": (
                dict(preflight_runner.values)
                if preflight_runner is not None else {}
            ),
            "log_dir": str(preflight_log_dir),
            "workspace_restored": True,
            "clean_infrastructure_provenance_bound": (
                clean_infrastructure_provenance is not None
            ),
            **gate_created_summary,
        }
        if (
            not preflight.classification.startswith("success")
            or preflight_runner is None
        ):
            returncode = (
                preflight_runner.returncode
                if preflight_runner is not None
                and preflight_runner.returncode != 0
                else 2
            )
            summary.update({
                "rc": int(returncode),
                "cascade_status": "failed",
                "authoring_status": "failed",
                "status": "setup_failed",
                "failure_phase": "dev_verify_preflight",
                "failure_kind": preflight.classification,
                "error": (
                    "clean development-verifier preflight failed before "
                    "provider invocation: "
                    f"classification={preflight.classification}, "
                    f"returncode={returncode}"
                ),
            })
            logger.warning(
                "deepswe: task %s development-verifier preflight failed "
                "(classification=%s, returncode=%s); skipping Architect",
                inst.task_id,
                preflight.classification,
                returncode,
            )
            return summary
        if clean_infrastructure_provenance is not None:
            from crucible.products.deepswe.runners import (  # noqa: PLC0415
                deepswe_dev_verify,
            )

            architect_parameters = dict(kwargs["extra_parameters"])
            architect_metadata = dict(
                architect_parameters["deepswe_metadata"]
            )
            architect_metadata[
                deepswe_dev_verify.CLEAN_INFRASTRUCTURE_PROVENANCE_KEY
            ] = clean_infrastructure_provenance
            architect_parameters["deepswe_metadata"] = architect_metadata
            kwargs["extra_parameters"] = architect_parameters
        if gate_created_paths:
            from crucible.products.deepswe.runners import (  # noqa: PLC0415
                deepswe_dev_verify,
            )

            # Bound, every later gate run removes these paths again when its
            # own commands created them.  A record that cannot be bound
            # exactly stays listed in the summary, unbound, and stops the
            # task here as a preflight failure: before the Architect
            # operation id is recorded and before any provider invocation.
            gate_created_evidence = (
                deepswe_dev_verify.gate_created_paths_evidence(
                    gate_created_paths
                )
            )
            if gate_created_evidence is None:
                summary.update({
                    "rc": 2,
                    "cascade_status": "failed",
                    "authoring_status": "failed",
                    "status": "setup_failed",
                    "failure_phase": "dev_verify_preflight",
                    "failure_kind": "gate_created_paths_unbindable",
                    "error": (
                        "clean development-verifier preflight recorded "
                        f"{len(gate_created_paths)} gate-created path(s) "
                        "that cannot be bound exactly before provider "
                        "invocation; a bound record holds at most "
                        f"{deepswe_dev_verify.MAX_GATE_CREATED_PATHS} paths "
                        "and "
                        f"{deepswe_dev_verify.MAX_GATE_CREATED_PATH_BYTES} "
                        "bytes of canonical UTF-8 names"
                    ),
                })
                logger.warning(
                    "deepswe: task %s development-verifier preflight "
                    "recorded %d gate-created path(s) that cannot be bound "
                    "exactly; skipping Architect",
                    inst.task_id,
                    len(gate_created_paths),
                )
                return summary
            architect_parameters = dict(kwargs["extra_parameters"])
            architect_metadata = dict(
                architect_parameters["deepswe_metadata"]
            )
            architect_metadata[
                deepswe_dev_verify.GATE_CREATED_PATHS_KEY
            ] = gate_created_evidence
            architect_parameters["deepswe_metadata"] = architect_metadata
            kwargs["extra_parameters"] = architect_parameters
            summary["dev_verify_preflight"]["gate_created_paths_bound"] = True
        operation_id = _architect_operation_id(inst.task_id)
        summary["architect_operation_id"] = operation_id
        architect_error: Exception | None = None
        try:
            with _outcome.child_operation(
                run_dir,
                operation_id=operation_id,
                workflow="architect",
            ):
                rc = run_architect(**kwargs)
        except Exception as exc:  # noqa: BLE001 — account before returning
            architect_error = exc
            rc = 3
            logger.warning(
                "deepswe: task %s architect raised %s: %s; "
                "recovering its run cost before halting",
                inst.task_id, type(exc).__name__, exc,
            )
        summary["rc"] = int(rc)
        summary["cascade_status"] = (
            "error" if architect_error is not None
            else "passed" if rc == 0 else "failed"
        )
        summary["authoring_status"] = summary["cascade_status"]
        summary["status"] = (
            "cascade_error"
            if architect_error is not None else "verification_pending"
        )
        if architect_error is not None:
            summary["error"] = (
                f"{type(architect_error).__name__}: {architect_error}"
            )[:500]

        # run_architect returns only an rc and keeps its cost rollup in its own
        # run summary; read it back so this task's summary AND the sweep-level
        # accumulator (which drives the --budget-usd halt + cost_total_usd)
        # reflect real spend instead of $0. The hidden verifier may still
        # finish the current task, but an incomplete rollup halts the outer
        # sweep before another paid task starts.
        child_run_dir: Path | None = None
        try:
            arch_costs, typed_child, child_run_dir = _architect_run_cost(
                run_dir,
                operation_id,
            )
            if typed_child is not None:
                summary["architect_run_id"] = typed_child.run_id
                summary["architect_detail_status"] = typed_child.detail_status
                summary["architect_terminal_class"] = (
                    typed_child.terminal_class.value
                )
            if child_run_dir is not None:
                summary["architect_run_dir"] = str(child_run_dir)
            if arch_costs is not None:
                summary["cost_usd"] = _rollup_architect_cost(
                    cost, arch_costs, task_id=inst.task_id
                )
                summary["cost_incomplete"] = bool(
                    arch_costs.get("rollup_incomplete", False)
                )
                if _cost_summary_has_pricing_extension(arch_costs):
                    # Retain validated physical calls so checkpoint resume
                    # does not collapse them back to dollars alone.
                    summary["architect_costs"] = arch_costs
            else:
                logger.warning(
                    "deepswe: task %s — no architect cost summary found; "
                    "aggregate budget is no longer trustworthy", inst.task_id,
                )
                summary["cost_incomplete"] = True
        except Exception as e:  # noqa: BLE001 — cost reporting must not fail the task
            logger.warning(
                "deepswe: task %s cost rollup failed (%s); aggregate budget "
                "is no longer trustworthy",
                inst.task_id, e,
            )
            summary["cost_incomplete"] = True

        architect_summary = _architect_run_summary(child_run_dir)
        if architect_summary is not None:
            development_budget = architect_summary.get(
                "development_budget"
            )
            if isinstance(development_budget, dict):
                summary["development_budget"] = development_budget
                metrics = development_budget.get(
                    "autonomous_development_metrics"
                )
                if isinstance(metrics, dict):
                    summary["autonomous_development_metrics"] = metrics
            decision = architect_summary.get("decision")
            if isinstance(decision, str) and decision:
                summary["architect_decision"] = decision
            reason = architect_summary.get("reason")
            if isinstance(reason, str) and reason:
                summary["architect_reason"] = reason[:2000]
            tiers_executed = architect_summary.get("tiers_executed")
            if (
                not isinstance(tiers_executed, bool)
                and isinstance(tiers_executed, int)
                and tiers_executed >= 0
            ):
                summary["architect_tiers_executed"] = tiers_executed
            phase = architect_summary.get("failure_phase")
            kind = architect_summary.get("failure_kind")
            if isinstance(phase, str) and phase:
                summary["architect_failure_phase"] = phase
            if isinstance(kind, str) and kind:
                summary["architect_failure_kind"] = kind

        if rc != 0 or architect_error is not None:
            primary_phase = summary.get("architect_failure_phase") or "architect"
            primary_kind = (
                summary.get("architect_failure_kind")
                or summary.get("architect_detail_status")
                or (
                    type(architect_error).__name__
                    if architect_error is not None else "architect_failed"
                )
            )
            summary["failure_phase"] = primary_phase
            summary["failure_kind"] = primary_kind
            if summary.get("architect_reason"):
                summary["error"] = summary["architect_reason"]

        # Candidate capture is a security transition, not final cleanup.
        # Both model-accessible containers must be proven stopped first; ids
        # are cleared only after Docker confirms each container is quiescent.
        stop_instance_container_strict(authoring_id)
        authoring_id = None
        active_container_id = baseline_id
        stop_instance_container_strict(baseline_id)
        baseline_id = None
        active_container_id = None

        try:
            summary["mutation_journal_settlement"] = (
                _establish_capture_mutation_settlement(
                    instance_corpus,
                    child_run_dir,
                    inst=checkout_inst,
                    host_workspace=host_workspace,
                    max_patch_bytes=args.max_patch_bytes,
                )
            )
        except Exception as exc:  # noqa: BLE001 — fail closed before capture
            settlement_error = (
                "mutation journal settlement is unproven: "
                f"{type(exc).__name__}: {exc}"
            )
            summary["status"] = "candidate_failed"
            if summary.get("failure_phase") is not None:
                _append_secondary_failure(
                    summary,
                    phase="mutation_journal_settlement",
                    kind="mutation_journal_unsettled",
                    error=settlement_error,
                )
            else:
                summary["failure_phase"] = "mutation_journal_settlement"
                summary["failure_kind"] = "mutation_journal_unsettled"
                summary["error"] = settlement_error[:2000]
            # A zero-tier Architect failure produced no implementation to
            # capture. Do not manufacture an empty failed candidate or expose
            # the hidden verifier merely because settlement itself also failed.
            if rc != 0 and summary.get("architect_tiers_executed") == 0:
                return summary
            candidate = DeepSWECandidate.failed(settlement_error)
            summary["candidate_capture_status"] = candidate.status.value
            summary["candidate"] = candidate.to_payload()
            try:
                summary.update(_persist_candidate_only_outcome(
                    instance_corpus,
                    inst,
                    candidate,
                    status="candidate_failed",
                    run_dir=run_dir,
                    checkpoint_record=_lifecycle_record(summary),
                ))
            except Exception as persist_exc:
                raise _HiddenOraclePersistenceError(
                    "unsettled Tier journal evidence could not be committed "
                    f"for {inst.task_id}"
                ) from persist_exc
            return summary

        capture_attestation = summary["mutation_journal_settlement"][
            "capture_attestation"
        ]
        settled_tiers = summary["mutation_journal_settlement"]["tiers_executed"]
        summary["architect_tiers_executed"] = settled_tiers
        summary["zero_tier_settlement_proven"] = settled_tiers == 0
        if rc != 0 and settled_tiers == 0:
            # Planning/validation ended before implementation. The final and
            # capture attestations prove the workspace stayed unchanged, but
            # an empty patch is not a candidate and must never consume hidden
            # verifier authority.
            summary["status"] = "candidate_failed"
            return summary
        # Image Git, not host Git, defines the bytes the hidden verifier will
        # emit. Render directly from the now-quiescent author workspace in a
        # no-tests container, with a streaming hard cap before persistence.
        render_id = start_patch_render_container(
            inst,
            host_workspace=host_workspace,
            container_name=managed_container_name(
                prefix="deepswe",
                scope=container_scope,
                task_id=inst.task_id,
                phase="source-render",
            ),
            labels=_settlement_labels(phase="source-patch-render"),
            register_container=register_container,
        )
        summary["render_container_id"] = render_id
        summary["render_container_ids"].append(render_id)
        active_container_id = render_id
        try:
            rendered_patch = capture_model_patch_from_container(
                render_id,
                base_commit=base_commit,
                max_bytes=args.max_patch_bytes,
            )
        finally:
            stop_instance_container_strict(render_id)
            render_id = None
            active_container_id = None
        candidate = candidate_from_image_patch(
            rendered_patch,
            files_modified=tuple(
                capture_attestation["workspace_files_modified"]
            ),
            raw_diff_bytes=int(capture_attestation["workspace_patch_bytes"]),
            max_bytes=args.max_patch_bytes,
        )
        summary["candidate_capture_status"] = candidate.status.value
        summary["candidate"] = candidate.to_payload()
        summary["candidate_diff_bytes"] = candidate.diff_bytes
        summary["raw_workspace_diff_bytes"] = candidate.raw_diff_bytes

        if architect_error is not None:
            summary["failure_phase"] = "authoring"
            summary["failure_kind"] = type(architect_error).__name__
            # Authoring failure is diagnostic, not a submission veto. Preserve
            # and grade any nonempty exact candidate the workspace contains.

        if rc != 0 and candidate.status == CandidateCaptureStatus.EMPTY:
            # A failed authoring cascade produced no submission bytes. Retain
            # the exact empty capture as retryable diagnostic evidence, but do
            # not consume replay or one-shot hidden-verifier authority. A
            # deliberate empty candidate from a successful cascade remains a
            # valid benchmark submission.
            summary["status"] = "candidate_failed"
            try:
                summary.update(_persist_candidate_only_outcome(
                    instance_corpus,
                    inst,
                    candidate,
                    status="candidate_failed",
                    run_dir=run_dir,
                    checkpoint_record=_lifecycle_record(summary),
                ))
            except Exception as exc:
                raise _HiddenOraclePersistenceError(
                    "empty failed-authoring candidate evidence could not be "
                    f"committed for {inst.task_id}"
                ) from exc
            return summary

        # Host Git applies B but cannot prove B: equivalent worktrees can have
        # host/image serialization drift. A fresh image container must render
        # C from this exact checkout, and only C == B issues proof.
        apply_report = prepare_candidate_replay(
            verifier_workspace,
            candidate,
            expected_base_commit=base_commit,
            max_bytes=args.max_patch_bytes,
        )
        summary["candidate_apply"] = apply_report.to_payload()
        if apply_report.status != CandidateApplyStatus.APPLIED:
            pristine_replay = prove_image_replay(
                candidate,
                apply_report,
                b"",
                expected_base_commit=base_commit,
                max_bytes=args.max_patch_bytes,
            )
        else:
            render_id = start_patch_render_container(
                inst,
                host_workspace=verifier_workspace,
                container_name=managed_container_name(
                    prefix="deepswe",
                    scope=container_scope,
                    task_id=inst.task_id,
                    phase="proof-render",
                ),
                labels=_settlement_labels(phase="proof-patch-render"),
                register_container=register_container,
            )
            summary["render_container_id"] = render_id
            summary["render_container_ids"].append(render_id)
            active_container_id = render_id
            try:
                replayed_render = capture_model_patch_from_container(
                    render_id,
                    base_commit=base_commit,
                    max_bytes=args.max_patch_bytes,
                )
            finally:
                stop_instance_container_strict(render_id)
                render_id = None
                active_container_id = None
            pristine_replay = prove_image_replay(
                candidate,
                apply_report,
                replayed_render,
                expected_base_commit=base_commit,
                max_bytes=args.max_patch_bytes,
            )
        summary["pristine_replay"] = pristine_replay.to_payload()
        summary["render_replay"] = pristine_replay.to_payload()
        if pristine_replay.status != PristineReplayStatus.PASSED:
            summary["status"] = "pristine_replay_failed"
            summary["failure_phase"] = "image_replay_proof"
            summary["failure_kind"] = pristine_replay.status.value
            summary["error"] = pristine_replay.error
            try:
                summary.update(_persist_candidate_only_outcome(
                    instance_corpus,
                    inst,
                    candidate,
                    status="pristine_replay_failed",
                    run_dir=run_dir,
                    checkpoint_record=_lifecycle_record(summary),
                    pristine_replay=pristine_replay,
                ))
            except Exception as exc:
                raise _HiddenOraclePersistenceError(
                    "DeepSWE image replay failure evidence could not be "
                    f"committed for {inst.task_id}"
                ) from exc
            return summary

        if storage_quota.exceeded:
            exceeded_quota = storage_quota
            storage_quota = None
            _finish_storage_quota(exceeded_quota, summary)

        # Snapshot only after exact replay proof. The live dataset tree is
        # never mounted; the private copy is quota-covered and read-only.
        if tests_identity is None:
            raise RuntimeError("hidden tests identity was not established")
        tests_snapshot = snapshot_task_tests_tree(
            args.dataset,
            inst.task_id,
            (private_root / "tests-snapshots").resolve(),
            expected_identity=tests_identity,
        )
        if tests_snapshot.sha256 != tests_identity.sha256:
            raise RuntimeError("hidden tests snapshot identity mismatch")
        summary["tests_snapshot"] = tests_snapshot.to_payload()
        # The whitelist sizes come from the exact private bytes the verifier
        # mounts, so the reward's totals are bound to the snapshot digest.
        # Read before the invocation marker: a malformed whitelist must not
        # consume the one-shot verifier.
        test_inventory = load_snapshot_test_inventory(tests_snapshot)
        summary["test_inventory"] = (
            None if test_inventory is None else test_inventory.to_payload()
        )
        host_tests_dir = tests_snapshot.root
        verifier_patch_path = instance_log_dir / "artifacts" / "model.patch"
        verifier_patch_path.unlink(missing_ok=True)
        # DeepSWE v1.1 consumes the submitted patch from /logs/artifacts
        # during ``grader.py prepare``. Publish only the already captured and
        # pristine-replay-proven candidate, then prove the exact bytes before
        # granting the one-shot hidden mount. The post-stop proof below is
        # retained so a verifier cannot replace the submitted artifact.
        write_atomic_bytes(verifier_patch_path, candidate.patch)
        verifier_patch_preload = verify_hidden_model_patch(
            verifier_patch_path,
            candidate,
        )
        summary["verifier_patch_preload"] = verifier_patch_preload.to_payload()
        if verifier_patch_preload.status != VerifierPatchStatus.MATCHED:
            raise RuntimeError(
                "hidden verifier candidate patch preload is not exact"
            )
        # Consume one-shot authority durably before any process receives the
        # private /tests mount. Even image startup must not precede this marker.
        invocation_summary = {
            **summary,
            "status": "verification_incomplete",
            "verifier_status": DeepSWEVerifierStatus.NOT_RUN.value,
            "failure_phase": "hidden_verifier",
            "failure_kind": "hidden_invocation_incomplete",
        }
        try:
            summary.update(_persist_hidden_invocation_marker(
                instance_corpus,
                inst,
                candidate,
                run_dir=run_dir,
                checkpoint_record=_lifecycle_record(invocation_summary),
                pristine_replay=pristine_replay,
                tests_snapshot=tests_snapshot,
            ))
        except Exception as exc:
            raise _HiddenOraclePersistenceError(
                "hidden verifier invocation marker could not be committed "
                f"for {inst.task_id}"
            ) from exc
        verifier_id = start_verifier_container(
            inst,
            host_logs_dir=instance_log_dir,
            host_workspace=grading_workspace,
            host_tests_dir=host_tests_dir,
            container_name=managed_container_name(
                prefix="deepswe",
                scope=container_scope,
                task_id=inst.task_id,
                phase="verify",
            ),
            labels=_settlement_labels(phase="verification"),
            register_container=register_container,
        )
        summary["verifier_container_id"] = verifier_id
        active_container_id = verifier_id
        if storage_quota.exceeded:
            exceeded_quota = storage_quota
            storage_quota = None
            _finish_storage_quota(
                exceeded_quota,
                summary,
                kill_if_already_exceeded=_kill_active_container,
            )
        verdict = _run_dataset_verify(
            inst, container_id=verifier_id,
            instance_log_dir=instance_log_dir,
        )
        stop_instance_container_strict(verifier_id)
        verifier_id = None
        active_container_id = None
        verdict = _validate_stopped_verifier_reward(
            verdict,
            instance_log_dir=instance_log_dir,
            test_inventory=test_inventory,
        )
        summary["reward"] = verdict["reward"]
        summary["resolved"] = verdict["resolved"]
        summary["reward_counts"] = verdict.get("reward_counts")
        summary["reward_error"] = verdict.get("reward_error")
        summary["verify_returncode"] = verdict.get("verify_returncode")
        for field in _VERIFIER_OBSERVABILITY_FIELDS:
            summary[field] = verdict.get(field)
        summary["verification_rc"] = verdict["rc"]
        summary["reward_valid"] = verdict.get("reward_valid")
        summary["reward_state"] = verdict.get("reward_state")
        verifier_status = _dataset_verifier_status(verdict)
        verifier_patch = verify_hidden_model_patch(
            verifier_patch_path,
            candidate,
        )
        summary["verifier_patch"] = verifier_patch.to_payload()
        if verifier_patch.status == VerifierPatchStatus.MATCHED:
            summary["verifier_status"] = verifier_status.value
        else:
            verifier_status_by_patch = {
                VerifierPatchStatus.MISSING: DeepSWEVerifierStatus.PATCH_MISSING,
                VerifierPatchStatus.MISMATCH: DeepSWEVerifierStatus.PATCH_MISMATCH,
                VerifierPatchStatus.UNREADABLE: (
                    DeepSWEVerifierStatus.PATCH_UNREADABLE
                ),
            }
            summary["verifier_status"] = verifier_status_by_patch[
                verifier_patch.status
            ].value
            summary["failure_phase"] = "hidden_verifier_artifact"
            summary["failure_kind"] = summary["verifier_status"]
            summary["error"] = verifier_patch.error
        # Close the quota before publishing an authoritative hidden-verifier
        # outcome. Its final synchronous sample closes the race between the
        # monitor's last poll and verifier completion.
        completed_quota = storage_quota
        storage_quota = None
        _finish_storage_quota(
            completed_quota,
            summary,
            kill_if_already_exceeded=_kill_active_container,
        )
        # The hidden dataset verifier is the benchmark oracle. Cascade rc
        # remains diagnostic, but cannot turn a reward-0 patch into success
        # or reject a reward-1 patch.
        status = (
            _dataset_verdict_status(verdict)
            if verifier_patch.status == VerifierPatchStatus.MATCHED
            else "verification_failed"
        )
        summary["status"] = status
        if status == "verification_failed" and summary["failure_kind"] is None:
            summary["failure_phase"] = "hidden_verifier"
            summary["failure_kind"] = verifier_status.value
        try:
            summary.update(_persist_dataset_outcome(
                instance_corpus,
                inst,
                verdict,
                status=status,
                run_dir=run_dir,
                checkpoint_record=_lifecycle_record(summary),
                candidate=candidate,
                pristine_replay=pristine_replay,
                verifier_patch=verifier_patch,
            ))
        except Exception as exc:
            raise _HiddenOraclePersistenceError(
                "authoritative hidden verifier result could not be committed "
                f"for {inst.task_id}"
            ) from exc
    except _HiddenOraclePersistenceError:
        # The verifier is one-shot. Never degrade a failed durable publication
        # into a retryable cascade status that can silently invoke it again.
        raise
    except StorageQuotaExceeded as e:
        logger.warning(
            "deepswe: task %s exceeded writable storage quota: %s",
            inst.task_id, e,
        )
        if not _preserve_architect_primary(
            summary,
            phase="storage_quota",
            kind=type(e).__name__,
            error=str(e),
        ):
            summary["status"] = (
                "verification_incomplete"
                if summary.get("hidden_invocation_measurement_ref")
                else "storage_quota_exceeded"
            )
            summary["failure_phase"] = "storage_quota"
            summary["failure_kind"] = type(e).__name__
            summary["error"] = str(e)[:500]
    except RustToolchainUnavailable as e:
        logger.warning(
            "deepswe: task %s Rust toolchain is unavailable: %s",
            inst.task_id,
            e,
        )
        hidden_authority_consumed = bool(
            summary.get("hidden_invocation_measurement_ref")
        )
        if hidden_authority_consumed:
            # Invocation authority is consumed before /tests is mounted. A
            # verifier-container bootstrap failure remains one-shot even
            # though no dataset command was reached.
            summary["status"] = "verification_incomplete"
            summary["failure_phase"] = "hidden_verifier"
        else:
            summary["status"] = "toolchain_unavailable"
            summary["failure_phase"] = "container_setup"
        summary["failure_kind"] = "toolchain_unavailable"
        summary["error"] = str(e)[:500]
    except InstanceContainerSetupError as e:
        logger.warning(
            "deepswe: task %s container setup/teardown failed: %s",
            inst.task_id, e,
        )
        if not _preserve_architect_primary(
            summary,
            phase="container_setup",
            kind=type(e).__name__,
            error=str(e),
        ):
            hidden_authority_consumed = bool(
                summary.get("hidden_invocation_measurement_ref")
            )
            summary["status"] = (
                "verification_incomplete"
                if hidden_authority_consumed
                else "setup_failed"
            )
            summary["failure_phase"] = (
                "hidden_verifier"
                if hidden_authority_consumed else "container_setup"
            )
            summary["failure_kind"] = type(e).__name__
            summary["error"] = str(e)[:500]
    except Exception as e:  # noqa: BLE001 — sweep must not abort
        logger.warning(
            "deepswe: task %s cascade raised %s: %s",
            inst.task_id, type(e).__name__, e,
        )
        hidden_authority_consumed = bool(
            summary.get("hidden_invocation_measurement_ref")
        )
        summary["status"] = (
            "verification_incomplete"
            if hidden_authority_consumed
            else "cascade_error"
        )
        summary["failure_phase"] = (
            "hidden_verifier"
            if hidden_authority_consumed
            else summary.get("failure_phase") or "workflow"
        )
        summary["failure_kind"] = summary.get("failure_kind") or type(e).__name__
        summary["error"] = f"{type(e).__name__}: {e}"[:500]
    finally:
        if storage_quota is not None:
            incomplete_quota = storage_quota
            storage_quota = None
            try:
                _finish_storage_quota(
                    incomplete_quota,
                    summary,
                    kill_if_already_exceeded=_kill_active_container,
                )
            except StorageQuotaExceeded as exc:
                logger.warning(
                    "deepswe: task %s exceeded writable storage quota: %s",
                    inst.task_id, exc,
                )
                if not _preserve_architect_primary(
                    summary,
                    phase="storage_quota",
                    kind=type(exc).__name__,
                    error=str(exc),
                ):
                    summary["status"] = (
                        "verification_incomplete"
                        if summary.get("hidden_invocation_measurement_ref")
                        else "storage_quota_exceeded"
                    )
                    summary["failure_phase"] = "storage_quota"
                    summary["failure_kind"] = type(exc).__name__
                    summary["error"] = str(exc)[:500]
        wall = time.monotonic() - started
        summary["wall_seconds"] = wall
        # Stop whichever containers are still up (idempotent,
        # image=None so neither call reclaims the image), then
        # reclaim the per-task image once at the end — covers every
        # path including a raise before Phase B tore the authoring
        # container down.
        cleanup_calls = []
        if authoring_id:
            cleanup_calls.append((authoring_id, None, "authoring"))
        if baseline_id:
            cleanup_calls.append((baseline_id, None, "baseline"))
        if render_id:
            cleanup_calls.append((render_id, None, "render"))
        if verifier_id:
            cleanup_calls.append((verifier_id, None, "verifier"))
        if summary["container_id"]:
            cleanup_calls.append((None, inst.docker_image, "image"))
        for container_id, image, label in cleanup_calls:
            try:
                teardown_instance_container(container_id, image=image)
            except Exception as exc:  # noqa: BLE001 — preserve outer summary
                message = f"{label}: {type(exc).__name__}: {exc}"[:500]
                summary["cleanup_errors"].append(message)
                logger.error(
                    "deepswe: task %s cleanup failed (%s)",
                    inst.task_id, message,
                )
        if summary["cleanup_errors"]:
            _preserve_architect_primary(
                summary,
                phase="cleanup",
                kind="cleanup_failed",
                error="; ".join(summary["cleanup_errors"]),
            )
        summary["cleanup_status"] = (
            "failed" if summary["cleanup_errors"] else "complete"
        )

    # Every caught task terminal must carry an exact process-style return
    # code. Failures before Architect assigns ``rc`` (including verifier
    # preflight settlement failures) otherwise publish a valid lifecycle row
    # with ``rc: null`` that the frozen cohort authority cannot classify as a
    # known failure.
    if type(summary.get("rc")) is not int:
        summary["rc"] = 3

    return summary


def _emit_summary(
    *,
    handle,
    args: argparse.Namespace,
    task_summaries: list[dict[str, Any]],
    cost: CostAccumulator,
    status: str,
    halt_reason: str | None,
    tasks_selected: int,
    tasks_not_started: int,
) -> Path:
    """Publish the outer run summary transactionally. Returns its path.

    ``tasks_selected`` is the size of the filtered selection and
    ``tasks_not_started`` the number of its tasks that were never handed to
    execution. The roll-up keeps apart what a single ``tasks_failed`` number
    cannot: a task graded 0 (``tasks_unresolved``), a task this run executed
    or resumed without a valid grade (``tasks_ungraded``), a task another
    process holds (``tasks_lease_busy``), a task that was handed to execution
    but ended without a task summary (``tasks_unrecorded``), and a task the
    run never reached (``tasks_not_started``).
    """
    total_cost = cost.total_cost_usd
    cost_summary = cost.to_dict()
    cost_incomplete = any(
        bool(summary.get("cost_incomplete"))
        for summary in task_summaries
    )
    tasks_by_status: dict[str, int] = {}
    for summary in task_summaries:
        task_status = str(summary["status"])
        tasks_by_status[task_status] = tasks_by_status.get(task_status, 0) + 1
    tasks_unresolved = tasks_by_status.get("unresolved", 0)
    tasks_graded = tasks_by_status.get("resolved", 0) + tasks_unresolved
    tasks_lease_busy = tasks_by_status.get("task_lease_busy", 0)
    counted = [
        RewardCounts.from_payload(summary["reward_counts"])
        for summary in task_summaries
        if summary["status"] in _GRADED_TASK_STATUSES
        and summary.get("reward_counts") is not None
    ]
    payload = {
        "workflow": "deepswe",
        "status": status,
        "halt_reason": halt_reason,
        "spec_dir": str(args.spec),
        "dataset": str(args.dataset),
        "tasks_selected": tasks_selected,
        "tasks_not_started": tasks_not_started,
        "tasks_unrecorded": (
            tasks_selected - len(task_summaries) - tasks_not_started
        ),
        "tasks_total": len(task_summaries),
        "tasks_ok": sum(
            1 for s in task_summaries if s["status"] == "resolved"
        ),
        "tasks_failed": sum(
            1 for s in task_summaries
            if s["status"] != "resolved"
        ),
        # Dataset-gate roll-up: how many tasks the hidden verifier
        # graded reward==1. This is the authoritative resolved count
        # (independent of the cascade rc, which reflects the
        # dev_verify gate against the Defender's per-round tests).
        "tasks_resolved": sum(
            1 for s in task_summaries if s.get("status") == "resolved"
        ),
        "tasks_cleanup_failed": sum(
            1 for s in task_summaries if s.get("cleanup_status") == "failed"
        ),
        "tasks_graded": tasks_graded,
        "tasks_unresolved": tasks_unresolved,
        "tasks_ungraded": (
            len(task_summaries) - tasks_graded - tasks_lease_busy
        ),
        "tasks_lease_busy": tasks_lease_busy,
        # Results a plain re-run does not reuse. This is the lifecycle
        # status policy, not a claim that the failure was transient.
        "tasks_rerun_by_default": sum(
            1
            for s in task_summaries
            if s["status"] in _LIFECYCLE_TERMINAL_STATUSES
            and _task_status_policy(s) == "retryable"
        ),
        "tasks_by_status": dict(sorted(tasks_by_status.items())),
        # Per-test counts summed over graded tasks that report them. A task
        # graded from a count-less v1.0 token is graded but not counted.
        "reward_counts": {
            "tasks_counted": len(counted),
            "tasks_apply_failed": sum(
                1 for counts in counted if counts.apply_failed
            ),
            "f2p_passed": sum(counts.f2p_passed for counts in counted),
            "f2p_total": sum(counts.f2p_total for counts in counted),
            "p2p_passed": sum(counts.p2p_passed for counts in counted),
            "p2p_total": sum(counts.p2p_total for counts in counted),
        },
        "cost_total_usd": total_cost,
        "planner_count": args.planner_count,
        "budget_usd": args.budget_usd,
        "budget_remaining_usd": (
            None
            if args.budget_usd is None
            else 0.0
            if cost_incomplete
            else max(0.0, args.budget_usd - total_cost)
        ),
        "cost_incomplete": cost_incomplete,
        "tasks": task_summaries,
    }
    if _cost_summary_has_pricing_extension(cost_summary):
        # Provider-free/legacy runs retain their exact historical shape.
        payload["costs"] = cost_summary
    if getattr(args, "diagnostic_cost_policy", None) is not None:
        payload.update({
            "diagnostic_cost_policy": args.diagnostic_cost_policy,
            "diagnostic_only": True,
            "rankable": False,
            "promotable": False,
        })
    handle.write_summary(payload)
    return handle.run_dir / "summary.json"


def _all_task_failures_are_retryable(
    task_summaries: list[dict[str, Any]],
) -> bool:
    """Preserve a retryable Architect terminal at the aggregate boundary.

    Hidden-verifier outcomes are one-shot and use their existing reusable
    policy.  This narrow projection applies only to pre-oracle task failures
    whose exact Architect outcome is retryable and whose accounting and
    cleanup are complete.  It changes no task execution or replay authority.
    """

    failed = [
        summary
        for summary in task_summaries
        if summary.get("status") != "resolved"
    ]
    return bool(failed) and all(
        summary.get("status") in _LIFECYCLE_RETRYABLE_STATUSES
        and summary.get("architect_terminal_class")
        == _outcome.TerminalClass.RETRYABLE_FAILURE.value
        and "hidden_invocation_measurement_ref" in summary
        and summary["hidden_invocation_measurement_ref"] is None
        and summary.get("cost_incomplete") is False
        and summary.get("cleanup_status") == "complete"
        for summary in failed
    )


_LIFECYCLE_SCHEMA = "crucible.deepswe.lifecycle.v5"
_LIFECYCLE_TERMINAL_STATUSES = frozenset({
    "candidate_failed",
    "cascade_error",
    "pristine_replay_failed",
    "resolved",
    "setup_failed",
    "storage_quota_exceeded",
    "toolchain_unavailable",
    "unresolved",
    "verification_failed",
    "verification_incomplete",
})
_LIFECYCLE_REUSABLE_STATUSES = frozenset({
    "resolved",
    "unresolved",
    # Once the hidden oracle starts it is one-shot. Failed/final-incomplete
    # invocations therefore require explicit rerun authority too.
    "verification_failed",
    "verification_incomplete",
})
_LIFECYCLE_RETRYABLE_STATUSES = (
    _LIFECYCLE_TERMINAL_STATUSES - _LIFECYCLE_REUSABLE_STATUSES
)
# Stops raised before Architect is started: no model was invoked and no work
# exists to judge.
_UNJUDGED_SETUP_STATUSES = frozenset({"setup_failed", "toolchain_unavailable"})


def _task_failure_is_unjudged(summary: dict[str, Any]) -> bool:
    """A settled pre-oracle task failure in which nothing judged the work.

    Either Architect was never started (a setup stop), or its exact typed
    outcome is not a ruling: inconclusive, retryable or cancelled. The
    accounting and cleanup terms are those of the retryable projection, so a
    task with an incomplete cost rollup never qualifies.
    """

    if not (
        "hidden_invocation_measurement_ref" in summary
        and summary["hidden_invocation_measurement_ref"] is None
        and summary.get("cost_incomplete") is False
        and summary.get("cleanup_status") == "complete"
    ):
        return False
    architect_class = summary.get("architect_terminal_class")
    if architect_class is None:
        # ``architect_operation_id`` is recorded immediately before Architect
        # starts. A started Architect without a typed outcome proves nothing.
        return (
            summary.get("architect_operation_id") is None
            and summary.get("status") in _UNJUDGED_SETUP_STATUSES
        )
    return (
        summary.get("status") in _LIFECYCLE_RETRYABLE_STATUSES
        and architect_class
        in {item.value for item in _outcome.UNJUDGED_TERMINAL_CLASSES}
    )


def _all_task_failures_are_unjudged(
    task_summaries: list[dict[str, Any]],
) -> bool:
    """Keep a run in which no failed task was judged out of ``tasks_failed``.

    This is a label for the aggregate run only. Like the retryable
    projection it changes no task execution, replay or rerun authority, and
    it is not a claim that a later attempt would succeed.
    """

    failed = [
        summary
        for summary in task_summaries
        if summary.get("status") != "resolved"
    ]
    return bool(failed) and all(
        _task_failure_is_unjudged(summary) for summary in failed
    )


# The hidden verifier produced a valid binary reward for the exact candidate.
_GRADED_TASK_STATUSES = frozenset({"resolved", "unresolved"})


def _effective_deepswe_backend_pricing_identity(
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Bind every model route and price that DeepSWE may consume."""

    routing = load_routing(
        config_path=args.backend_config,
        spec_dir=args.spec,
    )
    pricing = load_pricing(spec_dir=args.spec)
    roles = {
        "architect_planner",
        "architect_critic",
        "architect_judge",
        "implementer",
        "breach_defender",
        "review_conflict_authority",
        DEEPSWE_BREACH_POLICY.breaker_rubric,
        DEEPSWE_BREACH_POLICY.judge_rubric,
        *DEEPSWE_REVIEW_POLICY.rubrics,
    }
    effective: dict[str, Any] = {}
    for role in sorted(roles):
        config = routing.for_role(role)
        model = config.model
        if model is None:
            raise ValueError(f"DeepSWE role {role!r} has no priced model")
        price = pricing_for_model(model, pricing=pricing)
        effective[role] = {
            "route": {
                "backend": config.backend,
                "model": model,
                "effort": config.effort,
                "base_url": config.base_url,
                "api_key_configured": config.api_key is not None,
                "extra": config.extra,
            },
            "pricing": dataclasses.asdict(price),
        }
    return {
        "routing_fingerprint": routing_fingerprint(routing),
        "routing_source_sha256": routing.source_sha256,
        "roles": effective,
    }


def _deepswe_checkpoint_fingerprint(
    inst: DeepSWEInstance,
    *,
    args: argparse.Namespace,
    child_budget_usd: float | None,
    tests_identity: DeepSWETestsTreeIdentity | None = None,
) -> str:
    """Bind a lifecycle record to every input that can affect its result."""

    crucible_root = Path(__file__).resolve().parents[1]
    product_root = crucible_root / "products" / "deepswe"
    workflow_root = crucible_root / "workflows"
    tool_root = crucible_root / "tools"
    common_runner_root = crucible_root / "common" / "runners"
    assay_root = crucible_root.parent / "assay"
    backend_pricing = _effective_deepswe_backend_pricing_identity(args=args)
    development_budget_policy = (
        _deepswe_development_budget_policy(
            child_budget_usd, planner_count=args.planner_count,
        )
        if child_budget_usd is not None and child_budget_usd > 0
        else None
    )
    planner_tool_call_tranches = (
        (args.planner_tool_call_limit,)
        if args.planner_count > 1
        else (
            DEFAULT_PLANNER_TOOL_CALL_TRANCHES
            if args.planner_tool_call_limit
            == sum(DEFAULT_PLANNER_TOOL_CALL_TRANCHES)
            else None
        )
    )
    if tests_identity is None:
        tests_identity = fingerprint_task_tests_tree(args.dataset, inst.task_id)
    return canonical_fingerprint({
        "schema": "crucible.deepswe.checkpoint.v5",
        # Includes instruction, hidden verifier inputs, resource policy, base
        # commit, and the immutable image digest.  Only the hash is persisted.
        "task": dataclasses.asdict(inst),
        "execution": {
            "max_turns": args.max_turns,
            "max_patch_bytes": args.max_patch_bytes,
            "child_budget_usd": child_budget_usd,
            **({"diagnostic_cost_policy": args.diagnostic_cost_policy}
               if getattr(args, "diagnostic_cost_policy", None) is not None else {}),
            "planning_envelope": _deepswe_planning_envelope(
                child_budget_usd, planner_count=args.planner_count,
            ),
            "include_breach": not args.no_include_breach,
            "replan": args.replan,
            "planner_source_tool_budget": {
                "call_limit": args.planner_tool_call_limit,
                "result_bytes": args.planner_tool_result_bytes,
                "call_tranches": planner_tool_call_tranches,
            },
            "development_budget_policy": (
                {
                    "schema_version": development_budget_policy.schema_version,
                    "digest": development_budget_policy.digest,
                }
                if development_budget_policy is not None
                else None
            ),
            "backend_pricing": backend_pricing,
            "image_verification_policy": (
                image_verification_policy_identity()
            ),
            "hidden_tests": tests_identity.to_payload(),
        },
        "recipe": {
            "workflow": file_digest(Path(__file__)),
            "architect_workflow": file_digest(workflow_root / "architect.py"),
            "planning_contract": planning_recipe_identity(
                crucible_root, digest_file=file_digest,
            ),
            "tier_workflow": file_digest(workflow_root / "tier.py"),
            "breach_workflow": file_digest(workflow_root / "breach.py"),
            "review_workflow": file_digest(workflow_root / "_review.py"),
            "analysis_handoff": file_digest(
                workflow_root / "_analysis_handoff.py"
            ),
            "mutation_journal": file_digest(
                workflow_root / "_mutation_journal.py"
            ),
            "development_verifier_preflight": file_digest(
                workflow_root / "_deepswe_baseline_preflight.py"
            ),
            "build_verify": file_digest(workflow_root / "_build_verify.py"),
            "backend_pool": file_digest(workflow_root / "_backends.py"),
            "development_budget": file_digest(
                workflow_root / "_development_budget.py"
            ),
            "pre_review_plan_validation": file_digest(
                workflow_root / "_plan_validation.py"
            ),
            "planner_tool_budget": file_digest(
                tool_root / "_dispatch_budget.py"
            ),
            # Source navigation is part of the planning recipe. Bind both its
            # handlers and the support that defines tool schemas, read-only
            # classification, workspace context, and path containment.
            "planner_source_tools": {
                name: file_digest(tool_root / name)
                for name in (
                    "source/__init__.py",
                    "source/handlers.py",
                    "_builder.py",
                    "_context.py",
                    "_paths.py",
                )
            },
            "prompt": file_digest(product_root / "prompts.py"),
            "spec_synthesis": file_digest(product_root / "spec_synthesis.py"),
            "dev_verify": file_digest(product_root / "dev_verify_overrides.py"),
            "dev_verify_runner": file_digest(
                product_root / "runners" / "deepswe_dev_verify.py"
            ),
            "dev_verify_outcome": file_digest(
                product_root / "runners" / "dev_verify_outcome.py"
            ),
            # Exact per-attack observations are part of the acceptance
            # recipe. Bind both the target-environment pytest observer and
            # the host-side validator that decides which metadata is trusted.
            "pytest_observation_protocol": {
                name: file_digest(common_runner_root / name)
                for name in ("pytest_observer.py", "observations.py")
            },
            # The project-suite command and its shared failure budget affect
            # both repair feedback and clean-base comparison.  They are
            # intentionally separate from the exact one-node observation
            # protocol above.
            "python_dev_verify_protocol": {
                name: file_digest(common_runner_root / name)
                for name in ("pytest_policy.py", "lang/python.py")
            },
            "hidden_verifier_runner": file_digest(
                product_root / "runners" / "deepswe_verifier.py"
            ),
            # Whether a verifier result is a grade is decided by the reward
            # reader and by the whitelist rule of the dataset loader.
            "reward_artifact_reader": file_digest(
                product_root / "secure_artifacts.py"
            ),
            "dataset_loader": file_digest(product_root / "dataset.py"),
            "instance_container": file_digest(
                product_root / "instance_container.py"
            ),
            "git_safety": file_digest(product_root / "git_safety.py"),
            "rust_toolchain": file_digest(
                product_root / "rust_toolchain.py"
            ),
            "mutation_settlement": file_digest(
                product_root / "mutation_settlement.py"
            ),
            "pristine_verifier": file_digest(
                product_root / "pristine_verifier.py"
            ),
            "rubrics": file_set_fingerprint(
                crucible_root / "rubrics", "*.yaml",
            ),
            "rubric_loader": file_digest(
                crucible_root / "rubrics" / "__init__.py"
            ),
            "backend_runtime": backend_recipe_identity(
                crucible_root,
                digest_file=file_digest,
            ),
            "assay": {
                "version": read_assay_version(),
                "source": (
                    file_set_fingerprint(
                        assay_root / "src" / "assay",
                        "**/*.py",
                    )
                    if (assay_root / "src" / "assay").is_dir()
                    else None
                ),
                "pyproject": file_digest(assay_root / "pyproject.toml")
                if (assay_root / "pyproject.toml").is_file() else None,
            },
            "build_identity": {
                "python_version": file_digest(
                    crucible_root / ".python-version"
                ),
                "pyproject": file_digest(crucible_root / "pyproject.toml"),
                "uv_lock": file_digest(crucible_root / "uv.lock"),
            },
        },
    })


def _is_proven_permanent_preimplementation_terminal(
    value: dict[str, Any],
) -> bool:
    """Return whether a zero-tier Architect terminal is safe to cache.

    Hidden-oracle terminals have their own reusable policy and Corpus-backed
    authority. This narrower predicate is only for a deterministic Architect
    planning/validation veto that produced no candidate and completed every
    proof, accounting, and cleanup obligation.
    """

    rc = value.get("rc")
    tiers_executed = value.get("architect_tiers_executed")
    architect_phase = value.get("architect_failure_phase")
    architect_kind = value.get("architect_failure_kind")
    return bool(
        value.get("status") == "candidate_failed"
        and not isinstance(rc, bool)
        and isinstance(rc, int)
        and rc != 0
        and value.get("authoring_status") == "failed"
        and value.get("architect_terminal_class") == "permanent_failure"
        and isinstance(value.get("architect_detail_status"), str)
        and bool(value.get("architect_detail_status"))
        and tiers_executed == 0
        and not isinstance(tiers_executed, bool)
        and architect_phase in {
            "architect_planning",
            "architect_validation",
        }
        and isinstance(architect_kind, str)
        and bool(architect_kind)
        and value.get("failure_phase") == architect_phase
        and value.get("failure_kind") == architect_kind
        and value.get("zero_tier_settlement_proven") is True
        and value.get("candidate") is None
        and value.get("candidate_capture_status")
        == CandidateCaptureStatus.FAILED.value
        and value.get("pristine_replay") is None
        and value.get("verifier_status") == DeepSWEVerifierStatus.NOT_RUN.value
        and value.get("verifier_patch") is None
        and value.get("tests_snapshot") is None
        and value.get("reward") is None
        and value.get("resolved") is None
        and value.get("cost_incomplete") is False
        and value.get("cleanup_status") == "complete"
        and value.get("cleanup_errors") == []
        and value.get("secondary_failures") == []
    )


def _task_status_policy(summary: dict[str, Any]) -> str:
    """Classify what a later run may do with one task result.

    ``reusable`` and a proven ``permanent`` terminal are kept; ``retryable``
    carries no reuse authority, so a plain re-run executes the task again.
    """

    if summary.get("diagnostic_cost_policy") is not None:
        return "diagnostic_only"
    if summary["status"] in _LIFECYCLE_REUSABLE_STATUSES:
        return "reusable"
    if _is_proven_permanent_preimplementation_terminal(summary):
        return "permanent"
    return "retryable"


def _checkpoint_is_reusable(
    record: dict[str, Any], *, args: argparse.Namespace,
) -> bool:
    status = record["status"]
    if record.get("diagnostic_cost_policy") is not None or record.get("diagnostic_only") is True:
        return False
    if record.get("cost_incomplete") or args.rerun or status in set(args.rerun_status):
        return False
    policy = record.get("status_policy")
    if policy == "permanent":
        return _is_proven_permanent_preimplementation_terminal(record)
    return policy == "reusable"


def _restore_checkpoint_cost(
    cost: CostAccumulator, record: dict[str, Any], *, task_id: str,
) -> None:
    """Replay durable checkpoint spend into the resumed sweep ledger."""

    amount = float(record.get("cost_usd", 0.0) or 0.0)
    architect_costs = record.get("architect_costs")
    if architect_costs is not None:
        if type(architect_costs) is not dict:
            raise ValueError("checkpoint Architect cost summary is invalid")
        restored = _rollup_architect_cost(
            cost, architect_costs, task_id=task_id,
        )
        if not math.isclose(restored, amount, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "checkpoint Architect cost summary does not match cost_usd"
            )
        return
    cost.record(
        role=f"deepswe_resume:{task_id}",
        usage=_CostShim(amount),
        backend_name="checkpoint",
        model="checkpoint",
    )


def _lifecycle_record(summary: dict[str, Any]) -> dict[str, Any]:
    """Project one task summary into the durable lifecycle contract."""

    record = {
        "schema_version": _LIFECYCLE_SCHEMA,
        "task_id": summary["task_id"],
        "ext_id": summary["ext_id"],
        "repo": summary.get("repo"),
        "language": summary.get("language"),
        "category": summary.get("category"),
        "status": summary["status"],
        "rc": summary.get("rc"),
        "reward": summary.get("reward"),
        "resolved": summary.get("resolved"),
        "reward_counts": summary.get("reward_counts"),
        "verification_rc": summary.get("verification_rc"),
        "verify_returncode": summary.get("verify_returncode"),
        **{
            field: summary.get(field)
            for field in _VERIFIER_OBSERVABILITY_FIELDS
        },
        "cleanup_status": summary.get("cleanup_status"),
        "cleanup_errors": summary.get("cleanup_errors", []),
        "wall_seconds": summary.get("wall_seconds"),
        "cost_usd": summary.get("cost_usd", 0.0),
        "cost_incomplete": bool(summary.get("cost_incomplete", False)),
        "assigned_budget_usd": summary.get("budget_usd"),
        "checkpoint_fingerprint": summary["checkpoint_fingerprint"],
        "image_digest": summary.get("image_digest"),
        "authoring_status": summary.get("authoring_status"),
        "candidate_capture_status": summary.get("candidate_capture_status"),
        "candidate": summary.get("candidate"),
        "pristine_replay": summary.get("pristine_replay"),
        "verifier_status": summary.get("verifier_status"),
        "verifier_patch": summary.get("verifier_patch"),
        "tests_source_identity": summary.get("tests_source_identity"),
        "tests_snapshot": summary.get("tests_snapshot"),
        "failure_phase": summary.get("failure_phase"),
        "failure_kind": summary.get("failure_kind"),
        "architect_detail_status": summary.get("architect_detail_status"),
        "architect_decision": summary.get("architect_decision"),
        "architect_terminal_class": summary.get("architect_terminal_class"),
        "architect_failure_phase": summary.get("architect_failure_phase"),
        "architect_failure_kind": summary.get("architect_failure_kind"),
        "architect_tiers_executed": summary.get("architect_tiers_executed"),
        "zero_tier_settlement_proven": (
            summary.get("zero_tier_settlement_proven") is True
        ),
        "secondary_failures": summary.get("secondary_failures", []),
        "status_policy": _task_status_policy(summary),
    }
    architect_costs = summary.get("architect_costs")
    if summary.get("diagnostic_cost_policy") is not None:
        record.update({
            "diagnostic_cost_policy": summary["diagnostic_cost_policy"],
            "diagnostic_only": True,
            "rankable": False,
            "promotable": False,
        })
    if (
        type(architect_costs) is dict
        and _cost_summary_has_pricing_extension(architect_costs)
    ):
        record["architect_costs"] = architect_costs
    return record


def _validate_lifecycle_record(
    value: Any,
    *,
    line_number: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"lifecycle JSONL line {line_number} must be an object"
        )
    value = dict(value)
    legacy_schema = value.get("schema_version")
    if legacy_schema == "crucible.deepswe.lifecycle.v2":
        # Strict legacy normalization. V2 predates exact candidate evidence and
        # assigned-budget persistence; those fields are explicitly unknown,
        # never synthesized. The expanded checkpoint recipe will still veto
        # stale reuse unless the operator requests a rerun.
        value.setdefault("assigned_budget_usd", None)
        for field in (
            "authoring_status",
            "candidate_capture_status",
            "candidate",
            "pristine_replay",
            "verifier_status",
            "verifier_patch",
            "tests_source_identity",
            "tests_snapshot",
            "failure_phase",
            "failure_kind",
        ):
            value.setdefault(field, None)
    elif legacy_schema not in {
        "crucible.deepswe.lifecycle.v3",
        "crucible.deepswe.lifecycle.v4",
        _LIFECYCLE_SCHEMA,
    }:
        raise ValueError(
            f"lifecycle JSONL line {line_number} has unsupported schema"
        )
    value["schema_version"] = _LIFECYCLE_SCHEMA
    # V4 adds Architect terminal provenance and a permanent policy for
    # deterministic planning vetoes. Older rows remain readable and retain
    # their prior reusable/retryable behavior when this provenance is absent.
    for field in (
        "architect_detail_status",
        "architect_decision",
        "architect_terminal_class",
        "architect_failure_phase",
        "architect_failure_kind",
        "architect_tiers_executed",
    ):
        value.setdefault(field, None)
    for field in (
        "verification_rc",
        "verify_returncode",
        *_VERIFIER_OBSERVABILITY_FIELDS,
    ):
        value.setdefault(field, None)
    value.setdefault("zero_tier_settlement_proven", False)
    value.setdefault("secondary_failures", [])
    # V5 adds the per-test counts of a v1.1 reward. Older rows carry none;
    # the counts are explicitly unknown, never derived from the reward.
    value.setdefault("reward_counts", None)
    for field in (
        "task_id", "ext_id", "status", "checkpoint_fingerprint",
    ):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(
                f"lifecycle JSONL line {line_number} requires non-empty "
                f"string {field!r}"
            )
    if value["status"] not in _LIFECYCLE_TERMINAL_STATUSES:
        raise ValueError(
            f"lifecycle JSONL line {line_number} has non-terminal status "
            f"{value['status']!r}"
        )
    status_policy = value.get("status_policy")
    if status_policy not in {"reusable", "retryable", "permanent", "diagnostic_only"}:
        raise ValueError(
            f"lifecycle JSONL line {line_number} has invalid status_policy"
        )
    diagnostic_fields = {"diagnostic_cost_policy", "diagnostic_only", "rankable", "promotable"}
    if diagnostic_fields.intersection(value) or status_policy == "diagnostic_only":
        from ._diagnostic_cost import DIAGNOSTIC_COST_POLICY

        if (
            value.get("diagnostic_cost_policy") != DIAGNOSTIC_COST_POLICY
            or value.get("diagnostic_only") is not True
            or value.get("rankable") is not False
            or value.get("promotable") is not False
            or status_policy != "diagnostic_only"
        ):
            raise ValueError("diagnostic lifecycle authority is missing or contradictory")
    if (
        status_policy == "reusable"
        and value["status"] not in _LIFECYCLE_REUSABLE_STATUSES
    ):
        raise ValueError(
            f"lifecycle JSONL line {line_number} has inconsistent reusable policy"
        )
    tiers_executed = value.get("architect_tiers_executed")
    if tiers_executed is not None and (
        isinstance(tiers_executed, bool)
        or not isinstance(tiers_executed, int)
        or tiers_executed < 0
    ):
        raise ValueError(
            f"lifecycle JSONL line {line_number} has invalid Architect tier count"
        )
    if not isinstance(value.get("zero_tier_settlement_proven"), bool):
        raise ValueError(
            f"lifecycle JSONL line {line_number} has invalid zero-tier proof state"
        )
    secondary_failures = value.get("secondary_failures")
    if (
        not isinstance(secondary_failures, list)
        or not all(
            isinstance(item, dict)
            and isinstance(item.get("phase"), str)
            and bool(item.get("phase"))
            and isinstance(item.get("kind"), str)
            and bool(item.get("kind"))
            and isinstance(item.get("error"), str)
            for item in secondary_failures
        )
    ):
        raise ValueError(
            f"lifecycle JSONL line {line_number} has invalid secondary failures"
        )
    if status_policy == "permanent" and not (
        _is_proven_permanent_preimplementation_terminal(value)
    ):
        raise ValueError(
            f"lifecycle JSONL line {line_number} has unproven permanent policy"
        )
    if value["reward_counts"] is not None:
        try:
            reward_counts = RewardCounts.from_payload(value["reward_counts"])
        except ValueError as exc:
            raise ValueError(
                f"lifecycle JSONL line {line_number} has invalid reward_counts"
            ) from exc
        # Counts exist only beside the binary reward they earn.
        reward = value.get("reward")
        if (
            type(reward) not in (int, float)
            or reward != reward_counts.earned_reward
        ):
            raise ValueError(
                f"lifecycle JSONL line {line_number} reward_counts contradict "
                "its reward"
            )
    try:
        cost_usd = float(value.get("cost_usd"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"lifecycle JSONL line {line_number} has invalid cost_usd"
        ) from exc
    if cost_usd < 0.0 or not math.isfinite(cost_usd):
        raise ValueError(
            f"lifecycle JSONL line {line_number} has invalid cost_usd"
        )
    architect_costs = value.get("architect_costs")
    if architect_costs is not None:
        if (
            type(architect_costs) is not dict
            or not _cost_summary_has_pricing_extension(architect_costs)
        ):
            raise ValueError(
                f"lifecycle JSONL line {line_number} has invalid Architect "
                "cost summary"
            )
        try:
            replayed_cost = _rollup_architect_cost(
                CostAccumulator(), architect_costs,
                task_id=str(value.get("task_id", "")),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"lifecycle JSONL line {line_number} has invalid Architect "
                "cost summary"
            ) from exc
        if not math.isclose(
            replayed_cost, cost_usd, rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError(
                f"lifecycle JSONL line {line_number} Architect costs do not "
                "match cost_usd"
            )
    if not isinstance(value.get("cost_incomplete"), bool):
        raise ValueError(
            f"lifecycle JSONL line {line_number} has invalid cost_incomplete"
        )
    assigned_budget = value.get("assigned_budget_usd")
    if assigned_budget is not None:
        try:
            assigned_budget = float(assigned_budget)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"lifecycle JSONL line {line_number} has invalid "
                "assigned_budget_usd"
            ) from exc
        if assigned_budget <= 0.0 or not math.isfinite(assigned_budget):
            raise ValueError(
                f"lifecycle JSONL line {line_number} has invalid "
                "assigned_budget_usd"
            )
        value["assigned_budget_usd"] = assigned_budget
    return value


def _canonical_lifecycle_bytes(
    records: dict[str, dict[str, Any]],
) -> bytes:
    return b"".join(
        (
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        for record in records.values()
    )


def _fsync_directory(path: Path) -> None:
    """Best-effort durability barrier for a newly created checkpoint."""

    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Not every filesystem supports directory fsync.
        pass


def _read_lifecycle_unlocked(
    out_path: Path,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Read a checkpoint, ignoring only a torn final append.

    Corruption in a newline-terminated record is never skipped. Duplicate
    task ids are collapsed to their last record and reported via the repair
    flag so the caller can publish one canonical line per task.
    """

    if not out_path.exists():
        return {}, False
    payload = out_path.read_bytes()
    repair_needed = bool(payload and not payload.endswith(b"\n"))
    if repair_needed:
        boundary = payload.rfind(b"\n")
        payload = payload[:boundary + 1] if boundary >= 0 else b""

    records: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line.strip():
            raise ValueError(
                f"lifecycle JSONL line {line_number} is empty"
            )
        try:
            decoded = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"lifecycle JSONL line {line_number} is invalid: {exc}"
            ) from exc
        record = _validate_lifecycle_record(
            decoded,
            line_number=line_number,
        )
        task_id = record["task_id"]
        prior = records.get(task_id)
        if prior is not None:
            if prior["ext_id"] != record["ext_id"]:
                raise ValueError(
                    f"lifecycle task {task_id!r} has conflicting ext_id values"
                )
            repair_needed = True
        records[task_id] = record
    return records, repair_needed


@contextmanager
def _lifecycle_lock(out_path: Path) -> Iterator[None]:
    """Serialize readers, appenders, and repair replacements by sidecar."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_name(f".{out_path.name}.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _load_lifecycle_jsonl(out_path: Path) -> dict[str, dict[str, Any]]:
    """Load and repair the durable checkpoint under its writer lock."""

    with _lifecycle_lock(out_path):
        records, repair_needed = _read_lifecycle_unlocked(out_path)
        if repair_needed:
            write_atomic_bytes(
                out_path,
                _canonical_lifecycle_bytes(records),
            )
        return records


def _append_lifecycle_record(
    *,
    out_path: Path,
    summary: dict[str, Any],
) -> bool:
    """Durably upsert one task checkpoint.

    New tasks use ``O_APPEND`` under an advisory sidecar lock and are fsynced
    before returning. A prior duplicate is either a no-op (identical record)
    or replaced through one atomic canonical rewrite. A torn final write is
    truncated during the same locked transaction.

    Returns ``True`` when the file changed.
    """

    record = _validate_lifecycle_record(
        _lifecycle_record(summary),
        line_number=1,
    )
    with _lifecycle_lock(out_path):
        records, repair_needed = _read_lifecycle_unlocked(out_path)
        prior = records.get(record["task_id"])
        if prior is not None:
            if prior["ext_id"] != record["ext_id"]:
                raise ValueError(
                    f"lifecycle task {record['task_id']!r} has conflicting "
                    "ext_id values"
                )
            if prior == record and not repair_needed:
                return False
            records[record["task_id"]] = record
            write_atomic_bytes(
                out_path,
                _canonical_lifecycle_bytes(records),
            )
            return True

        if repair_needed:
            write_atomic_bytes(
                out_path,
                _canonical_lifecycle_bytes(records),
            )

        line = (
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        fd = os.open(
            out_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            remaining = memoryview(line)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("DeepSWE lifecycle append made no progress")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_directory(out_path.parent)
        return True


def _emit_lifecycle_jsonl(
    *,
    out_path: Path,
    task_summaries: list[dict[str, Any]],
) -> None:
    """Durably upsert lifecycle records with one canonical row per task.

    This is a lifecycle checkpoint, not a leaderboard predictions file. Each
    task is published as soon as it completes so process death cannot erase
    the completed prefix; a later invocation resumes from these records.
    """

    for summary in task_summaries:
        _append_lifecycle_record(out_path=out_path, summary=summary)


_TaskExecutionLeaseBusy = TaskLeaseBusy


class _LifecycleIdentityConflict(ValueError):
    """A checkpoint task id resolves to a different dataset identity."""


class _LifecycleCheckpointError(RuntimeError):
    """A lifecycle checkpoint could not be read or durably published."""


def _load_candidate_measurement_evidence(
    corpus: Corpus,
    measurement: Measurement,
    inst: DeepSWEInstance,
) -> tuple[dict[str, Any], DeepSWECandidate, dict[str, Any]]:
    """Load and validate one candidate Measurement without inventing proof."""

    expected_bench = f"deepswe/{inst.task_id}/candidate"
    if (
        measurement.bench_id != expected_bench
        or measurement.commit != inst.base_commit_hash
    ):
        raise _LifecycleCheckpointError(
            f"candidate measurement identity invalid for {inst.task_id}"
        )
    try:
        payload = json.loads(
            corpus.get_artifact(measurement.raw_output_ref).decode("utf-8")
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("kind") != "deepswe_candidate"
            or payload.get("task_id") != inst.task_id
            or payload.get("ext_id") != inst.ext_id
        ):
            raise ValueError("candidate envelope identity/schema invalid")
        candidate_payload = payload["candidate"]
        candidate_blob = corpus.get_artifact(candidate_payload["diff_ref"])
        candidate = DeepSWECandidate.from_payload(
            candidate_payload,
            patch=candidate_blob,
        )
        checkpoint = _validate_lifecycle_record(
            payload.get("checkpoint"),
            line_number=1,
        )
        if (
            checkpoint["task_id"] != inst.task_id
            or checkpoint["ext_id"] != inst.ext_id
        ):
            raise ValueError("candidate checkpoint identity mismatch")
    except _LifecycleCheckpointError:
        raise
    except Exception as exc:
        raise _LifecycleCheckpointError(
            f"candidate measurement evidence invalid for {inst.task_id}: {exc}"
        ) from exc
    return payload, candidate, checkpoint


def _load_hidden_oracle_checkpoint(
    spec_root: Path,
    inst: DeepSWEInstance,
) -> dict[str, Any] | None:
    """Recover the checkpoint atomically committed with hidden evidence.

    The per-task Corpus is the authority for the crash window between the
    one-shot verifier commit and lifecycle JSONL publication. Once a matching
    hidden measurement is visible, malformed or missing checkpoint metadata
    fails closed instead of re-running the oracle.
    """

    corpus_dir = Path(spec_root) / "instances" / inst.task_id / "corpus"
    if not corpus_dir.is_dir():
        return None
    try:
        corpus = Corpus(corpus_dir)
        latest = corpus.latest_spec()
    except Exception as exc:
        raise _LifecycleCheckpointError(
            f"cannot read hidden-oracle Corpus for {inst.task_id}: {exc}"
        ) from exc
    if latest is None:
        return None

    hidden_bench_id = f"deepswe/{inst.task_id}/hidden_verifier"
    invocation_bench_id = f"deepswe/{inst.task_id}/hidden_invocation"
    candidate_bench_id = f"deepswe/{inst.task_id}/candidate"
    refs = latest.measurement_refs
    for index in range(len(refs) - 1, -1, -1):
        ref = refs[index]
        try:
            measurement = corpus.get_measurement(ref)
        except Exception as exc:
            raise _LifecycleCheckpointError(
                f"cannot read hidden-oracle measurement for {inst.task_id}: {exc}"
            ) from exc
        if measurement is None or measurement.bench_id not in {
            hidden_bench_id,
            invocation_bench_id,
            candidate_bench_id,
        }:
            continue
        if measurement.bench_id == candidate_bench_id:
            # A newer candidate-only terminal supersedes every older oracle.
            # This is the durable marker produced when a rerun captured a
            # candidate but failed before authoritative hidden verification.
            candidate_payload, _candidate, _checkpoint = (
                _load_candidate_measurement_evidence(corpus, measurement, inst)
            )
            if candidate_payload.get("verifier_patch") is None:
                return None
            raise _LifecycleCheckpointError(
                "candidate claims verifier evidence without its atomically "
                f"paired hidden measurement for {inst.task_id}"
            )

        if measurement.bench_id == invocation_bench_id:
            try:
                payload = json.loads(
                    corpus.get_artifact(measurement.raw_output_ref).decode(
                        "utf-8"
                    )
                )
                if (
                    measurement.commit != inst.base_commit_hash
                    or not isinstance(payload, dict)
                    or payload.get("schema_version") != 1
                    or payload.get("kind") != "deepswe_hidden_invocation"
                    or payload.get("state") != "started"
                    or payload.get("task_id") != inst.task_id
                    or payload.get("ext_id") != inst.ext_id
                ):
                    raise ValueError("hidden invocation identity/schema invalid")
                if index == 0:
                    raise ValueError("hidden invocation has no paired candidate")
                paired_measurement = corpus.get_measurement(refs[index - 1])
                if (
                    paired_measurement is None
                    or paired_measurement.bench_id != candidate_bench_id
                ):
                    raise ValueError(
                        "hidden invocation is not adjacent to its candidate"
                    )
                (
                    candidate_envelope,
                    candidate,
                    paired_checkpoint,
                ) = _load_candidate_measurement_evidence(
                    corpus,
                    paired_measurement,
                    inst,
                )
                checkpoint = _validate_lifecycle_record(
                    payload.get("checkpoint"),
                    line_number=1,
                )
                pristine_payload = payload.get("pristine_replay")
                tests_snapshot_payload = payload.get("tests_snapshot")
                evidence_valid = (
                    checkpoint == paired_checkpoint
                    and checkpoint["status"] == "verification_incomplete"
                    and candidate_envelope.get("candidate")
                    == payload.get("candidate")
                    and candidate_envelope.get("checkpoint")
                    == payload.get("checkpoint")
                    and candidate_envelope.get("pristine_replay")
                    == pristine_payload
                    and candidate_envelope.get("verifier_patch") is None
                    and tests_snapshot_payload
                    == checkpoint.get("tests_snapshot")
                    == checkpoint.get("tests_source_identity")
                    and isinstance(tests_snapshot_payload, dict)
                    and tests_snapshot_payload.get("schema")
                    == "crucible.deepswe.tests-tree.v1"
                    and isinstance(pristine_payload, dict)
                    and pristine_payload.get("schema")
                    == "crucible.deepswe.pristine-replay.v1"
                    and pristine_payload.get("status")
                    == PristineReplayStatus.PASSED.value
                    and pristine_payload.get("candidate_sha256")
                    == candidate.sha256
                    and pristine_payload.get("observed_sha256")
                    == candidate.sha256
                )
                if not evidence_valid:
                    raise ValueError(
                        "hidden invocation is not bound to candidate bytes"
                    )
            except Exception as exc:
                raise _LifecycleCheckpointError(
                    f"hidden invocation evidence invalid for {inst.task_id}: "
                    f"{exc}"
                ) from exc
            return {
                **checkpoint,
                "candidate": candidate.to_payload(),
                "pristine_replay": pristine_payload,
                "verifier_patch": None,
                "hidden_invocation_state": "started",
            }

        try:
            payload = json.loads(
                corpus.get_artifact(measurement.raw_output_ref).decode("utf-8")
            )
        except Exception as exc:
            raise _LifecycleCheckpointError(
                f"cannot read hidden-oracle checkpoint for {inst.task_id}: {exc}"
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != "deepswe_hidden_verifier"
            or payload.get("task_id") != inst.task_id
            or payload.get("ext_id") != inst.ext_id
        ):
            raise _LifecycleCheckpointError(
                f"hidden-oracle checkpoint identity/schema invalid for {inst.task_id}"
            )
        # Legacy schema-v2 results predate paired candidate evidence. They are
        # deliberately non-reusable rather than fatal: --rerun and upgraded
        # workers can supersede them, but no reward is silently trusted.
        if payload.get("schema_version") == 2:
            return None
        if payload.get("schema_version") != 3:
            raise _LifecycleCheckpointError(
                f"hidden-oracle checkpoint identity/schema invalid for {inst.task_id}"
            )
        if index == 0:
            raise _LifecycleCheckpointError(
                f"hidden oracle has no paired candidate for {inst.task_id}"
            )
        try:
            candidate_measurement = corpus.get_measurement(refs[index - 1])
        except Exception as exc:
            raise _LifecycleCheckpointError(
                f"cannot read paired candidate for {inst.task_id}: {exc}"
            ) from exc
        if (
            candidate_measurement is None
            or candidate_measurement.bench_id != candidate_bench_id
        ):
            raise _LifecycleCheckpointError(
                f"hidden oracle is not adjacent to its candidate for {inst.task_id}"
            )
        (
            candidate_envelope,
            paired_candidate,
            paired_checkpoint,
        ) = _load_candidate_measurement_evidence(
            corpus,
            candidate_measurement,
            inst,
        )
        try:
            checkpoint = _validate_lifecycle_record(
                payload.get("checkpoint"),
                line_number=1,
            )
        except (TypeError, ValueError) as exc:
            raise _LifecycleCheckpointError(
                f"hidden-oracle checkpoint invalid for {inst.task_id}: {exc}"
            ) from exc
        if (
            checkpoint["task_id"] != inst.task_id
            or checkpoint["ext_id"] != inst.ext_id
            or checkpoint["status"] != payload.get("status")
        ):
            raise _LifecycleCheckpointError(
                f"hidden-oracle checkpoint content mismatch for {inst.task_id}"
            )
        try:
            candidate_payload = payload["candidate"]
            candidate_ref = candidate_payload["diff_ref"]
            candidate_blob = corpus.get_artifact(candidate_ref)
            candidate = DeepSWECandidate.from_payload(
                candidate_payload,
                patch=candidate_blob,
            )
            pristine_payload = payload["pristine_replay"]
            verifier_patch_payload = payload["verifier_patch"]
            candidate_bound = (
                isinstance(pristine_payload, dict)
                and pristine_payload.get("schema")
                == "crucible.deepswe.pristine-replay.v1"
                and pristine_payload.get("status")
                == PristineReplayStatus.PASSED.value
                and pristine_payload.get("candidate_sha256")
                == candidate.sha256
                and pristine_payload.get("observed_sha256")
                == candidate.sha256
                and isinstance(verifier_patch_payload, dict)
                and verifier_patch_payload.get("schema")
                == "crucible.deepswe.verifier-patch.v1"
                and verifier_patch_payload.get("status")
                == VerifierPatchStatus.MATCHED.value
                and verifier_patch_payload.get("candidate_sha256")
                == candidate.sha256
                and verifier_patch_payload.get("observed_sha256")
                == candidate.sha256
                and verifier_patch_payload.get("observed_bytes")
                == candidate.diff_bytes
            )
            evidence_malformed = (
                not isinstance(pristine_payload, dict)
                or not isinstance(verifier_patch_payload, dict)
                or pristine_payload.get("schema")
                != "crucible.deepswe.pristine-replay.v1"
                or verifier_patch_payload.get("schema")
                != "crucible.deepswe.verifier-patch.v1"
                or verifier_patch_payload.get("candidate_sha256")
                != candidate.sha256
                or pristine_payload.get("candidate_sha256")
                != candidate.sha256
            )
            paired_evidence_mismatch = (
                candidate.to_payload() != paired_candidate.to_payload()
                or candidate.patch != paired_candidate.patch
                or checkpoint != paired_checkpoint
                or candidate_envelope.get("candidate") != candidate_payload
                or candidate_envelope.get("checkpoint")
                != payload.get("checkpoint")
                or candidate_envelope.get("pristine_replay")
                != pristine_payload
                or candidate_envelope.get("verifier_patch")
                != verifier_patch_payload
            )
            if evidence_malformed or paired_evidence_mismatch or (
                checkpoint["status"] in {"resolved", "unresolved"}
                and not candidate_bound
            ):
                raise ValueError(
                    "hidden verifier is not bound to paired candidate bytes"
                )
        except Exception as exc:
            raise _LifecycleCheckpointError(
                "hidden-oracle candidate evidence invalid for "
                f"{inst.task_id}: {exc}"
            ) from exc
        checkpoint = {
            **checkpoint,
            "candidate": candidate.to_payload(),
            "pristine_replay": pristine_payload,
            "verifier_patch": verifier_patch_payload,
        }
        return checkpoint
    return None


@contextmanager
def _task_execution_lease(spec_dir: Path, task_id: str) -> Iterator[Path]:
    """Acquire one nonblocking cross-process lease for ``task_id``.

    The hashed filename prevents task ids from escaping the private lease
    directory. ``flock`` is released automatically on process death.
    """

    with task_execution_lease(
        spec_dir,
        namespace="deepswe",
        task_id=task_id,
    ) as lease_path:
        yield lease_path


def _execute_task_with_checkpoint(
    inst: DeepSWEInstance,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    cost: CostAccumulator,
    child_budget_usd: float | None,
    handle,
    index: int,
) -> tuple[dict[str, Any], bool]:
    """Execute one task while holding its lease through checkpoint fsync.

    Returns ``(summary, resumed)``. The checkpoint is re-read only after the
    lease is acquired, closing the load-before-lock race between workers.
    """

    with _task_execution_lease(args.spec, inst.task_id):
        oracle_checkpoint = _load_hidden_oracle_checkpoint(args.spec, inst)
        if oracle_checkpoint is not None and (
            getattr(args, "diagnostic_cost_policy", None) is not None
            or oracle_checkpoint.get("diagnostic_only") is True
        ):
            raise _LifecycleCheckpointError("diagnostic upper-bound pilot cannot reuse prior oracle authority")
        if args.lifecycle_out is not None:
            try:
                current = _load_lifecycle_jsonl(args.lifecycle_out).get(
                    inst.task_id
                )
            except (OSError, ValueError) as exc:
                raise _LifecycleCheckpointError(str(exc)) from exc
            if current is not None:
                if (
                    getattr(args, "diagnostic_cost_policy", None) is not None
                    or current.get("diagnostic_only") is True
                ):
                    raise _LifecycleCheckpointError("diagnostic upper-bound pilot cannot resume a prior task")
                if current["ext_id"] != inst.ext_id:
                    raise _LifecycleIdentityConflict(
                        f"checkpoint task {inst.task_id!r} ext_id="
                        f"{current['ext_id']!r} conflicts with dataset "
                        f"ext_id={inst.ext_id!r}"
                    )
                prior_inst = dataclasses.replace(
                    inst,
                    docker_image=(current.get("image_digest") or inst.docker_image),
                )
                prior_fingerprint = _deepswe_checkpoint_fingerprint(
                    prior_inst,
                    args=args,
                    child_budget_usd=current.get(
                        "assigned_budget_usd", child_budget_usd
                    ),
                )
                explicitly_rerun_current = (
                    args.rerun
                    or current["status"] in set(args.rerun_status)
                )
                if current["checkpoint_fingerprint"] != prior_fingerprint:
                    # Hidden-oracle results are one-shot and require explicit
                    # replacement authority when their recipe identity drifts.
                    # A permanent pre-implementation planning terminal has no
                    # oracle/candidate authority; a changed recipe may safely
                    # run it again, while an identical recipe is reused below.
                    if (
                        current["status"] in _LIFECYCLE_REUSABLE_STATUSES
                        and not explicitly_rerun_current
                    ):
                        raise _LifecycleIdentityConflict(
                            f"checkpoint task {inst.task_id!r} fingerprint "
                            "does not match dataset/policy/runtime inputs; "
                            "pass --rerun to replace it"
                        )
                elif _checkpoint_is_reusable(current, args=args):
                    oracle_bound_status = current["status"] in {
                        "verification_failed",
                        "verification_incomplete",
                    }
                    if (oracle_bound_status or current.get("candidate") is not None) and (
                        oracle_checkpoint is None
                        or oracle_checkpoint.get("checkpoint_fingerprint")
                        != current["checkpoint_fingerprint"]
                        or oracle_checkpoint.get("status") != current["status"]
                        or oracle_checkpoint.get("candidate")
                        != current.get("candidate")
                    ):
                        raise _LifecycleCheckpointError(
                            "lifecycle candidate is not backed by matching "
                            f"atomic hidden-oracle evidence for {inst.task_id}"
                        )
                    try:
                        verify_immutable_image(str(current.get("image_digest") or ""))
                    except ImmutableImageError as exc:
                        raise _LifecycleIdentityConflict(
                            "saved DeepSWE image no longer passes the current "
                            f"verification policy: {exc}"
                        ) from exc
                    _restore_checkpoint_cost(cost, current, task_id=inst.task_id)
                    return {**current, "resumed": True}, True

        if oracle_checkpoint is not None:
            oracle_inst = dataclasses.replace(
                inst,
                docker_image=(
                    oracle_checkpoint.get("image_digest") or inst.docker_image
                ),
            )
            oracle_fingerprint = _deepswe_checkpoint_fingerprint(
                oracle_inst,
                args=args,
                child_budget_usd=oracle_checkpoint.get(
                    "assigned_budget_usd", child_budget_usd
                ),
            )
            explicitly_rerun = (
                args.rerun
                or oracle_checkpoint["status"] in set(args.rerun_status)
            )
            # Corpus oracle checkpoints exist only after candidate/hidden
            # publication. Permanent zero-tier planning terminals deliberately
            # publish neither and are handled by the lifecycle policy above.
            oracle_is_authoritative = (
                oracle_checkpoint["status"] in _LIFECYCLE_REUSABLE_STATUSES
                and not explicitly_rerun
            )
            if (
                oracle_is_authoritative
                and oracle_checkpoint["checkpoint_fingerprint"]
                != oracle_fingerprint
            ):
                raise _LifecycleIdentityConflict(
                    f"hidden-oracle checkpoint task {inst.task_id!r} "
                    "does not match dataset/policy/runtime inputs; pass "
                    "--rerun to replace it"
                )
            if oracle_is_authoritative:
                try:
                    verify_immutable_image(
                        str(oracle_checkpoint.get("image_digest") or "")
                    )
                except ImmutableImageError as exc:
                    raise _LifecycleIdentityConflict(
                        "saved DeepSWE hidden-oracle image no longer passes "
                        f"the current verification policy: {exc}"
                    ) from exc
                if args.lifecycle_out is not None:
                    try:
                        _append_lifecycle_record(
                            out_path=args.lifecycle_out,
                            summary=oracle_checkpoint,
                        )
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        raise _LifecycleCheckpointError(str(exc)) from exc
                _restore_checkpoint_cost(
                    cost,
                    oracle_checkpoint,
                    task_id=inst.task_id,
                )
                return {**oracle_checkpoint, "resumed": True}, True

        handle.write_event({
            "phase": "task_start",
            "task_id": inst.task_id,
            "index": index,
            "budget_usd": child_budget_usd,
        })
        try:
            task_summary = _run_task_cascade(
                inst,
                args=args,
                run_dir=run_dir,
                cost=cost,
                child_budget_usd=child_budget_usd,
            )
        except _HiddenOraclePersistenceError as exc:
            raise _LifecycleCheckpointError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — preserve outer terminal
            logger.exception(
                "deepswe: task %s escaped its failure boundary: %s",
                inst.task_id, exc,
            )
            task_summary = {
                "task_id": inst.task_id,
                "ext_id": inst.ext_id,
                "repo": inst.repo,
                "language": inst.language,
                "category": inst.category,
                "status": "cascade_error",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "cost_usd": 0.0,
                "cost_incomplete": True,
                "budget_usd": child_budget_usd,
            }
        immutable_image = str(task_summary.get("image_digest") or inst.docker_image)
        runtime_inst = dataclasses.replace(inst, docker_image=immutable_image)
        checkpoint_fingerprint = task_summary.get("checkpoint_fingerprint")
        if not isinstance(checkpoint_fingerprint, str):
            checkpoint_fingerprint = _deepswe_checkpoint_fingerprint(
                runtime_inst,
                args=args,
                child_budget_usd=child_budget_usd,
            )
        task_summary["checkpoint_fingerprint"] = checkpoint_fingerprint
        task_summary["image_digest"] = immutable_image
        if args.lifecycle_out is not None:
            try:
                _append_lifecycle_record(
                    out_path=args.lifecycle_out,
                    summary=task_summary,
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise _LifecycleCheckpointError(str(exc)) from exc
        return task_summary, False


def _implement_serial(
    filtered: tuple[DeepSWEInstance, ...],
    *,
    args: argparse.Namespace,
    handle=None,
    corpus: Corpus | None = None,
) -> int:
    """Serial per-task cascade loop. Per-task failures are recorded
    + skipped; the loop runs to completion across all filtered tasks
    so one bad task doesn't abort the sweep."""
    if handle is None:
        routing_snapshot = load_routing(
            config_path=args.backend_config,
            spec_dir=args.spec,
        )
        corpus = corpus or _open_or_create_corpus(args.spec)
        with open_run_with_capture(
            corpus,
            "deepswe",
            provenance_root=None,
        ) as opened:
            _outcome.record_provenance_inputs(
                opened.run_dir,
                backend_routing=routing_fingerprint(routing_snapshot),
                backend_features=routing_feature_fingerprint(
                    routing_snapshot,
                ),
                backend_routing_source_sha256=(
                    routing_source_fingerprint(routing_snapshot)
                ),
            )
            with bind_routing_snapshot(routing_snapshot):
                return _implement_serial(
                    filtered,
                    args=args,
                    handle=opened,
                    corpus=corpus,
                )

    run_dir = handle.run_dir
    cost = CostAccumulator()
    task_summaries: list[dict[str, Any]] = []
    halt_reason: str | None = None
    completed: dict[str, dict[str, Any]] = {}
    if args.lifecycle_out is not None:
        try:
            completed = _load_lifecycle_jsonl(args.lifecycle_out)
        except (OSError, ValueError) as exc:
            logger.error("deepswe: lifecycle checkpoint is invalid: %s", exc)
            status = "lifecycle_output_invalid"
            handle.write_event({
                "phase": "terminal",
                "status": status,
                "halt_reason": status,
                "error": str(exc)[:2000],
            })
            _emit_summary(
                handle=handle,
                args=args,
                task_summaries=[],
                cost=cost,
                status=status,
                halt_reason=status,
                tasks_selected=len(filtered),
                tasks_not_started=len(filtered),
            )
            handle.finalize(RunResult(
                status=status,
                duration_s=handle.elapsed_s,
                error_kind=type(exc).__name__,
                error_message=str(exc)[:2000],
            ))
            return 1

    # Tasks handed to execution. A handed task normally ends with a task
    # summary; a lifecycle conflict, checkpoint failure, or lease error ends
    # it without one, possibly after its cascade and hidden verifier ran.
    tasks_handed = 0
    for i, inst in enumerate(filtered, 1):
        logger.info(
            "deepswe: [%d/%d] task=%s repo=%s lang=%s",
            i, len(filtered), inst.task_id, inst.repo, inst.language,
        )
        if args.budget_usd is not None and cost.total_cost_usd >= args.budget_usd:
            logger.warning(
                "deepswe: aggregate budget $%.2f reached at task %d/%d — "
                "halting before remaining %d tasks",
                args.budget_usd, i - 1, len(filtered),
                len(filtered) - i + 1,
            )
            halt_reason = "aggregate_budget_reached"
            handle.write_event({
                "phase": "halt",
                "reason": halt_reason,
                "spent_usd": cost.total_cost_usd,
                "budget_usd": args.budget_usd,
            })
            break
        child_budget = _resolve_child_budget(
            args.budget_usd_per_instance,
            args.budget_usd,
            cost.total_cost_usd,
        )
        tasks_handed += 1
        try:
            task_summary, resumed_after_lease = _execute_task_with_checkpoint(
                inst,
                args=args,
                run_dir=run_dir,
                cost=cost,
                child_budget_usd=child_budget,
                handle=handle,
                index=i,
            )
        except _TaskExecutionLeaseBusy as exc:
            halt_reason = "task_lease_busy"
            task_summary = {
                "task_id": inst.task_id,
                "ext_id": inst.ext_id,
                "repo": inst.repo,
                "language": inst.language,
                "category": inst.category,
                "status": halt_reason,
                "error": str(exc)[:500],
                "cost_usd": 0.0,
                "cost_incomplete": False,
                "budget_usd": child_budget,
                "cleanup_status": "not_run",
            }
            task_summaries.append(task_summary)
            handle.write_event({
                "phase": "halt",
                "reason": halt_reason,
                "task_id": inst.task_id,
                "error": str(exc)[:2000],
            })
            logger.warning("deepswe: %s", exc)
            break
        except _LifecycleIdentityConflict as exc:
            halt_reason = "lifecycle_identity_conflict"
            handle.write_event({
                "phase": "halt",
                "reason": halt_reason,
                "task_id": inst.task_id,
                "error": str(exc)[:2000],
            })
            logger.error("deepswe: %s", exc)
            break
        except _LifecycleCheckpointError as exc:
            halt_reason = "lifecycle_output_failed"
            handle.write_event({
                "phase": "halt",
                "reason": halt_reason,
                "task_id": inst.task_id,
                "error": str(exc)[:2000],
            })
            logger.error(
                "deepswe: lifecycle checkpoint failed for task %s: %s",
                inst.task_id, exc,
            )
            break
        except OSError as exc:
            halt_reason = "task_lease_error"
            handle.write_event({
                "phase": "halt",
                "reason": halt_reason,
                "task_id": inst.task_id,
                "error": str(exc)[:2000],
            })
            logger.error(
                "deepswe: task lease failed for task %s: %s",
                inst.task_id, exc,
            )
            break

        task_summaries.append(task_summary)
        if resumed_after_lease:
            completed[inst.task_id] = task_summary
            handle.write_event({
                "phase": "task_resumed",
                "task_id": inst.task_id,
                "index": i,
                "status": task_summary["status"],
            })
            logger.info(
                "deepswe: task %s completed by the prior lease owner as %s — "
                "skipping",
                inst.task_id, task_summary["status"],
            )
            if task_summary.get("cost_incomplete"):
                logger.error(
                    "deepswe: resumed task %s has incomplete durable cost "
                    "accounting; preserving its oracle result but halting",
                    inst.task_id,
                )
                halt_reason = "cost_incomplete"
                break
            if (
                args.budget_usd is not None
                and cost.total_cost_usd > args.budget_usd
            ):
                halt_reason = "aggregate_budget_exceeded"
                handle.write_event({
                    "phase": "halt",
                    "reason": halt_reason,
                    "task_id": inst.task_id,
                    "spent_usd": cost.total_cost_usd,
                    "budget_usd": args.budget_usd,
                })
                logger.error(
                    "deepswe: restored task %s pushes durable aggregate spend "
                    "over budget ($%.4f > $%.4f)",
                    inst.task_id,
                    cost.total_cost_usd,
                    args.budget_usd,
                )
                break
            if (
                args.budget_usd is not None
                and cost.total_cost_usd >= args.budget_usd
                and i < len(filtered)
            ):
                halt_reason = "aggregate_budget_reached"
                handle.write_event({
                    "phase": "halt",
                    "reason": halt_reason,
                    "task_id": inst.task_id,
                    "spent_usd": cost.total_cost_usd,
                    "budget_usd": args.budget_usd,
                })
                break
            continue
        handle.write_event({
            "phase": "task_complete",
            "task_id": inst.task_id,
            "status": task_summary.get("status"),
            "cost_usd": task_summary.get("cost_usd", 0.0),
            "cost_incomplete": bool(task_summary.get("cost_incomplete")),
        })
        if task_summary.get("cost_incomplete"):
            logger.error(
                "deepswe: halting after task %s because its cost rollup is "
                "incomplete; continuing could exceed the aggregate budget",
                inst.task_id,
            )
            halt_reason = "cost_incomplete"
            break
        if (
            args.budget_usd is not None
            and cost.total_cost_usd > args.budget_usd
        ):
            logger.error(
                "deepswe: task %s overshot aggregate budget "
                "($%.4f > $%.4f); finalizing non-successfully",
                inst.task_id, cost.total_cost_usd, args.budget_usd,
            )
            halt_reason = "aggregate_budget_exceeded"
            break

    failed_tasks = any(
        summary.get("status") != "resolved"
        for summary in task_summaries
    )
    cleanup_failed = any(
        summary.get("cleanup_status") == "failed"
        for summary in task_summaries
    )
    if halt_reason in {
        "lifecycle_identity_conflict",
        "lifecycle_output_failed",
        "task_lease_busy",
        "task_lease_error",
    }:
        status = halt_reason
    elif halt_reason == "cost_incomplete":
        status = "cost_incomplete"
    elif halt_reason is not None:
        status = "budget_exhausted"
    elif failed_tasks:
        if _all_task_failures_are_retryable(task_summaries):
            status = "retryable_tasks_failed"
        elif _all_task_failures_are_unjudged(task_summaries):
            status = "inconclusive_tasks_failed"
        else:
            status = "tasks_failed"
    elif cleanup_failed:
        status = "cleanup_failed"
    else:
        status = "deepswe_complete"
    rc = 1 if (halt_reason is not None or failed_tasks or cleanup_failed) else 0
    if getattr(args, "diagnostic_cost_policy", None) is not None:
        if status == "deepswe_complete":
            status = "diagnostic_completed"
        rc = 1

    handle.write_event({
        "phase": "terminal",
        "status": status,
        "halt_reason": halt_reason,
        "tasks_completed": len(task_summaries),
        "cost_usd": cost.total_cost_usd,
    })
    summary_path = _emit_summary(
        handle=handle,
        args=args,
        task_summaries=task_summaries,
        cost=cost,
        status=status,
        halt_reason=halt_reason,
        tasks_selected=len(filtered),
        tasks_not_started=len(filtered) - tasks_handed,
    )
    logger.info("deepswe: summary written to %s", summary_path)
    handle.finalize(RunResult(
        status=status,
        duration_s=handle.elapsed_s,
        cost_usd=cost.total_cost_usd,
    ))
    return rc


def _finalize_empty_run(
    handle,
    *,
    args: argparse.Namespace,
    status: str,
    rc: int,
    tasks_selected: int,
    error: Exception | str | None = None,
) -> int:
    """Publish summary, event, and result for an early implement terminal.

    ``tasks_selected`` is the filtered selection when the terminal is reached
    after filtering, and 0 when no selection exists yet. None of its tasks
    was started.
    """

    error_text = None if error is None else str(error)[:2000]
    _emit_summary(
        handle=handle,
        args=args,
        task_summaries=[],
        cost=CostAccumulator(),
        status=status,
        halt_reason=status,
        tasks_selected=tasks_selected,
        tasks_not_started=tasks_selected,
    )
    handle.write_event({
        "phase": "terminal",
        "status": status,
        "error": error_text,
    })
    handle.finalize(RunResult(
        status=status,
        duration_s=handle.elapsed_s,
        error_kind=(
            type(error).__name__
            if isinstance(error, Exception) else None
        ),
        error_message=error_text,
    ))
    return rc


def _run_implement_mode(args: argparse.Namespace) -> int:
    """Run the full implement mode inside one transactional RunHandle."""

    corpus = _open_or_create_corpus(args.spec)
    with open_run_with_capture(
        corpus,
        "deepswe",
        provenance_root=None,
    ) as handle:
        try:
            filtered = _load_and_filter(args)
        except FileNotFoundError as exc:
            logger.error("deepswe: dataset path missing: %s", exc)
            return _finalize_empty_run(
                handle,
                args=args,
                status="dataset_missing",
                rc=2,
                tasks_selected=0,
                error=exc,
            )
        except DeepSWEInstanceError as exc:
            logger.error("deepswe: dataset validation failed: %s", exc)
            return _finalize_empty_run(
                handle,
                args=args,
                status="dataset_invalid",
                rc=1,
                tasks_selected=0,
                error=exc,
            )

        if getattr(args, "diagnostic_cost_policy", None) is not None and len(filtered) != 1:
            return _finalize_empty_run(
                handle, args=args, status="diagnostic_task_shape_unsupported",
                rc=2, tasks_selected=len(filtered),
                error="diagnostic upper-bound pilot requires exactly one fresh task",
            )

        explicit_filters = bool(
            args.task_id or args.repo or args.language
            or args.category or args.limit
        )
        if not filtered and explicit_filters:
            message = (
                "no tasks matched filters "
                f"(task_id={args.task_id}, repo={args.repo}, "
                f"language={args.language}, category={args.category}, "
                f"limit={args.limit})"
            )
            logger.error("deepswe: %s", message)
            return _finalize_empty_run(
                handle,
                args=args,
                status="no_tasks_matched",
                rc=1,
                tasks_selected=0,
                error=message,
            )
        if not filtered:
            logger.warning("deepswe: dataset %s contains zero tasks", args.dataset)
            return _finalize_empty_run(
                handle,
                args=args,
                status="no_work",
                rc=1,
                tasks_selected=0,
                error=f"dataset {args.dataset} contains zero tasks",
            )

        try:
            routing_snapshot = load_routing(
                config_path=args.backend_config,
                spec_dir=args.spec,
            )
            args.routing_fingerprint = routing_fingerprint(routing_snapshot)
            _outcome.record_provenance_inputs(
                handle.run_dir,
                backend_routing=args.routing_fingerprint,
                backend_features=routing_feature_fingerprint(
                    routing_snapshot,
                ),
                backend_routing_source_sha256=(
                    routing_source_fingerprint(routing_snapshot)
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.error("deepswe: backend routing is invalid: %s", exc)
            return _finalize_empty_run(
                handle,
                args=args,
                status="setup_error",
                rc=2,
                tasks_selected=len(filtered),
                error=exc,
            )

        # --implement path: docker required + disk-headroom gated. These
        # handled terminals still publish summary.json + result.json.
        if not docker_available():
            logger.error(
                "deepswe: docker daemon not reachable. The cascade is "
                "docker-mode-only for now (no venv fallback). Bring up "
                "docker and retry."
            )
            return _finalize_empty_run(
                handle=handle,
                args=args,
                status="container_runtime_required",
                rc=2,
                tasks_selected=len(filtered),
                error="Docker is unavailable",
            )
        try:
            removed = reconcile_orphans(str(Path(args.spec).resolve()))
            if removed:
                logger.warning(
                    "deepswe: reconciled %d orphan container(s)", len(removed),
                )
            if not args.skip_disk_check:
                check_disk_headroom()
        except InstanceContainerSetupError as e:
            logger.error("deepswe: %s", e)
            return _finalize_empty_run(
                handle=handle,
                args=args,
                status="container_preflight_failed",
                rc=2,
                tasks_selected=len(filtered),
                error=e,
            )

        with bind_routing_snapshot(routing_snapshot):
            return _implement_serial(
                filtered,
                args=args,
                handle=handle,
                corpus=corpus,
            )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.implement:
        return _run_implement_mode(args)

    try:
        filtered = _load_and_filter(args)
    except FileNotFoundError as e:
        logger.error("deepswe: dataset path missing: %s", e)
        return 2
    except DeepSWEInstanceError as e:
        logger.error("deepswe: dataset validation failed: %s", e)
        return 1

    explicit_filters = bool(
        args.task_id or args.repo or args.language
        or args.category or args.limit
    )
    if not filtered:
        if explicit_filters:
            logger.error(
                "deepswe: no tasks matched filters (task_id=%s, repo=%s, "
                "language=%s, category=%s, limit=%s)",
                args.task_id, args.repo, args.language, args.category,
                args.limit,
            )
            return 1
        logger.warning(
            "deepswe: dataset %s contains zero tasks", args.dataset,
        )

    if args.list_only:
        return _list_mode(filtered)
    if args.dry_run:
        return _dry_run_mode(filtered)
    logger.error("deepswe: one of --list / --dry-run / --implement required.")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
