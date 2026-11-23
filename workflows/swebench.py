# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""SWE-bench Verified adapter — run the full Crucible cascade per instance.

Each live instance gets a standalone, remote-free checkout at its exact
``base_commit`` and a mandatory immutable Docker authoring environment. The
workflow synthesizes an instance-local Spec, invokes ``run_architect``, and
thereby composes plan → tier → implement → optional breach/review. It captures
the cumulative workspace diff as the leaderboard prediction, then optionally
runs the official hidden evaluation before the authoring image is removed.

Backend selection is role-routed. With no operator configuration the entire
cascade uses the built-in Codex / GPT-6 Astra / Max route; ``--backend-config``
may opt into another model or a mixed-role profile without changing workflow
flags.

Three operating modes:

- ``--list`` prints instance ids (and optional metadata) for the
  filtered subset. No backend invocation.
- ``--dry-run`` walks the filtered subset and emits a zero-patch
  predictions file in
  the canonical leaderboard shape. Useful for sanity-checking the
  dataset loader + filter logic + predictions format before
  spending LLM budget.
- ``--implement`` is the live path: per-instance, clone the repo at
  ``base_commit``, run the Architect/Tier cascade against the issue's
  ``problem_statement``, capture the resulting cumulative diff, replay the
  exact source-only patch through an independent pristine visible verifier,
  then surface it as ``Prediction.model_patch`` and run inline evaluation by
  default.
  Per-instance records land in an isolated Corpus; predictions land as
  ``<run_dir>/predictions.jsonl`` in the SWE-bench leaderboard
  contract.

Usage::

    uv run crucible swebench \\
        --spec <corpus_dir> \\
        --dataset /path/to/SWE-bench_Verified.json \\
        [--instance-id django__django-15347] \\
        [--repo django/django] \\
        [--limit 10] \\
        [--list | --dry-run | --implement] \\
        [--backend-config <routing.yaml>] \\
        [--max-turns 50] \\
        [--planner-count 3] \\
        [--budget-usd-per-instance 0.50] \\
        [--budget-usd 50] \\
        [--repo-cache-dir <path>] \\
        [--max-concurrent-instances 4] \\
        [--rerun] \\
        [--predictions-out predictions.jsonl] \\
        [--no-evaluate-inline] \\
        [--eval-timeout 7200]

Exit codes:
- 0: instances processed successfully (or nothing to do).
- 1: dataset validation failed; no instances matched filters when
     filters were explicitly set; total budget exhausted mid-run.
- 2: unrecoverable setup error (dataset path missing, etc.).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from assay import (
    ExecutionContext,
    Implementation,
    LifecycleEvent,
    Measurement,
    ProjectContext,
    hash_record,
)
from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus

from crucible.backends.config import (
    BackendRouting,
    bind_routing_snapshot,
    load_routing,
    routing_feature_fingerprint,
    routing_fingerprint,
    routing_source_fingerprint,
)
from crucible.common.benchmark_runtime import (
    BudgetReservationError,
    BudgetReservations,
    ImmutableImageError,
    TaskLeaseBusy,
    canonical_fingerprint,
    file_digest,
    file_set_fingerprint,
    image_verification_policy_identity,
    managed_container_name,
    managed_container_labels,
    task_execution_lease,
    verify_immutable_image,
)
from crucible.common.cost import CostAccumulator
from crucible.products.swebench.dataset import (
    SwebenchInstance,
    SwebenchInstanceError,
    filter_instances,
    load_dataset,
)
from crucible.products.swebench.eval import (
    EvalReport,
    InstanceEvalReport,
    SwebenchEvalError,
    run_evaluation,
)
from crucible.products.swebench.predictions import (
    Prediction,
    merge_predictions_jsonl,
    write_predictions_jsonl,
)
from crucible.products.swebench.instance_container import (
    InstanceContainerSetupError as _InstanceContainerSetupError,
    check_disk_headroom as _check_disk_headroom,
    container_immutable_image as _container_immutable_image,
    docker_available as _docker_available,
    reconcile_orphans as _reconcile_orphans,
    resolve_container_spec as _resolve_container_spec_impl,
    setup_instance_container as _setup_instance_container_impl,
    teardown_instance_container as _teardown_instance_container,
)
from crucible.products.swebench.paths import is_standard_test_path
from crucible.products.swebench.pristine_verifier import (
    PristineVerificationReport,
    PristineVerificationStatus,
    verify_source_patch,
)
from crucible.products.swebench.repo_workspace import (
    RepoWorkspaceError,
    repo_workspace,
)
from crucible.common.storage_quota import (
    DirectoryGrowthQuota,
    StorageQuotaBackendUnavailable,
    StorageQuotaCleanupError,
    StorageQuotaExceeded,
)

from ._args import (
    nonnegative_budget_arg,
    positive_budget_arg,
    positive_int_arg,
)
from . import _outcome
from ._backend_recipe import backend_recipe_identity
from ._benchmark_planning import (
    BENCHMARK_PLANNING_BUDGET_SHARE,
    BenchmarkPlanningPolicy,
    benchmark_planning_identity,
    planning_recipe_identity,
)
from ._plan_competition import (
    DEFAULT_PLANNER_COUNT,
    planner_count_arg,
    validate_competition_admission,
)
from ._run import (
    exec_ctx_now,
    open_run_with_capture,
    record_handled_terminal,
    register_subprocess_cleanup,
)

logger = logging.getLogger(__name__)


def load_e12_arm_set_for_swebench():
    """Load SWE-bench's exact, offline E12 A/B/C route checkpoint."""

    from ._abc_evaluation import load_e12_arm_set

    return load_e12_arm_set(
        Path(__file__).resolve().parents[1]
        / "products"
        / "swebench"
        / "e12_arms.yaml"
    )

_AUTHORING_STORAGE_GROWTH_MB = 8192


class _TerminalPersistenceError(RuntimeError):
    """A completed attempt could not publish its durable resume marker."""


class CandidateCaptureStatus(StrEnum):
    """How a non-empty candidate was normalized before policy evaluation."""

    SOURCE_FILTERED = "source_filtered"


class CandidateOrigin(StrEnum):
    """The state surface from which a candidate patch was recovered."""

    WORKSPACE = "workspace"
    CORPUS_FALLBACK = "corpus_fallback"
    SURFACED_OUTCOME = "surfaced_outcome"


class CandidateEligibilityStatus(StrEnum):
    """Whether the candidate is authorized by the verified-only policy."""

    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


class CandidateEligibilityBlocker(StrEnum):
    """Typed reasons a captured candidate cannot be verified for submission."""

    ARCHITECT_FAILED = "architect_failed"
    TERMINAL_CLASSIFICATION_NOT_SUCCESS = (
        "terminal_classification_not_success"
    )
    CANDIDATE_CLASSIFICATION_NOT_SUCCESS = (
        "candidate_classification_not_success"
    )
    CANDIDATE_PROVENANCE_UNBOUND = "candidate_provenance_unbound"
    RECORD_ROLLUP_INCOMPLETE = "record_rollup_incomplete"
    CAPTURE_FAILED = "capture_failed"
    NO_NONEMPTY_CANDIDATE = "no_nonempty_candidate"
    PRISTINE_VERIFY_PATCH_APPLY_FAILED = (
        "pristine_verify_patch_apply_failed"
    )
    PRISTINE_VERIFY_TEST_FAILED = "pristine_verify_test_failed"
    PRISTINE_VERIFY_INFRA_FAILED = "pristine_verify_infra_failed"


class PredictionSubmissionPolicy(StrEnum):
    """Policy controlling whether a captured candidate becomes model_patch."""

    VERIFIED_SUCCESS_ONLY = "verified_success_only"
    BEST_EFFORT_ANY_NONEMPTY = "best_effort_any_nonempty"
    LEGACY_V3 = "legacy_v3"


@dataclasses.dataclass(frozen=True)
class PredictionCandidateCapture:
    """A source-only candidate retained independently of submission policy."""

    source_patch: bytes = dataclasses.field(repr=False)
    diff_ref: str
    sha256: str
    diff_bytes: int
    files_modified: tuple[str, ...]
    origin: CandidateOrigin
    capture_status: CandidateCaptureStatus
    producer_slice_id: str | None = None
    producer_classification: str | None = None


@dataclasses.dataclass(frozen=True)
class PredictionCandidateEligibility:
    """Positive authorization state for one captured candidate."""

    status: CandidateEligibilityStatus
    blockers: tuple[CandidateEligibilityBlocker, ...] = ()


@dataclasses.dataclass(frozen=True)
class PredictionSubmissionDecision:
    """Policy decision kept separate from intrinsic candidate eligibility."""

    policy: PredictionSubmissionPolicy
    submitted: bool
    override_requested: bool
    override_used: bool
    submitted_patch_ref: str | None = None


# Per-instance starter Specs come from
# products.swebench.spec_synthesis.synthesize_starter_spec; the planner
# decomposes them through run_architect → run_tier → assay.implement.


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="crucible-workflow-swebench")
    p.add_argument(
        "--spec", required=True, type=Path,
        help="Corpus-shaped spec directory. Run artifacts (predictions, "
             "per-instance Implementation records, summary.json) land "
             "under <spec>/runs/<ts>_swebench_*/.",
    )
    p.add_argument(
        "--dataset", required=True, type=Path,
        help="Path to a SWE-bench Verified instances file (JSON list or "
             "JSONL). Fetch the dataset out-of-band, e.g. via "
             "``huggingface-cli download princeton-nlp/SWE-bench_Verified``.",
    )
    p.add_argument(
        "--instance-id", action="append", default=[],
        help="Whitelist by instance_id; repeatable. Empty (default) = "
             "no instance-id filter.",
    )
    p.add_argument(
        "--repo", action="append", default=[],
        help="Whitelist by repo (e.g. ``django/django``); repeatable. "
             "Empty (default) = no repo filter.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap the post-filter instance count (default: no cap).",
    )
    p.add_argument(
        "--backend-config", type=Path, default=None,
        help="Path to a backend routing YAML overriding discovery. "
             "Role consulted: implementer (same as `crucible implement`). "
             "The resolved backend's model id flows into "
             "Prediction.model_name_or_path (the SWE-bench leaderboard's "
             "required attribution field).",
    )
    p.add_argument(
        "--predictions-out", type=Path, default=None,
        help="Override the predictions JSONL path. Default: "
             "<run_dir>/predictions.jsonl.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--list", action="store_true", dest="list_only",
        help="List instance ids + repo + base_commit for the filtered "
             "subset; no backend invocation, no predictions emitted.",
    )
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Walk the filtered subset and emit a zero-patch predictions "
             "file (every Prediction.model_patch = empty string). Used "
             "to validate dataset + filter + predictions shape before "
             "spending LLM budget.",
    )
    mode.add_argument(
        "--implement", action="store_true",
        help="Live path: per-instance clone the repo at base_commit, "
             "run assay.implement against the issue's problem_statement, "
             "capture the resulting diff as Prediction.model_patch, "
             "then invoke the official SWE-bench harness inline against "
             "the prediction before image teardown. Per-instance "
             "Implementation + Measurement records persist to the Corpus.",
    )

    # Inline-eval timeout (harness call per instance). Eval is on by
    # default; ``--no-evaluate-inline`` skips it (the flag is declared
    # below).
    p.add_argument(
        "--eval-timeout", type=int, default=7200,
        help="Per-instance harness subprocess timeout in seconds "
             "(default 7200 = 2h). Inline eval against a cached image "
             "completes in ~1-2 minutes typically; the cap exists for "
             "pathological-test scenarios.",
    )
    p.add_argument(
        "--no-evaluate-inline", dest="evaluate_inline",
        action="store_false", default=True,
        help="Default: ON. "
             "Opt out of the default per-instance inline harness eval. "
             "With ``--implement``, the worker by default runs the "
             "official SWE-bench harness against each prediction "
             "immediately after diff capture and BEFORE the "
             "container's image_rm teardown — the image is cached "
             "from the implement phase so the harness's docker run "
             "reuses it (no re-pull). Disable for cheap-smoke / "
             "ablation runs where you don't want the per-instance "
             "harness cost (~1-2 min/instance). Verdicts land on the "
             "InstanceOutcome's ``eval_status`` field + as Measurement "
             "records on the per-instance corpus.",
    )
    p.add_argument(
        "--submit-any-nonempty-best-effort",
        action="store_true",
        default=False,
        help="Submit the latest non-empty source patch even when its "
             "Implementation is classified as build_failed, verify_failed, "
             "dev_verify_failed, another non-success outcome, or the final "
             "pristine verifier fails. Default: off; failed candidates and "
             "verifier reports remain in the Corpus for diagnostics but are "
             "surfaced as empty predictions.",
    )

    # Live-implement only flags
    p.add_argument(
        "--max-turns", type=positive_int_arg, default=90,
        help="Per-instance planner + implementer max turns (default: 90).",
    )
    p.add_argument(
        "--planner-count", type=planner_count_arg, default=DEFAULT_PLANNER_COUNT,
        help="Independent plans per task (default: 3). Competing plans share "
             "40%% of the task's existing dollar budget; the remaining budget "
             "is retained for implementation and verification. Use 1 explicitly "
             "for a diagnostic single-planner run.",
    )
    p.add_argument(
        "--budget-usd-per-instance", type=positive_budget_arg, default=None,
        help="Per-instance cost ceiling. The Backend halts past it and "
             "the worker's cascade-aborted handler salvages whatever "
             "diff was captured before the cap (eval skipped inline; "
             "retained as an ineligible Corpus candidate unless explicit "
             "best-effort submission is requested).",
    )
    p.add_argument(
        "--budget-usd", type=nonnegative_budget_arg, default=None,
        help="Total cumulative cost ceiling across all instances. "
             "Halts the run mid-loop (exit 1) when exceeded.",
    )
    p.add_argument(
        "--repo-cache-dir", type=Path, default=None,
        help="Directory for reusable acquisition caches plus isolated "
             "per-instance checkouts. The cache accelerates downloads but "
             "is never shared with or mounted into authoring sandboxes. "
             "Default: direct ephemeral fetches.",
    )
    p.add_argument(
        "--max-concurrent-instances", type=positive_int_arg, default=1,
        help="Cap on instances that run their per-instance implement "
             "phase concurrently. Default 1 (serial). Values >1 fan "
             "out via ThreadPoolExecutor with explicit per-instance "
             "workspace paths. Strongly recommended to "
             "pair with --repo-cache-dir so concurrent workers reuse "
             "acquired objects while retaining standalone authoring repos.",
    )
    p.add_argument(
        "--rerun", action="store_true",
        help="Re-implement instances that already have a recorded "
             "Implementation in the corpus. Default: skip them — a "
             "Verified pass at ~$0.50 × 500 ≈ $250 is one OOM away "
             "from being repeated from scratch otherwise, and "
             "resumability is the cheapest mitigation.",
    )
    p.add_argument(
        "--rerun-status", action="append", default=[], choices=(
            "success", "soft_fail_no_changes", "soft_fail_test_only_changes",
        ),
        help="Re-execute reusable terminal classifications of this kind, "
             "including empty source predictions whose test edits were "
             "filtered; repeatable. Infrastructure failures retry by default.",
    )
    p.add_argument(
        "--no-include-breach", dest="include_breach",
        action="store_false", default=True,
        help="Skip the per-instance breach (adversarial test) phase. "
             "Default: on. Off is for smoke and ablation runs only.",
    )
    p.add_argument(
        "--no-include-review", dest="include_review",
        action="store_false", default=True,
        help="Skip the per-instance review phase. Default: on "
             "(run_architect.include_review=True). Off is for "
             "cheap-smoke / ablation runs only.",
    )
    p.add_argument(
        "--no-build-verify", dest="enable_build_verify",
        action="store_false", default=True,
        help="Opt out of the per-slice visible-test gate. The isolated "
             "authoring container remains mandatory because benchmark "
             "code and agent shell commands are never executed on the "
             "host. The final patch-only pristine visible gate remains "
             "mandatory. Hidden evaluation tests are never used by either "
             "gate.",
    )
    return p.parse_args(argv)


@dataclasses.dataclass(frozen=True)
class InstanceOutcome:
    """One instance's terminal state for summary.json + leaderboard
    emit. ``model_patch`` may be empty when the implementer hard-failed
    or when the diff capture returned no bytes.

    ``eval_status`` is the inline-harness verdict (or sentinel) per
    instance:
      - ``"resolved"``                  — harness ran, FAIL_TO_PASS
                                          + PASS_TO_PASS all green
      - ``"unresolved"``                — harness ran, patch applied,
                                          but at least one expected-
                                          pass test failed (= the fix
                                          is conceptually wrong)
      - ``"patch_apply_failed"``        — harness ran, but the model's
                                          diff didn't apply cleanly
                                          (whitespace mismatch, base-
                                          commit drift, malformed
                                          unified diff). Distinct
                                          remediation from "unresolved":
                                          implementer produced bad-
                                          format output rather than a
                                          wrong fix.
      - ``"harness_failed"``            — harness raised (docker
                                          hiccup, image evicted between
                                          setup + eval); the prediction
                                          is still on disk in
                                          predictions.jsonl for post-
                                          hoc recovery
      - ``"cascade_aborted_partial"``   — cascade was truncated mid-
                                          flight (budget exhaustion,
                                          persistence-veto exception,
                                          backend transport error,
                                          etc), or returned without a
                                          ruling on its work (a gate,
                                          reviewer, breach loop or
                                          planning backend gave no
                                          verdict; the run was
                                          cancelled). The captured diff
                                          represents partial cascade
                                          work — eval skipped inline and
                                          verified-only predictions remain
                                          empty; candidate metadata + bytes
                                          remain in the Corpus.
      - ``"skipped_pristine_*"``        — the final patch-only visible
                                          verifier rejected patch application,
                                          visible tests, or infrastructure;
                                          the candidate remains in the Corpus
                                          but verified-only submission and
                                          hidden evaluation are suppressed
      - ``"skipped_empty"``             — model_patch was empty;
                                          harness would just report
                                          unresolved
      - ``"skipped_disabled"``          — ``--no-evaluate-inline`` was
                                          set
      - ``"skipped_setup_error"``       — workspace clone / setup
                                          failed before the cascade
                                          ran; eval never reached
      - ``"skipped_cascade_error"``     — cascade raised an unexpected
                                          exception; eval never reached
      - ``None``                        — dry-run / list mode (eval
                                          never part of the flow)
    """

    instance_id: str
    classification: str
    model_patch: str
    cost_usd: float
    diff_bytes: int
    files_modified: tuple[str, ...]
    eval_status: str | None = None
    # True when a cost-bearing corpus record (Implementation / Verdict)
    # failed to load during the rollup, so ``cost_usd`` is a known
    # under-count (CRU-05a / ARC-2 — mirrors ``costs.rollup_incomplete``
    # on the architect path).
    cost_incomplete: bool = False
    resumed: bool = False
    producer_model: str = ""
    checkpoint_fingerprint: str = ""
    image_digest: str = ""
    # End-to-end token usage and the separate logical planning-call count are
    # read from the saga-bound Architect child summary. These fields keep a
    # zero-dollar inference profile observable without changing the
    # authoritative dollar ledger above.
    input_tokens: int = 0
    output_tokens: int = 0
    planning_backend_calls: int = 0
    planning_transport_failure: bool = False
    # Typed terminal of this run's saga-bound Architect child: its detail
    # status and terminal class. Run-local like the planning fields above;
    # an instance restored from a prior run carries ``None``.
    architect_detail_status: str | None = None
    architect_terminal_class: str | None = None
    # Prediction capture is source-only.  Keep the filtering decision
    # auditable without placing filtered test contents in the prediction or
    # authoring context.
    stripped_test_paths: tuple[str, ...] = ()
    stripped_test_diff_bytes: int = 0
    raw_workspace_diff_bytes: int = 0
    diff_provenance: str = "none"
    # Candidate capture, verified-only eligibility, and the eventual submission
    # decision are deliberately separate. ``model_patch`` remains the exact
    # leaderboard payload, so an ineligible candidate can stay auditable here
    # without weakening the empty-patch fail-closed contract.
    candidate_capture: PredictionCandidateCapture | None = None
    candidate_eligibility: PredictionCandidateEligibility | None = None
    submission_decision: PredictionSubmissionDecision | None = None
    # Final visible verification runs the exact content-addressed candidate in
    # a second, pristine one-commit workspace. It is deliberately distinct
    # from both the mutable authoring-container gate and the hidden benchmark
    # verdict.
    pristine_verification: PristineVerificationReport | None = None


@dataclasses.dataclass(frozen=True)
class _PoolPayload:
    """Per-instance bundle handed to one ThreadPool worker.

    Each instance owns its own Corpus at
    ``<spec_dir>/instances/<instance_id>/corpus/`` (no shared writes
    across threads) and gets its own ``backend_pool`` constructed by
    ``run_architect`` against the routing config from
    ``backend_config_path``. The Spec is synthesized fresh per
    instance via ``synthesize_starter_spec`` — no shared workflow
    Spec ride-through.
    """

    instance: SwebenchInstance
    workspace_root_spec_dir: str
    backend_config_path: str | None
    max_turns: int
    budget_usd: float | None
    repo_cache_dir: str | None
    log_dir: str
    # The durable parent run that owns this per-instance Architect child.
    # Child and parent use different Corpora, so this path cannot be rebuilt
    # from the instance Corpus.
    parent_run_dir: str
    planner_count: int = DEFAULT_PLANNER_COUNT
    include_breach: bool = True
    include_review: bool = True
    enable_build_verify: bool = True
    # When True (default), the per-instance worker runs the official
    # SWE-bench harness against its just-produced prediction BEFORE
    # the container teardown fires ``docker image rm``. The image is
    # locally cached from setup_instance_container, so the harness's
    # docker run reuses the cache (no re-pull). The InstanceEvalReport
    # persists as a Measurement in the per-instance corpus. Opt out
    # via the CLI ``--no-evaluate-inline`` for ablation runs.
    evaluate_inline: bool = True
    # Per-instance harness subprocess timeout in seconds, plumbed
    # through from --eval-timeout.
    eval_timeout: int = 7200
    # Pre-resolved implementer model id from the backend routing
    # config. Hoisted onto the payload so each worker thread doesn't
    # re-parse the YAML at instance entry (which would be N parses on
    # an N-instance parallel sweep).
    routed_model: str = "unknown"
    routed_backend: str = "unknown"
    # One in-memory routing snapshot is bound around the complete nested
    # Architect/Tier/Breach cascade. Child spec-local discovery must not drift
    # away from the benchmark parent's resolved table.
    routing: BackendRouting | None = None
    # Every worker materializes a fresh repository workspace. If its
    # per-instance Corpus already has lineage, landed slice records refer
    # to a previous workspace and must be replayed, never skipped.
    restart: bool = False
    rerun: bool = False
    rerun_status: tuple[str, ...] = ()
    routing_fingerprint: str = ""
    routing_source_sha256: str | None = None
    checkpoint_fingerprint: str = ""
    image_digest: str = ""
    # Failed source candidates remain durable Corpus artifacts by default but
    # are not benchmark submissions. This opt-in preserves the old "submit any
    # non-empty patch" behavior for explicit best-effort experiments.
    submit_any_nonempty_best_effort: bool = False


_HIDDEN_EVALUATION_FIELDS = frozenset({
    "test_patch", "fail_to_pass", "pass_to_pass",
    "FAIL_TO_PASS", "PASS_TO_PASS", "patch",
})


def _authoring_metadata(
    inst: SwebenchInstance, *, container_id: str, image_digest: str,
) -> dict[str, object]:
    """Return the complete metadata boundary visible during authoring."""
    if not container_id:
        raise _InstanceContainerSetupError(
            "SWE-bench authoring metadata requires a live container"
        )
    payload: dict[str, object] = {
        "instance_id": inst.instance_id,
        "repo": inst.repo,
        "version": inst.version,
        "container_id": container_id,
        "image_digest": image_digest,
    }
    leaked = _HIDDEN_EVALUATION_FIELDS.intersection(payload)
    if leaked:
        raise RuntimeError(
            f"hidden evaluation fields crossed authoring boundary: {leaked}"
        )
    return payload


def _remove_owned_stopped_container(
    *,
    container_name: str,
    expected_labels: dict[str, str],
    workspace: Path,
    log_prefix: str,
) -> None:
    """Remove only a stopped container owned by this exact logical task."""

    inspected = subprocess.run(
        ["docker", "inspect", container_name],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode != 0:
        return
    try:
        documents = json.loads(inspected.stdout)
        document = documents[0]
        running = document["State"]["Running"]
        labels = document["Config"]["Labels"] or {}
        container_id = document["Id"]
        if not isinstance(running, bool):
            raise TypeError("State.Running is not boolean")
        if not isinstance(labels, dict) or not isinstance(container_id, str):
            raise TypeError("container identity is malformed")
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise _InstanceContainerSetupError(
            f"{log_prefix}: cannot validate existing container "
            f"{container_name!r} ownership"
        ) from exc
    if running:
        raise _InstanceContainerSetupError(
            f"{log_prefix}: container {container_name} is already running; "
            "refusing to clobber a concurrent worker"
        )
    ownership_keys = (
        "io.crucible.managed",
        "io.crucible.scope",
        "io.crucible.task",
    )
    if any(labels.get(key) != expected_labels[key] for key in ownership_keys):
        raise _InstanceContainerSetupError(
            f"{log_prefix}: stopped container {container_name!r} is not owned "
            "by this benchmark scope/task; refusing destructive removal"
        )
    removed = subprocess.run(
        ["docker", "rm", "-f", container_id],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=False,
    )
    if removed.returncode != 0:
        raise _InstanceContainerSetupError(
            f"{log_prefix}: could not remove stopped managed container "
            f"{container_name}: {(removed.stderr or '')[-500:]}"
        )


@contextlib.contextmanager
def _instance_env_lifecycle(
    inst,
    workspace: Path,
    *,
    enabled: bool,
    log_prefix: str,
    scope: str | None = None,
    quota_retirements: list[Callable[[], None]] | None = None,
):
    """Yield an isolated per-instance execution container.

    Yields a ``dict[str, str]`` keyed by environment kind:
      - ``{"container_id": "..."}``       Docker container ready
      - ``{}``                            build/verify explicitly disabled

    When build/verify is enabled, Docker is required and every setup
    failure is fatal for the instance.  There is deliberately no host
    venv fallback: benchmark repositories and their build hooks are
    untrusted, and running them with the Crucible process environment
    would expose host credentials and permit host mutation.

    Teardown ALWAYS runs in the finally block. ``log_prefix`` tags
    warnings for the call site so serial/parallel paths look
    distinguishable in logs.
    """
    if not enabled:
        yield {}
        return
    if not _docker_available():
        raise _InstanceContainerSetupError(
            "Docker is required for SWE-bench build/verify; daemon "
            "is unavailable (host execution fallback is disabled)"
        )
    cspec = _resolve_container_spec_impl(inst)
    if cspec is None:
        raise _InstanceContainerSetupError(
            f"no supported SWE-bench container spec for "
            f"{inst.instance_id} ({inst.repo} {inst.version})"
        )
    container_scope = scope or str(workspace.parent.resolve())
    container_labels = managed_container_labels(
        scope=container_scope,
        task_id=inst.instance_id,
        phase="authoring",
    )
    container_name = managed_container_name(
        prefix="crucible-swebench",
        scope=container_scope,
        task_id=inst.instance_id,
    )
    _remove_owned_stopped_container(
        container_name=container_name,
        expected_labels=container_labels,
        workspace=workspace,
        log_prefix=log_prefix,
    )

    container_id: str | None = None
    container_image: str | None = None

    def _kill_over_quota() -> None:
        if container_id is None:
            return
        subprocess.run(
            ["docker", "kill", container_id],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=False,
        )

    # Reserve before container setup: image-provided install/pre-install
    # commands can write into the bind-mounted checkout too, so enforcing only
    # around the later agent phase leaves an untrusted setup-time gap. Keep the
    # lease until teardown completes so background container writers never see
    # an unbounded release window.
    quota = DirectoryGrowthQuota(
        (workspace,),
        limit_bytes=_AUTHORING_STORAGE_GROWTH_MB * 1024 * 1024,
        on_exceeded=_kill_over_quota,
        require_kernel=True,
    )
    try:
        with quota:
            if quota_retirements is not None:
                # The first release happens when this context exits, after
                # the container has stopped.  At that point repo_workspace's
                # generated checkout still exists, so the fail-closed helper
                # leaves a RELEASING tombstone and its hard cap in place.
                # Register exact retirement proof for after the outer
                # workspace context has removed that checkout.
                quota_retirements.append(quota.retire_kernel_lease)
            container_id = _setup_instance_container_impl(
                workspace,
                container_name,
                cspec,
                labels=container_labels,
            )
            try:
                container_image = _container_immutable_image(container_id)
                yield {
                    "container_id": container_id,
                    "image_digest": container_image,
                }
            finally:
                _teardown_instance_container(container_id, container_image)
                container_id = None
    finally:
        # Retry cleanup if teardown raised. The helper keeps a tombstoned hard
        # cap until the workspace disappears and kernel usage reaches zero.
        if container_id is not None:
            _teardown_instance_container(container_id, container_image)


@contextlib.contextmanager
def _repo_workspace_with_quota_retirement(
    inst: SwebenchInstance,
    *,
    cache_dir: Path | None,
):
    """Remove a generated checkout before finally retiring its quota.

    The callback order is load-bearing: ``repo_workspace.__exit__`` deletes
    the checkout first; only then may the helper clear the tombstoned kernel
    cap.  Any callback failure remains a storage-quota cleanup failure rather
    than silently weakening the fail-closed boundary.
    """

    retirements: list[Callable[[], None]] = []
    try:
        with repo_workspace(inst, cache_dir=cache_dir) as workspace:
            yield workspace, retirements
    finally:
        for retire in retirements:
            retire()


def _replace_outcome_eval_status(
    outcome: InstanceOutcome,
    evaluate_inline_requested: bool,
    eval_report: InstanceEvalReport | None,
    *,
    cascade_aborted: bool = False,
) -> InstanceOutcome:
    """Derive the ``eval_status`` sentinel from the inline-eval signal
    and stamp it onto a copy of ``outcome``. Centralizing the decision
    table here keeps the call sites (parallel + serial paths) from
    duplicating the sentinel logic.

    Decision table (in priority order):
      - ``cascade_aborted=True``: ``"cascade_aborted_partial"`` —
        the cascade was truncated mid-flight (budget cap, persistence-
        veto exception, backend transport error, etc) or returned without
        a ruling on its work; the captured
        diff represents partial cascade work. Eval is skipped inline and the
        ineligible candidate is retained in the Corpus; predictions.jsonl
        remains empty unless best-effort submission was explicitly requested.
      - pristine verification failed and suppressed the patch:
        ``"skipped_pristine_<status>"``
      - evaluate-inline disabled (``--no-evaluate-inline``):
        ``"skipped_disabled"``
      - prediction was empty: ``"skipped_empty"`` (helper returned None
        AND model_patch is empty)
      - harness raised: ``"harness_failed"`` (helper returned None
        AND model_patch is non-empty)
      - harness ran, resolved=True: ``"resolved"``
      - harness ran, patch_applied=False: ``"patch_apply_failed"``
        — distinct from "unresolved" because the remediation is
        different (implementer produced bad-format output rather
        than a wrong fix)
      - harness ran, patch_applied=True, resolved=False:
        ``"unresolved"`` (= the fix is conceptually wrong)
    """
    pristine_status = (
        outcome.pristine_verification.status
        if outcome.pristine_verification is not None else None
    )
    if cascade_aborted:
        status = "cascade_aborted_partial"
    elif (
        not outcome.model_patch
        and pristine_status is not None
        and pristine_status != PristineVerificationStatus.PASSED
    ):
        status = f"skipped_pristine_{pristine_status.value}"
    elif not evaluate_inline_requested:
        status = "skipped_disabled"
    elif eval_report is None:
        status = "skipped_empty" if not outcome.model_patch else "harness_failed"
    elif eval_report.resolved:
        status = "resolved"
    elif not eval_report.patch_applied:
        status = "patch_apply_failed"
    else:
        status = "unresolved"
    return dataclasses.replace(outcome, eval_status=status)


_PRISTINE_FAILURE_POLICY: dict[
    PristineVerificationStatus,
    tuple[CandidateEligibilityBlocker, str],
] = {
    PristineVerificationStatus.PATCH_APPLY_FAILED: (
        CandidateEligibilityBlocker.PRISTINE_VERIFY_PATCH_APPLY_FAILED,
        "hard_fail_pristine_patch_apply",
    ),
    PristineVerificationStatus.TEST_FAILED: (
        CandidateEligibilityBlocker.PRISTINE_VERIFY_TEST_FAILED,
        "hard_fail_pristine_test",
    ),
    PristineVerificationStatus.INFRA_FAILED: (
        CandidateEligibilityBlocker.PRISTINE_VERIFY_INFRA_FAILED,
        "hard_fail_pristine_infra",
    ),
}


def _validate_live_pristine_report_identity(
    report: PristineVerificationReport,
    *,
    source_patch: bytes,
    base_commit: str,
    image_digest: str,
) -> None:
    """Bind a live verifier result to the exact bytes and environment."""

    if report.schema != "crucible.swebench.pristine-verification.v1":
        raise ValueError("pristine verifier returned an unsupported schema")
    if (
        report.patch_sha256 != hashlib.sha256(source_patch).hexdigest()
        or report.patch_bytes != len(source_patch)
    ):
        raise ValueError("pristine verifier report does not match patch bytes")
    if report.base_commit != base_commit:
        raise ValueError("pristine verifier report does not match base commit")
    if report.image_digest != image_digest:
        raise ValueError("pristine verifier report does not match image digest")


def _apply_pristine_verification(
    outcome: InstanceOutcome,
    report: PristineVerificationReport,
) -> InstanceOutcome:
    """Apply the final visible gate without discarding candidate evidence.

    Verified-only submissions fail closed: the surfaced patch becomes empty,
    while the content-addressed candidate and report remain durable. The
    explicit best-effort policy continues to submit the bytes, but its use is
    upgraded to an auditable override after any verifier failure.
    """

    candidate, eligibility, submission = _effective_prediction_metadata(outcome)
    if candidate is None or not outcome.model_patch or not submission.submitted:
        raise ValueError("pristine verification requires a submitted candidate")
    if report.status == PristineVerificationStatus.PASSED:
        return dataclasses.replace(outcome, pristine_verification=report)

    try:
        blocker, classification = _PRISTINE_FAILURE_POLICY[report.status]
    except KeyError as exc:  # defensive against a future unhandled status
        raise ValueError(
            f"unsupported pristine verification status: {report.status!r}"
        ) from exc

    blockers = tuple(dict.fromkeys((*eligibility.blockers, blocker)))
    if report.status == PristineVerificationStatus.INFRA_FAILED:
        verification_eligibility = (
            CandidateEligibilityStatus.INELIGIBLE
            if eligibility.status == CandidateEligibilityStatus.INELIGIBLE
            else CandidateEligibilityStatus.UNKNOWN
        )
    else:
        verification_eligibility = CandidateEligibilityStatus.INELIGIBLE
    eligibility = dataclasses.replace(
        eligibility,
        status=verification_eligibility,
        blockers=blockers,
    )

    provenance_suffix = f"pristine_{report.status.value}"
    if submission.override_requested:
        # The operator explicitly selected any-nonempty best effort. Preserve
        # that behavior, but record that it is now overriding this final gate
        # as well as any earlier eligibility blocker.
        return dataclasses.replace(
            outcome,
            candidate_eligibility=eligibility,
            submission_decision=dataclasses.replace(
                submission,
                policy=PredictionSubmissionPolicy.BEST_EFFORT_ANY_NONEMPTY,
                override_used=True,
            ),
            pristine_verification=report,
            diff_provenance=(
                f"{outcome.diff_provenance}+{provenance_suffix}_best_effort"
            ),
        )

    return dataclasses.replace(
        outcome,
        classification=classification,
        model_patch="",
        diff_bytes=0,
        files_modified=(),
        candidate_eligibility=eligibility,
        submission_decision=dataclasses.replace(
            submission,
            submitted=False,
            override_used=False,
            submitted_patch_ref=None,
        ),
        pristine_verification=report,
        diff_provenance=(
            f"{outcome.diff_provenance}+suppressed_{provenance_suffix}"
        ),
    )


def _verify_submitted_candidate(
    inst: SwebenchInstance,
    outcome: InstanceOutcome,
    *,
    image_digest: str,
    repo_cache_dir: Path | None,
    scope: str,
) -> InstanceOutcome:
    """Verify the exact submitted bytes in an independent pristine checkout."""

    if not outcome.model_patch:
        return outcome
    candidate, eligibility, submission = _effective_prediction_metadata(outcome)
    if candidate is None or not submission.submitted:
        raise ValueError("non-empty prediction has no submitted candidate")
    outcome = dataclasses.replace(
        outcome,
        candidate_capture=candidate,
        candidate_eligibility=eligibility,
        submission_decision=submission,
    )
    submitted_patch = outcome.model_patch.encode("utf-8")
    try:
        report = verify_source_patch(
            inst,
            # Verify the exact UTF-8 bytes that predictions.jsonl and the hidden
            # evaluator receive. ``expected_sha256`` is bound to the retained
            # raw candidate, so serialization drift becomes a typed failure
            # instead of accidentally certifying different bytes.
            source_patch=submitted_patch,
            expected_sha256=candidate.sha256,
            expected_image_digest=image_digest,
            repo_cache_dir=repo_cache_dir,
            scope=scope,
        )
        if not isinstance(report, PristineVerificationReport):
            raise TypeError("pristine verifier returned an invalid report")
        _validate_live_pristine_report_identity(
            report,
            source_patch=submitted_patch,
            base_commit=inst.base_commit,
            image_digest=image_digest,
        )
        return _apply_pristine_verification(outcome, report)
    except Exception as exc:  # preserve paid candidate on verifier defects
        logger.exception(
            "swebench pristine verifier failed unexpectedly for %s",
            inst.instance_id,
        )
        report = PristineVerificationReport(
            schema="crucible.swebench.pristine-verification.v1",
            status=PristineVerificationStatus.INFRA_FAILED,
            reason=(
                "unexpected verifier exception: "
                f"{type(exc).__name__}: {exc}"
            )[-2_000:],
            patch_sha256=hashlib.sha256(submitted_patch).hexdigest(),
            patch_bytes=len(submitted_patch),
            base_commit=inst.base_commit,
            image_digest=image_digest,
            apply_check_rc=None,
            apply_rc=None,
            runner_rc=None,
            candidate_rc=None,
            comparison_mode="unavailable",
            preexisting_failures=0,
            new_failures=0,
            worsened_failures=0,
            duration_s=0.0,
            cleanup_status="not_started",
            raw_output_tail="",
        )
        return _apply_pristine_verification(outcome, report)


def _evaluate_prediction_inline(
    inst: SwebenchInstance,
    outcome: InstanceOutcome,
    routed_model: str,
    instance_corpus: "Corpus",
    instance_log_dir: Path,
    timeout: int = 7200,
    terminal_marker_factory: (
        Callable[[InstanceEvalReport | None], Implementation] | None
    ) = None,
) -> InstanceEvalReport | None:
    """Run the official SWE-bench harness against a single just-produced
    prediction, BEFORE the per-instance container's image_rm fires.

    The image is cached locally from ``setup_instance_container``; the
    harness's ``docker run`` reuses that cache (no re-pull). On a
    20-instance sweep this avoids ~100 GB of redundant downloads vs.
    a hypothetical two-phase shape that tore down images between
    implement and eval.

    Returns the parsed ``InstanceEvalReport`` (or ``None`` when the
    prediction is empty / the harness raises). The report is also
    stamped onto the per-instance corpus as Measurement records via
    ``_stamp_eval_measurements`` so per-instance evidence readers
    surfaces the harness verdict alongside the cascade's own slice
    classifications.

    The hidden harness is invoked exactly once. Its result may be annotated by
    a post-hoc parser reclassifier, but it can never trigger environment
    mutation or an adaptive retry; authoring and evaluation remain separate
    phases with a stable evaluation environment.

    Failure shapes (caller's ``_replace_outcome_eval_status`` maps
    these to ``eval_status`` sentinels on the InstanceOutcome):
    - ``outcome.model_patch == ""``: nothing to evaluate; return None.
      The harness would treat an empty patch as "no fix attempted"
      and report ``resolved=False``; we skip the call entirely to
      avoid the docker run cost.
    - Harness exits non-zero (docker daemon hiccup, image evicted
      between setup and eval): log + return None. The cascade's
      Implementation record persists and predictions.jsonl is on disk
      for post-hoc inspection. The caller stamps
      ``eval_status="harness_failed"`` so summary.json distinguishes
      this from skipped-empty / disabled.
    """
    if not outcome.model_patch:
        return None
    if routed_model == "unknown":
        # The harness accepts any string for ``model_name_or_path``,
        # but landing "unknown" in the predictions.jsonl breaks
        # leaderboard attribution downstream. Surface the issue at
        # WARNING so an operator iterating on a misconfigured routing
        # config sees it in the run log rather than silently shipping
        # an unattributed sweep.
        logger.warning(
            "swebench inline-eval: routed_model is 'unknown' for %s — "
            "leaderboard attribution will be unusable. Likely cause: "
            "backend routing config doesn't define an 'implementer' "
            "role, or the YAML failed to parse. Check "
            "--backend-config.",
            inst.instance_id,
        )
    # Resolve to absolute before handing to the harness subprocess —
    # the harness's ``docker run`` runs in a different cwd and bare
    # relative paths fail with FileNotFoundError.
    instance_log_dir = Path(instance_log_dir).resolve()
    instance_log_dir.mkdir(parents=True, exist_ok=True)
    single_preds = instance_log_dir / f"{inst.instance_id}.predictions.jsonl"
    write_predictions_jsonl(
        [Prediction(
            instance_id=inst.instance_id,
            model_patch=outcome.model_patch,
            model_name_or_path=routed_model,
        )],
        single_preds,
    )
    harness_output_dir = instance_log_dir / "harness_output"
    def _invoke_harness() -> "EvalReport | None":
        try:
            return run_evaluation(
                single_preds,
                instance_ids=[inst.instance_id],
                run_id=f"inline_{inst.instance_id}",
                model=routed_model,
                output_dir=harness_output_dir,
                max_workers=1,
                timeout=timeout,
                log_path=instance_log_dir / "harness.log",
            )
        except SwebenchEvalError as e:
            logger.warning(
                "swebench inline-eval: harness failed for %s: %s — "
                "prediction survives in predictions.jsonl; outcome's "
                "eval_status will be 'harness_failed' for post-hoc "
                "inspection",
                inst.instance_id, e,
            )
            return None

    eval_report = _invoke_harness()
    if eval_report is None:
        return None

    instance_report = (
        eval_report.per_instance[0] if eval_report.per_instance else None
    )
    terminal_marker = (
        terminal_marker_factory(instance_report)
        if terminal_marker_factory is not None else None
    )
    # Stamp Measurement records onto the per-instance corpus so per-instance evidence readers
    # surfaces the harness verdict. When called from the live worker, the exact
    # surfaced prediction marker shares this commit and therefore the same Spec
    # visibility point. A crash after this line cannot expose the one-shot
    # hidden oracle without also exposing the resume authority.
    _stamp_eval_measurements(
        eval_report,
        {inst.instance_id: inst},
        corpus=instance_corpus,
        run_dir=instance_log_dir,
        run_id=f"inline_{inst.instance_id}",
        terminal_implementation=terminal_marker,
    )
    # Apply the post-hoc reclassifier inline as well — corrects the 4
    # known upstream parser misclassification modes (M1-M4) at eval
    # stamp time so the Measurement bin reflects the corrected verdict
    # without a separate post-hoc pass. Skips silently for instances
    # the harness already marked resolved and for non-matching
    # patterns. See products/swebench/harness_reclassify.py.
    try:
        _stamp_inline_reclassification(
            eval_report,
            inst,
            corpus=instance_corpus,
            harness_output_dir=harness_output_dir,
            run_id=f"inline_{inst.instance_id}",
        )
    except Exception as e:
        # Reclassifier failure must NOT block the workflow — the raw
        # harness Measurement is already stamped and reflects the
        # authoritative (if uncorrected) verdict.
        logger.warning(
            "swebench inline-reclassify: failed for %s: %s — raw "
            "Measurement bin remains authoritative",
            inst.instance_id, e,
        )
    # ``run_evaluation`` returns an aggregate ``EvalReport``; for a
    # single-instance scope, exactly one ``InstanceEvalReport`` lives
    # in ``per_instance``. Return it for caller-side summary inclusion.
    return instance_report


def _pool_worker_unleased(payload: _PoolPayload) -> InstanceOutcome:
    """ThreadPoolExecutor entry point — runs one instance through the
    plan → tier (implement + breach + review) cascade via
    ``run_architect``.

    Each thread:
      1. Materialises the per-instance workspace via repo_workspace.
      2. Opens a per-instance Corpus at
         ``<root_spec_dir>/instances/<instance_id>/corpus/``.
      3. Synthesizes a starter Spec via
         ``products.swebench.spec_synthesis.synthesize_starter_spec``
         and persists it to the per-instance Corpus.
      4. Calls ``run_architect`` with ``include_breach=True`` and
         ``include_review=True`` — full cascade.
      5. Reads the final Spec from the per-instance Corpus + composes
         the union diff across Implementation records → InstanceOutcome.

    Errors are caught per-worker and folded into the
    InstanceOutcome.classification. Top-level callable so import
    machinery can resolve it from worker threads.
    """
    from crucible.products.swebench.spec_synthesis import (
        SWEBENCH_REVIEW_POLICY,
        resolve_swebench_breach_policy,
        synthesize_starter_spec,
    )
    from crucible.products.swebench.prompts import SWEBENCH_SYSTEM_PROMPT
    from crucible.products.swebench.plan_normalization import (
        AuditedSwebenchPlanTransform,
    )
    from crucible.workflows.architect import run_architect

    inst = payload.instance
    log_dir = Path(payload.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    instance_corpus_dir = (
        Path(payload.workspace_root_spec_dir)
        / "instances" / inst.instance_id / "corpus"
    )
    completed_outcome: InstanceOutcome | None = None
    completed_corpus: Corpus | None = None
    completed_checkpoint_fingerprint = ""
    completed_image_digest = ""

    def _supersede_completed_for_quota_failure(
        classification: str,
    ) -> InstanceOutcome | None:
        """Replace a durable success marker after quota exit fails closed."""

        if completed_outcome is None:
            return None
        failure_outcome = dataclasses.replace(
            completed_outcome,
            classification=classification,
        )
        assert completed_corpus is not None
        try:
            # A success marker may have been committed atomically with hidden
            # evaluation before quota exit. Supersede it while the task lease
            # is still held so resume cannot reuse an incompletely cleaned-up
            # or over-quota attempt as a successful one.
            _persist_surfaced_prediction(
                completed_corpus,
                inst,
                failure_outcome,
                backend_name=payload.routed_backend,
                model=payload.routed_model,
                checkpoint_fingerprint=completed_checkpoint_fingerprint,
                image_digest=completed_image_digest,
                assigned_budget_usd=payload.budget_usd,
            )
        except Exception as exc:
            raise _TerminalPersistenceError(
                "could not publish storage-quota failure marker "
                f"for {inst.instance_id}"
            ) from exc
        return failure_outcome

    try:
        cache_dir = (
            Path(payload.repo_cache_dir)
            if payload.repo_cache_dir else None
        )
        with _repo_workspace_with_quota_retirement(
            inst, cache_dir=cache_dir,
        ) as (workspace, quota_retirements):
            # Create the per-instance Corpus and its fresh starter Spec.
            instance_corpus_dir.mkdir(parents=True, exist_ok=True)
            corpus = Corpus(instance_corpus_dir)
            starter_spec = synthesize_starter_spec(inst)
            preparation = _prepare_instance_corpus(
                corpus, starter_spec, instance_id=inst.instance_id,
            )
            attempt_base = corpus.latest_spec()
            assert attempt_base is not None
            implementation_start_index = len(
                attempt_base.implementation_refs
            )
            verdict_start_index = len(attempt_base.verdict_refs)

            # Per-instance isolated authoring container. It is mandatory
            # even when the optional tier build/verify phase is disabled:
            # implementer and breach shell execution must never fall back
            # to the host.
            with _instance_env_lifecycle(
                inst, workspace,
                enabled=True,
                log_prefix="swebench worker",
                scope=str(Path(payload.workspace_root_spec_dir).resolve()),
                quota_retirements=quota_retirements,
            ) as env_keys:
                if (
                    payload.image_digest
                    and env_keys["image_digest"] != payload.image_digest
                ):
                    raise _InstanceContainerSetupError(
                        "SWE-bench image digest changed between checkpoint "
                        "identity and container launch"
                    )
                image_digest = env_keys["image_digest"]
                checkpoint_fingerprint = _pool_checkpoint_fingerprint(
                    payload, image_digest=image_digest,
                )
                # Authoring metadata deliberately excludes test_patch and
                # FAIL_TO_PASS/PASS_TO_PASS. Those fields belong only to
                # the post-authoring official evaluator.
                swebench_metadata = _authoring_metadata(
                    inst,
                    container_id=env_keys["container_id"],
                    image_digest=env_keys["image_digest"],
                )
                # Full plan → tier → breach → review cascade.
                breach_policy = resolve_swebench_breach_policy(
                    enabled=payload.include_breach,
                )
                cascade_aborted = False
                operation_id = _architect_operation_id(inst.instance_id)
                try:
                    with _outcome.child_operation(
                        payload.parent_run_dir,
                        operation_id=operation_id,
                        workflow="architect",
                    ):
                        plan_transform = AuditedSwebenchPlanTransform(corpus)
                        rc = run_architect(
                            spec=instance_corpus_dir,
                            cwd=str(workspace),
                            critic_rubric="swebench_plan_critic",
                            backend_config=(
                                Path(payload.backend_config_path)
                                if payload.backend_config_path else None
                            ),
                            budget_usd=payload.budget_usd,
                            max_turns=payload.max_turns,
                            planner_count=payload.planner_count,
                            planning_policy=BenchmarkPlanningPolicy(
                                planner_count=payload.planner_count,
                                planning_budget_share=(
                                    BENCHMARK_PLANNING_BUDGET_SHARE
                                    if payload.planner_count > 1 else None
                                ),
                                execution_fingerprint=checkpoint_fingerprint,
                            ),
                            # Per-instance is already the unit of parallelism;
                            # within an instance, run slices serially.
                            max_concurrent_slices=1,
                            include_review=payload.include_review,
                            # Build/verify runs only in the harness-matched
                            # container; the host environment is never used.
                            skip_build_verify=not payload.enable_build_verify,
                            # The planner needs Read/Grep/Glob to ground
                            # files_in_scope in the real repository instead of
                            # inferring paths from issue text alone.
                            enable_source_tools=True,
                            # Per-call breach data (NOT policy): the SWE-bench
                            # issue text whose alternative interpretations the
                            # Breaker attacks.
                            breach_issue_text=inst.problem_statement,
                            # This worker always creates a fresh git workspace.
                            # Existing Corpus refs describe a prior workspace,
                            # so replay landed slices instead of skipping them.
                            restart=(
                                payload.restart or preparation.had_lineage
                            ),
                            replan=preparation.requires_replan,
                            extra_parameters={
                                "swebench_metadata": swebench_metadata,
                            },
                            breach_policy=breach_policy,
                            # SWE-bench-shaped review set — drops spec_clarity
                            # since the spec is dataset-authored (problem_
                            # statement verbatim) and done_criteria is empty
                            # by design (harness FAIL_TO_PASS owns completion).
                            review_policy=SWEBENCH_REVIEW_POLICY,
                            # Opt into the iterate-on-tests implementer contract
                            # (run tests, write-early, "empty diff scores ZERO").
                            # None ⇒ assay's don't-run-tests default, which is
                            # contraindicated for issue-fixing — see
                            # products/swebench/prompts.py.
                            impl_system_prompt=SWEBENCH_SYSTEM_PROMPT,
                            plan_candidate_transform=plan_transform,
                        )
                except Exception as e:  # noqa: BLE001 — bound the cascade
                    # ANY exception from run_architect — the implementer
                    # and possibly breach Defender have already WRITTEN
                    # to the workspace. Catch INSIDE the ``with`` block
                    # so ``workspace`` and ``corpus`` are still alive;
                    # ``_spec_to_outcome`` below salvages whatever was
                    # produced (live diff or
                    # ``_latest_nonempty_diff_candidate`` from the corpus).
                    # ``git diff HEAD`` is a local subprocess so the
                    # capture stays inside whatever cap was hit.
                    #
                    # Previously this clause was ``except BudgetExhausted``
                    # — but ``assay.BudgetExhausted`` is a ``@dataclass``,
                    # NEVER raised (only attached to
                    # ``InvocationResult.error``). The clause caught
                    # nothing on the happy path and crashed with
                    # ``TypeError: catching classes that do not inherit
                    # from BaseException`` on ANY real exception,
                    # silently losing partial cascade work to the
                    # outer ``except Exception`` (which produces
                    # ``eval_status=skipped_cascade_error`` with
                    # ``model_patch=""``). Observed on pylint-4551
                    # (recovery sweep 2026-05-24): persistence_veto-
                    # shape exception → empty prediction despite 10
                    # valid Implementation refs in the corpus.
                    logger.warning(
                        "swebench worker: instance %s cascade aborted "
                        "(%s: %s) — capturing partial workspace diff; "
                        "eval skipped inline (recoverable post-sweep)",
                        inst.instance_id, type(e).__name__, e,
                    )
                    rc = 3  # halt-shape rc; mirrors review_failed code
                    cascade_aborted = True
                if not cascade_aborted and _architect_stopped_on_budget(
                    payload.parent_run_dir, operation_id,
                ):
                    # ``rc`` stays the Architect's own return value; only the
                    # inline evaluation and the status label change.
                    logger.warning(
                        "swebench worker: instance %s cascade stopped by its "
                        "budget ceiling — capturing partial workspace diff; "
                        "eval skipped inline (recoverable post-sweep)",
                        inst.instance_id,
                    )
                    cascade_aborted = True
                architect_status, architect_class = _architect_terminal_label(
                    payload.parent_run_dir, operation_id,
                )
                if not cascade_aborted and _architect_ended_unjudged(
                    rc, architect_status, architect_class,
                ):
                    # Nothing ruled on the cascade's work: a gate, reviewer,
                    # breach loop or planning backend returned no verdict,
                    # or the run was cancelled. Like a budget stop this is
                    # partial work; ``rc`` stays the Architect's own value.
                    logger.warning(
                        "swebench worker: instance %s cascade ended without "
                        "a ruling on its work (%s) — capturing partial "
                        "workspace diff; eval skipped inline (recoverable "
                        "post-sweep)",
                        inst.instance_id, architect_status,
                    )
                    cascade_aborted = True
                architect_cost, architect_rollup_incomplete = (
                    _architect_cost_for_operation(
                        payload.parent_run_dir, operation_id,
                    )
                )
                (
                    architect_input_tokens,
                    architect_output_tokens,
                    architect_backend_calls,
                    architect_transport_failure,
                ) = _architect_usage_for_operation(
                    payload.parent_run_dir, operation_id,
                )
                outcome = _spec_to_outcome(
                    corpus, inst, rc, workspace=workspace,
                    implementation_start_index=implementation_start_index,
                    verdict_start_index=verdict_start_index,
                    submit_any_nonempty_best_effort=(
                        payload.submit_any_nonempty_best_effort
                    ),
                )
                outcome = _apply_architect_cost(
                    outcome,
                    authoritative_cost=architect_cost,
                    rollup_incomplete=architect_rollup_incomplete,
                    input_tokens=architect_input_tokens,
                    output_tokens=architect_output_tokens,
                    planning_backend_calls=architect_backend_calls,
                    planning_transport_failure=architect_transport_failure,
                )
                outcome = dataclasses.replace(
                    outcome,
                    architect_detail_status=architect_status,
                    architect_terminal_class=architect_class,
                )
                # Re-materialize the exact content-addressed source patch in a
                # second one-commit workspace and container. This final visible
                # gate never reuses or mounts the mutable authoring workspace.
                # It runs before hidden evaluation so the latter only receives
                # a verified candidate under the default submission policy.
                outcome = _verify_submitted_candidate(
                    inst,
                    outcome,
                    image_digest=image_digest,
                    repo_cache_dir=cache_dir,
                    scope=str(
                        Path(payload.workspace_root_spec_dir).resolve()
                    ),
                )
                # Inline harness eval BEFORE the with-block exits and
                # ``teardown_instance_container`` removes the image.
                # The image is cached locally from setup; the harness's
                # docker run reuses it (no re-pull). Empty-patch and
                # harness-failure cases skip silently — the cascade's
                # diff is already preserved in the corpus.
                eval_report = None
                terminal_marker_published = False
                # Skip inline eval when the cascade aborted — the
                # captured diff is partial-by-definition. Verified-only mode
                # retains it as candidate metadata in the Corpus but keeps the
                # leaderboard prediction empty.
                if payload.evaluate_inline and not cascade_aborted:
                    def _terminal_marker(
                        report: InstanceEvalReport | None,
                    ) -> Implementation:
                        nonlocal terminal_marker_published
                        evaluated = _replace_outcome_eval_status(
                            outcome,
                            True,
                            report,
                        )
                        evaluated = dataclasses.replace(
                            evaluated,
                            producer_model=payload.routed_model,
                            checkpoint_fingerprint=checkpoint_fingerprint,
                            image_digest=image_digest,
                        )
                        marker = _build_surfaced_prediction_marker(
                            corpus,
                            inst,
                            evaluated,
                            backend_name=payload.routed_backend,
                            model=payload.routed_model,
                            checkpoint_fingerprint=checkpoint_fingerprint,
                            image_digest=image_digest,
                            assigned_budget_usd=payload.budget_usd,
                        )
                        # This callback is immediately followed by the atomic
                        # Corpus commit. If that commit raises, control never
                        # returns to the success path below.
                        terminal_marker_published = True
                        return marker

                    try:
                        eval_report = _evaluate_prediction_inline(
                            inst, outcome, payload.routed_model,
                            instance_corpus=corpus,
                            instance_log_dir=log_dir,
                            timeout=payload.eval_timeout,
                            terminal_marker_factory=_terminal_marker,
                        )
                    except Exception as exc:
                        raise _TerminalPersistenceError(
                            "hidden evaluation completed but its atomic "
                            f"terminal checkpoint failed for {inst.instance_id}"
                        ) from exc
                # Stamp the eval outcome onto the InstanceOutcome so
                # the parent summary distinguishes "eval succeeded" from
                # "eval failed silently" from "eval skipped (empty
                # patch)" / "eval disabled by --no-evaluate-inline" /
                # "cascade aborted mid-flight, partial diff captured".
                outcome = _replace_outcome_eval_status(
                    outcome, payload.evaluate_inline, eval_report,
                    cascade_aborted=cascade_aborted,
                )
                outcome = dataclasses.replace(
                    outcome,
                    producer_model=payload.routed_model,
                    checkpoint_fingerprint=checkpoint_fingerprint,
                    image_digest=image_digest,
                )
                if not terminal_marker_published:
                    try:
                        _persist_surfaced_prediction(
                            corpus, inst, outcome,
                            backend_name=payload.routed_backend,
                            model=payload.routed_model,
                            checkpoint_fingerprint=checkpoint_fingerprint,
                            image_digest=image_digest,
                            assigned_budget_usd=payload.budget_usd,
                        )
                    except Exception as exc:
                        raise _TerminalPersistenceError(
                            "could not publish terminal prediction marker "
                            f"for {inst.instance_id}"
                        ) from exc
                # Assign only after the terminal marker is durable. A quota
                # release error is raised by the enclosing context manager
                # after this return begins; retaining these values lets the
                # outer handler report the completed patch and accounting
                # instead of replacing them with a zero-valued quota result.
                completed_outcome = outcome
                completed_corpus = corpus
                completed_checkpoint_fingerprint = checkpoint_fingerprint
                completed_image_digest = image_digest
                return completed_outcome
    except _TerminalPersistenceError:
        # The task lease is owned by _pool_worker and remains held until this
        # failure crosses that wrapper. Never turn an unpublished terminal
        # outcome into an apparent worker success.
        raise
    except RepoWorkspaceError as e:
        logger.error(
            "swebench worker: instance %s workspace error: %s",
            inst.instance_id, e,
        )
        return InstanceOutcome(
            instance_id=inst.instance_id,
            classification="hard_fail_workspace_error",
            model_patch="", cost_usd=0.0, diff_bytes=0,
            files_modified=(),
            # Eval never reached — the cascade didn't start. Distinct
            # from "skipped_empty" / "skipped_disabled" so post-sweep
            # inspection can attribute the gap to setup failure.
            eval_status="skipped_setup_error",
        )
    except StorageQuotaCleanupError as e:
        logger.error(
            "swebench worker: instance %s completed but its writable "
            "storage quota lease cleanup failed: %s",
            inst.instance_id, e,
        )
        failure_outcome = _supersede_completed_for_quota_failure(
            "hard_fail_storage_quota_cleanup"
        )
        if failure_outcome is None:
            if isinstance(e.__cause__, _TerminalPersistenceError):
                # Quota cleanup must not launder an unpublished hidden-oracle
                # result into an ordinary retryable worker outcome.
                raise e.__cause__ from e
            return InstanceOutcome(
                instance_id=inst.instance_id,
                classification="hard_fail_storage_quota_cleanup",
                model_patch="", cost_usd=0.0, diff_bytes=0,
                files_modified=(),
                eval_status="skipped_setup_error",
            )
        return failure_outcome
    except StorageQuotaBackendUnavailable as e:
        logger.error(
            "swebench worker: instance %s storage quota backend "
            "unavailable: %s",
            inst.instance_id, e,
        )
        return InstanceOutcome(
            instance_id=inst.instance_id,
            classification="hard_fail_storage_quota_backend",
            model_patch="", cost_usd=0.0, diff_bytes=0,
            files_modified=(),
            eval_status="skipped_setup_error",
        )
    except StorageQuotaExceeded as e:
        logger.error(
            "swebench worker: instance %s exceeded writable storage quota: %s",
            inst.instance_id, e,
        )
        failure_outcome = _supersede_completed_for_quota_failure(
            "hard_fail_storage_quota"
        )
        if failure_outcome is not None:
            return failure_outcome
        return InstanceOutcome(
            instance_id=inst.instance_id,
            classification="hard_fail_storage_quota",
            model_patch="", cost_usd=0.0, diff_bytes=0,
            files_modified=(),
            eval_status="skipped_cascade_error",
        )
    except _InstanceContainerSetupError as e:
        logger.error(
            "swebench worker: instance %s isolated container setup "
            "failed: %s",
            inst.instance_id, e,
        )
        return InstanceOutcome(
            instance_id=inst.instance_id,
            classification="hard_fail_container_setup",
            model_patch="", cost_usd=0.0, diff_bytes=0,
            files_modified=(), eval_status="skipped_setup_error",
        )
    except Exception as e:  # noqa: BLE001 — bound the worker
        logger.exception(
            "swebench worker: instance %s raised unexpectedly: %s",
            inst.instance_id, e,
        )
        return InstanceOutcome(
            instance_id=inst.instance_id,
            classification=f"hard_fail_{type(e).__name__.lower()}",
            model_patch="", cost_usd=0.0, diff_bytes=0,
            files_modified=(),
            # Cascade raised; eval never reached. Distinct from
            # "skipped_setup_error" so post-sweep inspection can tell
            # workspace-setup failures from in-cascade crashes.
            eval_status="skipped_cascade_error",
            # The exception may follow paid planner/agent calls before the
            # architect summary became readable. Never schedule more work
            # against an aggregate cap using this known under-count.
            cost_incomplete=True,
        )


def _pool_worker(payload: _PoolPayload) -> InstanceOutcome:
    """Bind the parent's immutable routing snapshot around the full cascade."""

    validate_competition_admission(payload.planner_count, payload.budget_usd)
    if payload.routing is None:
        return _pool_worker_with_routing(payload)
    with bind_routing_snapshot(payload.routing):
        return _pool_worker_with_routing(payload)


def _pool_worker_with_routing(payload: _PoolPayload) -> InstanceOutcome:
    """Execute/resume one task while holding its cross-process lease.

    The marker is re-read only after locking, and the lease remains held until
    the surfaced marker is durably attached by ``_pool_worker_unleased``.
    """

    inst = payload.instance
    root = Path(payload.workspace_root_spec_dir)
    try:
        with task_execution_lease(
            root,
            namespace="swebench",
            task_id=inst.instance_id,
        ):
            prior = _existing_predictions_by_id(
                root, (inst.instance_id,),
            ).get(inst.instance_id)
            if prior is not None:
                prior_budget_payload = dataclasses.replace(
                    payload,
                    budget_usd=prior.assigned_budget_usd,
                )
                fingerprint = _pool_checkpoint_fingerprint(
                    prior_budget_payload,
                    image_digest=prior.image_digest,
                )
                reusable = (
                    _is_skip_worthy_prior(prior.classification)
                    and not prior.cost_incomplete
                    and (
                        not payload.evaluate_inline
                        or not prior.model_patch
                        or prior.eval_status in {
                            "resolved",
                            "unresolved",
                            "patch_apply_failed",
                        }
                    )
                )
                explicitly_rerun = (
                    payload.rerun
                    or prior.classification in payload.rerun_status
                )
                if (
                    reusable
                    and prior.checkpoint_fingerprint != fingerprint
                    and not explicitly_rerun
                ):
                    return InstanceOutcome(
                        instance_id=inst.instance_id,
                        classification="hard_fail_checkpoint_identity",
                        model_patch="",
                        cost_usd=0.0,
                        diff_bytes=0,
                        files_modified=(),
                        eval_status="skipped_setup_error",
                        checkpoint_fingerprint=fingerprint,
                        image_digest=prior.image_digest,
                    )
                if (
                    reusable
                    and prior.checkpoint_fingerprint == fingerprint
                    and not explicitly_rerun
                ):
                    try:
                        verify_immutable_image(prior.image_digest)
                    except ImmutableImageError as exc:
                        raise _InstanceContainerSetupError(
                            "saved SWE-bench image no longer passes the "
                            f"current verification policy: {exc}"
                        ) from exc
                    return InstanceOutcome(
                        instance_id=inst.instance_id,
                        classification=prior.classification,
                        model_patch=prior.model_patch,
                        cost_usd=0.0,
                        diff_bytes=len(prior.model_patch.encode("utf-8")),
                        files_modified=(),
                        eval_status=prior.eval_status,
                        resumed=True,
                        producer_model=prior.model,
                        checkpoint_fingerprint=fingerprint,
                        image_digest=prior.image_digest,
                        stripped_test_paths=prior.stripped_test_paths,
                        stripped_test_diff_bytes=(
                            prior.stripped_test_diff_bytes
                        ),
                        raw_workspace_diff_bytes=(
                            prior.raw_workspace_diff_bytes
                        ),
                        diff_provenance=prior.diff_provenance,
                        candidate_capture=prior.candidate_capture,
                        candidate_eligibility=prior.candidate_eligibility,
                        submission_decision=prior.submission_decision,
                        pristine_verification=prior.pristine_verification,
                    )
            return _pool_worker_unleased(payload)
    except TaskLeaseBusy as exc:
        logger.warning("swebench worker: %s", exc)
        return InstanceOutcome(
            instance_id=inst.instance_id,
            classification="hard_fail_task_lease_busy",
            model_patch="",
            cost_usd=0.0,
            diff_bytes=0,
            files_modified=(),
            eval_status="skipped_setup_error",
        )
    except _InstanceContainerSetupError as exc:
        logger.error(
            "swebench worker: instance %s image resolution failed: %s",
            inst.instance_id, exc,
        )
        return InstanceOutcome(
            instance_id=inst.instance_id,
            classification="hard_fail_container_setup",
            model_patch="",
            cost_usd=0.0,
            diff_bytes=0,
            files_modified=(),
            eval_status="skipped_setup_error",
        )


# run_architect → run_tier construct their own per-call backend_pool with
# the instance workspace cwd; swebench.py does not construct backends.


# Install-artifact paths that pip/setuptools/tox/etc create during the
# instance container's setup step. ``git ls-files --others --exclude-standard``
# only filters by the repo's own .gitignore, and many SWE-bench-era
# repos predate modern .gitignore conventions — e.g. psf/requests at
# base_commit 22623bd8 does NOT gitignore ``build/``. Observed on
# psf__requests-1142 recovery sweep (2026-05-24): an 8-line edit in
# requests/models.py shipped as an 872 KB / 66-file prediction
# because the pip editable install left build/lib/requests/* behind.
# Filtering trims the patch to the actual model edit; verdicts are
# unchanged (the harness applied the install artifacts harmlessly,
# they just polluted the leaderboard JSONL).
_SWEBENCH_INSTALL_ARTIFACT_PREFIXES: tuple[str, ...] = (
    "build/",
    "dist/",
    ".tox/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "node_modules/",
)


def _is_install_artifact(path: str) -> bool:
    """Return True for paths under pip/setuptools/tox/cache prefixes
    or inside any ``*.egg-info`` directory (anywhere in the tree).
    Used by ``_capture_workspace_diff_report`` to drop install artifacts
    from prediction diffs.
    """
    if path.startswith(_SWEBENCH_INSTALL_ARTIFACT_PREFIXES):
        return True
    return any(part.endswith(".egg-info") for part in path.split("/"))


@dataclasses.dataclass(frozen=True)
class _WorkspaceDiffCapture:
    """Source-only patch plus auditable, content-free filter metadata."""

    source_patch: bytes = b""
    source_paths: tuple[str, ...] = ()
    raw_diff_bytes: int = 0
    stripped_test_paths: tuple[str, ...] = ()
    stripped_test_diff_bytes: int = 0
    provenance: str = "none"


def _merge_diff_captures(
    workspace_capture: _WorkspaceDiffCapture,
    fallback_capture: _WorkspaceDiffCapture,
) -> _WorkspaceDiffCapture:
    """Retain workspace filter evidence while surfacing a stored source diff."""

    return _WorkspaceDiffCapture(
        source_patch=fallback_capture.source_patch,
        source_paths=fallback_capture.source_paths,
        # This field is explicitly the live workspace's raw size when one was
        # observed; otherwise the fallback candidate is the only raw input.
        raw_diff_bytes=(
            workspace_capture.raw_diff_bytes
            or fallback_capture.raw_diff_bytes
        ),
        stripped_test_paths=tuple(sorted({
            *workspace_capture.stripped_test_paths,
            *fallback_capture.stripped_test_paths,
        })),
        stripped_test_diff_bytes=(
            workspace_capture.stripped_test_diff_bytes
            + fallback_capture.stripped_test_diff_bytes
        ),
        provenance=(
            f"{workspace_capture.provenance}+{fallback_capture.provenance}"
        ),
    )


def _git_pathspec_exclude_literal(path: str) -> str:
    """Return a top-anchored literal exclusion safe for Git pathspecs."""

    return f":(exclude,top,literal){path}"


def _decode_name_status_entries_z(
    payload: bytes,
) -> tuple[tuple[str, ...], ...]:
    """Decode changes, preserving rename/copy endpoint membership."""

    fields = payload.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    entries: list[tuple[str, ...]] = []
    index = 0
    while index < len(fields):
        status = fields[index].decode("ascii", errors="replace")
        index += 1
        path_count = 2 if status[:1] in {"C", "R"} else 1
        if index + path_count > len(fields):
            raise ValueError("truncated git --name-status -z output")
        entries.append(tuple(
            os.fsdecode(raw_path)
            for raw_path in fields[index:index + path_count]
        ))
        index += path_count
    return tuple(entries)


def _diff_capture_from_index(
    workspace: Path,
    inst: SwebenchInstance,
    *,
    cached: bool,
    env: dict[str, str] | None,
    provenance: str,
) -> _WorkspaceDiffCapture:
    """Capture one index/worktree state and remove whole test-file entries."""

    cwd = str(workspace)
    cached_arg = ["--cached"] if cached else []
    names = subprocess.run(
        [
            "git", "-c", "core.hooksPath=/dev/null", "diff", *cached_arg,
            "--name-status", "-z", "--find-renames", "--find-copies-harder",
            "HEAD", "--",
        ],
        cwd=cwd, env=env, capture_output=True, check=False, timeout=30,
    )
    if names.returncode != 0:
        logger.warning(
            "swebench %s: git diff --name-status failed (rc=%d): %s",
            inst.instance_id, names.returncode,
            names.stderr.decode("utf-8", errors="replace")[:500],
        )
        return _WorkspaceDiffCapture(provenance=f"{provenance}_error")
    try:
        change_entries = _decode_name_status_entries_z(names.stdout)
    except ValueError as exc:
        logger.warning(
            "swebench %s: could not parse changed paths: %s",
            inst.instance_id, exc,
        )
        return _WorkspaceDiffCapture(provenance=f"{provenance}_error")

    # A rename/copy is one semantic change.  If either endpoint is filtered,
    # exclude both; excluding just one endpoint can turn a source->test rename
    # into a source deletion (or a test->source rename into a source add).
    artifact_path_set: set[str] = set()
    test_path_set: set[str] = set()
    source_path_set: set[str] = set()
    for entry in change_entries:
        is_artifact = any(_is_install_artifact(path) for path in entry)
        is_test = any(is_standard_test_path(path) for path in entry)
        if is_artifact:
            artifact_path_set.update(entry)
        if is_test:
            test_path_set.update(entry)
        if not is_artifact and not is_test:
            source_path_set.update(entry)
    artifact_paths = tuple(sorted(artifact_path_set))
    test_paths = tuple(sorted(test_path_set - artifact_path_set))
    common = [
        "git", "-c", "core.hooksPath=/dev/null", "diff", *cached_arg,
        "--binary", "--find-renames", "--find-copies-harder", "HEAD", "--",
        ":(top,glob)**",
    ]
    artifact_excludes = [
        _git_pathspec_exclude_literal(path) for path in artifact_paths
    ]

    raw = subprocess.run(
        [*common, *artifact_excludes], cwd=cwd, env=env,
        capture_output=True, check=False, timeout=60,
    )
    source = subprocess.run(
        [
            *common,
            *artifact_excludes,
            *(_git_pathspec_exclude_literal(path) for path in test_paths),
        ],
        cwd=cwd, env=env, capture_output=True, check=False, timeout=60,
    )
    if raw.returncode != 0 or source.returncode != 0:
        failed = raw if raw.returncode != 0 else source
        logger.warning(
            "swebench %s: source-only git diff failed (rc=%d): %s",
            inst.instance_id, failed.returncode,
            failed.stderr.decode("utf-8", errors="replace")[:500],
        )
        return _WorkspaceDiffCapture(provenance=f"{provenance}_error")

    stripped_bytes = 0
    if test_paths:
        test_only = subprocess.run(
            [
                "git", "-c", "core.hooksPath=/dev/null", "diff",
                *cached_arg, "--binary", "--find-renames",
                "--find-copies-harder", "HEAD", "--",
                *(f":(top,literal){path}" for path in test_paths),
            ],
            cwd=cwd, env=env, capture_output=True, check=False, timeout=60,
        )
        if test_only.returncode == 0:
            stripped_bytes = len(test_only.stdout)
        else:
            logger.warning(
                "swebench %s: test-only provenance diff failed (rc=%d)",
                inst.instance_id, test_only.returncode,
            )

    return _WorkspaceDiffCapture(
        source_patch=source.stdout,
        source_paths=tuple(sorted(source_path_set)),
        raw_diff_bytes=len(raw.stdout),
        stripped_test_paths=test_paths,
        stripped_test_diff_bytes=stripped_bytes,
        provenance=provenance,
    )


def _capture_workspace_diff_report_impl(
    workspace: Path,
    inst: SwebenchInstance,
) -> _WorkspaceDiffCapture:
    """Capture a syntactically complete, source-only workspace patch.

    The workspace's ``HEAD`` IS ``base_commit`` (the per-instance
    isolated checkout pins that commit and never advances HEAD —
    ``assay.implement`` doesn't commit between slices). So
    ``git diff HEAD`` against the working tree captures the cumulative
    delta from the cascade: implementer slice edits + breach Defender
    edits, with no intermediate-state duplication.

    ``git add --intent-to-add`` runs first against untracked files so new-file
    diffs appear.  Changed paths are then classified with the product-local
    :func:`is_standard_test_path` policy.  Git pathspec exclusions remove
    whole tracked, untracked, renamed, or deleted test-file entries; test
    files stay present and usable throughout authoring.

    Returns an empty capture on git failure rather than raising — diff capture
    is best-effort; the outcome records the empty-diff case downstream.

    Safety gate: before running any git command, verify that
    ``git -C <workspace> rev-parse --show-toplevel`` resolves to
    ``workspace`` itself. If the checkout's ``.git`` directory is missing
    or corrupted, git may silently walk up
    to the next ancestor ``.git`` — for SWE-bench runs that's the
    outer workspace repository, and capturing that diff would emit
    submodule pointers + uncommitted operator state as the
    leaderboard prediction. Empty bytes + a loud error is the
    correct response.
    """
    cwd = str(workspace)
    expected_root = str(Path(workspace).resolve())
    toplevel = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "rev-parse", "--show-toplevel"],
        cwd=cwd, capture_output=True, check=False, timeout=30,
    )
    if toplevel.returncode != 0:
        logger.error(
            "swebench %s: git rev-parse failed in workspace %s "
            "(rc=%d, stderr=%s); refusing to capture diff",
            inst.instance_id, cwd, toplevel.returncode,
            toplevel.stderr.decode("utf-8", errors="replace")[:200],
        )
        return _WorkspaceDiffCapture(provenance="workspace_error")
    raw_actual = toplevel.stdout.decode("utf-8", errors="replace").strip()
    try:
        actual_root = str(Path(raw_actual).resolve())
    except (OSError, ValueError):
        actual_root = raw_actual
    if actual_root != expected_root:
        logger.error(
            "swebench %s: workspace cwd=%s resolves to git toplevel=%s — "
            "checkout .git missing or corrupted. Refusing to "
            "capture diff; would emit the parent repo's state as the "
            "leaderboard prediction.",
            inst.instance_id, expected_root, actual_root,
        )
        return _WorkspaceDiffCapture(provenance="workspace_error")
    untracked_result = subprocess.run(
        [
            "git", "-c", "core.hooksPath=/dev/null", "ls-files", "-z",
            "--others", "--exclude-standard",
        ],
        cwd=cwd, capture_output=True, check=False, timeout=30,
    )
    if untracked_result.returncode == 0 and untracked_result.stdout:
        new_paths = [
            os.fsdecode(p) for p in untracked_result.stdout.split(b"\0") if p
        ]
        new_paths = [p for p in new_paths if not _is_install_artifact(p)]
        if new_paths:
            encoded_paths = b"\0".join(os.fsencode(p) for p in new_paths) + b"\0"
            add_result = subprocess.run(
                [
                    "git", "--literal-pathspecs", "-c",
                    "core.hooksPath=/dev/null", "add",
                    "--intent-to-add", "--pathspec-from-file=-",
                    "--pathspec-file-nul",
                ],
                cwd=cwd, capture_output=True, check=False, timeout=30,
                input=encoded_paths,
            )
            if add_result.returncode != 0:
                logger.warning(
                    "swebench %s: git add --intent-to-add failed (rc=%d): %s",
                    inst.instance_id, add_result.returncode,
                    add_result.stderr.decode("utf-8", errors="replace")[:500],
                )

    return _diff_capture_from_index(
        workspace, inst, cached=False, env=None, provenance="workspace",
    )


def _capture_workspace_diff_report(
    workspace: Path,
    inst: SwebenchInstance,
) -> _WorkspaceDiffCapture:
    """Best-effort facade around the Git subprocess capture sequence."""

    try:
        return _capture_workspace_diff_report_impl(workspace, inst)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(
            "swebench %s: workspace diff capture failed in %s: %s",
            inst.instance_id, workspace, exc,
        )
        return _WorkspaceDiffCapture(provenance="workspace_error")


def _filter_corpus_patch_to_source(
    workspace: Path,
    inst: SwebenchInstance,
    patch: bytes,
) -> _WorkspaceDiffCapture:
    """Apply a stored candidate to a temporary index and re-diff source-only.

    Corpus fallback patches bypass the live-worktree capture path.  A private
    index lets Git parse renames, deletes, binary patches, and quoted paths
    without modifying the authoring checkout or trusting textual diff splits.
    """

    fd, raw_index_path = tempfile.mkstemp(prefix="crucible-swe-index-")
    os.close(fd)
    index_path = Path(raw_index_path)
    index_path.unlink()
    env = dict(os.environ)
    env["GIT_INDEX_FILE"] = str(index_path)
    try:
        read_tree = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "read-tree", "HEAD"],
            cwd=str(workspace), env=env, capture_output=True,
            check=False, timeout=30,
        )
        if read_tree.returncode != 0:
            logger.warning(
                "swebench %s: temporary-index read-tree failed: %s",
                inst.instance_id,
                read_tree.stderr.decode("utf-8", errors="replace")[:500],
            )
            return _WorkspaceDiffCapture(
                raw_diff_bytes=len(patch), provenance="corpus_fallback_error",
            )
        apply_result = subprocess.run(
            [
                "git", "-c", "core.hooksPath=/dev/null", "apply",
                "--cached", "--whitespace=nowarn", "-",
            ],
            cwd=str(workspace), env=env, input=patch, capture_output=True,
            check=False, timeout=60,
        )
        if apply_result.returncode != 0:
            logger.warning(
                "swebench %s: stored fallback patch did not apply to base; "
                "refusing unfiltered prediction: %s",
                inst.instance_id,
                apply_result.stderr.decode("utf-8", errors="replace")[:500],
            )
            return _WorkspaceDiffCapture(
                raw_diff_bytes=len(patch), provenance="corpus_fallback_error",
            )
        return _diff_capture_from_index(
            workspace, inst, cached=True, env=env,
            provenance="corpus_fallback",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(
            "swebench %s: stored fallback patch filtering failed: %s",
            inst.instance_id, exc,
        )
        return _WorkspaceDiffCapture(
            raw_diff_bytes=len(patch), provenance="corpus_fallback_error",
        )
    finally:
        index_path.unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True)
class _StoredDiffCandidate:
    """One persisted non-empty Implementation diff and its gate status."""

    patch: bytes
    classification: str
    slice_id: str


def _latest_nonempty_diff_candidate(
    corpus: Corpus,
    spec,
    instance_id: str,
    *,
    implementation_start_index: int = 0,
) -> _StoredDiffCandidate | None:
    """Walk ``spec.implementation_refs`` in reverse and return the
    first non-empty diff blob with its producing classification. Returns
    ``None`` if no Implementation carries usable bytes.

    Used by ``_spec_to_outcome`` as a fallback when the live
    workspace diff is empty but the corpus has prior work — typically
    because a breach Defender reverted the workspace mid-loop. The
    cascade's spend is recoverable from the corpus even when the
    on-disk state is gone.

    Walking in REVERSE (latest first) means the most-refined
    Implementation wins. Earlier-round diffs are skipped — they're
    superseded by later ones in the Defender's iteration.
    """
    for ref in reversed(
        spec.implementation_refs[implementation_start_index:]
    ):
        try:
            impl = corpus.get_record(ref, as_type=Implementation)
        except (FileNotFoundError, OSError, TypeError):
            continue
        if not isinstance(impl, Implementation) or not impl.diff_ref:
            continue
        try:
            blob = corpus.get_artifact(impl.diff_ref)
        except (FileNotFoundError, OSError):
            continue
        if blob:
            logger.info(
                "swebench %s: found stored diff_ref candidate from "
                "Implementation %r (%d bytes, classification=%r)",
                instance_id, impl.slice_id, len(blob),
                impl.classification,
            )
            return _StoredDiffCandidate(
                patch=blob,
                classification=str(impl.classification),
                slice_id=str(impl.slice_id),
            )
    return None


def _latest_matching_diff_candidate(
    corpus: Corpus,
    spec,
    inst: SwebenchInstance,
    workspace: Path,
    source_patch: bytes,
    *,
    implementation_start_index: int = 0,
) -> _StoredDiffCandidate | None:
    """Bind a live source patch only to an exact normalized producer.

    A later failed retry can be durably attached for accounting even after its
    workspace bytes are rolled back.  Therefore record order alone cannot
    identify the producer of a live patch.  Walk newest-first, normalize every
    stored diff through the same source-only filter used for submissions, and
    return only a byte-for-byte (and digest) match.

    ``None`` is intentionally fail-closed.  In particular, callers must not
    borrow the latest record's classification for an unmatched live patch.
    """

    expected_sha256 = _content_sha256(source_patch)
    for ref in reversed(
        spec.implementation_refs[implementation_start_index:]
    ):
        try:
            impl = corpus.get_record(ref, as_type=Implementation)
        except (FileNotFoundError, OSError, TypeError):
            continue
        if not isinstance(impl, Implementation) or not impl.diff_ref:
            continue
        try:
            stored_patch = corpus.get_artifact(impl.diff_ref)
        except (FileNotFoundError, OSError):
            continue
        if not stored_patch:
            continue

        if stored_patch == source_patch:
            normalized_patch = stored_patch
        else:
            normalized_patch = _filter_corpus_patch_to_source(
                workspace,
                inst,
                stored_patch,
            ).source_patch
        if (
            _content_sha256(normalized_patch) != expected_sha256
            or normalized_patch != source_patch
        ):
            continue

        logger.info(
            "swebench %s: bound live candidate to exact normalized "
            "Implementation %r (%d bytes, classification=%r)",
            inst.instance_id,
            impl.slice_id,
            len(normalized_patch),
            impl.classification,
        )
        return _StoredDiffCandidate(
            patch=stored_patch,
            classification=str(impl.classification),
            slice_id=str(impl.slice_id),
        )
    return None


def _reduce_implementation_classifications(
    classifications: list[str],
) -> str:
    """Return the terminal causal Implementation classification.

    Implementation refs are appended in lifecycle order.  A later repair is
    therefore authoritative over the failed attempt that caused it, while a
    later failed/empty attempt must not be hidden by an earlier success.
    Severity-based reduction got both cases wrong by allowing historical
    failures to dominate a successfully verified repair forever.
    """

    if not classifications:
        return "success"
    return classifications[-1]


def _candidate_is_submittable(classification: str) -> bool:
    """Only verified success-class candidates ship without an opt-in."""

    return classification == "success" or classification.startswith("success_")


def _content_sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _content_ref(blob: bytes) -> str:
    return f"sha256:{_content_sha256(blob)}"


def _persist_candidate_capture(
    corpus: Corpus,
    inst: SwebenchInstance,
    patch: bytes,
    *,
    files_modified: tuple[str, ...],
    origin: CandidateOrigin,
    capture_status: CandidateCaptureStatus,
    producer: _StoredDiffCandidate | None,
) -> PredictionCandidateCapture:
    """Persist and describe a normalized candidate before policy suppression.

    Production callers always provide a real :class:`Corpus`. A few focused
    unit tests intentionally use minimal corpus doubles; for those, the
    content-address itself remains deterministic even though the double has no
    storage method.
    """

    expected_ref = _content_ref(patch)
    put_artifact = getattr(corpus, "put_artifact", None)
    if callable(put_artifact):
        stored_ref = str(put_artifact(
            patch,
            kind="diffs",
            name_hint=f"{inst.instance_id}_candidate.diff",
        ))
        if stored_ref != expected_ref:
            raise ValueError(
                "candidate artifact store returned a non-content-addressed ref"
            )
    return PredictionCandidateCapture(
        source_patch=patch,
        diff_ref=expected_ref,
        sha256=_content_sha256(patch),
        diff_bytes=len(patch),
        files_modified=files_modified,
        origin=origin,
        capture_status=capture_status,
        producer_slice_id=(producer.slice_id if producer is not None else None),
        producer_classification=(
            producer.classification if producer is not None else None
        ),
    )


def _spec_to_outcome(
    corpus: Corpus,
    inst: SwebenchInstance,
    architect_rc: int,
    *,
    workspace: Path | None = None,
    implementation_start_index: int = 0,
    verdict_start_index: int = 0,
    submit_any_nonempty_best_effort: bool = False,
) -> InstanceOutcome:
    """Compose an InstanceOutcome from the final per-instance Spec.

    The ``model_patch`` is captured by a single source-only ``git diff HEAD``
    against the per-instance workspace's working tree, with standard test
    paths excluded via literal Git pathspecs. One canonical diff against
    ``base_commit`` rather than concatenating each ``Implementation
    .diff_ref`` — the latter shape produced duplicate hunks because
    the breach Defender's Implementation captures a cumulative diff
    (impl edits + Defender's breach test additions), which the
    implementer's per-slice diff already covered.

    ``Implementation.cost_usd`` + ``classification`` are still
    summed across the Spec's ``implementation_refs`` for the
    cost / outcome rollup — those are per-slice signals, not the
    leaderboard deliverable. ``Verdict.cost_usd`` is summed across
    the Spec's ``verdict_refs`` so plan-critic / breach-Breaker /
    breach-Judge / review costs land in ``outcome.cost_usd`` too —
    otherwise the per-instance total only reflects implementer spend
    and the ``--budget-usd`` total-budget cap can't trip on a
    runaway planner or breach loop.

    ``architect_rc`` is the exit code from run_architect (0 on
    success, 1 on tier failure, 2 on plan_failed); used to derive
    the InstanceOutcome.classification when no Implementation
    records exist yet (early-fail scenarios).

    ``workspace`` is the per-instance repo path. When ``None``
    (callers that finished the cascade before the workspace was
    materialised, e.g. early architect failure), ``model_patch``
    falls back to empty and the classification reflects the
    architect_rc.
    """
    spec = corpus.latest_spec()
    if (
        spec is None
        or len(spec.implementation_refs) <= implementation_start_index
    ):
        # Architect failed before any Implementation landed (plan
        # failed, validation gate halted, etc). Critic/judge calls may
        # still have incurred cost, so account current-attempt verdicts.
        verdict_cost = 0.0
        verdict_unloaded = 0
        if spec is not None:
            from assay.types.findings import Verdict
            for ref in spec.verdict_refs[verdict_start_index:]:
                try:
                    verdict = corpus.get_record(ref, as_type=Verdict)
                except (FileNotFoundError, OSError, TypeError):
                    verdict_unloaded += 1
                    continue
                if isinstance(verdict, Verdict):
                    verdict_cost += float(
                        getattr(verdict, "cost_usd", 0.0) or 0.0
                    )
        return InstanceOutcome(
            instance_id=inst.instance_id,
            classification=(
                "hard_fail_no_implementation"
                if architect_rc != 0 else "soft_fail_no_changes"
            ),
            model_patch="",
            cost_usd=verdict_cost,
            diff_bytes=0,
            files_modified=(),
            cost_incomplete=verdict_unloaded > 0,
        )

    cost_total = 0.0
    files_modified: list[str] = []
    classifications: list[str] = []
    unloaded = 0  # cost-bearing refs that failed to load → under-count
    for ref in spec.implementation_refs[implementation_start_index:]:
        try:
            impl = corpus.get_record(ref, as_type=Implementation)
        except (FileNotFoundError, OSError, TypeError) as e:
            logger.warning(
                "swebench %s: could not load implementation %s: %s",
                inst.instance_id, ref, e,
            )
            unloaded += 1
            continue
        if not isinstance(impl, Implementation):
            continue
        cost_total += float(getattr(impl, "cost_usd", 0.0) or 0.0)
        files_modified.extend(impl.files_modified)
        classifications.append(impl.classification)

    # Sum Verdict.cost_usd across the spec's verdict_refs. The plan
    # workflow's critic/judge rounds, the breach workflow's
    # Breaker/Judge rounds, and the tier workflow's Phase-4 review
    # verdicts all stamp records here. Without this loop, outcome
    # cost only reflects implementer spend and a $5/round planner
    # loop is invisible to the budget cap.
    from assay.types.findings import Verdict
    for ref in spec.verdict_refs[verdict_start_index:]:
        try:
            verdict = corpus.get_record(ref, as_type=Verdict)
        except (FileNotFoundError, OSError, TypeError) as e:
            logger.warning(
                "swebench %s: could not load verdict %s: %s",
                inst.instance_id, ref, e,
            )
            unloaded += 1
            continue
        if not isinstance(verdict, Verdict):
            continue
        cost_total += float(getattr(verdict, "cost_usd", 0.0) or 0.0)

    classification = _reduce_implementation_classifications(classifications)
    latest_candidate = _latest_nonempty_diff_candidate(
        corpus,
        spec,
        inst.instance_id,
        implementation_start_index=implementation_start_index,
    )

    if workspace is not None and workspace.exists():
        capture = _capture_workspace_diff_report(workspace, inst)
    else:
        capture = _WorkspaceDiffCapture(provenance="workspace_unavailable")
    diff_bytes = capture.source_patch

    # Defender-oscillation salvage: when the live workspace diff is empty but
    # the corpus has a non-empty prior Implementation, normalize the latest
    # stored diff through the same source-only filter before deciding whether
    # it may be submitted. Candidate capture and submission are separate: a
    # failed terminal run retains a filtered candidate artifact, while
    # ``model_patch`` remains empty under verified_success_only.
    used_fallback = False
    candidate_origin = CandidateOrigin.WORKSPACE
    if not diff_bytes and latest_candidate is not None:
        fallback_patch = latest_candidate.patch
        used_fallback = True
        candidate_origin = CandidateOrigin.CORPUS_FALLBACK
        if workspace is not None and workspace.exists():
            fallback_capture = _filter_corpus_patch_to_source(
                workspace, inst, fallback_patch,
            )
        else:
            # Stored bytes require a repository to establish source-only scope.
            # Retain diagnostic size and provenance without a publishable patch.
            fallback_capture = _WorkspaceDiffCapture(
                raw_diff_bytes=len(fallback_patch),
                provenance="corpus_fallback_workspace_unavailable",
            )
        capture = _merge_diff_captures(capture, fallback_capture)
        diff_bytes = capture.source_patch

    candidate_producer: _StoredDiffCandidate | None = None
    if diff_bytes:
        if used_fallback:
            # ``diff_bytes`` was derived directly from this stored record and
            # normalized above, so its provenance is already exact.
            candidate_producer = latest_candidate
        elif workspace is not None and workspace.exists():
            candidate_producer = _latest_matching_diff_candidate(
                corpus,
                spec,
                inst,
                workspace,
                diff_bytes,
                implementation_start_index=implementation_start_index,
            )

    if candidate_producer is not None:
        # Record order describes attempt chronology, not necessarily the
        # candidate left in the workspace.  Once an exact producer is bound,
        # its gate status is authoritative for those bytes in both directions:
        # an unrelated later failure cannot poison them and an unrelated later
        # success cannot launder a failed candidate.
        classification = candidate_producer.classification
    if (
        architect_rc != 0
        and _candidate_is_submittable(classification)
    ):
        # A failed record for different bytes can remain attached for audit and
        # cost accounting after its retry checkpoint restores an earlier exact
        # success.  In that case the downstream architect/tier result, not the
        # discarded retry's gate result, is the active candidate's terminal
        # failure.  Newest-first exact producer matching keeps a failed record
        # authoritative when it describes these same normalized bytes.
        classification = f"architect_rc_{architect_rc}"

    candidate_capture: PredictionCandidateCapture | None = None
    if diff_bytes:
        candidate_files = capture.source_paths or tuple(dict.fromkeys(
            path for path in files_modified
            if not is_standard_test_path(path)
        ))
        candidate_capture = _persist_candidate_capture(
            corpus,
            inst,
            diff_bytes,
            files_modified=candidate_files,
            origin=candidate_origin,
            capture_status=CandidateCaptureStatus.SOURCE_FILTERED,
            producer=candidate_producer,
        )

    candidate_producer_is_submittable = bool(
        candidate_producer is not None
        and _candidate_is_submittable(candidate_producer.classification)
    )
    terminal_run_is_submittable = (
        architect_rc == 0
        and unloaded == 0
        and _candidate_is_submittable(classification)
        and candidate_capture is not None
        and candidate_producer_is_submittable
    )

    blockers: list[CandidateEligibilityBlocker] = []
    definitive_blocker = False
    if architect_rc != 0:
        blockers.append(CandidateEligibilityBlocker.ARCHITECT_FAILED)
        definitive_blocker = True
    if not _candidate_is_submittable(classification):
        blockers.append(
            CandidateEligibilityBlocker.TERMINAL_CLASSIFICATION_NOT_SUCCESS
        )
        definitive_blocker = True
    if candidate_capture is not None:
        if candidate_producer is None:
            blockers.append(
                CandidateEligibilityBlocker.CANDIDATE_PROVENANCE_UNBOUND
            )
        elif not candidate_producer_is_submittable:
            blockers.append(
                CandidateEligibilityBlocker
                .CANDIDATE_CLASSIFICATION_NOT_SUCCESS
            )
            definitive_blocker = True
    if unloaded:
        blockers.append(CandidateEligibilityBlocker.RECORD_ROLLUP_INCOMPLETE)
    if candidate_capture is None:
        capture_failed = (
            "_error" in capture.provenance
            or "_unavailable" in capture.provenance
        )
        if capture_failed:
            blockers.append(CandidateEligibilityBlocker.CAPTURE_FAILED)
        else:
            blockers.append(CandidateEligibilityBlocker.NO_NONEMPTY_CANDIDATE)
            definitive_blocker = True

    if definitive_blocker:
        eligibility_status = CandidateEligibilityStatus.INELIGIBLE
    elif blockers:
        eligibility_status = CandidateEligibilityStatus.UNKNOWN
    elif candidate_capture is not None and terminal_run_is_submittable:
        eligibility_status = CandidateEligibilityStatus.ELIGIBLE
    else:
        eligibility_status = CandidateEligibilityStatus.UNKNOWN
    candidate_eligibility = PredictionCandidateEligibility(
        status=eligibility_status,
        blockers=tuple(dict.fromkeys(blockers)),
    )

    if unloaded:
        submission_block_reason = "incomplete_record_rollup"
    elif not _candidate_is_submittable(classification):
        submission_block_reason = classification
    elif architect_rc != 0:
        submission_block_reason = f"architect_rc_{architect_rc}"
    elif candidate_producer is None:
        submission_block_reason = "candidate_provenance_unbound"
    elif not candidate_producer_is_submittable:
        submission_block_reason = candidate_producer.classification
    else:
        submission_block_reason = "unverified_terminal_candidate"

    override_requested = submit_any_nonempty_best_effort
    submitted = bool(candidate_capture) and (
        terminal_run_is_submittable or override_requested
    )
    override_used = bool(
        submitted
        and override_requested
        and candidate_eligibility.status != CandidateEligibilityStatus.ELIGIBLE
    )

    # A failed live candidate can survive a custom workflow that leaves the
    # workspace dirty. Do not bypass the same policy merely because capture
    # found it on disk rather than through Corpus fallback.
    if candidate_capture is not None and not submitted:
        if used_fallback:
            logger.warning(
                "swebench %s: retaining but not submitting stored diff_ref "
                "from Implementation %r (%d filtered bytes, terminal=%r, "
                "reason=%r); use --submit-any-nonempty-best-effort to opt in",
                inst.instance_id,
                latest_candidate.slice_id if latest_candidate is not None else "",
                candidate_capture.diff_bytes,
                classification,
                submission_block_reason,
            )
        else:
            logger.warning(
                "swebench %s: suppressing %d-byte live prediction "
                "(terminal=%r, reason=%r); candidate remains available in "
                "the Corpus",
                inst.instance_id,
                candidate_capture.diff_bytes,
                classification,
                submission_block_reason,
            )
        capture = dataclasses.replace(
            capture,
            source_patch=b"",
            provenance=(
                f"{capture.provenance}"
                f"{'_' if used_fallback else '+'}"
                f"suppressed_{submission_block_reason}"
            ),
        )
        diff_bytes = b""

    try:
        model_patch = diff_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        bad_byte_count = sum(1 for b in diff_bytes if b > 0x7f)
        logger.warning(
            "swebench %s: workspace diff contains non-UTF8 bytes "
            "(first bad position %d, ~%d non-ASCII bytes total); "
            "decoding with errors=replace — model_patch may not "
            "apply cleanly in the eval harness",
            inst.instance_id, e.start, bad_byte_count,
        )
        model_patch = diff_bytes.decode("utf-8", errors="replace")

    submission_decision = PredictionSubmissionDecision(
        policy=(
            PredictionSubmissionPolicy.BEST_EFFORT_ANY_NONEMPTY
            if override_requested
            else PredictionSubmissionPolicy.VERIFIED_SUCCESS_ONLY
        ),
        submitted=submitted,
        override_requested=override_requested,
        override_used=override_used,
        submitted_patch_ref=(
            _content_ref(model_patch.encode("utf-8")) if submitted else None
        ),
    )

    if (
        not diff_bytes
        and capture.raw_diff_bytes > 0
        and capture.stripped_test_paths
        and architect_rc == 0
        and not classification.startswith("hard_fail")
    ):
        classification = "soft_fail_test_only_changes"

    surfaced_files = (
        tuple(dict.fromkeys(
            path for path in files_modified if not is_standard_test_path(path)
        ))
        if diff_bytes else ()
    )

    return InstanceOutcome(
        instance_id=inst.instance_id,
        classification=classification,
        model_patch=model_patch,
        cost_usd=cost_total,
        diff_bytes=len(diff_bytes),
        # The terminal marker describes the surfaced source-only prediction.
        files_modified=surfaced_files,
        cost_incomplete=unloaded > 0,
        stripped_test_paths=capture.stripped_test_paths,
        stripped_test_diff_bytes=capture.stripped_test_diff_bytes,
        raw_workspace_diff_bytes=capture.raw_diff_bytes,
        diff_provenance=capture.provenance,
        candidate_capture=candidate_capture,
        candidate_eligibility=candidate_eligibility,
        submission_decision=submission_decision,
    )


def _architect_operation_id(instance_id: str) -> str:
    """Return a path-safe, stable saga operation id for one instance."""

    identity = canonical_fingerprint({
        "workflow": "swebench",
        "instance_id": instance_id,
    })
    return f"architect-{identity.removeprefix('sha256:')}"


def _architect_stopped_on_budget(
    parent_run_dir: Path | str,
    operation_id: str,
) -> bool:
    """Return whether the saga-bound Architect child ended on its budget.

    ``run_architect`` returns 1 for a budget stop instead of raising, so the
    exception handler that marks a truncated cascade never sees it. The typed
    terminal of the parent's own child is the only authority; a missing or
    misbound child proves nothing.
    """

    try:
        outcome = _outcome.child_outcome(parent_run_dir, operation_id)
    except ValueError:
        return False
    return (
        outcome.workflow == "architect"
        and outcome.terminal_class is _outcome.TerminalClass.BUDGET_EXHAUSTED
    )


def _architect_terminal_label(
    parent_run_dir: Path | str,
    operation_id: str,
) -> tuple[str | None, str | None]:
    """Return detail status and terminal class of the Architect child.

    The typed terminal of the parent's own saga-bound child is the only
    authority, as for ``_architect_stopped_on_budget``. A missing or misbound
    child, or a child of another workflow, labels nothing.
    """

    try:
        outcome = _outcome.child_outcome(parent_run_dir, operation_id)
    except ValueError:
        return None, None
    if outcome.workflow != "architect":
        return None, None
    return outcome.detail_status, outcome.terminal_class.value


# Architect terminals that report an open cost rollup. They share the
# inconclusive class with unjudged work, but every tier may have been judged
# and landed; the instance's ``cost_incomplete`` flag carries them.
_ARCHITECT_ACCOUNTING_STATUSES = frozenset({
    "budget_accounting_incomplete",
    "cost_incomplete",
    "incomplete_accounting",
})


def _architect_ended_unjudged(
    rc: int,
    detail_status: str | None,
    terminal_class: str | None,
) -> bool:
    """Whether a returned Architect failure is not a ruling on its work.

    True for an inconclusive, retryable or cancelled terminal of a failed
    return. An accounting terminal and a returned success are excluded; a
    budget stop has its own rule.
    """

    return (
        rc != 0
        and detail_status is not None
        and detail_status not in _ARCHITECT_ACCOUNTING_STATUSES
        and terminal_class
        in {item.value for item in _outcome.UNJUDGED_TERMINAL_CLASSES}
    )


def _architect_cost_for_operation(
    parent_run_dir: Path | str,
    operation_id: str,
) -> tuple[float | None, bool]:
    """Read spend only from the saga-bound Architect child.

    Planner invocations have no standalone Spec record, so summing only
    Implementation/Verdict refs undercounts every instance. Architect owns the
    end-to-end budget ledger and publishes ``authoritative_spent_usd``; use it
    as the instance/sweep accounting source. The boolean reports a missing or
    incomplete rollup so callers can fail closed instead of trusting a lower
    fallback total. A concurrent or partially-created run directory has no
    bearing on this lookup because both the typed outcome and directory come
    from the parent's durable saga.
    """

    try:
        outcome = _outcome.child_outcome(parent_run_dir, operation_id)
        run_dir = _outcome.child_run_dir(parent_run_dir, operation_id)
    except ValueError:
        return None, True
    if outcome.workflow != "architect":
        return None, True
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        cost_source = summary
        if "authoritative_spent_usd" not in cost_source:
            partial = summary.get("partial_summary")
            if isinstance(partial, dict):
                cost_source = partial
        value = float(cost_source["authoritative_spent_usd"])
        incomplete = bool(
            (cost_source.get("costs") or {}).get("rollup_incomplete", False)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, True
    if value < 0.0 or not math.isfinite(value):
        return None, True
    return value, incomplete


def _architect_usage_for_operation(
    parent_run_dir: Path | str,
    operation_id: str,
) -> tuple[int, int, int, bool]:
    """Read total tokens, planning calls, and transport state from one child.

    Exception-shaped child summaries retain the Architect's last explicitly
    written summary under ``partial_summary``.  Reading only through the
    parent's durable saga prevents an unrelated sibling run from supplying
    plausible-looking metrics for this instance.
    """

    try:
        outcome = _outcome.child_outcome(parent_run_dir, operation_id)
        run_dir = _outcome.child_run_dir(parent_run_dir, operation_id)
    except ValueError:
        return 0, 0, 0, False
    if outcome.workflow != "architect":
        return 0, 0, 0, False
    try:
        summary = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )
        source = summary
        if not isinstance(source.get("costs"), dict):
            partial = summary.get("partial_summary")
            if isinstance(partial, dict):
                source = partial
        costs = source["costs"]
        input_tokens = int(costs.get("total_input_tokens", 0))
        output_tokens = int(costs.get("total_output_tokens", 0))
        calls = int(costs.get("planning_backend_calls", 0))
        transport_failure = bool(source.get("transport_failure", False))
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return 0, 0, 0, False
    if input_tokens < 0 or output_tokens < 0 or calls < 0:
        return 0, 0, 0, transport_failure
    return input_tokens, output_tokens, calls, transport_failure


def _apply_architect_cost(
    outcome: InstanceOutcome,
    *,
    authoritative_cost: float | None,
    rollup_incomplete: bool,
    input_tokens: int = 0,
    output_tokens: int = 0,
    planning_backend_calls: int = 0,
    planning_transport_failure: bool = False,
) -> InstanceOutcome:
    """Replace the ref-only undercount with the architect ledger."""

    if authoritative_cost is None:
        return dataclasses.replace(
            outcome,
            cost_incomplete=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            planning_backend_calls=planning_backend_calls,
            planning_transport_failure=planning_transport_failure,
        )
    return dataclasses.replace(
        outcome,
        cost_usd=authoritative_cost,
        cost_incomplete=outcome.cost_incomplete or rollup_incomplete,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        planning_backend_calls=planning_backend_calls,
        planning_transport_failure=planning_transport_failure,
    )


@dataclasses.dataclass(frozen=True)
class _InstanceCorpusPreparation:
    """Lineage state established before one SWE-bench authoring attempt."""

    had_lineage: bool
    requires_replan: bool


def _prepare_instance_corpus(
    corpus: Corpus,
    starter_spec,
    *,
    instance_id: str,
) -> _InstanceCorpusPreparation:
    """Initialize one instance Corpus or validate its existing lineage.

    A retry always gets a fresh repository workspace, so callers use
    ``had_lineage`` to replay landed slices. When the product-owned authoring
    policy changed, this publishes an explicit next Spec version with the
    current managed constraints and no executable DAG. ``requires_replan``
    then makes the worker's architect call regenerate and re-review the plan.

    Unknown constraints, all record refs, and ``plan_policy`` are preserved.
    Existing manifests remain immutable. Production callers hold the
    per-instance task lease, which serializes policy migrations; the atomic
    Corpus transform rebases over unrelated record commits without losing
    their refs.

    Reusing an instance id for different task content is rejected. Without
    this identity check a new dataset revision could silently inherit an old
    plan and patch from the stable ``instances/<id>/corpus`` path.
    """
    latest = corpus.latest_spec()
    if latest is None:
        corpus.put_spec(starter_spec)
        return _InstanceCorpusPreparation(
            had_lineage=False,
            requires_replan=False,
        )
    expected = starter_spec.project_context
    actual = latest.project_context
    identity_matches = (
        actual.project == expected.project
        and actual.repo_url == expected.repo_url
        and actual.base_commit == expected.base_commit
        and latest.intent == starter_spec.intent
    )
    if not identity_matches:
        raise ValueError(
            f"SWE-bench instance {instance_id!r} conflicts with existing "
            "Corpus lineage (repo/base commit/problem statement changed); "
            "use a new spec root"
        )

    from crucible.products.swebench.spec_synthesis import (
        SWEBENCH_AUTHORING_POLICY_ID,
        reconcile_swebench_authoring_constraints,
        swebench_managed_authoring_constraints,
    )

    reconciled_constraints = reconcile_swebench_authoring_constraints(
        latest.constraints,
    )
    if reconciled_constraints == latest.constraints:
        return _InstanceCorpusPreparation(
            had_lineage=True,
            requires_replan=False,
        )

    managed_before = swebench_managed_authoring_constraints(
        latest.constraints,
    )
    managed_after = swebench_managed_authoring_constraints(
        reconciled_constraints,
    )
    reason = json.dumps(
        {
            "constraints_after": hash_record(reconciled_constraints),
            "constraints_before": hash_record(latest.constraints),
            "instance_id": instance_id,
            "managed_after": hash_record(managed_after),
            "managed_before": hash_record(managed_before),
            "plan_policy": (
                hash_record(latest.plan_policy)
                if latest.plan_policy is not None
                else None
            ),
            "policy": SWEBENCH_AUTHORING_POLICY_ID,
            "retired_tier_dag": (
                hash_record(latest.tier_dag)
                if latest.tier_dag is not None
                else None
            ),
            "schema": "crucible.swebench.authoring-policy-migration.v1",
            "source_spec": hash_record(latest),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    execution = exec_ctx_now("swebench")

    def _apply_policy(candidate):
        # Operate on Corpus.commit's rebased candidate, not the stale pre-read
        # Spec, so concurrently attached evidence refs remain intact.
        return dataclasses.replace(
            candidate,
            constraints=reconciled_constraints,
            tier_dag=None,
        )

    result = corpus.commit(
        lifecycle_event_factory=lambda committed_version: LifecycleEvent(
            kind="authoring_policy_migrated",
            reason=reason,
            affected_versions=(committed_version,),
            execution=execution,
        ),
        transform=_apply_policy,
    )
    logger.info(
        "swebench: migrated authoring policy for %s to %s at spec v%d; "
        "retired prior tier_dag and forcing replan",
        instance_id,
        SWEBENCH_AUTHORING_POLICY_ID,
        result.spec.version,
    )
    return _InstanceCorpusPreparation(
        had_lineage=True,
        requires_replan=True,
    )


def _swe_checkpoint_fingerprint(
    inst: SwebenchInstance,
    *,
    routing_source_sha256: str | None,
    max_turns: int,
    child_budget_usd: float | None,
    include_breach: bool,
    include_review: bool,
    enable_build_verify: bool,
    evaluate_inline: bool,
    eval_timeout: int,
    routed_backend: str,
    routed_model: str,
    image_digest: str,
    planner_count: int = DEFAULT_PLANNER_COUNT,
    submit_any_nonempty_best_effort: bool = False,
) -> str:
    """Bind a surfaced prediction to complete task and policy inputs."""

    crucible_root = Path(__file__).resolve().parents[1]
    product_root = crucible_root / "products" / "swebench"
    return canonical_fingerprint({
        "schema": "crucible.swebench.checkpoint.v4",
        # Includes the issue, base commit, gold patch/test identities and
        # expected test sets.  Only this hash, never hidden content, persists.
        "task": dataclasses.asdict(inst),
        "execution": {
            "max_turns": max_turns,
            "child_budget_usd": child_budget_usd,
            "planning": benchmark_planning_identity(
                planner_count,
                planning_budget_share=(
                    BENCHMARK_PLANNING_BUDGET_SHARE if planner_count > 1 else None
                ),
            ),
            "include_breach": include_breach,
            "include_review": include_review,
            "enable_build_verify": enable_build_verify,
            "evaluate_inline": evaluate_inline,
            "eval_timeout": eval_timeout,
            "routed_backend": routed_backend,
            "routed_model": routed_model,
            "routing_source_sha256": routing_source_sha256,
            "image_digest": image_digest,
            "submit_any_nonempty_best_effort": (
                submit_any_nonempty_best_effort
            ),
            "image_verification_policy": (
                image_verification_policy_identity()
            ),
        },
        "recipe": {
            "workflow": file_digest(Path(__file__)),
            "prompt": file_digest(product_root / "prompts.py"),
            "spec_synthesis": file_digest(product_root / "spec_synthesis.py"),
            "plan_normalization": file_digest(
                product_root / "plan_normalization.py"
            ),
            "test_path_policy": file_digest(product_root / "paths.py"),
            "evaluator_compat": file_digest(
                product_root / "evaluator_compat.py"
            ),
            "visible_test_runner": file_digest(
                product_root / "runners" / "swebench_pytest.py"
            ),
            "pristine_verifier": file_digest(
                product_root / "pristine_verifier.py"
            ),
            "instance_container": file_digest(
                product_root / "instance_container.py"
            ),
            "repo_workspace": file_digest(
                product_root / "repo_workspace.py"
            ),
            "backend_runtime": backend_recipe_identity(
                crucible_root,
                digest_file=file_digest,
            ),
            "planning_runtime": planning_recipe_identity(
                crucible_root,
                digest_file=file_digest,
            ),
            "rubrics": file_set_fingerprint(
                crucible_root / "rubrics", "*.yaml",
            ),
            "build_identity": {
                "python_version": file_digest(
                    crucible_root / ".python-version"
                ),
                "pyproject": file_digest(crucible_root / "pyproject.toml"),
                "uv_lock": file_digest(crucible_root / "uv.lock"),
            },
        },
    })


def _pool_checkpoint_fingerprint(
    payload: _PoolPayload, *, image_digest: str,
) -> str:
    base = _swe_checkpoint_fingerprint(
        payload.instance,
        routing_source_sha256=payload.routing_source_sha256,
        max_turns=payload.max_turns,
        child_budget_usd=payload.budget_usd,
        include_breach=payload.include_breach,
        include_review=payload.include_review,
        enable_build_verify=payload.enable_build_verify,
        evaluate_inline=payload.evaluate_inline,
        eval_timeout=payload.eval_timeout,
        routed_backend=payload.routed_backend,
        routed_model=payload.routed_model,
        image_digest=image_digest,
        planner_count=payload.planner_count,
        submit_any_nonempty_best_effort=(
            payload.submit_any_nonempty_best_effort
        ),
    )
    return canonical_fingerprint({
        "task_fingerprint": base,
        "routing_fingerprint": payload.routing_fingerprint,
    })


def _effective_prediction_metadata(
    outcome: InstanceOutcome,
) -> tuple[
    PredictionCandidateCapture | None,
    PredictionCandidateEligibility,
    PredictionSubmissionDecision,
]:
    """Return complete metadata for new outcomes and legacy direct callers."""

    candidate = outcome.candidate_capture
    if candidate is None and outcome.model_patch:
        patch = outcome.model_patch.encode("utf-8")
        candidate = PredictionCandidateCapture(
            source_patch=patch,
            diff_ref=_content_ref(patch),
            sha256=_content_sha256(patch),
            diff_bytes=len(patch),
            files_modified=outcome.files_modified,
            origin=CandidateOrigin.SURFACED_OUTCOME,
            capture_status=CandidateCaptureStatus.SOURCE_FILTERED,
            producer_classification=outcome.classification or None,
        )

    eligibility = outcome.candidate_eligibility
    if eligibility is None:
        if candidate is None:
            eligibility = PredictionCandidateEligibility(
                status=CandidateEligibilityStatus.INELIGIBLE,
                blockers=(
                    CandidateEligibilityBlocker.NO_NONEMPTY_CANDIDATE,
                ),
            )
        elif _candidate_is_submittable(outcome.classification):
            eligibility = PredictionCandidateEligibility(
                status=CandidateEligibilityStatus.ELIGIBLE,
            )
        else:
            eligibility = PredictionCandidateEligibility(
                status=CandidateEligibilityStatus.INELIGIBLE,
                blockers=(
                    CandidateEligibilityBlocker
                    .TERMINAL_CLASSIFICATION_NOT_SUCCESS,
                ),
            )

    decision = outcome.submission_decision
    if decision is None:
        submitted = bool(outcome.model_patch)
        override_used = bool(
            submitted
            and eligibility.status != CandidateEligibilityStatus.ELIGIBLE
        )
        decision = PredictionSubmissionDecision(
            policy=(
                PredictionSubmissionPolicy.BEST_EFFORT_ANY_NONEMPTY
                if override_used
                else PredictionSubmissionPolicy.VERIFIED_SUCCESS_ONLY
            ),
            submitted=submitted,
            override_requested=override_used,
            override_used=override_used,
            submitted_patch_ref=(
                _content_ref(outcome.model_patch.encode("utf-8"))
                if submitted else None
            ),
        )

    if candidate is not None:
        if (
            candidate.diff_ref != _content_ref(candidate.source_patch)
            or candidate.sha256 != _content_sha256(candidate.source_patch)
            or candidate.diff_bytes != len(candidate.source_patch)
        ):
            raise ValueError("candidate capture metadata does not match its patch")
    if eligibility.status == CandidateEligibilityStatus.ELIGIBLE and (
        candidate is None or eligibility.blockers
    ):
        raise ValueError("eligible candidate must exist and have no blockers")
    if decision.submitted != bool(outcome.model_patch):
        raise ValueError("submission decision disagrees with model_patch")
    expected_submitted_ref = (
        _content_ref(outcome.model_patch.encode("utf-8"))
        if decision.submitted else None
    )
    if decision.submitted_patch_ref != expected_submitted_ref:
        raise ValueError("submitted patch ref disagrees with model_patch")
    if decision.override_used and (
        not decision.override_requested
        or decision.policy not in {
            PredictionSubmissionPolicy.BEST_EFFORT_ANY_NONEMPTY,
            PredictionSubmissionPolicy.LEGACY_V3,
        }
    ):
        raise ValueError("submission override flags disagree with policy")
    if (
        decision.submitted
        and eligibility.status != CandidateEligibilityStatus.ELIGIBLE
        and (
            not decision.override_used
            or decision.policy not in {
                PredictionSubmissionPolicy.BEST_EFFORT_ANY_NONEMPTY,
                PredictionSubmissionPolicy.LEGACY_V3,
            }
        )
    ):
        raise ValueError("ineligible submission requires an explicit override")
    return candidate, eligibility, decision


def _validate_pristine_verification_contract(
    outcome: InstanceOutcome,
    *,
    base_commit: str,
    image_digest: str,
    candidate: PredictionCandidateCapture | None,
    submission: PredictionSubmissionDecision,
) -> None:
    """Fail closed on persisted submissions that bypass the final gate."""

    report = outcome.pristine_verification
    if report is not None:
        if report.schema != "crucible.swebench.pristine-verification.v1":
            raise ValueError("unsupported pristine report schema")
        if report.base_commit != base_commit:
            raise ValueError("pristine report base commit does not match task")
        if not image_digest or report.image_digest != image_digest:
            raise ValueError("pristine report image digest does not match marker")
        if report.status == PristineVerificationStatus.PASSED:
            if candidate is None:
                raise ValueError("passed pristine report requires a candidate")
            if (
                report.patch_sha256 != candidate.sha256
                or report.patch_bytes != candidate.diff_bytes
                or outcome.model_patch.encode("utf-8")
                != candidate.source_patch
            ):
                raise ValueError(
                    "passed pristine report does not bind the submitted patch"
                )

    if submission.submitted:
        if report is None:
            raise ValueError(
                "submitted v5 prediction requires pristine verification"
            )
        if report.status != PristineVerificationStatus.PASSED and not (
            submission.policy
            == PredictionSubmissionPolicy.BEST_EFFORT_ANY_NONEMPTY
            and submission.override_requested
            and submission.override_used
        ):
            raise ValueError(
                "failed pristine verification requires best-effort override"
            )
    elif report is not None and (
        report.status == PristineVerificationStatus.PASSED
    ):
        raise ValueError("passed pristine verification cannot be suppressed")


def _candidate_capture_payload(
    candidate: PredictionCandidateCapture | None,
) -> dict[str, object] | None:
    if candidate is None:
        return None
    return {
        "diff_ref": candidate.diff_ref,
        "sha256": candidate.sha256,
        "diff_bytes": candidate.diff_bytes,
        "files_modified": list(candidate.files_modified),
        "origin": candidate.origin.value,
        "capture_status": candidate.capture_status.value,
        "producer_slice_id": candidate.producer_slice_id,
        "producer_classification": candidate.producer_classification,
    }


def _candidate_eligibility_payload(
    eligibility: PredictionCandidateEligibility,
) -> dict[str, object]:
    return {
        "status": eligibility.status.value,
        "blockers": [blocker.value for blocker in eligibility.blockers],
    }


def _submission_decision_payload(
    decision: PredictionSubmissionDecision,
) -> dict[str, object]:
    return {
        "policy": decision.policy.value,
        "submitted": decision.submitted,
        "override_requested": decision.override_requested,
        "override_used": decision.override_used,
        "submitted_patch_ref": decision.submitted_patch_ref,
    }


def _prediction_metadata_summary(outcome: InstanceOutcome) -> dict[str, object]:
    candidate, eligibility, submission = _effective_prediction_metadata(outcome)
    return {
        "prediction_capture": {
            "provenance": outcome.diff_provenance,
            "raw_diff_bytes": outcome.raw_workspace_diff_bytes,
            "source_diff_bytes": outcome.diff_bytes,
            "stripped_test_paths": list(outcome.stripped_test_paths),
            "stripped_test_diff_bytes": outcome.stripped_test_diff_bytes,
        },
        "candidate_capture": _candidate_capture_payload(candidate),
        "candidate_eligibility": _candidate_eligibility_payload(eligibility),
        "submission_decision": _submission_decision_payload(submission),
        "pristine_verification": (
            outcome.pristine_verification.to_payload()
            if outcome.pristine_verification is not None else None
        ),
    }


def _build_surfaced_prediction_marker(
    corpus: Corpus,
    inst: SwebenchInstance,
    outcome: InstanceOutcome,
    *,
    backend_name: str,
    model: str,
    checkpoint_fingerprint: str = "",
    image_digest: str = "",
    assigned_budget_usd: float | None = None,
) -> Implementation:
    """Build the exact surfaced prediction's terminal Corpus record.

    Tier Implementations are slice-local and may include intermediate or
    cumulative diffs. Resume needs the exact leaderboard patch captured at
    the end of the cascade. Returning the marker separately lets the hidden
    evaluation path commit it atomically with the authoritative measurements.
    """
    if not checkpoint_fingerprint:
        checkpoint_fingerprint = canonical_fingerprint({
            "schema": "crucible.swebench.direct-persist.v1",
            "task": dataclasses.asdict(inst),
            "backend": backend_name,
            "model": model,
            "image_digest": image_digest,
            "assigned_budget_usd": assigned_budget_usd,
        })
    candidate, eligibility, submission = _effective_prediction_metadata(outcome)
    _validate_pristine_verification_contract(
        outcome,
        base_commit=inst.base_commit,
        image_digest=image_digest,
        candidate=candidate,
        submission=submission,
    )
    if candidate is not None:
        persisted_candidate_ref = str(corpus.put_artifact(
            candidate.source_patch,
            kind="diffs",
            name_hint=f"{inst.instance_id}_candidate.diff",
        ))
        if persisted_candidate_ref != candidate.diff_ref:
            raise ValueError("persisted candidate ref does not match metadata")

    diff_ref = corpus.put_artifact(
        outcome.model_patch.encode("utf-8"),
        kind="diffs",
        name_hint=f"{inst.instance_id}_surfaced.diff",
    )
    if submission.submitted_patch_ref is not None and (
        str(diff_ref) != submission.submitted_patch_ref
    ):
        raise ValueError("persisted submitted patch ref does not match decision")
    transcript_ref = corpus.put_artifact(
        json.dumps({
            "schema_version": "crucible.swebench.prediction.v5",
            "kind": "swebench_surfaced_prediction",
            "instance_id": inst.instance_id,
            "classification": outcome.classification,
            "eval_status": outcome.eval_status,
            "cost_usd": outcome.cost_usd,
            "cost_incomplete": outcome.cost_incomplete,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "image_digest": image_digest,
            "assigned_budget_usd": assigned_budget_usd,
            "prediction_capture": {
                "provenance": outcome.diff_provenance,
                "raw_diff_bytes": outcome.raw_workspace_diff_bytes,
                "source_diff_bytes": outcome.diff_bytes,
                "stripped_test_paths": list(outcome.stripped_test_paths),
                "stripped_test_diff_bytes": (
                    outcome.stripped_test_diff_bytes
                ),
            },
            "candidate_capture": _candidate_capture_payload(candidate),
            "candidate_eligibility": _candidate_eligibility_payload(
                eligibility,
            ),
            "submission_decision": _submission_decision_payload(submission),
            "pristine_verification": (
                outcome.pristine_verification.to_payload()
                if outcome.pristine_verification is not None else None
            ),
        }, sort_keys=True).encode("utf-8"),
        kind="transcripts",
        name_hint=f"{inst.instance_id}_surfaced.json",
    )
    return Implementation(
        slice_id=f"swebench/{inst.instance_id}",
        base_commit=inst.base_commit,
        result_commit=None,
        diff_ref=diff_ref,
        transcript_ref=transcript_ref,
        files_modified=outcome.files_modified,
        scope_escapes=(),
        classification=outcome.classification,
        execution=ExecutionContext(
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds",
            ),
            machine=socket.gethostname(),
            cwd=str(corpus.spec_dir),
            toolchain={"python": platform.python_version()},
            backend=backend_name,
            model=model,
            recipe="swebench_surface_prediction",
            recipe_commit=None,
            gpu_model=None,
            run_id=None,
        ),
        cost_usd=0.0,
    )


def _persist_surfaced_prediction(
    corpus: Corpus,
    inst: SwebenchInstance,
    outcome: InstanceOutcome,
    *,
    backend_name: str,
    model: str,
    checkpoint_fingerprint: str = "",
    image_digest: str = "",
    assigned_budget_usd: float | None = None,
) -> None:
    """Durably publish a terminal marker when no hidden-eval commit does it."""

    marker = _build_surfaced_prediction_marker(
        corpus,
        inst,
        outcome,
        backend_name=backend_name,
        model=model,
        checkpoint_fingerprint=checkpoint_fingerprint,
        image_digest=image_digest,
        assigned_budget_usd=assigned_budget_usd,
    )
    corpus.commit(implementations=(marker,))

def _partition_for_resume(
    instances: tuple[SwebenchInstance, ...],
    prior_predictions: dict[str, _PriorPrediction],
) -> tuple[tuple[SwebenchInstance, ...], tuple[SwebenchInstance, ...]]:
    """Return ``(to_process, skipped)`` using the swebench-scoped
    prior-prediction map.

    An instance is skipped only when its prior classification
    is skip-worthy (success or a terminal no-source-change soft failure).
    Hard-fail
    records — timeouts, transport errors, empty-diff-on-mutating
    slice, workspace errors — get retried because they represent
    operator-paid runs that never produced a real attempt.
    ``--rerun`` bypasses this and re-implements everything.
    """
    to_process: list[SwebenchInstance] = []
    skipped: list[SwebenchInstance] = []
    for inst in instances:
        prior = prior_predictions.get(inst.instance_id)
        if prior is not None and _is_skip_worthy_prior(prior.classification):
            skipped.append(inst)
        else:
            to_process.append(inst)
    return tuple(to_process), tuple(skipped)


@dataclasses.dataclass(frozen=True)
class _PriorPrediction:
    """One previously-implemented instance's surfaced state.

    Carries the model_patch + the model that produced it + the
    classification so the resumability decision can re-attempt
    hard-failed instances while skipping success / soft-fail ones,
    and cross-model comparisons get correct attribution on the
    resumed entries.
    """
    model_patch: str
    model: str
    classification: str
    eval_status: str | None
    cost_usd: float
    cost_incomplete: bool
    checkpoint_fingerprint: str
    image_digest: str
    assigned_budget_usd: float | None
    stripped_test_paths: tuple[str, ...] = ()
    stripped_test_diff_bytes: int = 0
    raw_workspace_diff_bytes: int = 0
    diff_provenance: str = "none"
    candidate_capture: PredictionCandidateCapture | None = None
    candidate_eligibility: PredictionCandidateEligibility | None = None
    submission_decision: PredictionSubmissionDecision | None = None
    pristine_verification: PristineVerificationReport | None = None


def _existing_predictions_by_id(
    spec_root: Path,
    instance_ids: tuple[str, ...],
) -> dict[str, _PriorPrediction]:
    """Load terminal predictions from each instance's own Corpus.

    Live cascades persist under
    ``<spec_root>/instances/<instance_id>/corpus``. Scanning the root
    Corpus (the former implementation) can never see those refs and made
    resume a no-op. Only the reserved terminal marker written by
    ``_persist_surfaced_prediction`` is accepted; intermediate slice
    Implementations are intentionally ignored.

    Tolerates the corpus deserializer quirk where nested
    ``execution`` lands as a dict at runtime instead of an
    ExecutionContext.
    """
    out: dict[str, _PriorPrediction] = {}
    for instance_id in instance_ids:
        corpus_dir = spec_root / "instances" / instance_id / "corpus"
        if not corpus_dir.is_dir():
            continue
        corpus = Corpus(corpus_dir)
        try:
            spec = corpus.latest_spec()
        except Exception as e:  # noqa: BLE001 — corrupt lineage retries
            logger.warning(
                "swebench resume: cannot read Corpus for %s: %s; retrying",
                instance_id, e,
            )
            continue
        if spec is None:
            continue
        marker: Implementation | None = None
        for ref in reversed(spec.implementation_refs):
            try:
                candidate = corpus.get_record(ref, as_type=Implementation)
            except Exception:  # noqa: BLE001 — incomplete attempt retries
                continue
            if (
                isinstance(candidate, Implementation)
                and candidate.slice_id == f"swebench/{instance_id}"
            ):
                marker = candidate
                break
        if marker is None:
            continue
        try:
            diff_bytes = corpus.get_artifact(marker.diff_ref)
        except (FileNotFoundError, OSError):
            # A terminal marker without its patch is not resumable.
            continue
        try:
            transcript = json.loads(
                corpus.get_artifact(marker.transcript_ref).decode("utf-8")
            )
            marker_schema = transcript.get("schema_version")
            if (
                marker_schema not in {
                    "crucible.swebench.prediction.v3",
                    "crucible.swebench.prediction.v4",
                    "crucible.swebench.prediction.v5",
                }
                or transcript.get("instance_id") != instance_id
                or not isinstance(transcript.get("checkpoint_fingerprint"), str)
                or not transcript["checkpoint_fingerprint"]
                or not isinstance(transcript.get("cost_incomplete"), bool)
            ):
                raise ValueError("invalid surfaced prediction transcript")
            prior_cost = float(transcript["cost_usd"])
            if prior_cost < 0.0 or not math.isfinite(prior_cost):
                raise ValueError("invalid surfaced prediction cost")
            raw_budget = transcript["assigned_budget_usd"]
            assigned_budget_usd = (
                None if raw_budget is None else float(raw_budget)
            )
            if assigned_budget_usd is not None and (
                assigned_budget_usd <= 0.0
                or not math.isfinite(assigned_budget_usd)
            ):
                raise ValueError("invalid surfaced prediction assigned budget")
            capture_payload = transcript.get("prediction_capture")
            stripped_test_paths: tuple[str, ...] = ()
            stripped_test_diff_bytes = 0
            raw_workspace_diff_bytes = 0
            diff_provenance = "none"
            if capture_payload is not None:
                if not isinstance(capture_payload, dict):
                    raise ValueError("invalid prediction_capture payload")
                raw_paths = capture_payload.get("stripped_test_paths", [])
                if not isinstance(raw_paths, list) or not all(
                    isinstance(path, str) for path in raw_paths
                ):
                    raise ValueError("invalid stripped_test_paths")

                def _capture_nonnegative_int(
                    key: str,
                    payload: dict = capture_payload,
                ) -> int:
                    value = payload.get(key, 0)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ValueError(f"invalid prediction_capture {key}")
                    return value

                provenance = capture_payload.get("provenance", "none")
                if not isinstance(provenance, str):
                    raise ValueError("invalid prediction_capture provenance")
                stripped_test_paths = tuple(raw_paths)
                stripped_test_diff_bytes = _capture_nonnegative_int(
                    "stripped_test_diff_bytes",
                )
                raw_workspace_diff_bytes = _capture_nonnegative_int(
                    "raw_diff_bytes",
                )
                diff_provenance = provenance

            decoded_patch = diff_bytes.decode("utf-8", errors="replace")
            pristine_verification: PristineVerificationReport | None = None
            if marker_schema in {
                "crucible.swebench.prediction.v4",
                "crucible.swebench.prediction.v5",
            }:
                raw_candidate = transcript.get("candidate_capture")
                candidate_capture: PredictionCandidateCapture | None = None
                if raw_candidate is not None:
                    if not isinstance(raw_candidate, dict):
                        raise ValueError("invalid candidate_capture payload")
                    candidate_ref = raw_candidate.get("diff_ref")
                    candidate_sha256 = raw_candidate.get("sha256")
                    candidate_size = raw_candidate.get("diff_bytes")
                    candidate_files = raw_candidate.get("files_modified")
                    producer_slice_id = raw_candidate.get("producer_slice_id")
                    producer_classification = raw_candidate.get(
                        "producer_classification"
                    )
                    if (
                        not isinstance(candidate_ref, str)
                        or not isinstance(candidate_sha256, str)
                        or isinstance(candidate_size, bool)
                        or not isinstance(candidate_size, int)
                        or candidate_size < 0
                        or not isinstance(candidate_files, list)
                        or not all(isinstance(path, str) for path in candidate_files)
                        or producer_slice_id is not None
                        and not isinstance(producer_slice_id, str)
                        or producer_classification is not None
                        and not isinstance(producer_classification, str)
                    ):
                        raise ValueError("invalid candidate_capture fields")
                    candidate_blob = corpus.get_artifact(candidate_ref)
                    candidate_capture = PredictionCandidateCapture(
                        source_patch=candidate_blob,
                        diff_ref=candidate_ref,
                        sha256=candidate_sha256,
                        diff_bytes=candidate_size,
                        files_modified=tuple(candidate_files),
                        origin=CandidateOrigin(raw_candidate.get("origin")),
                        capture_status=CandidateCaptureStatus(
                            raw_candidate.get("capture_status")
                        ),
                        producer_slice_id=producer_slice_id,
                        producer_classification=producer_classification,
                    )

                raw_eligibility = transcript.get("candidate_eligibility")
                if not isinstance(raw_eligibility, dict):
                    raise ValueError("invalid candidate_eligibility payload")
                raw_blockers = raw_eligibility.get("blockers")
                if not isinstance(raw_blockers, list):
                    raise ValueError("invalid candidate eligibility blockers")
                candidate_eligibility = PredictionCandidateEligibility(
                    status=CandidateEligibilityStatus(
                        raw_eligibility.get("status")
                    ),
                    blockers=tuple(
                        CandidateEligibilityBlocker(blocker)
                        for blocker in raw_blockers
                    ),
                )

                raw_submission = transcript.get("submission_decision")
                if not isinstance(raw_submission, dict):
                    raise ValueError("invalid submission_decision payload")
                submitted = raw_submission.get("submitted")
                override_requested = raw_submission.get("override_requested")
                override_used = raw_submission.get("override_used")
                submitted_patch_ref = raw_submission.get("submitted_patch_ref")
                if (
                    not isinstance(submitted, bool)
                    or not isinstance(override_requested, bool)
                    or not isinstance(override_used, bool)
                    or submitted_patch_ref is not None
                    and not isinstance(submitted_patch_ref, str)
                ):
                    raise ValueError("invalid submission_decision fields")
                submission_decision = PredictionSubmissionDecision(
                    policy=PredictionSubmissionPolicy(
                        raw_submission.get("policy")
                    ),
                    submitted=submitted,
                    override_requested=override_requested,
                    override_used=override_used,
                    submitted_patch_ref=submitted_patch_ref,
                )
                if marker_schema == "crucible.swebench.prediction.v5":
                    if "pristine_verification" not in transcript:
                        raise ValueError(
                            "missing pristine_verification payload"
                        )
                    raw_pristine = transcript["pristine_verification"]
                    if raw_pristine is not None:
                        if not isinstance(raw_pristine, dict):
                            raise ValueError(
                                "invalid pristine_verification payload"
                            )
                        pristine_verification = (
                            PristineVerificationReport.from_payload(
                                raw_pristine
                            )
                        )
                restored_outcome = InstanceOutcome(
                    instance_id=instance_id,
                    classification=marker.classification or "",
                    model_patch=decoded_patch,
                    cost_usd=prior_cost,
                    diff_bytes=len(diff_bytes),
                    files_modified=marker.files_modified,
                    candidate_capture=candidate_capture,
                    candidate_eligibility=candidate_eligibility,
                    submission_decision=submission_decision,
                    pristine_verification=pristine_verification,
                )
                _effective_prediction_metadata(restored_outcome)
                if marker_schema == "crucible.swebench.prediction.v5":
                    _validate_pristine_verification_contract(
                        restored_outcome,
                        base_commit=marker.base_commit,
                        image_digest=str(transcript.get("image_digest") or ""),
                        candidate=candidate_capture,
                        submission=submission_decision,
                    )
            else:
                # V3 had only the surfaced patch. Synthesize conservative
                # metadata without changing its historical resume behavior.
                (
                    candidate_capture,
                    candidate_eligibility,
                    submission_decision,
                ) = _effective_prediction_metadata(InstanceOutcome(
                    instance_id=instance_id,
                    classification=marker.classification or "",
                    model_patch=decoded_patch,
                    cost_usd=prior_cost,
                    diff_bytes=len(diff_bytes),
                    files_modified=marker.files_modified,
                ))
                submission_decision = dataclasses.replace(
                    submission_decision,
                    policy=PredictionSubmissionPolicy.LEGACY_V3,
                )
        except (
            KeyError, OSError, TypeError, UnicodeDecodeError,
            json.JSONDecodeError, ValueError,
        ):
            logger.warning(
                "swebench resume: invalid terminal marker for %s; retrying",
                instance_id,
            )
            continue
        prior_model = marker.execution.model or ""
        out[instance_id] = _PriorPrediction(
            model_patch=decoded_patch,
            model=prior_model,
            classification=marker.classification or "",
            eval_status=transcript.get("eval_status"),
            cost_usd=prior_cost,
            cost_incomplete=transcript["cost_incomplete"],
            checkpoint_fingerprint=transcript["checkpoint_fingerprint"],
            image_digest=str(transcript.get("image_digest") or ""),
            assigned_budget_usd=assigned_budget_usd,
            stripped_test_paths=stripped_test_paths,
            stripped_test_diff_bytes=stripped_test_diff_bytes,
            raw_workspace_diff_bytes=raw_workspace_diff_bytes,
            diff_provenance=diff_provenance,
            candidate_capture=candidate_capture,
            candidate_eligibility=candidate_eligibility,
            submission_decision=submission_decision,
            pristine_verification=pristine_verification,
        )
    return out


_SWEBENCH_REUSABLE_CLASSIFICATIONS = frozenset({
    "success", "soft_fail_no_changes", "soft_fail_test_only_changes",
})
_SWEBENCH_RETRYABLE_PREFIXES = (
    "hard_fail_", "architect_rc_",
)


def _is_skip_worthy_prior(classification: str) -> bool:
    """Return True when a prior Implementation's classification means
    "the agent reached a terminal answer; don't redo the work."

    The skip set is intentionally narrow: only ``success`` (clean
    run with a diff) and ``soft_fail_no_changes`` (agent
    legitimately declined an instance with no fix needed under
    verify_mode=skip). Every ``hard_fail_*`` retries by default —
    timeouts, transport errors, empty-diff-on-mutating slice,
    workspace errors — those are infrastructure / agent failures
    where the operator paid but didn't get a real attempt.

    ``--rerun`` bypasses this entirely and re-implements all prior
    records.
    """
    return classification in _SWEBENCH_REUSABLE_CLASSIFICATIONS


_INSTANCE_ROLLUP_ACCOUNTING_DEBT = (
    "token_usage_complete",
    "cost_complete",
)


def _note_instance_rollup_debt(
    costs: CostAccumulator,
    outcome: InstanceOutcome,
) -> None:
    """Make an incomplete child rollup non-authorizing at the outer ledger."""

    if outcome.cost_incomplete:
        # ``InstanceOutcome`` has one aggregate incompleteness bit rather than
        # field-level provenance. Conservatively deny both token and cost
        # authority while retaining every numeric lower bound already found.
        costs.note_accounting_debt(
            role=outcome.instance_id,
            missing=_INSTANCE_ROLLUP_ACCOUNTING_DEBT,
        )


def _record_instance_cost(
    costs: CostAccumulator,
    outcome: InstanceOutcome,
    *,
    backend_name: str,
    model: str,
) -> None:
    """Record one numeric instance lower bound and its aggregate debt."""

    costs.record(
        role=outcome.instance_id,
        usage=outcome,
        backend_name=backend_name,
        model=model,
    )
    _note_instance_rollup_debt(costs, outcome)


def _run_implement_serial(
    filtered: tuple[SwebenchInstance, ...],
    *,
    args,
    run_dir: Path,
    handle,
    predictions_path: Path,
    routing,
    costs: "CostAccumulator",
) -> tuple[list[InstanceOutcome], float, bool, bool]:
    """Return outcomes, spend, budget halt, and accounting failure.

    Each instance's Prediction is appended to ``predictions_path``
    immediately so a Ctrl-C / crash mid-run doesn't lose results
    from earlier instances. The backend is constructed per-instance
    with the instance's log_dir so SDK debug dumps don't collide in
    run_dir/sdk_*.json across instances — without this, a
    500-instance run produces an unattributed pile of
    sdk_*_<backend>.json files.
    """
    # A single-worker executor now owns both serial and fan-out operation so
    # every path uses the same task lease, under-lock resume recheck, budget
    # reservation, and merged prediction publication protocol.
    return _run_implement_parallel(
        filtered,
        args=args,
        run_dir=run_dir,
        handle=handle,
        predictions_path=predictions_path,
        routing=routing,
        costs=costs,
    )


def _run_implement_parallel(
    filtered: tuple[SwebenchInstance, ...],
    *,
    args,
    run_dir: Path,
    handle,
    predictions_path: Path,
    routing,
    costs: "CostAccumulator",
) -> tuple[list[InstanceOutcome], float, bool, bool]:
    """ThreadPoolExecutor-backed implement loop.

    Each worker thread holds its own standalone per-instance workspace,
    instance-local Corpus, and role-routed backend pool. ``run_architect``
    and ``run_tier`` receive the workspace explicitly; no process-cwd state
    is used for agent tools or diff capture.

    The acquisition cache (when ``--repo-cache-dir`` is set) is shared
    across threads, but each authoring checkout has its own object database
    and exposes exactly the pinned commit. Routing config is parsed once by
    the parent and
    handed to every worker via ``_PoolPayload.routing`` so N threads
    don't redundantly re-parse the same YAML.

    Concurrent corpus writes from threads are safe:
    ``Corpus.put_record`` / ``put_artifact`` use content-addressed
    sha256 filenames + ``_atomic_write`` (tempfile + ``os.replace``).

    The scheduler submits at most ``pool_size`` instances and reserves a
    fair share of the remaining aggregate budget for each. Completed
    workers release unused reservation before another task is launched.
    Once the cap prevents new submissions, every in-flight future is still
    drained, persisted, and cost-accounted; no paid completion is dropped.
    """
    from concurrent.futures import (
        FIRST_COMPLETED,
        ThreadPoolExecutor,
        wait,
    )

    if _docker_available():
        # Refuse authoring below the required disk-headroom floor.
        _check_disk_headroom()

    pool_size = min(args.max_concurrent_instances, len(filtered))
    logger.info(
        "swebench: fanning out %d instances across %d thread(s)",
        len(filtered), pool_size,
    )

    # Routing was pre-resolved on the parent thread (see main()) and
    # passed in as a frozen dataclass; safe to share across worker
    # threads.
    implementer_route = routing.for_role("implementer")
    routed_model = implementer_route.model or "unknown"
    if implementer_route.backend is None:
        raise ValueError("implementer routing has no backend")
    routed_backend_name = implementer_route.backend
    try:
        resolved_routing_fingerprint = routing_fingerprint(routing)
        resolved_routing_source_sha256 = routing_source_fingerprint(routing)
    except (AttributeError, TypeError, ValueError):
        # Lightweight test/operator adapters may expose only ``for_role``.
        # Production ``load_routing`` always returns BackendRouting.
        resolved_routing_fingerprint = canonical_fingerprint({
            "implementer": {
                "backend": routed_backend_name,
                "model": routed_model,
            },
        })
        resolved_routing_source_sha256 = None

    payloads_by_id: dict[str, _PoolPayload] = {}
    for inst in filtered:
        instance_log_dir = run_dir / "instances" / inst.instance_id
        instance_log_dir.mkdir(parents=True, exist_ok=True)
        payloads_by_id[inst.instance_id] = _PoolPayload(
            instance=inst,
            workspace_root_spec_dir=str(args.spec),
            backend_config_path=(
                str(args.backend_config) if args.backend_config else None
            ),
            max_turns=args.max_turns,
            budget_usd=args.budget_usd_per_instance,
            planner_count=args.planner_count,
            repo_cache_dir=(
                str(args.repo_cache_dir) if args.repo_cache_dir else None
            ),
            log_dir=str(instance_log_dir),
            parent_run_dir=str(run_dir),
            include_breach=args.include_breach,
            include_review=args.include_review,
            enable_build_verify=args.enable_build_verify,
            evaluate_inline=args.evaluate_inline,
            eval_timeout=args.eval_timeout,
            routed_model=routed_model,
            routed_backend=routed_backend_name,
            routing=routing,
            restart=(
                args.rerun
                or (
                    Path(args.spec) / "instances" / inst.instance_id
                    / "corpus" / "versions"
                ).is_dir()
            ),
            rerun=args.rerun,
            rerun_status=tuple(getattr(args, "rerun_status", ())),
            routing_fingerprint=resolved_routing_fingerprint,
            routing_source_sha256=resolved_routing_source_sha256,
            submit_any_nonempty_best_effort=(
                getattr(args, "submit_any_nonempty_best_effort", False)
            ),
        )

    outcomes_by_id: dict[str, InstanceOutcome] = {}
    spent_total_usd = 0.0
    budget_ledger = BudgetReservations(args.budget_usd)
    halted_on_total_budget = False
    accounting_incomplete = False
    merge_predictions_jsonl((), predictions_path)

    ordered_payloads = [
        payloads_by_id[inst.instance_id] for inst in filtered
    ]

    def _record_completed(future, iid: str, reservation) -> None:
        nonlocal spent_total_usd, accounting_incomplete, halted_on_total_budget
        try:
            outcome = future.result()
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "swebench: worker for instance %s raised: %s", iid, e,
            )
            outcome = InstanceOutcome(
                instance_id=iid,
                classification=f"hard_fail_{type(e).__name__.lower()}",
                model_patch="", cost_usd=0.0, diff_bytes=0,
                files_modified=(),
                eval_status="skipped_cascade_error",
                cost_incomplete=True,
            )
        outcomes_by_id[iid] = outcome
        try:
            budget_ledger.settle(reservation, outcome.cost_usd)
        except BudgetReservationError as exc:
            logger.error("swebench: budget settlement failed for %s: %s", iid, exc)
            halted_on_total_budget = True
        spent_total_usd = budget_ledger.spent
        if outcome.cost_incomplete:
            accounting_incomplete = True
        _record_instance_cost(
            costs,
            outcome,
            backend_name=routed_backend_name,
            model=routed_model,
        )
        merge_predictions_jsonl((Prediction(
                instance_id=outcome.instance_id,
                model_patch=outcome.model_patch,
                model_name_or_path=outcome.producer_model or routed_model,
            ),),
            predictions_path,
        )
        handle.write_event({
            "phase": "instance",
            "instance_id": iid,
            "classification": outcome.classification,
            "diff_bytes": outcome.diff_bytes,
            "cost_usd": outcome.cost_usd,
            "cost_incomplete": outcome.cost_incomplete,
        })

    with ThreadPoolExecutor(max_workers=pool_size) as ex:
        in_flight: dict[object, tuple[str, object]] = {}
        next_index = 0

        def _submit_available() -> None:
            nonlocal next_index, halted_on_total_budget
            while (
                len(in_flight) < pool_size
                and next_index < len(ordered_payloads)
            ):
                free_slots = min(
                    pool_size - len(in_flight),
                    len(ordered_payloads) - next_index,
                )
                if args.budget_usd is None:
                    worker_budget = args.budget_usd_per_instance
                else:
                    available = budget_ledger.available or 0.0
                    if available <= 0.0:
                        # An in-flight reservation may come back mostly
                        # unused. Wait for it to settle before declaring a
                        # terminal halt; otherwise parallel mode strands
                        # valid aggregate budget.
                        if not in_flight:
                            halted_on_total_budget = True
                        return
                    fair_share = available / max(1, free_slots)
                    worker_budget = (
                        fair_share
                        if args.budget_usd_per_instance is None
                        else min(args.budget_usd_per_instance, fair_share)
                    )
                try:
                    reservation = budget_ledger.reserve(worker_budget)
                except BudgetReservationError:
                    if not in_flight:
                        halted_on_total_budget = True
                    return
                payload = dataclasses.replace(
                    ordered_payloads[next_index],
                    budget_usd=worker_budget,
                )
                future = ex.submit(_pool_worker, payload)
                in_flight[future] = (
                    payload.instance.instance_id,
                    reservation,
                )
                next_index += 1

        _submit_available()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            # Release all completed reservations before deciding whether
            # the next batch can launch. Process every member of ``done``;
            # breaking on the first budget crossing used to lose paid work.
            completed: list[tuple[object, str, object]] = []
            for future in done:
                iid, reservation = in_flight.pop(future)
                completed.append((future, iid, reservation))
            for future, iid, reservation in completed:
                _record_completed(future, iid, reservation)

            if (
                args.budget_usd is not None
                and spent_total_usd >= args.budget_usd
                and next_index < len(ordered_payloads)
            ):
                halted_on_total_budget = True
            if not halted_on_total_budget and not accounting_incomplete:
                _submit_available()

        if next_index < len(ordered_payloads):
            if accounting_incomplete:
                logger.error(
                    "swebench: cost accounting became incomplete; %d "
                    "instance(s) were not launched",
                    len(ordered_payloads) - next_index,
                )
            else:
                halted_on_total_budget = True
                logger.error(
                    "swebench: total budget exhausted after accounting all "
                    "in-flight completions (spent %.4f, cap %.4f); %d "
                    "instance(s) were not launched",
                    spent_total_usd,
                    args.budget_usd,
                    len(ordered_payloads) - next_index,
                )

    # Re-order outcomes to match the input filter order so summary.json
    # is deterministic regardless of completion ordering.
    outcomes = [
        outcomes_by_id[i.instance_id]
        for i in filtered
        if i.instance_id in outcomes_by_id
    ]
    return (
        outcomes,
        spent_total_usd,
        halted_on_total_budget,
        accounting_incomplete,
    )


def _stamp_eval_measurements(
    eval_report: EvalReport,
    instances_by_id: dict[str, SwebenchInstance],
    *,
    corpus: Corpus,
    run_dir: Path,
    run_id: str,
    terminal_implementation: Implementation | None = None,
) -> tuple[Measurement, ...]:
    """Persist one Measurement per evaluated instance + one tier-level
    aggregate Measurement carrying the resolved/total/rate triple.

    The per-instance Measurement carries the resolved + patch_applied
    booleans as 0.0/1.0 floats (the canonical assay.bench shape) plus
    test-status counts. The tier-level aggregate's bench_id is
    ``swebench/aggregate`` so per-instance evidence readers can find the headline
    number without iterating per-instance records.
    """
    import datetime as _dt
    import platform
    import socket

    def _exec() -> ExecutionContext:
        return ExecutionContext(
            timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds",
            ),
            machine=socket.gethostname(),
            cwd=str(run_dir),
            toolchain={"python": platform.python_version()},
            backend=None,
            model=eval_report.model,
            recipe="swebench_eval",
            recipe_commit=None,
            gpu_model=None,
            run_id=run_id,
        )

    # Sentinel commit shapes used below. The Measurement schema
    # expects a SHA-shaped string, but SWE-bench aggregate measurements
    # don't correspond to a single commit and per-instance records for
    # unknown instance_ids genuinely have no commit. Use distinct
    # non-SHA-shaped strings so evidence readers can identify aggregate and
    # unknown-instance records without treating them as repository commits.
    SENTINEL_AGGREGATE_COMMIT = "swebench-aggregate"
    SENTINEL_UNKNOWN_INSTANCE_COMMIT = "swebench-unknown-instance"

    stamped: list[Measurement] = []
    for r in eval_report.per_instance:
        inst = instances_by_id.get(r.instance_id)
        if inst is None:
            logger.warning(
                "swebench: eval report references instance_id %r that's "
                "not in the filtered set — Measurement commit will use "
                "the unknown-instance sentinel %r",
                r.instance_id, SENTINEL_UNKNOWN_INSTANCE_COMMIT,
            )
            commit = SENTINEL_UNKNOWN_INSTANCE_COMMIT
        else:
            commit = inst.base_commit
        # Persist the raw harness payload so postmortem can inspect
        # the per-instance tests_status without re-running the harness.
        raw_ref = corpus.put_artifact(
            json.dumps(r.raw_report, indent=2).encode("utf-8"),
            kind="bench_raw",
            name_hint=f"swebench_{r.instance_id}_report.json",
        )
        m = Measurement(
            bench_id=f"swebench/{r.instance_id}",
            commit=commit,
            config_hash=run_id,
            values={
                "resolved": 1.0 if r.resolved else 0.0,
                "patch_applied": 1.0 if r.patch_applied else 0.0,
                "fail_to_pass_passed": float(r.fail_to_pass_passed),
                "fail_to_pass_failed": float(r.fail_to_pass_failed),
                "pass_to_pass_passed": float(r.pass_to_pass_passed),
                "pass_to_pass_failed": float(r.pass_to_pass_failed),
            },
            raw_output_ref=raw_ref,
            execution=_exec(),
        )
        stamped.append(m)

    # Tier-level aggregate. raw_output_ref points at the workflow's
    # summary.json so callers can navigate from the headline number
    # back to per-instance context.
    summary_ref = corpus.put_artifact(
        json.dumps({
            "run_id": run_id,
            "model": eval_report.model,
            "total": eval_report.total,
            "resolved": eval_report.resolved_count,
            "patch_applied": eval_report.patch_applied_count,
            "resolved_rate": eval_report.resolved_rate,
        }, indent=2).encode("utf-8"),
        kind="bench_raw",
        name_hint=f"swebench_aggregate_{run_id}.json",
    )
    aggregate = Measurement(
        bench_id="swebench/aggregate",
        commit=SENTINEL_AGGREGATE_COMMIT,
        config_hash=run_id,
        values={
            "total": float(eval_report.total),
            "resolved": float(eval_report.resolved_count),
            "patch_applied": float(eval_report.patch_applied_count),
            "resolved_rate": eval_report.resolved_rate,
        },
        raw_output_ref=summary_ref,
        execution=_exec(),
    )
    stamped.append(aggregate)
    # The report is one evaluation fact set. Publish all per-instance bins and
    # the aggregate in one Spec version so readers cannot observe a partial
    # denominator while the loop is still stamping rows.
    evaluation_execution = _exec()
    corpus.commit(
        implementations=(
            (terminal_implementation,)
            if terminal_implementation is not None else ()
        ),
        measurements=tuple(stamped),
        lifecycle_event_factory=lambda committed_version: LifecycleEvent(
            kind=(
                "landed"
                if eval_report.total > 0
                and eval_report.resolved_count == eval_report.total
                else "abandoned"
            ),
            reason=(
                "authoritative SWE-bench evaluation "
                f"resolved={eval_report.resolved_count}/{eval_report.total}; "
                f"run_id={run_id}"
            ),
            affected_versions=(committed_version,),
            execution=evaluation_execution,
        ),
    )
    return tuple(stamped)


def _stamp_inline_reclassification(
    eval_report: EvalReport,
    inst: SwebenchInstance,
    *,
    corpus: Corpus,
    harness_output_dir: Path,
    run_id: str,
) -> None:
    """Retain parser interpretation alongside the official inline result.

    For the (single-instance) inline-eval scope, if the raw harness
    verdict for this iid is unresolved, attempt to reclassify via
    ``products/swebench/harness_reclassify.reclassify``. If a rule
    fires, persist a second Measurement bin
    (``bench_id=swebench/aggregate_reclassified``) and a
    ``LifecycleEvent(kind="reclassified")`` alongside the raw bins so
    downstream readers can prefer the reclassified verdict while the
    raw audit trail is preserved.

    Skips when:
      - The raw harness already marked this iid resolved.
      - No ``test_output.txt`` is locatable under harness_output_dir.
      - No reclassifier rule fires (e.g., true F2P/P2P fail).

    Failures bubble up to the caller, which logs a warning and
    continues — the raw Measurement is authoritative either way.
    """
    import datetime as _dt
    import platform
    import socket
    import tempfile

    # Bail early if no per-instance eval landed.
    if not eval_report.per_instance:
        return
    r = next(
        (x for x in eval_report.per_instance if x.instance_id == inst.instance_id),
        None,
    )
    if r is None or r.resolved:
        return

    # Find the test_output.txt under harness_output_dir. For a
    # single-instance inline eval there's exactly one.
    candidates = list(harness_output_dir.glob("**/test_output.txt"))
    if not candidates:
        return
    test_output = max(candidates, key=lambda p: p.stat().st_mtime)

    # Lazy imports — keeps the workflow-module import path light for
    # callers that never trigger inline eval.
    from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: PLC0415

    from crucible.products.swebench.harness_reclassify import (  # noqa: PLC0415
        ReclassifyVerdict,
        format_reclassification_reason,
        has_existing_reclassification,
        reclassify,
    )

    # Reconstruct the dataset-shape dict from the SwebenchInstance.
    # make_test_spec accepts either a dict or an existing TestSpec; we
    # pass through the parsed instance fields it needs.
    # ``make_test_spec`` reads these dataset-shape fields and fills the
    # rest (env_script_list / eval_script_list / repo_script_list /
    # arch / docker_specs / language / namespace) from
    # ``MAP_REPO_VERSION_TO_SPECS`` + per-(repo,version) templates
    # bundled into upstream swebench. The reclassifier doesn't run
    # those scripts (test_output is already on disk), but
    # ``make_test_spec`` requires all these keys at construction time.
    # If upstream's TestSpec construction shifts, this dict shape may
    # drift — covered by inline-reclassification tests in
    # ``tests/test_swebench_harness_reclassify.py``.
    inst_dict = {
        "instance_id": inst.instance_id,
        "repo": inst.repo,
        "version": inst.version,
        "base_commit": inst.base_commit,
        "patch": inst.patch,
        "test_patch": inst.test_patch,
        "problem_statement": inst.problem_statement,
        "FAIL_TO_PASS": list(inst.fail_to_pass),
        "PASS_TO_PASS": list(inst.pass_to_pass),
    }
    spec = make_test_spec(inst_dict)
    prediction = {
        "instance_id": inst.instance_id,
        "model_name_or_path": eval_report.model or "inline_reclassify",
        "model_patch": "(reclassifier — patch content irrelevant; "
                       "harness log already reflects post-patch state)",
    }
    with tempfile.TemporaryDirectory() as td:
        verdict: ReclassifyVerdict = reclassify(
            test_output, spec, prediction, tmp_log_dir=Path(td),
        )

    if not verdict.reclass_reason:
        # No rule fired; raw verdict stands. Nothing to write.
        return

    # Dedup against any prior reclassification of the same iid by the
    # same rule after a re-run or a restarted inline invocation.
    if has_existing_reclassification(corpus, verdict.reclass_reason):
        logger.info(
            "swebench inline-reclassify: %s already reclassified by "
            "rule %s — skipping",
            inst.instance_id, verdict.reclass_reason,
        )
        return

    def _exec() -> ExecutionContext:
        return ExecutionContext(
            timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds",
            ),
            machine=socket.gethostname(),
            cwd=str(harness_output_dir),
            toolchain={"python": platform.python_version()},
            backend=None,
            model=eval_report.model,
            recipe="swebench_inline_reclassify",
            recipe_commit=None,
            gpu_model=None,
            run_id=run_id,
        )

    inst_rep = verdict.upstream_report.get(inst.instance_id, {})
    resolved_float = 1.0 if verdict.status == "resolved" else 0.0
    patch_applied = 1.0 if inst_rep.get("patch_successfully_applied") else 0.0

    summary_payload = json.dumps(
        {
            "iid": inst.instance_id,
            "reclass_reason": verdict.reclass_reason,
            "status": verdict.status,
            "upstream_report": verdict.upstream_report,
            "evidence_excerpt": verdict.evidence_excerpt,
        },
        indent=2,
        default=str,
    ).encode("utf-8")
    summary_ref = corpus.put_artifact(
        summary_payload,
        kind="bench_raw",
        name_hint=f"swebench_aggregate_reclassified_{inst.instance_id}.json",
    )
    measurement = Measurement(
        bench_id="swebench/aggregate_reclassified",
        commit="swebench-aggregate-reclassified",
        config_hash=run_id,
        values={
            "total": 1.0,
            "resolved": resolved_float,
            "patch_applied": patch_applied,
            "resolved_rate": resolved_float,
        },
        raw_output_ref=summary_ref,
        execution=_exec(),
    )
    reason = format_reclassification_reason(verdict)
    execution = _exec()
    corpus.commit(
        measurements=(measurement,),
        lifecycle_event_factory=lambda committed_version: LifecycleEvent(
            kind="reclassified",
            reason=reason,
            affected_versions=(committed_version,),
            execution=execution,
        ),
    )
    logger.info(
        "swebench inline-reclassify: %s rule=%s status=%s",
        inst.instance_id, verdict.reclass_reason, verdict.status,
    )


def _finalize_handled_terminal(
    handle,
    *,
    args,
    status: str,
    rc: int,
    filtered_instances: int,
    error: Exception | str | None = None,
) -> int:
    """Publish a complete summary/result pair for a handled early exit."""

    error_text = None if error is None else str(error)[:2000]
    duration_s = handle.elapsed_s
    summary = {
        "mode": "implement",
        "status": status,
        "dataset": str(args.dataset),
        "filtered_instances": filtered_instances,
        "processed_instances": 0,
        "total_cost_usd": 0.0,
        "cost_incomplete": False,
        "halted_on_total_budget": False,
        "error": error_text,
        "duration_s": duration_s,
    }
    handle.write_summary(summary)
    handle.write_event({
        "phase": "terminal",
        "status": status,
        "error": error_text,
    })
    handle.finalize(RunResult(
        status=status,
        duration_s=duration_s,
        error_kind=(
            type(error).__name__
            if isinstance(error, Exception) else None
        ),
        error_message=error_text,
    ))
    return rc


def main(argv: list[str] | None = None) -> int:
    workflow_started = time.monotonic()
    args = _parse_args(argv)
    corpus = Corpus(args.spec)

    # Guarantee descendant cleanup on any exit path (atexit + SIGINT +
    # SIGTERM + SIGHUP). The bundled Claude CLI subprocess spawned per
    # implementer call is a direct child of this process; without
    # cleanup it can outlive the parent and keep burning Anthropic
    # API quota — empirically observed during this run when ``pkill``
    # on the parent left several bundled-claude processes orphaned.
    register_subprocess_cleanup()

    try:
        instances = load_dataset(args.dataset)
    except FileNotFoundError as e:
        logger.error("swebench: %s", e)
        record_handled_terminal(
            corpus, "swebench", status="dataset_unavailable",
            message=str(e), error_kind="input",
            duration_s=time.monotonic() - workflow_started,
            provenance_root=None,
        )
        return 2
    except SwebenchInstanceError as e:
        logger.error("swebench: dataset validation failed: %s", e)
        record_handled_terminal(
            corpus, "swebench", status="dataset_invalid",
            message=str(e), error_kind="input",
            duration_s=time.monotonic() - workflow_started,
            provenance_root=None,
        )
        return 1

    filtered = filter_instances(
        instances,
        instance_ids=args.instance_id or None,
        repos=args.repo or None,
        limit=args.limit,
    )

    explicit_filters = bool(args.instance_id or args.repo or args.limit)
    if not filtered:
        if explicit_filters:
            rendered = (
                "no instances matched filters "
                f"(instance_id={args.instance_id}, repo={args.repo}, "
                f"limit={args.limit}) over a dataset of "
                f"{len(instances)} instances"
            )
            logger.error("swebench: %s", rendered)
            record_handled_terminal(
                corpus, "swebench", status="no_instances_matched",
                message=rendered, error_kind="insufficient_input",
                duration_s=time.monotonic() - workflow_started,
                provenance_root=None,
            )
            return 1
        logger.warning(
            "swebench: dataset %s contains zero instances", args.dataset,
        )
        if args.implement:
            record_handled_terminal(
                corpus,
                "swebench",
                status="no_work",
                message=f"dataset {args.dataset} contains zero instances",
                error_kind="insufficient_input",
                extra={"filtered_instances": 0},
                duration_s=time.monotonic() - workflow_started,
                provenance_root=None,
            )
            return 1

    if args.list_only:
        for inst in filtered:
            print(f"{inst.instance_id}\t{inst.repo}\t{inst.base_commit}")
        return 0

    spec = corpus.latest_spec()
    if spec is None:
        # Synthesize the sweep-level parent Spec so the workflow is
        # self-contained. Each live instance gets its own starter Spec
        # from ``spec_synthesis.synthesize_starter_spec``; this parent
        # only anchors the sweep run envelope and aggregate records.
        from assay import Spec
        spec = Spec(
            version=0, parent=None, root=None, schema_version=1,
            project_context=ProjectContext(
                project="swebench", repo_url="https://swe-bench/",
                base_commit="0" * 40,
                branch=None, submodule_path=None, runners={},
            ),
            execution=None,
            intent="SWE-bench Verified evaluation run.",
            constraints=(), done_criteria=(), plan_policy=None,
        )
        corpus.put_spec(spec)
        logger.info(
            "swebench: synthesized sweep parent spec at %s; live "
            "instances receive independently synthesized specs",
            args.spec,
        )
    with open_run_with_capture(
        corpus,
        "swebench",
        provenance_root=None,
    ) as handle:
        run_dir = handle.run_dir
        # Resolve the routed model id ONCE for the whole run. All
        # Prediction.model_name_or_path attributes the route that produces
        # workspace edits, so it uses the implementer role even when the
        # surrounding cascade mixes models. spec_dir participates in normal
        # routing discovery, allowing a spec-local override.
        _routing = load_routing(
            config_path=args.backend_config, spec_dir=args.spec,
        )
        _outcome.record_provenance_inputs(
            handle.run_dir,
            backend_routing=routing_fingerprint(_routing),
            backend_features=routing_feature_fingerprint(_routing),
            backend_routing_source_sha256=(
                routing_source_fingerprint(_routing)
            ),
        )
        routed_model = _routing.for_role("implementer").model or "unknown"
        predictions_path = (
            args.predictions_out
            if args.predictions_out is not None
            else run_dir / "predictions.jsonl"
        )

        if args.dry_run:
            # Zero-patch predictions: every Prediction.model_patch == "".
            # The SWE-bench eval harness treats this as "no fix
            # attempted" — a useful baseline for "is the harness even
            # running?" smoke tests.
            preds = [
                Prediction(
                    instance_id=inst.instance_id,
                    model_patch="",
                    model_name_or_path=routed_model,
                )
                for inst in filtered
            ]
            count = write_predictions_jsonl(preds, predictions_path)
            duration_s = handle.elapsed_s
            summary = {
                "mode": "dry_run",
                "dataset": str(args.dataset),
                "total_instances": len(instances),
                "filtered_instances": len(filtered),
                "predictions_path": str(predictions_path),
                "predictions_count": count,
                "model": routed_model,
                "duration_s": duration_s,
            }
            handle.write_summary(summary)
            logger.info(
                "swebench dry-run: %d instances → %s",
                count, predictions_path,
            )
            handle.finalize(RunResult(
                status="dry_run_complete",
                duration_s=duration_s,
            ))
            return 0

        if not args.implement:
            # No mode flag was set and the workflow can't infer one.
            logger.error(
                "swebench: select a mode (--list / --dry-run / "
                "--implement)."
            )
            duration_s = handle.elapsed_s
            summary = {
                "mode": "no_mode_selected",
                "filtered_instances": len(filtered),
                "duration_s": duration_s,
            }
            handle.write_summary(summary)
            handle.finalize(RunResult(
                status="no_mode_selected",
                duration_s=duration_s,
            ))
            return 1

        # ------------------------------------------------------------------
        # --implement: per-instance Architect/Tier cascade.
        # ------------------------------------------------------------------
        # Aggregate zero-budget runs retain their ordinary clean halt. Every
        # admitted task otherwise receives a finite reservation before any
        # repository, container, or provider work can begin.
        if args.budget_usd != 0:
            task_caps = [
                cap for cap in (
                    args.budget_usd_per_instance, args.budget_usd,
                ) if cap is not None
            ]
            try:
                validate_competition_admission(
                    args.planner_count, min(task_caps) if task_caps else None,
                )
            except ValueError as exc:
                return _finalize_handled_terminal(
                    handle,
                    args=args,
                    status="planning_policy_invalid",
                    rc=2,
                    filtered_instances=len(filtered),
                    error=str(exc),
                )
        if not _docker_available():
            logger.error(
                "swebench: Docker is required for the authoring sandbox; "
                "host execution fallback is disabled."
            )
            return _finalize_handled_terminal(
                handle,
                args=args,
                status="container_runtime_required",
                rc=2,
                filtered_instances=len(filtered),
                error="Docker is unavailable for mandatory authoring isolation",
            )
        try:
            removed = _reconcile_orphans(str(Path(args.spec).resolve()))
            if removed:
                logger.warning(
                    "swebench: reconciled %d orphan container(s)", len(removed),
                )
            _check_disk_headroom()
        except _InstanceContainerSetupError as e:
            logger.error("swebench: %s", e)
            return _finalize_handled_terminal(
                handle,
                args=args,
                status="container_preflight_failed",
                rc=2,
                filtered_instances=len(filtered),
                error=e,
            )

        # Resumability: skip instances whose prior classification is
        # skip-worthy (success or soft_fail_no_changes) unless --rerun
        # is set. Hard-failed instances retry by default. One corpus
        # walk for the whole resumability decision (model_patch +
        # model + classification per instance).
        # Every selected task enters the shared worker protocol.  The worker
        # acquires its cross-process lease and only then re-reads the terminal
        # marker, closing the former partition-before-lock race.
        prior_predictions: dict[str, _PriorPrediction] = {}
        to_process, skipped = filtered, ()

        # Workflow-level cost accountant. NO cap here: the swebench
        # workflow already does its own cumulative budget gating in
        # the per-instance loop (clean halt + ``halted_on_total_budget``
        # in summary.json + sentinel model names for unreached
        # instances). Stacking the accumulator's CostLimitExceeded on
        # top would short-circuit that clean halt with an exception
        # and bypass the unreached-instance accounting.
        costs = CostAccumulator()
        try:
            if args.max_concurrent_instances > 1 and len(to_process) > 1:
                (
                    outcomes,
                    spent_total_usd,
                    halted_on_total_budget,
                    accounting_incomplete,
                ) = _run_implement_parallel(
                    to_process, args=args,
                    run_dir=run_dir, handle=handle,
                    predictions_path=predictions_path,
                    routing=_routing,
                    costs=costs,
                )
            else:
                (
                    outcomes,
                    spent_total_usd,
                    halted_on_total_budget,
                    accounting_incomplete,
                ) = _run_implement_serial(
                    to_process, args=args,
                    run_dir=run_dir, handle=handle,
                    predictions_path=predictions_path,
                    routing=_routing,
                    costs=costs,
                )
        except StorageQuotaExceeded as e:
            logger.error("swebench: %s", e)
            return _finalize_handled_terminal(
                handle,
                args=args,
                status="storage_quota_exceeded",
                rc=1,
                filtered_instances=len(filtered),
                error=e,
            )
        except _InstanceContainerSetupError as e:
            logger.error("swebench: %s", e)
            return _finalize_handled_terminal(
                handle,
                args=args,
                status="container_preflight_failed",
                rc=2,
                filtered_instances=len(filtered),
                error=e,
            )

        # Treat the returned flag as an optimization, not the sole safety
        # authority. Any outcome carrying an incomplete rollup vetoes a
        # successful finalization even if a future scheduler implementation
        # forgets to propagate its aggregate flag.
        accounting_incomplete = accounting_incomplete or any(
            outcome.cost_incomplete for outcome in outcomes
        )
        for outcome in outcomes:
            _note_instance_rollup_debt(costs, outcome)

        # Final predictions write — single source of truth.
        #
        # The per-instance locked merge calls inside
        # the loops are the crash-safety surface: if the workflow
        # dies mid-loop, those appends preserve completed work and
        # the next ``--rerun``/resume path picks up from there.
        # This rewrite at end-of-run is the deterministic-order
        # consolidation that the downstream eval harness reads.
        # Single write path means no "incremental + clobber" dead
        # layering ambiguity.
        #
        # Composition: outcomes from this run + skipped-instance
        # surfaced predictions (with their ORIGINAL model
        # attribution, not routed_model — cross-model comparisons
        # depend on that field) + zero-patch placeholders for
        # instances that never reached the loop (budget-halted).
        outcomes_by_id = {o.instance_id: o for o in outcomes}
        skipped_ids = {s.instance_id for s in skipped}
        # prior_predictions is already loaded above for the resume
        # decision; reuse the same map for the final write so we
        # don't walk the corpus twice (N4 from the second-pass
        # critique).
        preds: list[Prediction] = []
        unreached: list[str] = []
        for inst in filtered:
            iid = inst.instance_id
            if iid in outcomes_by_id:
                preds.append(Prediction(
                    instance_id=iid,
                    model_patch=outcomes_by_id[iid].model_patch,
                    model_name_or_path=(
                        outcomes_by_id[iid].producer_model or routed_model
                    ),
                ))
            elif iid in skipped_ids:
                prior = prior_predictions.get(iid)
                preds.append(Prediction(
                    instance_id=iid,
                    model_patch=prior.model_patch if prior else "",
                    # Preserve the model that ACTUALLY produced this
                    # patch. Falling back to routed_model only when
                    # the prior record is missing.
                    model_name_or_path=(
                        prior.model if prior and prior.model
                        else routed_model
                    ),
                ))
            else:
                # Filtered but neither processed nor skipped — the
                # budget-halt path leaves these unreached. Surface
                # them as zero-patch entries with a clear sentinel
                # model name so postmortem can attribute them.
                unreached.append(iid)
                preds.append(Prediction(
                    instance_id=iid,
                    model_patch="",
                    model_name_or_path=f"{routed_model}/unreached",
                ))
        if unreached:
            if accounting_incomplete:
                unreached_reason = "accounting_incomplete"
                halt_description = "incomplete cost accounting"
            elif halted_on_total_budget:
                unreached_reason = "budget_exhausted"
                halt_description = "budget halt"
            else:
                unreached_reason = "not_scheduled"
                halt_description = "scheduler halt"
            logger.warning(
                "swebench: %d instance(s) dropped due to %s "
                "before processing — emitted as zero-patch entries "
                "with ``/unreached`` model suffix: %s",
                len(unreached),
                halt_description,
                ", ".join(unreached[:10])
                + (" …" if len(unreached) > 10 else ""),
            )
        else:
            unreached_reason = None
        count = merge_predictions_jsonl(preds, predictions_path)

        cascade_success_count = sum(
            1 for o in outcomes if o.classification.startswith("success")
        )
        eval_status_counts: dict[str, int] = {}
        pristine_verification_status_counts: dict[str, int] = {}
        for outcome in outcomes:
            key = outcome.eval_status or "not_recorded"
            eval_status_counts[key] = eval_status_counts.get(key, 0) + 1
            pristine_key = (
                outcome.pristine_verification.status.value
                if outcome.pristine_verification is not None else "not_run"
            )
            pristine_verification_status_counts[pristine_key] = (
                pristine_verification_status_counts.get(pristine_key, 0) + 1
            )
        resolved_count = eval_status_counts.get("resolved", 0)
        nonempty_count = sum(
            1 for o in outcomes if o.model_patch
        )
        # Per-instance transport-failure counter — operator postmortem
        # signal for runs where SDK transport burps shaped outcomes
        # (orthogonal to budget exhaustion / real classification
        # failures). The per-instance ``_spec_to_outcome`` rolls
        # transport-classed Implementations into the InstanceOutcome's
        # classification with the ``hard_fail_transport`` suffix.
        transport_failure_instances = sum(
            1 for o in outcomes
            if (
                o.planning_transport_failure
                or "hard_fail_transport" in o.classification
            )
        )
        budget_exceeded = (
            args.budget_usd is not None
            and spent_total_usd > args.budget_usd
        )
        budget_failed = halted_on_total_budget or budget_exceeded
        if accounting_incomplete:
            terminal_status = "cost_incomplete"
        elif budget_failed:
            terminal_status = "budget_exhausted"
        else:
            terminal_status = "implement_complete"
        cost_summary = costs.to_dict()
        cost_summary["planning_backend_calls"] = sum(
            outcome.planning_backend_calls for outcome in outcomes
        )
        duration_s = handle.elapsed_s
        summary = {
            "mode": "implement",
            "status": terminal_status,
            "dataset": str(args.dataset),
            "total_instances": len(instances),
            "filtered_instances": len(filtered),
            "processed_instances": len(outcomes),
            "resumed_instances": sum(1 for outcome in outcomes if outcome.resumed),
            # Cascade completion and the official harness oracle are distinct.
            # A syntactically successful patch can still be unresolved; never
            # present the former as the benchmark success KPI.
            "cascade_success_count": cascade_success_count,
            "resolved_count": resolved_count,
            "eval_status_counts": eval_status_counts,
            "pristine_verification_status_counts": (
                pristine_verification_status_counts
            ),
            "nonempty_patch_count": nonempty_count,
            "transport_failure_instances": transport_failure_instances,
            "unreached_instances": len(unreached),
            "unreached_reason": unreached_reason,
            "halted_on_total_budget": halted_on_total_budget,
            "budget_exceeded": budget_exceeded,
            "cost_incomplete": accounting_incomplete,
            "total_cost_usd": spent_total_usd,
            "budget_usd": args.budget_usd,
            "planning": benchmark_planning_identity(
                args.planner_count,
                planning_budget_share=(
                    BENCHMARK_PLANNING_BUDGET_SHARE
                    if args.planner_count > 1 else None
                ),
            ),
            "costs": cost_summary,
            "predictions_path": str(predictions_path),
            "predictions_count": count,
            "model": routed_model,
            "duration_s": duration_s,
            "submission_policy": (
                "best_effort_any_nonempty"
                if getattr(
                    args, "submit_any_nonempty_best_effort", False
                )
                else "verified_success_only"
            ),
            "per_instance": [
                {
                    "instance_id": o.instance_id,
                    "classification": o.classification,
                    "eval_status": o.eval_status,
                    "diff_bytes": o.diff_bytes,
                    "files_modified": list(o.files_modified),
                    **_prediction_metadata_summary(o),
                    "cost_usd": o.cost_usd,
                    "cost_incomplete": o.cost_incomplete,
                    "input_tokens": o.input_tokens,
                    "output_tokens": o.output_tokens,
                    "planning_backend_calls": o.planning_backend_calls,
                    "planning_transport_failure": (
                        o.planning_transport_failure
                    ),
                    "architect_detail_status": o.architect_detail_status,
                    "architect_terminal_class": o.architect_terminal_class,
                    "resumed": o.resumed,
                    "checkpoint_fingerprint": o.checkpoint_fingerprint,
                    "image_digest": o.image_digest,
                }
                for o in outcomes
            ],
        }
        handle.write_summary(summary)
        logger.info(
            "swebench implement: processed=%d cascade_success=%d nonempty=%d "
            "cost=%.4f → %s",
            len(outcomes), cascade_success_count, nonempty_count,
            spent_total_usd, predictions_path,
        )
        rc = 1 if (budget_failed or accounting_incomplete) else 0
        handle.finalize(RunResult(
            status=terminal_status,
            duration_s=duration_s,
            cost_usd=spent_total_usd,
        ))
        return rc


if __name__ == "__main__":
    sys.exit(main())
