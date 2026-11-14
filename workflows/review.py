# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Review workflow — drive ``assay.review`` against an artifact + rubric.

Usage:

    python -m crucible.workflows review \\
        --spec <spec_dir> \\
        --rubric <rubric_name> \\
        (--artifact <path> | --artifact-hash sha256:<hex>) \\
        [--backend-config <routing.yaml>]

Loads the artifact (path → bytes, or content-hash → corpus lookup),
invokes the rubric via the strict Crucible review adapter, writes the complete
typed Verdict surface to ``<run_dir>/summary.json``, and returns the shared
decision exit code (pass=0, revise=1, reject=2, inconclusive=3).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from assay import ContentHash
from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus
from assay.primitives.review import MAX_CHUNK_RESPONSE_RETRY_BUDGET
from assay.types.artifacts import Implementation

from crucible.common.cost import CostAccumulator, CostLimitExceeded
from crucible.rubrics import contract as rubric_contract
from crucible.rubrics import load as load_rubric

from ._args import (
    add_backend_limit_args,
    add_budget_arg,
    add_spec_arg,
    nonnegative_int_arg,
    validate_backend_limit_args,
)
from ._analysis_handoff import decision_exit_code, enforce_budget_overrun
from ._backends import backend_pool
from . import _partition
from ._run import open_run_with_capture, record_handled_terminal
from ._review import AUTONOMOUS_REVIEW_CHUNK_CHARS, review

logger = logging.getLogger(__name__)


class _ConcurrentReviewSpecChange(RuntimeError):
    """A context-dependent review outlived its execution authority."""


def _response_retry_budget_arg(value: str) -> int:
    parsed = nonnegative_int_arg(value)
    if parsed > MAX_CHUNK_RESPONSE_RETRY_BUDGET:
        raise argparse.ArgumentTypeError(
            "must be at most "
            f"{MAX_CHUNK_RESPONSE_RETRY_BUDGET}"
        )
    return parsed


def _review_structure_guard(expected_structure_hash: str):
    def guard(candidate):
        if (
            _partition.spec_execution_structure_hash(candidate)
            != expected_structure_hash
        ):
            raise _ConcurrentReviewSpecChange(
                "Spec execution authority changed during review"
            )
        return candidate

    return guard


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="crucible-workflow-review")
    add_spec_arg(p)
    p.add_argument("--rubric", required=True,
                   help="Name of a bundled rubric (under crucible/rubrics/).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--artifact", type=Path,
                   help="Path to a file whose contents will be reviewed.")
    g.add_argument("--artifact-hash",
                   help="`sha256:<hex>` corpus key for an existing artifact.")
    p.add_argument(
        "--slice-id",
        default=None,
        help=(
            "Slice whose directive scopes a diff_scope review. Required for "
            "path artifacts; hash artifacts may resolve it from an attached "
            "Implementation."
        ),
    )
    # The --rubric value doubles as the role key (review.py:72 passes
    # args.rubric directly to resolve_for_role) — so the help text
    # below differs slightly from add_backend_config_arg's standard
    # "roles consulted: <fixed list>" shape.
    p.add_argument("--backend-config", type=Path, default=None,
                   help="Path to a backend routing YAML overriding "
                        "discovery; role consulted is the value of "
                        "--rubric (see crucible.backends.config for "
                        "the schema).")
    add_budget_arg(p, kind="per_call")
    p.add_argument(
        "--response-retry-budget",
        type=_response_retry_budget_arg,
        default=0,
        help=(
            "bounded fresh semantic response repairs after the initial "
            "attempt (default: 0, maximum: "
            f"{MAX_CHUNK_RESPONSE_RETRY_BUDGET}); chunk reviews delegate "
            "this allowance to Assay's current-map-only retry seam"
        ),
    )
    add_backend_limit_args(p)
    parsed = p.parse_args(argv)
    validate_backend_limit_args(p, parsed)
    return parsed


def _implementation_slice_for_artifact(
    corpus: Corpus,
    artifact_ref: ContentHash,
    *,
    explicit_slice_id: str | None = None,
    spec=None,
) -> str | None:
    """Resolve one unambiguous slice from an attached Implementation diff."""
    spec = spec or corpus.latest_spec()
    if spec is None:
        return None
    matches: set[str] = set()
    for ref in reversed(spec.implementation_refs):
        try:
            impl = corpus.get_record(ref, as_type=Implementation)
        except Exception:  # noqa: BLE001 — tolerate old/unavailable records
            continue
        if impl.diff_ref == artifact_ref:
            matches.add(impl.slice_id)
    if explicit_slice_id is not None:
        if explicit_slice_id in matches:
            return explicit_slice_id
        if matches:
            raise SystemExit(
                f"--slice-id {explicit_slice_id!r} does not match artifact's "
                f"Implementation slices {sorted(matches)!r}"
            )
    if len(matches) > 1:
        raise SystemExit(
            "artifact diff is attached to multiple slices; pass --slice-id"
        )
    return next(iter(matches), None)


def _scope_context(
    corpus: Corpus,
    *,
    slice_id: str | None,
    artifact_ref: ContentHash | None,
    spec=None,
) -> tuple[str, dict[str, str]]:
    """Return the exact directive scope for one slice.

    A whole-DAG union is unsafe: it lets a slice modify a sibling's files and
    still pass ``diff_scope``.  Hash artifacts can infer their slice from a
    persisted Implementation; path artifacts require ``--slice-id``.
    """
    spec = spec or corpus.latest_spec()
    if spec is None or spec.tier_dag is None:
        raise SystemExit("diff_scope review requires a Spec with a tier_dag")

    inferred = (
        _implementation_slice_for_artifact(
            corpus,
            artifact_ref,
            explicit_slice_id=slice_id,
            spec=spec,
        )
        if artifact_ref is not None
        else None
    )
    if slice_id is not None and inferred is not None and slice_id != inferred:
        raise SystemExit(
            f"--slice-id {slice_id!r} does not match artifact's "
            f"Implementation slice {inferred!r}"
        )
    effective_slice_id = slice_id or inferred
    if effective_slice_id is None:
        raise SystemExit(
            "diff_scope review needs --slice-id, or an --artifact-hash "
            "attached to an Implementation"
        )

    owner_tier = None
    selected = None
    tiers = {tier.tier_id: tier for tier in spec.tier_dag.tiers}
    for tier in spec.tier_dag.tiers:
        for candidate in tier.slices:
            if candidate.id == effective_slice_id:
                owner_tier = tier
                selected = candidate
                break
        if selected is not None:
            break
    if selected is None or owner_tier is None:
        raise SystemExit(f"slice {effective_slice_id!r} not found in tier_dag")

    dependency_ids: set[str] = set()

    def visit(tier_id: str) -> None:
        if tier_id in dependency_ids:
            return
        dependency_ids.add(tier_id)
        dependency = tiers.get(tier_id)
        if dependency is None:
            return
        for parent in dependency.depends_on:
            visit(parent)

    for dependency_id in owner_tier.depends_on:
        visit(dependency_id)
    dependency_provided: set[str] = set()
    for tier_id in dependency_ids:
        dependency = tiers.get(tier_id)
        if dependency is None:
            continue
        for dependency_slice in dependency.slices:
            dependency_provided.update(dependency_slice.directive.creates_paths)
    directive = selected.directive
    return effective_slice_id, {
        "files_in_scope": "\n".join(sorted(directive.files_in_scope)) or "(none)",
        "creates_paths": "\n".join(sorted(directive.creates_paths)) or "(none)",
        "dependency_provided_files": (
            "\n".join(sorted(dependency_provided)) or "(none)"
        ),
        "directive_intent": directive.intent.strip() or "(none)",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus = Corpus(args.spec)
    review_spec = corpus.latest_spec()
    if review_spec is None:
        message = f"no spec manifest found under {args.spec / 'versions'}"
        record_handled_terminal(
            corpus, "review", status="setup_error",
            message=message, error_kind="missing_spec",
            provenance_root=None,
        )
        return 2
    meta = rubric_contract(args.rubric)
    if "workflows/review.py" not in meta.consumers:
        raise SystemExit(
            f"rubric {args.rubric!r} is not wired for the generic review "
            "workflow; invoke its declared producer workflow instead"
        )
    if meta.tools != "none":
        raise SystemExit(
            f"rubric {args.rubric!r} requires a retained read-tool source "
            "registry that the generic review workflow does not provide"
        )
    rubric = load_rubric(args.rubric)

    if args.artifact is not None:
        artifact = args.artifact
        artifact_ref = None
    else:
        artifact_ref = ContentHash(args.artifact_hash)
        artifact = artifact_ref

    review_slice_id = args.slice_id
    review_context = None
    expected_structure_hash = None
    if args.rubric == "diff_scope":
        expected_structure_hash = (
            _partition.spec_execution_structure_hash(review_spec)
        )
        review_slice_id, review_context = _scope_context(
            corpus,
            slice_id=args.slice_id,
            artifact_ref=artifact_ref,
            spec=review_spec,
        )

    with open_run_with_capture(
        corpus,
        "review",
        provenance_root=None,
    ) as handle, \
            backend_pool(args, handle.run_dir) as pool:
        backend = pool.get(args.rubric)
        verdict = review(
            artifact,
            rubric,
            backend=backend,
            corpus=corpus,
            budget_usd=args.budget_usd,
            diagnostic_cost_policy=args.diagnostic_cost_policy,
            context=review_context,
            chunk_chars=AUTONOMOUS_REVIEW_CHUNK_CHARS,
            response_retry_budget=args.response_retry_budget,
        )
        costs = CostAccumulator(
            max_cost_usd=args.budget_usd,
            diagnostic_cost_policy=args.diagnostic_cost_policy,
        )
        try:
            costs.record(
                role=args.rubric, usage=verdict,
                backend_name=backend.name,
                model=backend.model,
            )
        except CostLimitExceeded:
            # The backend ignored its per-call cap. Preserve the response for
            # audit, but never let an over-budget nominal approval control the
            # workflow as a pass.
            verdict = dataclasses.replace(
                verdict,
                decision="inconclusive",
                reasoning=(
                    f"review exceeded the ${args.budget_usd:.4f} budget; "
                    + verdict.reasoning
                ),
                coverage=dataclasses.replace(
                    verdict.coverage,
                    complete=False,
                    omissions=verdict.coverage.omissions
                    + ("backend-reported cost exceeded the review budget",),
                ),
            )
        verdict = enforce_budget_overrun(verdict, costs)
        try:
            committed = corpus.commit(
                verdicts=(verdict,),
                transform=(
                    _review_structure_guard(expected_structure_hash)
                    if expected_structure_hash is not None
                    else None
                ),
            )
        except _ConcurrentReviewSpecChange as exc:
            summary = {
                "status": "concurrent_spec_change",
                "error": str(exc),
                "rubric": verdict.rubric,
                "original_decision": verdict.decision,
                "cost_usd": verdict.cost_usd,
                "costs": costs.to_dict(),
                "accounting_debt": list(verdict.accounting_debt),
                "slice_id": review_slice_id,
            }
            handle.write_summary(summary)
            logger.error("review: %s", exc)
            handle.finalize_elapsed(RunResult(
                status="concurrent_spec_change",
                cost_usd=verdict.cost_usd,
            ))
            return 3
        committed_spec = committed.spec
        verdict_ref = committed.refs.verdict_refs[0]

        summary = {
            "rubric": verdict.rubric,
            "decision": verdict.decision,
            "passed": verdict.passed,
            "coverage": dataclasses.asdict(verdict.coverage),
            "finding_count": len(verdict.findings),
            "findings": [dataclasses.asdict(f) for f in verdict.findings],
            "cost_usd": verdict.cost_usd,
            "costs": costs.to_dict(),
            "accounting_debt": list(verdict.accounting_debt),
            "reasoning": verdict.reasoning,
            "artifact_ref": verdict.artifact_ref,
            "transcript_ref": verdict.transcript_ref,
            "verdict_ref": verdict_ref,
            "spec_version": committed_spec.version,
            "slice_id": review_slice_id,
        }
        handle.write_summary(summary)
        logger.info(
            "review: rubric=%s passed=%s findings=%d cost=$%.4f",
            verdict.rubric, verdict.passed, len(verdict.findings),
            verdict.cost_usd,
        )
        handle.finalize_elapsed(RunResult(
            status=verdict.decision,
            cost_usd=verdict.cost_usd,
        ))
        return decision_exit_code(verdict.decision)


if __name__ == "__main__":
    sys.exit(main())
