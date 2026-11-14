# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Validate workflow — drive ``assay.validate`` against a Spec.

Usage:

    python -m crucible.workflows validate \\
        --spec <spec_dir> \\
        [--rules cycle,blast_radius,parallel_tier_conflicts,...] \\
        [--ruleset-name strict]

If ``--rules`` is omitted, runs the entire bundled rule set from
``crucible.rules.available()``. No backend invocation, no cost; runs
in milliseconds (per arch §5.5). Writes the verdict as
``<run_dir>/summary.json``. Exits 0 if all findings are below the
Critical/High severity floor; else 1.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import sys
from pathlib import Path

from assay import RuleContext, ValidationRuleset
from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus
from assay.primitives.validate import validate

from crucible.rules import available as available_rules
from crucible.rules import context_aware as context_aware_rules

from ._args import add_spec_arg
from ._atomic_persistence import persist_and_attach
from ._finding_counts import finding_counts
from ._run import open_run_with_capture, record_handled_terminal

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="crucible-workflow-validate")
    add_spec_arg(p)
    p.add_argument("--rules", default=None,
                   help="Comma-separated subset of rule_ids. Default: all bundled rules.")
    p.add_argument("--ruleset-name", default="default",
                   help="Name embedded in the Verdict.rubric field.")
    p.add_argument("--workspace-root", type=Path, default=None,
                   help=(
                       "Workspace root for context-aware rules (e.g. "
                       "cited_path_exists). When set, context-aware "
                       "rules fire against the local working tree. "
                       "Default: the spec's grandparent dir "
                       "(<workspace>/.crucible/specs/<target>/<spec>/ → "
                       "<workspace>)."
                   ))
    return p.parse_args(argv)


def _default_workspace_root(spec_dir: Path) -> Path | None:
    """Default workspace_root for the canonical Corpus layout
    ``<workspace>/.crucible/specs/<target>/<spec>/``: peel four parent
    components off the resolved spec dir.

    Returns None if the inferred path doesn't look workspace-shaped
    (e.g. a flat test fixture). Callers can pass --workspace-root
    explicitly to override.
    """
    # spec → <target> → specs → .crucible → <workspace>
    candidate = spec_dir.resolve().parent.parent.parent.parent
    if (candidate / ".crucible").is_dir():
        return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus = Corpus(args.spec)
    spec = corpus.latest_spec()
    if spec is None:
        message = f"no spec manifest found under {args.spec / 'versions'}"
        record_handled_terminal(
            corpus, "validate", status="setup_error",
            message=message, error_kind="missing_spec",
            provenance_root=None,
        )
        return 2

    rule_ids = (
        tuple(r.strip() for r in args.rules.split(",") if r.strip())
        if args.rules
        else available_rules()
    )
    ruleset = ValidationRuleset(name=args.ruleset_name, rule_ids=rule_ids)

    deduped_rule_ids = tuple(dict.fromkeys(rule_ids))
    context_rule_ids = set(context_aware_rules())
    selected_context_rules = tuple(
        rule_id
        for rule_id in deduped_rule_ids
        if rule_id in context_rule_ids
    )

    # A selected context-aware rule is a hard contract: it either receives a
    # real workspace root or the workflow fails setup without publishing an
    # incomplete Verdict. Callers wanting a machine-independent gate select
    # only pure rules through ``--rules``.
    ctx: RuleContext | None = None
    if selected_context_rules:
        workspace_root = args.workspace_root or _default_workspace_root(args.spec)
        if workspace_root is None:
            message = (
                "context-aware validation requires --workspace-root when "
                "the spec is outside the canonical workspace layout"
            )
            record_handled_terminal(
                corpus,
                "validate",
                status="setup_error",
                message=message,
                error_kind="missing_rule_context",
                extra={"context_rule_ids": list(selected_context_rules)},
                provenance_root=None,
            )
            return 2
        ctx = RuleContext(
            workspace_root=workspace_root,
            submodule_path=spec.project_context.submodule_path,
        )

    provenance_root = None
    if ctx is not None:
        provenance_root = Path(ctx.workspace_root)
        if ctx.submodule_path:
            provenance_root /= ctx.submodule_path

    with open_run_with_capture(
        corpus,
        "validate",
        provenance_root=provenance_root,
    ) as handle:
        verdict = validate(spec, ruleset=ruleset, ctx=ctx)
        committed_spec, verdict_ref = persist_and_attach(
            corpus,
            verdict,
            kind="verdicts",
            attachment="verdict",
        )
        counts = finding_counts(verdict.findings)
        summary = {
            "rubric": verdict.rubric,
            "decision": verdict.decision,
            "passed": verdict.passed,
            "coverage": asdict(verdict.coverage),
            "requested_rule_ids": list(rule_ids),
            "run_rule_ids": list(deduped_rule_ids),
            "context": {
                "available": ctx is not None,
                "workspace_root": (
                    str(ctx.workspace_root) if ctx is not None else None
                ),
            },
            "finding_count": counts["total"],
            "finding_counts": counts,
            "findings": [
                {
                    "id": f.id,
                    "severity": f.severity,
                    "category": f.category,
                    "suggested_action": f.suggested_action,
                }
                for f in verdict.findings
            ],
            "verdict_ref": verdict_ref,
            "spec_version": committed_spec.version,
        }
        handle.write_summary(summary)
        logger.info(
            "validate: rubric=%s passed=%s findings=%d blocking=%d advisory=%d",
            verdict.rubric,
            verdict.passed,
            counts["total"],
            counts["blocking"],
            counts["advisory"],
        )
        handle.finalize_elapsed(RunResult(
            status="passed" if verdict.passed else "failed",
        ))
        return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
