# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Validate or freeze the provider-free DeepSWE quality-pilot plan.

There is intentionally no run, launch, dispatch, or implement action.  Cell
argv are reviewable proposals only; this command never invokes them and never
constructs a backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crucible.products.deepswe.quality_cohort import (
    DeepSWEQualityCohortError,
    build_quality_cohort_plan,
    freeze_quality_cohort_plan,
    load_quality_cohort_matrix,
    validate_frozen_plan,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MATRIX = _ROOT / "products" / "deepswe" / "backend_convergence_matrix.yaml"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="deepswe-backend-convergence")
    parser.add_argument(
        "action",
        choices=("validate", "dry-run", "freeze"),
        help="Provider-free operation; no action can launch a cell.",
    )
    parser.add_argument("--matrix", type=Path, default=_DEFAULT_MATRIX)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Pinned DeepSWE clone's tasks/manifest.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Proposed isolated cell root; this planner does not create it.",
    )
    parser.add_argument(
        "--plan-out",
        type=Path,
        default=None,
        help="Exclusive output path; required only by freeze.",
    )
    parser.add_argument(
        "--plan-in",
        type=Path,
        default=None,
        help="Existing frozen plan to byte-validate; valid only with validate.",
    )
    args = parser.parse_args(argv)
    if args.action == "freeze" and args.plan_out is None:
        parser.error("freeze requires --plan-out")
    if args.action != "freeze" and args.plan_out is not None:
        parser.error("--plan-out is valid only with freeze")
    if args.action != "validate" and args.plan_in is not None:
        parser.error("--plan-in is valid only with validate")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        matrix = load_quality_cohort_matrix(args.matrix)
        plan = build_quality_cohort_plan(
            matrix,
            dataset_path=args.dataset,
            output_root=args.output_root,
        )
        if args.action == "validate" and args.plan_in is not None:
            validate_frozen_plan(args.plan_in, expected=plan)
        elif args.action == "dry-run":
            sys.stdout.buffer.write(plan.to_json_bytes())
        elif args.action == "freeze":
            assert args.plan_out is not None
            freeze_quality_cohort_plan(plan, args.plan_out)
    except FileExistsError:
        print(
            f"plan output already exists; refusing overwrite: {args.plan_out}",
            file=sys.stderr,
        )
        return 2
    except (DeepSWEQualityCohortError, OSError) as exc:
        print(f"quality-pilot plan rejected: {exc}", file=sys.stderr)
        return 2

    print(
        f"action={args.action} tasks={len(plan.corpus.tasks)} "
        f"blocks=30 cells={len(plan.cells)} nominal_soft_budget_usd="
        f"{plan.matrix.execution.nominal_budget_usd} "
        f"spend_authorized=false dispatch_enabled=false "
        f"plan_digest={plan.payload()['plan_digest']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
