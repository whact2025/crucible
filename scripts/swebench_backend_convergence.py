# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Expand an explicit checked-in SWE-bench matrix without launching it.

This command is deliberately a planner.  It authenticates the matrix, route
files, and local SWE-bench Verified dataset, then emits the exact isolated
cell argv/dependency graph and nominal configured budget. The plan confers no
execution authority. A separate reviewed launcher must reauthenticate all
inputs at dispatch before any provider call is allowed. The default remains
the frozen Astra/Fable v1 matrix; successor matrices require ``--matrix``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crucible.products.swebench.backend_convergence import (
    BackendConvergenceMatrixError,
    build_backend_convergence_plan,
    load_backend_convergence_matrix,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MATRIX = _ROOT / "products" / "swebench" / "backend_convergence_matrix.yaml"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="swebench-backend-convergence")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=_DEFAULT_MATRIX,
        help="Matrix to authenticate; defaults to the frozen Astra/Fable v1 file.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("calibration", "pilot", "heldout"),
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "implement"),
        default="dry-run",
        help="Mode embedded in cell commands; this planner never executes them.",
    )
    parser.add_argument(
        "--plan-out",
        type=Path,
        default=None,
        help="Create this plan file exclusively instead of writing JSON to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        matrix = load_backend_convergence_matrix(args.matrix)
        plan = build_backend_convergence_plan(
            matrix,
            phase=args.phase,
            dataset_path=args.dataset,
            output_root=args.output_root,
            repo_cache_dir=args.repo_cache_dir,
            mode=args.mode,
        )
    except BackendConvergenceMatrixError as exc:
        print(f"matrix rejected: {exc}", file=sys.stderr)
        return 2

    payload = plan.to_json_bytes()
    if args.plan_out is None:
        sys.stdout.buffer.write(payload)
    else:
        try:
            args.plan_out.parent.mkdir(parents=True, exist_ok=True)
            with args.plan_out.open("xb") as stream:
                stream.write(payload)
        except FileExistsError:
            print(
                f"plan output already exists; refusing overwrite: {args.plan_out}",
                file=sys.stderr,
            )
            return 2
        except OSError as exc:
            print(f"cannot create plan output: {exc}", file=sys.stderr)
            return 2
    print(
        f"phase={plan.phase.name} tasks={len(plan.selected_tasks)} "
        f"cells={len(plan.cells)} nominal_budget_usd="
        f"{plan.nominal_configured_budget_usd} "
        f"plan_digest={plan.payload()['plan_digest']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
