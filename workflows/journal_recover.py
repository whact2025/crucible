# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Check or apply an exact, externally-authorized mutation-journal recovery."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from assay.corpus.store import Corpus

from ._args import add_spec_arg
from ._journal_recovery import (
    ExternalRecoveryRequest,
    inspect_external_recovery,
    load_accepted_ancestor_submodule,
    preflight_external_recovery_authority,
)
from ._mutation_journal import (
    MutationJournalError,
    recovery_workspace_mutation_lock,
)
from .reconcile import _resolve_source_root


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="crucible journal_recover")
    add_spec_arg(parser)
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help=(
            "outer workspace root; the latest Spec submodule_path is resolved "
            "exactly once beneath this directory"
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--transaction-id", required=True)
    parser.add_argument("--expected-diff-ref", required=True)
    parser.add_argument(
        "--accept-removed-linked-worktree",
        action="append",
        default=[],
        metavar="PATH",
        help="exact removed guard-only linked worktree path (repeatable)",
    )
    parser.add_argument("--accept-appended-stash-tip", required=True)
    parser.add_argument(
        "--reason",
        required=True,
        help="nonempty operator reason retained in the durable audit",
    )
    parser.add_argument(
        "--accept-ancestor-change-ref",
        metavar="SHA256_REF",
        help=(
            "transaction-bound Corpus manifest for one exact preserved "
            "ancestor change"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform recovery; without this flag the workflow is check-only",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus = Corpus(Path(args.spec))
    spec = corpus.latest_spec()
    if spec is None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"no spec manifest under {args.spec}",
                },
                indent=2,
            )
        )
        return 2
    outer = Path(args.cwd).resolve()
    source_root = _resolve_source_root(
        outer,
        spec.project_context.submodule_path,
    )
    if source_root is None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "Spec submodule_path escapes the requested outer workspace",
                },
                indent=2,
            )
        )
        return 2
    try:
        accepted_ancestor = (
            load_accepted_ancestor_submodule(
                corpus,
                args.accept_ancestor_change_ref,
            )
            if args.accept_ancestor_change_ref is not None
            else None
        )
    except (MutationJournalError, FileNotFoundError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_kind": type(exc).__name__,
                    "error": str(exc),
                    "apply": bool(args.apply),
                },
                indent=2,
            )
        )
        return 2
    request = ExternalRecoveryRequest(
        run_id=args.run_id,
        transaction_id=args.transaction_id,
        expected_diff_ref=args.expected_diff_ref,
        removed_linked_worktrees=tuple(args.accept_removed_linked_worktree),
        appended_stash_tip=args.accept_appended_stash_tip,
        reason=args.reason,
        ancestor_submodule=accepted_ancestor,
    )
    try:
        preflight_external_recovery_authority(corpus, source_root, request)
        with ExitStack() as workspace_release:
            workspace_context = recovery_workspace_mutation_lock(
                corpus,
                source_root,
            )
            workspace_release.callback(workspace_context.retry_cleanup)
            workspace_context.__enter__()
            report = inspect_external_recovery(
                corpus,
                source_root,
                request,
                apply=bool(args.apply),
            )
    except (MutationJournalError, FileNotFoundError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_kind": type(exc).__name__,
                    "error": str(exc),
                    "apply": bool(args.apply),
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps(report.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
