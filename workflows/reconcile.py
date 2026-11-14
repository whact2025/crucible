# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Deterministically reconcile crash state and Spec lifecycle authority.

This workflow never invokes a model.  It recovers abandoned mutation
journals, closes dead run owners, checks that every planned slice has a
plan-bound successful Implementation, requires exact passed verification for
Handoff candidates, reconstructs dependency-ordered Git postimages across the
full planned surface, reruns deterministic Spec rules, and atomically publishes
the verdict plus ``landed`` lifecycle only if the validated Spec remains Corpus
latest. Missing evidence remains incomplete; use ``handoff`` to capture
operator-authored work before reconciling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from assay import (
    ContentHash,
    ExecutionContext,
    Implementation,
    LifecycleEvent,
    Spec,
    ValidationRuleset,
)
from assay.canonical import hash_record
from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus
from assay.primitives.validate import validate

from crucible.rules import available as available_rules
from crucible.rules.cited_path_exists import CitedPathExistsContext

from . import _build_verify, _outcome, _partition
from ._args import add_spec_arg
from ._mutation_journal import (
    MutationJournalError,
    _sanitized_git_env,
    normalize_scope_path,
    recover_mutation_journals,
    recovery_workspace_mutation_lock,
)
from ._run import open_run_with_capture, record_handled_terminal


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="crucible reconcile")
    add_spec_arg(parser)
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Git workspace associated with mutation journals (default: cwd)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the reconciliation decision without publishing records",
    )
    return parser.parse_args(argv)


def _execution(cwd: Path) -> ExecutionContext:
    return ExecutionContext(
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
        machine=socket.gethostname(),
        cwd=str(cwd),
        toolchain={},
        backend=None,
        model=None,
        recipe="reconcile",
        recipe_commit=None,
        gpu_model=None,
        run_id=None,
    )


def _already_landed(corpus: Corpus, spec) -> bool:
    if not spec.lifecycle_event_refs:
        return False
    event = corpus.get_record(
        spec.lifecycle_event_refs[-1],
        as_type=LifecycleEvent,
    )
    return event.kind == "landed" and spec.version in event.affected_versions


@dataclass(frozen=True)
class _GitPathState:
    mode: str | None
    object_id: str | None
    content: bytes | None


@dataclass(frozen=True)
class _PlanImplementationEvidence:
    tier_id: str
    slice_id: str
    implementation: Implementation
    scope: frozenset[str]
    files_in_scope: frozenset[str]
    base_compatible: bool


@dataclass(frozen=True)
class _DirectiveScope:
    exact_paths: frozenset[str]
    recursive_directories: frozenset[str]


class _SpecSnapshotChanged(RuntimeError):
    """The validated Spec stopped being Corpus latest before publication."""


def _git_env(**overrides: str) -> dict[str, str]:
    """Return the closed Git environment used by every reconcile probe."""

    return _sanitized_git_env(GIT_NO_REPLACE_OBJECTS="1", **overrides)


def _expected_spec_transform(expected: Spec, expected_digest: str):
    expected_hash = ContentHash(expected_digest)

    def require_expected_parent(candidate: Spec) -> Spec:
        if candidate.parent != expected_hash:
            raise _SpecSnapshotChanged(
                "latest Spec changed after reconcile captured its snapshot"
            )
        return candidate

    return require_expected_parent


def _resolve_checkout_commit(repo: Path, revision: str) -> str | None:
    """Resolve a checkout-local commit to its full id, failing closed."""

    if not revision:
        return None

    try:
        result = subprocess.run(
            [
                "git", "-c", "core.hooksPath=/dev/null",
                "rev-parse", "--verify", "--end-of-options",
                f"{revision}^{{commit}}",
            ],
            cwd=repo,
            env=_git_env(),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    head = result.stdout.strip()
    return head or None


def _path_absent_at_commit(repo: Path, commit: str, path: str) -> bool:
    """Prove an exact repo-relative path is absent from ``commit``.

    ``False`` is the fail-closed answer for malformed paths, Git failures, and
    timeouts. The literal pathspec prevents wildcard/pathspec magic from
    turning one declared create path into broader tree authority.
    """

    if not _canonical_repo_path(repo, path):
        return False
    try:
        result = subprocess.run(
            [
                "git", "-c", "core.hooksPath=/dev/null",
                "ls-tree", "-z", "--full-tree", commit, "--",
                f":(literal){path}",
            ],
            cwd=repo,
            env=_git_env(),
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and not result.stdout


def _implementation_postimages(
    repo: Path,
    base_commit: str,
    diff_bytes: bytes | None,
    paths: set[str],
) -> tuple[
    dict[str, _GitPathState],
    frozenset[str],
    frozenset[tuple[str, str]],
] | None:
    """Reconstruct exact Git postimages without mutating the source repo.

    Implementation diffs are cumulative ``git diff --binary HEAD`` patches.
    Apply one selected final diff to a fresh base index backed by an isolated
    object directory, then read each requested path's Git mode and blob bytes.
    Any setup, apply, or lookup failure returns ``None`` (no authority).
    """

    if any(not _canonical_repo_path(repo, path) for path in paths):
        return None
    try:
        object_result = subprocess.run(
            [
                "git", "-c", "core.hooksPath=/dev/null", "rev-parse",
                "--path-format=absolute", "--git-path", "objects",
            ],
            cwd=repo,
            env=_git_env(),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if object_result.returncode != 0 or not object_result.stdout.strip():
        return None
    source_objects = object_result.stdout.strip()

    try:
        with tempfile.TemporaryDirectory(prefix="crucible-reconcile-") as raw_tmp:
            temp_root = Path(raw_tmp)
            temp_objects = temp_root / "objects"
            temp_objects.mkdir()
            env = _git_env(
                GIT_INDEX_FILE=str(temp_root / "index"),
                GIT_OBJECT_DIRECTORY=str(temp_objects),
                GIT_ALTERNATE_OBJECT_DIRECTORIES=source_objects,
            )

            read_tree = subprocess.run(
                [
                    "git", "-c", "core.hooksPath=/dev/null",
                    "read-tree", base_commit,
                ],
                cwd=repo,
                env=env,
                capture_output=True,
                timeout=10,
            )
            if read_tree.returncode != 0:
                return None
            if diff_bytes:
                apply = subprocess.run(
                    [
                        "git", "-c", "core.hooksPath=/dev/null",
                        "apply", "--cached", "--whitespace=nowarn", "-",
                    ],
                    cwd=repo,
                    env=env,
                    input=diff_bytes,
                    capture_output=True,
                    timeout=10,
                )
                if apply.returncode != 0:
                    return None

            changed = subprocess.run(
                [
                    "git", "-c", "core.hooksPath=/dev/null",
                    "diff-index", "--cached", "-M", "--name-status", "-z",
                    base_commit, "--",
                ],
                cwd=repo,
                env=env,
                capture_output=True,
                timeout=5,
            )
            if changed.returncode != 0:
                return None
            changed_records = [
                item for item in changed.stdout.split(b"\0") if item
            ]
            touched: set[str] = set()
            rename_pairs: set[tuple[str, str]] = set()
            index = 0
            while index < len(changed_records):
                status = changed_records[index]
                index += 1
                if not status:
                    return None
                if status.startswith((b"R", b"C")):
                    if index + 1 >= len(changed_records):
                        return None
                    source = os.fsdecode(changed_records[index])
                    destination = os.fsdecode(changed_records[index + 1])
                    index += 2
                    if status.startswith(b"R"):
                        touched.update((source, destination))
                        rename_pairs.add((source, destination))
                    else:
                        touched.add(destination)
                else:
                    if index >= len(changed_records):
                        return None
                    touched.add(os.fsdecode(changed_records[index]))
                    index += 1
            touched_paths = frozenset(touched)

            result: dict[str, _GitPathState] = {}
            for path in sorted(paths | touched):
                entry = subprocess.run(
                    [
                        "git", "-c", "core.hooksPath=/dev/null",
                        "ls-files", "--stage", "-z", "--",
                        f":(literal){path}",
                    ],
                    cwd=repo,
                    env=env,
                    capture_output=True,
                    timeout=5,
                )
                records = [item for item in entry.stdout.split(b"\0") if item]
                if entry.returncode != 0 or len(records) > 1:
                    return None
                if not records:
                    result[path] = _GitPathState(
                        mode=None,
                        object_id=None,
                        content=None,
                    )
                    continue
                try:
                    metadata, encoded_path = records[0].split(b"\t", 1)
                    mode, object_id, stage = metadata.split()
                except ValueError:
                    return None
                if stage != b"0" or os.fsdecode(encoded_path) != path:
                    return None
                blob = subprocess.run(
                    [
                        "git", "-c", "core.hooksPath=/dev/null",
                        "cat-file", "blob", object_id.decode("ascii"),
                    ],
                    cwd=repo,
                    env=env,
                    capture_output=True,
                    timeout=5,
                )
                if blob.returncode != 0:
                    return None
                result[path] = _GitPathState(
                    mode=mode.decode("ascii"),
                    object_id=object_id.decode("ascii"),
                    content=blob.stdout,
                )
            return result, touched_paths, frozenset(rename_pairs)
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return None


def _canonical_repo_path(repo: Path, path: str) -> bool:
    try:
        return normalize_scope_path(repo, path) == path
    except (MutationJournalError, OSError, ValueError):
        return False


def _base_path_is_tree(
    repo: Path,
    base_commit: str,
    path: str,
) -> bool | None:
    try:
        result = subprocess.run(
            [
                "git", "-c", "core.hooksPath=/dev/null",
                "ls-tree", "-z", "--full-tree", base_commit, "--",
                f":(literal){path}",
            ],
            cwd=repo,
            env=_git_env(),
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    records = [item for item in result.stdout.split(b"\0") if item]
    if not records:
        return False
    if len(records) != 1:
        return None
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        _mode, object_type, _object_id = metadata.split()
    except ValueError:
        return None
    if os.fsdecode(encoded_path) != path:
        return None
    return object_type == b"tree"


def _directive_scope(
    repo: Path,
    base_commit: str,
    rendered_paths: frozenset[str],
) -> _DirectiveScope | None:
    exact: set[str] = set()
    recursive: set[str] = set()
    for rendered in rendered_paths:
        try:
            normalized = normalize_scope_path(repo, rendered)
        except (MutationJournalError, OSError, ValueError):
            return None
        if rendered not in {normalized, f"{normalized}/"}:
            return None
        base_tree = _base_path_is_tree(repo, base_commit, normalized)
        if base_tree is None:
            return None
        base_absent = (
            not base_tree
            and _path_absent_at_commit(repo, base_commit, normalized)
        )
        if rendered.endswith("/") and not base_tree and not base_absent:
            return None
        if rendered.endswith("/") or base_tree:
            recursive.add(normalized)
        else:
            exact.add(normalized)
    return _DirectiveScope(
        exact_paths=frozenset(exact),
        recursive_directories=frozenset(recursive),
    )


def _path_allowed_by_scope(path: str, scope: _DirectiveScope) -> bool:
    if path in scope.exact_paths:
        return True
    return any(
        path == directory or path.startswith(f"{directory}/")
        for directory in scope.recursive_directories
    )


def _recursive_scope_children(
    repo: Path,
    base_commit: str,
    directory: str,
) -> set[str] | None:
    try:
        base_result = subprocess.run(
            [
                "git", "-c", "core.hooksPath=/dev/null",
                "ls-tree", "-r", "-z", "--name-only", base_commit, "--",
                f":(literal){directory}",
            ],
            cwd=repo,
            env=_git_env(),
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if base_result.returncode != 0:
        return None
    children = {
        os.fsdecode(item)
        for item in base_result.stdout.split(b"\0")
        if item
    }

    try:
        current_result = subprocess.run(
            [
                "git", "-c", "core.hooksPath=/dev/null",
                "ls-files", "-z", "--cached", "--others",
                "--exclude-standard", "--", f":(literal){directory}",
            ],
            cwd=repo,
            env=_git_env(),
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if current_result.returncode != 0:
        return None
    children.update(
        os.fsdecode(item)
        for item in current_result.stdout.split(b"\0")
        if item
    )
    if any(not _canonical_repo_path(repo, child) for child in children):
        return None
    return children


def _resolve_source_root(cwd: Path, submodule_path: str | None) -> Path | None:
    """Resolve a declared child checkout without permitting workspace escape."""

    workspace = cwd.resolve()
    if not submodule_path:
        return workspace
    if "\\" in submodule_path:
        return None
    relative = Path(submodule_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(
            part.casefold() in {".git", ".hg", ".svn"}
            for part in relative.parts
        )
    ):
        return None
    source_root = (workspace / relative).resolve()
    try:
        source_root.relative_to(workspace)
    except ValueError:
        return None
    return source_root


def _workspace_matches_postimage(
    repo: Path,
    path: str,
    postimage: _GitPathState,
) -> bool:
    """Compare Git-representable type, executable mode, and exact bytes."""

    if not _canonical_repo_path(repo, path):
        return False
    mode = postimage.mode
    expected_object_id = postimage.object_id
    expected_bytes = postimage.content
    full = repo / path
    if mode is None:
        return not os.path.lexists(full)
    if expected_object_id is None or expected_bytes is None:
        return False
    try:
        current = os.lstat(full)
        if mode in {"100644", "100755"}:
            if not stat.S_ISREG(current.st_mode):
                return False
            actual_mode = (
                "100755" if current.st_mode & stat.S_IXUSR else "100644"
            )
            if actual_mode != mode:
                return False
            actual_bytes = full.read_bytes()
            if actual_bytes == expected_bytes:
                return True

            # The isolated index contains Git's canonical blob bytes, while
            # the checkout may contain a built-in representation such as
            # CRLF under ``core.autocrlf``.  Never ask Git to canonicalize a
            # path with an effective filter driver: ``hash-object --path``
            # would execute project-controlled clean/process commands during
            # this nominally read-only gate.  ``check-attr`` only resolves
            # metadata and does not invoke the driver.
            attribute = subprocess.run(
                [
                    "git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "check-attr",
                    "-z",
                    "filter",
                    "--stdin",
                ],
                cwd=repo,
                env=_git_env(),
                input=os.fsencode(path) + b"\0",
                capture_output=True,
                check=False,
            )
            if attribute.returncode != 0:
                return False
            fields = attribute.stdout.split(b"\0")
            if fields and fields[-1] == b"":
                fields.pop()
            if len(fields) != 3 or fields[0] != os.fsencode(path):
                return False
            if fields[1].lower() != b"filter":
                return False
            if fields[2].lower() not in {b"unset", b"unspecified"}:
                return False

            canonical = subprocess.run(
                [
                    "git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "hash-object",
                    f"--path={path}",
                    "--stdin",
                ],
                cwd=repo,
                env=_git_env(),
                input=actual_bytes,
                capture_output=True,
                check=False,
            )
            if canonical.returncode != 0:
                return False
            try:
                actual_object_id = canonical.stdout.decode("ascii").strip()
            except UnicodeDecodeError:
                return False
            return actual_object_id == expected_object_id
        if mode == "120000":
            return (
                stat.S_ISLNK(current.st_mode)
                and os.fsencode(os.readlink(full)) == expected_bytes
            )
    except OSError:
        return False
    # A planned file create cannot safely authorize gitlinks, directories, or
    # unknown future index modes.
    return False


def main(argv: list[str] | None = None) -> int:
    return _main(argv, lock_held=False, locked_root=None)


def _main(
    argv: list[str] | None,
    *,
    lock_held: bool,
    locked_root: Path | None,
) -> int:
    args = _parse_args(argv)
    corpus = Corpus(args.spec)
    spec = corpus.latest_spec()
    if spec is None:
        record_handled_terminal(
            corpus,
            "reconcile",
            status="setup_error",
            message=f"no spec manifest under {args.spec}",
            error_kind="missing_spec",
            provenance_root=None,
        )
        return 2
    cwd = (args.cwd or Path.cwd()).resolve()
    source_root = _resolve_source_root(
        cwd,
        spec.project_context.submodule_path,
    )
    if source_root is None:
        record_handled_terminal(
            corpus,
            "reconcile",
            status="setup_error",
            message="spec submodule_path escapes the requested workspace",
            error_kind="invalid_submodule_path",
            provenance_root=None,
        )
        return 2

    # Re-enter once under the process+kernel workspace lock so the
    # authoritative Spec read, journal recovery, evidence reconstruction,
    # filesystem comparison, validation, and Corpus lifecycle commit are one
    # TOCTOU-free critical section. Argument parsing is deterministic; the
    # second invocation deliberately re-reads the latest Spec under the lock.
    if not lock_held:
        with ExitStack() as workspace_release:
            workspace_context = recovery_workspace_mutation_lock(
                corpus,
                source_root,
            )
            workspace_release.callback(workspace_context.retry_cleanup)
            workspace_context.__enter__()
            return _main(
                argv,
                lock_held=True,
                locked_root=source_root,
            )
    if locked_root != source_root:
        record_handled_terminal(
            corpus,
            "reconcile",
            status="setup_error",
            message="spec source root changed while acquiring workspace lock",
            error_kind="workspace_lock_root_changed",
            provenance_root=None,
        )
        return 2

    stale_runs = _outcome.reconcile_stale_runs(corpus)
    recovery = recover_mutation_journals(corpus, source_root)
    if recovery.failures:
        record_handled_terminal(
            corpus,
            "reconcile",
            status="setup_error",
            message="; ".join(recovery.failures),
            error_kind="mutation_recovery_failed",
            provenance_root=None,
        )
        return 2

    try:
        with corpus.locked_spec_authority() as authority_spec:
            if authority_spec != spec:
                raise _SpecSnapshotChanged(
                    "latest Spec changed while reconcile captured its "
                    "execution snapshot"
                )
            persisted_spec, persisted_digest = corpus.get_spec_with_digest(
                spec.version
            )
            if persisted_spec != spec:
                raise _SpecSnapshotChanged(
                    "hydrated Spec differs from its persisted manifest "
                    "authority"
                )
            spec_digest = str(persisted_digest)
    except Exception as exc:  # noqa: BLE001 - setup authority fails closed
        record_handled_terminal(
            corpus,
            "reconcile",
            status="setup_error",
            message=(
                "Spec authority is unreadable before reconciliation: "
                f"{type(exc).__name__}: {exc}"
            ),
            error_kind="invalid_spec_authority",
            provenance_root=None,
        )
        return 2

    with open_run_with_capture(
        corpus,
        "reconcile",
        # CLI cwd may be a non-Git outer workspace. Source reconciliation and
        # mutation ownership both bind the resolved target checkout.
        provenance_root=source_root,
    ) as handle:
        if spec.tier_dag is None:
            summary: dict[str, object] = {
                "status": "incomplete",
                "reason": "spec has no tier_dag",
                "stale_runs_reconciled": list(stale_runs),
                "mutation_recovery": {
                    "recovered": list(recovery.recovered),
                    "preserved": list(recovery.preserved),
                },
            }
            handle.write_summary(summary)
            handle.finalize_elapsed(RunResult(status="incomplete"))
            print(json.dumps(summary, indent=2))
            return 1

        rows: list[dict[str, object]] = []
        all_complete = True
        all_verification_complete = True
        successful_plan_bound_slices: set[tuple[str, str]] = set()
        required_create_paths: set[tuple[str, str, str]] = set()
        candidate_create_paths: set[tuple[str, str, str]] = set()
        verified_create_paths: set[tuple[str, str, str]] = set()
        successful_records: list[_PlanImplementationEvidence] = []
        tier_ancestors: dict[str, frozenset[str]] = {}
        current_head = _resolve_checkout_commit(source_root, "HEAD")
        project_base = _resolve_checkout_commit(
            source_root,
            spec.project_context.base_commit,
        )
        resolved_implementation_bases: dict[str, str | None] = {}
        base_path_absence: dict[str, bool] = {}
        for tier in spec.tier_dag.topo_order():
            ancestors = set(tier.depends_on)
            for dependency in tier.depends_on:
                ancestors.update(tier_ancestors.get(dependency, ()))
            tier_ancestors[tier.tier_id] = frozenset(ancestors)
            partition = _partition.partition_for_required_verification(
                corpus, tier, spec=spec,
            )
            landed = partition.retained
            pending = partition.implementation_pending
            verified_retained = dict(partition.verified_retained)
            verification_pending = dict(partition.verification_pending)
            lifecycle_accepted = False
            lifecycle_reason = "not evaluated"
            if not pending and set(landed) == {
                item.id for item in tier.slices
            }:
                lifecycle_accepted, lifecycle_reason = (
                    _build_verify._tier_lifecycle_acceptance(
                        corpus,
                        spec,
                        tier,
                        landed,
                    )
                )
            # Tier manifests prove one exact candidate closure, but Tier's
            # terminal landed lifecycle is the authority that accepted those
            # exact records. Reconcile may consume that pair; it must never
            # manufacture the missing lifecycle by treating the manifest as
            # a standalone pass.
            if (
                partition.tier_manifest_obligations
                and not lifecycle_accepted
            ):
                for slice_id in partition.tier_manifest_obligations:
                    implementation = landed.get(slice_id)
                    if implementation is None:
                        continue
                    verified_retained.pop(slice_id, None)
                    verification_pending[slice_id] = implementation
            # The structural partition remains the source-byte authority: it
            # uses only the latest lineage-attached record per slice and
            # accepts it only when its resume identity matches this exact
            # TierDAG, directive, project base, and implementation base.  The
            # required-verification partition is deliberately narrower for
            # landing: an exact Handoff candidate with missing, waived,
            # unavailable, corrupt, or stale gate evidence remains retained
            # for provider-free deterministic replay, but cannot be reconciled
            # into a pass. A create-path exemption must never be inferred from
            # an orphan, stale-plan, or superseded success record.
            for slice_ in tier.slices:
                implementation = landed.get(slice_.id)
                if (
                    implementation is None
                    or implementation.classification != "success"
                ):
                    continue
                successful_plan_bound_slices.add((tier.tier_id, slice_.id))
                required_create_paths.update(
                    (tier.tier_id, slice_.id, path)
                    for path in slice_.directive.creates_paths
                )
                # Existing-path authority is narrower than completion: the
                # exact declared create must be in this Implementation's
                # files_modified, and all three base identities must agree.
                # Any mismatch fails closed as an ordinary collision.
                if (
                    implementation.base_commit
                    not in resolved_implementation_bases
                ):
                    resolved_implementation_bases[implementation.base_commit] = (
                        _resolve_checkout_commit(
                            source_root,
                            implementation.base_commit,
                        )
                    )
                implementation_base = resolved_implementation_bases[
                    implementation.base_commit
                ]
                base_compatible = bool(
                    current_head is not None
                    and project_base is not None
                    and implementation_base is not None
                    and current_head == project_base
                    and implementation_base == project_base
                )
                successful_records.append(_PlanImplementationEvidence(
                    tier_id=tier.tier_id,
                    slice_id=slice_.id,
                    implementation=implementation,
                    scope=frozenset((
                        *slice_.directive.files_in_scope,
                        *slice_.directive.creates_paths,
                    )),
                    files_in_scope=frozenset(
                        slice_.directive.files_in_scope,
                    ),
                    base_compatible=base_compatible,
                ))
                if not base_compatible or project_base is None:
                    continue
                modified = set(implementation.files_modified)
                for path in slice_.directive.creates_paths:
                    if path not in modified:
                        continue
                    if path not in base_path_absence:
                        base_path_absence[path] = _path_absent_at_commit(
                            source_root,
                            project_base,
                            path,
                        )
                    if base_path_absence[path]:
                        candidate_create_paths.add(
                            (tier.tier_id, slice_.id, path),
                        )
            pending_ids = [item.id for item in pending]
            rows.append({
                "tier_id": tier.tier_id,
                "landed_slices": sorted(landed),
                "pending_slices": pending_ids,
                "verified_slices": sorted(verified_retained),
                "verification_pending_slices": sorted(
                    verification_pending
                ),
                "tier_manifest_obligations": list(
                    partition.tier_manifest_obligations
                ),
                "tier_manifest_ref": partition.tier_manifest_ref,
                "tier_manifest_verified": (
                    partition.tier_manifest_verified
                ),
                "lifecycle_accepted": lifecycle_accepted,
                "lifecycle_reason": lifecycle_reason,
            })
            all_complete = all_complete and not pending_ids
            all_verification_complete = (
                all_verification_complete and not verification_pending
            )

        postimage_errors: list[str] = []
        evidence_states: dict[
            tuple[str, str],
            dict[str, _GitPathState],
        ] = {}
        evidence_touched_paths: dict[
            tuple[str, str],
            frozenset[str],
        ] = {}
        record_scope_policies: dict[
            tuple[str, str],
            _DirectiveScope,
        ] = {}
        record_files_in_scope_policies: dict[
            tuple[str, str],
            _DirectiveScope,
        ] = {}
        evidence_valid: set[tuple[str, str]] = set()

        # Establish that every plan-bound success is internally consistent:
        # its cumulative patch must apply to the agreed base, its actual
        # touched paths must exactly equal files_modified, and both must stay
        # inside the directive surface. This prevents either record metadata
        # or an attached diff from granting broader workspace authority.
        if project_base is None:
            postimage_errors.append(
                "declared project base does not resolve in the current checkout"
            )
        for record in successful_records:
            identity = (record.tier_id, record.slice_id)
            implementation = record.implementation
            if project_base is None or not record.base_compatible:
                postimage_errors.append(
                    f"{record.tier_id}/{record.slice_id}: implementation, "
                    "project, and checkout bases do not agree"
                )
                continue
            scope_policy = _directive_scope(
                source_root,
                project_base,
                record.scope,
            )
            if scope_policy is None:
                postimage_errors.append(
                    f"{record.tier_id}/{record.slice_id}: directive surface "
                    "is unsafe or cannot be resolved"
                )
                continue
            record_scope_policies[identity] = scope_policy
            files_in_scope_policy = _directive_scope(
                source_root,
                project_base,
                record.files_in_scope,
            )
            if files_in_scope_policy is None:
                postimage_errors.append(
                    f"{record.tier_id}/{record.slice_id}: files_in_scope "
                    "surface is unsafe or cannot be resolved"
                )
                continue
            record_files_in_scope_policies[identity] = files_in_scope_policy
            files_modified = tuple(implementation.files_modified)
            modified = set(files_modified)
            if implementation.scope_escapes:
                postimage_errors.append(
                    f"{record.tier_id}/{record.slice_id}: successful "
                    "Implementation carries scope escapes "
                    f"{sorted(implementation.scope_escapes)!r}"
                )
                continue
            if len(modified) != len(files_modified):
                postimage_errors.append(
                    f"{record.tier_id}/{record.slice_id}: files_modified "
                    "contains duplicate paths"
                )
                continue
            if (
                any(not _canonical_repo_path(source_root, p) for p in modified)
                or any(
                    not _path_allowed_by_scope(p, scope_policy)
                    for p in modified
                )
            ):
                postimage_errors.append(
                    f"{record.tier_id}/{record.slice_id}: implementation paths "
                    "are non-canonical or outside the directive surface"
                )
                continue
            try:
                diff_bytes = corpus.get_artifact(implementation.diff_ref)
            except (OSError, TypeError, ValueError, KeyError):
                postimage_errors.append(
                    f"{record.tier_id}/{record.slice_id}: diff_ref is unavailable"
                )
                continue
            reconstructed = _implementation_postimages(
                source_root,
                project_base,
                diff_bytes,
                modified,
            )
            if reconstructed is None:
                postimage_errors.append(
                    f"{record.tier_id}/{record.slice_id}: cumulative diff "
                    "cannot be reconstructed against the agreed base"
                )
                continue
            postimages, touched, rename_pairs = reconstructed
            rename_sources = {source for source, _ in rename_pairs}
            metadata_touched = touched - rename_sources
            if (
                any(
                    not _path_allowed_by_scope(p, scope_policy)
                    for p in touched
                )
                or metadata_touched != frozenset(modified)
            ):
                postimage_errors.append(
                    f"{record.tier_id}/{record.slice_id}: diff touched paths "
                    f"{sorted(touched)!r} (renames {sorted(rename_pairs)!r}) "
                    "do not match the directive surface/files_modified "
                    f"{sorted(modified)!r}"
                )
                continue
            evidence_states[identity] = postimages
            evidence_touched_paths[identity] = touched
            evidence_valid.add(identity)

        planned_surface: set[str] = set()
        surface_valid = True
        recursive_children_cache: dict[str, set[str] | None] = {}
        for tier in spec.tier_dag.tiers:
            for slice_ in tier.slices:
                identity = (tier.tier_id, slice_.id)
                raw_scope = frozenset((
                    *slice_.directive.files_in_scope,
                    *slice_.directive.creates_paths,
                ))
                scope_policy = record_scope_policies.get(identity)
                if scope_policy is None and project_base is not None:
                    scope_policy = _directive_scope(
                        source_root,
                        project_base,
                        raw_scope,
                    )
                if scope_policy is None:
                    surface_valid = False
                    postimage_errors.append(
                        f"{tier.tier_id}/{slice_.id}: planned surface is "
                        "unsafe or cannot be resolved"
                    )
                    continue
                planned_surface.update(scope_policy.exact_paths)
                for directory in scope_policy.recursive_directories:
                    if directory not in recursive_children_cache:
                        recursive_children_cache[directory] = (
                            _recursive_scope_children(
                                source_root,
                                project_base,
                                directory,
                            )
                            if project_base is not None
                            else None
                        )
                    children = recursive_children_cache[directory]
                    if children is None:
                        surface_valid = False
                        postimage_errors.append(
                            f"recursive scope '{directory}' cannot be expanded"
                        )
                    else:
                        planned_surface.update(children)
                planned_surface.update(
                    evidence_touched_paths.get(identity, ())
                )

        expected_states: dict[str, _GitPathState] = {}
        chain_valid_paths: set[str] = set()
        planned_deletion_paths: set[str] = set()
        untouched_paths: set[str] = set()
        if all_complete and project_base is not None and surface_valid:
            for path in sorted(planned_surface):
                writers = [
                    record
                    for record in successful_records
                    if path in evidence_touched_paths.get(
                        (record.tier_id, record.slice_id),
                        (),
                    )
                ]
                if not writers:
                    untouched_paths.add(path)
                    continue
                chain_valid = True
                for prior, current in zip(
                    writers, writers[1:], strict=False,
                ):
                    if prior.tier_id not in tier_ancestors.get(
                        current.tier_id,
                        (),
                    ):
                        postimage_errors.append(
                            f"{path}: incomparable writers "
                            f"{prior.tier_id}/{prior.slice_id} and "
                            f"{current.tier_id}/{current.slice_id}"
                        )
                        chain_valid = False
                        break
                final_writer = writers[-1]
                final_identity = (
                    final_writer.tier_id,
                    final_writer.slice_id,
                )
                postimage = evidence_states.get(final_identity, {}).get(path)
                if not chain_valid or postimage is None:
                    continue
                expected_states[path] = postimage
                chain_valid_paths.add(path)
                if postimage.mode is None:
                    planned_deletion_paths.add(path)

            if untouched_paths:
                base_reconstruction = _implementation_postimages(
                    source_root,
                    project_base,
                    None,
                    untouched_paths,
                )
                if base_reconstruction is None:
                    postimage_errors.append(
                        "untouched planned surface cannot be reconstructed "
                        "from the agreed base"
                    )
                else:
                    base_states, base_touched, base_renames = base_reconstruction
                    if base_touched:
                        postimage_errors.append(
                            "base reconstruction unexpectedly reported "
                            f"changed paths: {sorted(base_touched)!r}"
                        )
                    elif base_renames:
                        postimage_errors.append(
                            "base reconstruction unexpectedly reported "
                            f"renames: {sorted(base_renames)!r}"
                        )
                    else:
                        expected_states.update(base_states)
                        chain_valid_paths.update(base_states)

        expected_plan_surface_paths = (
            frozenset(planned_surface) if all_complete else frozenset()
        )
        verified_plan_surface_paths = frozenset(
            path
            for path, postimage in expected_states.items()
            if _workspace_matches_postimage(source_root, path, postimage)
        )

        for candidate in candidate_create_paths:
            creator_identity = (candidate[0], candidate[1])
            path = candidate[2]
            postimage = expected_states.get(path)
            creator_postimage = evidence_states.get(
                creator_identity,
                {},
            ).get(path)
            if (
                creator_identity in evidence_valid
                and creator_postimage is not None
                and creator_postimage.mode is not None
                and path in chain_valid_paths
                and path in verified_plan_surface_paths
                and postimage is not None
                and postimage.mode is not None
            ):
                verified_create_paths.add(candidate)

        verified_recursive_scope_paths = frozenset(
            (*identity, directory)
            for identity in (
                evidence_valid & successful_plan_bound_slices
            )
            for directory in record_files_in_scope_policies[
                identity
            ].recursive_directories
        )

        context = CitedPathExistsContext(
            workspace_root=cwd,
            submodule_path=spec.project_context.submodule_path,
            successful_plan_bound_slices=frozenset(
                successful_plan_bound_slices,
            ),
            verified_create_paths=frozenset(verified_create_paths),
            expected_plan_surface_paths=expected_plan_surface_paths,
            verified_plan_surface_paths=verified_plan_surface_paths,
            verified_recursive_scope_paths=verified_recursive_scope_paths,
            planned_deletion_paths=frozenset(planned_deletion_paths),
            postimage_evidence_errors=tuple(postimage_errors),
        )
        verdict = validate(
            spec,
            ruleset=ValidationRuleset(
                name="reconcile",
                rule_ids=available_rules(),
            ),
            ctx=context,
        )
        workspace_evidence_passed = (
            not postimage_errors
            and expected_plan_surface_paths == verified_plan_surface_paths
            and required_create_paths.issubset(verified_create_paths)
        )
        eligible = (
            all_complete
            and all_verification_complete
            and workspace_evidence_passed
            and verdict.passed
        )
        already_landed = _already_landed(corpus, spec)
        committed_version: int | None = None
        lifecycle_ref: str | None = None
        verdict_ref: str | None = None
        publication_error: str | None = None
        expected_spec_transform = _expected_spec_transform(
            spec,
            spec_digest,
        )

        try:
            if eligible and already_landed:
                # A no-op Corpus commit is an atomic latest-Spec read under
                # the Corpus write lock. Compare that immutable snapshot to
                # the validated Spec before reporting no_work; otherwise a
                # concurrent DAG extension can turn stale success into a
                # misleading terminal result even though we publish nothing.
                latest_result = corpus.commit()
                observed_spec, observed_digest = corpus.get_spec_with_digest(
                    latest_result.spec.version
                )
                if (
                    observed_spec != latest_result.spec
                    or latest_result.spec != spec
                    or str(observed_digest) != spec_digest
                ):
                    raise _SpecSnapshotChanged(
                        "latest Spec changed after reconcile captured its "
                        "already-landed snapshot"
                    )
                status = "no_work"
            elif args.dry_run:
                status = "dry_run_complete" if eligible else "incomplete"
            elif eligible:
                result = corpus.commit(
                    verdicts=(verdict,),
                    lifecycle_event_factory=lambda version: LifecycleEvent(
                        kind="landed",
                        reason=(
                            "deterministic reconciliation confirmed all slices"
                        ),
                        affected_versions=(version,),
                        execution=_execution(cwd),
                    ),
                    transform=expected_spec_transform,
                )
                committed_version = result.spec.version
                verdict_ref = str(result.refs.verdict_refs[0])
                lifecycle_ref = str(result.refs.lifecycle_event_refs[0])
                status = "landed"
            else:
                result = corpus.commit(
                    verdicts=(verdict,),
                    transform=expected_spec_transform,
                )
                committed_version = result.spec.version
                verdict_ref = str(result.refs.verdict_refs[0])
                status = "incomplete"
        except _SpecSnapshotChanged as exc:
            # Corpus.commit evaluates transforms while holding its Spec write
            # lock, making this an atomic compare-and-swap against the exact
            # Spec whose DAG/evidence was validated above.
            eligible = False
            status = "concurrent_spec_change"
            publication_error = str(exc)

        summary = {
            "status": status,
            "eligible_to_land": eligible,
            "already_landed": already_landed,
            "tiers": rows,
            "validation": {
                "decision": verdict.decision,
                "finding_count": len(verdict.findings),
                "verdict_ref": verdict_ref,
            },
            "workspace_evidence": {
                "passed": workspace_evidence_passed,
                "expected_path_count": len(expected_plan_surface_paths),
                "verified_path_count": len(verified_plan_surface_paths),
                "required_create_path_count": len(required_create_paths),
                "verified_create_path_count": len(verified_create_paths),
                "error_count": len(postimage_errors),
            },
            "committed_version": committed_version,
            "lifecycle_ref": lifecycle_ref,
            "publication_error": publication_error,
            "stale_runs_reconciled": list(stale_runs),
            "mutation_recovery": {
                "recovered": list(recovery.recovered),
                "preserved": list(recovery.preserved),
            },
        }
        handle.write_summary(summary)
        handle.finalize_elapsed(RunResult(status=status))
        print(json.dumps(summary, indent=2))
        return 0 if status in {"landed", "no_work", "dry_run_complete"} else 1


if __name__ == "__main__":
    sys.exit(main())
