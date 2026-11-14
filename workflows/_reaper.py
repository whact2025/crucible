# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Lifecycle-ordered recovery of workflow-owned workspace mutations.

Exact write-ahead mutation journals are always recovered first. Legacy
Implementation records are considered only when they are attached to the
locked latest Spec, bind the current plan identity, and cannot overlap newer
or accepted bytes. Every legacy write is then made inside one durable outer
transaction and settled by a CAS-bound ``reaped`` lifecycle event.

Unattached content-addressed records, timestamps, artifact enumeration order,
and reused slice identifiers never authorize filesystem mutation.

The public entry point acquires the reentrant workspace mutation lock for
``cwd`` so direct and canonical Tier callers share the same serialization.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterator

from assay import ContentHash, Implementation, LifecycleEvent

from . import _build_verify, _partition
from ._mutation_journal import (
    MutationJournalError,
    RecoveryReport,
    WorkspaceMutationJournal,
    _RetainedAcquisitionCleanupError,
    _SpecAuthorityEntryError,
    _assert_workspace_mutation_authority,
    _git_argv,
    _recover_mutation_journals_locked,
    _readable_spec_authority,
    _sanitized_git_env,
    _validate_mutation_lifecycle_event,
    mutation_lifecycle_reason_base,
    prepare_mutation_lifecycle_settlement,
    recovery_workspace_mutation_lock,
)
from ._rollback import (
    _assert_checkout_root_binding,
    _assert_workspace_parent_binding,
    _descriptor_git_cwd,
    _directory_open_flags,
    _git_tree_entry,
    _open_checkout_root,
    _open_workspace_parent_if_present,
    _revert_scope_escapes_bound,
    _verify_worktree_path_matches_tree,
)
from ._run import exec_ctx_now

if TYPE_CHECKING:
    from assay.corpus.store import Corpus

logger = logging.getLogger(__name__)

_UNSPECIFIED = object()
_REAPER_EVENT_KIND = "reaped"
_REAPER_EVENT_PREFIX = "orphaned-dirt cleanup settlement="
_REAPER_SETTLEMENT_SCHEMA = "crucible.orphaned-dirt-settlement.v1"
_MAX_REAPER_DIFF_BYTES = 64 * 1024 * 1024
_MAX_REAPER_SETTLEMENT_BYTES = 8 * 1024 * 1024
_MAX_REAPER_ARTIFACT_AGGREGATE_BYTES = 192 * 1024 * 1024
_MAX_REAPER_ARTIFACTS = 4096
_VCS_METADATA_COMPONENTS = frozenset({
    ".git", ".hg", ".svn", ".bzr", "_darcs",
})


class _ConcurrentSpecChange(RuntimeError):
    """The cleanup decision lost its exact Spec parent before publication."""


def _is_retained_workspace_root_identity_change(
    error: MutationJournalError,
) -> bool:
    """Recognize only Crucible's typed retained-error graph."""

    pending: list[MutationJournalError] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if type(current) is MutationJournalError and current.args == (
            "external workspace root changed identity",
        ):
            return True
        if type(current) is not _RetainedAcquisitionCleanupError:
            continue
        fields = vars(current)
        for name in ("primary_error", "cleanup_error"):
            nested = fields.get(name)
            if isinstance(nested, MutationJournalError):
                pending.append(nested)
    return False


@dataclasses.dataclass
class _ReaperArtifactReader:
    """One invocation-wide bounded, deduplicating artifact authority."""

    corpus: "Corpus"
    total_bytes: int = 0
    artifact_count: int = 0
    cache: dict[str, bytes] = dataclasses.field(default_factory=dict)

    def read(
        self,
        ref: object,
        *,
        max_bytes: int,
        label: str,
    ) -> bytes:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 0
            or max_bytes > _MAX_REAPER_DIFF_BYTES
        ):
            raise ValueError(f"invalid reaper artifact ceiling for {label}")
        rendered_ref = str(ref)
        cached = self.cache.get(rendered_ref)
        if cached is not None:
            if len(cached) > max_bytes:
                raise ValueError(
                    f"cached reaper artifact exceeds {label} byte ceiling"
                )
            return cached
        if self.artifact_count >= _MAX_REAPER_ARTIFACTS:
            raise ValueError(
                "reaper artifact aggregate count budget exhausted"
            )
        remaining = _MAX_REAPER_ARTIFACT_AGGREGATE_BYTES - self.total_bytes
        if remaining < 0:
            raise ValueError(
                "reaper artifact aggregate byte budget exhausted"
            )
        read_ceiling = min(max_bytes, remaining)
        try:
            payload = self.corpus.get_artifact_bounded(
                ref,
                max_bytes=read_ceiling,
            )
        except Exception as exc:  # noqa: BLE001 - authority read fails closed
            if read_ceiling < max_bytes and isinstance(exc, ValueError):
                detail = "reaper artifact aggregate byte budget exhausted"
            else:
                detail = f"bounded {label} artifact is unreadable"
            raise ValueError(
                f"{detail} ({type(exc).__name__}: {exc})"
            ) from exc
        if not isinstance(payload, bytes) or len(payload) > read_ceiling:
            raise ValueError(
                f"bounded {label} artifact violated its byte ceiling"
            )
        self.cache[rendered_ref] = payload
        self.artifact_count += 1
        self.total_bytes += len(payload)
        return payload


@dataclasses.dataclass(frozen=True)
class _AttachedImplementation:
    index: int
    ref: str
    implementation: Implementation


@dataclasses.dataclass(frozen=True)
class _PositiveOccurrence:
    """One positive ref at its actual append-only acceptance position."""

    item: _AttachedImplementation
    # Implementation ref ``i`` occurs at boundary ``i + 1``. A lifecycle
    # attached after N refs occurs at boundary N with phase 1, so it sorts
    # after all refs visible to its accepted Spec and before the next ref.
    chronology: tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True)
class _PathAuthority:
    path: str
    chronology: tuple[int, int, int, int]
    source: str
    postimage: tuple[str, str, str] | None
    implementation_ref: str | None = None


@dataclasses.dataclass(frozen=True)
class _LegacyCandidate:
    attached: _AttachedImplementation
    diff_bytes: bytes | None
    diff_paths: tuple[str, ...]
    escape_paths: tuple[str, ...]

    @property
    def mutation_paths(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.diff_paths, *self.escape_paths)))


def _attached_implementation_history(
    corpus: "Corpus",
    spec,
) -> tuple[tuple[_AttachedImplementation, ...], dict[str, list[_AttachedImplementation]]]:
    """Hydrate every attached Implementation in authoritative ref order."""

    attached: list[_AttachedImplementation] = []
    history: dict[str, list[_AttachedImplementation]] = {}
    seen_refs: set[str] = set()
    for index, ref in enumerate(getattr(spec, "implementation_refs", ())):
        rendered_ref = str(ref)
        if rendered_ref in seen_refs:
            raise ValueError(
                "Spec lineage repeats attached Implementation ref "
                f"{rendered_ref}"
            )
        seen_refs.add(rendered_ref)
        try:
            implementation = corpus.get_record(ref, as_type=Implementation)
        except Exception as exc:  # noqa: BLE001 - mutation authority fails closed
            raise ValueError(
                f"attached Implementation {ref} is unreadable "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        if not isinstance(implementation, Implementation):
            raise TypeError(
                f"attached Implementation {ref} decoded as "
                f"{type(implementation).__name__}"
            )
        item = _AttachedImplementation(index, rendered_ref, implementation)
        attached.append(item)
        history.setdefault(implementation.slice_id, []).append(item)
    return tuple(attached), history


def _attached_lifecycle_records(
    corpus: "Corpus",
    spec,
) -> tuple[tuple[str, LifecycleEvent], ...]:
    """Hydrate every attached lifecycle event without forward-compat skips."""

    records: list[tuple[str, LifecycleEvent]] = []
    seen_refs: set[str] = set()
    for ref in getattr(spec, "lifecycle_event_refs", ()):
        rendered_ref = str(ref)
        if rendered_ref in seen_refs:
            raise ValueError(
                "Spec lineage repeats attached LifecycleEvent ref "
                f"{rendered_ref}"
            )
        seen_refs.add(rendered_ref)
        try:
            event = corpus.get_record(ref, as_type=LifecycleEvent)
        except Exception as exc:  # noqa: BLE001 - mutation authority fails closed
            raise ValueError(
                f"attached LifecycleEvent {ref} is unreadable "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        if not isinstance(event, LifecycleEvent):
            raise TypeError(
                f"attached LifecycleEvent {ref} decoded as "
                f"{type(event).__name__}"
            )
        records.append((rendered_ref, event))
    return tuple(records)


def _unattached_implementation_refs(
    corpus: "Corpus",
    attached_refs: Iterable[object],
) -> tuple[str, ...]:
    """Inventory unattached records for diagnostics, never chronology."""

    attached = {str(ref) for ref in attached_refs}
    return tuple(sorted({
        str(ref)
        for ref in corpus.list_artifacts(kind="implementations")
        if str(ref) not in attached
    }))


def _is_failure_class(classification: str) -> bool:
    if classification.startswith("hard_fail_"):
        return True
    return classification in {
        "build_failed",
        "dev_verify_failed",
        "dev_verify_unavailable",
        "partial",
        "runner_missing",
        "verify_failed",
    }


def _diff_applies_in_reverse(diff_bytes: bytes, root_fd: int) -> bool:
    """Best-effort compatibility wrapper for legacy cleanup checks."""

    try:
        return _diff_reverse_check(diff_bytes, root_fd)
    except MutationJournalError:
        return False


def _diff_reverse_check(diff_bytes: bytes, root_fd: int) -> bool:
    """Return exact reverse applicability, distinguishing Git failures.

    A stale attached record may be treated as absent only after Git has
    actually compared its accepted postimage with the workspace.  Process
    launch failures and abnormal Git exits are authority failures, not proof
    that the postimage is absent.
    """

    try:
        result = subprocess.run(
            _git_argv("apply", "-R", "--check"),
            input=diff_bytes,
            cwd=_descriptor_git_cwd(root_fd),
            pass_fds=(root_fd,),
            env=_sanitized_git_env(),
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MutationJournalError(
            "cannot inspect attached diff postimage: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise MutationJournalError(
        "cannot inspect attached diff postimage: "
        + result.stderr.decode("utf-8", errors="replace").strip()
    )


def _git_apply_reverse(diff_bytes: bytes, root_fd: int) -> bool:
    try:
        result = subprocess.run(
            _git_argv("apply", "-R"),
            input=diff_bytes,
            cwd=_descriptor_git_cwd(root_fd),
            pass_fds=(root_fd,),
            env=_sanitized_git_env(),
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _git_apply_forward(diff_bytes: bytes, root_fd: int) -> bool:
    """Compensate a completed reverse apply on its held checkout."""

    try:
        result = subprocess.run(
            _git_argv("apply"),
            input=diff_bytes,
            cwd=_descriptor_git_cwd(root_fd),
            pass_fds=(root_fd,),
            env=_sanitized_git_env(),
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _head(cwd: Path, root_fd: int | None = None) -> str:
    if root_fd is not None:
        _assert_checkout_root_binding(cwd, root_fd)
    command_cwd = (
        cwd if root_fd is None else Path(_descriptor_git_cwd(root_fd))
    )
    result = subprocess.run(
        _git_argv("rev-parse", "--verify", "HEAD"),
        cwd=command_cwd,
        pass_fds=(() if root_fd is None else (root_fd,)),
        env=_sanitized_git_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise MutationJournalError(
            "cannot resolve checkout HEAD for legacy reaper authority: "
            + result.stderr.strip()
        )
    if root_fd is not None:
        _assert_checkout_root_binding(cwd, root_fd)
    return result.stdout.strip()


def _paths_overlap(left: str, right: str) -> bool:
    return (
        left == right
        or left.startswith(right.rstrip("/") + "/")
        or right.startswith(left.rstrip("/") + "/")
    )


def _overlaps_any(paths: Iterable[str], roots: Iterable[str]) -> bool:
    roots_tuple = tuple(roots)
    return any(_paths_overlap(path, root) for path in paths for root in roots_tuple)


def _covered_by(path: str, roots: Iterable[str]) -> bool:
    """Whether one protected root covers the complete candidate path."""

    return any(
        path == root or path.startswith(root.rstrip("/") + "/")
        for root in roots
    )


def _overlapping_authorities(
    paths: Iterable[str],
    authorities: Iterable[_PathAuthority],
) -> tuple[_PathAuthority, ...]:
    paths_tuple = tuple(paths)
    return tuple(
        authority
        for authority in authorities
        if any(_paths_overlap(path, authority.path) for path in paths_tuple)
    )


def _authority_is_newer(
    authority: _PathAuthority,
    implementation_index: int,
) -> bool:
    candidate_chronology = (
        implementation_index + 1,
        0,
        implementation_index,
        0,
    )
    return authority.chronology > candidate_chronology


def _authority_covers_candidate_postimage(
    authority: _PathAuthority,
    *,
    path: str,
    postimage: tuple[str, str, str] | None,
    implementation_index: int,
) -> bool:
    """Prove that one positive authority disposes of a candidate path.

    A later positive occurrence supersedes an earlier candidate over every
    path it covers.  An older occurrence is also sufficient for the same
    literal path when both immutable diff projections name the exact same Git
    tree entry.  ``_proven_positive_authorities`` has already proved that
    entry is live, so an intervening cumulative failure cannot have left a
    distinct Git-visible postimage behind.  Prefix coverage is deliberately
    limited to the chronological case: a parent entry cannot attest an older
    descendant's independent object identity.
    """

    if not _covered_by(path, (authority.path,)):
        return False
    if _authority_is_newer(authority, implementation_index):
        return True
    return authority.path == path and authority.postimage == postimage


@dataclasses.dataclass(frozen=True)
class _MissingAncestorBinding:
    """Held identity of the exact ancestor frontier that was absent."""

    parent_fd: int
    existing_components: tuple[str, ...]
    missing_component: str


@contextmanager
def _open_bound_entry_parent_or_missing(
    root_fd: int,
    path: str,
) -> Iterator[tuple[int, str] | _MissingAncestorBinding]:
    """Hold either the final parent or one exact missing-ancestor frontier."""

    components = tuple(path.split("/"))
    if not components or any(
        component in {"", ".", ".."} for component in components
    ):
        raise MutationJournalError(
            f"cannot safely inspect non-normalized reaper path {path!r}"
        )
    try:
        descriptor = os.dup(root_fd)
    except OSError as exc:
        raise MutationJournalError(
            f"cannot duplicate held reaper checkout descriptor: {exc}"
        ) from exc
    try:
        for index, component in enumerate(components[:-1]):
            try:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                yield _MissingAncestorBinding(
                    parent_fd=descriptor,
                    existing_components=components[:index],
                    missing_component=component,
                )
                return
            except (OSError, TypeError, ValueError) as exc:
                raise MutationJournalError(
                    f"cannot safely open parent of reaper path {path!r}: "
                    f"{exc}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        yield descriptor, components[-1]
    finally:
        os.close(descriptor)


def _assert_missing_ancestor_binding(
    root_fd: int,
    path: str,
    binding: _MissingAncestorBinding,
) -> None:
    """Reprove one held deepest-parent identity and its missing component."""

    try:
        held = os.fstat(binding.parent_fd)
    except OSError as exc:
        raise MutationJournalError(
            f"cannot inspect missing-ancestor parent for {path!r}: {exc}"
        ) from exc
    if binding.existing_components:
        probe = "/".join((*binding.existing_components, ".reaper-probe"))
        with _open_workspace_parent_if_present(root_fd, probe) as current:
            if current is None:
                raise MutationJournalError(
                    f"existing ancestor of reaper path {path!r} disappeared"
                )
            current_fd, _name = current
            current_metadata = os.fstat(current_fd)
    else:
        current_metadata = os.fstat(root_fd)
    if (held.st_dev, held.st_ino) != (
        current_metadata.st_dev,
        current_metadata.st_ino,
    ):
        raise MutationJournalError(
            f"missing-ancestor parent of reaper path {path!r} changed identity"
        )
    try:
        os.stat(
            binding.missing_component,
            dir_fd=binding.parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot reprove missing ancestor of reaper path {path!r}: {exc}"
        ) from exc
    raise MutationJournalError(
        f"missing ancestor of reaper path {path!r} appeared during audit"
    )


def _bound_entry_metadata(
    cwd: Path,
    root_fd: int,
    path: str,
) -> os.stat_result | None:
    """Inspect one stable root-relative entry without following aliases."""

    _assert_checkout_root_binding(cwd, root_fd)
    with _open_bound_entry_parent_or_missing(root_fd, path) as binding:
        if isinstance(binding, _MissingAncestorBinding):
            # Both probes must bind the same deepest-existing parent and
            # missing component; two unqualified ``None`` results can describe
            # different topology after a swap.
            _assert_missing_ancestor_binding(root_fd, path, binding)
            with _open_bound_entry_parent_or_missing(root_fd, path) as retry:
                if not isinstance(retry, _MissingAncestorBinding):
                    raise MutationJournalError(
                        f"missing frontier of reaper path {path!r} changed "
                        "during audit"
                    )
                first_parent = os.fstat(binding.parent_fd)
                retry_parent = os.fstat(retry.parent_fd)
                if (
                    retry.existing_components != binding.existing_components
                    or retry.missing_component != binding.missing_component
                    or (retry_parent.st_dev, retry_parent.st_ino)
                    != (first_parent.st_dev, first_parent.st_ino)
                ):
                    raise MutationJournalError(
                        f"missing frontier of reaper path {path!r} changed "
                        "during audit"
                    )
                _assert_missing_ancestor_binding(root_fd, path, retry)
            _assert_checkout_root_binding(cwd, root_fd)
            return None
        parent_fd, name = binding
        try:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            _assert_workspace_parent_binding(root_fd, path, parent_fd)
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                # The pathname parent can be replaced after the first parent
                # reproof while the held descriptor continues to report
                # absence.  Re-bind it after the second ENOENT before treating
                # that absence as authority for the current checkout path.
                _assert_workspace_parent_binding(root_fd, path, parent_fd)
                _assert_checkout_root_binding(cwd, root_fd)
                return None
            raise MutationJournalError(
                f"reaper path {path!r} appeared during absence audit"
            ) from None
        _assert_workspace_parent_binding(root_fd, path, parent_fd)
        after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
        ):
            raise MutationJournalError(
                f"reaper path {path!r} changed during audit"
            )
        # Both entry observations used the held parent. Re-bind that parent
        # once more afterward so a pathname swap between the first parent
        # proof and the second stat cannot make stable parked bytes stand in
        # for the current checkout path.
        _assert_workspace_parent_binding(root_fd, path, parent_fd)
    _assert_checkout_root_binding(cwd, root_fd)
    return after


def _normalize_bound_workspace_path(
    cwd: Path,
    root_fd: int,
    rendered: object,
) -> str:
    """Normalize a machine path against the held checkout descriptor."""

    if not isinstance(rendered, str) or "\0" in rendered:
        raise MutationJournalError(
            f"reaper workspace path is not one safe string: {rendered!r}"
        )
    normalized = rendered.replace("\\", "/") if os.name == "nt" else rendered
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise MutationJournalError(
            f"reaper workspace path escapes checkout: {rendered!r}"
        )
    text = candidate.as_posix()
    if (
        text in {"", "."}
        or any(
            part.casefold() in _VCS_METADATA_COMPONENTS
            for part in candidate.parts
        )
    ):
        raise MutationJournalError(
            f"reaper workspace path targets metadata: {rendered!r}"
        )
    _assert_checkout_root_binding(cwd, root_fd)
    # Opening the parent chain with O_NOFOLLOW proves that an existing parent
    # cannot redirect authority outside this checkout. Missing parents are
    # safe for a not-yet-created path and remain revalidated before mutation.
    with _open_workspace_parent_if_present(root_fd, text):
        pass
    _assert_checkout_root_binding(cwd, root_fd)
    return text


def _normalize_bound_scope_path(
    cwd: Path,
    root_fd: int,
    rendered: object,
) -> str:
    """Apply legacy redundant-root handling using descriptor-bound reads."""

    if not isinstance(rendered, str) or "\0" in rendered:
        raise MutationJournalError(
            f"reaper directive path is not one safe string: {rendered!r}"
        )
    normalized = rendered.replace("\\", "/") if os.name == "nt" else rendered
    candidate = PurePosixPath(normalized)
    parts = candidate.parts
    if parts and parts[0] == cwd.name and len(parts) > 1:
        unstripped = candidate.as_posix()
        full_metadata = _bound_entry_metadata(cwd, root_fd, unstripped)
        parent = PurePosixPath(unstripped).parent.as_posix()
        parent_metadata = _bound_entry_metadata(cwd, root_fd, parent)
        parent_is_directory = (
            parent_metadata is not None
            and stat.S_ISDIR(parent_metadata.st_mode)
            and not stat.S_ISLNK(parent_metadata.st_mode)
        )
        if full_metadata is None and not parent_is_directory:
            candidate = PurePosixPath(*parts[1:])
    return _normalize_bound_workspace_path(
        cwd,
        root_fd,
        candidate.as_posix(),
    )


def _recorded_paths(
    cwd: Path,
    root_fd: int,
    impl: Implementation,
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        _normalize_bound_workspace_path(cwd, root_fd, rendered)
        for rendered in (
            *(getattr(impl, "files_modified", ()) or ()),
            *(getattr(impl, "scope_escapes", ()) or ()),
        )
    ))


def _diff_paths(
    diff_bytes: bytes,
    cwd: Path,
    root_fd: int,
    *,
    base_commit: str,
) -> tuple[str, ...]:
    """Parse one diff through Git while bound to the held checkout inode."""

    if not diff_bytes.strip():
        return ()
    _assert_checkout_root_binding(cwd, root_fd)
    object_result = subprocess.run(
        _git_argv(
            "rev-parse", "--path-format=absolute", "--git-path", "objects"
        ),
        cwd=_descriptor_git_cwd(root_fd),
        pass_fds=(root_fd,),
        env=_sanitized_git_env(
            GIT_NO_LAZY_FETCH="1",
            GIT_NO_REPLACE_OBJECTS="1",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    _assert_checkout_root_binding(cwd, root_fd)
    if object_result.returncode != 0 or not object_result.stdout.strip():
        raise MutationJournalError(
            "cannot resolve Git objects for attached diff parsing: "
            + object_result.stderr.strip()
        )
    source_objects = object_result.stdout.strip()
    with tempfile.TemporaryDirectory(prefix="crucible-reaper-diff-index-") as tmp:
        temp_root = Path(tmp)
        temp_objects = temp_root / "objects"
        temp_objects.mkdir()
        env = _sanitized_git_env(
            GIT_INDEX_FILE=str(temp_root / "index"),
            GIT_OBJECT_DIRECTORY=str(temp_objects),
            GIT_ALTERNATE_OBJECT_DIRECTORIES=source_objects,
            GIT_LITERAL_PATHSPECS="1",
            GIT_NO_LAZY_FETCH="1",
            GIT_NO_REPLACE_OBJECTS="1",
        )

        def _run(*args: str, input_bytes: bytes | None = None):
            _assert_checkout_root_binding(cwd, root_fd)
            result = subprocess.run(
                _git_argv(*args),
                cwd=_descriptor_git_cwd(root_fd),
                pass_fds=(root_fd,),
                env=env,
                input=input_bytes,
                capture_output=True,
                check=False,
            )
            _assert_checkout_root_binding(cwd, root_fd)
            return result

        read_tree = _run(
            "-c",
            "core.splitIndex=false",
            "read-tree",
            "--",
            base_commit,
        )
        if read_tree.returncode != 0:
            raise MutationJournalError(
                "cannot seed attached diff parser: "
                + read_tree.stderr.decode("utf-8", errors="replace").strip()
            )
        applied = _run(
            "apply",
            "--cached",
            "--binary",
            "--whitespace=nowarn",
            "-",
            input_bytes=diff_bytes,
        )
        if applied.returncode != 0:
            raise MutationJournalError(
                "attached diff is malformed or does not apply to its base: "
                + applied.stderr.decode("utf-8", errors="replace").strip()
            )
        listed = _run(
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            "--name-status",
            "-z",
            "--find-renames",
            base_commit,
            "--",
        )
        if listed.returncode != 0:
            raise MutationJournalError(
                "cannot enumerate attached diff paths: "
                + listed.stderr.decode("utf-8", errors="replace").strip()
            )

    records = listed.stdout.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(records):
        status_raw = records[index]
        index += 1
        if not status_raw:
            continue
        status = os.fsdecode(status_raw)
        path_count = 2 if status[:1] in {"R", "C"} else 1
        if index + path_count > len(records):
            raise MutationJournalError(
                "Git returned malformed attached diff path output"
            )
        for raw_path in records[index:index + path_count]:
            if not raw_path:
                raise MutationJournalError(
                    "Git returned an empty attached diff pathname"
                )
            paths.append(_normalize_bound_workspace_path(
                cwd,
                root_fd,
                os.fsdecode(raw_path),
            ))
        index += path_count
    _assert_checkout_root_binding(cwd, root_fd)
    return tuple(dict.fromkeys(paths))


def _diff_postimage_matches_base_tree(
    diff_bytes: bytes,
    cwd: Path,
    root_fd: int,
    *,
    source_base: str,
    target_base: str,
    mutation_paths: Iterable[str],
) -> bool:
    """Prove an old diff's exact path postimages are in ``target_base``.

    ``HEAD == target_base`` does not prove that a historical dirty-worktree
    landing was committed: Git can advance an unrelated path while leaving
    those bytes dirty.  Apply the accepted diff to an isolated index seeded
    from its immutable source base, then compare only its actual touched paths
    with the target tree.  The live index, worktree, and object database are
    never write targets.
    """

    paths = tuple(dict.fromkeys(mutation_paths))
    if not paths:
        return True
    _assert_checkout_root_binding(cwd, root_fd)
    object_result = subprocess.run(
        _git_argv(
            "rev-parse", "--path-format=absolute", "--git-path", "objects"
        ),
        cwd=_descriptor_git_cwd(root_fd),
        pass_fds=(root_fd,),
        env=_sanitized_git_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    _assert_checkout_root_binding(cwd, root_fd)
    if object_result.returncode != 0 or not object_result.stdout.strip():
        raise ValueError(
            "cannot resolve Git object directory for landed-postimage proof: "
            + object_result.stderr.strip()
        )
    source_objects = object_result.stdout.strip()
    with tempfile.TemporaryDirectory(prefix="crucible-landed-index-") as tmp:
        temp_root = Path(tmp)
        temp_objects = temp_root / "objects"
        temp_objects.mkdir()
        env = _sanitized_git_env(
            GIT_INDEX_FILE=str(temp_root / "index"),
            GIT_OBJECT_DIRECTORY=str(temp_objects),
            GIT_ALTERNATE_OBJECT_DIRECTORIES=source_objects,
            GIT_LITERAL_PATHSPECS="1",
            GIT_NO_LAZY_FETCH="1",
            GIT_NO_REPLACE_OBJECTS="1",
        )

        def _run(*args: str, input_bytes: bytes | None = None):
            _assert_checkout_root_binding(cwd, root_fd)
            result = subprocess.run(
                _git_argv(*args),
                cwd=_descriptor_git_cwd(root_fd),
                pass_fds=(root_fd,),
                env=env,
                input=input_bytes,
                capture_output=True,
                check=False,
            )
            _assert_checkout_root_binding(cwd, root_fd)
            return result

        read_tree = _run("read-tree", "--", source_base)
        if read_tree.returncode != 0:
            raise ValueError(
                "cannot seed isolated landed-postimage index: "
                + read_tree.stderr.decode("utf-8", errors="replace").strip()
            )
        applied = _run(
            "apply",
            "--cached",
            "--binary",
            "--whitespace=nowarn",
            "-",
            input_bytes=diff_bytes,
        )
        if applied.returncode != 0:
            raise ValueError(
                "cannot apply accepted diff to isolated landed-postimage "
                "index: "
                + applied.stderr.decode("utf-8", errors="replace").strip()
            )
        # Bound argv size while retaining literal path semantics. ``--quiet``
        # compares tree entries (mode, object identity, and existence) and
        # never consults or mutates the live worktree/index.
        for offset in range(0, len(paths), 128):
            compared = _run(
                "diff",
                "--cached",
                "--quiet",
                "--no-ext-diff",
                "--no-textconv",
                target_base,
                "--",
                *paths[offset:offset + 128],
            )
            if compared.returncode == 1:
                return False
            if compared.returncode != 0:
                raise ValueError(
                    "cannot compare isolated landed postimage with current "
                    "base tree: "
                    + compared.stderr.decode(
                        "utf-8", errors="replace"
                    ).strip()
                )
    _assert_checkout_root_binding(cwd, root_fd)
    return True


def _worktree_paths_match_base_tree(
    cwd: Path,
    root_fd: int | None = None,
    *,
    base_commit: str,
    paths: Iterable[str],
) -> bool:
    """Prove exact live path bytes/types/modes equal an immutable base.

    Whole-patch reverse applicability cannot establish absence: one path from
    a multi-file failed patch may remain dirty after another is restored. Use
    rollback's descriptor-bound verifier for each uncovered literal path. It
    authenticates regular bytes and symlink payloads against the selected
    full object ID without trusting the live index, status flags, replacement
    objects, or lazy fetching.
    """

    normalized_paths = tuple(dict.fromkeys(paths))
    if not normalized_paths:
        return True
    owns_descriptor = root_fd is None
    if root_fd is None:
        bound_cwd, root_fd = _open_checkout_root(cwd)
    else:
        bound_cwd = cwd
    try:
        if owns_descriptor and bound_cwd != cwd.resolve():
            raise MutationJournalError(
                "descriptor-bound checkout differs from reaper workspace"
            )
        _assert_checkout_root_binding(bound_cwd, root_fd)
        for path in normalized_paths:
            _assert_checkout_root_binding(bound_cwd, root_fd)
            tree_entry = _git_tree_entry(root_fd, base_commit, path)
            _assert_checkout_root_binding(bound_cwd, root_fd)
            try:
                _verify_worktree_path_matches_tree(
                    root_fd,
                    path,
                    tree_entry,
                    tree_label=f"base {base_commit}",
                )
            except MutationJournalError:
                return False
            _assert_checkout_root_binding(bound_cwd, root_fd)
        _assert_checkout_root_binding(bound_cwd, root_fd)
    finally:
        if owns_descriptor:
            os.close(root_fd)
    return True


def _accepted_postimages_are_in_base(
    artifacts: _ReaperArtifactReader,
    cwd: Path,
    root_fd: int,
    items: Iterable[_AttachedImplementation],
    *,
    current_base: str,
) -> bool:
    """Prove the lifecycle's final per-path postimage is immutable.

    Slice diffs are cumulative from their common base. When accepted slices
    share a path, applying them sequentially is invalid because a later patch
    still expects the base preimage. The caller supplies global lifecycle/ref
    chronology; compare each occurrence only on paths not superseded by a
    later accepted occurrence, while parsing every diff against its own
    immutable base.
    """

    snapshots: list[tuple[_AttachedImplementation, bytes, tuple[str, ...]]] = []
    for item in items:
        implementation = item.implementation
        if implementation.scope_escapes:
            return False
        diff_bytes = artifacts.read(
            implementation.diff_ref,
            max_bytes=_MAX_REAPER_DIFF_BYTES,
            label="Implementation diff",
        )
        diff_paths = _diff_paths(
            diff_bytes,
            cwd,
            root_fd,
            base_commit=implementation.base_commit,
        )
        if bool(diff_bytes.strip()) != bool(diff_paths):
            return False
        recorded = tuple(
            _normalize_bound_workspace_path(cwd, root_fd, path)
            for path in (implementation.files_modified or ())
        )
        if not set(recorded).issubset(diff_paths):
            return False
        snapshots.append((item, diff_bytes, diff_paths))

    later_paths: list[str] = []
    for item, diff_bytes, diff_paths in reversed(snapshots):
        final_paths = tuple(
            path
            for path in diff_paths
            if not _overlaps_any((path,), later_paths)
        )
        if not _diff_postimage_matches_base_tree(
            diff_bytes,
            cwd,
            root_fd,
            source_base=item.implementation.base_commit,
            target_base=current_base,
            mutation_paths=final_paths,
        ):
            return False
        later_paths.extend(diff_paths)
    return True


def _positive_postimage_entries(
    cwd: Path,
    root_fd: int,
    item: _AttachedImplementation,
    diff_bytes: bytes,
    paths: tuple[str, ...],
) -> dict[str, tuple[str, str, str] | None]:
    """Project one accepted diff into an isolated index at its exact base."""

    if not paths:
        return {}
    object_result = subprocess.run(
        _git_argv(
            "rev-parse", "--path-format=absolute", "--git-path", "objects"
        ),
        cwd=_descriptor_git_cwd(root_fd),
        pass_fds=(root_fd,),
        env=_sanitized_git_env(
            GIT_NO_LAZY_FETCH="1",
            GIT_NO_REPLACE_OBJECTS="1",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if object_result.returncode != 0 or not object_result.stdout.strip():
        raise MutationJournalError(
            "cannot resolve Git objects for positive postimage proof: "
            + object_result.stderr.strip()
        )
    source_objects = object_result.stdout.strip()
    with tempfile.TemporaryDirectory(prefix="crucible-positive-index-") as tmp:
        temp_root = Path(tmp)
        temp_objects = temp_root / "objects"
        temp_objects.mkdir()
        env = _sanitized_git_env(
            GIT_INDEX_FILE=str(temp_root / "index"),
            GIT_OBJECT_DIRECTORY=str(temp_objects),
            GIT_ALTERNATE_OBJECT_DIRECTORIES=source_objects,
            GIT_LITERAL_PATHSPECS="1",
            GIT_NO_LAZY_FETCH="1",
            GIT_NO_REPLACE_OBJECTS="1",
        )

        def _run(*args: str, input_bytes: bytes | None = None):
            _assert_checkout_root_binding(cwd, root_fd)
            result = subprocess.run(
                _git_argv(*args),
                cwd=_descriptor_git_cwd(root_fd),
                pass_fds=(root_fd,),
                env=env,
                input=input_bytes,
                capture_output=True,
                check=False,
            )
            _assert_checkout_root_binding(cwd, root_fd)
            return result

        read_tree = _run(
            "-c",
            "core.splitIndex=false",
            "read-tree",
            "--",
            item.implementation.base_commit,
        )
        if read_tree.returncode != 0:
            raise MutationJournalError(
                "cannot seed positive postimage index: "
                + read_tree.stderr.decode("utf-8", errors="replace").strip()
            )
        applied = _run(
            "apply",
            "--cached",
            "--binary",
            "--whitespace=nowarn",
            "-",
            input_bytes=diff_bytes,
        )
        if applied.returncode != 0:
            raise MutationJournalError(
                "cannot project positive diff into isolated index: "
                + applied.stderr.decode("utf-8", errors="replace").strip()
            )
        entries: dict[str, tuple[str, str, str]] = {}
        for offset in range(0, len(paths), 128):
            result = _run(
                "ls-files",
                "--stage",
                "-z",
                "--",
                *paths[offset:offset + 128],
            )
            if result.returncode != 0:
                raise MutationJournalError(
                    "cannot inspect projected positive index: "
                    + result.stderr.decode(
                        "utf-8", errors="replace"
                    ).strip()
                )
            for record in result.stdout.split(b"\0"):
                if not record:
                    continue
                metadata, separator, encoded_path = record.partition(b"\t")
                fields = metadata.split()
                if not separator or len(fields) != 3:
                    raise MutationJournalError(
                        "projected positive index returned malformed metadata"
                    )
                mode, oid, stage = fields
                rendered = os.fsdecode(encoded_path)
                if (
                    rendered not in paths
                    or stage != b"0"
                    or not mode.isdigit()
                    or len(oid) not in {40, 64}
                    or any(byte not in b"0123456789abcdef" for byte in oid)
                    or rendered in entries
                ):
                    raise MutationJournalError(
                        "projected positive index entry is ambiguous or "
                        "malformed"
                    )
                rendered_mode = mode.decode("ascii")
                entries[rendered] = (
                    rendered_mode,
                    "commit" if rendered_mode == "160000" else "blob",
                    oid.decode("ascii"),
                )
        return {path: entries.get(path) for path in paths}


def _proven_positive_authorities(
    artifacts: _ReaperArtifactReader,
    cwd: Path,
    root_fd: int,
    occurrences: Iterable[_PositiveOccurrence],
) -> tuple[_PathAuthority, ...]:
    """Unconditionally prove the final live postimage of positive refs."""

    ordered = tuple(sorted(
        occurrences,
        key=lambda occurrence: occurrence.chronology,
    ))
    snapshots: list[
        tuple[_PositiveOccurrence, bytes, tuple[str, ...]]
    ] = []
    seen_chronology: set[tuple[int, int, int, int]] = set()
    for occurrence in ordered:
        item = occurrence.item
        if occurrence.chronology in seen_chronology:
            raise ValueError(
                "positive Implementation chronology is ambiguous at "
                f"{occurrence.chronology!r}"
            )
        seen_chronology.add(occurrence.chronology)
        implementation = item.implementation
        if implementation.scope_escapes:
            raise ValueError(
                f"positive Implementation {item.ref} records scope escapes"
            )
        diff_bytes = artifacts.read(
            implementation.diff_ref,
            max_bytes=_MAX_REAPER_DIFF_BYTES,
            label="Implementation diff",
        )
        diff_paths = _diff_paths(
            diff_bytes,
            cwd,
            root_fd,
            base_commit=implementation.base_commit,
        )
        if bool(diff_bytes.strip()) != bool(diff_paths):
            raise ValueError(
                f"positive Implementation {item.ref} diff projection is "
                "incoherent"
            )
        recorded = tuple(
            _normalize_bound_workspace_path(cwd, root_fd, path)
            for path in (implementation.files_modified or ())
        )
        if not set(recorded).issubset(diff_paths):
            raise ValueError(
                f"positive Implementation {item.ref} files_modified is not "
                "a subset of its exact diff paths"
            )
        snapshots.append((occurrence, diff_bytes, diff_paths))

    selected: list[
        tuple[_PositiveOccurrence, bytes, tuple[str, ...]]
    ] = []
    later_paths: list[str] = []
    for occurrence, diff_bytes, diff_paths in reversed(snapshots):
        final_paths = tuple(
            path
            for path in diff_paths
            if not _overlaps_any((path,), later_paths)
        )
        selected.append((occurrence, diff_bytes, final_paths))
        later_paths.extend(diff_paths)

    authorities: list[_PathAuthority] = []
    _assert_checkout_root_binding(cwd, root_fd)
    for occurrence, diff_bytes, final_paths in reversed(selected):
        item = occurrence.item
        entries = _positive_postimage_entries(
            cwd,
            root_fd,
            item,
            diff_bytes,
            final_paths,
        )
        for path in final_paths:
            try:
                _verify_worktree_path_matches_tree(
                    root_fd,
                    path,
                    entries[path],
                    tree_label=f"positive Implementation {item.ref}",
                )
            except MutationJournalError as exc:
                raise ValueError(
                    f"positive Implementation {item.ref} postimage for "
                    f"{path!r} is not live: {exc}"
                ) from exc
            _assert_checkout_root_binding(cwd, root_fd)
            authorities.append(_PathAuthority(
                path=path,
                chronology=occurrence.chronology,
                source="positive_postimage",
                implementation_ref=item.ref,
                postimage=entries[path],
            ))
    _assert_checkout_root_binding(cwd, root_fd)
    return tuple(authorities)


def _nonblocking_lifecycle_reason(reason: str) -> bool:
    """Recognized non-acceptance states that are not corrupt authority."""

    return (
        reason == "no tier lifecycle event is attached"
        or reason.startswith("latest tier lifecycle is not landed:")
        or reason == "landed lifecycle does not bind the latest slice records"
        or reason == "landed lifecycle belongs to different execution authority"
        or reason == "landed lifecycle Spec does not contain the complete tier"
    )


def _plan_slice_map(spec) -> dict[str, tuple[object, object]]:
    result: dict[str, tuple[object, object]] = {}
    tier_dag = getattr(spec, "tier_dag", None)
    if tier_dag is None:
        return result
    for tier in tier_dag.tiers:
        for slice_ in tier.slices:
            if slice_.id in result:
                raise ValueError(
                    f"current TierDAG repeats slice id {slice_.id!r}"
                )
            result[slice_.id] = (tier, slice_)
    return result


def _implementation_paths(
    artifacts: _ReaperArtifactReader,
    cwd: Path,
    root_fd: int,
    item: _AttachedImplementation,
    *,
    slice_=None,
) -> tuple[str, ...]:
    """Resolve every path named by one attached Implementation."""

    paths: list[str] = []
    if slice_ is not None:
        paths.extend(
            _normalize_bound_scope_path(cwd, root_fd, rendered)
            for rendered in (
                *slice_.directive.files_in_scope,
                *slice_.directive.creates_paths,
            )
        )
    diff_bytes = artifacts.read(
        item.implementation.diff_ref,
        max_bytes=_MAX_REAPER_DIFF_BYTES,
        label="Implementation diff",
    )
    paths.extend(_diff_paths(
        diff_bytes,
        cwd,
        root_fd,
        base_commit=item.implementation.base_commit,
    ))
    paths.extend(_recorded_paths(cwd, root_fd, item.implementation))
    return tuple(dict.fromkeys(paths))


def _tier_lifecycle_target(event: LifecycleEvent) -> str | None:
    execution = getattr(event, "execution", None)
    if getattr(execution, "recipe", None) != "tier":
        return None
    try:
        reason = mutation_lifecycle_reason_base(event.reason)
    except MutationJournalError:
        return None
    if not isinstance(reason, str) or not reason.startswith("tier "):
        return None
    tier_id, separator, status = reason[5:].rpartition(": ")
    if (
        not separator
        or event.kind != "landed"
        or status != "complete"
    ):
        return None
    return tier_id


def _historical_landed_authorities(
    corpus: "Corpus",
    artifacts: _ReaperArtifactReader,
    spec,
    attached: tuple[_AttachedImplementation, ...],
    lifecycle_records: Iterable[tuple[str, LifecycleEvent]],
    plan_slices: Mapping[str, tuple[object, object]],
    cwd: Path,
    root_fd: int,
) -> tuple[
    tuple[_PositiveOccurrence, ...],
    frozenset[str],
    frozenset[str],
]:
    """Protect every exact postimage accepted by a historical Tier landing.

    A later failed attempt for one sibling must not invalidate the accepted
    postimage of another sibling.  Each lifecycle is therefore resolved
    against its own affected Spec rather than against the current latest map.
    """

    by_ref = {item.ref: item for item in attached}
    current_structure = _partition.spec_execution_structure_hash(spec)
    authoritative_occurrences: list[_PositiveOccurrence] = []
    authoritative_refs: set[str] = set()
    absorbed_refs: set[str] = set()
    absorption_occurrences: list[
        tuple[int, _AttachedImplementation]
    ] = []
    tiers = {
        tier.tier_id: tier
        for tier in (getattr(getattr(spec, "tier_dag", None), "tiers", ()) or ())
    }
    current_lifecycle_refs = tuple(
        event_ref for event_ref, _event in lifecycle_records
    )
    for event_index, (event_ref, event) in enumerate(lifecycle_records):
        tier_id = _tier_lifecycle_target(event)
        if tier_id is None:
            continue
        if (
            len(event.affected_versions) != 1
            or isinstance(event.affected_versions[0], bool)
            or not isinstance(event.affected_versions[0], int)
            or event.affected_versions[0] < 0
            or event.affected_versions[0] > spec.version
        ):
            raise ValueError(
                f"landed lifecycle {event_ref} has invalid affected_versions"
            )
        try:
            accepted_spec = corpus.get_spec(event.affected_versions[0])
        except Exception as exc:  # noqa: BLE001 - protection is fail closed
            raise ValueError(
                f"landed lifecycle {event_ref} Spec is unavailable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        accepted_lifecycle_refs = tuple(
            str(ref) for ref in accepted_spec.lifecycle_event_refs
        )
        if (
            current_lifecycle_refs[:len(accepted_lifecycle_refs)]
            != accepted_lifecycle_refs
        ):
            raise ValueError(
                f"landed lifecycle {event_ref} event lineage is not an "
                "append-only prefix of current authority"
            )
        if event_ref not in accepted_lifecycle_refs:
            raise ValueError(
                f"landed lifecycle {event_ref} is not attached to its Spec"
            )
        accepted_tiers = {
            tier.tier_id: tier
            for tier in (
                getattr(
                    getattr(accepted_spec, "tier_dag", None),
                    "tiers",
                    (),
                )
                or ()
            )
        }
        accepted_tier = accepted_tiers.get(tier_id)
        if accepted_tier is None:
            raise ValueError(
                f"landed lifecycle {event_ref} names absent tier {tier_id!r}"
            )
        accepted_history: dict[str, _AttachedImplementation] = {}
        accepted_refs = tuple(
            str(ref) for ref in accepted_spec.implementation_refs
        )
        current_refs = tuple(item.ref for item in attached)
        if current_refs[:len(accepted_refs)] != accepted_refs:
            raise ValueError(
                f"landed lifecycle {event_ref} Implementation lineage is "
                "not an append-only prefix of current authority"
            )
        for ref in accepted_refs:
            item = by_ref.get(str(ref))
            if item is None:
                raise ValueError(
                    f"landed lifecycle {event_ref} binds Implementation "
                    f"{ref} absent from current lineage"
                )
            accepted_history[item.implementation.slice_id] = item
        latest = {
            slice_.id: accepted_history[slice_.id].implementation
            for slice_ in accepted_tier.slices
            if slice_.id in accepted_history
        }
        try:
            accepted, reason = _build_verify._tier_lifecycle_acceptance(
                corpus,
                accepted_spec,
                accepted_tier,
                latest,
                artifact_reader=artifacts.read,
            )
        except Exception as exc:  # noqa: BLE001 - ambiguous proof blocks writes
            raise ValueError(
                f"tier {tier_id}: lifecycle validation raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not accepted:
            # A later exact current-plan proof may supersede a stale landing,
            # including one written with a resume identity that the current
            # reader intentionally rejects. Do not treat that stale event as
            # positive postimage authority. The attached-record audit below
            # still requires every path to be covered by a newer live proof or
            # to match the immutable base, so this cannot hide surviving dirt.
            if _nonblocking_lifecycle_reason(reason):
                continue
            raise ValueError(f"tier {tier_id}: {reason}")

        accepted_structure = _partition.spec_execution_structure_hash(
            accepted_spec
        )
        if accepted_structure != current_structure:
            accepted_base = accepted_spec.project_context.base_commit
            current_base = spec.project_context.base_commit
            # HEAD equality alone cannot retire historical dirt: an unrelated
            # commit may advance HEAD while leaving the accepted postimage in
            # the worktree. Retire only exact accepted refs whose diff-derived
            # path postimages are independently present in the current base
            # tree. Otherwise leave them unproved so overlap checks fail
            # closed. A same-base plan/tool/runner rewrite is never equivalent.
            if (
                accepted_base != current_base
                and _head(cwd, root_fd) == current_base
            ):
                absorption_occurrences.extend(
                    (event_index, accepted_history[slice_.id])
                    for slice_ in accepted_tier.slices
                )
                continue
            if accepted_base == current_base:
                # A same-base plan revision makes this historical landing
                # stale; it does not make the old postimage current authority.
                # Do not fail here, however, because the operator may have
                # re-attested the exact live bytes with a newer current-plan
                # Implementation.  Leaving the historical refs out of
                # ``authoritative_refs`` makes the normal attached-record
                # audit below prove one of three safe dispositions for every
                # old path: a strictly newer retained authority covers it, an
                # independently live positive authority projects the exact
                # same literal-path postimage, or the worktree matches the old
                # record's immutable base. Any unprotected stale postimage
                # still fails closed before mutation or provider dispatch.
                continue
            raise ValueError(
                f"landed lifecycle {event_ref} belongs to different "
                "execution authority"
            )
        tier = tiers.get(tier_id)
        if tier is None:
            raise ValueError(
                f"landed lifecycle {event_ref} names absent tier {tier_id!r}"
            )
        accepted_items = tuple(sorted(
            (accepted_history[slice_.id] for slice_ in accepted_tier.slices),
            key=lambda item: item.index,
        ))
        for item in accepted_items:
            authoritative_refs.add(item.ref)
            authoritative_occurrences.append(_PositiveOccurrence(
                item=item,
                chronology=(
                    len(accepted_refs),
                    1,
                    event_index,
                    item.index,
                ),
            ))

    if absorption_occurrences:
        current_base = spec.project_context.base_commit
        # Lifecycle ref order is primary. Within one accepted event, Spec
        # Implementation ref order resolves overlapping cumulative records.
        ordered = tuple(
            item
            for _event_index, item in sorted(
                absorption_occurrences,
                key=lambda occurrence: (
                    occurrence[0],
                    occurrence[1].index,
                ),
            )
        )
        if _accepted_postimages_are_in_base(
            artifacts,
            cwd,
            root_fd,
            ordered,
            current_base=current_base,
        ):
            absorbed_refs.update(item.ref for item in ordered)
        # A base advance that did not absorb the historical landing is not,
        # by itself, proof of unresolved dirt.  Keep those refs active so the
        # normal attached-record audit below can require one of its exact safe
        # dispositions: a newer live positive authority covers every path, or
        # the worktree still matches the record's immutable base.  That audit
        # remains fail-closed when neither proof exists.  Raising here would
        # run before current-plan retained authorities are resolved and would
        # incorrectly reject an exact fresh recapture of the live postimage.
    return (
        tuple(authoritative_occurrences),
        frozenset(authoritative_refs),
        frozenset(absorbed_refs.difference(authoritative_refs)),
    )


def _retained_authorities(
    corpus: "Corpus",
    artifacts: _ReaperArtifactReader,
    spec,
    history: Mapping[str, list[_AttachedImplementation]],
    plan_slices: Mapping[str, tuple[object, object]],
    *,
    negative_refs: frozenset[str],
    tool_authority_digests: Mapping[str, str | None],
) -> tuple[tuple[_PositiveOccurrence, ...], frozenset[str]]:
    """Protect current-plan provisional successes that still survive."""

    authoritative_items: list[_PositiveOccurrence] = []
    authoritative_refs: set[str] = set()
    for slice_id, (tier, slice_) in plan_slices.items():
        item = next(
            (
                candidate
                for candidate in reversed(history.get(slice_id, ()))
                if candidate.ref not in negative_refs
            ),
            None,
        )
        if item is None:
            continue
        impl = item.implementation
        if (
            impl.scope_escapes
            or not _partition._is_skip_worthy(impl.classification, slice_)
        ):
            continue
        if not _partition._identity_matches(
            corpus,
            spec,
            tier,
            slice_,
            impl,
            tool_authority_digest=tool_authority_digests.get(slice_id),
            artifact_reader=artifacts.read,
        ):
            continue
        authoritative_refs.add(item.ref)
        authoritative_items.append(_PositiveOccurrence(
            item=item,
            chronology=(item.index + 1, 0, item.index, 0),
        ))
    return tuple(authoritative_items), frozenset(authoritative_refs)


def _reaped_authority(
    corpus: "Corpus",
    artifacts: _ReaperArtifactReader,
    spec,
    lifecycle_records: Iterable[tuple[str, LifecycleEvent]],
    cwd: Path,
    root_fd: int,
) -> tuple[tuple[str, ...], dict[str, frozenset[str]]]:
    """Resolve exact legacy-cleanup settlements attached to Spec lineage."""

    result: list[str] = []
    accepted_lifecycle_transactions: dict[str, frozenset[str]] = {}
    claimed_transactions: dict[str, str] = {}
    for event_ref, event in lifecycle_records:
        exact_operations: dict[str, dict[str, object]] | None = None
        if " mutation-settlement=" in event.reason:
            try:
                _settlement_ref, _payload, operations = (
                    _validate_mutation_lifecycle_event(
                        corpus,
                        spec,
                        event_ref,
                        event,
                    )
                )
            except MutationJournalError as exc:
                raise ValueError(
                    f"lifecycle {event_ref} has invalid exact mutation "
                    f"settlement ({exc})"
                ) from exc
            for transaction_id in operations:
                previous = claimed_transactions.get(transaction_id)
                if previous is not None and previous != event_ref:
                    raise ValueError(
                        "mutation transaction is claimed by multiple "
                        f"lifecycle events: {transaction_id}"
                    )
                claimed_transactions[transaction_id] = event_ref
            accepted_lifecycle_transactions[event_ref] = frozenset(operations)
            exact_operations = operations
            # Non-reaper events have no legacy cleanup semantics to resolve.
            if event.kind != _REAPER_EVENT_KIND:
                continue
        if event.kind != _REAPER_EVENT_KIND:
            continue
        execution = getattr(event, "execution", None)
        if getattr(execution, "recipe", None) != "reaper":
            raise ValueError(
                f"reaped lifecycle {event_ref} has non-reaper execution authority"
            )
        try:
            semantic_reason = mutation_lifecycle_reason_base(event.reason)
        except MutationJournalError as exc:
            raise ValueError(
                f"reaped lifecycle {event_ref} has malformed reason"
            ) from exc
        if not semantic_reason.startswith(_REAPER_EVENT_PREFIX):
            raise ValueError(f"reaped lifecycle {event_ref} has malformed reason")
        settlement_ref = semantic_reason[len(_REAPER_EVENT_PREFIX):]
        try:
            payload = json.loads(artifacts.read(
                ContentHash(settlement_ref),
                max_bytes=_MAX_REAPER_SETTLEMENT_BYTES,
                label="reaper settlement",
            ).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - settlement ambiguity blocks writes
            raise ValueError(
                f"reaped lifecycle {event_ref} settlement is unreadable "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"reaped lifecycle {event_ref} settlement is malformed")
        implementation_refs = payload.get("implementation_refs")
        transaction_id = payload.get("transaction_id")
        reaped_slices = payload.get("crashed_rollbacks_reaped")
        cleaned_escapes = payload.get("scope_escapes_cleaned")
        mutation_paths = payload.get("mutation_paths")
        if (
            payload.get("schema") != _REAPER_SETTLEMENT_SCHEMA
            or not isinstance(transaction_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", transaction_id) is None
            or not isinstance(implementation_refs, list)
            or not implementation_refs
            or any(not isinstance(ref, str) or not ref for ref in implementation_refs)
            or len(set(implementation_refs)) != len(implementation_refs)
            or not isinstance(reaped_slices, list)
            or any(not isinstance(value, str) or not value for value in reaped_slices)
            or len(set(reaped_slices)) != len(reaped_slices)
            or not isinstance(cleaned_escapes, list)
            or any(not isinstance(value, str) or not value for value in cleaned_escapes)
            or len(set(cleaned_escapes)) != len(cleaned_escapes)
            or not isinstance(mutation_paths, list)
            or any(not isinstance(value, str) or not value for value in mutation_paths)
            or len(set(mutation_paths)) != len(mutation_paths)
        ):
            raise ValueError(f"reaped lifecycle {event_ref} settlement is malformed")
        try:
            normalized_mutation_paths = tuple(
                _normalize_bound_workspace_path(cwd, root_fd, path)
                for path in mutation_paths
            )
        except Exception as exc:  # pragma: no cover - defensive shape guard
            raise ValueError(
                f"reaped lifecycle {event_ref} settlement paths are malformed"
            ) from exc
        if tuple(mutation_paths) != normalized_mutation_paths:
            raise ValueError(f"reaped lifecycle {event_ref} settlement is malformed")
        if (
            len(event.affected_versions) != 1
            or isinstance(event.affected_versions[0], bool)
            or not isinstance(event.affected_versions[0], int)
            or event.affected_versions[0] < 0
            or event.affected_versions[0] > spec.version
        ):
            raise ValueError(
                f"reaped lifecycle {event_ref} has invalid affected_versions"
            )
        try:
            accepted_spec = corpus.get_spec(event.affected_versions[0])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"reaped lifecycle {event_ref} Spec is unavailable: "
                f"{type(exc).__name__}"
            ) from exc
        valid_lineage, lineage_reason = (
            _build_verify._validate_spec_lineage(corpus, accepted_spec)
        )
        if not valid_lineage:
            raise ValueError(
                f"reaped lifecycle {event_ref} Spec lineage is invalid: "
                f"{lineage_reason}"
            )
        if not _build_verify._spec_refs_are_prefix(accepted_spec, spec):
            raise ValueError(
                f"reaped lifecycle {event_ref} Spec refs are not a current "
                "prefix"
            )
        accepted_lifecycle_refs = tuple(
            str(ref) for ref in accepted_spec.lifecycle_event_refs
        )
        if event_ref not in accepted_lifecycle_refs:
            raise ValueError(
                f"reaped lifecycle {event_ref} is not attached to its Spec"
            )
        if event.affected_versions[0] == 0:
            previous_lifecycle_refs: tuple[str, ...] = ()
        else:
            previous_spec = corpus.get_spec(event.affected_versions[0] - 1)
            previous_lifecycle_refs = tuple(
                str(ref) for ref in previous_spec.lifecycle_event_refs
            )
        if event_ref not in accepted_lifecycle_refs[
            len(previous_lifecycle_refs):
        ]:
            raise ValueError(
                f"reaped lifecycle {event_ref} was not first attached at its "
                "affected Spec"
            )
        accepted_impls = {str(ref) for ref in accepted_spec.implementation_refs}
        if not set(implementation_refs).issubset(accepted_impls):
            raise ValueError(
                f"reaped lifecycle {event_ref} names unattached Implementations"
            )
        referenced: dict[str, Implementation] = {}
        try:
            for ref in implementation_refs:
                referenced[ref] = corpus.get_record(
                    ref,
                    as_type=Implementation,
                )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"reaped lifecycle {event_ref} names unreadable "
                f"Implementation evidence ({type(exc).__name__}: {exc})"
            ) from exc
        by_slice: dict[str, list[tuple[str, Implementation]]] = {}
        for ref, implementation in referenced.items():
            by_slice.setdefault(implementation.slice_id, []).append(
                (ref, implementation)
            )
        if any(len(items) != 1 for items in by_slice.values()):
            raise ValueError(
                f"reaped lifecycle {event_ref} has ambiguous slice evidence"
            )

        operation_refs: set[str] = set()
        expected_paths: list[str] = []
        for slice_id in reaped_slices:
            items = by_slice.get(slice_id, ())
            if len(items) != 1 or not _is_failure_class(
                items[0][1].classification
            ):
                raise ValueError(
                    f"reaped lifecycle {event_ref} has invalid reaped slice"
                )
            ref, implementation = items[0]
            diff_bytes = artifacts.read(
                implementation.diff_ref,
                max_bytes=_MAX_REAPER_DIFF_BYTES,
                label="Implementation diff",
            )
            diff_paths = _diff_paths(
                diff_bytes,
                cwd,
                root_fd,
                base_commit=implementation.base_commit,
            )
            if not diff_bytes.strip() or not diff_paths:
                raise ValueError(
                    f"reaped lifecycle {event_ref} has empty reaped diff"
                )
            operation_refs.add(ref)
            expected_paths.extend(diff_paths)

        for path in cleaned_escapes:
            normalized = _normalize_bound_workspace_path(cwd, root_fd, path)
            owners = [
                ref
                for ref, implementation in referenced.items()
                if normalized in {
                    _normalize_bound_workspace_path(cwd, root_fd, rendered)
                    for rendered in (implementation.scope_escapes or ())
                }
            ]
            if len(owners) != 1:
                raise ValueError(
                    f"reaped lifecycle {event_ref} has unbound cleaned path"
                )
            operation_refs.add(owners[0])
            expected_paths.append(normalized)

        if (
            operation_refs != set(implementation_refs)
            or not expected_paths
            or set(expected_paths) != set(mutation_paths)
            or len(set(expected_paths)) != len(mutation_paths)
        ):
            raise ValueError(
                f"reaped lifecycle {event_ref} settlement operations do not "
                "bind its exact Implementation refs and paths"
            )
        result.extend(implementation_refs)
        previous = claimed_transactions.get(transaction_id)
        if previous is not None and previous != event_ref:
            raise ValueError(
                "mutation transaction is claimed by multiple lifecycle events: "
                f"{transaction_id}"
            )
        claimed_transactions[transaction_id] = event_ref
        if exact_operations is not None:
            if transaction_id not in exact_operations:
                raise ValueError(
                    f"reaped lifecycle {event_ref} legacy and exact "
                    "transaction authority disagree"
                )
        else:
            accepted_lifecycle_transactions[event_ref] = frozenset(
                (transaction_id,)
            )
    return (
        tuple(dict.fromkeys(result)),
        accepted_lifecycle_transactions,
    )


def _spec_cas_guard(expected_spec, expected_digest: str):
    expected_parent = ContentHash(expected_digest)

    def _guard(candidate):
        if (
            candidate.version != expected_spec.version + 1
            or candidate.parent != expected_parent
        ):
            raise _ConcurrentSpecChange(
                "concurrent Spec change while publishing reaper settlement"
            )
        return candidate

    return _guard


@dataclasses.dataclass(frozen=True)
class ReaperReport:
    """Outcome of one journal-first, lifecycle-ordered recovery pass."""

    crashed_rollbacks_reaped: tuple[str, ...]
    scope_escapes_deleted: tuple[str, ...]
    crashed_rollbacks_skipped: tuple[str, ...]
    failures: tuple[str, ...]
    mutation_journals_recovered: tuple[str, ...] = ()
    mutation_journals_preserved: tuple[str, ...] = ()
    unattached_legacy_skipped: tuple[str, ...] = ()
    legacy_mutations_skipped: tuple[str, ...] = ()
    cleanup_settlement_ref: str | None = None
    authority_spec_version: int | None = None
    authority_spec_hash: str | None = None
    cleanup_lifecycle_ref: str | None = None

    @property
    def reaped_anything(self) -> bool:
        return bool(
            self.crashed_rollbacks_reaped
            or self.scope_escapes_deleted
            or self.mutation_journals_recovered
        )

    @property
    def has_signal(self) -> bool:
        return bool(
            self.crashed_rollbacks_reaped
            or self.scope_escapes_deleted
            or self.crashed_rollbacks_skipped
            or self.failures
            or self.mutation_journals_recovered
            or self.mutation_journals_preserved
            or self.unattached_legacy_skipped
            or self.legacy_mutations_skipped
            or self.cleanup_settlement_ref
        )


def _reap_orphaned_dirt_locked(
    corpus: "Corpus",
    cwd: str | Path,
    *,
    expected_spec=_UNSPECIFIED,
    expected_spec_digest=_UNSPECIFIED,
    journal_root: str | Path | None = None,
    run_id: str | None = None,
    tool_authority_digests: Mapping[str, str | None] | None = None,
) -> ReaperReport:
    """Recover exact journals, then safely settle eligible legacy dirt.

    ``expected_spec`` and ``expected_spec_digest`` bind a parent workflow's
    cached semantic and raw-manifest authority. No recovery or legacy cleanup
    may mutate source if either differs from the locked persisted Spec. A
    successful legacy pass publishes one lifecycle settlement, so a later
    startup never mistakes already-clean bytes for unexplained drift.
    """

    authority = _assert_workspace_mutation_authority(cwd)
    cwd_path = authority.root
    root_fd = authority.root_descriptor
    journal_report = RecoveryReport((), (), ())
    crashed_reaped: list[str] = []
    crashed_skipped: list[str] = []
    escapes_deleted: list[str] = []
    failures: list[str] = []
    legacy_skipped: list[str] = []
    unattached_legacy_skipped: tuple[str, ...] = ()
    cleanup_settlement_ref: str | None = None
    cleanup_journal: WorkspaceMutationJournal | None = None
    locked_spec = None
    settlement_reason: str | None = None
    base_settlement_reason: str | None = None
    settled_implementation_refs: tuple[str, ...] = ()
    settlement_run_id = run_id or "reaper"
    authority_spec_version: int | None = None
    authority_spec_hash: str | None = None
    cleanup_lifecycle_ref: str | None = None
    current_tool_authority = dict(tool_authority_digests or {})
    lifecycle_records: tuple[tuple[str, LifecycleEvent], ...] = ()
    already_reaped: set[str] = set()
    artifacts = _ReaperArtifactReader(corpus)

    def _report() -> ReaperReport:
        return ReaperReport(
            crashed_rollbacks_reaped=tuple(crashed_reaped),
            scope_escapes_deleted=tuple(escapes_deleted),
            crashed_rollbacks_skipped=tuple(crashed_skipped),
            failures=tuple(failures),
            mutation_journals_recovered=journal_report.recovered,
            mutation_journals_preserved=journal_report.preserved,
            unattached_legacy_skipped=unattached_legacy_skipped,
            legacy_mutations_skipped=tuple(legacy_skipped),
            cleanup_settlement_ref=cleanup_settlement_ref,
            authority_spec_version=authority_spec_version,
            authority_spec_hash=authority_spec_hash,
            cleanup_lifecycle_ref=cleanup_lifecycle_ref,
        )

    with _readable_spec_authority(corpus) as spec:
        _assert_workspace_mutation_authority(cwd_path)
        locked_spec = spec
        if spec is not None:
            authority_spec_version = spec.version
            persisted_spec, persisted_digest = corpus.get_spec_with_digest(
                spec.version
            )
            if persisted_spec != spec:
                failures.append(
                    "locked Spec differs from its persisted manifest authority"
                )
                return _report()
            authority_spec_hash = str(persisted_digest)
            valid_lineage, lineage_reason = (
                _build_verify._validate_spec_lineage(corpus, spec)
            )
            if not valid_lineage:
                failures.append(
                    "latest Spec lineage is invalid before recovery: "
                    f"{lineage_reason}"
                )
                return _report()
            try:
                lifecycle_records = _attached_lifecycle_records(corpus, spec)
                (
                    reaped_refs,
                    accepted_lifecycle_transactions,
                ) = _reaped_authority(
                    corpus,
                    artifacts,
                    spec,
                    lifecycle_records,
                    cwd_path,
                    root_fd,
                )
                already_reaped = set(reaped_refs)
            except Exception as exc:  # noqa: BLE001 - authority fails closed
                failures.append(str(exc))
                return _report()
        else:
            accepted_lifecycle_transactions = {}

        if expected_spec is not _UNSPECIFIED and spec != expected_spec:
            failures.append(
                "Spec authority changed before mutation recovery: "
                f"expected version {getattr(expected_spec, 'version', None)}, "
                f"observed {getattr(spec, 'version', None)}"
            )
            return _report()
        if (
            expected_spec_digest is not _UNSPECIFIED
            and authority_spec_hash != str(expected_spec_digest)
        ):
            failures.append(
                "raw Spec authority changed before recovery: "
                f"expected {expected_spec_digest}, observed "
                f"{authority_spec_hash}"
            )
            return _report()

        journal_report = _recover_mutation_journals_locked(
            corpus,
            cwd_path,
            accepted_lifecycle_transactions=accepted_lifecycle_transactions,
        )
        failures.extend(journal_report.failures)
        if failures:
            return _report()

        if spec is None:
            try:
                unattached_legacy_skipped = _unattached_implementation_refs(
                    corpus,
                    (),
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic read only
                failures.append(
                    "failed to inventory unattached Implementations "
                    f"({type(exc).__name__}: {exc})"
                )
            return _report()

        try:
            _assert_workspace_mutation_authority(cwd_path)
            attached, history = _attached_implementation_history(corpus, spec)
            unattached_legacy_skipped = _unattached_implementation_refs(
                corpus,
                spec.implementation_refs,
            )
            plan_slices = _plan_slice_map(spec)
            recovered_refs = {
                settlement.implementation_ref
                for settlement in journal_report.settlements
                if (
                    settlement.disposition == "recovered"
                    and settlement.implementation_ref is not None
                )
            }
            preserved_refs = {
                settlement.implementation_ref
                for settlement in journal_report.settlements
                if (
                    settlement.disposition == "preserved"
                    and settlement.implementation_ref is not None
                )
            }
            preserved_lifecycle_settlements = tuple(
                settlement
                for settlement in journal_report.settlements
                if (
                    settlement.disposition == "preserved"
                    and settlement.implementation_ref is None
                )
            )
            negative_refs = frozenset((*already_reaped, *recovered_refs))
            historical, historical_refs, absorbed_refs = (
                _historical_landed_authorities(
                    corpus,
                    artifacts,
                    spec,
                    attached,
                    lifecycle_records,
                    plan_slices,
                    cwd_path,
                    root_fd,
                )
            )
            retained, retained_refs = _retained_authorities(
                corpus,
                artifacts,
                spec,
                history,
                plan_slices,
                negative_refs=negative_refs,
                tool_authority_digests=current_tool_authority,
            )
            positive_refs = frozenset((
                *historical_refs,
                *retained_refs,
            ))
            absorbed_refs = frozenset(
                absorbed_refs.difference(positive_refs)
            )
            inactive_refs = frozenset((*negative_refs, *absorbed_refs))
            for item in attached:
                if item.ref in absorbed_refs:
                    legacy_skipped.append(
                        f"slice {item.implementation.slice_id}: exact record "
                        f"{item.ref} historical landed postimage is absorbed "
                        "into current base "
                        f"{spec.project_context.base_commit}"
                    )
            contradictory = positive_refs.intersection(negative_refs)
            if contradictory:
                raise ValueError(
                    "Implementation evidence is both surviving and recovered: "
                    f"{sorted(contradictory)!r}"
                )
            attached_by_ref = {item.ref: item for item in attached}
            missing_preserved = preserved_refs.difference(attached_by_ref)
            if missing_preserved:
                raise ValueError(
                    "preserved recovery journal names unattached "
                    f"Implementations: {sorted(missing_preserved)!r}"
                )
            unadopted_preserved = preserved_refs.difference(positive_refs)
            if unadopted_preserved:
                raise ValueError(
                    "preserved recovery journal Implementation lacks "
                    "independent current-retained or landed authority: "
                    f"{sorted(unadopted_preserved)!r}"
                )
            contradictory_preserved = preserved_refs.intersection(
                negative_refs
            )
            if contradictory_preserved:
                raise ValueError(
                    "preserved recovery journal Implementation is also "
                    "settled negatively: "
                    f"{sorted(contradictory_preserved)!r}"
                )
            attached_lifecycle_refs = {
                event_ref for event_ref, _event in lifecycle_records
            }
            preserved_barrier_paths: list[str] = []
            for settlement in preserved_lifecycle_settlements:
                if (
                    settlement.lifecycle_ref is None
                    or settlement.lifecycle_ref not in attached_lifecycle_refs
                ):
                    raise ValueError(
                        "preserved lifecycle-only recovery journal lacks one "
                        "exact attached lifecycle authority: "
                        f"{settlement.transaction_id}"
                    )
                preserved_barrier_paths.extend(
                    _normalize_bound_workspace_path(cwd_path, root_fd, path)
                    for path in settlement.protected_scope
                )
            preserved_barrier_paths = list(dict.fromkeys(
                preserved_barrier_paths
            ))
            missing_positive = positive_refs.difference(attached_by_ref)
            if missing_positive:
                raise ValueError(
                    "positive recovery authority names unattached "
                    f"Implementations: {sorted(missing_positive)!r}"
                )
            authorities = list(_proven_positive_authorities(
                artifacts,
                cwd_path,
                root_fd,
                (
                    *historical,
                    *retained,
                ),
            ))
            attached_paths: dict[int, tuple[str, ...]] = {}
            for item in attached:
                if item.ref in inactive_refs:
                    attached_paths[item.index] = ()
                    continue
                plan_entry = plan_slices.get(item.implementation.slice_id)
                try:
                    attached_paths[item.index] = _implementation_paths(
                        artifacts,
                        cwd_path,
                        root_fd,
                        item,
                        slice_=(
                            plan_entry[1] if plan_entry is not None else None
                        ),
                    )
                except Exception as exc:
                    raise ValueError(
                        f"slice {item.implementation.slice_id}: attached "
                        "mutation paths are unreadable or malformed "
                        f"({type(exc).__name__}: {exc})"
                    ) from exc
        except Exception as exc:  # noqa: BLE001 - authority fails closed
            failures.append(str(exc))
            return _report()

        expected_base = spec.project_context.base_commit
        candidates: list[_LegacyCandidate] = []
        observed_head: str | None = None
        cleanup_authority_refs: set[str] = set()

        # Classify the one latest active record that current execution
        # authority can clean for each current slice. Non-skip-worthy states
        # such as pending_gates remain unproved: reaper must never erase them,
        # but exact surviving bytes must block provider dispatch unless a
        # prepared journal or later positive authority protects them.
        try:
            for slice_id, (tier, slice_) in plan_slices.items():
                item = next(
                    (
                        candidate
                        for candidate in reversed(history.get(slice_id, ()))
                        if candidate.ref not in inactive_refs
                        and candidate.ref not in positive_refs
                    ),
                    None,
                )
                if item is None:
                    continue
                impl = item.implementation
                identity_matches = (
                    impl.base_commit == expected_base
                    and _partition._identity_matches(
                        corpus,
                        spec,
                        tier,
                        slice_,
                        impl,
                        tool_authority_digest=current_tool_authority.get(
                            slice_id
                        ),
                        artifact_reader=artifacts.read,
                    )
                )
                if not identity_matches:
                    continue
                if _is_failure_class(impl.classification) or impl.scope_escapes:
                    cleanup_authority_refs.add(item.ref)
        except Exception as exc:  # noqa: BLE001 - authority fails closed
            failures.append(
                "cannot classify active attached mutation authority "
                f"({type(exc).__name__}: {exc})"
            )
            return _report()

        # Every attached record needs an explicit disposition before cleanup.
        # Previous code inspected only the latest failure for slice IDs still
        # present in the current DAG.  Removing/renaming a slice, revising a
        # directive under the same ID, or appending an empty non-candidate
        # record could therefore hide exact dirty bytes and let Tier dispatch
        # a provider atop them.  Audit every other active record before any
        # write. Newer surviving authorities may legitimately supersede an
        # old path; every other recorded path must exactly match that record's
        # immutable base tree or the current reviewed source baseline. The
        # latter protects committed baseline bytes only at the exact pinned
        # HEAD; it grants neither cleanup nor current-plan completion authority.
        # Every unprotected escape must still be absent.
        for item in attached:
            if (
                item.ref in inactive_refs
                or item.ref in positive_refs
                or item.ref in cleanup_authority_refs
            ):
                continue
            impl = item.implementation
            try:
                diff_bytes = artifacts.read(
                    impl.diff_ref,
                    max_bytes=_MAX_REAPER_DIFF_BYTES,
                    label="Implementation diff",
                )
                diff_paths = _diff_paths(
                    diff_bytes,
                    cwd_path,
                    root_fd,
                    base_commit=impl.base_commit,
                )
                projected_entries = _positive_postimage_entries(
                    cwd_path,
                    root_fd,
                    item,
                    diff_bytes,
                    diff_paths,
                )
                recorded_paths = tuple(
                    _normalize_bound_workspace_path(cwd_path, root_fd, path)
                    for path in (impl.files_modified or ())
                )
                if bool(diff_bytes.strip()) != bool(diff_paths):
                    raise ValueError(
                        "attached diff bytes and parsed path projection "
                        "disagree"
                    )
                if not set(recorded_paths).issubset(diff_paths):
                    raise ValueError(
                        "files_modified contains paths absent from the "
                        "attached diff"
                    )
            except Exception as exc:  # noqa: BLE001 - absence must be proved
                failures.append(
                    f"slice {impl.slice_id}: non-current attached evidence "
                    "is unreadable or cannot be inspected "
                    f"({type(exc).__name__}: {exc})"
                )
                continue

            unprotected_present_paths: list[str] = []
            uncovered_diff_paths = tuple(
                path
                for path in diff_paths
                if not any(
                    _authority_covers_candidate_postimage(
                        authority,
                        path=path,
                        postimage=projected_entries[path],
                        implementation_index=item.index,
                    )
                    for authority in _overlapping_authorities(
                        (path,),
                        authorities,
                    )
                )
            )
            try:
                paths_match_base = _worktree_paths_match_base_tree(
                    cwd_path,
                    root_fd,
                    base_commit=impl.base_commit,
                    paths=uncovered_diff_paths,
                )
                if (
                    not paths_match_base
                    and expected_base != impl.base_commit
                    and _head(cwd_path, root_fd) == expected_base
                ):
                    # A reviewed source advance may already contain an old
                    # Implementation's changes, including records whose slice
                    # was removed from the new DAG. Exact baseline equality is
                    # preservation evidence, not evidence that those old gates
                    # passed. Keep the original per-path absence proof as well.
                    paths_match_base = all(
                        _worktree_paths_match_base_tree(
                            cwd_path,
                            root_fd,
                            base_commit=impl.base_commit,
                            paths=(path,),
                        )
                        or _worktree_paths_match_base_tree(
                            cwd_path,
                            root_fd,
                            base_commit=expected_base,
                            paths=(path,),
                        )
                        for path in uncovered_diff_paths
                    )
                    if _head(cwd_path, root_fd) != expected_base:
                        raise MutationJournalError(
                            "checkout HEAD changed during reviewed baseline proof"
                        )
            except Exception as exc:  # noqa: BLE001 - absence must be proved
                failures.append(
                    f"slice {impl.slice_id}: cannot prove non-current paths "
                    "match their recorded or current reviewed base "
                    f"({type(exc).__name__}: {exc})"
                )
                continue
            if not paths_match_base:
                unprotected_present_paths.extend(uncovered_diff_paths)

            for rendered in impl.scope_escapes or ():
                try:
                    path = _normalize_bound_workspace_path(
                        cwd_path,
                        root_fd,
                        rendered,
                    )
                    if _bound_entry_metadata(cwd_path, root_fd, path) is None:
                        continue
                except Exception as exc:  # noqa: BLE001 - absence unproved
                    failures.append(
                        f"slice {impl.slice_id}: cannot prove non-current "
                        f"scope escape absent ({type(exc).__name__}: {exc})"
                    )
                    continue
                protected = any(
                    _authority_is_newer(authority, item.index)
                    and _covered_by(path, (authority.path,))
                    for authority in _overlapping_authorities(
                        (path,),
                        authorities,
                    )
                )
                if not protected:
                    unprotected_present_paths.append(path)

            if unprotected_present_paths:
                failures.append(
                    f"slice {impl.slice_id}: attached record {item.ref} is "
                    "not current-plan authority for cleanup but its mutation "
                    "may still be present"
                )
            else:
                legacy_skipped.append(
                    f"slice {impl.slice_id}: attached record {item.ref} is "
                    "not current-plan authority and no unprotected exact "
                    "postimage is present"
                )

        for slice_id, (_tier, _slice) in plan_slices.items():
            records = history.get(slice_id)
            if not records:
                continue
            for settled in reversed(records):
                if settled.ref in negative_refs:
                    legacy_skipped.append(
                        f"slice {slice_id}: exact record {settled.ref} "
                        "already settled"
                    )
            item = next(
                (
                    candidate
                    for candidate in reversed(records)
                    if candidate.ref not in inactive_refs
                ),
                None,
            )
            if item is None:
                continue
            impl = item.implementation
            if item.ref in positive_refs:
                legacy_skipped.append(
                    f"slice {slice_id}: exact record {item.ref} is retained"
                )
                continue
            if item.ref not in cleanup_authority_refs:
                continue
            if any(
                prior.ref not in inactive_refs
                for prior in records
                if prior.index < item.index
            ):
                failures.append(
                    f"slice {slice_id}: legacy failure diff is cumulative over "
                    "an earlier attached record; exact journal preimage is "
                    "required"
                )
                continue
            if observed_head is None:
                try:
                    observed_head = _head(cwd_path, root_fd)
                except MutationJournalError as exc:
                    failures.append(str(exc))
                    continue
            if observed_head != expected_base:
                failures.append(
                    "legacy reaper checkout HEAD differs from Spec base_commit: "
                    f"HEAD={observed_head}, base_commit={expected_base}"
                )
                continue

            later_unproved_paths = tuple(dict.fromkeys(
                path
                for later in attached
                if (
                    later.index > item.index
                    and later.ref not in inactive_refs
                    and later.ref not in positive_refs
                )
                for path in attached_paths[later.index]
            ))
            earlier_unproved_paths = tuple(dict.fromkeys(
                path
                for earlier in attached
                if (
                    earlier.index < item.index
                    and earlier.ref not in inactive_refs
                    and earlier.ref not in positive_refs
                )
                for path in attached_paths[earlier.index]
            ))
            diff_bytes: bytes | None = None
            diff_paths: tuple[str, ...] = ()
            if _is_failure_class(impl.classification):
                try:
                    resolved = artifacts.read(
                        impl.diff_ref,
                        max_bytes=_MAX_REAPER_DIFF_BYTES,
                        label="Implementation diff",
                    )
                    parsed_paths = _diff_paths(
                        resolved,
                        cwd_path,
                        root_fd,
                        base_commit=impl.base_commit,
                    )
                    projected_entries = _positive_postimage_entries(
                        cwd_path,
                        root_fd,
                        item,
                        resolved,
                        parsed_paths,
                    )
                except Exception as exc:  # noqa: BLE001 - exact diff required
                    failures.append(
                        f"slice {slice_id}: diff_ref unresolvable or malformed "
                        f"({type(exc).__name__}: {exc})"
                    )
                    continue
                if resolved.strip():
                    if _overlaps_any(parsed_paths, preserved_barrier_paths):
                        failures.append(
                            f"slice {slice_id}: failure diff overlaps a "
                            "preserved lifecycle journal scope without exact "
                            "postimage authority"
                        )
                        continue
                    overlaps = _overlapping_authorities(
                        parsed_paths,
                        authorities,
                    )
                    fully_covered_by_newer = all(
                        any(
                            _authority_is_newer(authority, item.index)
                            and _covered_by(path, (authority.path,))
                            for authority in overlaps
                        )
                        for path in parsed_paths
                    )
                    fully_covered_by_surviving_postimage = all(
                        any(
                            _authority_covers_candidate_postimage(
                                authority,
                                path=path,
                                postimage=projected_entries[path],
                                implementation_index=item.index,
                            )
                            for authority in overlaps
                        )
                        for path in parsed_paths
                    )
                    if fully_covered_by_newer:
                        legacy_skipped.append(
                            f"slice {slice_id}: failure diff is superseded by "
                            "newer surviving path authority"
                        )
                    elif fully_covered_by_surviving_postimage:
                        legacy_skipped.append(
                            f"slice {slice_id}: cumulative failure diff "
                            "matches exact surviving path authority"
                        )
                    elif overlaps:
                        failures.append(
                            f"slice {slice_id}: failure diff overlaps a retained "
                            "postimage without an exact attempt preimage"
                        )
                    elif _overlaps_any(parsed_paths, earlier_unproved_paths):
                        failures.append(
                            f"slice {slice_id}: failure diff overlaps earlier "
                            "unproved Implementation evidence"
                        )
                    elif _overlaps_any(parsed_paths, later_unproved_paths):
                        failures.append(
                            f"slice {slice_id}: failure diff overlaps later "
                            "unproved Implementation evidence"
                        )
                    elif not _diff_applies_in_reverse(resolved, root_fd):
                        crashed_skipped.append(slice_id)
                        failures.append(
                            f"slice {slice_id}: failure diff does not apply in "
                            "reverse; workspace substrate is unproved"
                        )
                    else:
                        diff_bytes = resolved
                        diff_paths = parsed_paths

            eligible_escapes: list[str] = []
            for rendered in impl.scope_escapes or ():
                path = _normalize_bound_workspace_path(
                    cwd_path,
                    root_fd,
                    rendered,
                )
                if _overlaps_any((path,), preserved_barrier_paths):
                    failures.append(
                        f"slice {slice_id}: scope escape {path!r} overlaps a "
                        "preserved lifecycle journal scope without exact "
                        "postimage authority"
                    )
                    continue
                overlaps = _overlapping_authorities((path,), authorities)
                covered_by_newer = any(
                    _authority_is_newer(authority, item.index)
                    and _covered_by(path, (authority.path,))
                    for authority in overlaps
                )
                if covered_by_newer:
                    legacy_skipped.append(
                        f"slice {slice_id}: scope escape {path!r} is protected "
                        "by newer surviving path authority"
                    )
                    continue
                if overlaps:
                    failures.append(
                        f"slice {slice_id}: scope escape {path!r} overlaps a "
                        "retained postimage without an exact attempt preimage"
                    )
                    continue
                if _overlaps_any((path,), earlier_unproved_paths):
                    failures.append(
                        f"slice {slice_id}: scope escape {path!r} overlaps "
                        "earlier unproved Implementation evidence"
                    )
                    continue
                if _overlaps_any((path,), later_unproved_paths):
                    failures.append(
                        f"slice {slice_id}: scope escape {path!r} overlaps "
                        "later unproved Implementation evidence"
                    )
                    continue
                eligible_escapes.append(path)

            if diff_bytes is not None or eligible_escapes:
                candidates.append(
                    _LegacyCandidate(
                        attached=item,
                        diff_bytes=diff_bytes,
                        diff_paths=diff_paths,
                        escape_paths=tuple(dict.fromkeys(eligible_escapes)),
                    )
                )

        if failures:
            return _report()

        for index, left in enumerate(candidates):
            for right in candidates[index + 1:]:
                if _overlaps_any(left.mutation_paths, right.mutation_paths):
                    failures.append(
                        "legacy cleanup candidates overlap: "
                        f"{left.attached.implementation.slice_id} and "
                        f"{right.attached.implementation.slice_id}"
                    )
        if failures or not candidates:
            return _report()

        _assert_workspace_mutation_authority(cwd_path)
        root = (
            Path(journal_root)
            if journal_root is not None
            else Path(corpus.spec_dir) / "runs" / "reaper" / "mutations"
        )
        cleanup_journal = WorkspaceMutationJournal.begin(
            cwd=cwd_path,
            journal_root=root,
            slice_id="__legacy_reaper__",
            scope=tuple(dict.fromkeys(
                path for candidate in candidates for path in candidate.mutation_paths
            )),
            sequence=0,
            spec=spec,
        )
        staged_reaped: list[str] = []
        staged_escapes: list[str] = []
        try:
            for candidate in sorted(
                candidates,
                key=lambda value: value.attached.index,
                reverse=True,
            ):
                impl = candidate.attached.implementation
                if candidate.diff_bytes is not None:
                    _assert_workspace_mutation_authority(cwd_path)
                    if not _git_apply_reverse(candidate.diff_bytes, root_fd):
                        raise MutationJournalError(
                            f"slice {impl.slice_id}: reverse check passed but "
                            "legacy apply failed"
                        )
                    try:
                        _assert_workspace_mutation_authority(cwd_path)
                    except BaseException as authority_exc:
                        if not _git_apply_forward(candidate.diff_bytes, root_fd):
                            raise MutationJournalError(
                                f"slice {impl.slice_id}: workspace authority "
                                "changed after reverse apply and descriptor-"
                                "bound compensation failed"
                            ) from authority_exc
                        raise
                    staged_reaped.append(impl.slice_id)
                if candidate.escape_paths:
                    _assert_workspace_mutation_authority(cwd_path)
                    scoped = dataclasses.replace(
                        impl,
                        scope_escapes=candidate.escape_paths,
                    )
                    reverted, deleted = _revert_scope_escapes_bound(
                        scoped,
                        cwd_path,
                        root_fd,
                        strict=True,
                    )
                    _assert_workspace_mutation_authority(cwd_path)
                    staged_escapes.extend((*reverted, *deleted))

            _assert_workspace_mutation_authority(cwd_path)
            cleanup_journal.assert_head_unchanged()
            cleanup_journal.assert_repository_metadata_unchanged()
            cleanup_journal.assert_index_unchanged()
            escapes = cleanup_journal.detected_scope_escapes()
            if escapes:
                raise MutationJournalError(
                    f"legacy cleanup escaped aggregate journal scope: {list(escapes)!r}"
                )

            if not staged_reaped and not staged_escapes:
                cleanup_journal.commit()
                cleanup_journal = None
                return _report()

            settled_implementation_refs = tuple(dict.fromkeys(
                candidate.attached.ref for candidate in candidates
            ))
            payload = {
                "schema": _REAPER_SETTLEMENT_SCHEMA,
                "transaction_id": cleanup_journal.transaction_id,
                "implementation_refs": list(settled_implementation_refs),
                "crashed_rollbacks_reaped": list(staged_reaped),
                "scope_escapes_cleaned": list(staged_escapes),
                "mutation_paths": list(dict.fromkeys((
                    *(
                        path
                        for candidate in candidates
                        if candidate.diff_bytes is not None
                        for path in candidate.diff_paths
                    ),
                    *staged_escapes,
                ))),
            }
            payload_ref = corpus.put_artifact(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
                kind="reaper_settlements",
                name_hint=f"{cleanup_journal.transaction_id}.json",
            )
            cleanup_settlement_ref = str(payload_ref)
            base_settlement_reason = (
                _REAPER_EVENT_PREFIX + cleanup_settlement_ref
            )
        except BaseException as exc:
            restored = cleanup_journal.restore(full=True)
            cleanup_journal = None
            cleanup_settlement_ref = None
            if isinstance(exc, Exception):
                failures.append(
                    "legacy cleanup transaction failed and "
                    f"{'restored' if restored else 'could not restore'}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return _report()
            raise

    # Corpus.commit deliberately occurs after releasing locked_spec_authority;
    # its CAS transform proves that no writer advanced the parent in between.
    assert locked_spec is not None
    assert cleanup_journal is not None
    assert base_settlement_reason is not None
    try:
        settlement_reason, _mutation_settlement_ref = (
            prepare_mutation_lifecycle_settlement(
                corpus,
                (cleanup_journal,),
                kind=_REAPER_EVENT_KIND,
                reason=base_settlement_reason,
                run_id=settlement_run_id,
                implementation_refs=settled_implementation_refs,
            )
        )
    except BaseException as exc:
        restored = cleanup_journal.restore(full=True)
        cleanup_journal = None
        failures.append(
            "legacy cleanup exact settlement preparation failed and "
            f"{'restored' if restored else 'could not restore'}: "
            f"{type(exc).__name__}: {exc}"
        )
        if not isinstance(exc, Exception):
            raise
        return _report()
    try:
        _assert_workspace_mutation_authority(cwd_path)
    except Exception as exc:  # noqa: BLE001 - retain journal on lost root
        failures.append(
            "legacy cleanup workspace authority changed before settlement; "
            f"durable journal retained: {type(exc).__name__}: {exc}"
        )
        return _report()
    try:
        committed = corpus.commit(
            lifecycle_event_factory=lambda version: LifecycleEvent(
                kind=_REAPER_EVENT_KIND,
                reason=settlement_reason,
                affected_versions=(version,),
                execution=exec_ctx_now("reaper", run_id=settlement_run_id),
            ),
            transform=_spec_cas_guard(locked_spec, authority_spec_hash),
        )
    except BaseException as exc:
        # A raised transport/IO error does not prove that Corpus.commit failed
        # before publication.  Reconcile the prepared lifecycle journal first;
        # it preserves an attached event and restores only when exact
        # post-baseline evidence is definitively absent.  Unreadable authority
        # raises and leaves the durable journal armed for startup recovery.
        try:
            with corpus.locked_spec_authority() as latest:
                outcome = cleanup_journal._settle_or_restore_outcome_locked(
                    corpus,
                    full=True,
                )
                authority_spec_version = latest.version
                persisted_latest, persisted_digest = (
                    corpus.get_spec_with_digest(latest.version)
                )
                if persisted_latest != latest:
                    raise _ConcurrentSpecChange(
                        "locked Spec differs from persisted authority"
                    )
                authority_spec_hash = str(persisted_digest)
        except BaseException as reconcile_exc:
            failures.append(
                "legacy cleanup settlement outcome is ambiguous; durable "
                "journal retained: "
                f"{type(reconcile_exc).__name__}: {reconcile_exc}"
            )
            if not isinstance(exc, Exception):
                raise
            return _report()

        if outcome.completed:
            cleanup_journal = None
        if (
            outcome.disposition == "preserved"
            and outcome.lifecycle_ref is not None
        ):
            cleanup_lifecycle_ref = outcome.lifecycle_ref
            crashed_reaped.extend(staged_reaped)
            escapes_deleted.extend(staged_escapes)
            legacy_skipped.append(
                "legacy cleanup commit acknowledgement was lost; exact "
                "attached lifecycle reconciled"
            )
            if not isinstance(exc, Exception):
                raise
            return _report()

        cleanup_settlement_ref = None
        failures.append(
            "legacy cleanup settlement lost Spec CAS and "
            f"{'restored' if outcome.completed else 'could not restore'}: "
            f"{type(exc).__name__}: {exc}"
        )
        if not isinstance(exc, Exception):
            raise
        return _report()

    crashed_reaped.extend(staged_reaped)
    escapes_deleted.extend(staged_escapes)
    authority_spec_version = committed.spec.version
    persisted_committed, persisted_digest = corpus.get_spec_with_digest(
        committed.spec.version
    )
    if persisted_committed != committed.spec:
        failures.append(
            "legacy cleanup committed Spec differs from persisted authority"
        )
        return _report()
    authority_spec_hash = str(persisted_digest)
    refs = tuple(getattr(committed.refs, "lifecycle_event_refs", ()) or ())
    if len(refs) != 1:
        failures.append(
            "legacy cleanup commit did not return one lifecycle reference"
        )
    else:
        cleanup_lifecycle_ref = str(refs[0])
    try:
        _assert_workspace_mutation_authority(cwd_path)
        cleanup_journal.commit(attached=True, corpus=corpus)
    except Exception as exc:  # noqa: BLE001 - attached bytes must not restore
        failures.append(
            "legacy cleanup lifecycle is attached but journal finalization "
            f"failed: {type(exc).__name__}: {exc}"
        )
    return _report()


def reap_orphaned_dirt(
    corpus: "Corpus",
    cwd: str | Path,
    *,
    expected_spec=_UNSPECIFIED,
    expected_spec_digest=_UNSPECIFIED,
    journal_root: str | Path | None = None,
    run_id: str | None = None,
    tool_authority_digests: Mapping[str, str | None] | None = None,
) -> ReaperReport:
    """Run journal-first recovery under the checkout's mutation lock.

    The lock is reentrant for canonical Tier callers and mandatory here for
    direct/composed callers so recovery, legacy mutation, Spec settlement, and
    journal finalization remain one serialized workspace operation.
    """

    # Preserve the lexical spelling until the lock has bound the exact root.
    # Resolving first would create a race in which an attacker can replace the
    # pathname between ``resolve`` and descriptor acquisition.
    cwd_path = Path(os.path.abspath(os.fspath(cwd)))
    report: ReaperReport | None = None
    try:
        with ExitStack() as workspace_release:
            workspace_context = recovery_workspace_mutation_lock(
                corpus,
                cwd_path,
            )
            workspace_release.callback(workspace_context.retry_cleanup)
            authority = workspace_context.__enter__()
            report = _reap_orphaned_dirt_locked(
                corpus,
                authority.root,
                expected_spec=expected_spec,
                expected_spec_digest=expected_spec_digest,
                journal_root=journal_root,
                run_id=run_id,
                tool_authority_digests=tool_authority_digests,
            )
    except _SpecAuthorityEntryError as exc:
        coordination_failure = (
            "latest Spec authority is unreadable before recovery; "
            f"journals retained: {type(exc).__name__}: {exc}"
        )
    except MutationJournalError as exc:
        if _is_retained_workspace_root_identity_change(exc):
            coordination_failure = (
                "workspace authority root changed identity during recovery "
                "coordination; journals retained: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            coordination_failure = (
                "workspace recovery coordination failed; journals retained: "
                f"{type(exc).__name__}: {exc}"
            )
    else:
        assert report is not None
        return report

    if report is not None:
        return dataclasses.replace(
            report,
            failures=(*report.failures, coordination_failure),
        )
    return ReaperReport(
        crashed_rollbacks_reaped=(),
        scope_escapes_deleted=(),
        crashed_rollbacks_skipped=(),
        failures=(coordination_failure,),
    )


__all__ = ["ReaperReport", "reap_orphaned_dirt"]
