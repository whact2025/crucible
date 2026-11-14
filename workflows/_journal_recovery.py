# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Narrow recovery for a journal interrupted by an external Git cleanup.

This module deliberately does *not* provide a force switch.  It supports one
closed recovery shape: guard-only linked worktrees were removed, exactly one
pre-existing stash was advanced by one append that preserves the dirty peer,
and the primary journal-owned source postimage is still byte-identical to an
explicit Corpus diff artifact.  The accepted Git state is never rewritten;
only the primary worktree and index are restored from the journal baseline.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Callable, Iterator, cast

from assay.primitives.implement import GitCaptureError, capture_workspace_delta

# The journal replaces process-local authority objects after fork.  Keep the
# module live so recovery never retains pre-fork TLS, locks, or owner maps.
from . import _mutation_journal as _mutation_journal_state
from ._mutation_journal import (
    _DURABLE_JOURNAL_ROOTS,
    _GIT_SAFE_OPTIONS,
    _MANIFEST,
    _REGISTRY_DIRECTORY,
    _SCHEMA,
    _SettlementCaptureBudget,
    _assert_authority_directory_binding,
    _assert_settlement_parent_binding,
    _assert_workspace_mutation_authority,
    _git,
    _git_common_dir,
    _git_control_path,
    _git_dir,
    _git_worktree_root,
    _head,
    _hidden_index_paths,
    _index_path,
    _index_semantic_views,
    _is_live,
    _module_admin_roots,
    _module_namespace_topology,
    _nested_clean_status,
    _object_inventory,
    _open_settlement_parent_if_present,
    _refs_snapshot,
    _registered_worktrees,
    _sanitized_git_env,
    _symbolic_head,
    _worktree_namespace_topology,
    _workspace_authority_common_dir,
    _xattr_inventory,
    _current_postimage_state,
    _capture_snapshot_blob_record,
    _bound_mutation_journal_directory,
    _bound_mutation_journal_run_directory,
    _bounded_directory_names,
    _MUTATION_JOURNAL_MAX_NESTED_MEMBERS,
    _MUTATION_JOURNAL_MAX_RUNS,
    _MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_BYTES,
    _MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_PATHS,
    _MUTATION_JOURNAL_MAX_MANIFEST_BYTES,
    _open_authority_directory,
    _open_authority_directory_at,
    _remove_bound_directory_tree,
    _remove_bound_regular_entry,
    _terminal_cleanup_journal_inventory,
    discover_mutation_journals,
    read_corpus_run_control_bounded,
    refresh_mutation_journal_artifact_binding,
    _writer_epoch_manifest_proof,
    MutationJournalError,
    WorkspaceMutationJournal,
)
from ._writer_epoch import (
    WriterEpochError,
    WriterEpochStorage,
    WriterEpochTerminalProof,
    parse_writer_epoch_terminal_proof,
    require_writer_epoch_quiescence,
)
from ._outcome import WorkflowOutcome
from ._run import write_atomic_bytes, write_atomic_json

if TYPE_CHECKING:
    from assay.corpus.store import Corpus


_ACCEPTANCE_SCHEMA = "crucible.external-git-journal-recovery.v2"
_ACCEPTANCE_FILE = "external_recovery.json"
_CLEANUP_RECEIPT_SCHEMA = (
    "crucible.external-git-journal-recovery-cleanup.v1"
)
_CLEANUP_RECEIPT_DIRECTORY = "journal_recovery_receipts"
_WRITER_EPOCH_PROOF_FIELD = "writer_epoch_terminal_proof"
_WRITER_EPOCH_MANIFEST_FIELD = "writer_epoch_manifest"
_RECOVERY_DIRECTORY = "external_recovery"
_TOMBSTONE_DIRECTORY = "journal_recovery_tombstones"
_HEX_OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_HEX_TRANSACTION = re.compile(r"[0-9a-f]{32}")
_ANCESTOR_INDEX_ARTIFACT_KIND = "journal_recovery_indexes"
_ANCESTOR_GITMODULES_ARTIFACT_KIND = "journal_recovery_gitmodules"
_ANCESTOR_ACCEPTANCE_ARTIFACT_KIND = "journal_recovery_acceptances"
_ANCESTOR_ACCEPTANCE_SCHEMA = "crucible.accepted-ancestor-submodule.v1"
_MAX_RECOVERY_EVENT_BYTES = 8 * 1024 * 1024
# A canonical event embedded as a JSON string can at most double through
# quote, backslash, and newline escaping.  The receipt also repeats a bounded
# subset of its binding metadata, whose encoded bytes are already present in
# the event.  Keep one explicit, rigorously derived ceiling for that envelope.
_MAX_RECOVERY_CLEANUP_RECEIPT_BYTES = (
    3 * _MAX_RECOVERY_EVENT_BYTES + 64 * 1024
)
_MAX_RECOVERY_DIFF_BYTES = 64 * 1024 * 1024
_MAX_ANCESTOR_ACCEPTANCE_BYTES = 1024 * 1024
_MAX_ANCESTOR_INDEX_BYTES = 64 * 1024 * 1024
_MAX_ANCESTOR_GITMODULES_BYTES = 1024 * 1024
_MAX_RECOVERY_ARTIFACT_AGGREGATE_BYTES = 192 * 1024 * 1024
_MAX_RECOVERY_ARTIFACTS = 8


@dataclass
class _RecoveryArtifactBudget:
    """Bound all Corpus artifact reads performed by one recovery inspection."""

    max_bytes: int = _MAX_RECOVERY_ARTIFACT_AGGREGATE_BYTES
    max_artifacts: int = _MAX_RECOVERY_ARTIFACTS
    bytes: int = 0
    artifacts: int = 0

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.max_bytes - self.bytes)

    def charge(self, size: int, *, kind: str) -> None:
        if size < 0:
            raise MutationJournalError(
                f"Corpus {kind} artifact has a negative size"
            )
        self.bytes += size
        self.artifacts += 1
        if self.bytes > self.max_bytes:
            raise MutationJournalError(
                "recovery Corpus artifacts exceed their aggregate byte ceiling"
            )
        if self.artifacts > self.max_artifacts:
            raise MutationJournalError(
                "recovery Corpus artifacts exceed their aggregate count ceiling"
            )


@dataclass(frozen=True)
class AcceptedAncestorSubmodule:
    """Exact preserved postimage for one intentional ancestor submodule add."""

    run_id: str
    transaction_id: str
    ancestor_transaction_id: str
    ancestor_root: str
    submodule_path: str
    submodule_url: str
    gitlink_oid: str
    index_ref: str
    gitmodules_ref: str
    acceptance_ref: str

    def normalized(self) -> "AcceptedAncestorSubmodule":
        run_id = self.run_id.strip()
        transaction_id = self.transaction_id.strip().lower()
        ancestor_transaction_id = self.ancestor_transaction_id.strip().lower()
        if not run_id or any(separator in run_id for separator in ("/", "\\")):
            raise MutationJournalError("accepted ancestor run id is invalid")
        if (
            _HEX_TRANSACTION.fullmatch(transaction_id) is None
            or _HEX_TRANSACTION.fullmatch(ancestor_transaction_id) is None
        ):
            raise MutationJournalError(
                "accepted ancestor transaction binding is invalid"
            )
        raw_ancestor_root = self.ancestor_root.strip()
        if not raw_ancestor_root:
            raise MutationJournalError("accepted ancestor root is required")
        ancestor_root = str(Path(raw_ancestor_root).resolve())
        submodule_path = self.submodule_path.strip()
        posix_path = PurePosixPath(submodule_path)
        if (
            not submodule_path
            or submodule_path != posix_path.as_posix()
            or posix_path.is_absolute()
            or any(part in {"", ".", "..", ".git"} for part in posix_path.parts)
            or "\\" in submodule_path
            or re.fullmatch(r"[A-Za-z0-9._/-]+", submodule_path) is None
        ):
            raise MutationJournalError("invalid accepted ancestor submodule path")
        submodule_url = self.submodule_url.strip()
        if (
            not submodule_url
            or submodule_url != self.submodule_url
            or any(not 0x21 <= ord(char) <= 0x7E for char in submodule_url)
            or any(char in {'"', "\\"} for char in submodule_url)
        ):
            raise MutationJournalError("invalid accepted ancestor submodule URL")
        gitlink_oid = self.gitlink_oid.strip().lower()
        if _HEX_OID.fullmatch(gitlink_oid) is None:
            raise MutationJournalError("invalid accepted ancestor gitlink OID")
        index_ref = self.index_ref.strip().lower()
        gitmodules_ref = self.gitmodules_ref.strip().lower()
        acceptance_ref = self.acceptance_ref.strip().lower()
        for label, value in (
            ("index", index_ref),
            (".gitmodules", gitmodules_ref),
            ("acceptance manifest", acceptance_ref),
        ):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise MutationJournalError(
                    f"accepted ancestor {label} ref must be an exact sha256 ref"
                )
        return AcceptedAncestorSubmodule(
            run_id=run_id,
            transaction_id=transaction_id,
            ancestor_transaction_id=ancestor_transaction_id,
            ancestor_root=ancestor_root,
            submodule_path=submodule_path,
            submodule_url=submodule_url,
            gitlink_oid=gitlink_oid,
            index_ref=index_ref,
            gitmodules_ref=gitmodules_ref,
            acceptance_ref=acceptance_ref,
        )

    def to_json(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "transaction_id": self.transaction_id,
            "ancestor_transaction_id": self.ancestor_transaction_id,
            "ancestor_root": self.ancestor_root,
            "submodule_path": self.submodule_path,
            "submodule_url": self.submodule_url,
            "gitlink_oid": self.gitlink_oid,
            "index_ref": self.index_ref,
            "gitmodules_ref": self.gitmodules_ref,
            "acceptance_ref": self.acceptance_ref,
        }

    def to_manifest_json(self) -> dict[str, str]:
        return {
            "schema": _ANCESTOR_ACCEPTANCE_SCHEMA,
            "run_id": self.run_id,
            "transaction_id": self.transaction_id,
            "ancestor_transaction_id": self.ancestor_transaction_id,
            "ancestor_root": self.ancestor_root,
            "submodule_path": self.submodule_path,
            "submodule_url": self.submodule_url,
            "gitlink_oid": self.gitlink_oid,
            "index_ref": self.index_ref,
            "gitmodules_ref": self.gitmodules_ref,
        }

    @classmethod
    def from_json(cls, record: dict[str, object]) -> "AcceptedAncestorSubmodule":
        return cls(
            run_id=str(record["run_id"]),
            transaction_id=str(record["transaction_id"]),
            ancestor_transaction_id=str(record["ancestor_transaction_id"]),
            ancestor_root=str(record["ancestor_root"]),
            submodule_path=str(record["submodule_path"]),
            submodule_url=str(record["submodule_url"]),
            gitlink_oid=str(record["gitlink_oid"]),
            index_ref=str(record["index_ref"]),
            gitmodules_ref=str(record["gitmodules_ref"]),
            acceptance_ref=str(record["acceptance_ref"]),
        ).normalized()


@dataclass(frozen=True)
class ExternalRecoveryRequest:
    """Exact operator authority for one retained journal."""

    run_id: str
    transaction_id: str
    expected_diff_ref: str
    removed_linked_worktrees: tuple[str, ...]
    appended_stash_tip: str
    reason: str
    ancestor_submodule: AcceptedAncestorSubmodule | None = None

    def normalized(self) -> "ExternalRecoveryRequest":
        run_id = self.run_id.strip()
        transaction_id = self.transaction_id.strip().lower()
        expected_diff_ref = self.expected_diff_ref.strip().lower()
        stash_tip = self.appended_stash_tip.strip().lower()
        reason = self.reason.strip()
        if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
            raise MutationJournalError("journal recovery requires an exact run id")
        if _HEX_TRANSACTION.fullmatch(transaction_id) is None:
            raise MutationJournalError("invalid mutation transaction id")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", expected_diff_ref) is None:
            raise MutationJournalError("expected diff ref must be an exact sha256 ref")
        if _HEX_OID.fullmatch(stash_tip) is None:
            raise MutationJournalError("invalid appended stash tip")
        if not reason:
            raise MutationJournalError("journal recovery requires an operator reason")
        if len(reason) > 4096:
            raise MutationJournalError("journal recovery operator reason is too large")
        paths = tuple(
            sorted(
                {str(Path(path).resolve()) for path in self.removed_linked_worktrees}
            )
        )
        if not paths:
            raise MutationJournalError(
                "at least one removed linked worktree is required"
            )
        return ExternalRecoveryRequest(
            run_id=run_id,
            transaction_id=transaction_id,
            expected_diff_ref=expected_diff_ref,
            removed_linked_worktrees=paths,
            appended_stash_tip=stash_tip,
            reason=reason,
            ancestor_submodule=(
                self.ancestor_submodule.normalized()
                if self.ancestor_submodule is not None
                else None
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "transaction_id": self.transaction_id,
            "expected_diff_ref": self.expected_diff_ref,
            "removed_linked_worktrees": list(self.removed_linked_worktrees),
            "appended_stash_tip": self.appended_stash_tip,
            "reason": self.reason,
            "ancestor_submodule": (
                self.ancestor_submodule.to_json()
                if self.ancestor_submodule is not None
                else None
            ),
        }

    @classmethod
    def from_json(cls, record: dict[str, object]) -> "ExternalRecoveryRequest":
        ancestor_record = record.get("ancestor_submodule")
        if ancestor_record is not None and not isinstance(ancestor_record, dict):
            raise MutationJournalError(
                "accepted ancestor submodule record is malformed"
            )
        return cls(
            run_id=str(record["run_id"]),
            transaction_id=str(record["transaction_id"]),
            expected_diff_ref=str(record["expected_diff_ref"]),
            removed_linked_worktrees=tuple(
                str(path)
                for path in record["removed_linked_worktrees"]  # type: ignore[index]
            ),
            appended_stash_tip=str(record["appended_stash_tip"]),
            reason=str(record["reason"]),
            ancestor_submodule=(
                AcceptedAncestorSubmodule.from_json(ancestor_record)
                if isinstance(ancestor_record, dict)
                else None
            ),
        ).normalized()


@dataclass(frozen=True)
class ExternalRecoveryReport:
    status: str
    run_id: str
    transaction_id: str
    source_root: str
    removed_linked_worktrees: tuple[str, ...]
    removed_peer_transactions: tuple[str, ...]
    baseline_stash_tip: str
    appended_stash_tip: str
    object_addition_oids: tuple[str, ...]
    expected_diff_ref: str
    ancestor_transaction_id: str | None = None
    ancestor_root: str | None = None
    added_submodule_path: str | None = None
    ancestor_gitmodules_blob_oid: str | None = None
    audit_path: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "transaction_id": self.transaction_id,
            "source_root": self.source_root,
            "removed_linked_worktrees": list(self.removed_linked_worktrees),
            "removed_peer_transactions": list(self.removed_peer_transactions),
            "baseline_stash_tip": self.baseline_stash_tip,
            "appended_stash_tip": self.appended_stash_tip,
            "object_addition_oids": list(self.object_addition_oids),
            "expected_diff_ref": self.expected_diff_ref,
            "ancestor_transaction_id": self.ancestor_transaction_id,
            "ancestor_root": self.ancestor_root,
            "added_submodule_path": self.added_submodule_path,
            "ancestor_gitmodules_blob_oid": self.ancestor_gitmodules_blob_oid,
            "audit_path": self.audit_path,
        }


@dataclass(frozen=True)
class _AcceptedAncestorInspection:
    request: AcceptedAncestorSubmodule
    journal: WorkspaceMutationJournal
    index_bytes: bytes
    index_views: tuple[bytes, bytes, bytes]
    gitmodules_bytes: bytes
    gitmodules_blob_oid: str
    config_bytes: bytes
    guard_status: bytes
    object_inventory: dict[str, dict[str, object]]
    module_admin_roots: tuple[str, ...]
    module_namespace_topology: dict[str, dict[str, int | str]]


@dataclass(frozen=True)
class _Inspection:
    request: ExternalRecoveryRequest
    journal: WorkspaceMutationJournal
    family: tuple[WorkspaceMutationJournal, ...]
    removed_peers: tuple[WorkspaceMutationJournal, ...]
    baseline_stash_tip: str
    baseline_stash_log: bytes
    current_stash_log: bytes
    appended_reflog: bytes
    object_inventory: dict[str, dict[str, object]]
    object_addition_oids: tuple[str, ...]
    current_registered_worktrees: tuple[str, ...]
    removed_admin_names: tuple[str, ...]
    accepted_ancestor: _AcceptedAncestorInspection | None


def _run_directory(journal: WorkspaceMutationJournal) -> Path:
    for parent in (journal.journal_dir, *journal.journal_dir.parents):
        if parent.parent.name == "runs":
            metadata = parent.lstat()
            runs_metadata = parent.parent.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(runs_metadata.st_mode)
                or stat.S_ISLNK(runs_metadata.st_mode)
                or parent.resolve() != parent.absolute()
                or parent.parent.resolve() != parent.parent.absolute()
            ):
                raise MutationJournalError("mutation run directory is aliased")
            return parent
    raise MutationJournalError("mutation journal is not beneath a run directory")


def _reserved_directory(
    run_dir: Path,
    name: str,
    *,
    create: bool,
) -> Path | None:
    if name not in {"journal_recovery", _TOMBSTONE_DIRECTORY}:
        raise MutationJournalError("unknown journal recovery reserved directory")
    path = run_dir / name
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if not create:
            return None
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or path.resolve() != path.absolute()
        or path.parent.resolve() != run_dir.resolve()
    ):
        raise MutationJournalError(f"journal recovery directory is aliased: {path}")
    return path


def _assert_safe_output_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise MutationJournalError(f"journal recovery output is aliased: {path}")


def _read_bound_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    max_bytes: int = _MAX_RECOVERY_EVENT_BYTES,
    required: bool = True,
) -> bytes | None:
    """Read one stable, singly-linked regular entry through a held directory."""

    descriptor: int | None = None
    try:
        try:
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            if required:
                raise MutationJournalError(f"{label} is missing") from exc
            return None
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > max_bytes
        ):
            raise MutationJournalError(
                f"{label} is unsafe or exceeds its byte ceiling"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise MutationJournalError(f"{label} changed while opening")
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1))
            if not chunk:
                break
            if len(payload) + len(chunk) > max_bytes:
                raise MutationJournalError(f"{label} exceeds its byte ceiling")
            payload.extend(chunk)
        finished = os.fstat(descriptor)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(opened, field) != getattr(finished, field)
            or getattr(opened, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise MutationJournalError(f"{label} changed during read")
        return bytes(payload)
    except MutationJournalError:
        raise
    except OSError as exc:
        raise MutationJournalError(f"cannot read {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_stable_regular_path(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read a bounded file while holding and re-proving its exact parent."""

    parent_fd, parent_opened = _open_authority_directory(
        path.parent,
        label=f"{label} parent",
    )
    try:
        payload = _read_bound_regular_at(
            parent_fd,
            path.name,
            label=label,
            max_bytes=max_bytes,
        )
        assert payload is not None
        _assert_authority_directory_binding(
            path.parent,
            parent_fd,
            parent_opened,
            label=f"{label} parent",
        )
        return payload
    finally:
        os.close(parent_fd)


def _hash_bound_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    expected_size: int,
) -> str:
    """Hash one stable bound file without buffering it in memory."""

    if (
        expected_size < 0
        or expected_size > _MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_BYTES
    ):
        raise MutationJournalError(f"{label} exceeds its byte ceiling")
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
        ):
            raise MutationJournalError(f"{label} is unsafe or changed size")
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise MutationJournalError(f"{label} changed while opening")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > expected_size:
                raise MutationJournalError(f"{label} grew during read")
            digest.update(chunk)
        finished = os.fstat(descriptor)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if observed != expected_size or any(
            getattr(opened, field) != getattr(finished, field)
            or getattr(opened, field) != getattr(after, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        ):
            raise MutationJournalError(f"{label} changed during read")
        return digest.hexdigest()
    except MutationJournalError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(f"cannot hash {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _find_journal(
    corpus: "Corpus",
    source_root: Path,
    request: ExternalRecoveryRequest,
) -> WorkspaceMutationJournal:
    journals = discover_mutation_journals(corpus, run_id=request.run_id)
    for candidate in journals:
        with _bound_mutation_journal_directory(candidate):
            pass
    matches = [
        candidate
        for candidate in journals
        if candidate.transaction_id == request.transaction_id
    ]
    if len(matches) != 1:
        raise MutationJournalError(
            f"run {request.run_id!r} does not contain exactly one top-level "
            f"transaction {request.transaction_id}"
        )
    journal = matches[0]
    journal._external_recovery_inventory = journals
    return _validate_recovery_root(
        corpus,
        source_root,
        request,
        journal,
    )


def _validate_recovery_root(
    corpus: "Corpus",
    source_root: Path,
    request: ExternalRecoveryRequest,
    journal: WorkspaceMutationJournal,
) -> WorkspaceMutationJournal:
    """Revalidate the exact root selected by one recovery authority path."""

    try:
        outcome_record = json.loads(
            read_corpus_run_control_bounded(
                corpus,
                run_id=request.run_id,
                name="outcome.json",
            )
        )
        outcome = WorkflowOutcome.from_mapping(outcome_record)
    except (
        MutationJournalError,
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise MutationJournalError(
            f"run {request.run_id!r} has no valid terminal outcome"
        ) from exc
    if outcome.run_id != request.run_id:
        raise MutationJournalError("terminal outcome run binding changed")
    if journal.transaction_id != request.transaction_id:
        raise MutationJournalError("recovery root transaction binding changed")
    if journal.parent_transaction_id is not None:
        raise MutationJournalError("external recovery must target a top-level journal")
    if Path(journal._manifest["cwd"]).resolve() != source_root.resolve():
        raise MutationJournalError(
            "transaction cwd does not match resolved Spec source"
        )
    if _run_directory(journal).name != request.run_id:
        raise MutationJournalError("transaction run binding changed")
    if journal.state not in {
        "invoking",
        "external_recovery_prepared",
        "external_recovery_verified",
    }:
        raise MutationJournalError(
            f"transaction state {journal.state!r} is not externally recoverable"
        )
    if journal._manifest.get("expected_implementation") is not None:
        raise MutationJournalError(
            "journal has an expected Implementation; external recovery refused"
        )
    return journal


def _startup_resume_journal(
    corpus: "Corpus",
    source_root: Path,
    request: ExternalRecoveryRequest,
    journal: WorkspaceMutationJournal,
    startup_inventory: tuple[WorkspaceMutationJournal, ...],
) -> WorkspaceMutationJournal:
    """Bind resume to the exact family installed by startup discovery."""

    installed = getattr(
        _mutation_journal_state._LOCK_STATE,
        "writer_epoch_recovery_members",
        None,
    )
    if not isinstance(installed, dict) or installed.get(
        journal.transaction_id
    ) is not journal:
        raise MutationJournalError(
            "startup recovery root is not the exact installed writer epoch owner"
        )
    startup_by_id: dict[str, WorkspaceMutationJournal] = {}
    for candidate in startup_inventory:
        if (
            not isinstance(candidate, WorkspaceMutationJournal)
            or candidate.transaction_id in startup_by_id
        ):
            raise MutationJournalError(
                "startup recovery inventory is malformed or duplicated"
            )
        startup_by_id[candidate.transaction_id] = candidate
    if set(startup_by_id) != set(installed) or any(
        installed.get(transaction_id) is not candidate
        for transaction_id, candidate in startup_by_id.items()
    ):
        raise MutationJournalError(
            "startup recovery inventory changed after writer epoch installation"
        )

    same_run: list[WorkspaceMutationJournal] = []
    for candidate in startup_inventory:
        if _run_directory(candidate).name != request.run_id:
            continue
        with _bound_mutation_journal_directory(candidate):
            pass
        if _run_directory(candidate).name != request.run_id:
            raise MutationJournalError(
                "startup recovery run binding changed after revalidation"
            )
        same_run.append(candidate)
    if not same_run or journal not in same_run:
        raise MutationJournalError(
            "startup recovery inventory does not contain its exact run root"
        )
    journal._external_recovery_inventory = tuple(same_run)
    _validate_recovery_root(corpus, source_root, request, journal)
    _walk_family(corpus, journal, request.run_id)
    return journal


def _walk_family(
    corpus: "Corpus",
    journal: WorkspaceMutationJournal,
    run_id: str,
) -> tuple[WorkspaceMutationJournal, ...]:
    inventory = getattr(journal, "_external_recovery_inventory", None)
    if inventory is None:
        inventory = discover_mutation_journals(corpus, run_id=run_id)
    root_prefix = journal.journal_dir / "nested_mutations"
    result = [
        candidate
        for candidate in inventory
        if candidate.transaction_id == journal.transaction_id
        or root_prefix in candidate.journal_dir.parents
    ]
    parent_edges = {
        candidate.transaction_id: candidate.parent_transaction_id
        for candidate in result
    }
    parent_edges[journal.transaction_id] = None
    ids = [member.transaction_id for member in result]
    if len(ids) != len(set(ids)):
        raise MutationJournalError("duplicate transaction in mutation family")
    physical_ids = set(ids)
    root_family = tuple(
        str(item) for item in journal._manifest.get("registry_transaction_family", ())
    )
    if len(root_family) != len(set(root_family)) or set(root_family) != physical_ids:
        raise MutationJournalError(
            "durable transaction family does not match nested journals"
        )
    root_parents = journal._manifest.get("registry_family_parents")
    if not isinstance(root_parents, dict) or root_parents != parent_edges:
        raise MutationJournalError(
            "durable transaction family parent map does not match nested journals"
        )
    marker_paths: set[Path] = set()
    for member in result:
        # A transaction unregister refreshes only the lineage components that
        # remain live.  A crash-retained child can therefore carry an older,
        # shrinking view of the same physical family.  The top-level journal
        # and nested directory graph are authoritative; local child views must
        # be internally complete subsets of that graph, never alternate
        # membership or parentage.
        local_family = tuple(
            str(item)
            for item in member._manifest.get("registry_transaction_family", ())
        )
        local_ids = set(local_family)
        if (
            len(local_family) != len(local_ids)
            or member.transaction_id not in local_ids
            or not local_ids.issubset(physical_ids)
        ):
            raise MutationJournalError(
                "transaction family contains an inconsistent local membership view"
            )
        local_parents = member._manifest.get("registry_family_parents")
        if not isinstance(local_parents, dict) or set(local_parents) != local_ids:
            raise MutationJournalError(
                "transaction family contains an incomplete local parent map"
            )
        if any(
            local_parents[transaction_id] != parent_edges[transaction_id]
            for transaction_id in local_ids
        ):
            raise MutationJournalError(
                "transaction family contains an inconsistent local parent map"
            )
        connected: set[str] = set()
        pending = [member.transaction_id]
        while pending:
            transaction_id = pending.pop()
            if transaction_id in connected:
                continue
            connected.add(transaction_id)
            parent_id = parent_edges[transaction_id]
            if parent_id in local_ids:
                pending.append(parent_id)
            pending.extend(
                child_id
                for child_id, candidate_parent in parent_edges.items()
                if child_id in local_ids and candidate_parent == transaction_id
            )
        if connected != local_ids:
            raise MutationJournalError(
                "transaction family contains a disconnected local membership view"
            )
        failed_view = tuple(
            str(item) for item in member._manifest.get("failed_family_members", ())
        )
        failed = set(failed_view)
        if (
            len(failed_view) != len(failed)
            or not failed.issubset(local_ids)
            or not failed.issubset({journal.transaction_id})
        ):
            raise MutationJournalError(
                "transaction family contains a failed descendant"
            )
        if member.transaction_id != journal.transaction_id and (
            member.state != "invoking"
            or not member._manifest.get("nested_gitlink_guard")
            or member._manifest.get("scope")
            or member._manifest.get("expected_implementation") is not None
            or member._manifest.get("expected_lifecycle") is not None
        ):
            raise MutationJournalError(
                "transaction family contains a non-guard child: "
                f"{member.transaction_id}"
            )
        marker = Path(str(member._manifest.get("registry_marker", "")))
        registry = Path(str(member._manifest.get("registry_directory", "")))
        if (
            marker.name != f"{member.transaction_id}.json"
            or marker.parent != registry
            or registry.name != _REGISTRY_DIRECTORY
            or marker in marker_paths
        ):
            raise MutationJournalError("transaction family registry binding changed")
        marker_paths.add(marker)
        registry_metadata = registry.lstat()
        local_marker = member.journal_dir / "registry_marker"
        local_metadata = local_marker.lstat()
        if (
            not stat.S_ISDIR(registry_metadata.st_mode)
            or stat.S_ISLNK(registry_metadata.st_mode)
            or registry.resolve() != registry.absolute()
            or not stat.S_ISREG(local_metadata.st_mode)
            or local_metadata.st_nlink != 1
        ):
            raise MutationJournalError("transaction family registry is aliased")
        authority = member._manifest.get("authority_lock") or {}
        authority_common = Path(str(authority.get("common_dir", "")))
        member_cwd = Path(member._manifest["cwd"])
        if member_cwd.is_dir() and (member_cwd / ".git").exists():
            if _workspace_authority_common_dir(member_cwd) != authority_common:
                raise MutationJournalError(
                    "transaction family authority common dir changed"
                )
        canonical_registry = authority_common / _REGISTRY_DIRECTORY
        if registry != canonical_registry:
            raise MutationJournalError("transaction family registry is non-canonical")
    return tuple(result)


def _status_records(raw: bytes) -> tuple[tuple[str, str], ...]:
    records = raw.split(b"\0")
    parsed: list[tuple[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4:
            raise MutationJournalError("malformed recorded peer status")
        code = record[:2].decode("ascii", errors="strict")
        if "R" in code or "C" in code:
            raise MutationJournalError(
                "rename/copy peer status is unsupported by narrow recovery"
            )
        parsed.append((code, os.fsdecode(record[3:])))
    return tuple(parsed)


def _tree_entry(cwd: Path, treeish: str, path: str) -> tuple[str, str, bytes] | None:
    raw = _git(cwd, "ls-tree", "-z", treeish, "--", path)
    records = [record for record in raw.split(b"\0") if record]
    if not records:
        return None
    if len(records) != 1:
        raise MutationJournalError(f"ambiguous stash tree path {path!r}")
    metadata, separator, encoded_path = records[0].partition(b"\t")
    fields = metadata.split()
    if not separator or len(fields) != 3 or os.fsdecode(encoded_path) != path:
        raise MutationJournalError(f"malformed stash tree entry for {path!r}")
    mode, kind, oid = (field.decode("ascii") for field in fields)
    payload = _git(cwd, "cat-file", "-p", oid) if kind == "blob" else b""
    return mode, oid, payload


def _snapshot_matches_tree(
    repository: Path,
    journal: WorkspaceMutationJournal,
    treeish: str,
    path: str,
) -> None:
    expected = journal._manifest.get("entries", {}).get(path)
    if expected is None:
        raise MutationJournalError(f"dirty peer path lacks a journal snapshot: {path}")
    observed = _tree_entry(repository, treeish, path)
    kind = str(expected["kind"])
    if kind == "missing":
        if observed is not None:
            raise MutationJournalError(f"stash retained deleted peer path {path}")
        return
    if observed is None:
        raise MutationJournalError(f"stash omitted dirty peer path {path}")
    mode, _oid, payload = observed
    if expected.get("xattrs"):
        raise MutationJournalError(f"stash cannot preserve peer xattrs for {path}")
    if kind == "file":
        snapshot = (
            journal.journal_dir / "blobs" / str(expected["sha256"])
        ).read_bytes()
        expected_mode = "100755" if int(expected["mode"]) & 0o111 else "100644"
        if mode != expected_mode or payload != snapshot:
            raise MutationJournalError(f"stash content does not cover peer path {path}")
        return
    if kind == "symlink":
        if mode != "120000" or payload != os.fsencode(str(expected["target"])):
            raise MutationJournalError(f"stash symlink does not cover peer path {path}")
        return
    raise MutationJournalError(f"unsupported dirty peer snapshot kind {kind!r}")


def _index_stage_entries(raw: bytes) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, encoded_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3 or fields[2] != b"0":
            raise MutationJournalError("unsupported staged peer index entry")
        result[os.fsdecode(encoded_path)] = (
            fields[0].decode("ascii"),
            fields[1].decode("ascii"),
        )
    return result


def _tree_delta_paths(root: Path, left: str, right: str) -> set[str]:
    raw = _git(
        root,
        "diff-tree",
        "--no-ext-diff",
        "--no-textconv",
        "--no-commit-id",
        "--name-only",
        "-r",
        "-z",
        left,
        right,
    )
    return {os.fsdecode(path) for path in raw.split(b"\0") if path}


def _tree_paths(root: Path, treeish: str) -> set[str]:
    raw = _git(root, "ls-tree", "-r", "--name-only", "-z", treeish)
    return {os.fsdecode(path) for path in raw.split(b"\0") if path}


def _assert_stash_covers_peer(
    root: Path,
    stash_tip: str,
    dirty_peer: WorkspaceMutationJournal,
    records: tuple[tuple[str, str], ...],
) -> None:
    parents = _git(root, "rev-list", "--parents", "-n", "1", stash_tip).decode().split()
    if not parents or parents[0] != stash_tip:
        raise MutationJournalError("cannot resolve accepted stash commit")
    parent_oids = parents[1:]
    if len(parent_oids) not in {2, 3}:
        raise MutationJournalError("accepted stash has an unsupported parent layout")
    if parent_oids[0] != dirty_peer._manifest["head"]:
        raise MutationJournalError("accepted stash is not based on the dirty peer HEAD")
    staged = _index_stage_entries((dirty_peer.journal_dir / "index_stage").read_bytes())
    tracked_paths = {path for code, path in records if code != "??"}
    staged_paths = {path for code, path in records if code != "??" and code[0] != " "}
    untracked_paths = {path for code, path in records if code == "??"}
    if _tree_delta_paths(root, parent_oids[0], stash_tip) != tracked_paths:
        raise MutationJournalError(
            "accepted stash worktree tree contains changes outside recorded peer status"
        )
    if _tree_delta_paths(root, parent_oids[0], parent_oids[1]) != staged_paths:
        raise MutationJournalError(
            "accepted stash index tree contains changes outside recorded peer status"
        )
    has_untracked = any(code == "??" for code, _path in records)
    if has_untracked != (len(parent_oids) == 3):
        raise MutationJournalError(
            "accepted stash does not preserve peer untracked state"
        )
    if has_untracked and _tree_paths(root, parent_oids[2]) != untracked_paths:
        raise MutationJournalError(
            "accepted stash untracked tree differs from recorded peer status"
        )
    for code, path in records:
        if code == "??":
            _snapshot_matches_tree(root, dirty_peer, parent_oids[2], path)
            continue
        _snapshot_matches_tree(root, dirty_peer, stash_tip, path)
        if code[0] != " ":
            expected_stage = staged.get(path)
            observed = _tree_entry(root, parent_oids[1], path)
            if expected_stage is None:
                if observed is not None:
                    raise MutationJournalError(
                        f"stash index unexpectedly contains {path}"
                    )
            elif observed is None or observed[:2] != expected_stage:
                raise MutationJournalError(
                    f"stash index does not cover staged peer path {path}"
                )


def _stash_log_snapshot(
    journal: WorkspaceMutationJournal,
) -> tuple[bytes, dict[str, object]]:
    record = (journal._manifest.get("log_refs_tree") or {}).get("stash")
    if record is None or record.get("kind") != "file":
        raise MutationJournalError(
            "narrow recovery requires a pre-existing loose stash reflog"
        )
    payload = (
        journal.journal_dir / "git_ref_logs" / str(record["snapshot"])
    ).read_bytes()
    return payload, record


def _assert_stash_append(
    journal: WorkspaceMutationJournal,
    accepted_tip: str,
) -> tuple[str, bytes, bytes, bytes]:
    cwd = journal._bound_workspace_cwd()
    baseline_tip = str((journal._manifest.get("refs") or {}).get("refs/stash", ""))
    if _HEX_OID.fullmatch(baseline_tip) is None:
        raise MutationJournalError("narrow recovery requires a pre-existing stash ref")
    current_refs = _refs_snapshot(cwd)
    expected_refs = dict(journal._manifest["refs"])
    changed = {
        ref
        for ref in set(expected_refs) | set(current_refs)
        if expected_refs.get(ref) != current_refs.get(ref)
    }
    if changed != {"refs/stash"} or current_refs.get("refs/stash") != accepted_tip:
        raise MutationJournalError(
            f"logical refs changed outside the accepted stash append: {sorted(changed)!r}"
        )
    baseline_log, log_record = _stash_log_snapshot(journal)
    current_log_path = _git_control_path(cwd, "logs/refs/stash")
    metadata = current_log_path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != int(log_record["mode"])
        or _xattr_inventory(current_log_path) != log_record.get("xattrs", {})
    ):
        raise MutationJournalError("accepted stash reflog metadata changed")
    current_log = current_log_path.read_bytes()
    if not current_log.startswith(baseline_log):
        raise MutationJournalError(
            "stash reflog is not an append of the journal baseline"
        )
    appended = current_log[len(baseline_log) :]
    if not appended.endswith(b"\n") or appended.count(b"\n") != 1:
        raise MutationJournalError(
            "narrow recovery requires exactly one stash reflog append"
        )
    fields = appended.rstrip(b"\n").split(b" ", 2)
    if len(fields) < 3:
        raise MutationJournalError("malformed appended stash reflog record")
    old_oid = fields[0].decode("ascii", errors="strict")
    new_oid = fields[1].decode("ascii", errors="strict")
    if old_oid != baseline_tip or new_oid != accepted_tip:
        raise MutationJournalError(
            "stash reflog append does not match accepted transition"
        )
    return baseline_tip, baseline_log, current_log, appended


def _loose_object_oid(path: str) -> str | None:
    match = re.fullmatch(r"([0-9a-f]{2})/([0-9a-f]{38}|[0-9a-f]{62})", path)
    return "".join(match.groups()) if match else None


def _assert_object_additions(
    journal: WorkspaceMutationJournal,
    stash_tip: str,
) -> tuple[dict[str, dict[str, object]], tuple[str, ...]]:
    cwd = journal._bound_workspace_cwd()
    expected = journal._manifest.get("object_inventory")
    if expected is None:
        raise MutationJournalError("journal lacks an object-store baseline")
    _root, current = _object_inventory(cwd)
    if not set(expected).issubset(current):
        raise MutationJournalError("pre-existing Git objects disappeared")
    ignored_times = {"mtime_ns", "ctime_ns"}
    for relative, baseline in expected.items():
        observed = current[relative]
        if observed == baseline:
            continue
        baseline_authority = {
            key: value for key, value in baseline.items() if key not in ignored_times
        }
        observed_authority = {
            key: value for key, value in observed.items() if key not in ignored_times
        }
        loose = _loose_object_oid(relative) is not None
        pack = re.fullmatch(
            r"pack/pack-([0-9a-f]{40}|[0-9a-f]{64})\.pack",
            relative,
        )
        indexed_pack = False
        if pack is not None:
            index_path = f"pack/pack-{pack.group(1)}.idx"
            indexed_pack = (
                expected.get(index_path) == current.get(index_path)
                and expected.get(index_path, {}).get("kind") == "file"
                and expected.get(index_path, {}).get("nlink") == 1
            )
        if baseline_authority != observed_authority or not (loose or indexed_pack):
            raise MutationJournalError(
                f"pre-existing Git object changed beyond verified timestamps: {relative}"
            )
    additions = set(current).difference(expected)
    object_oids: set[str] = set()
    for relative in sorted(additions):
        record = current[relative]
        if record.get("kind") == "directory":
            if re.fullmatch(r"[0-9a-f]{2}", relative) is None:
                raise MutationJournalError(
                    f"unsupported added object directory {relative}"
                )
            continue
        oid = _loose_object_oid(relative)
        if (
            oid is None
            or record.get("kind") != "file"
            or record.get("nlink") != 1
            or record.get("xattrs")
        ):
            raise MutationJournalError(f"unsupported Git object addition {relative}")
        object_oids.add(oid)
    if not object_oids:
        raise MutationJournalError("stash append did not create any loose Git objects")
    reachable = {
        line.split(" ", 1)[0]
        for line in _git(cwd, "rev-list", "--objects", stash_tip).decode().splitlines()
        if line
    }
    missing = object_oids.difference(reachable)
    if missing:
        raise MutationJournalError(
            f"added Git objects are not reachable from the accepted stash: {sorted(missing)!r}"
        )
    fsck = subprocess.run(
        ["git", *_GIT_SAFE_OPTIONS, "fsck", "--full", "--no-dangling", stash_tip],
        cwd=cwd,
        env=_sanitized_git_env(GIT_OPTIONAL_LOCKS="0"),
        capture_output=True,
        check=False,
    )
    if fsck.returncode != 0:
        diagnostic = b"\n".join((fsck.stdout, fsck.stderr)).decode(
            "utf-8", errors="replace"
        )
        raise MutationJournalError(
            f"accepted stash failed native Git fsck: {diagnostic[-1000:]}"
        )
    return current, tuple(sorted(object_oids))


def _shadow_children(
    source: Path,
    destination: Path,
    *,
    copied_directories: set[str] | None = None,
) -> None:
    copied = copied_directories or set()
    destination.mkdir(parents=True)
    for child in source.iterdir():
        if child.name in {
            _MANIFEST,
            "nested_mutations",
            _ACCEPTANCE_FILE,
            _RECOVERY_DIRECTORY,
        }:
            continue
        target = destination / child.name
        if child.name == "registry_marker" or child.name in copied:
            if child.is_dir() and not child.is_symlink():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
        else:
            target.symlink_to(child, target_is_directory=child.is_dir())


def _same_durable_workspace_journal_owner(
    expected: object,
    observed: object,
) -> bool:
    """Prove two reconstructed objects bind the same durable journal owner."""

    if expected is observed:
        return True
    if not isinstance(expected, WorkspaceMutationJournal) or not isinstance(
        observed,
        WorkspaceMutationJournal,
    ):
        return False
    try:
        expected_manifest = copy.deepcopy(expected._manifest)
        observed_manifest = copy.deepcopy(observed._manifest)
        return (
            expected.transaction_id == observed.transaction_id
            and expected.parent_transaction_id
            == observed.parent_transaction_id
            and expected.read_writer_epoch_record()
            == observed.read_writer_epoch_record()
            and expected_manifest == observed_manifest
            and expected.journal_dir == observed.journal_dir
            and expected._logical_journal_dir
            == observed._logical_journal_dir
            and expected._logical_journal_identity
            == observed._logical_journal_identity
            and expected._logical_journal_parent_identity
            == observed._logical_journal_parent_identity
            and expected._registry_authority_journal_dir
            == observed._registry_authority_journal_dir
            and expected._discovery_binding == observed._discovery_binding
        )
    except BaseException:
        return False


@contextmanager
def _shadow_journal(
    journal: WorkspaceMutationJournal,
    *,
    manifest: dict[str, object] | None = None,
    copied_directories: set[str] | None = None,
) -> Iterator[WorkspaceMutationJournal]:
    with tempfile.TemporaryDirectory(prefix="crucible-journal-recovery-") as raw:
        root = Path(raw) / "journal"
        copied = {"blobs", *(copied_directories or set())}
        _shadow_children(
            journal.journal_dir,
            root,
            copied_directories=copied,
        )
        rendered = copy.deepcopy(
            manifest if manifest is not None else journal._manifest
        )
        (root / _MANIFEST).write_text(json.dumps(rendered), encoding="utf-8")
        shadow = WorkspaceMutationJournal(root, rendered)
        shadow._registry_authority_journal_dir = (
            journal._registry_authority_journal_dir
        )
        installed = getattr(
            _mutation_journal_state._LOCK_STATE,
            "writer_epoch_recovery_members",
            None,
        )
        installed_owner = (
            installed.get(journal.transaction_id)
            if isinstance(installed, dict)
            else None
        )
        if (
            isinstance(installed, dict)
            and _same_durable_workspace_journal_owner(
                installed_owner,
                journal,
            )
        ):
            replacement = dict(installed)
            replacement[journal.transaction_id] = shadow
            _mutation_journal_state._LOCK_STATE.writer_epoch_recovery_members = (
                replacement
            )
            try:
                yield shadow
            finally:
                _mutation_journal_state._LOCK_STATE.writer_epoch_recovery_members = (
                    installed
                )
        else:
            yield shadow


def _assert_raw_index(journal: WorkspaceMutationJournal) -> None:
    index = journal._bound_manifest_path(str(journal._manifest["index_path"]))
    expected_exists = bool(journal._manifest["index_exists"])
    if index.is_file() != expected_exists:
        raise MutationJournalError(f"slice {journal.slice_id} changed index existence")
    if not expected_exists:
        return
    metadata = index.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != int(journal._manifest["index_mode"])
        or index.read_bytes() != (journal.journal_dir / "index").read_bytes()
    ):
        raise MutationJournalError(f"slice {journal.slice_id} changed raw Git index")
    current_views = _index_semantic_views(journal._bound_workspace_cwd(), index)
    expected_views = tuple(
        (journal.journal_dir / name).read_bytes()
        for name in ("index_stage", "index_flags_v", "index_flags_f")
    )
    if current_views != expected_views:
        raise MutationJournalError(
            f"slice {journal.slice_id} changed Git index semantics"
        )


def _assert_unaffected_journal(
    journal: WorkspaceMutationJournal,
    *,
    ignore_registry_marker: bool = False,
) -> None:
    _assert_raw_index(journal)
    manifest = copy.deepcopy(journal._manifest)
    if ignore_registry_marker:
        manifest["registry_marker"] = None
    with _shadow_journal(journal, manifest=manifest) as shadow:
        shadow.assert_exact_baseline()


def _replace_snapshot_payload(
    shadow_root: Path,
    directory: str,
    record: dict[str, object],
    payload: bytes,
) -> None:
    path = shadow_root / directory / str(record["snapshot"])
    if path.is_symlink():
        path.unlink()
    write_atomic_bytes(path, payload)
    record.update(_capture_snapshot_blob_record(shadow_root, payload))


def _overlay_manifest(
    inspection: _Inspection,
) -> tuple[dict[str, object], set[str]]:
    journal = inspection.journal
    manifest: dict[str, object] = copy.deepcopy(journal._manifest)
    manifest["refs"]["refs/stash"] = inspection.request.appended_stash_tip  # type: ignore[index]
    manifest["registered_worktrees"] = list(inspection.current_registered_worktrees)
    retained_guards = set(manifest.get("guarded_worktree_admin_roots", ())).difference(
        inspection.removed_admin_names
    )
    manifest["guarded_worktree_admin_roots"] = sorted(retained_guards)
    manifest["worktree_namespace_topology"] = _worktree_namespace_topology(
        Path(manifest["cwd"]),
        guarded_admin_roots=retained_guards,
    )
    manifest["object_inventory"] = inspection.object_inventory
    # Source bytes/modes are verified separately against the immutable
    # journal baseline.  Rewriting a scoped file necessarily changes inode
    # timestamps, so exclude only those exact paths from the metadata-only
    # guard while retaining every unrelated tracked-path check.
    tracked_metadata = manifest.get("tracked_path_metadata")
    if isinstance(tracked_metadata, dict):
        for relative in manifest.get("scope", ()):
            tracked_metadata.pop(str(relative), None)
    copied = {"git_refs", "git_ref_logs"}
    admin = manifest.get("git_admin_topology", [])
    common = _git_common_dir(Path(manifest["cwd"]))
    worktrees = common / "worktrees"
    for root_record in admin:  # type: ignore[assignment]
        admin_root = Path(str(root_record["root"]))
        entries = root_record["entries"]
        if admin_root == common and not worktrees.exists():
            entries.pop("worktrees", None)
        for member in inspection.family:
            authority = member._manifest.get("authority_lock") or {}
            authority_path = Path(str(authority.get("path", "")))
            if (
                Path(str(authority.get("common_dir", ""))) == admin_root
                and authority_path.parent == admin_root
                and authority_path.name not in entries
            ):
                metadata = authority_path.lstat()
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_dev != int(authority["device"])
                    or metadata.st_ino != int(authority["inode"])
                    or metadata.st_nlink != int(authority["nlink"])
                    or stat.S_IMODE(metadata.st_mode) != int(authority["mode"])
                    or _xattr_inventory(authority_path) != authority.get("xattrs", {})
                ):
                    raise MutationJournalError(
                        "nested family authority lock changed during recovery"
                    )
                entries[authority_path.name] = {
                    "kind": "authority_lock",
                    "mode": int(authority["mode"]),
                    "device": int(authority["device"]),
                    "inode": int(authority["inode"]),
                    "nlink": int(authority["nlink"]),
                    "xattrs": authority.get("xattrs", {}),
                }
            registry = Path(str(member._manifest.get("registry_directory", "")))
            if registry.parent == admin_root and registry.name not in entries:
                registry_metadata = registry.lstat()
                if (
                    not stat.S_ISDIR(registry_metadata.st_mode)
                    or stat.S_ISLNK(registry_metadata.st_mode)
                    or registry_metadata.st_dev
                    != int(member._manifest["registry_directory_device"])
                    or registry_metadata.st_ino
                    != int(member._manifest["registry_directory_inode"])
                    or stat.S_IMODE(registry_metadata.st_mode)
                    != int(member._manifest["registry_directory_mode"])
                ):
                    raise MutationJournalError(
                        "nested family registry changed during recovery"
                    )
                entries[registry.name] = {
                    "kind": "owned_directory",
                    "device": registry_metadata.st_dev,
                    "inode": registry_metadata.st_ino,
                    "mode": stat.S_IMODE(registry_metadata.st_mode),
                    "xattrs": _xattr_inventory(registry),
                }
        for name in ("logs", "refs"):
            record = entries.get(name)
            if record and record.get("kind") == "guarded_directory":
                copied.add(str(record["snapshot_name"]))
    return manifest, copied


def _assert_parent_overlay(
    inspection: _Inspection,
    *,
    ignore_registry_marker: bool = False,
) -> None:
    journal = inspection.journal
    _assert_raw_index(journal)
    manifest, copied = _overlay_manifest(inspection)
    if ignore_registry_marker:
        manifest["registry_marker"] = None
    with _shadow_journal(
        journal,
        manifest=manifest,
        copied_directories=copied,
    ) as shadow:
        shadow_manifest = shadow._manifest
        refs_record = shadow_manifest["refs_tree"]["stash"]  # type: ignore[index]
        stash_ref = _git_control_path(
            Path(shadow_manifest["cwd"]), "refs/stash"
        ).read_bytes()
        _replace_snapshot_payload(
            shadow.journal_dir, "git_refs", refs_record, stash_ref
        )
        log_record = shadow_manifest["log_refs_tree"]["stash"]  # type: ignore[index]
        _replace_snapshot_payload(
            shadow.journal_dir,
            "git_ref_logs",
            log_record,
            inspection.current_stash_log,
        )
        for root_record in shadow_manifest.get("git_admin_topology", []):
            entries = root_record["entries"]
            refs_admin = entries.get("refs")
            if refs_admin and "stash" in refs_admin.get("entries", {}):
                _replace_snapshot_payload(
                    shadow.journal_dir,
                    str(refs_admin["snapshot_name"]),
                    refs_admin["entries"]["stash"],
                    stash_ref,
                )
            logs_admin = entries.get("logs")
            if logs_admin and "refs/stash" in logs_admin.get("entries", {}):
                _replace_snapshot_payload(
                    shadow.journal_dir,
                    str(logs_admin["snapshot_name"]),
                    logs_admin["entries"]["refs/stash"],
                    inspection.current_stash_log,
                )
        write_atomic_json(shadow.journal_dir / _MANIFEST, shadow._manifest)
        shadow.assert_head_unchanged()
        shadow.assert_repository_metadata_unchanged()
        shadow.assert_index_unchanged()


def _source_matches_expected_diff(
    corpus: "Corpus",
    journal: WorkspaceMutationJournal,
    expected_diff_ref: str,
    *,
    artifact_budget: _RecoveryArtifactBudget | None = None,
) -> None:
    if journal._manifest.get("baseline_dirty"):
        raise MutationJournalError(
            "narrow external recovery requires a clean primary baseline"
        )
    expected = _load_namespaced_artifact(
        corpus,
        expected_diff_ref,
        kind="diffs",
        max_bytes=_MAX_RECOVERY_DIFF_BYTES,
        budget=artifact_budget,
    )
    if not expected:
        raise MutationJournalError("expected recovery diff must be non-empty")
    scope = tuple(str(path) for path in journal._manifest.get("scope", ()))
    if not scope or journal._manifest.get("recursive_scope"):
        raise MutationJournalError(
            "narrow external recovery requires a non-recursive file scope"
        )
    if journal.detected_scope_escapes():
        raise MutationJournalError(
            "primary source contains writes outside the journal scope"
        )
    try:
        captured, _files, base = capture_workspace_delta(
            cwd=str(journal._bound_workspace_cwd()),
            path_scope=scope,
            max_diff_bytes=len(expected),
        )
    except (GitCaptureError, OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            "cannot capture the bounded primary source recovery diff"
        ) from exc
    if base != journal._manifest["head"] or captured != expected:
        raise MutationJournalError(
            "primary source postimage does not match expected diff ref"
        )


def _load_namespaced_artifact(
    corpus: "Corpus",
    reference: str,
    *,
    kind: str,
    max_bytes: int = _MAX_RECOVERY_DIFF_BYTES,
    budget: _RecoveryArtifactBudget | None = None,
) -> bytes:
    """Read one exact CAS artifact through a held, no-follow directory chain."""

    if re.fullmatch(r"sha256:[0-9a-f]{64}", reference) is None:
        raise MutationJournalError(
            f"artifact ref must be an exact sha256 value: {reference!r}"
        )
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", kind) is None:
        raise MutationJournalError(
            f"artifact namespace is non-canonical: {kind!r}"
        )
    if max_bytes < 0 or max_bytes > _MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_BYTES:
        raise MutationJournalError(
            f"artifact byte ceiling is invalid for {kind}: {max_bytes}"
        )
    active_budget = budget or _RecoveryArtifactBudget()
    if (
        isinstance(active_budget.max_bytes, bool)
        or not isinstance(active_budget.max_bytes, int)
        or active_budget.max_bytes < 0
        or isinstance(active_budget.max_artifacts, bool)
        or not isinstance(active_budget.max_artifacts, int)
        or active_budget.max_artifacts < 1
        or isinstance(active_budget.bytes, bool)
        or not isinstance(active_budget.bytes, int)
        or active_budget.bytes < 0
        or active_budget.bytes > active_budget.max_bytes
        or isinstance(active_budget.artifacts, bool)
        or not isinstance(active_budget.artifacts, int)
        or active_budget.artifacts < 0
    ):
        raise MutationJournalError("recovery artifact budget is malformed")
    if active_budget.artifacts >= active_budget.max_artifacts:
        raise MutationJournalError(
            "recovery Corpus artifacts exceed their aggregate count ceiling"
        )
    digest = reference.removeprefix("sha256:")
    spec_root = Path(corpus.spec_dir)
    descriptors: list[int] = []
    try:
        spec_fd, spec_opened = _open_authority_directory(
            spec_root,
            label="Corpus Spec artifact root",
        )
        descriptors.append(spec_fd)
        artifacts_fd, artifacts_opened = _open_authority_directory_at(
            spec_fd,
            "artifacts",
            label="Corpus artifact directory",
        )
        descriptors.append(artifacts_fd)
        kind_fd, kind_opened = _open_authority_directory_at(
            artifacts_fd,
            kind,
            label=f"Corpus artifact namespace {kind!r}",
        )
        descriptors.append(kind_fd)
        payload = _read_bound_regular_at(
            kind_fd,
            f"{digest}.bin",
            label=f"Corpus {kind} artifact {reference}",
            max_bytes=min(max_bytes, active_budget.remaining_bytes),
        )
        if payload is None:  # Required reads are never optional; narrow the type.
            raise MutationJournalError(
                f"Corpus {kind} artifact is missing: {reference}"
            )
        for parent_fd, name, descriptor, expected, label in (
            (
                artifacts_fd,
                kind,
                kind_fd,
                kind_opened,
                f"Corpus artifact namespace {kind!r}",
            ),
            (
                spec_fd,
                "artifacts",
                artifacts_fd,
                artifacts_opened,
                "Corpus artifact directory",
            ),
        ):
            current = os.stat(
                name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            held = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(current.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or not stat.S_ISDIR(held.st_mode)
                or (current.st_dev, current.st_ino)
                != (expected.st_dev, expected.st_ino)
                or (held.st_dev, held.st_ino)
                != (expected.st_dev, expected.st_ino)
            ):
                raise MutationJournalError(
                    f"{label} changed during artifact read"
                )
        _assert_authority_directory_binding(
            spec_root,
            spec_fd,
            spec_opened,
            label="Corpus Spec artifact root",
        )
    except MutationJournalError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot read Corpus {kind} artifact {reference}: {exc}"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise MutationJournalError(
            f"Corpus {kind} artifact failed CAS validation: {reference}"
        )
    active_budget.charge(len(payload), kind=kind)
    return payload


def load_accepted_ancestor_submodule(
    corpus: "Corpus",
    reference: str,
    *,
    _artifact_budget: _RecoveryArtifactBudget | None = None,
) -> AcceptedAncestorSubmodule:
    normalized_ref = reference.strip().lower()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", normalized_ref) is None:
        raise MutationJournalError(
            "ancestor change manifest ref must be an exact sha256 ref"
        )
    payload = _load_namespaced_artifact(
        corpus,
        normalized_ref,
        kind=_ANCESTOR_ACCEPTANCE_ARTIFACT_KIND,
        max_bytes=_MAX_ANCESTOR_ACCEPTANCE_BYTES,
        budget=_artifact_budget,
    )
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MutationJournalError(
            "accepted ancestor change manifest is not valid JSON"
        ) from exc
    expected_fields = {
        "schema",
        "run_id",
        "transaction_id",
        "ancestor_transaction_id",
        "ancestor_root",
        "submodule_path",
        "submodule_url",
        "gitlink_oid",
        "index_ref",
        "gitmodules_ref",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected_fields
        or record.get("schema") != _ANCESTOR_ACCEPTANCE_SCHEMA
        or any(
            not isinstance(record[field], str)
            for field in expected_fields.difference({"schema"})
        )
    ):
        raise MutationJournalError(
            "accepted ancestor change manifest schema or fields differ"
        )
    return AcceptedAncestorSubmodule(
        run_id=str(record["run_id"]),
        transaction_id=str(record["transaction_id"]),
        ancestor_transaction_id=str(record["ancestor_transaction_id"]),
        ancestor_root=str(record["ancestor_root"]),
        submodule_path=str(record["submodule_path"]),
        submodule_url=str(record["submodule_url"]),
        gitlink_oid=str(record["gitlink_oid"]),
        index_ref=str(record["index_ref"]),
        gitmodules_ref=str(record["gitmodules_ref"]),
        acceptance_ref=normalized_ref,
    ).normalized()


def _index_flag_entries(raw: bytes) -> dict[str, str]:
    entries: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            raise MutationJournalError("malformed accepted ancestor index flags")
        tag = record[:1].decode("ascii", errors="strict")
        path = os.fsdecode(record[2:])
        if path in entries:
            raise MutationJournalError("duplicate accepted ancestor index flag path")
        entries[path] = tag
    return entries


def _canonical_submodule_sections(
    accepted: AcceptedAncestorSubmodule,
) -> tuple[bytes, bytes]:
    path = accepted.submodule_path
    url = accepted.submodule_url
    worktree = (
        f'[submodule "{path}"]\n'
        f"\tpath = {path}\n"
        f"\turl = {url}\n"
    ).encode("ascii")
    local = (
        f'[submodule "{path}"]\n'
        f"\turl = {url}\n"
        "\tactive = true\n"
    ).encode("ascii")
    return worktree, local


def _ancestor_config_postimage(
    journal: WorkspaceMutationJournal,
    accepted: AcceptedAncestorSubmodule,
) -> bytes:
    matches = [
        record
        for record in journal._manifest.get("control_files", ())
        if record.get("name") == "config"
    ]
    if len(matches) != 1 or not matches[0].get("exists"):
        raise MutationJournalError("ancestor config control binding is missing")
    record = matches[0]
    config_path = Path(str(record["path"]))
    ancestor_root = Path(str(journal._manifest["cwd"]))
    admin_root = Path(str(journal._manifest.get("git_admin_dir", "")))
    try:
        admin_metadata = admin_root.lstat()
    except FileNotFoundError as exc:
        raise MutationJournalError("ancestor Git admin root is missing") from exc
    if (
        config_path != admin_root / "config"
        or not stat.S_ISDIR(admin_metadata.st_mode)
        or stat.S_ISLNK(admin_metadata.st_mode)
        or admin_root.resolve() != admin_root.absolute()
        or (ancestor_root / ".git").resolve() != admin_root
    ):
        raise MutationJournalError("ancestor config path is not canonically bound")
    baseline = (
        journal.journal_dir / "git_controls" / str(record["snapshot"])
    ).read_bytes()
    metadata = config_path.lstat()
    _worktree_section, config_section = _canonical_submodule_sections(accepted)
    current = config_path.read_bytes()
    if (
        not baseline.endswith(b"\n")
        or current != baseline + config_section
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != int(record["mode"])
        or _xattr_inventory(config_path) != record.get("xattrs", {})
    ):
        raise MutationJournalError(
            "ancestor config is not exactly one canonical submodule append"
        )
    entries = _read_bounded_git_config(config_path)
    if any(
        key.lower() in {"core.hookspath", "core.fsmonitor"}
        for key, _value in entries
    ):
        raise MutationJournalError("accepted ancestor config redirects Git execution")
    return current


def _preflight_accepted_ancestor_config(
    parent: WorkspaceMutationJournal,
    accepted: AcceptedAncestorSubmodule,
) -> None:
    matches = [
        member
        for member in parent._nested_journals()
        if member.transaction_id == accepted.ancestor_transaction_id
        and str(Path(member._manifest["cwd"]).resolve()) == accepted.ancestor_root
        and "::ancestor-superproject" in member.slice_id
    ]
    if len(matches) != 1:
        raise MutationJournalError(
            "accepted ancestor manifest does not bind one direct guard"
        )
    _ancestor_config_postimage(matches[0], accepted)


def _assert_exact_ancestor_object_addition(
    journal: WorkspaceMutationJournal,
    blob_oid: str,
    blob_payload: bytes,
) -> dict[str, dict[str, object]]:
    root = journal._bound_workspace_cwd()
    expected = journal._manifest.get("object_inventory")
    if not isinstance(expected, dict):
        raise MutationJournalError("ancestor journal lacks an object baseline")
    _objects_root, current = _object_inventory(root)
    if any(current.get(path) != record for path, record in expected.items()):
        raise MutationJournalError("pre-existing ancestor Git objects changed")
    relative = f"{blob_oid[:2]}/{blob_oid[2:]}"
    allowed = {relative}
    if blob_oid[:2] not in expected:
        allowed.add(blob_oid[:2])
    additions = set(current).difference(expected)
    if additions != allowed:
        raise MutationJournalError(
            "ancestor object delta is not exactly the accepted .gitmodules blob"
        )
    record = current.get(relative, {})
    if (
        record.get("kind") != "file"
        or record.get("nlink") != 1
        or record.get("xattrs")
    ):
        raise MutationJournalError("accepted ancestor blob object is aliased")
    if _git(root, "cat-file", "-t", blob_oid).strip() != b"blob":
        raise MutationJournalError("accepted ancestor object is not a blob")
    if _git(root, "cat-file", "-p", blob_oid) != blob_payload:
        raise MutationJournalError(
            "accepted ancestor blob payload differs from .gitmodules"
        )
    result = subprocess.run(
        [
            "git",
            *_GIT_SAFE_OPTIONS,
            "fsck",
            "--full",
            "--no-dangling",
            blob_oid,
        ],
        cwd=root,
        env=_sanitized_git_env(GIT_OPTIONAL_LOCKS="0"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        diagnostic = b"\n".join((result.stdout, result.stderr)).decode(
            "utf-8", errors="replace"
        )
        raise MutationJournalError(
            f"accepted ancestor object delta failed Git fsck: {diagnostic[-1000:]}"
        )
    return current


def _read_bounded_git_config(path: Path) -> list[tuple[str, str]]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise MutationJournalError(f"required Git config is missing: {path}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > 4 * 1024 * 1024
    ):
        raise MutationJournalError(f"Git config is aliased or too large: {path}")
    result = subprocess.run(
        [
            "git",
            *_GIT_SAFE_OPTIONS,
            "config",
            "--no-includes",
            "--null",
            "--file",
            str(path),
            "--list",
        ],
        cwd=path.parent,
        env=_sanitized_git_env(GIT_OPTIONAL_LOCKS="0"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise MutationJournalError(
            "cannot parse bounded Git config: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    entries: list[tuple[str, str]] = []
    for encoded in result.stdout.split(b"\0"):
        if not encoded:
            continue
        key, separator, value = encoded.partition(b"\n")
        if not separator:
            raise MutationJournalError(f"malformed Git config record in {path}")
        rendered_key = key.decode("utf-8", errors="strict")
        rendered_value = value.decode("utf-8", errors="surrogateescape")
        lowered = rendered_key.lower()
        if lowered == "include.path" or (
            lowered.startswith("includeif.") and lowered.endswith(".path")
        ):
            raise MutationJournalError("accepted module config contains includes")
        entries.append((rendered_key, rendered_value))
    return entries


def _committed_submodule_bindings(repository: Path) -> dict[str, str]:
    head = _head(repository)
    entry = _tree_entry(repository, head, ".gitmodules")
    path = repository / ".gitmodules"
    if entry is None:
        if path.exists() or path.is_symlink():
            raise MutationJournalError(
                "accepted module has an uncommitted .gitmodules file"
            )
        return {}
    metadata = path.lstat()
    if (
        entry[0] != "100644"
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or path.read_bytes() != entry[2]
    ):
        raise MutationJournalError(
            "accepted module .gitmodules differs from its committed tree"
        )
    sections: dict[str, dict[str, list[str]]] = {}
    for key, value in _read_bounded_git_config(path):
        lowered = key.lower()
        if not lowered.startswith("submodule."):
            raise MutationJournalError(
                "accepted module .gitmodules contains a non-submodule key"
            )
        section, separator, field = key[len("submodule.") :].rpartition(".")
        if not separator or not section or not field:
            raise MutationJournalError(
                "accepted module .gitmodules key is malformed"
            )
        section_fields = sections.setdefault(section, {})
        if field.lower() not in {"path", "url"}:
            continue
        section_fields.setdefault(field.lower(), []).append(value)
    bindings: dict[str, str] = {}
    for fields in sections.values():
        paths = fields.get("path", [])
        urls = fields.get("url", [])
        if len(paths) != 1 or len(urls) != 1:
            raise MutationJournalError(
                "accepted module .gitmodules path/URL binding is ambiguous"
            )
        relative = paths[0]
        if (
            re.fullmatch(r"[A-Za-z0-9._/-]+", relative) is None
            or PurePosixPath(relative).is_absolute()
            or any(
                part in {"", ".", "..", ".git"}
                for part in PurePosixPath(relative).parts
            )
            or relative in bindings
        ):
            raise MutationJournalError(
                "accepted module .gitmodules path binding is non-canonical"
            )
        tree_entry = _tree_entry(repository, head, relative)
        if tree_entry is None or tree_entry[0] != "160000":
            raise MutationJournalError(
                "accepted module .gitmodules path is not a committed gitlink"
            )
        bindings[relative] = urls[0]
    return bindings


def _assert_no_operation_state(repository: Path) -> None:
    forbidden = (
        "index.lock",
        "config.lock",
        "packed-refs.lock",
        "shallow.lock",
        "MERGE_HEAD",
        "MERGE_MSG",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "REBASE_HEAD",
        "BISECT_START",
        "sequencer",
        "rebase-apply",
        "rebase-merge",
    )
    present = [
        name
        for name in forbidden
        if (
            (path := _git_control_path(repository, name)).exists()
            or path.is_symlink()
        )
    ]
    if present:
        raise MutationJournalError(
            f"accepted module has in-progress Git operation state: {present!r}"
        )


def _assert_unaliased_module_admin_controls(admin: Path) -> None:
    entries = 0
    for current, dirs, files in os.walk(
        admin,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        retained: list[str] = []
        for name in dirs:
            child = current_path / name
            if name == "objects":
                continue
            metadata = child.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise MutationJournalError(
                    "accepted module admin controls contain an aliased directory"
                )
            retained.append(name)
            entries += 1
        dirs[:] = retained
        for name in files:
            child = current_path / name
            metadata = child.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise MutationJournalError(
                    "accepted module admin controls contain an aliased file"
                )
            entries += 1
        if entries > 100_000:
            raise MutationJournalError(
                "accepted module admin control namespace is unexpectedly large"
            )


def _assert_module_repository(
    repository: Path,
    *,
    expected_head: str,
    expected_admin: Path,
    expected_url: str | None,
) -> Path:
    metadata = repository.lstat()
    git_entry = repository / ".git"
    git_metadata = git_entry.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or repository.resolve() != repository.absolute()
        or not stat.S_ISREG(git_metadata.st_mode)
        or git_metadata.st_nlink != 1
        or _xattr_inventory(git_entry)
    ):
        raise MutationJournalError("accepted module worktree binding is aliased")
    pointer = git_entry.read_text(
        encoding="utf-8", errors="surrogateescape"
    )
    if not pointer.startswith("gitdir: ") or pointer.count("\n") != 1:
        raise MutationJournalError("accepted module .git indirection is malformed")
    rendered_admin = pointer.removeprefix("gitdir: ").rstrip("\n")
    if (
        not rendered_admin
        or Path(rendered_admin).is_absolute()
        or os.path.normpath(rendered_admin) != rendered_admin
    ):
        raise MutationJournalError("accepted module .git indirection is non-canonical")
    admin = (repository / rendered_admin).resolve()
    admin_metadata = admin.lstat()
    if admin != expected_admin:
        raise MutationJournalError("accepted module Git admin binding is non-canonical")
    config_entries = _read_bounded_git_config(admin / "config")
    lowered_config = [(key.lower(), value) for key, value in config_entries]
    if any(
        key in {"core.hookspath", "core.fsmonitor"}
        for key, _value in lowered_config
    ):
        raise MutationJournalError("accepted module config redirects Git execution")
    if expected_url is not None and [
        value
        for key, value in lowered_config
        if key == "remote.origin.url"
    ] != [expected_url]:
        raise MutationJournalError("accepted module origin URL changed")
    alternates = admin / "objects" / "info" / "alternates"
    if (
        alternates.is_symlink()
        or (alternates.exists() and alternates.read_bytes().strip())
    ):
        raise MutationJournalError("accepted module uses an alternate object store")
    _assert_unaliased_module_admin_controls(admin)
    if (
        (repository / rendered_admin).resolve() != admin
        or not stat.S_ISDIR(admin_metadata.st_mode)
        or stat.S_ISLNK(admin_metadata.st_mode)
        or admin.resolve() != admin.absolute()
        or _git_dir(repository) != admin
        or _git_common_dir(repository) != admin
        or _git_worktree_root(repository) != repository
    ):
        raise MutationJournalError("accepted module Git admin binding changed")
    registered = {Path(path).resolve() for path in _registered_worktrees(repository)}
    if len(registered) != 1 or not registered.issubset({repository, admin}):
        raise MutationJournalError("accepted module has linked worktrees")
    if _head(repository) != expected_head:
        raise MutationJournalError("accepted module HEAD differs from staged gitlink")
    if _git(repository, "cat-file", "-t", expected_head).strip() != b"commit":
        raise MutationJournalError("accepted module gitlink is not a commit")
    index = _index_path(repository)
    views = _index_semantic_views(repository, index)
    if _hidden_index_paths(views[1]) or _hidden_index_paths(views[2]):
        raise MutationJournalError("accepted module index contains hidden flags")
    if (admin / "worktrees").exists() or (admin / "worktrees").is_symlink():
        raise MutationJournalError("accepted module has linked-worktree admin state")
    _assert_no_operation_state(repository)
    hooks = admin / "hooks"
    if hooks.is_dir() and any(
        item.name != "hooks.readme" and not item.name.endswith(".sample")
        for item in hooks.iterdir()
    ):
        raise MutationJournalError("accepted module contains active repository hooks")
    _objects, inventory = _object_inventory(repository)
    invalid_objects = [
        path
        for path, record in inventory.items()
        if record.get("kind") not in {"file", "directory"}
        or (
            record.get("kind") == "file"
            and (record.get("nlink") != 1 or record.get("xattrs"))
        )
    ]
    if invalid_objects:
        raise MutationJournalError(
            f"accepted module object store is aliased: {invalid_objects[:10]!r}"
        )
    fsck = subprocess.run(
        ["git", *_GIT_SAFE_OPTIONS, "fsck", "--full", "--no-dangling"],
        cwd=repository,
        env=_sanitized_git_env(GIT_OPTIONAL_LOCKS="0"),
        capture_output=True,
        check=False,
    )
    if fsck.returncode != 0:
        diagnostic = b"\n".join((fsck.stdout, fsck.stderr)).decode(
            "utf-8", errors="replace"
        )
        raise MutationJournalError(
            f"accepted module failed Git fsck: {diagnostic[-1000:]}"
        )
    return admin


def _assert_accepted_module_tree(
    ancestor_root: Path,
    accepted: AcceptedAncestorSubmodule,
) -> None:
    module_root = ancestor_root / accepted.submodule_path
    outer_admin = _git_common_dir(ancestor_root) / "modules" / accepted.submodule_path
    repositories: dict[str, str] = {}
    known_admins: set[Path] = set()
    known_git_entries: set[str] = set()

    def visit(relative: str, oid: str, url: str, expected_admin: Path) -> None:
        if len(repositories) >= 256 or relative.count("/") > 32:
            raise MutationJournalError("accepted module descendant graph is too large")
        if relative in repositories:
            raise MutationJournalError("accepted module descendant path is duplicated")
        repository = module_root if not relative else module_root / relative
        repositories[relative] = oid
        known_git_entries.add(
            ".git" if not relative else f"{relative}/.git"
        )
        admin = _assert_module_repository(
            repository,
            expected_head=oid,
            expected_admin=expected_admin,
            expected_url=url,
        )
        if admin in known_admins:
            raise MutationJournalError("accepted module admin roots are not unique")
        known_admins.add(admin)
        for child_path, child_url in sorted(
            _committed_submodule_bindings(repository).items()
        ):
            child_entry = _tree_entry(repository, oid, child_path)
            assert child_entry is not None
            child = repository / child_path
            child_relative = (
                child_path if not relative else f"{relative}/{child_path}"
            )
            git_entry = child / ".git"
            if git_entry.exists() or git_entry.is_symlink():
                visit(
                    child_relative,
                    child_entry[1],
                    child_url,
                    admin / "modules" / child_path,
                )
                continue
            if child.exists() and (
                not child.is_dir()
                or child.is_symlink()
                or any(child.iterdir())
            ):
                raise MutationJournalError(
                    "uninitialized accepted descendant path is occupied"
                )

    visit("", accepted.gitlink_oid, accepted.submodule_url, outer_admin)
    for relative in sorted(repositories, key=lambda path: (path.count("/"), path)):
        repository = module_root if not relative else module_root / relative
        if _nested_clean_status(repository):
            raise MutationJournalError("accepted module or descendant is dirty")
    observed_git_entries: set[str] = set()
    for current, dirs, files in os.walk(
        module_root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        relative_root = (
            "" if current_path == module_root else current_path.relative_to(module_root).as_posix()
        )
        if ".git" in dirs:
            observed_git_entries.add(f"{relative_root}/.git".lstrip("/"))
            dirs.remove(".git")
        if ".git" in files:
            observed_git_entries.add(f"{relative_root}/.git".lstrip("/"))
        dirs[:] = [
            name for name in dirs if not (current_path / name).is_symlink()
        ]
    if observed_git_entries != known_git_entries:
        raise MutationJournalError(
            "accepted module contains an unbound nested Git administration entry"
        )


def _assert_module_namespace_delta(
    journal: WorkspaceMutationJournal,
    accepted: AcceptedAncestorSubmodule,
) -> tuple[tuple[str, ...], dict[str, dict[str, int | str]]]:
    root = journal._bound_workspace_cwd()
    baseline_roots = tuple(journal._manifest.get("module_admin_roots", ()))
    current_roots = _module_admin_roots(root)
    if set(current_roots) != {*baseline_roots, accepted.submodule_path}:
        raise MutationJournalError(
            "ancestor module-admin delta differs from accepted submodule"
        )
    guarded = (
        *journal._manifest.get("initialized_module_admin_roots", ()),
        *journal._manifest.get("cached_module_admin_roots", ()),
        accepted.submodule_path,
    )
    current_topology = _module_namespace_topology(root, guarded_roots=guarded)
    baseline_topology = journal._manifest.get("module_namespace_topology")
    if not isinstance(baseline_topology, dict):
        raise MutationJournalError("ancestor journal lacks module topology")
    parent_paths = {
        parent.as_posix()
        for parent in PurePosixPath(accepted.submodule_path).parents
        if parent.as_posix() not in {"", "."}
    }
    additions = set(current_topology).difference(baseline_topology)
    if not additions.issubset(parent_paths) or any(
        current_topology[path].get("kind") != "directory"
        for path in additions
    ):
        raise MutationJournalError(
            "ancestor module namespace changed outside accepted connective directories"
        )
    if any(
        current_topology.get(path) != record
        for path, record in baseline_topology.items()
    ):
        raise MutationJournalError("pre-existing ancestor module namespace changed")
    return current_roots, current_topology


def _replace_shadow_file(root: Path, name: str, payload: bytes) -> None:
    path = root / name
    if path.is_symlink():
        path.unlink()
    write_atomic_bytes(path, payload)


def _assert_accepted_ancestor_overlay(
    inspection: _AcceptedAncestorInspection,
    *,
    ignore_registry_marker: bool = False,
) -> None:
    journal = inspection.journal
    accepted = inspection.request
    manifest: dict[str, object] = copy.deepcopy(journal._manifest)
    manifest["guard_status"] = inspection.guard_status.hex()
    manifest["object_inventory"] = inspection.object_inventory
    manifest["module_admin_roots"] = list(inspection.module_admin_roots)
    manifest["module_namespace_topology"] = inspection.module_namespace_topology
    manifest["gitlinks"] = sorted(
        {*manifest.get("gitlinks", ()), accepted.submodule_path}
    )
    manifest["initialized_gitlinks"] = sorted(
        {*manifest.get("initialized_gitlinks", ()), accepted.submodule_path}
    )
    manifest["initialized_module_admin_roots"] = sorted(
        {
            *manifest.get("initialized_module_admin_roots", ()),
            accepted.submodule_path,
        }
    )
    if ignore_registry_marker:
        manifest["registry_marker"] = None
    with _shadow_journal(
        journal,
        manifest=manifest,
        copied_directories={"git_controls", "git_admin_topology"},
    ) as shadow:
        shadow_manifest = shadow._manifest
        for name, payload in zip(
            ("index", "index_stage", "index_flags_v", "index_flags_f"),
            (inspection.index_bytes, *inspection.index_views),
            strict=True,
        ):
            _replace_shadow_file(shadow.journal_dir, name, payload)
        shadow_manifest["index_blob"] = _capture_snapshot_blob_record(
            shadow.journal_dir,
            inspection.index_bytes,
        )
        shadow_manifest["index_semantic_blobs"] = {
            name: _capture_snapshot_blob_record(shadow.journal_dir, payload)
            for name, payload in zip(
                ("index_stage", "index_flags_v", "index_flags_f"),
                inspection.index_views,
                strict=True,
            )
        }
        config_path = _git_control_path(
            journal._bound_workspace_cwd(), "config"
        )
        control_matches = [
            record
            for record in shadow_manifest.get("control_files", ())
            if Path(str(record.get("path", ""))) == config_path
        ]
        if len(control_matches) != 1:
            raise MutationJournalError("ancestor config control binding is ambiguous")
        _replace_snapshot_payload(
            shadow.journal_dir,
            "git_controls",
            control_matches[0],
            inspection.config_bytes,
        )
        common_dir = _git_common_dir(journal._bound_workspace_cwd())
        admin_matches = [
            record["entries"]["config"]
            for record in shadow_manifest.get("git_admin_topology", ())
            if Path(str(record["root"])) == common_dir
            and "config" in record["entries"]
        ]
        if len(admin_matches) != 1 or admin_matches[0].get("kind") != "file":
            raise MutationJournalError("ancestor admin config binding is ambiguous")
        _replace_snapshot_payload(
            shadow.journal_dir,
            "git_admin_topology",
            admin_matches[0],
            inspection.config_bytes,
        )
        write_atomic_json(shadow.journal_dir / _MANIFEST, shadow_manifest)
        shadow.assert_exact_baseline()


def _inspect_accepted_ancestor(
    corpus: "Corpus",
    family: tuple[WorkspaceMutationJournal, ...],
    accepted: AcceptedAncestorSubmodule,
    parent: WorkspaceMutationJournal,
    *,
    ignore_registry_marker: bool,
    artifact_budget: _RecoveryArtifactBudget,
    accepted_authority: AcceptedAncestorSubmodule,
) -> _AcceptedAncestorInspection:
    accepted = accepted.normalized()
    if accepted_authority != accepted:
        raise MutationJournalError(
            "accepted ancestor request differs from its transaction manifest"
        )
    if (
        accepted.run_id != _run_directory(parent).name
        or accepted.transaction_id != parent.transaction_id
    ):
        raise MutationJournalError(
            "accepted ancestor manifest is bound to a different run or transaction"
        )
    matches = [
        member
        for member in family[1:]
        if member.parent_transaction_id == parent.transaction_id
        and member.transaction_id == accepted.ancestor_transaction_id
        and str(Path(member._manifest["cwd"]).resolve()) == accepted.ancestor_root
        and "::ancestor-superproject" in member.slice_id
    ]
    if len(matches) != 1:
        raise MutationJournalError(
            "accepted ancestor root does not identify exactly one direct guard"
        )
    journal = matches[0]
    root = Path(accepted.ancestor_root)
    if (
        journal._manifest.get("guard_status") not in {None, ""}
        or journal._manifest.get("baseline_dirty")
        or not journal._manifest.get("external_worktree_guard")
        or not journal._manifest.get("nested_gitlink_guard")
        or journal._manifest.get("scope")
    ):
        raise MutationJournalError(
            "accepted ancestor requires a clean, empty-scope ancestor guard"
        )
    index_bytes = _load_namespaced_artifact(
        corpus,
        accepted.index_ref,
        kind=_ANCESTOR_INDEX_ARTIFACT_KIND,
        max_bytes=_MAX_ANCESTOR_INDEX_BYTES,
        budget=artifact_budget,
    )
    index = _index_path(root)
    index_payload = _read_stable_regular_path(
        index,
        label="accepted ancestor live Git index",
        max_bytes=len(index_bytes),
    )
    index_metadata = index.lstat()
    if (
        not stat.S_ISREG(index_metadata.st_mode)
        or index_metadata.st_nlink != 1
        or stat.S_IMODE(index_metadata.st_mode) != int(journal._manifest["index_mode"])
        or index_payload != index_bytes
    ):
        raise MutationJournalError(
            "ancestor raw index differs from exact accepted artifact"
        )
    index_views = _index_semantic_views(root, index)
    baseline_stage = _index_stage_entries(
        (journal.journal_dir / "index_stage").read_bytes()
    )
    current_stage = _index_stage_entries(index_views[0])
    baseline_modules = baseline_stage.get(".gitmodules")
    current_modules = current_stage.get(".gitmodules")
    if baseline_modules is None or current_modules is None:
        raise MutationJournalError("ancestor .gitmodules index entry is missing")
    expected_stage = dict(baseline_stage)
    expected_stage[".gitmodules"] = current_modules
    expected_stage[accepted.submodule_path] = ("160000", accepted.gitlink_oid)
    if (
        current_modules[0] != baseline_modules[0]
        or current_modules[1] == baseline_modules[1]
        or accepted.submodule_path in baseline_stage
        or current_stage != expected_stage
    ):
        raise MutationJournalError(
            "ancestor staged delta is not exactly .gitmodules plus one gitlink"
        )
    for baseline_name, current_view in zip(
        ("index_flags_v", "index_flags_f"),
        index_views[1:],
        strict=True,
    ):
        baseline_flags = _index_flag_entries(
            (journal.journal_dir / baseline_name).read_bytes()
        )
        expected_flags = dict(baseline_flags)
        expected_flags[accepted.submodule_path] = "H"
        if _index_flag_entries(current_view) != expected_flags:
            raise MutationJournalError(
                "ancestor index flags differ outside accepted normal gitlink"
            )
    gitmodules_bytes = _load_namespaced_artifact(
        corpus,
        accepted.gitmodules_ref,
        kind=_ANCESTOR_GITMODULES_ARTIFACT_KIND,
        max_bytes=_MAX_ANCESTOR_GITMODULES_BYTES,
        budget=artifact_budget,
    )
    gitmodules_payload = _read_stable_regular_path(
        root / ".gitmodules",
        label="accepted ancestor live .gitmodules",
        max_bytes=len(gitmodules_bytes),
    )
    if (
        gitmodules_payload != gitmodules_bytes
        or _git(root, "cat-file", "-p", current_modules[1]) != gitmodules_bytes
    ):
        raise MutationJournalError(
            "ancestor .gitmodules differs from accepted worktree/index artifact"
        )
    baseline_tree = _tree_entry(
        root, str(journal._manifest["head"]), ".gitmodules"
    )
    if (
        baseline_tree is None
        or baseline_tree[:2] != baseline_modules
        or not baseline_tree[2].endswith(b"\n")
    ):
        raise MutationJournalError("ancestor .gitmodules baseline is not canonical")
    worktree_section, _config_section = _canonical_submodule_sections(accepted)
    if gitmodules_bytes != baseline_tree[2] + worktree_section:
        raise MutationJournalError(
            "ancestor .gitmodules is not exactly one canonical section append"
        )
    config_bytes = _ancestor_config_postimage(journal, accepted)
    guard_status = _nested_clean_status(
        root,
        excluded_roots=journal._manifest.get("opaque_gitlink_roots", ()),
    )
    status = _status_records(guard_status)
    if set(status) != {
        ("M ", ".gitmodules"),
        ("A ", accepted.submodule_path),
    }:
        raise MutationJournalError(
            "ancestor worktree status differs from accepted staged submodule add"
        )
    objects = _assert_exact_ancestor_object_addition(
        journal,
        current_modules[1],
        gitmodules_bytes,
    )
    _assert_accepted_module_tree(root, accepted)
    module_roots, module_topology = _assert_module_namespace_delta(
        journal, accepted
    )
    inspection = _AcceptedAncestorInspection(
        request=accepted,
        journal=journal,
        index_bytes=index_bytes,
        index_views=index_views,
        gitmodules_bytes=gitmodules_bytes,
        gitmodules_blob_oid=current_modules[1],
        config_bytes=config_bytes,
        guard_status=guard_status,
        object_inventory=objects,
        module_admin_roots=module_roots,
        module_namespace_topology=module_topology,
    )
    _assert_accepted_ancestor_overlay(
        inspection,
        ignore_registry_marker=ignore_registry_marker,
    )
    return inspection


def _peer_admin_name(peer: WorkspaceMutationJournal, source_root: Path) -> str:
    git_dir = Path(str(peer._manifest["git_admin_dir"]))
    root = _git_common_dir(source_root) / "worktrees"
    try:
        relative = git_dir.relative_to(root)
    except ValueError as exc:
        raise MutationJournalError(
            "accepted peer is not a linked-worktree guard"
        ) from exc
    if len(relative.parts) != 1:
        raise MutationJournalError("accepted peer admin root is not direct")
    return relative.parts[0]


def _inspect(
    corpus: "Corpus",
    source_root: Path,
    request: ExternalRecoveryRequest,
    *,
    journal: WorkspaceMutationJournal | None = None,
    allow_partially_restored_source: bool = False,
    allow_missing_family_markers: bool = False,
) -> _Inspection:
    request = request.normalized()
    artifact_budget = _RecoveryArtifactBudget()
    if journal is None:
        journal = _find_journal(corpus, source_root, request)
    run_dir = _run_directory(journal)
    _reserved_directory(run_dir, "journal_recovery", create=False)
    _reserved_directory(run_dir, _TOMBSTONE_DIRECTORY, create=False)
    if journal._manifest.get("schema") != _SCHEMA:
        raise MutationJournalError("unsupported retained journal schema")
    if journal._manifest.get("expected_lifecycle") is not None:
        raise MutationJournalError(
            "journal has pending lifecycle acceptance; external recovery refused"
        )
    accepted_authority: AcceptedAncestorSubmodule | None = None
    if request.ancestor_submodule is not None:
        accepted = request.ancestor_submodule
        accepted_authority = load_accepted_ancestor_submodule(
            corpus,
            accepted.acceptance_ref,
            _artifact_budget=artifact_budget,
        )
        if (
            accepted_authority != accepted
            or accepted.run_id != request.run_id
            or accepted.transaction_id != journal.transaction_id
        ):
            raise MutationJournalError(
                "accepted ancestor manifest differs from the requested transaction"
            )
        _preflight_accepted_ancestor_config(journal, accepted)
    family = _walk_family(corpus, journal, request.run_id)
    live = [
        member.transaction_id for member in family if _is_live(member.transaction_id)
    ]
    if live:
        raise MutationJournalError(
            f"mutation transaction family still has a live owner: {live!r}"
        )
    for member in family[1:]:
        if (
            member.state != "invoking"
            or not member._manifest.get("nested_gitlink_guard")
            or member._manifest.get("scope")
            or member._manifest.get("expected_implementation") is not None
            or member._manifest.get("expected_lifecycle") is not None
        ):
            raise MutationJournalError(
                "transaction family contains a non-guard child: "
                f"{member.transaction_id}"
            )
    if request.ancestor_submodule is not None:
        assert accepted_authority is not None
        with _installed_writer_epoch_recovery_members(
            cast(tuple[object, ...], family)
        ):
            accepted_ancestor = _inspect_accepted_ancestor(
                corpus,
                family,
                request.ancestor_submodule,
                journal,
                ignore_registry_marker=allow_missing_family_markers,
                artifact_budget=artifact_budget,
                accepted_authority=accepted_authority,
            )
    else:
        accepted_ancestor = None
    direct_children = {
        str(Path(member._manifest["cwd"]).resolve()): member
        for member in family[1:]
        if member.parent_transaction_id == journal.transaction_id
    }
    requested = set(request.removed_linked_worktrees)
    linked_peer_children = {
        path
        for path, member in direct_children.items()
        if "::linked-peer-worktree::" in member.slice_id
    }
    if requested != linked_peer_children:
        raise MutationJournalError(
            "accepted paths must name every and only direct linked-peer guard"
        )
    if not requested.issubset(direct_children):
        raise MutationJournalError(
            "accepted removed paths are not direct journal children"
        )
    removed_peers = tuple(direct_children[path] for path in sorted(requested))
    if len(removed_peers) != len(requested):
        raise MutationJournalError("accepted removed peer paths are not unique")
    admin_names: list[str] = []
    for peer in removed_peers:
        if (
            not peer._manifest.get("nested_gitlink_guard")
            or not peer._manifest.get("external_worktree_guard")
            or peer._manifest.get("scope")
            or "::linked-peer-worktree::" not in peer.slice_id
        ):
            raise MutationJournalError(
                "only direct empty-scope linked-peer guards may be adopted"
            )
        peer_path = Path(peer._manifest["cwd"])
        if peer_path.exists() or peer_path.is_symlink():
            raise MutationJournalError(
                f"accepted removed peer still exists: {peer_path}"
            )
        admin_name = _peer_admin_name(peer, source_root)
        admin_path = _git_common_dir(source_root) / "worktrees" / admin_name
        if admin_path.exists() or admin_path.is_symlink():
            raise MutationJournalError(
                f"accepted removed peer admin still exists: {admin_path}"
            )
        admin_names.append(admin_name)
    baseline_worktrees = tuple(
        str(path) for path in journal._manifest["registered_worktrees"]
    )
    current_worktrees = _registered_worktrees(source_root)
    removed_observed = {
        str(Path(path).resolve()) for path in baseline_worktrees
    }.difference(str(Path(path).resolve()) for path in current_worktrees)
    if removed_observed != requested or len(current_worktrees) + len(requested) != len(
        baseline_worktrees
    ):
        raise MutationJournalError(
            "registered worktree delta differs from exact acceptance"
        )
    if _head(source_root) != journal._manifest["head"] or _symbolic_head(
        source_root
    ) != journal._manifest.get("head_symbolic_ref"):
        raise MutationJournalError("primary HEAD changed")
    _assert_raw_index(journal)
    baseline_tip, baseline_log, current_log, appended = _assert_stash_append(
        journal,
        request.appended_stash_tip,
    )
    dirty: list[tuple[WorkspaceMutationJournal, tuple[tuple[str, str], ...]]] = []
    for peer in removed_peers:
        status = bytes.fromhex(str(peer._manifest.get("guard_status") or ""))
        records = _status_records(status)
        if records:
            dirty.append((peer, records))
    if len(dirty) != 1:
        raise MutationJournalError(
            "exactly one accepted removed peer must carry dirty state"
        )
    _assert_stash_covers_peer(source_root, request.appended_stash_tip, *dirty[0])
    current_objects, object_oids = _assert_object_additions(
        journal,
        request.appended_stash_tip,
    )
    inspection = _Inspection(
        request=request,
        journal=journal,
        family=family,
        removed_peers=removed_peers,
        baseline_stash_tip=baseline_tip,
        baseline_stash_log=baseline_log,
        current_stash_log=current_log,
        appended_reflog=appended,
        object_inventory=current_objects,
        object_addition_oids=object_oids,
        current_registered_worktrees=current_worktrees,
        removed_admin_names=tuple(sorted(admin_names)),
        accepted_ancestor=accepted_ancestor,
    )
    with _installed_writer_epoch_recovery_members(
        cast(tuple[object, ...], family)
    ):
        _assert_parent_overlay(
            inspection,
            ignore_registry_marker=allow_missing_family_markers,
        )
        removed_ids = {peer.transaction_id for peer in removed_peers}
        accepted_ancestor_id = (
            accepted_ancestor.journal.transaction_id
            if accepted_ancestor is not None
            else None
        )
        for member in family[1:]:
            if (
                member.transaction_id not in removed_ids
                and member.transaction_id != accepted_ancestor_id
            ):
                _assert_unaffected_journal(
                    member,
                    ignore_registry_marker=allow_missing_family_markers,
                )
        _assert_registry_cleanup_safe(
            family,
            allow_missing=allow_missing_family_markers,
        )
        if not allow_partially_restored_source:
            _source_matches_expected_diff(
                corpus,
                journal,
                request.expected_diff_ref,
                artifact_budget=artifact_budget,
            )
    return inspection


def _read_bound_workspace_file(
    root_fd: int,
    relative: str,
    *,
    expected_size: int,
    expected_sha256: str,
) -> bytes:
    """Read one source file through a held root and no-follow parent chain."""

    with _open_settlement_parent_if_present(root_fd, relative) as binding:
        parent_fd, name, parent_chain, missing_index = binding
        if missing_index is not None:
            raise MutationJournalError(
                f"source postimage parent disappeared: {relative}"
            )
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)
                or opened.st_size != expected_size
            ):
                raise MutationJournalError(
                    f"source postimage changed while opening: {relative}"
                )
            payload = bytearray()
            while len(payload) <= expected_size:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, expected_size + 1 - len(payload)),
                )
                if not chunk:
                    break
                payload.extend(chunk)
            finished = os.fstat(descriptor)
            _assert_settlement_parent_binding(
                root_fd,
                relative,
                parent_fd,
                parent_chain,
            )
            after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                len(payload) != expected_size
                or hashlib.sha256(payload).hexdigest() != expected_sha256
                or any(
                    getattr(opened, field) != getattr(finished, field)
                    or getattr(opened, field) != getattr(after, field)
                    for field in (
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_nlink",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                )
            ):
                raise MutationJournalError(
                    f"source postimage changed during read: {relative}"
                )
            return bytes(payload)
        finally:
            os.close(descriptor)


def _capture_postimage(
    journal: WorkspaceMutationJournal,
    journal_fd: int,
) -> dict[str, dict[str, object]]:
    authority = _assert_workspace_mutation_authority(
        Path(journal._manifest["cwd"])
    )
    try:
        os.mkdir(_RECOVERY_DIRECTORY, mode=0o700, dir_fd=journal_fd)
    except FileExistsError:
        pass
    recovery_fd, recovery_opened = _open_authority_directory_at(
        journal_fd,
        _RECOVERY_DIRECTORY,
        label="external recovery artifact directory",
    )
    postimage_fd: int | None = None
    try:
        try:
            postimage_metadata = os.stat(
                "postimage",
                dir_fd=recovery_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            postimage_fd, postimage_opened = _open_authority_directory_at(
                recovery_fd,
                "postimage",
                label="external recovery postimage directory",
            )
            _remove_bound_directory_tree(
                recovery_fd,
                "postimage",
                postimage_fd,
                postimage_opened,
            )
            os.close(postimage_fd)
            postimage_fd = None
            if not stat.S_ISDIR(postimage_metadata.st_mode):
                raise MutationJournalError(
                    "external recovery postimage is not a directory"
                )
        os.mkdir("postimage", mode=0o700, dir_fd=recovery_fd)
        postimage_fd, postimage_opened = _open_authority_directory_at(
            recovery_fd,
            "postimage",
            label="external recovery postimage directory",
        )
        records: dict[str, dict[str, object]] = {}
        budget = _SettlementCaptureBudget()
        expected_snapshots: set[str] = set()
        for index, rendered in enumerate(journal._manifest.get("scope", ())):
            relative = str(rendered)
            budget.add_path(relative)
            state = _current_postimage_state(
                authority.root_descriptor,
                relative,
                budget,
            )
            kind = state.get("kind")
            if kind == "missing":
                record: dict[str, object] = {"kind": "missing"}
            elif kind == "symlink":
                record = {
                    "kind": "symlink",
                    "target": state["target"],
                    "mode": state["mode"],
                    "xattrs": state["xattrs"],
                }
            elif kind == "file":
                snapshot = str(index)
                size = int(state["size"])
                digest = str(state["sha256"])
                payload = _read_bound_workspace_file(
                    authority.root_descriptor,
                    relative,
                    expected_size=size,
                    expected_sha256=digest,
                )
                write_atomic_bytes(
                    Path(f"/proc/self/fd/{postimage_fd}") / snapshot,
                    payload,
                )
                if _read_bound_regular_at(
                    postimage_fd,
                    snapshot,
                    label=f"recovery postimage snapshot {snapshot}",
                    max_bytes=max(size, 1),
                ) != payload:
                    raise MutationJournalError(
                        f"recovery postimage snapshot changed: {snapshot}"
                    )
                expected_snapshots.add(snapshot)
                record = {
                    "kind": "file",
                    "mode": state["mode"],
                    "snapshot": snapshot,
                    "size": size,
                    "sha256": digest,
                    "xattrs": state["xattrs"],
                }
            else:
                raise MutationJournalError(
                    f"unsupported source postimage type: {relative}"
                )
            records[relative] = record
        if set(_bounded_directory_names(
            postimage_fd,
            label="external recovery postimage directory",
            max_entries=len(expected_snapshots) + 1,
        )) != expected_snapshots:
            raise MutationJournalError(
                "external recovery postimage contains unreferenced snapshots"
            )
        current_postimage = os.stat(
            "postimage",
            dir_fd=recovery_fd,
            follow_symlinks=False,
        )
        current_recovery = os.stat(
            _RECOVERY_DIRECTORY,
            dir_fd=journal_fd,
            follow_symlinks=False,
        )
        if (
            (current_postimage.st_dev, current_postimage.st_ino)
            != (postimage_opened.st_dev, postimage_opened.st_ino)
            or (current_recovery.st_dev, current_recovery.st_ino)
            != (recovery_opened.st_dev, recovery_opened.st_ino)
        ):
            raise MutationJournalError(
                "external recovery postimage path changed during capture"
            )
        return records
    finally:
        if postimage_fd is not None:
            os.close(postimage_fd)
        os.close(recovery_fd)


@contextmanager
def _bound_recovery_postimage_directory(
    journal: WorkspaceMutationJournal,
    *,
    journal_fd: int | None = None,
) -> Iterator[int]:
    """Hold the exact recovery/postimage directory chain while it is used."""

    with ExitStack() as stack:
        if journal_fd is None:
            bound = stack.enter_context(_bound_mutation_journal_directory(journal))
            journal_fd = bound[1]
        recovery_fd, recovery_opened = _open_authority_directory_at(
            journal_fd,
            _RECOVERY_DIRECTORY,
            label="external recovery artifact directory",
        )
        postimage_fd: int | None = None
        try:
            postimage_fd, postimage_opened = _open_authority_directory_at(
                recovery_fd,
                "postimage",
                label="external recovery postimage directory",
            )
            yield postimage_fd
            held_postimage = os.fstat(postimage_fd)
            held_recovery = os.fstat(recovery_fd)
            current_postimage = os.stat(
                "postimage",
                dir_fd=recovery_fd,
                follow_symlinks=False,
            )
            current_recovery = os.stat(
                _RECOVERY_DIRECTORY,
                dir_fd=journal_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(held_postimage.st_mode)
                or not stat.S_ISDIR(held_recovery.st_mode)
                or (held_postimage.st_dev, held_postimage.st_ino)
                != (postimage_opened.st_dev, postimage_opened.st_ino)
                or (current_postimage.st_dev, current_postimage.st_ino)
                != (postimage_opened.st_dev, postimage_opened.st_ino)
                or (held_recovery.st_dev, held_recovery.st_ino)
                != (recovery_opened.st_dev, recovery_opened.st_ino)
                or (current_recovery.st_dev, current_recovery.st_ino)
                != (recovery_opened.st_dev, recovery_opened.st_ino)
            ):
                raise MutationJournalError(
                    "external recovery postimage directory changed during use"
                )
        except MutationJournalError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise MutationJournalError(
                f"cannot revalidate external recovery postimage: {exc}"
            ) from exc
        finally:
            if postimage_fd is not None:
                os.close(postimage_fd)
            os.close(recovery_fd)


def _validated_postimage_records(
    journal: WorkspaceMutationJournal,
    event: dict[str, object],
    *,
    journal_fd: int | None = None,
    postimage_fd: int | None = None,
) -> dict[str, dict[str, object]]:
    if postimage_fd is None:
        with _bound_recovery_postimage_directory(
            journal,
            journal_fd=journal_fd,
        ) as bound_postimage_fd:
            return _validated_postimage_records(
                journal,
                event,
                postimage_fd=bound_postimage_fd,
            )
    raw_records = event.get("postimage_records")
    if not isinstance(raw_records, dict):
        raise MutationJournalError("recovery postimage records are missing")
    scope = tuple(str(path) for path in journal._manifest.get("scope", ()))
    if set(raw_records) != set(scope):
        raise MutationJournalError("recovery postimage scope binding changed")
    records: dict[str, dict[str, object]] = {}
    expected_snapshots: set[str] = set()
    snapshot_bytes = 0
    for index, relative in enumerate(scope):
        raw = raw_records[relative]
        if not isinstance(raw, dict):
            raise MutationJournalError(
                f"recovery postimage record is malformed: {relative}"
            )
        record = {str(key): value for key, value in raw.items()}
        kind = record.get("kind")
        if kind == "missing":
            expected_keys = {"kind"}
        elif kind == "symlink":
            expected_keys = {"kind", "target", "mode", "xattrs"}
            if (
                not isinstance(record.get("target"), str)
                or not isinstance(record.get("mode"), int)
                or not isinstance(record.get("xattrs"), dict)
            ):
                raise MutationJournalError(
                    f"recovery symlink postimage is malformed: {relative}"
                )
        elif kind == "file":
            expected_keys = {
                "kind",
                "mode",
                "snapshot",
                "size",
                "sha256",
                "xattrs",
            }
            snapshot = str(index)
            if (
                record.get("snapshot") != snapshot
                or not isinstance(record.get("mode"), int)
                or not isinstance(record.get("size"), int)
                or int(record["size"]) < 0
                or int(record["size"])
                > _MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_BYTES
                or not isinstance(record.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"]))
                is None
                or not isinstance(record.get("xattrs"), dict)
            ):
                raise MutationJournalError(
                    f"recovery file postimage is malformed: {relative}"
                )
            snapshot_bytes += int(record["size"])
            if (
                snapshot_bytes
                > _MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_BYTES
                or _hash_bound_regular_at(
                    postimage_fd,
                    snapshot,
                    label=f"recovery postimage payload for {relative}",
                    expected_size=int(record["size"]),
                )
                != record["sha256"]
            ):
                raise MutationJournalError(
                    f"recovery postimage payload changed: {relative}"
                )
            expected_snapshots.add(snapshot)
        else:
            raise MutationJournalError(
                f"unsupported recovery postimage kind for {relative}: {kind!r}"
            )
        if set(record) != expected_keys:
            raise MutationJournalError(
                f"recovery postimage fields changed: {relative}"
            )
        records[relative] = record
    observed_snapshots = set(_bounded_directory_names(
        postimage_fd,
        label="external recovery postimage directory",
        max_entries=len(expected_snapshots) + 1,
    ))
    if observed_snapshots != expected_snapshots:
        raise MutationJournalError("recovery postimage payload set changed")
    return records


def _matches_postimage(
    journal: WorkspaceMutationJournal,
    relative: str,
    record: dict[str, object],
    *,
    postimage_fd: int | None = None,
    budget: _SettlementCaptureBudget | None = None,
    state: dict[str, object] | None = None,
) -> bool:
    if postimage_fd is None:
        with _bound_recovery_postimage_directory(journal) as bound_postimage_fd:
            return _matches_postimage(
                journal,
                relative,
                record,
                postimage_fd=bound_postimage_fd,
                budget=budget,
                state=state,
            )
    if budget is None:
        budget = _SettlementCaptureBudget()
    if state is None:
        state = _capture_recovery_source_state(journal, relative, budget)
    kind = record["kind"]
    if state.get("kind") != kind:
        return False
    if kind == "missing":
        return True
    if kind == "symlink":
        return (
            state.get("target") == record["target"]
            and state.get("mode") == record["mode"]
            and state.get("xattrs") == record.get("xattrs", {})
        )
    if kind != "file":
        return False
    if _hash_bound_regular_at(
        postimage_fd,
        str(record["snapshot"]),
        label=f"recovery postimage payload for {relative}",
        expected_size=int(record["size"]),
    ) != record["sha256"]:
        raise MutationJournalError(
            f"recovery postimage payload changed: {relative}"
        )
    return (
        state.get("mode") == record["mode"]
        and state.get("size") == record["size"]
        and state.get("sha256") == record["sha256"]
        and state.get("xattrs") == record.get("xattrs", {})
    )


def _capture_recovery_source_state(
    journal: WorkspaceMutationJournal,
    relative: str,
    budget: _SettlementCaptureBudget,
) -> dict[str, object]:
    """Capture one live source node once for baseline/postimage comparison."""

    budget.add_path(relative)
    authority = _assert_workspace_mutation_authority(
        Path(journal._manifest["cwd"])
    )
    return _current_postimage_state(
        authority.root_descriptor,
        relative,
        budget,
    )


@contextmanager
def _bound_recovery_baseline_blobs(
    journal: WorkspaceMutationJournal,
    *,
    budget: _SettlementCaptureBudget | None = None,
) -> Iterator[dict[str, int]]:
    """Bind and inventory the journal once, then hold its baseline blobs."""

    entries = journal._manifest.get("entries", {})
    if not isinstance(entries, dict):
        raise MutationJournalError("mutation journal baseline entries are malformed")
    expected_blobs: dict[str, int] = {}
    for relative, raw in entries.items():
        if not isinstance(raw, dict) or raw.get("kind") != "file":
            continue
        digest = raw.get("sha256")
        size = raw.get("size")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > _MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_BYTES
            or (digest in expected_blobs and expected_blobs[digest] != size)
        ):
            raise MutationJournalError(
                f"mutation journal baseline file is malformed: {relative}"
            )
        expected_blobs[digest] = size
    with _bound_mutation_journal_directory(journal) as (
        _artifact_dir,
        journal_fd,
        _parent_fd,
        _entry_name,
    ):
        if not expected_blobs:
            yield {}
            return
        blobs_fd, blobs_opened = _open_authority_directory_at(
            journal_fd,
            "blobs",
            label="mutation journal baseline blobs",
        )
        try:
            active_budget = budget or _SettlementCaptureBudget()
            for digest, size in sorted(expected_blobs.items()):
                active_budget.add_hashed_bytes(size)
                if _hash_bound_regular_at(
                    blobs_fd,
                    digest,
                    label=f"mutation baseline payload {digest}",
                    expected_size=size,
                ) != digest:
                    raise MutationJournalError(
                        f"corrupt mutation snapshot blob: {digest}"
                    )
            yield expected_blobs
            current = os.stat(
                "blobs",
                dir_fd=journal_fd,
                follow_symlinks=False,
            )
            held = os.fstat(blobs_fd)
            if (
                not stat.S_ISDIR(current.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or not stat.S_ISDIR(held.st_mode)
                or (current.st_dev, current.st_ino)
                != (blobs_opened.st_dev, blobs_opened.st_ino)
                or (held.st_dev, held.st_ino)
                != (blobs_opened.st_dev, blobs_opened.st_ino)
            ):
                raise MutationJournalError(
                    "mutation journal baseline blob directory changed"
                )
        finally:
            os.close(blobs_fd)


def _matches_baseline(
    journal: WorkspaceMutationJournal,
    relative: str,
    *,
    budget: _SettlementCaptureBudget | None = None,
    state: dict[str, object] | None = None,
    baseline_blobs: dict[str, int] | None = None,
) -> bool:
    """Compare one source path with its baseline through held descriptors."""

    if budget is None:
        budget = _SettlementCaptureBudget()
    authority = _assert_workspace_mutation_authority(
        Path(journal._manifest["cwd"])
    )
    if state is None:
        state = _capture_recovery_source_state(journal, relative, budget)
    entries = journal._manifest.get("entries", {})
    if not isinstance(entries, dict):
        raise MutationJournalError("mutation journal baseline entries are malformed")
    expected = entries.get(relative)
    if expected is not None:
        if not isinstance(expected, dict):
            raise MutationJournalError(
                f"mutation journal baseline entry is malformed: {relative}"
            )
        kind = expected.get("kind")
        if kind not in {
            "missing",
            "directory",
            "symlink",
            "file",
            "opaque_directory",
        }:
            raise MutationJournalError(
                f"mutation journal baseline entry kind is malformed: {relative}"
            )
        xattrs = expected.get("xattrs", {})
        if not isinstance(xattrs, dict) or not all(
            isinstance(name, str)
            and isinstance(value, str)
            and len(value) % 2 == 0
            and re.fullmatch(r"[0-9a-f]*", value) is not None
            for name, value in xattrs.items()
        ):
            raise MutationJournalError(
                f"mutation journal baseline xattrs are malformed: {relative}"
            )
        if kind in {"directory", "symlink", "file"}:
            mode = expected.get("mode")
            if (
                isinstance(mode, bool)
                or not isinstance(mode, int)
                or mode < 0
                or mode > 0o7777
            ):
                raise MutationJournalError(
                    f"mutation journal baseline mode is malformed: {relative}"
                )
        if kind in {"symlink", "file"}:
            nlink = expected.get("nlink", 1)
            if isinstance(nlink, bool) or not isinstance(nlink, int) or nlink != 1:
                raise MutationJournalError(
                    "mutation journal baseline link count is malformed: "
                    f"{relative}"
                )
        if kind == "symlink" and not isinstance(expected.get("target"), str):
            raise MutationJournalError(
                f"mutation journal baseline symlink is malformed: {relative}"
            )
        if kind == "file":
            expected_size_value = expected.get("size")
            expected_digest_value = expected.get("sha256")
            if (
                isinstance(expected_size_value, bool)
                or not isinstance(expected_size_value, int)
                or expected_size_value < 0
                or expected_size_value
                > _MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_BYTES
                or not isinstance(expected_digest_value, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_digest_value) is None
            ):
                raise MutationJournalError(
                    f"mutation journal baseline file is malformed: {relative}"
                )
        if kind == "opaque_directory":
            return state.get("kind") == "directory"
        if state.get("kind") != kind:
            return False
        if kind == "missing":
            return True
        if kind == "directory":
            return (
                state.get("mode") == expected.get("mode")
                and state.get("xattrs") == expected.get("xattrs", {})
            )
        if kind == "symlink":
            return (
                state.get("mode") == expected.get("mode")
                and state.get("nlink") == expected.get("nlink", 1)
                and state.get("target") == expected.get("target")
                and state.get("xattrs") == expected.get("xattrs", {})
            )
        if kind != "file":
            return False
        if (
            state.get("mode") != expected.get("mode")
            or state.get("nlink") != expected.get("nlink", 1)
            or state.get("size") != expected.get("size")
            or state.get("sha256") != expected.get("sha256")
            or state.get("xattrs") != expected.get("xattrs", {})
        ):
            return False
        expected_size = cast(int, expected_size_value)
        expected_digest = cast(str, expected_digest_value)
        if baseline_blobs is None:
            with _bound_recovery_baseline_blobs(
                journal,
                budget=budget,
            ) as bound_blobs:
                return _matches_baseline(
                    journal,
                    relative,
                    budget=budget,
                    state=state,
                    baseline_blobs=bound_blobs,
                )
        if baseline_blobs.get(expected_digest) != expected_size:
            raise MutationJournalError(
                f"corrupt mutation snapshot blob for {relative}"
            )
        return True

    from ._rollback import (  # noqa: PLC0415
        _git_tree_entry,
        _verify_worktree_path_matches_tree,
    )

    tree_entry = _git_tree_entry(
        authority.root_descriptor,
        str(journal._manifest["head"]),
        relative,
    )
    try:
        _verify_worktree_path_matches_tree(
            authority.root_descriptor,
            relative,
            tree_entry,
            tree_label=f"baseline {journal._manifest['head']}",
        )
    except MutationJournalError:
        return False
    return True


def _assert_resumable_source(
    journal: WorkspaceMutationJournal, event: dict[str, object]
) -> None:
    budget = _SettlementCaptureBudget()
    with (
        _bound_recovery_postimage_directory(journal) as postimage_fd,
        _bound_recovery_baseline_blobs(
            journal,
            budget=budget,
        ) as baseline_blobs,
    ):
        postimages = _validated_postimage_records(
            journal,
            event,
            postimage_fd=postimage_fd,
        )
        for relative in journal._manifest.get("scope", ()):
            relative = str(relative)
            state = _capture_recovery_source_state(journal, relative, budget)
            if _matches_baseline(
                journal,
                relative,
                budget=budget,
                state=state,
                baseline_blobs=baseline_blobs,
            ):
                continue
            record = postimages.get(relative)
            if record is None or not _matches_postimage(
                journal,
                relative,
                record,
                postimage_fd=postimage_fd,
                budget=budget,
                state=state,
            ):
                raise MutationJournalError(
                    f"source path changed after recovery acceptance: {relative}"
                )


def _assert_restored_source(journal: WorkspaceMutationJournal) -> None:
    scope = {str(path) for path in journal._manifest.get("scope", ())}
    budget = _SettlementCaptureBudget()
    with _bound_recovery_baseline_blobs(
        journal,
        budget=budget,
    ) as baseline_blobs:
        for relative in sorted(scope):
            state = _capture_recovery_source_state(journal, relative, budget)
            if not _matches_baseline(
                journal,
                relative,
                budget=budget,
                state=state,
                baseline_blobs=baseline_blobs,
            ):
                raise MutationJournalError(
                    f"source path regressed after source restore: {relative}"
                )
    cwd = journal._bound_workspace_cwd()
    journal._assert_workspace_baseline(cwd, roots=scope, full=True)
    _assert_raw_index(journal)


def _assert_source_for_stage(
    journal: WorkspaceMutationJournal,
    event: dict[str, object],
    stage: str,
) -> None:
    if stage == "prepared":
        # A process may die between individual source-path restores.  Only
        # this stage admits the exact accepted postimage/baseline union.
        _assert_resumable_source(journal, event)
        return
    if stage in {"source_restored", "verified", "tombstone_prepared"}:
        # Once restoration is durably acknowledged, every path must remain at
        # baseline.  Reintroduced model bytes must never be accepted merely
        # because they still match the saved pre-restore postimage.
        _assert_restored_source(journal)
        return
    raise MutationJournalError(f"unsupported external recovery stage {stage!r}")


def _all_paths_match_postimage(
    journal: WorkspaceMutationJournal,
    event: dict[str, object],
    *,
    journal_fd: int | None = None,
) -> bool:
    with _bound_recovery_postimage_directory(
        journal,
        journal_fd=journal_fd,
    ) as postimage_fd:
        records = _validated_postimage_records(
            journal,
            event,
            postimage_fd=postimage_fd,
        )
        budget = _SettlementCaptureBudget()
        return all(
            str(relative) in records
            and _matches_postimage(
                journal,
                str(relative),
                records[str(relative)],
                postimage_fd=postimage_fd,
                budget=budget,
            )
            for relative in journal._manifest.get("scope", ())
        )


def _event_path(journal: WorkspaceMutationJournal) -> Path:
    return journal.journal_dir / _ACCEPTANCE_FILE


def _audit_path(
    journal: WorkspaceMutationJournal,
    *,
    create: bool = True,
) -> Path:
    run_dir = _run_directory(journal)
    root = _reserved_directory(run_dir, "journal_recovery", create=create)
    if root is None:
        root = run_dir / "journal_recovery"
    path = root / f"{journal.transaction_id}.json"
    if root.exists():
        _assert_safe_output_file(path)
    return path


def _encode_recovery_event(event: dict[str, object]) -> bytes:
    """Return the one canonical byte representation for recovery events."""

    payload = json.dumps(
        event,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    if len(payload) > _MAX_RECOVERY_EVENT_BYTES:
        raise MutationJournalError(
            "external recovery event exceeds its byte ceiling"
        )
    return payload


def _prepared_control_identity(
    event: dict[str, object],
) -> tuple[int, int, int, int]:
    """Parse the non-circular inode binding carried inside a prepared event."""

    raw = event.get("tombstone_control_identity")
    if not isinstance(raw, dict) or set(raw) != {
        "device",
        "inode",
        "mode",
        "links",
    }:
        raise MutationJournalError(
            "external recovery tombstone control identity is missing or "
            "malformed"
        )
    values = tuple(raw[key] for key in ("device", "inode", "mode", "links"))
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise MutationJournalError(
            "external recovery tombstone control identity is malformed"
        )
    device, inode, mode, links = cast(tuple[int, int, int, int], values)
    if device < 0 or inode <= 0 or mode < 0 or mode > 0o7777 or links != 1:
        raise MutationJournalError(
            "external recovery tombstone control identity is invalid"
        )
    return device, inode, mode, links


def _require_prepared_control_at(
    journal_fd: int,
    event: dict[str, object],
    *,
    name: str = _ACCEPTANCE_FILE,
) -> tuple[tuple[object, ...], bytes]:
    """Hold and authenticate the exact prepared control inode and bytes."""

    expected_identity = _prepared_control_identity(event)
    expected_payload = _prepared_external_event_payload(event)
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=journal_fd, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=journal_fd,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or (
                opened.st_dev,
                opened.st_ino,
                stat.S_IMODE(opened.st_mode),
                opened.st_nlink,
            )
            != expected_identity
            or opened.st_size != len(expected_payload)
        ):
            raise MutationJournalError(
                "external recovery tombstone control identity changed"
            )
        payload = bytearray()
        while len(payload) <= len(expected_payload):
            chunk = os.read(
                descriptor,
                min(1024 * 1024, len(expected_payload) + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        finished = os.fstat(descriptor)
        current = os.stat(name, dir_fd=journal_fd, follow_symlinks=False)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if bytes(payload) != expected_payload or any(
            getattr(opened, field) != getattr(finished, field)
            or getattr(opened, field) != getattr(current, field)
            for field in stable_fields
        ):
            raise MutationJournalError(
                "external recovery tombstone control changed during read"
            )
        return (
            (
                "file",
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_size,
                hashlib.sha256(payload).hexdigest(),
            ),
            bytes(payload),
        )
    except MutationJournalError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot authenticate external recovery tombstone control: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_prepared_control_at(
    journal_fd: int,
    event: dict[str, object],
) -> bytes:
    """Publish a prepared event whose bytes bind their own known temp inode."""

    if "tombstone_control_identity" in event:
        _record, payload = _require_prepared_control_at(journal_fd, event)
        return payload

    temporary = f".tmp_{_ACCEPTANCE_FILE}_prepared_{secrets.token_hex(16)}"
    descriptor: int | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=journal_fd,
        )
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            raise MutationJournalError(
                "external recovery prepared control temporary is unsafe"
            )
        temporary_identity = (created.st_dev, created.st_ino)
        event["tombstone_control_identity"] = {
            "device": created.st_dev,
            "inode": created.st_ino,
            "mode": stat.S_IMODE(created.st_mode),
            "links": created.st_nlink,
        }
        payload = _encode_recovery_event(event)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise MutationJournalError(
                    "external recovery prepared control write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        finished = os.fstat(descriptor)
        if (
            (finished.st_dev, finished.st_ino) != temporary_identity
            or finished.st_nlink != 1
            or finished.st_size != len(payload)
        ):
            raise MutationJournalError(
                "external recovery prepared control changed while writing"
            )
        os.replace(
            temporary,
            _ACCEPTANCE_FILE,
            src_dir_fd=journal_fd,
            dst_dir_fd=journal_fd,
        )
        published = os.stat(
            _ACCEPTANCE_FILE,
            dir_fd=journal_fd,
            follow_symlinks=False,
        )
        if (published.st_dev, published.st_ino) != temporary_identity:
            raise MutationJournalError(
                "external recovery prepared control published a replacement"
            )
        os.fsync(journal_fd)
        _record, observed = _require_prepared_control_at(journal_fd, event)
        if observed != payload:
            raise MutationJournalError(
                "external recovery prepared control differs after publication"
            )
        return payload
    except MutationJournalError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot publish external recovery prepared control: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_identity is not None:
            try:
                residue = os.stat(
                    temporary,
                    dir_fd=journal_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                if (residue.st_dev, residue.st_ino) == temporary_identity:
                    os.unlink(temporary, dir_fd=journal_fd)


def _write_event(journal: WorkspaceMutationJournal, event: dict[str, object]) -> None:
    prepared_control = (
        event.get("stage") == "tombstone_prepared"
        and "tombstone_inventory" in event
    )
    payload = b""
    if journal._discovery_binding is None:
        if prepared_control:
            journal_fd, _opened = _open_authority_directory(
                journal.journal_dir,
                label="external recovery prepared journal",
            )
            try:
                payload = _publish_prepared_control_at(journal_fd, event)
            finally:
                os.close(journal_fd)
        else:
            payload = _encode_recovery_event(event)
            event_path = _event_path(journal)
            _assert_safe_output_file(event_path)
            write_atomic_bytes(event_path, payload)
        audit_path = _audit_path(journal)
        _assert_safe_output_file(audit_path)
        write_atomic_bytes(audit_path, payload)
    else:
        expected_artifacts: dict[str, tuple[str, str | None]] = {}
        if not prepared_control:
            payload = _encode_recovery_event(event)
            expected_artifacts[_ACCEPTANCE_FILE] = (
                "file",
                hashlib.sha256(payload).hexdigest(),
            )
        with refresh_mutation_journal_artifact_binding(
            journal,
            expected_artifacts=expected_artifacts,
        ) as (artifact_dir, journal_fd):
            assert journal_fd is not None
            if prepared_control:
                payload = _publish_prepared_control_at(journal_fd, event)
                expected_artifacts[_ACCEPTANCE_FILE] = (
                    "file",
                    hashlib.sha256(payload).hexdigest(),
                )
            else:
                event_path = artifact_dir / _ACCEPTANCE_FILE
                _assert_safe_output_file(event_path)
                write_atomic_bytes(event_path, payload)
                if _read_bound_regular_at(
                    journal_fd,
                    _ACCEPTANCE_FILE,
                    label="external recovery event",
                ) != payload:
                    raise MutationJournalError(
                        "external recovery event differs from its authored "
                        "payload"
                    )
            with _bound_mutation_journal_run_directory(journal) as (
                _run_alias,
                run_fd,
            ):
                try:
                    os.mkdir("journal_recovery", mode=0o700, dir_fd=run_fd)
                except FileExistsError:
                    pass
                audit_root_fd, audit_root_opened = _open_authority_directory_at(
                    run_fd,
                    "journal_recovery",
                    label="external recovery audit directory",
                )
                try:
                    audit_name = f"{journal.transaction_id}.json"
                    _read_bound_regular_at(
                        audit_root_fd,
                        audit_name,
                        label="external recovery audit",
                        required=False,
                    )
                    write_atomic_bytes(
                        Path(f"/proc/self/fd/{audit_root_fd}") / audit_name,
                        payload,
                    )
                    if _read_bound_regular_at(
                        audit_root_fd,
                        audit_name,
                        label="external recovery audit",
                    ) != payload:
                        raise MutationJournalError(
                            "external recovery audit differs from its authored "
                            "payload"
                        )
                    current_root = os.stat(
                        "journal_recovery",
                        dir_fd=run_fd,
                        follow_symlinks=False,
                    )
                    held_root = os.fstat(audit_root_fd)
                    if (
                        current_root.st_dev,
                        current_root.st_ino,
                    ) != (
                        audit_root_opened.st_dev,
                        audit_root_opened.st_ino,
                    ) or (
                        held_root.st_dev,
                        held_root.st_ino,
                    ) != (
                        audit_root_opened.st_dev,
                        audit_root_opened.st_ino,
                    ):
                        raise MutationJournalError(
                            "external recovery audit directory changed during "
                            "write"
                        )
                finally:
                    os.close(audit_root_fd)


def _new_event(
    inspection: _Inspection,
    journal_fd: int,
) -> dict[str, object]:
    request = inspection.request
    return {
        "schema": _ACCEPTANCE_SCHEMA,
        "stage": "prepared",
        "request": request.to_json(),
        "parent_slice_id": inspection.journal.slice_id,
        "source_root": str(Path(inspection.journal._manifest["cwd"]).resolve()),
        "journal_dir": str(inspection.journal.journal_dir.resolve()),
        "removed_peer_transactions": [
            peer.transaction_id for peer in inspection.removed_peers
        ],
        "baseline_stash_tip": inspection.baseline_stash_tip,
        "appended_reflog_b64": base64.b64encode(inspection.appended_reflog).decode(
            "ascii"
        ),
        "object_addition_oids": list(inspection.object_addition_oids),
        "accepted_ancestor_transaction_id": (
            inspection.accepted_ancestor.journal.transaction_id
            if inspection.accepted_ancestor is not None
            else None
        ),
        "accepted_ancestor_gitmodules_blob_oid": (
            inspection.accepted_ancestor.gitmodules_blob_oid
            if inspection.accepted_ancestor is not None
            else None
        ),
        "postimage_records": _capture_postimage(
            inspection.journal,
            journal_fd,
        ),
    }


def _load_event(journal: WorkspaceMutationJournal) -> dict[str, object] | None:
    if journal._discovery_binding is None:
        path = _event_path(journal)
        if not path.is_file():
            return None
        _assert_safe_output_file(path)
        payload = path.read_bytes()
        if len(payload) > _MAX_RECOVERY_EVENT_BYTES:
            raise MutationJournalError(
                "external recovery event exceeds its byte ceiling"
            )
    else:
        with _bound_mutation_journal_directory(journal) as (
            _artifact_dir,
            journal_fd,
            _parent_fd,
            _entry_name,
        ):
            try:
                before = os.stat(
                    _ACCEPTANCE_FILE,
                    dir_fd=journal_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return None
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > _MAX_RECOVERY_EVENT_BYTES
            ):
                raise MutationJournalError(
                    "external recovery event is unsafe or exceeds its byte "
                    "ceiling"
                )
            descriptor = os.open(
                _ACCEPTANCE_FILE,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=journal_fd,
            )
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)
                ):
                    raise MutationJournalError(
                        "external recovery event changed while opening"
                    )
                chunks: list[bytes] = []
                observed = 0
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    observed += len(chunk)
                    if observed > _MAX_RECOVERY_EVENT_BYTES:
                        raise MutationJournalError(
                            "external recovery event exceeds its byte ceiling"
                        )
                    chunks.append(chunk)
                finished = os.fstat(descriptor)
                after = os.stat(
                    _ACCEPTANCE_FILE,
                    dir_fd=journal_fd,
                    follow_symlinks=False,
                )
                stable_fields = (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
                if any(
                    getattr(opened, field) != getattr(finished, field)
                    or getattr(opened, field) != getattr(after, field)
                    for field in stable_fields
                ):
                    raise MutationJournalError(
                        "external recovery event changed during read"
                    )
                payload = b"".join(chunks)
            finally:
                os.close(descriptor)
    record = json.loads(payload)
    if record.get("schema") != _ACCEPTANCE_SCHEMA:
        raise MutationJournalError("unsupported external recovery event")
    return record


def _event_matches_inspection(
    event: dict[str, object], inspection: _Inspection
) -> None:
    if ExternalRecoveryRequest.from_json(event["request"]) != inspection.request:  # type: ignore[arg-type]
        raise MutationJournalError("recovery request differs from durable acceptance")
    journal = inspection.journal
    if event.get("parent_slice_id") != journal.slice_id:
        raise MutationJournalError("recovery parent slice binding changed")
    if event.get("source_root") != str(Path(journal._manifest["cwd"]).resolve()):
        raise MutationJournalError("recovery source-root binding changed")
    if event.get("journal_dir") != str(journal.journal_dir.resolve()):
        raise MutationJournalError("recovery journal-directory binding changed")
    if tuple(event.get("removed_peer_transactions", ())) != tuple(
        peer.transaction_id for peer in inspection.removed_peers
    ):
        raise MutationJournalError("removed peer transaction binding changed")
    if event.get("baseline_stash_tip") != inspection.baseline_stash_tip:
        raise MutationJournalError("baseline stash binding changed")
    if (
        base64.b64decode(str(event.get("appended_reflog_b64", "")))
        != inspection.appended_reflog
    ):
        raise MutationJournalError("accepted stash reflog append changed")
    if tuple(event.get("object_addition_oids", ())) != inspection.object_addition_oids:
        raise MutationJournalError("accepted object additions changed")
    accepted_ancestor = inspection.accepted_ancestor
    if event.get("accepted_ancestor_transaction_id") != (
        accepted_ancestor.journal.transaction_id
        if accepted_ancestor is not None
        else None
    ):
        raise MutationJournalError("accepted ancestor transaction binding changed")
    if event.get("accepted_ancestor_gitmodules_blob_oid") != (
        accepted_ancestor.gitmodules_blob_oid
        if accepted_ancestor is not None
        else None
    ):
        raise MutationJournalError("accepted ancestor blob binding changed")


def _restore_source_only(journal: WorkspaceMutationJournal) -> None:
    if journal._discovery_binding is not None:
        original_dir = journal.journal_dir
        with _bound_mutation_journal_directory(journal) as (
            artifact_dir,
            _journal_fd,
            _parent_fd,
            _entry_name,
        ):
            journal.journal_dir = artifact_dir
            try:
                _restore_source_only_bound(journal)
            finally:
                journal.journal_dir = original_dir
        return
    _restore_source_only_bound(journal)


def _restore_source_only_bound(journal: WorkspaceMutationJournal) -> None:
    _assert_workspace_mutation_authority(Path(journal._manifest["cwd"]))
    cwd = journal._bound_workspace_cwd()
    for relative in journal._manifest.get("scope", ()):
        journal._restore_snapshot_root(cwd, str(relative))
    journal._restore_index(cwd)
    journal._assert_workspace_baseline(
        cwd,
        roots={str(path) for path in journal._manifest.get("scope", ())},
        full=True,
    )
    _assert_raw_index(journal)


@contextmanager
def _bound_family_registry_entries(
    family: tuple[WorkspaceMutationJournal, ...],
    *,
    allow_missing: bool,
) -> Iterator[tuple[dict[str, object], ...]]:
    """Bind every family journal and exact registry directory as one unit."""

    groups: dict[tuple[int, int], dict[str, object]] = {}
    canonical_registry_identities: dict[str, tuple[int, int]] = {}
    with ExitStack() as stack:
        for member in family:
            if member._discovery_binding is None:
                raise MutationJournalError(
                    "external recovery family lacks discovery authority"
                )
            original_dir = member.journal_dir
            artifact_dir, journal_fd, _parent_fd, _entry_name = (
                stack.enter_context(_bound_mutation_journal_directory(member))
            )
            member.journal_dir = artifact_dir
            stack.callback(setattr, member, "journal_dir", original_dir)
            local_payload = _read_bound_regular_at(
                journal_fd,
                "registry_marker",
                label=f"journal {member.transaction_id} registry marker",
            )
            assert local_payload is not None
            marker = member._registry_marker()
            if marker is None:
                raise MutationJournalError(
                    "external recovery family member has no registry marker"
                )
            registry_fd, registry_opened = _open_authority_directory(
                marker.parent,
                label="bound mutation journal registry",
            )
            key = (registry_opened.st_dev, registry_opened.st_ino)
            canonical_registry = str(Path(str(
                member._manifest.get("registry_directory", marker.parent)
            )).absolute())
            previous_identity = canonical_registry_identities.get(
                canonical_registry
            )
            if previous_identity is not None and previous_identity != key:
                os.close(registry_fd)
                raise MutationJournalError(
                    "journal family registry changed between member bindings"
                )
            canonical_registry_identities[canonical_registry] = key
            group = groups.get(key)
            if group is None:
                stack.callback(os.close, registry_fd)
                group = {
                    "path": marker.parent,
                    "fd": registry_fd,
                    "opened": registry_opened,
                    "markers": {},
                }
                groups[key] = group
            else:
                os.close(registry_fd)
            markers = group["markers"]
            assert isinstance(markers, dict)
            if marker.name in markers:
                raise MutationJournalError(
                    "external recovery family has duplicate registry authority"
                )
            markers[marker.name] = (member, local_payload)

        for group in groups.values():
            registry_fd = int(group["fd"])
            markers = group["markers"]
            assert isinstance(markers, dict)
            current = set(_bounded_directory_names(
                registry_fd,
                label="mutation journal registry",
                max_entries=_MUTATION_JOURNAL_MAX_NESTED_MEMBERS,
            ))
            expected = set(markers)
            if not current.issubset(expected) or (
                not allow_missing and current != expected
            ):
                raise MutationJournalError(
                    "journal registry contains unrelated entries"
                )
            for name in current:
                _member, expected_payload = markers[name]
                if _read_bound_regular_at(
                    registry_fd,
                    name,
                    label=f"journal registry marker {name!r}",
                ) != expected_payload:
                    raise MutationJournalError(
                        f"journal registry marker changed: {name}"
                    )
        def revalidate_groups() -> None:
            for group in groups.values():
                path = Path(str(group["path"]))
                opened = group["opened"]
                assert isinstance(opened, os.stat_result)
                current = path.lstat()
                held = os.fstat(int(group["fd"]))
                if (
                    not stat.S_ISDIR(current.st_mode)
                    or stat.S_ISLNK(current.st_mode)
                    or (current.st_dev, current.st_ino)
                    != (opened.st_dev, opened.st_ino)
                    or (held.st_dev, held.st_ino)
                    != (opened.st_dev, opened.st_ino)
                ):
                    raise MutationJournalError(
                        "journal registry changed during family operation"
                    )

        revalidate_groups()
        yield tuple(groups.values())
        revalidate_groups()


def _assert_registry_cleanup_safe(
    family: tuple[WorkspaceMutationJournal, ...],
    *,
    allow_missing: bool,
) -> None:
    with _bound_family_registry_entries(
        family,
        allow_missing=allow_missing,
    ):
        pass


@dataclass(frozen=True, slots=True)
class _CertifiedExternalEpochMember:
    """Immutable storage for a family member covered by one audit proof."""

    transaction_id: str
    parent_transaction_id: str | None
    epoch_record: dict[str, object]

    @property
    def active(self) -> bool:
        return True

    @property
    def cleanup_pending(self) -> bool:
        return True

    def read_writer_epoch_record(self) -> object:
        return json.loads(json.dumps(
            self.epoch_record,
            sort_keys=True,
            separators=(",", ":"),
        ))

    def compare_and_write_writer_epoch_record(
        self,
        expected: object,
        replacement: dict[str, object],
    ) -> object:
        del expected, replacement
        raise MutationJournalError(
            "external recovery epoch certificates are immutable"
        )


@dataclass(frozen=True, slots=True)
class _PreinstalledWriterEpochAttestation:
    """Exact startup owners and proof frozen before their journal is renamed."""

    proof_payload: bytes
    prepared_event_payload: bytes
    journal_identity: tuple[int, int, int]
    deletion_inventory: tuple[tuple[object, ...], ...]
    members: tuple[
        tuple[str, str | None, WorkspaceMutationJournal],
        ...,
    ]


@dataclass(frozen=True, slots=True)
class _ExternalRecoveryAuditBinding:
    """Exact read-only inventory binding for one applicable recovery audit."""

    run_name: str
    audit_name: str
    audit_identity: tuple[int, int, int, int, int, int, int]
    audit_sha256: str
    tombstone_identity: tuple[int, int, int] | None


@dataclass(frozen=True, slots=True)
class _ExternalRecoveryEpochInventory:
    """Complete epoch and audit inventory frozen before destructive replay."""

    members: tuple[_CertifiedExternalEpochMember, ...]
    audits: tuple[_ExternalRecoveryAuditBinding, ...]


def _require_preinstalled_writer_epoch_members(
    members: tuple[object, ...],
) -> None:
    """Bind a second-pass event to the exact globally preflighted family."""

    installed = getattr(
        _mutation_journal_state._LOCK_STATE,
        "writer_epoch_recovery_members",
        None,
    )
    if not isinstance(installed, dict):
        raise MutationJournalError(
            "external recovery writer epoch inventory is not installed"
        )
    for member in members:
        transaction_id = getattr(member, "transaction_id", None)
        parent_transaction_id = getattr(member, "parent_transaction_id", None)
        expected = installed.get(transaction_id)
        if (
            expected is None
            or getattr(expected, "parent_transaction_id", object())
            != parent_transaction_id
            or expected.read_writer_epoch_record()
            != member.read_writer_epoch_record()
        ):
            raise MutationJournalError(
                "external recovery writer epoch differs from the global "
                f"inventory for transaction {transaction_id!r}"
            )


def _prevalidate_installed_writer_epoch_family(
    family: tuple[WorkspaceMutationJournal, ...],
    event: dict[str, object],
    journal_fd: int,
) -> _PreinstalledWriterEpochAttestation:
    """Freeze exact installed epoch authority while every original path exists."""

    family_by_id: dict[str, WorkspaceMutationJournal] = {}
    for member in family:
        if member.transaction_id in family_by_id:
            raise MutationJournalError(
                "startup recovery epoch family contains duplicate transactions"
            )
        family_by_id[member.transaction_id] = member
    roots = tuple(
        member.transaction_id
        for member in family
        if member.parent_transaction_id is None
    )
    if len(roots) != 1:
        raise MutationJournalError(
            "startup recovery epoch family does not have exactly one root"
        )
    proof = _parse_external_writer_epoch_proof(
        event,
        transaction_id=roots[0],
    )
    if set(family_by_id) != {
        transaction_id for transaction_id, _parent in proof.family
    } or {
        transaction_id: member.parent_transaction_id
        for transaction_id, member in family_by_id.items()
    } != dict(proof.family):
        raise MutationJournalError(
            "startup recovery family differs from its durable epoch proof"
        )
    bindings = tuple(
        (
            transaction_id,
            parent_transaction_id,
            family_by_id[transaction_id],
        )
        for transaction_id, parent_transaction_id in proof.family
    )
    if tuple(
        (transaction_id, parent) for transaction_id, parent, _member in bindings
    ) != proof.family:
        # Keep this separate from the map comparison so a future parser that
        # changes canonical ordering cannot silently alter owner installation.
        raise MutationJournalError(
            "startup recovery epoch proof ordering is non-canonical"
        )
    ordered_family = tuple(member for _transaction_id, _parent, member in bindings)
    if roots[0] != proof.transaction_id:
        raise MutationJournalError(
            "startup recovery epoch family root differs from its durable proof"
        )
    if not ordered_family:
        raise MutationJournalError(
            "startup recovery epoch family is empty"
        )
    proof_root = family_by_id[proof.transaction_id]
    if proof_root.parent_transaction_id is not None:
        raise MutationJournalError(
            "startup recovery epoch proof root has a parent"
        )
    journal_metadata = os.fstat(journal_fd)
    if not stat.S_ISDIR(journal_metadata.st_mode) or stat.S_ISLNK(
        journal_metadata.st_mode
    ):
        raise MutationJournalError(
            "startup recovery root inode changed before rename"
        )
    installed = getattr(
        _mutation_journal_state._LOCK_STATE,
        "writer_epoch_recovery_members",
        None,
    )
    if not isinstance(installed, dict):
        raise MutationJournalError(
            "startup recovery writer epoch inventory is not installed"
        )
    with _mutation_journal_state._LIVE_TRANSACTIONS_LOCK:
        if any(
            installed.get(transaction_id) is not member
            or transaction_id in _mutation_journal_state._LIVE_JOURNAL_OBJECTS
            for transaction_id, _parent, member in bindings
        ):
            raise MutationJournalError(
                "startup recovery writer epoch owner changed before rename"
            )
    for _transaction_id, _parent, member in bindings:
        with _bound_mutation_journal_directory(member) as (
            _artifact_dir,
            member_fd,
            _parent_fd,
            _entry_name,
        ):
            if member is proof_root:
                bound_root = os.fstat(member_fd)
                if (
                    bound_root.st_dev,
                    bound_root.st_ino,
                    stat.S_IMODE(bound_root.st_mode),
                ) != (
                    journal_metadata.st_dev,
                    journal_metadata.st_ino,
                    stat.S_IMODE(journal_metadata.st_mode),
                ):
                    raise MutationJournalError(
                        "startup recovery journal fd is not its exact proof root"
                    )
    _require_preinstalled_writer_epoch_members(
        cast(tuple[object, ...], ordered_family)
    )
    try:
        require_writer_epoch_quiescence(
            cast(tuple[WriterEpochStorage, ...], ordered_family),
            allow_never_armed=True,
        )
    except WriterEpochError as exc:
        raise MutationJournalError(
            f"startup recovery writer epoch is not quiescent: {exc}"
        ) from exc
    deletion_inventory = _external_deletion_inventory(
        journal_fd,
        event=event,
    )
    if deletion_inventory != _external_tombstone_inventory(event):
        raise MutationJournalError(
            "startup recovery control inventory changed before rename"
        )
    proof_payload = json.dumps(
        proof.to_record(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _PreinstalledWriterEpochAttestation(
        proof_payload=proof_payload,
        prepared_event_payload=_prepared_external_event_payload(event),
        journal_identity=(
            journal_metadata.st_dev,
            journal_metadata.st_ino,
            stat.S_IMODE(journal_metadata.st_mode),
        ),
        deletion_inventory=deletion_inventory,
        members=bindings,
    )


def _require_preinstalled_writer_epoch_attestation(
    attestation: _PreinstalledWriterEpochAttestation,
    proof: WriterEpochTerminalProof,
    *,
    journal_fd: int,
    event: dict[str, object],
) -> None:
    """Consume only frozen owner identity after the held journal is renamed."""

    proof_payload = json.dumps(
        proof.to_record(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if (
        proof_payload != attestation.proof_payload
        or _prepared_external_event_payload(event)
        != attestation.prepared_event_payload
        or tuple(
            (transaction_id, parent)
            for transaction_id, parent, _member in attestation.members
        )
        != proof.family
    ):
        raise MutationJournalError(
            "startup recovery writer epoch attestation changed after rename"
        )
    journal_metadata = os.fstat(journal_fd)
    if (
        journal_metadata.st_dev,
        journal_metadata.st_ino,
        stat.S_IMODE(journal_metadata.st_mode),
    ) != attestation.journal_identity or _external_deletion_inventory(
        journal_fd,
        event=event,
    ) != attestation.deletion_inventory:
        raise MutationJournalError(
            "startup recovery held inode or control inventory changed after rename"
        )
    installed = getattr(
        _mutation_journal_state._LOCK_STATE,
        "writer_epoch_recovery_members",
        None,
    )
    if not isinstance(installed, dict):
        raise MutationJournalError(
            "startup recovery writer epoch inventory disappeared after rename"
        )
    with _mutation_journal_state._LIVE_TRANSACTIONS_LOCK:
        if any(
            installed.get(transaction_id) is not member
            or transaction_id in _mutation_journal_state._LIVE_JOURNAL_OBJECTS
            for transaction_id, _parent, member in attestation.members
        ):
            raise MutationJournalError(
                "startup recovery writer epoch owner changed after rename"
            )


@contextmanager
def _installed_writer_epoch_recovery_members(
    members: tuple[object, ...],
) -> Iterator[None]:
    """Expose one complete certified family to journal-level epoch checks."""

    previous = getattr(
        _mutation_journal_state._LOCK_STATE,
        "writer_epoch_recovery_members",
        None,
    )
    if previous is not None and not isinstance(previous, dict):
        raise MutationJournalError(
            "external recovery writer epoch thread-local inventory is "
            "malformed"
        )
    combined = dict(previous) if previous is not None else {}

    def same_durable_owner(expected: object, observed: object) -> bool:
        if expected is observed:
            return True
        if type(expected) is not type(observed):
            return False
        if isinstance(expected, _CertifiedExternalEpochMember):
            return expected == observed
        return _same_durable_workspace_journal_owner(expected, observed)

    with _mutation_journal_state._LIVE_TRANSACTIONS_LOCK:
        seen: set[str] = set()
        additions: dict[str, object] = {}
        for member in members:
            transaction_id = getattr(member, "transaction_id", None)
            if not isinstance(transaction_id, str):
                raise MutationJournalError(
                    "external recovery writer epoch member identity is malformed"
                )
            if transaction_id in seen:
                raise MutationJournalError(
                    "external recovery writer epoch duplicates recovery "
                    f"inventory transaction {transaction_id}"
                )
            seen.add(transaction_id)
            if (
                transaction_id
                in _mutation_journal_state._LIVE_JOURNAL_OBJECTS
            ):
                raise MutationJournalError(
                    "external recovery writer epoch duplicates live "
                    f"authority transaction {transaction_id}"
                )
            expected = combined.get(transaction_id)
            if expected is not None:
                if not same_durable_owner(expected, member):
                    raise MutationJournalError(
                        "external recovery writer epoch conflicts with "
                        "preinstalled recovery inventory transaction "
                        f"{transaction_id}"
                    )
                # Retain the already-installed exact authority rather than
                # replacing it with an equivalent reconstruction.
                continue
            additions[transaction_id] = member
        combined.update(additions)
    _mutation_journal_state._LOCK_STATE.writer_epoch_recovery_members = combined
    try:
        yield
    finally:
        if previous is None:
            try:
                del _mutation_journal_state._LOCK_STATE.writer_epoch_recovery_members
            except AttributeError:
                pass
        else:
            _mutation_journal_state._LOCK_STATE.writer_epoch_recovery_members = (
                previous
            )


def _parse_external_writer_epoch_proof(
    event: dict[str, object],
    *,
    transaction_id: str,
) -> WriterEpochTerminalProof:
    """Parse the exact root/family certificate carried by an audit event."""

    try:
        proof = parse_writer_epoch_terminal_proof(
            event.get(_WRITER_EPOCH_PROOF_FIELD)
        )
    except (TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"external recovery writer epoch proof is malformed: {exc}"
        ) from exc
    if (
        proof.transaction_id != transaction_id
        or proof.parent_transaction_id is not None
        or proof.terminal_state != "rolled_back"
        or not proof.family
        or proof.family[0] != (transaction_id, None)
    ):
        raise MutationJournalError(
            "external recovery writer epoch proof binding changed"
        )
    manifest = event.get(_WRITER_EPOCH_MANIFEST_FIELD)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != _SCHEMA
        or manifest.get("transaction_id") != transaction_id
        or manifest.get("parent_transaction_id") is not None
        or manifest.get("state") != "external_recovery_verified"
        or manifest.get("cwd") != event.get("source_root")
    ):
        raise MutationJournalError(
            "external recovery writer epoch manifest binding changed"
        )
    expected = _writer_epoch_manifest_proof(
        manifest,
        terminal_state="rolled_back",
    )
    if proof.to_record() != expected.to_record():
        raise MutationJournalError(
            "external recovery writer epoch proof differs from its manifest"
        )
    if event.get("stage") in {"tombstone_prepared", "complete"}:
        _external_tombstone_inventory(event)
        _prepared_control_identity(event)
    return proof


def _external_tombstone_inventory(
    event: dict[str, object],
) -> tuple[tuple[object, ...], ...]:
    """Parse the exact deletion-only node inventory carried by a v2 audit."""

    raw = event.get("tombstone_inventory")
    if (
        not isinstance(raw, list)
        or len(raw) > _MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_PATHS
    ):
        raise MutationJournalError(
            "external recovery tombstone inventory is missing or malformed"
        )
    records: list[tuple[object, ...]] = []
    for item in raw:
        if not isinstance(item, list) or not item:
            raise MutationJournalError(
                "external recovery tombstone inventory is malformed"
            )
        record = tuple(item)
        if any(isinstance(value, (dict, list)) for value in record):
            raise MutationJournalError(
                "external recovery tombstone inventory contains nested data"
            )
        records.append(record)
    if len(records) != len(set(records)):
        raise MutationJournalError(
            "external recovery tombstone inventory contains duplicates"
        )
    return tuple(sorted(records, key=repr))


def _prepared_external_event_payload(event: dict[str, object]) -> bytes:
    """Derive the exact in-journal control payload from a durable audit.

    The deletion inventory intentionally excludes ``external_recovery.json``
    to avoid making that event contain its own digest.  Before rename both
    named copies are authored by :func:`_write_event`; after rename the outer
    audit advances only ``stage`` to ``complete``.  Reconstructing the prepared
    form therefore gives deletion a separate exact payload authority.
    """

    if event.get("stage") not in {"tombstone_prepared", "complete"}:
        raise MutationJournalError(
            "external recovery event cannot bind a tombstone control file"
        )
    prepared = copy.deepcopy(event)
    prepared["stage"] = "tombstone_prepared"
    return _encode_recovery_event(prepared)


def _prepared_control_inventory_record(
    event: dict[str, object],
) -> tuple[object, ...]:
    """Derive the exact control inode record from its audit-bound event."""

    device, inode, mode, links = _prepared_control_identity(event)
    payload = _prepared_external_event_payload(event)
    return (
        "file",
        device,
        inode,
        stat.S_IFREG | mode,
        links,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def _external_deletion_inventory(
    journal_fd: int,
    *,
    event: dict[str, object] | None = None,
) -> tuple[tuple[object, ...], ...]:
    """Inventory deletable nodes while separately binding the control file."""

    records = list(_terminal_cleanup_journal_inventory(journal_fd))
    if event is None:
        payload = _read_bound_regular_at(
            journal_fd,
            _ACCEPTANCE_FILE,
            label="external recovery tombstone control event",
            max_bytes=_MAX_RECOVERY_EVENT_BYTES,
            required=False,
        )
        if payload is None:
            return tuple(sorted(records, key=repr))
        metadata = os.stat(
            _ACCEPTANCE_FILE,
            dir_fd=journal_fd,
            follow_symlinks=False,
        )
        control_record = (
            "file",
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            hashlib.sha256(payload).hexdigest(),
        )
    else:
        control_record = _prepared_control_inventory_record(event)
        try:
            os.stat(
                _ACCEPTANCE_FILE,
                dir_fd=journal_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            # A permanent receipt may resume after the separately authorized
            # control inode was quarantined or unlinked.  Its exact record is
            # still excluded from the deletion-only tree inventory below.
            pass
        else:
            observed_record, _payload = _require_prepared_control_at(
                journal_fd,
                event,
            )
            if observed_record != control_record:
                raise MutationJournalError(
                    "external recovery tombstone control inventory changed"
                )
    try:
        records.remove(control_record)
    except ValueError as exc:
        if event is not None:
            # Absence is the only valid post-receipt state.  A named control
            # was authenticated above, so failure to find that inode there is
            # never treated as absence.
            try:
                os.stat(
                    _ACCEPTANCE_FILE,
                    dir_fd=journal_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return tuple(sorted(records, key=repr))
        raise MutationJournalError(
            "external recovery tombstone control inventory changed"
        ) from exc
    return tuple(sorted(records, key=repr))


def _remove_bound_external_tombstone(
    parent_fd: int,
    name: str,
    journal_fd: int,
    expected: os.stat_result,
    *,
    event: dict[str, object],
    pre_remove_guard: Callable[[], None],
) -> None:
    """Remove the separately-bound control, then only frozen journal nodes."""

    pre_remove_guard()
    expected_control = _prepared_external_event_payload(event)
    expected_control_record = _prepared_control_inventory_record(event)
    control_name: str | None
    try:
        os.stat(
            _ACCEPTANCE_FILE,
            dir_fd=journal_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        control_name = None
        expected_identity = (
            int(expected_control_record[1]),
            int(expected_control_record[2]),
        )
        for candidate in _bounded_directory_names(
            journal_fd,
            label="external recovery tombstone root",
            max_entries=_MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_PATHS,
        ):
            if not candidate.startswith(
                ".crucible-cleanup-external-control-"
            ):
                continue
            metadata = os.stat(
                candidate,
                dir_fd=journal_fd,
                follow_symlinks=False,
            )
            if (metadata.st_dev, metadata.st_ino) == expected_identity:
                if control_name is not None:
                    raise MutationJournalError(
                        "external recovery tombstone control quarantine is "
                        "ambiguous"
                    ) from None
                control_name = candidate
    else:
        observed_record, observed_control = _require_prepared_control_at(
            journal_fd,
            event,
        )
        if (
            observed_record != expected_control_record
            or observed_control != expected_control
        ):
            raise MutationJournalError(
                "external recovery tombstone control changed before deletion"
            )
        control_name = _ACCEPTANCE_FILE
    if control_name is not None:
        _remove_bound_regular_entry(
            journal_fd,
            control_name,
            label="external recovery tombstone control event",
            expected_payload=expected_control,
            expected_inventory_record=expected_control_record,
            quarantine_prefix=".crucible-cleanup-external-control-",
        )
    pre_remove_guard()
    _remove_bound_directory_tree(
        parent_fd,
        name,
        journal_fd,
        expected,
        pre_remove_guard=pre_remove_guard,
        authorized_inventory=_external_tombstone_inventory(event),
    )


def _certified_writer_epoch_members(
    root: WorkspaceMutationJournal,
    proof: WriterEpochTerminalProof,
) -> tuple[object, ...]:
    members: list[object] = []
    for transaction_id, parent_transaction_id in proof.family:
        if transaction_id == root.transaction_id:
            members.append(root)
        else:
            members.append(_CertifiedExternalEpochMember(
                transaction_id=transaction_id,
                parent_transaction_id=parent_transaction_id,
                epoch_record=proof.epoch_record,
            ))
    if not members or members[0] is not root:
        raise MutationJournalError(
            "external recovery writer epoch family root changed"
        )
    return tuple(members)


def _require_audit_writer_epoch_proof(
    event: dict[str, object],
    *,
    transaction_id: str,
    members_preinstalled: bool = False,
) -> WriterEpochTerminalProof:
    """Require a self-consistent settled family when its tombstone is absent."""

    proof = _parse_external_writer_epoch_proof(
        event,
        transaction_id=transaction_id,
    )
    members = tuple(
        _CertifiedExternalEpochMember(
            transaction_id=member_id,
            parent_transaction_id=parent_id,
            epoch_record=proof.epoch_record,
        )
        for member_id, parent_id in proof.family
    )
    try:
        if members_preinstalled:
            _require_preinstalled_writer_epoch_members(
                cast(tuple[object, ...], members)
            )
            require_writer_epoch_quiescence(
                cast(tuple[WriterEpochStorage, ...], members),
                allow_never_armed=True,
            )
        else:
            with _installed_writer_epoch_recovery_members(
                cast(tuple[object, ...], members)
            ):
                require_writer_epoch_quiescence(
                    cast(tuple[WriterEpochStorage, ...], members),
                    allow_never_armed=True,
                )
    except WriterEpochError as exc:
        raise MutationJournalError(
            f"external recovery writer epoch is not quiescent: {exc}"
        ) from exc
    return proof


def _require_bound_external_writer_epoch(
    *,
    journal_path: Path,
    journal_fd: int,
    event: dict[str, object],
    transaction_id: str,
    members_preinstalled: bool = False,
    preinstalled_attestation: (
        _PreinstalledWriterEpochAttestation | None
    ) = None,
) -> WriterEpochTerminalProof:
    """Bind an audit proof to a held manifest and require family settlement."""

    payload = _read_bound_regular_at(
        journal_fd,
        _MANIFEST,
        label="external recovery tombstone manifest",
        max_bytes=_MUTATION_JOURNAL_MAX_MANIFEST_BYTES,
        required=False,
    )
    proof = _parse_external_writer_epoch_proof(
        event,
        transaction_id=transaction_id,
    )
    embedded = event.get(_WRITER_EPOCH_MANIFEST_FIELD)
    assert isinstance(embedded, dict)
    manifest = cast(dict[str, object], copy.deepcopy(embedded))
    if payload is not None:
        try:
            observed = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MutationJournalError(
                "external recovery tombstone manifest is malformed"
            ) from exc
        if observed != manifest:
            raise MutationJournalError(
                "external recovery tombstone manifest binding changed"
            )
    expected_inventory = set(_external_tombstone_inventory(event))
    current_inventory = set(
        _external_deletion_inventory(journal_fd, event=event)
    )
    if not current_inventory.issubset(expected_inventory):
        raise MutationJournalError(
            "external recovery tombstone inventory contains an unauthorized "
            "or changed entry"
        )
    journal = WorkspaceMutationJournal(journal_path, manifest)
    journal._bind_writer_epoch_recovery_proof(proof)
    members = _certified_writer_epoch_members(journal, proof)
    if members_preinstalled and preinstalled_attestation is not None:
        raise MutationJournalError(
            "external recovery supplied conflicting preinstalled epoch authority"
        )
    try:
        require_writer_epoch_quiescence(
            cast(tuple[WriterEpochStorage, ...], members),
            allow_never_armed=True,
        )
        if preinstalled_attestation is not None:
            _require_preinstalled_writer_epoch_attestation(
                preinstalled_attestation,
                proof,
                journal_fd=journal_fd,
                event=event,
            )
        elif members_preinstalled:
            _require_preinstalled_writer_epoch_members(members)
            journal._require_writer_epoch_quiescence(
                frontier="external recovery tombstone cleanup"
            )
        else:
            with _installed_writer_epoch_recovery_members(members):
                journal._require_writer_epoch_quiescence(
                    frontier="external recovery tombstone cleanup"
                )
    except WriterEpochError as exc:
        raise MutationJournalError(
            f"external recovery writer epoch is not quiescent: {exc}"
        ) from exc
    return proof


def _prepare_external_writer_epoch_proof(
    inspection: _Inspection,
    event: dict[str, object],
) -> WriterEpochTerminalProof:
    """Persist or revalidate epoch authority before registry destruction."""

    family = inspection.family
    try:
        require_writer_epoch_quiescence(
            cast(tuple[WriterEpochStorage, ...], family),
            allow_never_armed=True,
        )
        with _installed_writer_epoch_recovery_members(
            cast(tuple[object, ...], family)
        ):
            inspection.journal._require_writer_epoch_quiescence(
                frontier="external recovery registry cleanup"
            )
    except WriterEpochError as exc:
        raise MutationJournalError(
            f"external recovery writer epoch is not quiescent: {exc}"
        ) from exc

    expected = _writer_epoch_manifest_proof(
        inspection.journal._manifest,
        terminal_state="rolled_back",
    )
    durable = event.get(_WRITER_EPOCH_PROOF_FIELD)
    if durable is None:
        event[_WRITER_EPOCH_MANIFEST_FIELD] = copy.deepcopy(
            inspection.journal._manifest
        )
        event[_WRITER_EPOCH_PROOF_FIELD] = expected.to_record()
        _write_event(inspection.journal, event)
    else:
        observed = _parse_external_writer_epoch_proof(
            event,
            transaction_id=inspection.journal.transaction_id,
        )
        if observed.to_record() != expected.to_record():
            raise MutationJournalError(
                "external recovery writer epoch proof differs from manifest"
            )

    # The event write is a crash boundary.  Re-read epoch authority and bind
    # the durable record again before the first registry or namespace removal.
    try:
        with _installed_writer_epoch_recovery_members(
            cast(tuple[object, ...], family)
        ):
            inspection.journal._require_writer_epoch_quiescence(
                frontier="external recovery registry cleanup"
            )
    except WriterEpochError as exc:
        raise MutationJournalError(
            f"external recovery writer epoch is not quiescent: {exc}"
        ) from exc
    observed = _parse_external_writer_epoch_proof(
        event,
        transaction_id=inspection.journal.transaction_id,
    )
    current = _writer_epoch_manifest_proof(
        inspection.journal._manifest,
        terminal_state="rolled_back",
    )
    if observed.to_record() != current.to_record():
        raise MutationJournalError(
            "external recovery writer epoch changed after audit persistence"
        )
    return observed


def _cleanup_verified(
    inspection: _Inspection,
    event: dict[str, object],
    *,
    members_preinstalled: bool = False,
) -> ExternalRecoveryReport:
    family = inspection.family
    _prepare_external_writer_epoch_proof(inspection, event)
    rename_attestation: _PreinstalledWriterEpochAttestation | None = None
    with _bound_family_registry_entries(
        family,
        allow_missing=True,
    ) as registry_groups:
        for group in registry_groups:
            registry_fd = int(group["fd"])
            markers = group["markers"]
            assert isinstance(markers, dict)
            for name, (_member, expected_payload) in reversed(
                tuple(markers.items())
            ):
                try:
                    os.stat(name, dir_fd=registry_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                _remove_bound_regular_entry(
                    registry_fd,
                    name,
                    label=f"journal registry marker {name!r}",
                    expected_payload=expected_payload,
                )
            if _bounded_directory_names(
                registry_fd,
                label="mutation journal registry",
                max_entries=1,
            ):
                raise MutationJournalError(
                    "journal registry not empty after cleanup"
                )
    run_dir = _run_directory(inspection.journal)
    if inspection.journal._discovery_binding is not None:
        with (
            _bound_mutation_journal_directory(inspection.journal) as (
                _artifact_dir,
                journal_fd,
                journal_parent_fd,
                journal_entry_name,
            ),
            _bound_mutation_journal_run_directory(inspection.journal) as (
                _run_alias,
                run_fd,
            ),
        ):
            try:
                os.mkdir(_TOMBSTONE_DIRECTORY, mode=0o700, dir_fd=run_fd)
            except FileExistsError:
                pass
            tombstone_fd, _tombstone_opened = _open_authority_directory_at(
                run_fd,
                _TOMBSTONE_DIRECTORY,
                label="external recovery tombstone directory",
            )
            try:
                try:
                    os.stat(
                        inspection.journal.transaction_id,
                        dir_fd=tombstone_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    raise MutationJournalError(
                        "journal recovery tombstone already exists"
                    )
                journal_metadata = os.fstat(journal_fd)
                logical_tombstone = (
                    Path(run_dir)
                    / _TOMBSTONE_DIRECTORY
                    / inspection.journal.transaction_id
                )
                event["stage"] = "tombstone_prepared"
                event["tombstone"] = str(logical_tombstone)
                event["tombstone_identity"] = {
                    "device": journal_metadata.st_dev,
                    "inode": journal_metadata.st_ino,
                    "mode": stat.S_IMODE(journal_metadata.st_mode),
                }
                event["tombstone_inventory"] = [
                    list(record)
                    for record in _external_deletion_inventory(journal_fd)
                ]
                # Persist the complete, self-excluding deletion inventory in
                # the same prepared event that authorizes missing registry
                # markers.  A crash after this write but before rename can
                # safely resume from the still-bound original journal; a
                # crash after rename retains the same complete outer audit.
                _write_event(inspection.journal, event)
                if _external_deletion_inventory(
                    journal_fd,
                    event=event,
                ) != _external_tombstone_inventory(event):
                    raise MutationJournalError(
                        "external recovery tombstone inventory changed while "
                        "preparing rename"
                    )
                if members_preinstalled:
                    rename_attestation = (
                        _prevalidate_installed_writer_epoch_family(
                            family,
                            event,
                            journal_fd,
                        )
                    )
                # The event writer is an injectable crash boundary. Reopen
                # the complete named chain after it returns so a run/root/tx
                # swap cannot be hidden by the fds held above.
                with _bound_mutation_journal_directory(inspection.journal):
                    pass
                current = os.stat(
                    journal_entry_name,
                    dir_fd=journal_parent_fd,
                    follow_symlinks=False,
                )
                if (current.st_dev, current.st_ino) != (
                    journal_metadata.st_dev,
                    journal_metadata.st_ino,
                ):
                    raise MutationJournalError(
                        "journal directory changed before bound tombstoning"
                    )
                os.rename(
                    journal_entry_name,
                    inspection.journal.transaction_id,
                    src_dir_fd=journal_parent_fd,
                    dst_dir_fd=tombstone_fd,
                )
                renamed = os.stat(
                    inspection.journal.transaction_id,
                    dir_fd=tombstone_fd,
                    follow_symlinks=False,
                )
                if (renamed.st_dev, renamed.st_ino) != (
                    journal_metadata.st_dev,
                    journal_metadata.st_ino,
                ):
                    raise MutationJournalError(
                        "bound journal tombstone changed after rename"
                    )
                event["stage"] = "complete"
                try:
                    os.mkdir("journal_recovery", mode=0o700, dir_fd=run_fd)
                except FileExistsError:
                    pass
                audit_root_fd, _audit_root_opened = _open_authority_directory_at(
                    run_fd,
                    "journal_recovery",
                    label="external recovery audit directory",
                )
                try:
                    audit_name = f"{inspection.journal.transaction_id}.json"
                    write_atomic_bytes(
                        Path(f"/proc/self/fd/{audit_root_fd}") / audit_name,
                        _encode_recovery_event(event),
                    )
                    with _bound_recovery_audit(
                        audit_root_fd,
                        audit_name,
                    ) as (
                        durable_event,
                        durable_audit_fd,
                        durable_audit_opened,
                        durable_audit_sha256,
                    ):
                        if durable_event != event:
                            raise MutationJournalError(
                                "external recovery audit differs before "
                                "tombstone cleanup"
                            )
                        renamed = os.stat(
                            inspection.journal.transaction_id,
                            dir_fd=tombstone_fd,
                            follow_symlinks=False,
                        )
                        if (renamed.st_dev, renamed.st_ino) != (
                            journal_metadata.st_dev,
                            journal_metadata.st_ino,
                        ):
                            raise MutationJournalError(
                                "bound journal tombstone changed before cleanup"
                            )
                        _require_bound_external_writer_epoch(
                            journal_path=logical_tombstone,
                            journal_fd=journal_fd,
                            event=event,
                            transaction_id=(
                                inspection.journal.transaction_id
                            ),
                            preinstalled_attestation=rename_attestation,
                        )
                        _require_bound_recovery_audit_unchanged(
                            audit_root_fd,
                            audit_name,
                            durable_audit_fd,
                            durable_audit_opened,
                            durable_audit_sha256,
                        )
                        with _bound_external_cleanup_receipt(
                            run_fd=run_fd,
                            logical_run=Path(run_dir),
                            audit_root_fd=audit_root_fd,
                            audit_name=audit_name,
                            audit_fd=durable_audit_fd,
                            audit_opened=durable_audit_opened,
                            audit_payload_sha256=durable_audit_sha256,
                            event=event,
                        ) as cleanup_receipt:
                            def cleanup_guard() -> None:
                                _require_bound_recovery_audit_unchanged(
                                    audit_root_fd,
                                    audit_name,
                                    durable_audit_fd,
                                    durable_audit_opened,
                                    durable_audit_sha256,
                                )
                                _require_external_cleanup_receipt_unchanged(
                                    cleanup_receipt
                                )

                            _remove_bound_external_tombstone(
                                tombstone_fd,
                                inspection.journal.transaction_id,
                                journal_fd,
                                journal_metadata,
                                event=event,
                                pre_remove_guard=cleanup_guard,
                            )
                            _reconcile_external_cleanup_audit(
                                receipt=cleanup_receipt,
                                audit_root_fd=audit_root_fd,
                                audit_name=audit_name,
                                audit_fd=durable_audit_fd,
                                audit_opened=durable_audit_opened,
                                audit_payload_sha256=durable_audit_sha256,
                            )
                finally:
                    os.close(audit_root_fd)
            finally:
                os.close(tombstone_fd)
        return ExternalRecoveryReport(
            status="recovered",
            run_id=inspection.request.run_id,
            transaction_id=inspection.request.transaction_id,
            source_root=str(Path(inspection.journal._manifest["cwd"])),
            removed_linked_worktrees=(
                inspection.request.removed_linked_worktrees
            ),
            removed_peer_transactions=tuple(
                peer.transaction_id for peer in inspection.removed_peers
            ),
            baseline_stash_tip=inspection.baseline_stash_tip,
            appended_stash_tip=inspection.request.appended_stash_tip,
            object_addition_oids=inspection.object_addition_oids,
            expected_diff_ref=inspection.request.expected_diff_ref,
            ancestor_transaction_id=(
                inspection.accepted_ancestor.journal.transaction_id
                if inspection.accepted_ancestor is not None
                else None
            ),
            ancestor_root=(
                inspection.accepted_ancestor.request.ancestor_root
                if inspection.accepted_ancestor is not None
                else None
            ),
            added_submodule_path=(
                inspection.accepted_ancestor.request.submodule_path
                if inspection.accepted_ancestor is not None
                else None
            ),
            ancestor_gitmodules_blob_oid=(
                inspection.accepted_ancestor.gitmodules_blob_oid
                if inspection.accepted_ancestor is not None
                else None
            ),
            audit_path=str(
                Path(run_dir)
                / "journal_recovery"
                / f"{inspection.journal.transaction_id}.json"
            ),
        )
    tombstone_root = _reserved_directory(
        run_dir,
        _TOMBSTONE_DIRECTORY,
        create=True,
    )
    assert tombstone_root is not None
    tombstone = tombstone_root / inspection.journal.transaction_id
    if tombstone.exists() or tombstone.is_symlink():
        raise MutationJournalError(
            f"journal recovery tombstone already exists: {tombstone}"
        )
    journal_metadata = inspection.journal.journal_dir.lstat()
    if not stat.S_ISDIR(journal_metadata.st_mode) or stat.S_ISLNK(
        journal_metadata.st_mode
    ):
        raise MutationJournalError("journal directory changed before tombstoning")
    event["stage"] = "tombstone_prepared"
    event["tombstone"] = str(tombstone)
    event["tombstone_identity"] = {
        "device": journal_metadata.st_dev,
        "inode": journal_metadata.st_ino,
        "mode": stat.S_IMODE(journal_metadata.st_mode),
    }
    direct_inventory_fd, inventory_opened = _open_authority_directory(
        inspection.journal.journal_dir,
        label="external recovery journal inventory",
    )
    try:
        if (inventory_opened.st_dev, inventory_opened.st_ino) != (
            journal_metadata.st_dev,
            journal_metadata.st_ino,
        ):
            raise MutationJournalError(
                "external recovery journal changed before inventory"
            )
        event["tombstone_inventory"] = [
            list(record)
            for record in _external_deletion_inventory(direct_inventory_fd)
        ]
    finally:
        os.close(direct_inventory_fd)
    _write_event(inspection.journal, event)
    direct_inventory_fd, inventory_opened = _open_authority_directory(
        inspection.journal.journal_dir,
        label="external recovery prepared journal inventory",
    )
    try:
        if (inventory_opened.st_dev, inventory_opened.st_ino) != (
            journal_metadata.st_dev,
            journal_metadata.st_ino,
        ) or _external_deletion_inventory(
            direct_inventory_fd,
            event=event,
        ) != _external_tombstone_inventory(event):
            raise MutationJournalError(
                "external recovery tombstone inventory changed while "
                "preparing rename"
            )
    finally:
        os.close(direct_inventory_fd)
    audit = str(_audit_path(inspection.journal))
    if tombstone.exists() or tombstone.is_symlink():
        raise MutationJournalError("journal recovery tombstone appeared before rename")
    inspection.journal.journal_dir.replace(tombstone)
    event["stage"] = "complete"
    write_atomic_bytes(Path(audit), _encode_recovery_event(event))
    audit_path = Path(audit)
    direct_audit_root_fd: int | None = None
    direct_tombstone_root_fd: int | None = None
    direct_tombstone_fd: int | None = None
    direct_run_fd: int | None = None
    try:
        direct_run_fd, _direct_run_opened = _open_authority_directory(
            Path(run_dir),
            label="external recovery run directory",
        )
        direct_audit_root_fd, _audit_root_opened = _open_authority_directory(
            audit_path.parent,
            label="external recovery audit directory",
        )
        direct_tombstone_root_fd, _tombstone_root_opened = (
            _open_authority_directory(
            tombstone_root,
            label="external recovery tombstone directory",
        )
        )
        direct_tombstone_fd, tombstone_opened = _open_authority_directory_at(
            direct_tombstone_root_fd,
            inspection.journal.transaction_id,
            label="bound external recovery tombstone",
        )
        if (tombstone_opened.st_dev, tombstone_opened.st_ino) != (
            journal_metadata.st_dev,
            journal_metadata.st_ino,
        ):
            raise MutationJournalError(
                "external recovery tombstone changed while opening"
            )
        _require_bound_external_writer_epoch(
            journal_path=tombstone,
            journal_fd=direct_tombstone_fd,
            event=event,
            transaction_id=inspection.journal.transaction_id,
        )
        with _bound_recovery_audit(
            direct_audit_root_fd,
            audit_path.name,
        ) as (
            durable_event,
            durable_audit_fd,
            durable_audit_opened,
            durable_audit_sha256,
        ):
            if durable_event != event:
                raise MutationJournalError(
                    "external recovery audit differs before tombstone cleanup"
                )
            _require_bound_recovery_audit_unchanged(
                direct_audit_root_fd,
                audit_path.name,
                durable_audit_fd,
                durable_audit_opened,
                durable_audit_sha256,
            )
            with _bound_external_cleanup_receipt(
                run_fd=direct_run_fd,
                logical_run=Path(run_dir),
                audit_root_fd=direct_audit_root_fd,
                audit_name=audit_path.name,
                audit_fd=durable_audit_fd,
                audit_opened=durable_audit_opened,
                audit_payload_sha256=durable_audit_sha256,
                event=event,
            ) as cleanup_receipt:
                def cleanup_guard() -> None:
                    _require_bound_recovery_audit_unchanged(
                        direct_audit_root_fd,
                        audit_path.name,
                        durable_audit_fd,
                        durable_audit_opened,
                        durable_audit_sha256,
                    )
                    _require_external_cleanup_receipt_unchanged(
                        cleanup_receipt
                    )

                _remove_bound_external_tombstone(
                    direct_tombstone_root_fd,
                    inspection.journal.transaction_id,
                    direct_tombstone_fd,
                    tombstone_opened,
                    event=event,
                    pre_remove_guard=cleanup_guard,
                )
                _reconcile_external_cleanup_audit(
                    receipt=cleanup_receipt,
                    audit_root_fd=direct_audit_root_fd,
                    audit_name=audit_path.name,
                    audit_fd=durable_audit_fd,
                    audit_opened=durable_audit_opened,
                    audit_payload_sha256=durable_audit_sha256,
                )
    finally:
        if direct_tombstone_fd is not None:
            os.close(direct_tombstone_fd)
        if direct_tombstone_root_fd is not None:
            os.close(direct_tombstone_root_fd)
        if direct_audit_root_fd is not None:
            os.close(direct_audit_root_fd)
        if direct_run_fd is not None:
            os.close(direct_run_fd)
    return ExternalRecoveryReport(
        status="recovered",
        run_id=inspection.request.run_id,
        transaction_id=inspection.request.transaction_id,
        source_root=str(Path(inspection.journal._manifest["cwd"])),
        removed_linked_worktrees=inspection.request.removed_linked_worktrees,
        removed_peer_transactions=tuple(
            peer.transaction_id for peer in inspection.removed_peers
        ),
        baseline_stash_tip=inspection.baseline_stash_tip,
        appended_stash_tip=inspection.request.appended_stash_tip,
        object_addition_oids=inspection.object_addition_oids,
        expected_diff_ref=inspection.request.expected_diff_ref,
        ancestor_transaction_id=(
            inspection.accepted_ancestor.journal.transaction_id
            if inspection.accepted_ancestor is not None
            else None
        ),
        ancestor_root=(
            inspection.accepted_ancestor.request.ancestor_root
            if inspection.accepted_ancestor is not None
            else None
        ),
        added_submodule_path=(
            inspection.accepted_ancestor.request.submodule_path
            if inspection.accepted_ancestor is not None
            else None
        ),
        ancestor_gitmodules_blob_oid=(
            inspection.accepted_ancestor.gitmodules_blob_oid
            if inspection.accepted_ancestor is not None
            else None
        ),
        audit_path=audit,
    )


@contextmanager
def _bound_recovery_audit(
    audit_root_fd: int,
    name: str,
    *,
    max_bytes: int = _MAX_RECOVERY_EVENT_BYTES,
) -> Iterator[tuple[dict[str, object], int, os.stat_result, str]]:
    """Hold one bounded, stable recovery audit through its consequences."""

    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=audit_root_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > max_bytes
        ):
            raise MutationJournalError(
                "journal recovery audit is unsafe or exceeds its byte ceiling"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=audit_root_fd,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise MutationJournalError(
                "journal recovery audit changed while opening"
            )
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            if len(payload) + len(chunk) > max_bytes:
                raise MutationJournalError(
                    "journal recovery audit exceeds its byte ceiling"
                )
            payload.extend(chunk)
        finished = os.fstat(descriptor)
        after = os.stat(name, dir_fd=audit_root_fd, follow_symlinks=False)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(opened, field) != getattr(finished, field)
            or getattr(opened, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise MutationJournalError(
                "journal recovery audit changed during read"
            )
        try:
            event = json.loads(bytes(payload))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MutationJournalError(
                "unreadable journal recovery audit"
            ) from exc
        if not isinstance(event, dict):
            raise MutationJournalError("journal recovery audit is not an object")
        yield event, descriptor, opened, hashlib.sha256(payload).hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)


# Keep receipt parsing on its own ceiling and implementation seam.  Recovery
# audit adversarial hooks deliberately target ``_bound_recovery_audit`` and
# must not accidentally rewrite the separately retained receipt authority.
_bound_json_authority = _bound_recovery_audit


@contextmanager
def _bound_external_cleanup_receipt_record(
    receipt_root_fd: int,
    name: str,
) -> Iterator[tuple[dict[str, object], int, os.stat_result, str]]:
    with _bound_json_authority(
        receipt_root_fd,
        name,
        max_bytes=_MAX_RECOVERY_CLEANUP_RECEIPT_BYTES,
    ) as bound:
        yield bound


def _require_bound_recovery_audit_unchanged(
    audit_root_fd: int,
    name: str,
    audit_fd: int,
    opened: os.stat_result,
    payload_sha256: str,
    *,
    max_bytes: int = _MAX_RECOVERY_EVENT_BYTES,
) -> None:
    """Reauthenticate exact audit bytes through held and named authority."""

    try:
        before = os.fstat(audit_fd)
        current = os.stat(name, dir_fd=audit_root_fd, follow_symlinks=False)
    except OSError as exc:
        raise MutationJournalError(
            "journal recovery audit changed before destructive commit"
        ) from exc
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > max_bytes
        or any(
            getattr(opened, field) != getattr(before, field)
            or getattr(opened, field) != getattr(current, field)
            for field in stable_fields
        )
    ):
        raise MutationJournalError(
            "journal recovery audit changed before destructive commit"
        )
    payload = bytearray()
    offset = 0
    while True:
        chunk = os.pread(audit_fd, 1024 * 1024, offset)
        if not chunk:
            break
        payload.extend(chunk)
        offset += len(chunk)
        if len(payload) > max_bytes:
            raise MutationJournalError(
                "journal recovery audit exceeds its byte ceiling"
            )
    finished = os.fstat(audit_fd)
    after = os.stat(name, dir_fd=audit_root_fd, follow_symlinks=False)
    if any(
        getattr(opened, field) != getattr(finished, field)
        or getattr(opened, field) != getattr(after, field)
        for field in stable_fields
    ) or hashlib.sha256(payload).hexdigest() != payload_sha256:
        raise MutationJournalError(
            "journal recovery audit content changed before destructive commit"
        )


@dataclass(frozen=True, slots=True)
class _BoundExternalCleanupReceipt:
    """Held independent authority for one external tombstone deletion."""

    root_fd: int
    name: str
    descriptor: int
    opened: os.stat_result
    payload_sha256: str
    record: dict[str, object]
    audit_payload: bytes


@dataclass(frozen=True, slots=True)
class _ExternalCleanupReceiptCandidate:
    """Frozen restart authority carried outside a deleted tombstone."""

    run_name: str
    receipt_name: str
    receipt_identity: tuple[int, int, int, int, int, int, int]
    receipt_sha256: str
    record: dict[str, object]
    event: dict[str, object]
    audit_payload: bytes
    tombstone_present: bool
    certified_members: tuple[_CertifiedExternalEpochMember, ...]


def _read_exact_bound_payload(
    descriptor: int,
    opened: os.stat_result,
    expected_sha256: str,
    *,
    label: str,
    max_bytes: int = _MAX_RECOVERY_EVENT_BYTES,
) -> bytes:
    """Read exact bytes from a held regular-file authority."""

    payload = bytearray()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            break
        payload.extend(chunk)
        offset += len(chunk)
        if len(payload) > max_bytes:
            raise MutationJournalError(f"{label} exceeds its byte ceiling")
    finished = os.fstat(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(
            getattr(opened, field) != getattr(finished, field)
            for field in stable_fields
        )
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise MutationJournalError(f"{label} changed during exact read")
    return bytes(payload)


def _external_cleanup_receipt_record(
    *,
    event: dict[str, object],
    audit_payload: bytes,
) -> dict[str, object]:
    """Build the independently retained authority for tombstone deletion."""

    try:
        audit_text = audit_payload.decode("utf-8", errors="strict")
        observed_event = json.loads(audit_payload)
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MutationJournalError(
            "external recovery audit cannot seed a cleanup receipt"
        ) from exc
    if observed_event != event:
        raise MutationJournalError(
            "external recovery audit payload differs from its parsed event"
        )
    if audit_payload != _encode_recovery_event(event):
        raise MutationJournalError(
            "external recovery audit cannot seed a noncanonical cleanup "
            "receipt"
        )
    request = ExternalRecoveryRequest.from_json(
        event["request"]  # type: ignore[arg-type]
    )
    source_root = str(event.get("source_root", ""))
    tombstone = str(event.get("tombstone", ""))
    identity = event.get("tombstone_identity")
    if (
        event.get("schema") != _ACCEPTANCE_SCHEMA
        or event.get("stage") not in {"tombstone_prepared", "complete"}
        or not source_root
        or not tombstone
        or not isinstance(identity, dict)
        or set(identity) != {"device", "inode", "mode"}
    ):
        raise MutationJournalError(
            "external recovery event cannot authorize tombstone deletion"
        )
    _parse_external_writer_epoch_proof(
        event,
        transaction_id=request.transaction_id,
    )
    return {
        "schema": _CLEANUP_RECEIPT_SCHEMA,
        "run_id": request.run_id,
        "transaction_id": request.transaction_id,
        "source_root": source_root,
        "tombstone": tombstone,
        "tombstone_identity": copy.deepcopy(identity),
        "audit_payload": audit_text,
        "audit_sha256": hashlib.sha256(audit_payload).hexdigest(),
    }


def _parse_external_cleanup_receipt(
    record: dict[str, object],
    *,
    logical_run: Path,
    receipt_name: str,
) -> tuple[dict[str, object], bytes]:
    """Authenticate one permanent external-cleanup receipt."""

    required = {
        "schema",
        "run_id",
        "transaction_id",
        "source_root",
        "tombstone",
        "tombstone_identity",
        "audit_payload",
        "audit_sha256",
    }
    if set(record) != required or record.get("schema") != _CLEANUP_RECEIPT_SCHEMA:
        raise MutationJournalError(
            "external recovery cleanup receipt fields changed"
        )
    if not isinstance(record.get("audit_payload"), str):
        raise MutationJournalError(
            "external recovery cleanup receipt payload is malformed"
        )
    audit_payload = str(record["audit_payload"]).encode(
        "utf-8",
        errors="strict",
    )
    digest = record.get("audit_sha256")
    if (
        not isinstance(digest, str)
        or hashlib.sha256(audit_payload).hexdigest() != digest
    ):
        raise MutationJournalError(
            "external recovery cleanup receipt audit digest changed"
        )
    try:
        parsed_event = json.loads(audit_payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MutationJournalError(
            "external recovery cleanup receipt audit is malformed"
        ) from exc
    if not isinstance(parsed_event, dict):
        raise MutationJournalError(
            "external recovery cleanup receipt event is not an object"
        )
    event = cast(dict[str, object], parsed_event)
    if audit_payload != _encode_recovery_event(event):
        # The receipt becomes permanent restart and destructive authority once
        # its tombstone disappears.  Reject semantically equivalent reorder,
        # whitespace drift, and duplicate-key collapse: only the canonical
        # bytes originally authored by Crucible may repair the named audit.
        raise MutationJournalError(
            "external recovery cleanup receipt audit is noncanonical"
        )
    request = ExternalRecoveryRequest.from_json(
        event["request"]  # type: ignore[arg-type]
    )
    expected_tombstone = (
        logical_run / _TOMBSTONE_DIRECTORY / request.transaction_id
    )
    if (
        record.get("run_id") != request.run_id
        or logical_run.name != request.run_id
        or record.get("transaction_id") != request.transaction_id
        or receipt_name != f"{request.transaction_id}.json"
        or record.get("source_root") != event.get("source_root")
        or Path(str(record.get("tombstone", ""))) != expected_tombstone
        or record.get("tombstone") != event.get("tombstone")
        or record.get("tombstone_identity")
        != event.get("tombstone_identity")
        or event.get("schema") != _ACCEPTANCE_SCHEMA
        or event.get("stage") not in {"tombstone_prepared", "complete"}
    ):
        raise MutationJournalError(
            "external recovery cleanup receipt authority changed"
        )
    _parse_external_writer_epoch_proof(
        event,
        transaction_id=request.transaction_id,
    )
    return event, audit_payload


def _write_exclusive_bound_json(
    directory_fd: int,
    name: str,
    payload: dict[str, object],
) -> None:
    """Create one durable JSON authority without replacing any existing name."""

    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_RECOVERY_CLEANUP_RECEIPT_BYTES:
        raise MutationJournalError(
            "external recovery cleanup receipt exceeds its byte ceiling"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise MutationJournalError(
                    "external recovery cleanup receipt write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        os.fsync(directory_fd)
    except FileExistsError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot persist external recovery cleanup receipt: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _bound_external_cleanup_receipt(
    *,
    run_fd: int,
    logical_run: Path,
    audit_root_fd: int,
    audit_name: str,
    audit_fd: int,
    audit_opened: os.stat_result,
    audit_payload_sha256: str,
    event: dict[str, object],
) -> Iterator[_BoundExternalCleanupReceipt]:
    """Persist and hold proof independent of the journal being removed."""

    _require_bound_recovery_audit_unchanged(
        audit_root_fd,
        audit_name,
        audit_fd,
        audit_opened,
        audit_payload_sha256,
    )
    audit_payload = _read_exact_bound_payload(
        audit_fd,
        audit_opened,
        audit_payload_sha256,
        label="external recovery audit",
    )
    expected_record = _external_cleanup_receipt_record(
        event=event,
        audit_payload=audit_payload,
    )
    try:
        os.mkdir(
            _CLEANUP_RECEIPT_DIRECTORY,
            mode=0o700,
            dir_fd=run_fd,
        )
        os.fsync(run_fd)
    except FileExistsError:
        pass
    receipt_root_fd, _receipt_root_opened = _open_authority_directory_at(
        run_fd,
        _CLEANUP_RECEIPT_DIRECTORY,
        label="external recovery cleanup receipt directory",
    )
    request = ExternalRecoveryRequest.from_json(
        event["request"]  # type: ignore[arg-type]
    )
    receipt_name = f"{request.transaction_id}.json"
    try:
        try:
            os.stat(
                receipt_name,
                dir_fd=receipt_root_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            try:
                _write_exclusive_bound_json(
                    receipt_root_fd,
                    receipt_name,
                    expected_record,
                )
            except FileExistsError:
                # A racing creator is accepted only through the same exact
                # parser and byte/event binding below.
                pass
        with _bound_external_cleanup_receipt_record(
            receipt_root_fd,
            receipt_name,
        ) as (
            observed_record,
            receipt_fd,
            receipt_opened,
            receipt_payload_sha256,
        ):
            parsed_event, parsed_audit_payload = (
                _parse_external_cleanup_receipt(
                    observed_record,
                    logical_run=logical_run,
                    receipt_name=receipt_name,
                )
            )
            if (
                observed_record != expected_record
                or parsed_event != event
                or parsed_audit_payload != audit_payload
            ):
                raise MutationJournalError(
                    "external recovery cleanup receipt conflicts with audit"
                )
            yield _BoundExternalCleanupReceipt(
                root_fd=receipt_root_fd,
                name=receipt_name,
                descriptor=receipt_fd,
                opened=receipt_opened,
                payload_sha256=receipt_payload_sha256,
                record=observed_record,
                audit_payload=audit_payload,
            )
    finally:
        os.close(receipt_root_fd)


def _require_external_cleanup_receipt_unchanged(
    receipt: _BoundExternalCleanupReceipt,
) -> None:
    _require_bound_recovery_audit_unchanged(
        receipt.root_fd,
        receipt.name,
        receipt.descriptor,
        receipt.opened,
        receipt.payload_sha256,
        max_bytes=_MAX_RECOVERY_CLEANUP_RECEIPT_BYTES,
    )


def _reconcile_external_cleanup_audit(
    *,
    receipt: _BoundExternalCleanupReceipt,
    audit_root_fd: int,
    audit_name: str,
    audit_fd: int,
    audit_opened: os.stat_result,
    audit_payload_sha256: str,
) -> None:
    """Keep exact named audit evidence after the receipt commits deletion."""

    _require_external_cleanup_receipt_unchanged(receipt)
    try:
        _require_bound_recovery_audit_unchanged(
            audit_root_fd,
            audit_name,
            audit_fd,
            audit_opened,
            audit_payload_sha256,
        )
    except MutationJournalError as exc:
        # The independently retained receipt is now the commit authority.  It
        # can safely repair a replaced or same-inode-mutated audit without the
        # deleted journal, and remains durable if repair itself is interrupted.
        write_atomic_bytes(
            Path(f"/proc/self/fd/{audit_root_fd}") / audit_name,
            receipt.audit_payload,
        )
        with _bound_recovery_audit(audit_root_fd, audit_name) as (
            restored_event,
            _restored_fd,
            _restored_opened,
            restored_sha256,
        ):
            expected_event = json.loads(receipt.audit_payload)
            if not isinstance(expected_event, dict) or (
                restored_event != expected_event
                or restored_sha256
                != hashlib.sha256(receipt.audit_payload).hexdigest()
            ):
                raise MutationJournalError(
                    "external recovery audit repair differs from receipt"
                ) from exc
        _require_external_cleanup_receipt_unchanged(receipt)
        raise MutationJournalError(
            "external recovery audit changed during destructive commit; "
            "independent cleanup receipt retained"
        ) from exc
    _require_external_cleanup_receipt_unchanged(receipt)


def _inventory_external_cleanup_receipts(
    corpus: "Corpus",
    source_root: str | Path,
) -> tuple[_ExternalCleanupReceiptCandidate, ...]:
    """Read and authenticate every restart receipt before any audit repair."""

    source = Path(source_root).resolve()
    spec_dir = Path(corpus.spec_dir).absolute()
    candidates: list[_ExternalCleanupReceiptCandidate] = []
    certified_by_id: dict[str, _CertifiedExternalEpochMember] = {}
    spec_fd, _spec_opened = _open_authority_directory(
        spec_dir,
        label="Corpus Spec directory",
    )
    runs_fd: int | None = None
    try:
        try:
            os.stat("runs", dir_fd=spec_fd, follow_symlinks=False)
        except FileNotFoundError:
            return ()
        runs_fd, _runs_opened = _open_authority_directory_at(
            spec_fd,
            "runs",
            label="Corpus runs directory",
        )
        run_names = _bounded_directory_names(
            runs_fd,
            label="Corpus runs directory",
            max_entries=_MUTATION_JOURNAL_MAX_RUNS,
        )
        for run_name in run_names:
            run_fd: int | None = None
            receipt_root_fd: int | None = None
            tombstone_root_fd: int | None = None
            try:
                run_fd, _run_opened = _open_authority_directory_at(
                    runs_fd,
                    run_name,
                    label=f"Corpus run directory {run_name!r}",
                )
                try:
                    os.stat(
                        _CLEANUP_RECEIPT_DIRECTORY,
                        dir_fd=run_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                receipt_root_fd, _receipt_root_opened = (
                    _open_authority_directory_at(
                        run_fd,
                        _CLEANUP_RECEIPT_DIRECTORY,
                        label="external recovery cleanup receipt directory",
                    )
                )
                receipt_names = _bounded_directory_names(
                    receipt_root_fd,
                    label="external recovery cleanup receipt directory",
                    max_entries=_MUTATION_JOURNAL_MAX_NESTED_MEMBERS,
                )
                for receipt_name in receipt_names:
                    if not receipt_name.endswith(".json"):
                        raise MutationJournalError(
                            "external recovery cleanup receipt directory "
                            "contains an unexpected entry"
                        )
                    with _bound_external_cleanup_receipt_record(
                        receipt_root_fd,
                        receipt_name,
                    ) as (
                        record,
                        receipt_fd,
                        receipt_opened,
                        receipt_sha256,
                    ):
                        logical_run = spec_dir / "runs" / run_name
                        event, audit_payload = (
                            _parse_external_cleanup_receipt(
                                record,
                                logical_run=logical_run,
                                receipt_name=receipt_name,
                            )
                        )
                        event_source = Path(str(event.get("source_root", "")))
                        if (
                            not event_source.is_absolute()
                            or event_source.resolve() != event_source
                        ):
                            raise MutationJournalError(
                                "external recovery cleanup receipt source is "
                                "non-canonical"
                            )
                        if event_source != source:
                            continue
                        request = ExternalRecoveryRequest.from_json(
                            event["request"]  # type: ignore[arg-type]
                        )
                        original = Path(str(event.get("journal_dir", "")))
                        if (
                            not original.is_absolute()
                            or original.parent.parent != logical_run
                            or original.parent.name
                            not in _DURABLE_JOURNAL_ROOTS
                        ):
                            raise MutationJournalError(
                                "external recovery cleanup receipt journal "
                                "binding changed"
                            )
                        durable_fd, _durable_opened = (
                            _open_authority_directory_at(
                                run_fd,
                                original.parent.name,
                                label="durable mutation journal root",
                            )
                        )
                        try:
                            try:
                                os.stat(
                                    original.name,
                                    dir_fd=durable_fd,
                                    follow_symlinks=False,
                                )
                            except FileNotFoundError:
                                pass
                            else:
                                raise MutationJournalError(
                                    "external cleanup receipt appeared before "
                                    "journal tombstoning"
                                )
                        finally:
                            os.close(durable_fd)
                        identity = event.get("tombstone_identity")
                        assert isinstance(identity, dict)
                        try:
                            os.stat(
                                _TOMBSTONE_DIRECTORY,
                                dir_fd=run_fd,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            tombstone_present = False
                        else:
                            tombstone_root_fd, _tombstone_root_opened = (
                                _open_authority_directory_at(
                                    run_fd,
                                    _TOMBSTONE_DIRECTORY,
                                    label=(
                                        "journal recovery tombstone directory"
                                    ),
                                )
                            )
                            try:
                                tombstone_metadata = os.stat(
                                    request.transaction_id,
                                    dir_fd=tombstone_root_fd,
                                    follow_symlinks=False,
                                )
                            except FileNotFoundError:
                                tombstone_present = False
                            else:
                                tombstone_present = True
                                if (
                                    not stat.S_ISDIR(tombstone_metadata.st_mode)
                                    or stat.S_ISLNK(tombstone_metadata.st_mode)
                                    or tombstone_metadata.st_dev
                                    != int(identity["device"])
                                    or tombstone_metadata.st_ino
                                    != int(identity["inode"])
                                    or stat.S_IMODE(tombstone_metadata.st_mode)
                                    != int(identity["mode"])
                                ):
                                    raise MutationJournalError(
                                        "external recovery cleanup receipt "
                                        "tombstone identity changed"
                                    )
                                tombstone_fd, tombstone_opened = (
                                    _open_authority_directory_at(
                                        tombstone_root_fd,
                                        request.transaction_id,
                                        label=(
                                            "bound journal recovery tombstone"
                                        ),
                                    )
                                )
                                try:
                                    if (
                                        tombstone_opened.st_dev,
                                        tombstone_opened.st_ino,
                                    ) != (
                                        tombstone_metadata.st_dev,
                                        tombstone_metadata.st_ino,
                                    ):
                                        raise MutationJournalError(
                                            "external recovery cleanup receipt "
                                            "tombstone changed while opening"
                                        )
                                    _require_bound_external_writer_epoch(
                                        journal_path=(
                                            logical_run
                                            / _TOMBSTONE_DIRECTORY
                                            / request.transaction_id
                                        ),
                                        journal_fd=tombstone_fd,
                                        event=event,
                                        transaction_id=(
                                            request.transaction_id
                                        ),
                                    )
                                finally:
                                    os.close(tombstone_fd)
                        if tombstone_root_fd is not None:
                            os.close(tombstone_root_fd)
                            tombstone_root_fd = None
                        proof = _require_audit_writer_epoch_proof(
                            event,
                            transaction_id=request.transaction_id,
                        )
                        certified = tuple(
                            _CertifiedExternalEpochMember(
                                transaction_id=transaction_id,
                                parent_transaction_id=parent_transaction_id,
                                epoch_record=proof.epoch_record,
                            )
                            for transaction_id, parent_transaction_id
                            in proof.family
                        )
                        _require_bound_recovery_audit_unchanged(
                            receipt_root_fd,
                            receipt_name,
                            receipt_fd,
                            receipt_opened,
                            receipt_sha256,
                            max_bytes=(
                                _MAX_RECOVERY_CLEANUP_RECEIPT_BYTES
                            ),
                        )
                        for member in certified:
                            if member.transaction_id in certified_by_id:
                                raise MutationJournalError(
                                    "external recovery cleanup receipt "
                                    "inventory contains duplicate or "
                                    "overlapping transaction "
                                    f"{member.transaction_id}"
                                )
                            certified_by_id[member.transaction_id] = member
                        candidates.append(
                            _ExternalCleanupReceiptCandidate(
                                run_name=run_name,
                                receipt_name=receipt_name,
                                receipt_identity=(
                                    receipt_opened.st_dev,
                                    receipt_opened.st_ino,
                                    receipt_opened.st_mode,
                                    receipt_opened.st_nlink,
                                    receipt_opened.st_size,
                                    receipt_opened.st_mtime_ns,
                                    receipt_opened.st_ctime_ns,
                                ),
                                receipt_sha256=receipt_sha256,
                                record=copy.deepcopy(record),
                                event=copy.deepcopy(event),
                                audit_payload=audit_payload,
                                tombstone_present=tombstone_present,
                                certified_members=certified,
                            )
                        )
            finally:
                if tombstone_root_fd is not None:
                    os.close(tombstone_root_fd)
                if receipt_root_fd is not None:
                    os.close(receipt_root_fd)
                if run_fd is not None:
                    os.close(run_fd)
    finally:
        if runs_fd is not None:
            os.close(runs_fd)
        os.close(spec_fd)
    members = tuple(certified_by_id.values())
    with _installed_writer_epoch_recovery_members(
        cast(tuple[object, ...], members)
    ):
        pass
    return tuple(sorted(
        candidates,
        key=lambda candidate: (
            candidate.run_name,
            candidate.receipt_name,
        ),
    ))


def _reconcile_external_cleanup_receipts(
    corpus: "Corpus",
    source_root: str | Path,
) -> tuple[str, ...]:
    """Repair named audits from exact receipts after a deletion-side crash."""

    candidates = _inventory_external_cleanup_receipts(corpus, source_root)
    confirmed = _inventory_external_cleanup_receipts(corpus, source_root)
    if confirmed != candidates:
        raise MutationJournalError(
            "external recovery cleanup receipt inventory changed before repair"
        )
    if not candidates:
        return ()
    spec_dir = Path(corpus.spec_dir).absolute()
    repaired: list[str] = []
    members = tuple(
        member
        for candidate in candidates
        for member in candidate.certified_members
    )
    with _installed_writer_epoch_recovery_members(
        cast(tuple[object, ...], members)
    ):
        for candidate in candidates:
            logical_run = spec_dir / "runs" / candidate.run_name
            run_fd: int | None = None
            receipt_root_fd: int | None = None
            audit_root_fd: int | None = None
            try:
                run_fd, _run_opened = _open_authority_directory(
                    logical_run,
                    label=(
                        "external recovery cleanup receipt run directory"
                    ),
                )
                receipt_root_fd, _receipt_root_opened = (
                    _open_authority_directory_at(
                        run_fd,
                        _CLEANUP_RECEIPT_DIRECTORY,
                        label="external recovery cleanup receipt directory",
                    )
                )
                with _bound_external_cleanup_receipt_record(
                    receipt_root_fd,
                    candidate.receipt_name,
                ) as (
                    record,
                    receipt_fd,
                    receipt_opened,
                    receipt_sha256,
                ):
                    receipt_identity = (
                        receipt_opened.st_dev,
                        receipt_opened.st_ino,
                        receipt_opened.st_mode,
                        receipt_opened.st_nlink,
                        receipt_opened.st_size,
                        receipt_opened.st_mtime_ns,
                        receipt_opened.st_ctime_ns,
                    )
                    event, audit_payload = _parse_external_cleanup_receipt(
                        record,
                        logical_run=logical_run,
                        receipt_name=candidate.receipt_name,
                    )
                    if (
                        receipt_identity != candidate.receipt_identity
                        or receipt_sha256 != candidate.receipt_sha256
                        or record != candidate.record
                        or event != candidate.event
                        or audit_payload != candidate.audit_payload
                    ):
                        raise MutationJournalError(
                            "external recovery cleanup receipt changed before "
                            "audit repair"
                        )
                    request = ExternalRecoveryRequest.from_json(
                        event["request"]  # type: ignore[arg-type]
                    )
                    try:
                        os.stat(
                            _TOMBSTONE_DIRECTORY,
                            dir_fd=run_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        tombstone_present = False
                    else:
                        tombstone_root_fd, _tombstone_root_opened = (
                            _open_authority_directory_at(
                                run_fd,
                                _TOMBSTONE_DIRECTORY,
                                label=(
                                    "journal recovery tombstone directory"
                                ),
                            )
                        )
                        try:
                            os.stat(
                                request.transaction_id,
                                dir_fd=tombstone_root_fd,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            tombstone_present = False
                        else:
                            tombstone_present = True
                        finally:
                            os.close(tombstone_root_fd)
                    if tombstone_present != candidate.tombstone_present:
                        raise MutationJournalError(
                            "external recovery cleanup tombstone inventory "
                            "changed before audit repair"
                        )
                    expected_event = copy.deepcopy(event)
                    if not tombstone_present:
                        expected_event["stage"] = "complete"
                    expected_payload = (
                        audit_payload
                        if expected_event == event
                        else _encode_recovery_event(expected_event)
                    )
                    audit_root_fd, _audit_root_opened = (
                        _open_authority_directory_at(
                            run_fd,
                            "journal_recovery",
                            label="external recovery audit directory",
                        )
                    )
                    audit_name = f"{request.transaction_id}.json"
                    audit_matches = False
                    try:
                        with _bound_recovery_audit(
                            audit_root_fd,
                            audit_name,
                        ) as (observed, _fd, _opened, _sha256):
                            audit_matches = observed == expected_event
                    except MutationJournalError:
                        try:
                            metadata = os.stat(
                                audit_name,
                                dir_fd=audit_root_fd,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            pass
                        else:
                            if (
                                not stat.S_ISREG(metadata.st_mode)
                                or stat.S_ISLNK(metadata.st_mode)
                                or metadata.st_nlink != 1
                            ):
                                raise MutationJournalError(
                                    "external recovery audit replacement is "
                                    "unsafe"
                                )
                    if tombstone_present and not audit_matches:
                        raise MutationJournalError(
                            "external recovery audit changed while its "
                            "cleanup tombstone remains"
                        )
                    if not audit_matches:
                        _require_bound_recovery_audit_unchanged(
                            receipt_root_fd,
                            candidate.receipt_name,
                            receipt_fd,
                            receipt_opened,
                            receipt_sha256,
                            max_bytes=(
                                _MAX_RECOVERY_CLEANUP_RECEIPT_BYTES
                            ),
                        )
                        write_atomic_bytes(
                            Path(f"/proc/self/fd/{audit_root_fd}")
                            / audit_name,
                            expected_payload,
                        )
                        with _bound_recovery_audit(
                            audit_root_fd,
                            audit_name,
                        ) as (
                            restored,
                            _restored_fd,
                            _restored_opened,
                            restored_sha256,
                        ):
                            if (
                                restored != expected_event
                                or restored_sha256
                                != hashlib.sha256(expected_payload).hexdigest()
                            ):
                                raise MutationJournalError(
                                    "external recovery audit repair differs "
                                    "from cleanup receipt"
                                )
                        _require_bound_recovery_audit_unchanged(
                            receipt_root_fd,
                            candidate.receipt_name,
                            receipt_fd,
                            receipt_opened,
                            receipt_sha256,
                            max_bytes=(
                                _MAX_RECOVERY_CLEANUP_RECEIPT_BYTES
                            ),
                        )
                        repaired.append(request.transaction_id)
            finally:
                if audit_root_fd is not None:
                    os.close(audit_root_fd)
                if receipt_root_fd is not None:
                    os.close(receipt_root_fd)
                if run_fd is not None:
                    os.close(run_fd)
    return tuple(repaired)


def _inventory_external_recovery_epoch_members(
    corpus: "Corpus",
    source_root: str | Path,
    *,
    check_live_collisions: bool = True,
    members_preinstalled: bool = False,
) -> _ExternalRecoveryEpochInventory:
    """Preflight every applicable audit before any tombstone is mutated."""

    source = Path(source_root).resolve()
    spec_dir = Path(corpus.spec_dir).absolute()
    members_by_id: dict[str, _CertifiedExternalEpochMember] = {}
    audit_bindings: list[_ExternalRecoveryAuditBinding] = []
    spec_fd, _spec_opened = _open_authority_directory(
        spec_dir,
        label="Corpus Spec directory",
    )
    runs_fd: int | None = None
    try:
        try:
            os.stat("runs", dir_fd=spec_fd, follow_symlinks=False)
        except FileNotFoundError:
            return _ExternalRecoveryEpochInventory(members=(), audits=())
        runs_fd, _runs_opened = _open_authority_directory_at(
            spec_fd,
            "runs",
            label="Corpus runs directory",
        )
        run_names = _bounded_directory_names(
            runs_fd,
            label="Corpus runs directory",
            max_entries=_MUTATION_JOURNAL_MAX_RUNS,
        )
        for run_name in run_names:
            run_fd: int | None = None
            audit_root_fd: int | None = None
            tombstone_root_fd: int | None = None
            try:
                run_fd, _run_opened = _open_authority_directory_at(
                    runs_fd,
                    run_name,
                    label=f"Corpus run directory {run_name!r}",
                )
                try:
                    os.stat(
                        "journal_recovery",
                        dir_fd=run_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                audit_root_fd, _audit_root_opened = (
                    _open_authority_directory_at(
                        run_fd,
                        "journal_recovery",
                        label="journal recovery audit directory",
                    )
                )
                audit_names = _bounded_directory_names(
                    audit_root_fd,
                    label="journal recovery audit directory",
                    max_entries=_MUTATION_JOURNAL_MAX_NESTED_MEMBERS,
                )
                for audit_name in audit_names:
                    if not audit_name.endswith(".json"):
                        raise MutationJournalError(
                            "journal recovery audit directory contains an "
                            "unexpected entry"
                        )
                    with _bound_recovery_audit(
                        audit_root_fd,
                        audit_name,
                    ) as (
                        event,
                        audit_fd,
                        audit_opened,
                        audit_payload_sha256,
                    ):
                        if event.get("schema") != _ACCEPTANCE_SCHEMA:
                            continue
                        if event.get("stage") not in {
                            "tombstone_prepared",
                            "complete",
                        }:
                            continue
                        request = ExternalRecoveryRequest.from_json(
                            event["request"]  # type: ignore[arg-type]
                        )
                        if (
                            run_name != request.run_id
                            or audit_name
                            != f"{request.transaction_id}.json"
                        ):
                            raise MutationJournalError(
                                "journal recovery audit run binding changed"
                            )
                        event_source = Path(str(event.get("source_root", "")))
                        if (
                            not event_source.is_absolute()
                            or event_source.resolve() != event_source
                        ):
                            raise MutationJournalError(
                                "journal recovery audit source binding is "
                                "non-canonical"
                            )
                        if event_source != source:
                            continue
                        parent_slice_id = event.get("parent_slice_id")
                        if (
                            not isinstance(parent_slice_id, str)
                            or not parent_slice_id
                        ):
                            raise MutationJournalError(
                                "journal recovery audit parent-slice binding "
                                "is missing"
                            )
                        logical_run = spec_dir / "runs" / run_name
                        expected_tombstone = (
                            logical_run
                            / _TOMBSTONE_DIRECTORY
                            / request.transaction_id
                        )
                        if Path(str(event.get("tombstone", ""))) != (
                            expected_tombstone
                        ):
                            raise MutationJournalError(
                                "journal recovery tombstone binding changed"
                            )
                        original = Path(str(event.get("journal_dir", "")))
                        if (
                            not original.is_absolute()
                            or original.parent.parent != logical_run
                            or original.parent.name
                            not in _DURABLE_JOURNAL_ROOTS
                        ):
                            raise MutationJournalError(
                                "journal recovery original-directory binding "
                                "changed"
                            )
                        durable_fd, _durable_opened = (
                            _open_authority_directory_at(
                                run_fd,
                                original.parent.name,
                                label="durable mutation journal root",
                            )
                        )
                        try:
                            try:
                                os.stat(
                                    original.name,
                                    dir_fd=durable_fd,
                                    follow_symlinks=False,
                                )
                            except FileNotFoundError:
                                pass
                            else:
                                # The discoverable journal, not this audit,
                                # still owns ordinary recovery authority.
                                continue
                        finally:
                            os.close(durable_fd)
                        identity = event.get("tombstone_identity")
                        if (
                            not isinstance(identity, dict)
                            or set(identity) != {"device", "inode", "mode"}
                            or any(
                                isinstance(value, bool)
                                or not isinstance(value, int)
                                or value < 0
                                for value in identity.values()
                            )
                        ):
                            raise MutationJournalError(
                                "journal recovery tombstone identity is missing"
                            )
                        proof = _parse_external_writer_epoch_proof(
                            event,
                            transaction_id=request.transaction_id,
                        )
                        certified = tuple(
                            _CertifiedExternalEpochMember(
                                transaction_id=transaction_id,
                                parent_transaction_id=parent_transaction_id,
                                epoch_record=proof.epoch_record,
                            )
                            for transaction_id, parent_transaction_id
                            in proof.family
                        )
                        try:
                            require_writer_epoch_quiescence(
                                cast(
                                    tuple[WriterEpochStorage, ...],
                                    certified,
                                ),
                                allow_never_armed=True,
                            )
                        except WriterEpochError as exc:
                            raise MutationJournalError(
                                "external recovery writer epoch is not "
                                f"quiescent: {exc}"
                            ) from exc

                        try:
                            os.stat(
                                _TOMBSTONE_DIRECTORY,
                                dir_fd=run_fd,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            tombstone_present = False
                        else:
                            tombstone_present = True
                            tombstone_root_fd, _tombstone_root_opened = (
                                _open_authority_directory_at(
                                    run_fd,
                                    _TOMBSTONE_DIRECTORY,
                                    label=(
                                        "journal recovery tombstone directory"
                                    ),
                                )
                            )
                        observed_tombstone_identity: (
                            tuple[int, int, int] | None
                        ) = None
                        if tombstone_present:
                            assert tombstone_root_fd is not None
                            try:
                                metadata = os.stat(
                                    request.transaction_id,
                                    dir_fd=tombstone_root_fd,
                                    follow_symlinks=False,
                                )
                            except FileNotFoundError:
                                metadata = None
                            if metadata is not None:
                                if (
                                    not stat.S_ISDIR(metadata.st_mode)
                                    or stat.S_ISLNK(metadata.st_mode)
                                    or metadata.st_dev != identity["device"]
                                    or metadata.st_ino != identity["inode"]
                                    or stat.S_IMODE(metadata.st_mode)
                                    != identity["mode"]
                                ):
                                    raise MutationJournalError(
                                        "journal recovery tombstone identity "
                                        "changed"
                                    )
                                observed_tombstone_identity = (
                                    metadata.st_dev,
                                    metadata.st_ino,
                                    stat.S_IMODE(metadata.st_mode),
                                )
                                tombstone_fd, tombstone_opened = (
                                    _open_authority_directory_at(
                                        tombstone_root_fd,
                                        request.transaction_id,
                                        label=(
                                            "bound journal recovery tombstone"
                                        ),
                                    )
                                )
                                try:
                                    if (
                                        tombstone_opened.st_dev,
                                        tombstone_opened.st_ino,
                                    ) != (
                                        metadata.st_dev,
                                        metadata.st_ino,
                                    ):
                                        raise MutationJournalError(
                                            "journal recovery tombstone changed "
                                            "while opening"
                                        )
                                    _require_bound_external_writer_epoch(
                                        journal_path=expected_tombstone,
                                        journal_fd=tombstone_fd,
                                        event=event,
                                        transaction_id=(
                                            request.transaction_id
                                        ),
                                        members_preinstalled=(
                                            members_preinstalled
                                        ),
                                    )
                                finally:
                                    os.close(tombstone_fd)
                        if tombstone_root_fd is not None:
                            os.close(tombstone_root_fd)
                            tombstone_root_fd = None
                        _require_bound_recovery_audit_unchanged(
                            audit_root_fd,
                            audit_name,
                            audit_fd,
                            audit_opened,
                            audit_payload_sha256,
                        )
                        audit_bindings.append(
                            _ExternalRecoveryAuditBinding(
                                run_name=run_name,
                                audit_name=audit_name,
                                audit_identity=(
                                    audit_opened.st_dev,
                                    audit_opened.st_ino,
                                    audit_opened.st_mode,
                                    audit_opened.st_nlink,
                                    audit_opened.st_size,
                                    audit_opened.st_mtime_ns,
                                    audit_opened.st_ctime_ns,
                                ),
                                audit_sha256=audit_payload_sha256,
                                tombstone_identity=(
                                    observed_tombstone_identity
                                ),
                            )
                        )
                        for member in certified:
                            if member.transaction_id in members_by_id:
                                raise MutationJournalError(
                                    "external recovery writer epoch inventory "
                                    "contains duplicate or overlapping "
                                    f"transaction {member.transaction_id}"
                                )
                            members_by_id[member.transaction_id] = member
            finally:
                if tombstone_root_fd is not None:
                    os.close(tombstone_root_fd)
                if audit_root_fd is not None:
                    os.close(audit_root_fd)
                if run_fd is not None:
                    os.close(run_fd)
    finally:
        if runs_fd is not None:
            os.close(runs_fd)
        os.close(spec_fd)
    members = tuple(members_by_id.values())
    # Reject collisions with discovered candidates, receipts, or live owners
    # while this pass remains read-only. The complete map is reinstalled for
    # the second pass so no event can shadow another event's authority.
    if check_live_collisions:
        with _installed_writer_epoch_recovery_members(
            cast(tuple[object, ...], members)
        ):
            pass
    return _ExternalRecoveryEpochInventory(
        members=members,
        audits=tuple(sorted(
            audit_bindings,
            key=lambda binding: (binding.run_name, binding.audit_name),
        )),
    )


def _reap_external_recovery_tombstones_bound(
    corpus: "Corpus",
    source_root: str | Path,
    inventory: _ExternalRecoveryEpochInventory,
) -> tuple[str, ...]:
    """Finish deletion after an atomically tombstoned recovery journal."""

    confirmed = _inventory_external_recovery_epoch_members(
        corpus,
        source_root,
        check_live_collisions=False,
        members_preinstalled=True,
    )
    if confirmed != inventory:
        raise MutationJournalError(
            "external recovery global inventory changed before bound deletion"
        )
    expected_audits = {
        (binding.run_name, binding.audit_name): binding
        for binding in inventory.audits
    }
    seen_audits: set[tuple[str, str]] = set()
    source = Path(source_root).resolve()
    recovered: list[str] = []
    spec_dir = Path(corpus.spec_dir).absolute()
    spec_fd, _spec_opened = _open_authority_directory(
        spec_dir,
        label="Corpus Spec directory",
    )
    runs_fd: int | None = None
    try:
        try:
            os.stat("runs", dir_fd=spec_fd, follow_symlinks=False)
        except FileNotFoundError:
            return ()
        runs_fd, _runs_opened = _open_authority_directory_at(
            spec_fd,
            "runs",
            label="Corpus runs directory",
        )
        run_names = _bounded_directory_names(
            runs_fd,
            label="Corpus runs directory",
            max_entries=_MUTATION_JOURNAL_MAX_RUNS,
        )
        for run_name in run_names:
            run_fd: int | None = None
            audit_root_fd: int | None = None
            tombstone_root_fd: int | None = None
            try:
                run_fd, _run_opened = _open_authority_directory_at(
                    runs_fd,
                    run_name,
                    label=f"Corpus run directory {run_name!r}",
                )
                try:
                    os.stat(
                        "journal_recovery",
                        dir_fd=run_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                audit_root_fd, _audit_root_opened = (
                    _open_authority_directory_at(
                        run_fd,
                        "journal_recovery",
                        label="journal recovery audit directory",
                    )
                )
                audit_names = _bounded_directory_names(
                    audit_root_fd,
                    label="journal recovery audit directory",
                    max_entries=_MUTATION_JOURNAL_MAX_NESTED_MEMBERS,
                )
                for audit_name in audit_names:
                    if not audit_name.endswith(".json"):
                        raise MutationJournalError(
                            "journal recovery audit directory contains an "
                            "unexpected entry"
                        )
                    with _bound_recovery_audit(
                        audit_root_fd,
                        audit_name,
                    ) as (
                        event,
                        audit_fd,
                        audit_opened,
                        audit_payload_sha256,
                    ):
                        if event.get("schema") != _ACCEPTANCE_SCHEMA:
                            continue
                        if event.get("stage") not in {
                            "tombstone_prepared",
                            "complete",
                        }:
                            continue
                        request = ExternalRecoveryRequest.from_json(
                            event["request"]  # type: ignore[arg-type]
                        )
                        if (
                            run_name != request.run_id
                            or audit_name
                            != f"{request.transaction_id}.json"
                        ):
                            raise MutationJournalError(
                                "journal recovery audit run binding changed"
                            )
                        event_source = Path(str(event.get("source_root", "")))
                        if (
                            not event_source.is_absolute()
                            or event_source.resolve() != event_source
                        ):
                            raise MutationJournalError(
                                "journal recovery audit source binding is "
                                "non-canonical"
                            )
                        if event_source != source:
                            continue
                        audit_key = (run_name, audit_name)
                        expected_audit = expected_audits.get(audit_key)
                        if expected_audit is None:
                            # A post-preflight event is never allowed to reuse
                            # an installed transaction certificate.  It remains
                            # untouched for the next complete inventory pass.
                            continue
                        parent_slice_id = event.get("parent_slice_id")
                        if (
                            not isinstance(parent_slice_id, str)
                            or not parent_slice_id
                        ):
                            raise MutationJournalError(
                                "journal recovery audit parent-slice binding "
                                "is missing"
                            )
                        logical_run = spec_dir / "runs" / run_name
                        expected_tombstone = (
                            logical_run
                            / _TOMBSTONE_DIRECTORY
                            / request.transaction_id
                        )
                        if Path(str(event.get("tombstone", ""))) != (
                            expected_tombstone
                        ):
                            raise MutationJournalError(
                                "journal recovery tombstone binding changed"
                            )
                        original = Path(str(event.get("journal_dir", "")))
                        if (
                            not original.is_absolute()
                            or original.parent.parent != logical_run
                            or original.parent.name
                            not in _DURABLE_JOURNAL_ROOTS
                        ):
                            raise MutationJournalError(
                                "journal recovery original-directory binding "
                                "changed"
                            )
                        durable_fd, _durable_opened = (
                            _open_authority_directory_at(
                                run_fd,
                                original.parent.name,
                                label="durable mutation journal root",
                            )
                        )
                        try:
                            try:
                                os.stat(
                                    original.name,
                                    dir_fd=durable_fd,
                                    follow_symlinks=False,
                                )
                            except FileNotFoundError:
                                pass
                            else:
                                # Rename has not occurred. Ordinary recovery
                                # still owns the discoverable journal.
                                continue
                        finally:
                            os.close(durable_fd)
                        identity = event.get("tombstone_identity")
                        if not isinstance(identity, dict) or set(identity) != {
                            "device",
                            "inode",
                            "mode",
                        }:
                            raise MutationJournalError(
                                "journal recovery tombstone identity is missing"
                            )
                        _require_audit_writer_epoch_proof(
                            event,
                            transaction_id=request.transaction_id,
                            members_preinstalled=True,
                        )
                        try:
                            os.stat(
                                _TOMBSTONE_DIRECTORY,
                                dir_fd=run_fd,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            tombstone_present = False
                        else:
                            tombstone_present = True
                            tombstone_root_fd, _tombstone_root_opened = (
                                _open_authority_directory_at(
                                    run_fd,
                                    _TOMBSTONE_DIRECTORY,
                                    label="journal recovery tombstone directory",
                                )
                            )
                        changed = False
                        if tombstone_present:
                            assert tombstone_root_fd is not None
                            try:
                                metadata = os.stat(
                                    request.transaction_id,
                                    dir_fd=tombstone_root_fd,
                                    follow_symlinks=False,
                                )
                            except FileNotFoundError:
                                metadata = None
                            if metadata is not None:
                                if (
                                    not stat.S_ISDIR(metadata.st_mode)
                                    or stat.S_ISLNK(metadata.st_mode)
                                    or metadata.st_dev
                                    != int(identity["device"])
                                    or metadata.st_ino
                                    != int(identity["inode"])
                                    or stat.S_IMODE(metadata.st_mode)
                                    != int(identity["mode"])
                                ):
                                    raise MutationJournalError(
                                        "journal recovery tombstone identity "
                                        "changed"
                                    )
                                current_tombstone_identity: (
                                    tuple[int, int, int] | None
                                ) = (
                                    metadata.st_dev,
                                    metadata.st_ino,
                                    stat.S_IMODE(metadata.st_mode),
                                )
                            else:
                                current_tombstone_identity = None
                        else:
                            current_tombstone_identity = None
                        current_audit_binding = _ExternalRecoveryAuditBinding(
                            run_name=run_name,
                            audit_name=audit_name,
                            audit_identity=(
                                audit_opened.st_dev,
                                audit_opened.st_ino,
                                audit_opened.st_mode,
                                audit_opened.st_nlink,
                                audit_opened.st_size,
                                audit_opened.st_mtime_ns,
                                audit_opened.st_ctime_ns,
                            ),
                            audit_sha256=audit_payload_sha256,
                            tombstone_identity=current_tombstone_identity,
                        )
                        if current_audit_binding != expected_audit:
                            raise MutationJournalError(
                                "external recovery bound audit inventory changed"
                            )
                        seen_audits.add(audit_key)
                        if tombstone_present:
                            assert tombstone_root_fd is not None
                            if metadata is not None:
                                current_audit = os.stat(
                                    audit_name,
                                    dir_fd=audit_root_fd,
                                    follow_symlinks=False,
                                )
                                if (current_audit.st_dev, current_audit.st_ino) != (
                                    audit_opened.st_dev,
                                    audit_opened.st_ino,
                                ) or (os.fstat(audit_fd).st_dev, os.fstat(audit_fd).st_ino) != (
                                    audit_opened.st_dev,
                                    audit_opened.st_ino,
                                ):
                                    raise MutationJournalError(
                                        "journal recovery audit changed before "
                                        "tombstone cleanup"
                                    )
                                tombstone_fd, tombstone_opened = (
                                    _open_authority_directory_at(
                                        tombstone_root_fd,
                                        request.transaction_id,
                                        label=(
                                            "bound journal recovery tombstone"
                                        ),
                                    )
                                )
                                try:
                                    if (
                                        tombstone_opened.st_dev,
                                        tombstone_opened.st_ino,
                                    ) != (
                                        metadata.st_dev,
                                        metadata.st_ino,
                                    ):
                                        raise MutationJournalError(
                                            "journal recovery tombstone changed "
                                            "while opening"
                                        )
                                    _require_bound_external_writer_epoch(
                                        journal_path=expected_tombstone,
                                        journal_fd=tombstone_fd,
                                        event=event,
                                        transaction_id=(
                                            request.transaction_id
                                        ),
                                        members_preinstalled=True,
                                    )
                                    _require_bound_recovery_audit_unchanged(
                                        audit_root_fd,
                                        audit_name,
                                        audit_fd,
                                        audit_opened,
                                        audit_payload_sha256,
                                    )
                                    assert run_fd is not None
                                    with _bound_external_cleanup_receipt(
                                        run_fd=run_fd,
                                        logical_run=logical_run,
                                        audit_root_fd=audit_root_fd,
                                        audit_name=audit_name,
                                        audit_fd=audit_fd,
                                        audit_opened=audit_opened,
                                        audit_payload_sha256=(
                                            audit_payload_sha256
                                        ),
                                        event=event,
                                    ) as cleanup_receipt:
                                        def cleanup_guard(
                                            audit_root_fd: int = audit_root_fd,
                                            audit_name: str = audit_name,
                                            audit_fd: int = audit_fd,
                                            audit_opened: os.stat_result = (
                                                audit_opened
                                            ),
                                            audit_payload_sha256: str = (
                                                audit_payload_sha256
                                            ),
                                            cleanup_receipt: (
                                                _BoundExternalCleanupReceipt
                                            ) = cleanup_receipt,
                                        ) -> None:
                                            _require_bound_recovery_audit_unchanged(
                                                audit_root_fd,
                                                audit_name,
                                                audit_fd,
                                                audit_opened,
                                                audit_payload_sha256,
                                            )
                                            _require_external_cleanup_receipt_unchanged(
                                                cleanup_receipt
                                            )

                                        _remove_bound_external_tombstone(
                                            tombstone_root_fd,
                                            request.transaction_id,
                                            tombstone_fd,
                                            tombstone_opened,
                                            event=event,
                                            pre_remove_guard=cleanup_guard,
                                        )
                                        _reconcile_external_cleanup_audit(
                                            receipt=cleanup_receipt,
                                            audit_root_fd=audit_root_fd,
                                            audit_name=audit_name,
                                            audit_fd=audit_fd,
                                            audit_opened=audit_opened,
                                            audit_payload_sha256=(
                                                audit_payload_sha256
                                            ),
                                        )
                                finally:
                                    os.close(tombstone_fd)
                                changed = True
                        if event.get("stage") != "complete":
                            _require_bound_recovery_audit_unchanged(
                                audit_root_fd,
                                audit_name,
                                audit_fd,
                                audit_opened,
                                audit_payload_sha256,
                            )
                            event["stage"] = "complete"
                            write_atomic_bytes(
                                Path(f"/proc/self/fd/{audit_root_fd}")
                                / audit_name,
                                _encode_recovery_event(event),
                            )
                            changed = True
                        if changed:
                            recovered.append(parent_slice_id)
                        if tombstone_root_fd is not None:
                            os.close(tombstone_root_fd)
                            tombstone_root_fd = None
            finally:
                if tombstone_root_fd is not None:
                    os.close(tombstone_root_fd)
                if audit_root_fd is not None:
                    os.close(audit_root_fd)
                if run_fd is not None:
                    os.close(run_fd)
    finally:
        if runs_fd is not None:
            os.close(runs_fd)
        os.close(spec_fd)
    if seen_audits != set(expected_audits):
        raise MutationJournalError(
            "external recovery bound audit inventory is incomplete"
        )
    return tuple(recovered)


def reap_external_recovery_tombstones(
    corpus: "Corpus",
    source_root: str | Path,
) -> tuple[str, ...]:
    """Globally preflight epoch authority, then revalidate and reap."""

    _reconcile_external_cleanup_receipts(corpus, source_root)
    inventory = _inventory_external_recovery_epoch_members(
        corpus,
        source_root,
    )
    confirmed = _inventory_external_recovery_epoch_members(
        corpus,
        source_root,
    )
    if confirmed != inventory:
        raise MutationJournalError(
            "external recovery global inventory changed before deletion"
        )
    with _installed_writer_epoch_recovery_members(
        cast(tuple[object, ...], inventory.members)
    ):
        return _reap_external_recovery_tombstones_bound(
            corpus,
            source_root,
            inventory,
        )


def _inspect_external_recovery(
    corpus: "Corpus",
    source_root: str | Path,
    request: ExternalRecoveryRequest,
    *,
    apply: bool = False,
    journal: WorkspaceMutationJournal | None = None,
) -> ExternalRecoveryReport:
    """Check or apply recovery using fresh or startup-bound authority."""

    source = Path(source_root).resolve()
    request = request.normalized()
    startup_bound = journal is not None
    if journal is None:
        journal = _find_journal(corpus, source, request)
    else:
        _validate_recovery_root(corpus, source, request, journal)
    event = _load_event(journal)
    if event is not None and journal._discovery_binding is not None:
        with _bound_mutation_journal_directory(journal):
            pass
    if event is None:
        inspection = _inspect(
            corpus,
            source,
            request,
            journal=journal,
        )
        if not apply:
            return ExternalRecoveryReport(
                status="ready",
                run_id=request.run_id,
                transaction_id=request.transaction_id,
                source_root=str(source),
                removed_linked_worktrees=request.removed_linked_worktrees,
                removed_peer_transactions=tuple(
                    peer.transaction_id for peer in inspection.removed_peers
                ),
                baseline_stash_tip=inspection.baseline_stash_tip,
                appended_stash_tip=request.appended_stash_tip,
                object_addition_oids=inspection.object_addition_oids,
                expected_diff_ref=request.expected_diff_ref,
                ancestor_transaction_id=(
                    inspection.accepted_ancestor.journal.transaction_id
                    if inspection.accepted_ancestor is not None
                    else None
                ),
                ancestor_root=(
                    inspection.accepted_ancestor.request.ancestor_root
                    if inspection.accepted_ancestor is not None
                    else None
                ),
                added_submodule_path=(
                    inspection.accepted_ancestor.request.submodule_path
                    if inspection.accepted_ancestor is not None
                    else None
                ),
                ancestor_gitmodules_blob_oid=(
                    inspection.accepted_ancestor.gitmodules_blob_oid
                    if inspection.accepted_ancestor is not None
                    else None
                ),
            )
        if journal._discovery_binding is None:
            raise MutationJournalError(
                "external recovery lacks bound journal authority"
            )
        original_dir = journal.journal_dir
        expected_postimage_artifacts: dict[
            str,
            tuple[str, str | None],
        ] = {}
        with refresh_mutation_journal_artifact_binding(
            journal,
            expected_artifacts=expected_postimage_artifacts,
            exclusive_prefixes=(_RECOVERY_DIRECTORY,),
        ) as (
            artifact_dir,
            journal_fd,
        ):
            assert journal_fd is not None
            journal.journal_dir = artifact_dir
            try:
                event = _new_event(inspection, journal_fd)
                postimage_records = _validated_postimage_records(
                    journal,
                    event,
                    journal_fd=journal_fd,
                )
                if not _all_paths_match_postimage(
                    journal,
                    event,
                    journal_fd=journal_fd,
                ):
                    raise MutationJournalError(
                        "captured recovery postimage differs from its "
                        "authored source bytes"
                    )
                expected_postimage_artifacts.update({
                    _RECOVERY_DIRECTORY: ("directory", None),
                    f"{_RECOVERY_DIRECTORY}/postimage": (
                        "directory",
                        None,
                    ),
                })
                for record in postimage_records.values():
                    if record.get("kind") != "file":
                        continue
                    snapshot = str(record["snapshot"])
                    expected_postimage_artifacts[
                        f"{_RECOVERY_DIRECTORY}/postimage/{snapshot}"
                    ] = ("file", str(record["sha256"]))
            finally:
                journal.journal_dir = original_dir
        _write_event(journal, event)
        journal._manifest["state"] = "external_recovery_prepared"
        journal._write()
    else:
        durable = ExternalRecoveryRequest.from_json(event["request"])  # type: ignore[arg-type]
        if durable != request:
            raise MutationJournalError(
                "CLI authority differs from durable recovery acceptance"
            )
        if not apply:
            durable_stage = str(event.get("stage", ""))
            inspection = _inspect(
                corpus,
                source,
                request,
                journal=journal,
                allow_partially_restored_source=True,
                allow_missing_family_markers=durable_stage
                in {"verified", "tombstone_prepared"},
            )
            _event_matches_inspection(event, inspection)
            _assert_source_for_stage(journal, event, durable_stage)
            report = ExternalRecoveryReport(
                status=durable_stage,
                run_id=request.run_id,
                transaction_id=request.transaction_id,
                source_root=str(source),
                removed_linked_worktrees=request.removed_linked_worktrees,
                removed_peer_transactions=tuple(
                    peer.transaction_id for peer in inspection.removed_peers
                ),
                baseline_stash_tip=inspection.baseline_stash_tip,
                appended_stash_tip=request.appended_stash_tip,
                object_addition_oids=inspection.object_addition_oids,
                expected_diff_ref=request.expected_diff_ref,
                ancestor_transaction_id=(
                    inspection.accepted_ancestor.journal.transaction_id
                    if inspection.accepted_ancestor is not None
                    else None
                ),
                ancestor_root=(
                    inspection.accepted_ancestor.request.ancestor_root
                    if inspection.accepted_ancestor is not None
                    else None
                ),
                added_submodule_path=(
                    inspection.accepted_ancestor.request.submodule_path
                    if inspection.accepted_ancestor is not None
                    else None
                ),
                ancestor_gitmodules_blob_oid=(
                    inspection.accepted_ancestor.gitmodules_blob_oid
                    if inspection.accepted_ancestor is not None
                    else None
                ),
                audit_path=str(_audit_path(journal, create=False)),
            )
            return report

    durable_stage = str(event.get("stage", ""))
    inspection = _inspect(
        corpus,
        source,
        request,
        journal=journal,
        allow_partially_restored_source=True,
        allow_missing_family_markers=durable_stage
        in {"verified", "tombstone_prepared"},
    )
    _event_matches_inspection(event, inspection)
    _assert_source_for_stage(journal, event, durable_stage)
    stage = durable_stage
    if stage == "prepared":
        with _installed_writer_epoch_recovery_members(
            cast(tuple[object, ...], inspection.family)
        ):
            # On the first attempt, bind the captured path snapshots back to
            # the exact Corpus diff immediately before the first source write.
            # A resumed partially-restored attempt instead uses the per-path
            # preimage/postimage union proven above.
            if _all_paths_match_postimage(journal, event):
                _source_matches_expected_diff(
                    corpus,
                    journal,
                    request.expected_diff_ref,
                )
            _restore_source_only(journal)
        event["stage"] = "source_restored"
        _write_event(journal, event)
        stage = "source_restored"
    if stage == "source_restored":
        with _installed_writer_epoch_recovery_members(
            cast(tuple[object, ...], inspection.family)
        ):
            _assert_parent_overlay(inspection)
            removed_ids = {
                peer.transaction_id for peer in inspection.removed_peers
            }
            accepted_ancestor_id = (
                inspection.accepted_ancestor.journal.transaction_id
                if inspection.accepted_ancestor is not None
                else None
            )
            for member in inspection.family[1:]:
                if (
                    member.transaction_id not in removed_ids
                    and member.transaction_id != accepted_ancestor_id
                ):
                    _assert_unaffected_journal(member)
            if inspection.accepted_ancestor is not None:
                _assert_accepted_ancestor_overlay(
                    inspection.accepted_ancestor
                )
        _assert_registry_cleanup_safe(inspection.family, allow_missing=False)
        event["stage"] = "verified"
        _write_event(journal, event)
        journal._manifest["state"] = "external_recovery_verified"
        journal._write()
        stage = "verified"
    if stage not in {"verified", "tombstone_prepared"}:
        raise MutationJournalError(f"unsupported external recovery stage {stage!r}")
    with _installed_writer_epoch_recovery_members(
        cast(tuple[object, ...], inspection.family)
    ):
        _assert_source_for_stage(journal, event, stage)
        _assert_parent_overlay(
            inspection,
            ignore_registry_marker=stage in {"verified", "tombstone_prepared"},
        )
        if inspection.accepted_ancestor is not None:
            _assert_accepted_ancestor_overlay(
                inspection.accepted_ancestor,
                ignore_registry_marker=stage
                in {"verified", "tombstone_prepared"},
            )
    return _cleanup_verified(
        inspection,
        event,
        members_preinstalled=startup_bound,
    )


def inspect_external_recovery(
    corpus: "Corpus",
    source_root: str | Path,
    request: ExternalRecoveryRequest,
    *,
    apply: bool = False,
) -> ExternalRecoveryReport:
    """Check or apply one exact external Git journal recovery."""

    return _inspect_external_recovery(
        corpus,
        source_root,
        request,
        apply=apply,
    )


def preflight_external_recovery_authority(
    corpus: "Corpus",
    source_root: str | Path,
    request: ExternalRecoveryRequest,
) -> None:
    """Validate config-bearing external authority before acquiring a Git lock."""

    source = Path(source_root).resolve()
    request = request.normalized()
    journal = _find_journal(corpus, source, request)
    accepted = request.ancestor_submodule
    if accepted is None:
        return
    if (
        load_accepted_ancestor_submodule(corpus, accepted.acceptance_ref)
        != accepted
        or accepted.run_id != request.run_id
        or accepted.transaction_id != journal.transaction_id
    ):
        raise MutationJournalError(
            "accepted ancestor manifest differs from the requested transaction"
        )
    _preflight_accepted_ancestor_config(journal, accepted)


def resume_prepared_external_recovery(
    corpus: "Corpus",
    journal: WorkspaceMutationJournal,
    *,
    _startup_inventory: tuple[WorkspaceMutationJournal, ...] | None = None,
) -> ExternalRecoveryReport | None:
    """Resume a durable acceptance; return ``None`` when no acceptance exists."""

    event = _load_event(journal)
    if event is None:
        return None
    request = ExternalRecoveryRequest.from_json(event["request"])  # type: ignore[arg-type]
    if request.transaction_id != journal.transaction_id:
        raise MutationJournalError("durable recovery transaction binding changed")
    source = Path(journal._manifest["cwd"])
    if _startup_inventory is None:
        return inspect_external_recovery(
            corpus,
            source,
            request,
            apply=True,
        )
    startup_journal = _startup_resume_journal(
        corpus,
        source.resolve(),
        request,
        journal,
        _startup_inventory,
    )
    return _inspect_external_recovery(
        corpus,
        source,
        request,
        apply=True,
        journal=startup_journal,
    )


__all__ = [
    "AcceptedAncestorSubmodule",
    "ExternalRecoveryReport",
    "ExternalRecoveryRequest",
    "inspect_external_recovery",
    "load_accepted_ancestor_submodule",
    "preflight_external_recovery_authority",
    "reap_external_recovery_tombstones",
    "resume_prepared_external_recovery",
]
