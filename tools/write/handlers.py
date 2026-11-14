# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Scoped text-edit and binary-artifact write tool handlers.

These tools mutate files in the workspace and implement the ``assay.ToolSet``
protocol with schemas declared inline via ``@builder.tool``.

``write_source`` replaces a file's full contents. ``apply_patch`` does
a unique-match before/after replacement, returning an error if the
``before`` text is missing or matches multiple locations (the model
must add surrounding context to disambiguate). ``delete_source`` removes
only an exact, explicitly declared write-scope root; directory removal
requires an explicit recursive acknowledgement.

``materialize_artifact`` copies an already-built regular file from the
active repository/worktree into the authorized write surface without
round-tripping binary bytes through a model response.

Path resolution goes through ``crucible.tools._paths`` — strict
in-repo by default, worktree-aware if ``ctx.path_resolver`` is set.
"""

from __future__ import annotations

import hashlib
import ctypes
import errno
import logging
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._builder import ToolSetBuilder
from .._context import ToolContext
from .._dispatch_budget import ToolMutationIndeterminate
from .._paths import (
    _VCS_METADATA_COMPONENTS,
    _context_relative,
    resolve_for_read,
    resolve_for_write,
)

logger = logging.getLogger(__name__)


builder = ToolSetBuilder("write")

_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_PUBLICATION_TOKEN_HEX_LENGTH = 32
_MAX_DELETE_ENTRIES = 100_000
_MAX_DELETE_DEPTH = 256
_MAX_DELETE_REGULAR_BYTES = 512 * 1024 * 1024
_DELETE_QUARANTINE_PAYLOAD = "payload"


class DeleteMutationIndeterminate(ToolMutationIndeterminate):
    """A quarantined deletion needs host settlement and journal rollback.

    This is deliberately not converted into an ordinary model-visible tool
    error by the isolated mutation child.  Once the public namespace has been
    mutated, continuing the provider conversation before process settlement
    and workspace-journal rollback would grant authority over an unproved
    checkout state.
    """


def _validate_publication_token(token: str | None) -> str:
    """Return one bounded token suitable for an adjacent staging filename."""

    if token is None:
        return secrets.token_hex(_PUBLICATION_TOKEN_HEX_LENGTH // 2)
    if (
        not isinstance(token, str)
        or len(token) != _PUBLICATION_TOKEN_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in token)
    ):
        raise ValueError("invalid atomic-publication token")
    return token


def _atomic_stage_path(path: Path, publication_token: str) -> Path:
    """Return the exact adjacent staging path used for one publication."""

    return path.parent / (
        f".{path.name}.crucible-{publication_token}.tmp"
    )


def _delete_quarantine_path(path: Path, publication_token: str) -> Path:
    """Return the exact sibling quarantine owned by one hosted deletion."""

    return path.parent / (
        f".{path.name}.crucible-delete-{publication_token}.quarantine"
    )


def _rename_noreplace_at(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename one descriptor-relative entry without replacement."""

    for name in (source_name, destination_name):
        if name in {"", ".", ".."} or "/" in name or "\0" in name:
            raise ValueError("unsafe atomic rename entry name")
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise ValueError(
            "atomic no-replace delete quarantine is unavailable"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_fd,
        os.fsencode(source_name),
        destination_fd,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise ValueError(
            "atomic no-replace delete quarantine is unsupported"
        )
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error,
            os.strerror(error),
            destination_name,
        )
    raise OSError(error, os.strerror(error), destination_name)


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write every byte, treating a zero-length write as an I/O failure."""

    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write during atomic text publication")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_text_atomic(
    path: Path,
    content: str,
    *,
    expected_current_sha256: str | None,
    publication_token: str | None,
    publication_barrier: Callable[[], None] | None = None,
) -> None:
    """Crash-atomically publish UTF-8 text with a final stale-state check.

    The complete replacement is written and fsynced in an adjacent private
    file before ``os.replace`` makes it visible.  Existing-file mode bits are
    retained.  A second whole-file digest immediately before publication
    preserves the caller's stale-read guard across the staging interval.
    """

    token = _validate_publication_token(publication_token)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = _atomic_stage_path(path, token)
    existing_mode: int | None = None
    try:
        observed = path.stat()
    except FileNotFoundError:
        observed = None
    if observed is not None:
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise ValueError(f"unsafe replacement target: {path.name}")
        existing_mode = stat.S_IMODE(observed.st_mode)

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
    )
    descriptor = os.open(stage, flags, 0o600)
    try:
        if existing_mode is not None:
            os.fchmod(descriptor, existing_mode)
        _write_all(descriptor, content.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    try:
        if publication_barrier is not None:
            # The process host uses this point to attest that complete bytes
            # are durable in staging and to re-check the logical deadline.
            # Revalidate the destination only after that acknowledgement so
            # the IPC wait cannot widen the stale-read race.
            publication_barrier()
        create_without_replace = expected_current_sha256 is None
        if create_without_replace:
            if path.exists():
                raise ValueError(
                    f"destination appeared during publication: {path.name}"
                )
        else:
            try:
                current_bytes = path.read_bytes()
            except FileNotFoundError as exc:
                raise ValueError(
                    f"destination disappeared during publication: {path.name}"
                ) from exc
            if hashlib.sha256(current_bytes).hexdigest() != (
                expected_current_sha256
            ):
                raise ValueError(
                    f"stale file changed during publication: {path.name}"
                )
        if create_without_replace:
            try:
                os.link(stage, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise ValueError(
                    f"destination appeared during publication: {path.name}"
                ) from exc
            stage.unlink()
        else:
            os.replace(stage, path)
        _fsync_directory(path.parent)
    finally:
        try:
            stage.unlink()
        except FileNotFoundError:
            pass


@builder.tool(
    description=(
        "Write content to a source file. Provide the COMPLETE file "
        "content -- this tool replaces the entire file. Use read_source "
        "first to get the current content, then modify and write back. "
        "Best for files under 500 lines. Every existing-file replacement "
        "requires expected_sha256 from a complete read_source response; use "
        "apply_patch for ordinary targeted edits."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path relative to repo root (e.g. '<project>/source/foo.py')",
            },
            "content": {
                "type": "string",
                "description": "Complete file content to write",
            },
            "expected_sha256": {
                "type": "string",
                "description": (
                    "Current whole-file SHA-256 from read_source; required "
                    "when replacing any existing file"
                ),
            },
        },
        "required": ["file_path", "content"],
    },
    write=True,
)
def write_source(
    ctx: ToolContext,
    *,
    file_path: str,
    content: str,
    expected_sha256: str = "",
    **_: Any,
) -> str:
    """Write *content* to *file_path*, replacing any existing contents."""
    return _write_source_impl(
        ctx,
        file_path=file_path,
        content=content,
        expected_sha256=expected_sha256,
        publication_token=None,
        publication_barrier=None,
    )


def _write_source_impl(
    ctx: ToolContext,
    *,
    file_path: str,
    content: str,
    expected_sha256: str = "",
    publication_token: str | None,
    publication_barrier: Callable[[], None] | None = None,
) -> str:
    """Canonical implementation with an optional host-owned stage token."""

    path = resolve_for_write(ctx, file_path)
    if path is None:
        return f"Error: cannot resolve write path for '{file_path}'"
    observed_sha256: str | None = None
    if path.is_file():
        try:
            original_bytes = path.read_bytes()
            original = original_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return f"Error reading {file_path} before write: {exc}"
        observed_sha256 = hashlib.sha256(original_bytes).hexdigest()
        if expected_sha256 and expected_sha256 != observed_sha256:
            return (
                f"Error: stale expected_sha256 for {file_path}; re-read the "
                "file before replacing it."
            )
        replacement_bytes = content.encode("utf-8")
        if replacement_bytes == original_bytes:
            return (
                f"Error: replacement for {file_path} is identical to the "
                "current file; write_source requires an actual edit."
            )
        original_lines = len(original.splitlines())
        replacement_lines = len(content.splitlines())
        shrinks_existing_file = (
            len(replacement_bytes) < len(original_bytes)
            or replacement_lines < original_lines
        )
        if not expected_sha256:
            if shrinks_existing_file:
                return (
                    f"Error: refusing to shrink existing file {file_path} "
                    f"from {len(original_bytes)} to {len(replacement_bytes)} "
                    f"bytes ({original_lines} to {replacement_lines} lines) "
                    "without an exact expected_sha256 from read_source. Use "
                    "apply_patch for a targeted deletion or re-read the "
                    "complete file."
                )
            return (
                f"Error: refusing to replace existing file {file_path} "
                "without an exact expected_sha256 from a complete "
                "read_source response. Use apply_patch for a targeted edit."
            )
    try:
        _publish_text_atomic(
            path,
            content,
            expected_current_sha256=observed_sha256,
            publication_token=publication_token,
            publication_barrier=publication_barrier,
        )
    except (OSError, ValueError) as exc:
        return f"Error writing {file_path}: {exc}"
    n_lines = content.count("\n") + (
        1 if content and not content.endswith("\n") else 0
    )
    return f"OK: wrote {n_lines} lines to {file_path}"


@builder.tool(
    description=(
        "Apply a targeted edit to a source file using before/after text. "
        "Provide the exact lines to find (before) and their replacement "
        "(after). Best for large files (500+ lines) where write_source "
        "would waste tokens. If the before text is not found exactly, "
        "returns an error -- retry with corrected context or fall back "
        "to write_source for small files."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path relative to repo root",
            },
            "before": {
                "type": "string",
                "description": "Exact text to find in the file (must match uniquely)",
            },
            "after": {
                "type": "string",
                "description": "Replacement text",
            },
        },
        "required": ["file_path", "before", "after"],
    },
    write=True,
)
def apply_patch(
    ctx: ToolContext, *, file_path: str, before: str, after: str, **_: Any,
) -> str:
    """Apply a unique find-and-replace edit to ``file_path``."""
    return _apply_patch_impl(
        ctx,
        file_path=file_path,
        before=before,
        after=after,
        publication_token=None,
        publication_barrier=None,
    )


def _apply_patch_impl(
    ctx: ToolContext,
    *,
    file_path: str,
    before: str,
    after: str,
    publication_token: str | None,
    publication_barrier: Callable[[], None] | None = None,
) -> str:
    """Canonical targeted edit with an optional host-owned stage token."""

    if before == after:
        return (
            f"Error: replacement for {file_path} is identical to the "
            "matched text; apply_patch requires an actual edit."
        )
    read_path = resolve_for_read(ctx, file_path)
    if read_path is None or not read_path.is_file():
        return f"Error: file not found: {file_path}"

    try:
        original = read_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Error reading {file_path}: {exc}"

    count = original.count(before)
    if count == 0:
        preview = before[:200].replace("\n", "\\n")
        return (
            f"Error: before text not found in {file_path}. "
            f"Searched for: {preview!r}... "
            f"File has {len(original.splitlines())} lines. "
            "Re-read the file and try again with exact text."
        )
    if count > 1:
        return (
            f"Error: before text matches {count} locations in {file_path}. "
            "Add more surrounding context to make the match unique."
        )

    patched = original.replace(before, after, 1)
    write_path = resolve_for_write(ctx, file_path)
    if write_path is None:
        return f"Error: cannot resolve write path for '{file_path}'"
    expected_current_sha256 = hashlib.sha256(
        original.encode("utf-8")
    ).hexdigest()
    # If the resolver pointed reads/writes at different paths
    # (worktree-aware), re-apply against the write-side current state.
    if write_path != read_path and write_path.is_file():
        original_wt_bytes = write_path.read_bytes()
        original_wt = original_wt_bytes.decode("utf-8")
        write_count = original_wt.count(before)
        if write_count == 0:
            return (
                f"Error: before text not found in write-side {file_path}; "
                "re-read the file and retry."
            )
        if write_count > 1:
            return (
                f"Error: before text matches {write_count} write-side "
                f"locations in {file_path}."
            )
        patched = original_wt.replace(before, after, 1)
        expected_current_sha256 = hashlib.sha256(
            original_wt_bytes
        ).hexdigest()

    try:
        _publish_text_atomic(
            write_path,
            patched,
            expected_current_sha256=expected_current_sha256,
            publication_token=publication_token,
            publication_barrier=publication_barrier,
        )
    except (OSError, ValueError) as exc:
        return f"Error writing patched {file_path}: {exc}"

    before_lines = before.count("\n") + 1
    after_lines = after.count("\n") + 1
    return (
        f"OK: replaced {before_lines} lines with {after_lines} lines "
        f"in {file_path}"
    )


def _same_directory_entry(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    """Return whether two no-follow observations identify the same entry."""

    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and (
            stat.S_ISDIR(left.st_mode)
            or left.st_nlink == right.st_nlink
        )
    )


def _mount_id_for_descriptor(descriptor: int) -> int:
    """Return Linux's mount identity for one held filesystem descriptor.

    ``st_dev`` cannot distinguish same-filesystem bind mounts. Recursive
    deletion therefore relies on the kernel's mount ID and fails closed on a
    platform where that identity cannot be proved.
    """

    try:
        payload = Path(f"/proc/self/fdinfo/{descriptor}").read_text(
            encoding="utf-8",
        )
    except OSError as exc:
        raise ValueError(
            "delete safety cannot read kernel mount identity"
        ) from exc
    mount_ids: list[int] = []
    for line in payload.splitlines():
        key, separator, value = line.partition(":")
        if separator and key == "mnt_id":
            try:
                mount_ids.append(int(value.strip()))
            except ValueError as exc:
                raise ValueError(
                    "delete safety received malformed kernel mount identity"
                ) from exc
    if len(mount_ids) != 1 or mount_ids[0] <= 0:
        raise ValueError("delete safety cannot prove kernel mount identity")
    return mount_ids[0]


def _require_mount_identity(
    descriptor: int,
    expected_mount_id: int,
    *,
    label: str,
) -> None:
    if _mount_id_for_descriptor(descriptor) != expected_mount_id:
        raise ValueError(f"delete target crosses a mount boundary: {label}")


def _exact_declared_delete_target(
    ctx: ToolContext,
    file_path: str,
) -> str | None:
    """Authorize deletion only for one explicitly named write-scope root.

    Ordinary text writes may target descendants of an authorized directory.
    Recursive deletion is intentionally narrower: a model cannot infer
    authority to remove a subtree merely because it may edit files beneath
    that subtree.  Denied descendants also make their ancestor undeletable.
    """

    normalized = _context_relative(ctx, file_path)
    if normalized is None:
        return None
    declared: list[str] = []
    for rendered in getattr(ctx, "write_scope", ()):
        root = _context_relative(ctx, rendered)
        if root is None:
            logger.error(
                "invalid ToolContext write_scope entry: %r",
                rendered,
            )
            return None
        declared.append(root)
    if normalized not in declared:
        return None
    exact_declared: list[str] = []
    for rendered in getattr(ctx, "write_exact_scope", ()):
        root = _context_relative(ctx, rendered)
        if root is None:
            logger.error(
                "invalid ToolContext write_exact_scope entry: %r",
                rendered,
            )
            return None
        exact_declared.append(root)
    if exact_declared and normalized not in exact_declared:
        return None
    for rendered in getattr(ctx, "write_deny_scope", ()):
        denied = _context_relative(ctx, rendered)
        if denied is None:
            logger.error(
                "invalid ToolContext write_deny_scope entry: %r",
                rendered,
            )
            return None
        if denied == normalized or denied.startswith(normalized + "/"):
            return None
    return normalized


def _validated_delete_entry(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
) -> os.stat_result:
    """Re-prove one no-follow directory entry immediately before mutation."""

    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not _same_directory_entry(expected, current):
        raise ValueError(f"delete target changed during traversal: {name}")
    return current


def _open_delete_directory_at(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    expected_mount_id: int,
) -> int:
    """Open and authenticate one directory without following a replacement."""

    descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    try:
        actual = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(actual.st_mode)
            or not _same_directory_entry(expected, actual)
        ):
            raise ValueError(f"delete directory changed while opening: {name}")
        _require_mount_identity(
            descriptor,
            expected_mount_id,
            label=name,
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_delete_parent_components(
    root_fd: int,
    components: tuple[str, ...],
    *,
    expected_mount_id: int,
) -> int:
    """Open an existing parent path while rejecting every mount transition."""

    current_fd = os.dup(root_fd)
    try:
        _require_mount_identity(
            current_fd,
            expected_mount_id,
            label="repository root",
        )
        for component in components:
            observed = os.stat(
                component,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(observed.st_mode):
                raise ValueError(
                    f"unsafe non-directory delete path component: {component}"
                )
            child_fd = _open_delete_directory_at(
                current_fd,
                component,
                observed,
                expected_mount_id=expected_mount_id,
            )
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_delete_regular_at(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    expected_mount_id: int,
) -> int:
    """Open one private regular leaf and reject a file bind mount."""

    opened = _open_regular_at(parent_fd, name, missing_ok=False)
    assert opened is not None
    descriptor, actual = opened
    try:
        if not _same_directory_entry(expected, actual):
            raise ValueError(f"delete file changed while opening: {name}")
        _require_mount_identity(
            descriptor,
            expected_mount_id,
            label=name,
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@dataclass(frozen=True)
class _DeleteTreeEntry:
    """One content-bound entry retained by recursive-delete preflight."""

    name: str
    observed: os.stat_result
    children: tuple[_DeleteTreeEntry, ...] | None
    metadata: tuple[int, ...]
    digest: str | None = None
    symlink_target: str | None = None


def _delete_entry_metadata(observed: os.stat_result) -> tuple[int, ...]:
    """Return metadata that must survive between both tree proofs."""

    return (
        observed.st_dev,
        observed.st_ino,
        stat.S_IFMT(observed.st_mode),
        stat.S_IMODE(observed.st_mode),
        observed.st_uid,
        observed.st_gid,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _delete_root_metadata(observed: os.stat_result) -> tuple[int, ...]:
    """Return root metadata unaffected by renaming into quarantine.

    Linux updates an entry's ctime when it is renamed, so root ctime is
    intentionally excluded. Descendant ctime remains part of the recursive
    manifest because moving the root does not mutate its children.
    """

    return (
        observed.st_dev,
        observed.st_ino,
        stat.S_IFMT(observed.st_mode),
        stat.S_IMODE(observed.st_mode),
        observed.st_uid,
        observed.st_gid,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
    )


def _digest_delete_regular(
    descriptor: int,
    expected: os.stat_result,
    *,
    label: str,
) -> str:
    """Hash one held regular file while proving it stayed stable."""

    before = os.fstat(descriptor)
    if not _same_directory_entry(expected, before):
        raise ValueError(f"delete file changed while hashing: {label}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_DELETE_REGULAR_BYTES:
            raise ValueError(
                "delete file grew beyond the regular-file byte ceiling "
                f"while hashing: {label}"
            )
        digest.update(chunk)
    after = os.fstat(descriptor)
    if _delete_entry_metadata(before) != _delete_entry_metadata(after):
        raise ValueError(f"delete file changed while hashing: {label}")
    return digest.hexdigest()


def _snapshot_delete_regular_at(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    expected_mount_id: int,
) -> str:
    descriptor = _open_delete_regular_at(
        parent_fd,
        name,
        expected,
        expected_mount_id=expected_mount_id,
    )
    try:
        return _digest_delete_regular(descriptor, expected, label=name)
    finally:
        os.close(descriptor)


def _bounded_delete_names(
    directory_fd: int,
    *,
    maximum: int,
    overflow_message: str,
    shared_count: list[int] | None = None,
) -> tuple[str, ...]:
    """Return names without enumerating past a local or shared ceiling."""

    if maximum < 0 or (
        shared_count is not None
        and (len(shared_count) != 1 or shared_count[0] < 0)
    ):
        raise ValueError(overflow_message)
    names: list[str] = []
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            names.append(entry.name)
            observed = len(names)
            if shared_count is not None:
                shared_count[0] += 1
                observed = shared_count[0]
            if observed > maximum:
                raise ValueError(overflow_message)
    return tuple(sorted(names))


def _delete_directory_is_empty(directory_fd: int) -> bool:
    """Prove emptiness while reading at most one directory entry."""

    with os.scandir(directory_fd) as entries:
        return next(iter(entries), None) is None


def _validate_delete_tree(
    directory_fd: int,
    *,
    root_device: int,
    root_mount_id: int,
    depth: int,
    count: list[int],
    regular_bytes: list[int],
) -> tuple[tuple[_DeleteTreeEntry, ...], int, int]:
    """Preflight one recursive deletion without following child symlinks."""

    if depth > _MAX_DELETE_DEPTH:
        raise ValueError(
            f"recursive delete exceeds {_MAX_DELETE_DEPTH} levels"
        )
    files = 0
    directories = 0
    manifest: list[_DeleteTreeEntry] = []
    entry_limit_message = (
        f"recursive delete exceeds {_MAX_DELETE_ENTRIES} entries"
    )
    ordered_names = _bounded_delete_names(
        directory_fd,
        maximum=_MAX_DELETE_ENTRIES,
        overflow_message=entry_limit_message,
        shared_count=count,
    )
    for name in ordered_names:
        if name.casefold() in _VCS_METADATA_COMPONENTS:
            raise ValueError(
                f"recursive delete contains protected VCS metadata: {name}"
            )
        observed = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(observed.st_mode):
            child_fd = _open_delete_directory_at(
                directory_fd,
                name,
                observed,
                expected_mount_id=root_mount_id,
            )
            try:
                actual = os.fstat(child_fd)
                if actual.st_dev != root_device:
                    raise ValueError(
                        f"recursive delete crosses a filesystem boundary: {name}"
                    )
                children, child_files, child_directories = _validate_delete_tree(
                    child_fd,
                    root_device=root_device,
                    root_mount_id=root_mount_id,
                    depth=depth + 1,
                    count=count,
                    regular_bytes=regular_bytes,
                )
            finally:
                os.close(child_fd)
            files += child_files
            directories += child_directories + 1
            manifest.append(_DeleteTreeEntry(
                name,
                observed,
                children,
                _delete_entry_metadata(observed),
            ))
            continue
        if stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            if observed.st_nlink != 1:
                raise ValueError(
                    f"recursive delete contains a multiply-linked entry: {name}"
                )
            digest: str | None = None
            symlink_target: str | None = None
            if stat.S_ISREG(observed.st_mode):
                regular_bytes[0] += observed.st_size
                if regular_bytes[0] > _MAX_DELETE_REGULAR_BYTES:
                    raise ValueError(
                        "recursive delete exceeds "
                        f"{_MAX_DELETE_REGULAR_BYTES} regular-file bytes"
                    )
                digest = _snapshot_delete_regular_at(
                    directory_fd,
                    name,
                    observed,
                    expected_mount_id=root_mount_id,
                )
            else:
                symlink_target = os.readlink(name, dir_fd=directory_fd)
            files += 1
            manifest.append(_DeleteTreeEntry(
                name,
                observed,
                None,
                _delete_entry_metadata(observed),
                digest,
                symlink_target,
            ))
            continue
        raise ValueError(
            f"recursive delete contains unsupported filesystem entry: {name}"
        )
    return tuple(manifest), files, directories


def _revalidate_delete_tree(
    directory_fd: int,
    *,
    manifest: tuple[_DeleteTreeEntry, ...],
    root_device: int,
    root_mount_id: int,
) -> None:
    """Re-prove the complete quarantined tree before the first unlink."""

    expected_names = tuple(entry.name for entry in manifest)
    current_names = _bounded_delete_names(
        directory_fd,
        maximum=len(expected_names),
        overflow_message="delete tree changed after validation",
    )
    if current_names != expected_names:
        raise ValueError("delete tree changed after validation")
    for entry in manifest:
        observed = os.stat(
            entry.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if _delete_entry_metadata(observed) != entry.metadata:
            raise ValueError(
                f"delete target metadata changed after validation: {entry.name}"
            )
        if stat.S_ISDIR(observed.st_mode):
            child_fd = _open_delete_directory_at(
                directory_fd,
                entry.name,
                entry.observed,
                expected_mount_id=root_mount_id,
            )
            try:
                if os.fstat(child_fd).st_dev != root_device:
                    raise ValueError(
                        "recursive delete crosses a filesystem boundary: "
                        f"{entry.name}"
                    )
                _revalidate_delete_tree(
                    child_fd,
                    manifest=entry.children or (),
                    root_device=root_device,
                    root_mount_id=root_mount_id,
                )
            finally:
                os.close(child_fd)
            continue
        if stat.S_ISREG(observed.st_mode):
            digest = _snapshot_delete_regular_at(
                directory_fd,
                entry.name,
                entry.observed,
                expected_mount_id=root_mount_id,
            )
            if digest != entry.digest:
                raise ValueError(
                    f"delete file content changed after validation: {entry.name}"
                )
            continue
        if os.readlink(entry.name, dir_fd=directory_fd) != entry.symlink_target:
            raise ValueError(
                f"delete symlink target changed after validation: {entry.name}"
            )


def _delete_tree_contents(
    directory_fd: int,
    *,
    manifest: tuple[_DeleteTreeEntry, ...],
    root_device: int,
    root_mount_id: int,
) -> tuple[int, int]:
    """Delete validated descendants relative to a held directory descriptor."""

    files = 0
    directories = 0
    expected_names = tuple(entry.name for entry in manifest)
    current_names = _bounded_delete_names(
        directory_fd,
        maximum=len(expected_names),
        overflow_message="delete tree changed after validation",
    )
    if current_names != expected_names:
        raise ValueError("delete tree changed after validation")
    for entry in manifest:
        name = entry.name
        observed = entry.observed
        _validated_delete_entry(directory_fd, name, observed)
        if stat.S_ISDIR(observed.st_mode):
            child_fd = _open_delete_directory_at(
                directory_fd,
                name,
                observed,
                expected_mount_id=root_mount_id,
            )
            try:
                if os.fstat(child_fd).st_dev != root_device:
                    raise ValueError(
                        f"recursive delete crosses a filesystem boundary: {name}"
                    )
                child_files, child_directories = _delete_tree_contents(
                    child_fd,
                    manifest=entry.children or (),
                    root_device=root_device,
                    root_mount_id=root_mount_id,
                )
                os.fsync(child_fd)
            finally:
                os.close(child_fd)
            _validated_delete_entry(directory_fd, name, observed)
            probe_fd = _open_delete_directory_at(
                directory_fd,
                name,
                observed,
                expected_mount_id=root_mount_id,
            )
            os.close(probe_fd)
            os.rmdir(name, dir_fd=directory_fd)
            files += child_files
            directories += child_directories + 1
            continue
        _validated_delete_entry(directory_fd, name, observed)
        if stat.S_ISREG(observed.st_mode):
            descriptor = _open_delete_regular_at(
                directory_fd,
                name,
                observed,
                expected_mount_id=root_mount_id,
            )
            os.close(descriptor)
        os.unlink(name, dir_fd=directory_fd)
        files += 1
    os.fsync(directory_fd)
    return files, directories


def _create_delete_quarantine(
    parent_fd: int,
    name: str,
    *,
    expected_mount_id: int,
    precreated: bool = False,
) -> tuple[int, os.stat_result]:
    """Create or open one empty, exact, private sibling directory."""

    created = False
    descriptor = -1
    try:
        if not precreated:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            created = True
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
        os.fchmod(descriptor, 0o700)
        held = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(held.st_mode)
            or not _same_directory_entry(held, named)
            or stat.S_IMODE(held.st_mode) != 0o700
            or held.st_uid != os.geteuid()
        ):
            raise ValueError("delete quarantine binding is unsafe")
        _require_mount_identity(
            descriptor,
            expected_mount_id,
            label=name,
        )
        if not _delete_directory_is_empty(descriptor):
            raise ValueError("delete quarantine was not created empty")
        os.fsync(descriptor)
        os.fsync(parent_fd)
        return descriptor, held
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            try:
                os.rmdir(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                # Never recurse into a private directory whose binding or
                # contents could not be proved.
                pass
        raise


def _discard_empty_delete_quarantine(
    parent_fd: int,
    quarantine_fd: int,
    quarantine_name: str,
    expected: os.stat_result,
) -> None:
    """Discard only the exact held quarantine after proving it is empty."""

    held = os.fstat(quarantine_fd)
    named = os.stat(
        quarantine_name,
        dir_fd=parent_fd,
        follow_symlinks=False,
    )
    if (
        not _same_directory_entry(expected, held)
        or not _same_directory_entry(expected, named)
        or not _delete_directory_is_empty(quarantine_fd)
    ):
        raise DeleteMutationIndeterminate(
            "delete quarantine changed before empty-directory settlement"
        )
    os.rmdir(quarantine_name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _restore_quarantined_delete(
    parent_fd: int,
    leaf: str,
    quarantine_fd: int,
    quarantine_name: str,
    quarantine_expected: os.stat_result,
    payload_expected: os.stat_result,
) -> None:
    """Restore the retained payload without replacing a public collision."""

    try:
        payload = os.stat(
            _DELETE_QUARANTINE_PAYLOAD,
            dir_fd=quarantine_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise DeleteMutationIndeterminate(
            "delete quarantine payload disappeared before restoration"
        ) from exc
    if not _same_directory_entry(payload_expected, payload):
        raise DeleteMutationIndeterminate(
            "delete quarantine payload changed before restoration"
        )
    try:
        _rename_noreplace_at(
            quarantine_fd,
            _DELETE_QUARANTINE_PAYLOAD,
            parent_fd,
            leaf,
        )
    except FileExistsError as exc:
        raise DeleteMutationIndeterminate(
            "delete target was recreated while its quarantined payload "
            "remained retained; both entries were preserved"
        ) from exc
    restored = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    if not _same_directory_entry(payload_expected, restored):
        raise DeleteMutationIndeterminate(
            "restored delete target binding could not be proved"
        )
    os.fsync(quarantine_fd)
    os.fsync(parent_fd)
    _discard_empty_delete_quarantine(
        parent_fd,
        quarantine_fd,
        quarantine_name,
        quarantine_expected,
    )


def _require_public_delete_name_absent(parent_fd: int, leaf: str) -> None:
    try:
        os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise DeleteMutationIndeterminate(
        "delete target was recreated after quarantine; public and private "
        "entries were preserved"
    )


@builder.tool(
    description=(
        "Delete one source path that is itself an exact declared directive "
        "scope entry. This cannot delete an inferred descendant or ancestor. "
        "Set recursive=true to delete a declared directory tree; directory "
        "deletion is rejected otherwise. Protected VCS metadata, final-root "
        "symlinks, special files, filesystem crossings, and denied descendants "
        "are rejected."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "Exact repository-relative path present in the directive "
                    "write scope"
                ),
            },
            "recursive": {
                "type": "boolean",
                "description": (
                    "Required explicit acknowledgement for directory deletion; "
                    "use false for an ordinary file"
                ),
            },
        },
        "required": ["file_path", "recursive"],
        "additionalProperties": False,
    },
    write=True,
)
def delete_source(
    ctx: ToolContext,
    *,
    file_path: str,
    recursive: bool,
    **_: Any,
) -> str:
    """Delete one exact directive root, recursively only when acknowledged."""

    return _delete_source_impl(
        ctx,
        file_path=file_path,
        recursive=recursive,
        publication_token=None,
        publication_barrier=None,
    )


def _delete_source_impl(
    ctx: ToolContext,
    *,
    file_path: str,
    recursive: bool,
    publication_token: str | None = None,
    publication_barrier: Callable[[], None] | None = None,
    quarantine_barrier: Callable[[], None] | None = None,
    quarantine_precreated: bool = False,
) -> str:
    """Quarantine and delete one exact descriptor-relative source root.

    The public source name is atomically moved into a tokenized 0700 sibling
    before any unlink. The complete retained manifest is then proved again
    inside that held quarantine. This closes public-name replacement races;
    the workflow's writer epoch remains responsible for excluding injected
    child code, root/CAP_DAC_OVERRIDE, and peers already holding a quarantine
    descriptor after the final proof.
    """

    if not isinstance(recursive, bool):
        return "Error: recursive must be a boolean"
    normalized = _exact_declared_delete_target(ctx, file_path)
    if normalized is None:
        return (
            "Error: delete target must exactly match an explicit, permitted "
            f"write_scope entry: '{file_path}'"
        )
    path = resolve_for_write(ctx, file_path)
    if path is None:
        return f"Error: cannot resolve delete path for '{file_path}'"
    root = Path(_active_repository_root(ctx)).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return f"Error: delete path escaped repository root: '{file_path}'"
    if relative.as_posix() != normalized or not relative.parts:
        return f"Error: unsafe or broad delete target: '{file_path}'"

    try:
        token = _validate_publication_token(publication_token)
    except ValueError as exc:
        return f"Error deleting {file_path}: {exc}"
    quarantine_name = _delete_quarantine_path(path, token).name
    root_fd = -1
    parent_fd = -1
    target_fd = -1
    quarantine_fd = -1
    quarantine_expected: os.stat_result | None = None
    observed: os.stat_result | None = None
    payload_expected: os.stat_result | None = None
    quarantine_created = False
    payload_moved = False
    destructive_started = False
    try:
        root_fd = os.open(root, _directory_open_flags())
        root_mount_id = _mount_id_for_descriptor(root_fd)
        parent_fd = _open_delete_parent_components(
            root_fd,
            tuple(relative.parts[:-1]),
            expected_mount_id=root_mount_id,
        )
        leaf = relative.parts[-1]
        observed = os.stat(
            leaf,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError("final delete target cannot be a symlink")
        root_metadata = _delete_root_metadata(observed)
        manifest: tuple[_DeleteTreeEntry, ...] = ()
        files = 0
        directories = 0
        root_digest: str | None = None
        if stat.S_ISDIR(observed.st_mode):
            if not recursive:
                return (
                    f"Error: {file_path} is a directory; retry with "
                    "recursive=true only if the whole declared tree must be "
                    "removed"
                )
            if observed.st_dev != os.fstat(parent_fd).st_dev:
                raise ValueError(
                    "recursive delete root crosses a filesystem boundary"
                )
            target_fd = _open_delete_directory_at(
                parent_fd,
                leaf,
                observed,
                expected_mount_id=root_mount_id,
            )
            root_device = os.fstat(target_fd).st_dev
            manifest, files, directories = _validate_delete_tree(
                target_fd,
                root_device=root_device,
                root_mount_id=root_mount_id,
                depth=0,
                count=[0],
                regular_bytes=[0],
            )
        elif stat.S_ISREG(observed.st_mode):
            if observed.st_nlink != 1:
                raise ValueError("delete target is multiply linked")
            if observed.st_size > _MAX_DELETE_REGULAR_BYTES:
                raise ValueError(
                    "delete target exceeds "
                    f"{_MAX_DELETE_REGULAR_BYTES} regular-file bytes"
                )
            target_fd = _open_delete_regular_at(
                parent_fd,
                leaf,
                observed,
                expected_mount_id=root_mount_id,
            )
            root_device = os.fstat(target_fd).st_dev
            root_digest = _digest_delete_regular(
                target_fd,
                observed,
                label=leaf,
            )
        else:
            raise ValueError("delete target must be a regular file or directory")

        if publication_barrier is not None:
            publication_barrier()

        quarantine_fd, quarantine_expected = _create_delete_quarantine(
            parent_fd,
            quarantine_name,
            expected_mount_id=root_mount_id,
            precreated=quarantine_precreated,
        )
        assert quarantine_expected is not None
        quarantine_created = True
        try:
            _rename_noreplace_at(
                parent_fd,
                leaf,
                quarantine_fd,
                _DELETE_QUARANTINE_PAYLOAD,
            )
            payload_moved = True
            os.fsync(quarantine_fd)
            os.fsync(parent_fd)

            moved = os.stat(
                _DELETE_QUARANTINE_PAYLOAD,
                dir_fd=quarantine_fd,
                follow_symlinks=False,
            )
            # Whatever the atomic move claimed must be restored on a failed
            # proof. Authorization is decided against ``observed`` below;
            # recovery is bound to the actual moved inode so a racing
            # replacement is never stranded or deleted.
            payload_expected = moved
            if quarantine_barrier is not None:
                quarantine_barrier()
            if (
                not _same_directory_entry(observed, moved)
                or not _same_directory_entry(os.fstat(target_fd), moved)
                or _delete_root_metadata(moved) != root_metadata
            ):
                raise ValueError(
                    "delete target changed before atomic quarantine"
                )
            _require_public_delete_name_absent(parent_fd, leaf)

            if stat.S_ISDIR(moved.st_mode):
                _revalidate_delete_tree(
                    target_fd,
                    manifest=manifest,
                    root_device=root_device,
                    root_mount_id=root_mount_id,
                )
            else:
                moved_digest = _digest_delete_regular(
                    target_fd,
                    moved,
                    label=leaf,
                )
                if moved_digest != root_digest:
                    raise ValueError(
                        "delete file content changed before quarantine proof"
                    )
            _require_public_delete_name_absent(parent_fd, leaf)

            destructive_started = True
            if stat.S_ISDIR(moved.st_mode):
                deleted_files, deleted_directories = _delete_tree_contents(
                    target_fd,
                    manifest=manifest,
                    root_device=root_device,
                    root_mount_id=root_mount_id,
                )
                if (deleted_files, deleted_directories) != (
                    files,
                    directories,
                ):
                    raise ValueError("delete tree changed after validation")
                os.rmdir(
                    _DELETE_QUARANTINE_PAYLOAD,
                    dir_fd=quarantine_fd,
                )
                result = (
                    f"OK: recursively deleted {deleted_files} files and "
                    f"{deleted_directories + 1} directories from {file_path}"
                )
            else:
                os.unlink(
                    _DELETE_QUARANTINE_PAYLOAD,
                    dir_fd=quarantine_fd,
                )
                result = f"OK: deleted file {file_path}"
            payload_moved = False
            os.fsync(quarantine_fd)
            _require_public_delete_name_absent(parent_fd, leaf)
            _discard_empty_delete_quarantine(
                parent_fd,
                quarantine_fd,
                quarantine_name,
                quarantine_expected,
            )
            quarantine_created = False
            return result
        except BaseException as exc:
            if payload_moved:
                if payload_expected is None or quarantine_expected is None:
                    raise DeleteMutationIndeterminate(
                        "delete quarantine payload identity could not be "
                        "bound for restoration"
                    ) from exc
                try:
                    _restore_quarantined_delete(
                        parent_fd,
                        leaf,
                        quarantine_fd,
                        quarantine_name,
                        quarantine_expected,
                        payload_expected,
                    )
                except DeleteMutationIndeterminate:
                    raise
                except BaseException as restore_exc:
                    raise DeleteMutationIndeterminate(
                        "quarantined delete could not be restored exactly"
                    ) from restore_exc
                payload_moved = False
                quarantine_created = False
                if destructive_started or not isinstance(exc, ValueError):
                    raise DeleteMutationIndeterminate(
                        "delete failed after namespace mutation; host "
                        "settlement and journal rollback are required"
                    ) from exc
                return f"Error deleting {file_path}: {exc}"
            if quarantine_created:
                if quarantine_expected is None:
                    raise DeleteMutationIndeterminate(
                        "delete quarantine identity could not be bound for "
                        "empty-directory settlement"
                    ) from exc
                try:
                    _discard_empty_delete_quarantine(
                        parent_fd,
                        quarantine_fd,
                        quarantine_name,
                        quarantine_expected,
                    )
                except BaseException as cleanup_exc:
                    raise DeleteMutationIndeterminate(
                        "empty delete quarantine could not be settled"
                    ) from cleanup_exc
                quarantine_created = False
            if destructive_started:
                raise DeleteMutationIndeterminate(
                    "delete failed after payload removal; host settlement "
                    "and journal rollback are required"
                ) from exc
            raise
    except DeleteMutationIndeterminate:
        raise
    except FileNotFoundError:
        return f"Error: delete target not found: {file_path}"
    except (OSError, ValueError) as exc:
        return f"Error deleting {file_path}: {exc}"
    finally:
        for descriptor in (
            quarantine_fd,
            target_fd,
            parent_fd,
            root_fd,
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _active_repository_root(ctx: ToolContext) -> str:
    """Return the root whose files the current invocation is mutating."""

    resolver = getattr(ctx, "path_resolver", None)
    if resolver is not None:
        root = resolver.resolve_worktree_root(".")
        if root is not None:
            return os.fspath(root)
    return os.fspath(ctx.repo_root.resolve())


def _copy_regular_file(source_fd: int, destination_fd: int) -> int:
    """Copy one descriptor exactly, enforcing the artifact size ceiling."""

    copied = 0
    while True:
        chunk = os.read(source_fd, _COPY_CHUNK_BYTES)
        if not chunk:
            return copied
        copied += len(chunk)
        if copied > _MAX_ARTIFACT_BYTES:
            raise ValueError(
                f"artifact exceeds {_MAX_ARTIFACT_BYTES} byte limit"
            )
        view = memoryview(chunk)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:
                raise OSError("short write while materializing artifact")
            view = view[written:]


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_CLOEXEC
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
    )


def _regular_read_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
    )


def _open_directory_components(
    root_fd: int,
    components: tuple[str, ...],
    *,
    create: bool,
) -> int:
    """Open a relative directory path without following any component."""

    current_fd = os.dup(root_fd)
    try:
        for component in components:
            if create:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
                except FileExistsError:
                    pass
            observed = os.stat(
                component,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(observed.st_mode):
                raise ValueError(
                    f"unsafe non-directory path component: {component}"
                )
            child_fd = os.open(
                component,
                _directory_open_flags(),
                dir_fd=current_fd,
            )
            actual = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(actual.st_mode)
                or actual.st_dev != observed.st_dev
                or actual.st_ino != observed.st_ino
            ):
                os.close(child_fd)
                raise ValueError(
                    f"directory path component changed while opening: {component}"
                )
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_regular_at(
    parent_fd: int,
    leaf: str,
    *,
    missing_ok: bool,
) -> tuple[int, os.stat_result] | None:
    """Open a private regular leaf anchored to *parent_fd*."""

    try:
        observed = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise ValueError(f"unsafe non-private regular file: {leaf}")
    descriptor = os.open(leaf, _regular_read_flags(), dir_fd=parent_fd)
    try:
        actual = os.fstat(descriptor)
        if (
            not stat.S_ISREG(actual.st_mode)
            or actual.st_nlink != 1
            or actual.st_dev != observed.st_dev
            or actual.st_ino != observed.st_ino
        ):
            raise ValueError(f"regular file changed while opening: {leaf}")
        return descriptor, actual
    except BaseException:
        os.close(descriptor)
        raise


def _create_temporary_at(parent_fd: int) -> tuple[int, str]:
    """Create one private temporary file anchored to *parent_fd*."""

    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
    )
    for _ in range(32):
        name = f".crucible-artifact-{secrets.token_hex(16)}"
        try:
            return os.open(name, flags, 0o600, dir_fd=parent_fd), name
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a unique artifact temporary file")


def _descriptor_bytes_equal(
    left_fd: int,
    left_size: int,
    right_fd: int,
    right_size: int,
) -> bool:
    """Compare descriptor bytes without loading an artifact into memory."""

    if left_size != right_size:
        return False
    os.lseek(left_fd, 0, os.SEEK_SET)
    os.lseek(right_fd, 0, os.SEEK_SET)
    while True:
        left_chunk = os.read(left_fd, _COPY_CHUNK_BYTES)
        right_chunk = os.read(right_fd, _COPY_CHUNK_BYTES)
        if left_chunk != right_chunk:
            return False
        if not left_chunk:
            return True


def _link_temporary_without_replace(
    parent_fd: int,
    temporary_name: str,
    destination_name: str,
) -> bool:
    """Publish with no-clobber semantics; return whether temp was removed.

    If removing the temporary hardlink fails, roll back the destination so an
    error never falsely reports that no publication occurred. If rollback is
    itself impossible, retain the valid destination and let the caller report
    success while making a best-effort final temporary cleanup.
    """

    os.link(
        temporary_name,
        destination_name,
        src_dir_fd=parent_fd,
        dst_dir_fd=parent_fd,
        follow_symlinks=False,
    )
    try:
        os.unlink(temporary_name, dir_fd=parent_fd)
    except FileNotFoundError:
        return True
    except OSError as cleanup_error:
        try:
            os.unlink(destination_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        except OSError as rollback_error:
            logger.error(
                "artifact temp cleanup and destination rollback both failed: "
                "%s; %s",
                cleanup_error,
                rollback_error,
            )
            return False
        raise OSError(
            "temporary cleanup failed; destination publication was rolled back"
        ) from cleanup_error
    return True


@builder.tool(
    description=(
        "Byte-copy a built binary artifact from the active repository or "
        "worktree into a destination authorized by the current write scope. "
        "The source may be in ignored build staging. Publication is atomic, "
        "byte-and-permission-identical existing output is idempotent, and "
        "differing output requires replace=true. Inputs are limited to 256 MiB."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "source_path": {
                "type": "string",
                "description": "Repository-relative path to a built regular file",
            },
            "destination_path": {
                "type": "string",
                "description": "Repository-relative authorized output path",
            },
            "replace": {
                "type": "boolean",
                "description": "Replace differing existing output when true",
                "default": False,
            },
        },
        "required": ["source_path", "destination_path"],
    },
    write=True,
)
def materialize_artifact(
    ctx: ToolContext,
    *,
    source_path: str,
    destination_path: str,
    replace: bool = False,
    **_: Any,
) -> str:
    """Atomically copy a bounded binary artifact into the write scope."""

    if not isinstance(replace, bool):
        return "Error: replace must be an explicit boolean"

    if resolve_for_write(ctx, destination_path) is None:
        return f"Error: cannot resolve write path for '{destination_path}'"

    normalized_source = _context_relative(ctx, source_path)
    normalized_destination = _context_relative(ctx, destination_path)
    if normalized_source is None:
        return f"Error: cannot resolve artifact source '{source_path}'"
    if normalized_destination is None:
        return f"Error: cannot resolve write path for '{destination_path}'"
    if normalized_source == normalized_destination:
        return "Error: artifact source and destination must be different paths"

    root_fd = -1
    source_parent_fd = -1
    destination_parent_fd = -1
    source_fd = -1
    temporary_fd = -1
    destination_fd = -1
    temporary_name = ""
    try:
        root_fd = os.open(
            _active_repository_root(ctx),
            _directory_open_flags(),
        )
        source_parts = tuple(normalized_source.split("/"))
        destination_parts = tuple(normalized_destination.split("/"))
        source_parent_fd = _open_directory_components(
            root_fd,
            source_parts[:-1],
            create=False,
        )
        source_entry = _open_regular_at(
            source_parent_fd,
            source_parts[-1],
            missing_ok=False,
        )
        assert source_entry is not None
        source_fd, source_metadata = source_entry
        if source_metadata.st_size > _MAX_ARTIFACT_BYTES:
            return (
                f"Error: artifact source exceeds {_MAX_ARTIFACT_BYTES} byte "
                f"limit: {source_path}"
            )

        destination_parent_fd = _open_directory_components(
            root_fd,
            destination_parts[:-1],
            create=True,
        )
        destination_entry = _open_regular_at(
            destination_parent_fd,
            destination_parts[-1],
            missing_ok=True,
        )
        destination_metadata: os.stat_result | None = None
        if destination_entry is not None:
            destination_fd, destination_metadata = destination_entry

        temporary_fd, temporary_name = _create_temporary_at(
            destination_parent_fd
        )
        publication_mode = stat.S_IMODE(source_metadata.st_mode) & 0o777
        os.fchmod(temporary_fd, publication_mode)
        copied = _copy_regular_file(source_fd, temporary_fd)
        os.fsync(temporary_fd)

        final_source_metadata = os.fstat(source_fd)
        if (
            final_source_metadata.st_dev != source_metadata.st_dev
            or final_source_metadata.st_ino != source_metadata.st_ino
            or final_source_metadata.st_nlink != 1
            or final_source_metadata.st_mode != source_metadata.st_mode
            or final_source_metadata.st_size != source_metadata.st_size
            or final_source_metadata.st_mtime_ns != source_metadata.st_mtime_ns
            or final_source_metadata.st_ctime_ns != source_metadata.st_ctime_ns
            or copied != final_source_metadata.st_size
        ):
            return f"Error: artifact source changed while reading: {source_path}"

        if destination_metadata is not None:
            mode_matches = (
                stat.S_IMODE(destination_metadata.st_mode) == publication_mode
            )
            bytes_match = _descriptor_bytes_equal(
                temporary_fd,
                copied,
                destination_fd,
                destination_metadata.st_size,
            )
            if mode_matches and bytes_match:
                return (
                    f"OK: artifact already materialized at {destination_path} "
                    f"({copied} bytes)"
                )
            if not replace:
                return (
                    f"Error: destination differs in bytes or mode: "
                    f"{destination_path}; "
                    "retry with replace=true to replace it"
                )
            os.replace(
                temporary_name,
                destination_parts[-1],
                src_dir_fd=destination_parent_fd,
                dst_dir_fd=destination_parent_fd,
            )
            temporary_name = ""
        elif replace:
            os.replace(
                temporary_name,
                destination_parts[-1],
                src_dir_fd=destination_parent_fd,
                dst_dir_fd=destination_parent_fd,
            )
            temporary_name = ""
        else:
            try:
                temporary_removed = _link_temporary_without_replace(
                    destination_parent_fd,
                    temporary_name,
                    destination_parts[-1],
                )
            except FileExistsError:
                return (
                    f"Error: destination appeared during publication: "
                    f"{destination_path}"
                )
            if temporary_removed:
                temporary_name = ""
    except (OSError, ValueError) as exc:
        return f"Error materializing {source_path} to {destination_path}: {exc}"
    finally:
        if temporary_name and destination_parent_fd >= 0:
            try:
                os.unlink(temporary_name, dir_fd=destination_parent_fd)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "could not clean artifact temporary %s: %s",
                    temporary_name,
                    exc,
                )
        for descriptor in (
            destination_fd,
            temporary_fd,
            source_fd,
            destination_parent_fd,
            source_parent_fd,
            root_fd,
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    return f"OK: materialized {copied} bytes to {destination_path}"


def build_write_tools(context: ToolContext):
    """Build the write ToolSet bound to *context*."""
    return builder.build(context=context)


__all__ = [
    "build_write_tools",
    "builder",
    "write_source",
    "apply_patch",
    "delete_source",
    "materialize_artifact",
]
