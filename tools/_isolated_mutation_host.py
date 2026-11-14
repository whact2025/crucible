# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Spawn-safe, invocation-scoped host for backend source mutations.

Subprocess-backed adapters reach Crucible tools through bridges whose sync
handlers normally run in parent-process threads. Python cannot terminate a
running handler thread after the logical invocation has timed out. This module
moves the narrowly qualified source-mutation surface into a fresh external
interpreter whose process group can be terminated and reaped before an
invocation returns. Adapters retain their own transport and trace logic; they
only wrap a frozen ToolSet with :class:`ProcessHostedToolSet`.

No callable or object graph crosses the boundary.  The child receives a
versioned JSON specification containing only paths, write scopes, and one of
four canonical handler kinds.  Unsupported or custom mutating ToolSets fail
preflight before provider credentials are read or inference begins.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

if __package__:
    from ._dispatch_budget import ToolMutationIndeterminate
else:
    class ToolMutationIndeterminate(RuntimeError):
        """Child-local marker mirrored into a typed terminal frame."""

_SPEC_SCHEMA = "crucible.isolated-toolset.v3"
_MAX_FRAME_BYTES = 32 * 1024 * 1024
_START_GRACE_SECONDS = 10.0
_TERM_GRACE_SECONDS = 0.5
_KILL_GRACE_SECONDS = 2.0
_MAX_SETTLEMENT_INTERRUPTS = 2
_PUBLICATION_TOKEN_HEX_LENGTH = 32
_MAX_PROC_IDENTITY_BYTES = 4096
_CHILD_CWD_ENV = "CRUCIBLE_ISOLATED_MUTATION_PRIVATE_ROOT"
_CHILD_ROOT_DESCRIPTOR_ENV = "CRUCIBLE_ISOLATED_MUTATION_ROOT_FD"
_CHILD_SOURCE_DESCRIPTOR_ENV = "CRUCIBLE_ISOLATED_MUTATION_SOURCE_FD"
_CHILD_SOURCE_SHA256_ENV = "CRUCIBLE_ISOLATED_MUTATION_SOURCE_SHA256"
_CHILD_MUTATION_ROOTS_ENV = "CRUCIBLE_ISOLATED_MUTATION_BOUND_ROOTS"
_SOURCE_HASH_CHUNK_BYTES = 1024 * 1024
_PIPE_CHUNK_BYTES = 64 * 1024
_MAX_DELETE_ENTRIES = 100_000
_MAX_DELETE_DEPTH = 256
_MAX_DELETE_REGULAR_BYTES = 512 * 1024 * 1024
_DELETE_QUARANTINE_PAYLOAD = "payload"
_CHILD_SOURCE_DESCRIPTOR = 3
_CHILD_ROOT_DESCRIPTOR = 4
_SPAWN_PROJECTION_FD_MINIMUM = 64
_POSIX_SPAWN_SETPGROUP = 0x02
_POSIX_SPAWN_SETSIGDEF = 0x04
_POSIX_SPAWN_SETSIGMASK = 0x08
_SPAWN_STORAGE_BYTES = 1024
_PRIVATE_ROOT_PREFIX = "crucible-isolated-mutation-"
_VCS_METADATA_COMPONENTS = frozenset({".git", ".hg", ".svn", ".bzr", "_darcs"})
_CHILD_BOOTSTRAP = """
import fcntl
import hashlib
import os
import stat

source_raw = os.environ.get("CRUCIBLE_ISOLATED_MUTATION_SOURCE_FD", "")
expected = os.environ.pop("CRUCIBLE_ISOLATED_MUTATION_SOURCE_SHA256", "")
root_raw = os.environ["CRUCIBLE_ISOLATED_MUTATION_ROOT_FD"]
if not source_raw.isascii() or not source_raw.isdigit():
    raise SystemExit(70)
source = int(source_raw)
metadata = os.fstat(source)
seals = int(fcntl.fcntl(source, getattr(fcntl, "F_GET_SEALS", 1034)))
if (
    source <= 2
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_nlink not in {0, 1}
    or metadata.st_size <= 0
    or seals & 15 != 15
    or len(expected) != 64
):
    raise SystemExit(70)
digest = hashlib.sha256()
offset = 0
while offset < metadata.st_size:
    chunk = os.pread(source, min(1024 * 1024, metadata.st_size - offset), offset)
    if not chunk:
        raise SystemExit(70)
    digest.update(chunk)
    offset += len(chunk)
if os.pread(source, 1, metadata.st_size) or digest.hexdigest() != expected:
    raise SystemExit(70)
os.set_inheritable(source, False)
root = int(root_raw)
os.fchdir(root)
os.close(root)
import runpy
runpy.run_path("/proc/self/fd/" + source_raw, run_name="__main__")
"""
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_SOURCE_IMAGE_REQUIRED_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
_SUPPORTED_KINDS = frozenset(
    {
        "write_source",
        "apply_patch",
        "delete_source",
        "exact_write_source",
    }
)
_PR_SET_PDEATHSIG = 1

MutationKind = Literal[
    "write_source",
    "apply_patch",
    "delete_source",
    "exact_write_source",
]


class MutationHostQualificationError(RuntimeError):
    """The attached mutating ToolSet cannot be safely reconstructed."""


class MutationHostLifecycleError(RuntimeError):
    """The isolated mutation process group could not be confirmed settled."""


class _ProjectImportBlocker:
    """Reject project imports without retaining an import callable."""

    __slots__ = ()

    def find_spec(
        self,
        fullname: str,
        _path: object = None,
        _target: object = None,
    ) -> None:
        if fullname == "crucible" or fullname.startswith("crucible."):
            raise MutationHostQualificationError(
                "project imports are sealed after mutation-child readiness"
            )
        return None


class MutationHostProtocolError(RuntimeError):
    """The isolated child violated its bounded JSON protocol."""


class MutationHostTimeoutError(TimeoutError):
    """The logical invocation deadline expired during mutation dispatch."""


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return exact metadata used to reject source replacement or mutation."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sealed_source_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return immutable identity fields; failed writes may update ctime."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
    )


def _descriptor_sha256(
    descriptor: int,
    *,
    expected_identity: tuple[int, ...],
) -> str:
    """Hash one held source descriptor without changing its file offset."""

    before = os.fstat(descriptor)
    if (
        _stat_identity(before) != expected_identity
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink not in {0, 1}
        or not before.st_size
    ):
        raise MutationHostQualificationError(
            "isolated mutation child source identity changed"
        )
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(
            descriptor,
            min(_SOURCE_HASH_CHUNK_BYTES, before.st_size - offset),
            offset,
        )
        if not chunk:
            raise MutationHostQualificationError(
                "isolated mutation child source ended before its exact size"
            )
        digest.update(chunk)
        offset += len(chunk)
    if (
        os.pread(descriptor, 1, before.st_size)
        or _stat_identity(os.fstat(descriptor)) != expected_identity
    ):
        raise MutationHostQualificationError(
            "isolated mutation child source changed while hashing"
        )
    return digest.hexdigest()


def _spawn_projection_targets(
    excluded: frozenset[int],
    *,
    count: int,
) -> tuple[int, ...]:
    """Choose collision-free child staging descriptors for atomic spawn."""

    try:
        open_max = os.sysconf("SC_OPEN_MAX")
    except (OSError, ValueError) as exc:
        raise MutationHostQualificationError(
            "isolated mutation child descriptor ceiling is unavailable"
        ) from exc
    if (
        not isinstance(open_max, int)
        or open_max <= _SPAWN_PROJECTION_FD_MINIMUM + count
    ):
        raise MutationHostQualificationError(
            "isolated mutation child descriptor ceiling is unsafe"
        )
    admitted = [
        descriptor
        for descriptor in range(
            _SPAWN_PROJECTION_FD_MINIMUM,
            min(open_max, 4096),
        )
        if descriptor not in excluded
    ]
    if len(admitted) < count:
        raise MutationHostQualificationError(
            "isolated mutation child projection descriptors are unavailable"
        )
    return tuple(admitted[:count])


def _stable_directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return rename-stable identity for one privately owned directory."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
    )


def _descriptor_mount_id(descriptor: int) -> int:
    """Read Linux's mount identity without acquiring another descriptor."""

    return _statx_mount_id(Path(f"/proc/self/fd/{descriptor}"), nofollow=False)


def _rename_noreplace(
    parent_descriptor: int,
    source: str,
    destination: str,
) -> None:
    """Atomically move one name without replacing an existing destination."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise MutationHostQualificationError(
            "atomic no-replace rename is unavailable"
        ) from exc
    renameat2.restype = ctypes.c_int
    result = int(
        renameat2(
            parent_descriptor,
            os.fsencode(source),
            parent_descriptor,
            os.fsencode(destination),
            0x1,  # RENAME_NOREPLACE
        )
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"renameat2: {os.strerror(error)}")


def _mkdirat_open_directory(
    parent_descriptor: int,
    name: str,
    mode: int,
) -> int:
    """Create and open an unpredictable private directory without Python hooks.

    ``mkdir(2)`` has no descriptor-returning form.  Keeping both syscalls in
    this sealed helper, beneath an unpredictable name, prevents a Python-level
    mkdir wrapper from substituting another inode before authority is bound.
    The caller treats any exceptional return as an uncertain acquisition and
    therefore cannot claim cleanup or publication success.
    """

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        mkdirat = libc.mkdirat
        openat = libc.openat
    except (AttributeError, OSError) as exc:
        raise MutationHostQualificationError(
            "descriptor-bound directory creation is unavailable"
        ) from exc
    mkdirat.restype = ctypes.c_int
    result = int(mkdirat(parent_descriptor, os.fsencode(name), mode))
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"mkdirat: {os.strerror(error)}")
    openat.restype = ctypes.c_int
    descriptor = int(
        openat(
            parent_descriptor,
            os.fsencode(name),
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            0,
        )
    )
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, f"openat: {os.strerror(error)}")
    return descriptor


def _unlinkat_direct(parent_descriptor: int, name: str, *, directory: bool) -> None:
    """Remove a quarantined entry without dispatching through ``os`` hooks."""

    try:
        unlinkat = ctypes.CDLL(None, use_errno=True).unlinkat
    except (AttributeError, OSError) as exc:
        raise MutationHostQualificationError(
            "descriptor-relative quarantine removal is unavailable"
        ) from exc
    unlinkat.restype = ctypes.c_int
    result = int(
        unlinkat(
            parent_descriptor,
            os.fsencode(name),
            0x200 if directory else 0,  # AT_REMOVEDIR
        )
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"unlinkat: {os.strerror(error)}")


def _entry_ownership_key(metadata: os.stat_result) -> tuple[int, ...]:
    """Return fields that identify an inode across rename and link changes."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
    )


def _named_entry_metadata(parent_descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise MutationHostLifecycleError(
            "descriptor-relative entry lookup is unavailable"
        ) from exc


@dataclass(slots=True)
class _QuarantineRemovalState:
    """Retry authority for an owned inode moved under a private name."""

    binding: tuple[str, str] | None = None

    @property
    def active(self) -> bool:
        return self.binding is not None

    @property
    def source_name(self) -> str | None:
        return None if self.binding is None else self.binding[0]

    @property
    def quarantine_name(self) -> str | None:
        return None if self.binding is None else self.binding[1]


def _quarantine_remove_bound_entry(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    *,
    directory: bool,
    token: str,
    removal_state: _QuarantineRemovalState | None = None,
) -> None:
    """Move an owned entry to an unpredictable name before deleting it.

    The visible name is never passed to a destructive syscall.  After every
    authorized writer is positively quiesced, the exact moved inode is checked
    through both its held descriptor and an unpredictable quarantine name
    before the direct ``unlinkat`` call.  POSIX has no inode-conditional
    unlink; deliberate same-UID replacement inside that final syscall window
    remains outside the documented transaction threat boundary in
    ``docs/workflows/tier.md``.
    """

    expected = _entry_ownership_key(os.fstat(descriptor))
    state = removal_state if removal_state is not None else _QuarantineRemovalState()
    if not state.active:
        state.binding = (
            name,
            f".crucible-remove-{token}-{secrets.token_hex(16)}",
        )
    source = state.source_name
    quarantine = state.quarantine_name
    if source is None or quarantine is None:
        raise MutationHostLifecycleError(
            "entry quarantine retry authority is incomplete"
        )

    source_before = _named_entry_metadata(parent_descriptor, source)
    quarantine_before = _named_entry_metadata(parent_descriptor, quarantine)
    if quarantine_before is None:
        if source_before is None:
            if os.fstat(descriptor).st_nlink == 0:
                state.binding = None
                return
            raise MutationHostLifecycleError(
                "entry quarantine lost its exact owned name"
            )
        if _entry_ownership_key(source_before) != expected:
            raise MutationHostLifecycleError(
                "entry quarantine refused a rebound source name"
            )
        rename_error: BaseException | None = None
        try:
            _rename_noreplace(parent_descriptor, source, quarantine)
        except BaseException as exc:
            rename_error = exc
        source_before = _named_entry_metadata(parent_descriptor, source)
        quarantine_before = _named_entry_metadata(parent_descriptor, quarantine)
        if source_before is not None and quarantine_before is None:
            if _entry_ownership_key(source_before) != expected:
                raise MutationHostLifecycleError(
                    "entry quarantine source was rebound during rename"
                ) from rename_error
            raise MutationHostLifecycleError(
                "entry quarantine rename had no effect; exact retry is required"
            ) from rename_error
    if (
        source_before is not None
        or quarantine_before is None
        or _entry_ownership_key(quarantine_before) != expected
        or _entry_ownership_key(os.fstat(descriptor)) != expected
    ):
        raise MutationHostLifecycleError("entry quarantine rename outcome is uncertain")
    unlink_error: BaseException | None = None
    try:
        _unlinkat_direct(parent_descriptor, quarantine, directory=directory)
    except BaseException as exc:
        unlink_error = exc
    quarantine_after = _named_entry_metadata(parent_descriptor, quarantine)
    if quarantine_after is None:
        state.binding = None
        return
    if (
        _entry_ownership_key(quarantine_after) != expected
        or _entry_ownership_key(os.fstat(descriptor)) != expected
        or _named_entry_metadata(parent_descriptor, name) is not None
    ):
        raise MutationHostLifecycleError(
            "quarantined entry removal outcome is uncertain"
        ) from unlink_error
    # A no-effect removal failure must not strand the exact owned inode under
    # a random name that its owner cannot rediscover on retry.  Restore it to
    # the retained source name and settle effect-then-error by observation.
    restore_error: BaseException | None = None
    try:
        _rename_noreplace(parent_descriptor, quarantine, name)
    except BaseException as exc:
        restore_error = exc
    source_after = _named_entry_metadata(parent_descriptor, name)
    quarantine_after = _named_entry_metadata(parent_descriptor, quarantine)
    if (
        source_after is not None
        and _entry_ownership_key(source_after) == expected
        and quarantine_after is None
        and _entry_ownership_key(os.fstat(descriptor)) == expected
    ):
        state.binding = None
        raise MutationHostLifecycleError(
            "quarantine removal failed; source was restored for an exact retry"
        ) from (unlink_error if unlink_error is not None else restore_error)
    raise MutationHostLifecycleError(
        "quarantine removal recovery orientation is uncertain"
    ) from (restore_error if restore_error is not None else unlink_error)


@dataclass(slots=True)
class _BoundDirectoryCreation:
    """Retained state for private-name creation and no-replace publication."""

    parent_descriptor: int
    parent_identity: tuple[int, ...]
    expected_mount_id: int
    final_name: str
    private_name: str
    mode: int
    token: str
    descriptor: int | None = None
    identity: tuple[int, ...] | None = None
    acquisition_uncertain: bool = False
    rename_attempted: bool = False
    published: bool = False
    creation_unsynced: bool = False
    removal_unsynced: bool = False
    descriptor_settlement_uncertain: bool = False
    removal_state: _QuarantineRemovalState | None = None

    def _verify_parent(self) -> None:
        if (
            _stable_directory_identity(os.fstat(self.parent_descriptor))
            != self.parent_identity
            or _descriptor_mount_id(self.parent_descriptor) != self.expected_mount_id
        ):
            raise MutationHostQualificationError(
                "directory-creation parent authority changed"
            )

    def _owned_name(self) -> str | None:
        descriptor = self.descriptor
        identity = self.identity
        if descriptor is None or identity is None:
            return None
        names: list[str] = []
        for name in (self.private_name, self.final_name):
            observed = _named_entry_metadata(self.parent_descriptor, name)
            if (
                observed is not None
                and _stable_directory_identity(observed) == identity
            ):
                names.append(name)
        if len(names) > 1:
            raise MutationHostLifecycleError(
                "directory creation acquired ambiguous names"
            )
        return names[0] if names else None

    def create(self) -> None:
        self._verify_parent()
        if _named_entry_metadata(self.parent_descriptor, self.final_name) is not None:
            raise FileExistsError(self.final_name)
        self.acquisition_uncertain = True
        try:
            descriptor = _mkdirat_open_directory(
                self.parent_descriptor,
                self.private_name,
                self.mode,
            )
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "descriptor-bound directory creation outcome is uncertain"
            ) from exc
        self.descriptor = descriptor
        self.acquisition_uncertain = False
        metadata = os.fstat(descriptor)
        self.identity = _stable_directory_identity(metadata)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _descriptor_mount_id(descriptor) != self.expected_mount_id
            or _stable_directory_identity(
                os.stat(
                    self.private_name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            )
            != self.identity
        ):
            raise MutationHostQualificationError(
                "private directory changed before descriptor binding"
            )
        chmod_error: BaseException | None = None
        try:
            os.fchmod(descriptor, self.mode)
        except BaseException as exc:
            chmod_error = exc
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != self.mode:
            raise MutationHostLifecycleError(
                "private directory mode outcome is uncertain"
            ) from chmod_error

    def publish_and_sync(self) -> None:
        descriptor = self.descriptor
        identity = self.identity
        if descriptor is None or identity is None or self.acquisition_uncertain:
            raise MutationHostLifecycleError(
                "directory creation authority is incomplete"
            )
        self._verify_parent()
        private = _named_entry_metadata(self.parent_descriptor, self.private_name)
        final = _named_entry_metadata(self.parent_descriptor, self.final_name)
        if (
            private is None
            or _stable_directory_identity(private) != identity
            or final is not None
        ):
            raise MutationHostLifecycleError(
                "directory publication precondition changed"
            )
        self.rename_attempted = True
        rename_error: BaseException | None = None
        try:
            _rename_noreplace(
                self.parent_descriptor,
                self.private_name,
                self.final_name,
            )
        except BaseException as exc:
            rename_error = exc
        private = _named_entry_metadata(self.parent_descriptor, self.private_name)
        final = _named_entry_metadata(self.parent_descriptor, self.final_name)
        if (
            private is not None
            or final is None
            or _stable_directory_identity(final) != identity
            or _stable_directory_identity(os.fstat(descriptor)) != identity
        ):
            raise MutationHostLifecycleError(
                "directory publication rename outcome is uncertain"
            ) from rename_error
        self.rename_attempted = False
        self.published = True
        self.creation_unsynced = True
        try:
            os.fsync(self.parent_descriptor)
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "directory publication fsync outcome is uncertain"
            ) from exc
        self._verify_parent()
        final = _named_entry_metadata(self.parent_descriptor, self.final_name)
        if (
            final is None
            or _stable_directory_identity(final) != identity
            or _stable_directory_identity(os.fstat(descriptor)) != identity
            or _descriptor_mount_id(descriptor) != self.expected_mount_id
            or _named_entry_metadata(self.parent_descriptor, self.private_name)
            is not None
        ):
            raise MutationHostQualificationError(
                "directory publication changed after fsync"
            )
        self.creation_unsynced = False

    def rollback(self) -> None:
        if self.descriptor_settlement_uncertain:
            raise MutationHostLifecycleError(
                "directory creation descriptor settlement remains uncertain"
            )
        if self.acquisition_uncertain:
            raise MutationHostLifecycleError(
                "directory creation acquisition remains uncertain"
            )
        descriptor = self.descriptor
        identity = self.identity
        if descriptor is None:
            return
        if identity is None:
            raise MutationHostLifecycleError(
                "directory creation identity is unavailable"
            )
        self._verify_parent()
        removal_state = self.removal_state
        if removal_state is not None and removal_state.active:
            owned_name = removal_state.source_name
        else:
            owned_name = self._owned_name()
        if owned_name is None:
            if os.fstat(descriptor).st_nlink != 0:
                raise MutationHostLifecycleError(
                    "owned directory has an unknown surviving name"
                )
        else:
            if removal_state is None:
                removal_state = _QuarantineRemovalState()
                self.removal_state = removal_state
            _quarantine_remove_bound_entry(
                self.parent_descriptor,
                owned_name,
                descriptor,
                directory=True,
                token=self.token,
                removal_state=removal_state,
            )
            if os.fstat(descriptor).st_nlink != 0:
                raise MutationHostLifecycleError(
                    "owned directory survived quarantine removal"
                )
        self.removal_unsynced = True
        try:
            os.fsync(self.parent_descriptor)
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "directory rollback fsync outcome is uncertain"
            ) from exc
        self._verify_parent()
        if self._owned_name() is not None or os.fstat(descriptor).st_nlink != 0:
            raise MutationHostLifecycleError(
                "directory rollback did not settle exact absence"
            )
        self.removal_unsynced = False
        self.descriptor = None
        try:
            os.close(descriptor)
        except OSError as exc:
            self.descriptor_settlement_uncertain = True
            raise MutationHostLifecycleError(
                "directory creation descriptor close failed"
            ) from exc


class _StatxTimestamp(ctypes.Structure):
    _fields_ = [
        ("seconds", ctypes.c_int64),
        ("nanoseconds", ctypes.c_uint32),
        ("reserved", ctypes.c_int32),
    ]


class _StatxRecord(ctypes.Structure):
    _fields_ = [
        ("mask", ctypes.c_uint32),
        ("block_size", ctypes.c_uint32),
        ("attributes", ctypes.c_uint64),
        ("link_count", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("mode", ctypes.c_uint16),
        ("reserved0", ctypes.c_uint16),
        ("inode", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("blocks", ctypes.c_uint64),
        ("attributes_mask", ctypes.c_uint64),
        ("atime", _StatxTimestamp),
        ("btime", _StatxTimestamp),
        ("ctime", _StatxTimestamp),
        ("mtime", _StatxTimestamp),
        ("rdev_major", ctypes.c_uint32),
        ("rdev_minor", ctypes.c_uint32),
        ("dev_major", ctypes.c_uint32),
        ("dev_minor", ctypes.c_uint32),
        ("mount_id", ctypes.c_uint64),
        ("dio_memory_alignment", ctypes.c_uint32),
        ("dio_offset_alignment", ctypes.c_uint32),
        ("reserved3", ctypes.c_uint64 * 12),
    ]


def _statx_mount_id(path: Path, *, nofollow: bool) -> int:
    """Read a Linux mount identity without acquiring a descriptor."""

    try:
        statx = ctypes.CDLL(None, use_errno=True).statx
    except (AttributeError, OSError) as exc:
        raise MutationHostQualificationError(
            "path mount identity is unavailable"
        ) from exc
    statx.restype = ctypes.c_int
    record = _StatxRecord()
    result = statx(
        -100,  # AT_FDCWD
        os.fsencode(path),
        0x100 if nofollow else 0,  # AT_SYMLINK_NOFOLLOW
        0x1000,  # STATX_MNT_ID
        ctypes.byref(record),
    )
    if result != 0:
        error = ctypes.get_errno()
        raise MutationHostQualificationError(
            "path mount identity is unavailable"
        ) from OSError(error, os.strerror(error))
    if record.mask & 0x1000 != 0x1000 or record.mount_id <= 0:
        raise MutationHostQualificationError("path mount identity is unavailable")
    return int(record.mount_id)


def _path_mount_id(path: Path) -> int:
    return _statx_mount_id(path, nofollow=True)


@dataclass(frozen=True, slots=True)
class _LoadedSourceImage:
    """Write-sealed source acquired by the already-trusted package loader."""

    descriptor: int
    identity: tuple[int, ...]
    sha256: str
    seals: int


_PRESEALED_HOST_SOURCE_DESCRIPTOR: int
_PRESEALED_HOST_SOURCE_IDENTITY: tuple[int, ...]
_PRESEALED_HOST_SOURCE_SHA256: str
_PRESEALED_HOST_SOURCE_SEALS: int
_PRESEALED_HOST_SOURCE_LOCK: threading.RLock
_PRESEALED_HOST_MODULE_LOADER: object
__loader__: object


if __name__ == "__main__":
    _LOADED_SOURCE_IMAGE: _LoadedSourceImage | None = None
    _SOURCE_IMAGE_LOCK: threading.RLock | None = None
else:
    try:
        _TRUSTED_SEALED_MODULE_LOADER = _PRESEALED_HOST_MODULE_LOADER
        _presealed_descriptor = int(_PRESEALED_HOST_SOURCE_DESCRIPTOR)
        _presealed_identity = tuple(_PRESEALED_HOST_SOURCE_IDENTITY)
        _presealed_sha256 = str(_PRESEALED_HOST_SOURCE_SHA256)
        _presealed_seals = int(_PRESEALED_HOST_SOURCE_SEALS)
        _SOURCE_IMAGE_LOCK = _PRESEALED_HOST_SOURCE_LOCK
    except (NameError, TypeError, ValueError) as exc:
        raise MutationHostQualificationError(
            "isolated mutation host was not executed by its trusted sealed loader"
        ) from exc
    if (
        len(_presealed_identity) != 7
        or len(_presealed_sha256) != 64
        or _presealed_seals & _SOURCE_IMAGE_REQUIRED_SEALS
        != _SOURCE_IMAGE_REQUIRED_SEALS
    ):
        raise MutationHostQualificationError(
            "isolated mutation sealed-loader authority is malformed"
        )
    _LOADED_SOURCE_IMAGE = _LoadedSourceImage(
        descriptor=_presealed_descriptor,
        identity=_presealed_identity,
        sha256=_presealed_sha256,
        seals=_presealed_seals,
    )


def _capture_loaded_source_image() -> _LoadedSourceImage | None:
    """Return only the image established by the trusted enforcing loader."""

    if __name__ == "__main__":
        return None
    if __loader__ is not _TRUSTED_SEALED_MODULE_LOADER:
        raise MutationHostQualificationError(
            "isolated mutation loader disagrees with its trusted digest authority"
        )
    return _LOADED_SOURCE_IMAGE


@dataclass(slots=True)
class _ChildSourceAuthority:
    """Invocation-owned duplicate of the package's sealed source image."""

    image: _LoadedSourceImage
    descriptor: int
    child_source_descriptor: int = _CHILD_SOURCE_DESCRIPTOR
    acquisition_uncertain: bool = False
    closed: bool = False

    @property
    def pass_descriptor(self) -> int:
        self.revalidate()
        return self.descriptor

    def revalidate(self) -> None:
        if self.acquisition_uncertain or self.descriptor < 0 or self.closed:
            raise MutationHostLifecycleError(
                "isolated mutation source duplicate authority is incomplete"
            )
        try:
            observed = os.fstat(self.descriptor)
            seals = int(fcntl.fcntl(self.descriptor, _F_GET_SEALS))
            if (
                _sealed_source_identity(observed) != self.image.identity
                or not stat.S_ISREG(observed.st_mode)
                or observed.st_size <= 0
                or seals != self.image.seals
                or seals & _SOURCE_IMAGE_REQUIRED_SEALS != _SOURCE_IMAGE_REQUIRED_SEALS
                or os.get_inheritable(self.descriptor)
                or _descriptor_sha256(
                    self.descriptor,
                    expected_identity=_stat_identity(observed),
                )
                != self.image.sha256
            ):
                raise MutationHostQualificationError(
                    "isolated mutation sealed source authority changed"
                )
        except MutationHostQualificationError:
            raise
        except OSError as exc:
            raise MutationHostQualificationError(
                "isolated mutation sealed source authority is unavailable"
            ) from exc

    def close(self) -> None:
        if self.acquisition_uncertain:
            raise MutationHostLifecycleError(
                "isolated mutation source duplication remains uncertain"
            )
        if self.closed:
            return
        descriptor = self.descriptor
        self.closed = True
        try:
            os.close(descriptor)
        except OSError as exc:
            raise MutationHostLifecycleError(
                "isolated mutation source duplicate cleanup failed"
            ) from exc

    def expected_ready_frame(self, *, pid: int, sid: int) -> dict[str, Any]:
        self.revalidate()
        return {
            "type": "ready",
            "pid": pid,
            "pgid": pid,
            "sid": sid,
            "source_identity": list(self.image.identity),
            "source_sha256": self.image.sha256,
            "source_seals": self.image.seals,
            "closure": "sealed-loaded-source-v2",
        }


def _provisional_child_source_authority() -> _ChildSourceAuthority:
    if (
        not Path("/proc/self/fd").is_dir()
        or _LOADED_SOURCE_IMAGE is None
        or _SOURCE_IMAGE_LOCK is None
    ):
        raise MutationHostQualificationError(
            "isolated mutation sealed source binding is unavailable"
        )
    return _ChildSourceAuthority(image=_LOADED_SOURCE_IMAGE, descriptor=-1)


def _open_child_source_authority(authority: _ChildSourceAuthority) -> None:
    if _SOURCE_IMAGE_LOCK is None:
        raise MutationHostQualificationError(
            "isolated mutation sealed source lock is unavailable"
        )
    with _SOURCE_IMAGE_LOCK:
        image = authority.image
        observed = os.fstat(image.descriptor)
        if (
            _sealed_source_identity(observed) != image.identity
            or int(fcntl.fcntl(image.descriptor, _F_GET_SEALS)) != image.seals
            or _descriptor_sha256(
                image.descriptor,
                expected_identity=_stat_identity(observed),
            )
            != image.sha256
        ):
            raise MutationHostQualificationError(
                "isolated mutation package source image changed"
            )
        authority.acquisition_uncertain = True
        try:
            descriptor = int(
                fcntl.fcntl(
                    image.descriptor,
                    fcntl.F_DUPFD_CLOEXEC,
                    _SPAWN_PROJECTION_FD_MINIMUM,
                )
            )
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "isolated mutation source duplication outcome is uncertain"
            ) from exc
        authority.descriptor = descriptor
        authority.acquisition_uncertain = False
        authority.revalidate()


@dataclass(slots=True)
class _PrivateRootAuthority:
    """Descriptor- and mount-bound ownership of one private execution root."""

    parent_path: Path
    parent_descriptor: int | None = None
    parent_identity: tuple[int, ...] | None = None
    parent_mount_id: int | None = None
    parent_acquisition_uncertain: bool = False
    root_name: str | None = None
    root_path: Path | None = None
    root_descriptor: int | None = None
    root_identity: tuple[int, ...] | None = None
    root_mount_id: int | None = None
    root_creation_uncertain: bool = False
    root_descriptor_acquisition_uncertain: bool = False
    pending_creation: _BoundDirectoryCreation | None = None
    cleanup_acquisition_uncertain: bool = False
    acquisition_complete: bool = False
    root_removed: bool = False
    root_removal_unsynced: bool = False

    @property
    def has_created_root(self) -> bool:
        return self.root_name is not None

    @property
    def child_descriptor(self) -> int:
        if not self.acquisition_complete or self.root_descriptor is None:
            raise MutationHostQualificationError(
                "isolated mutation private-root authority is incomplete"
            )
        return self.root_descriptor

    def ready_fields(self) -> dict[str, Any]:
        if self.root_identity is None or self.root_mount_id is None:
            raise MutationHostQualificationError(
                "isolated mutation private-root identity is incomplete"
            )
        return {
            "private_root_identity": list(self.root_identity),
            "private_root_mount_id": self.root_mount_id,
        }

    def revalidate(self, *, require_named_root: bool) -> None:
        parent_descriptor = self.parent_descriptor
        root_descriptor = self.root_descriptor
        if (
            not self.acquisition_complete
            or parent_descriptor is None
            or root_descriptor is None
            or self.parent_identity is None
            or self.parent_mount_id is None
            or self.root_name is None
            or self.root_identity is None
            or self.root_mount_id is None
        ):
            raise MutationHostQualificationError(
                "isolated mutation private-root authority is incomplete"
            )
        try:
            if (
                _stable_directory_identity(os.fstat(parent_descriptor))
                != self.parent_identity
                or _descriptor_mount_id(parent_descriptor) != self.parent_mount_id
                or _stable_directory_identity(os.fstat(root_descriptor))
                != self.root_identity
                or _descriptor_mount_id(root_descriptor) != self.root_mount_id
            ):
                raise MutationHostQualificationError(
                    "isolated mutation private-root descriptor identity changed"
                )
            if require_named_root:
                named = os.stat(
                    self.root_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if _stable_directory_identity(
                    named
                ) != self.root_identity or not stat.S_ISDIR(named.st_mode):
                    raise MutationHostQualificationError(
                        "isolated mutation private-root name was rebound"
                    )
        except MutationHostQualificationError:
            raise
        except OSError as exc:
            raise MutationHostQualificationError(
                "isolated mutation private-root authority is unavailable"
            ) from exc


def _provisional_private_root_authority() -> _PrivateRootAuthority:
    """Create parent authority before attempting any private-root mutation."""

    parent_path = Path(tempfile.gettempdir()).resolve(strict=True)
    if not parent_path.is_dir():
        raise MutationHostQualificationError(
            "isolated mutation temporary parent is not a directory"
        )
    return _PrivateRootAuthority(parent_path=parent_path)


def _open_private_root_authority(authority: _PrivateRootAuthority) -> None:
    """Create and bind one private root beneath an exact held parent."""

    parent_before = authority.parent_path.stat()
    parent_descriptor: int | None = None
    try:
        parent_descriptor = os.open(
            authority.parent_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except BaseException as exc:
        if parent_descriptor is not None:
            authority.parent_descriptor = parent_descriptor
        else:
            authority.parent_acquisition_uncertain = True
        raise MutationHostLifecycleError(
            "private-root parent descriptor acquisition outcome is uncertain"
        ) from exc
    assert parent_descriptor is not None
    authority.parent_descriptor = parent_descriptor
    authority.parent_identity = _stable_directory_identity(parent_before)
    authority.parent_mount_id = _descriptor_mount_id(parent_descriptor)
    if (
        _stable_directory_identity(os.fstat(parent_descriptor))
        != authority.parent_identity
        or _stable_directory_identity(authority.parent_path.stat())
        != authority.parent_identity
    ):
        raise MutationHostQualificationError(
            "isolated mutation temporary parent changed while opening"
        )
    root_name = f"{_PRIVATE_ROOT_PREFIX}{secrets.token_hex(16)}"
    authority.root_name = root_name
    authority.root_path = authority.parent_path / root_name
    parent_identity = authority.parent_identity
    parent_mount_id = authority.parent_mount_id
    if parent_identity is None or parent_mount_id is None:
        raise MutationHostLifecycleError("private-root parent authority is incomplete")
    pending = _BoundDirectoryCreation(
        parent_descriptor=parent_descriptor,
        parent_identity=parent_identity,
        expected_mount_id=parent_mount_id,
        final_name=root_name,
        private_name=f".{root_name}.bind-{secrets.token_hex(16)}",
        mode=0o700,
        token=secrets.token_hex(16),
    )
    authority.pending_creation = pending
    pending.create()
    pending.publish_and_sync()
    root_descriptor = pending.descriptor
    root_identity = pending.identity
    if root_descriptor is None or root_identity is None:
        raise MutationHostLifecycleError(
            "private-root creation lost its retained authority"
        )
    root_mount_id = _descriptor_mount_id(root_descriptor)
    if root_mount_id != parent_mount_id:
        raise MutationHostQualificationError(
            "private-root creation crossed its retained mount"
        )
    # Complete every fallible validation while ``pending`` remains the sole
    # descriptor owner.  Prepopulate non-owning metadata, then make the final
    # detach/attach handoff inside a recovery guard so an asynchronous
    # exception cannot leave two cleanup owners for one numeric descriptor.
    authority.root_identity = root_identity
    authority.root_mount_id = root_mount_id
    try:
        pending.descriptor = None
        authority.root_descriptor = root_descriptor
    except BaseException:
        if authority.root_descriptor is None:
            pending.descriptor = root_descriptor
        raise
    authority.pending_creation = None
    authority.acquisition_complete = True


def _bounded_string(
    value: Any,
    *,
    setting: str,
    maximum: int = 16_384,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or (not value and not allow_empty)
        or len(value) > maximum
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise MutationHostQualificationError(
            f"{setting} must be a bounded string without control characters"
        )
    return value


def _string_tuple(value: Any, *, setting: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise MutationHostQualificationError(f"{setting} must be a string list")
    rendered = tuple(
        _bounded_string(item, setting=f"{setting} entry") for item in value
    )
    if len(rendered) > 4096:
        raise MutationHostQualificationError(f"{setting} is too large")
    return rendered


def _positive_int(value: Any, *, setting: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MutationHostQualificationError(f"{setting} must be a positive integer")
    return value


def _identity_tuple_from_wire(value: Any, *, setting: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 5
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise MutationHostQualificationError(
            f"{setting} must be one closed directory identity"
        )
    return tuple(value)


@dataclass(frozen=True)
class MutationContextSpec:
    """The ToolContext subset consumed by canonical source mutators."""

    workspace_root: str
    workspace_root_identity: tuple[int, ...]
    workspace_root_mount_id: int
    repo_root: str
    repo_root_identity: tuple[int, ...]
    repo_root_mount_id: int
    workspace_relative_paths: bool
    write_scope: tuple[str, ...]
    write_exact_scope: tuple[str, ...]
    write_deny_scope: tuple[str, ...]
    mutation_journal_dir: str | None
    mutation_journal_transaction_id: str | None
    container_id: str | None
    container_workdir: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "workspace_root": self.workspace_root,
            "workspace_root_identity": list(self.workspace_root_identity),
            "workspace_root_mount_id": self.workspace_root_mount_id,
            "repo_root": self.repo_root,
            "repo_root_identity": list(self.repo_root_identity),
            "repo_root_mount_id": self.repo_root_mount_id,
            "workspace_relative_paths": self.workspace_relative_paths,
            "write_scope": list(self.write_scope),
            "write_exact_scope": list(self.write_exact_scope),
            "write_deny_scope": list(self.write_deny_scope),
            "mutation_journal_dir": self.mutation_journal_dir,
            "mutation_journal_transaction_id": (
                self.mutation_journal_transaction_id
            ),
            "container_id": self.container_id,
            "container_workdir": self.container_workdir,
        }

    @classmethod
    def from_wire(cls, payload: Any) -> MutationContextSpec:
        if not isinstance(payload, Mapping):
            raise MutationHostQualificationError(
                "mutation context specification must be an object"
            )
        expected = {
            "workspace_root",
            "workspace_root_identity",
            "workspace_root_mount_id",
            "repo_root",
            "repo_root_identity",
            "repo_root_mount_id",
            "workspace_relative_paths",
            "write_scope",
            "write_exact_scope",
            "write_deny_scope",
            "mutation_journal_dir",
            "mutation_journal_transaction_id",
            "container_id",
            "container_workdir",
        }
        if set(payload) != expected:
            raise MutationHostQualificationError(
                "mutation context specification has unknown or missing fields"
            )
        workspace_root = Path(
            _bounded_string(
                payload["workspace_root"],
                setting="mutation workspace_root",
            )
        )
        repo_root = Path(
            _bounded_string(
                payload["repo_root"],
                setting="mutation repo_root",
            )
        )
        if (
            not workspace_root.is_absolute()
            or not repo_root.is_absolute()
            or str(workspace_root) != payload["workspace_root"]
            or str(repo_root) != payload["repo_root"]
            or any(part in {"", ".", ".."} for part in workspace_root.parts[1:])
            or any(part in {"", ".", ".."} for part in repo_root.parts[1:])
        ):
            raise MutationHostQualificationError(
                "mutation roots must be absolute directory labels"
            )
        workspace_root_identity = _identity_tuple_from_wire(
            payload["workspace_root_identity"],
            setting="mutation workspace_root_identity",
        )
        repo_root_identity = _identity_tuple_from_wire(
            payload["repo_root_identity"],
            setting="mutation repo_root_identity",
        )
        workspace_root_mount_id = _positive_int(
            payload["workspace_root_mount_id"],
            setting="mutation workspace_root_mount_id",
        )
        repo_root_mount_id = _positive_int(
            payload["repo_root_mount_id"],
            setting="mutation repo_root_mount_id",
        )
        workspace_relative_paths = payload["workspace_relative_paths"]
        if not isinstance(workspace_relative_paths, bool):
            raise MutationHostQualificationError(
                "workspace_relative_paths must be boolean"
            )
        container_id = payload["container_id"]
        if container_id is not None:
            container_id = _bounded_string(
                container_id,
                setting="mutation container_id",
                maximum=1024,
            )
        container_workdir = _bounded_string(
            payload["container_workdir"],
            setting="mutation container_workdir",
            maximum=4096,
        )
        journal_dir_raw = payload["mutation_journal_dir"]
        journal_transaction_id = payload[
            "mutation_journal_transaction_id"
        ]
        if (journal_dir_raw is None) != (journal_transaction_id is None):
            raise MutationHostQualificationError(
                "mutation journal directory and transaction identity must "
                "be supplied together"
            )
        journal_dir: str | None = None
        if journal_dir_raw is not None:
            journal_path = Path(_bounded_string(
                journal_dir_raw,
                setting="mutation journal directory",
                maximum=4096,
            ))
            journal_transaction_id = _bounded_string(
                journal_transaction_id,
                setting="mutation journal transaction identity",
                maximum=32,
            )
            if (
                not journal_path.is_absolute()
                or len(journal_transaction_id) != 32
                or any(
                    character not in "0123456789abcdef"
                    for character in journal_transaction_id
                )
            ):
                raise MutationHostQualificationError(
                    "mutation journal identity is malformed"
                )
            journal_path = Path(os.path.abspath(os.fspath(journal_path)))
            try:
                journal_metadata = journal_path.lstat()
            except OSError as exc:
                raise MutationHostQualificationError(
                    "mutation journal directory is unavailable"
                ) from exc
            if (
                not stat.S_ISDIR(journal_metadata.st_mode)
                or stat.S_ISLNK(journal_metadata.st_mode)
            ):
                raise MutationHostQualificationError(
                    "mutation journal directory must be a real directory"
                )
            journal_dir = str(journal_path)
        return cls(
            workspace_root=str(workspace_root),
            workspace_root_identity=workspace_root_identity,
            workspace_root_mount_id=workspace_root_mount_id,
            repo_root=str(repo_root),
            repo_root_identity=repo_root_identity,
            repo_root_mount_id=repo_root_mount_id,
            workspace_relative_paths=workspace_relative_paths,
            write_scope=_string_tuple(
                payload["write_scope"],
                setting="mutation write_scope",
            ),
            write_exact_scope=_string_tuple(
                payload["write_exact_scope"],
                setting="mutation write_exact_scope",
            ),
            write_deny_scope=_string_tuple(
                payload["write_deny_scope"],
                setting="mutation write_deny_scope",
            ),
            mutation_journal_dir=journal_dir,
            mutation_journal_transaction_id=journal_transaction_id,
            container_id=container_id,
            container_workdir=container_workdir,
        )


@dataclass(frozen=True)
class MutationToolBinding:
    """One canonical writer reconstructed by name in the child."""

    name: str
    kind: MutationKind
    context: MutationContextSpec
    exact_target: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "context": self.context.to_wire(),
            "exact_target": self.exact_target,
        }

    @classmethod
    def from_wire(cls, payload: Any) -> MutationToolBinding:
        if not isinstance(payload, Mapping) or set(payload) != {
            "name",
            "kind",
            "context",
            "exact_target",
        }:
            raise MutationHostQualificationError(
                "mutation tool binding has unknown or missing fields"
            )
        name = _bounded_string(
            payload["name"],
            setting="mutation tool name",
            maximum=128,
        )
        kind = payload["kind"]
        if kind not in _SUPPORTED_KINDS:
            raise MutationHostQualificationError(
                f"unsupported isolated mutation kind: {kind!r}"
            )
        exact_target = payload["exact_target"]
        if exact_target is not None:
            exact_target = _bounded_string(
                exact_target,
                setting="exact mutation target",
            )
        if kind == "exact_write_source":
            if name != "write_source" or exact_target is None:
                raise MutationHostQualificationError(
                    "exact writer must bind write_source to one target"
                )
        elif exact_target is not None or name != kind:
            raise MutationHostQualificationError(
                "ordinary writer binding does not match its canonical name"
            )
        return cls(
            name=name,
            kind=kind,
            context=MutationContextSpec.from_wire(payload["context"]),
            exact_target=exact_target,
        )


@dataclass(frozen=True)
class IsolatedToolSetSpec:
    """Versioned reconstructable authority for one invocation."""

    bindings: tuple[MutationToolBinding, ...]

    @property
    def names(self) -> frozenset[str]:
        return frozenset(binding.name for binding in self.bindings)

    def binding_for(self, name: str) -> MutationToolBinding:
        for binding in self.bindings:
            if binding.name == name:
                return binding
        raise MutationHostProtocolError(
            f"isolated mutation host has no binding for {name!r}"
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema": _SPEC_SCHEMA,
            "bindings": [binding.to_wire() for binding in self.bindings],
        }

    @classmethod
    def from_wire(cls, payload: Any) -> IsolatedToolSetSpec:
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema", "bindings"}
            or payload.get("schema") != _SPEC_SCHEMA
            or not isinstance(payload.get("bindings"), list)
        ):
            raise MutationHostQualificationError(
                "invalid isolated mutation ToolSet specification"
            )
        bindings = tuple(
            MutationToolBinding.from_wire(item) for item in payload["bindings"]
        )
        if not bindings:
            raise MutationHostQualificationError(
                "isolated mutation ToolSet must contain at least one binding"
            )
        names = tuple(binding.name for binding in bindings)
        if len(set(names)) != len(names) or tuple(sorted(names)) != names:
            raise MutationHostQualificationError(
                "isolated mutation bindings must be sorted and unique"
            )
        return cls(bindings=bindings)


@dataclass(slots=True)
class _MutationRootAuthority:
    """One exact path label bound to a retained directory capability."""

    path: Path
    expected_identity: tuple[int, ...]
    expected_mount_id: int
    child_descriptor: int
    descriptor: int | None = None
    acquisition_in_progress: bool = False
    acquisition_uncertain: bool = False
    cleanup_acquisition_uncertain: bool = False

    def acquire(self) -> None:
        if (
            self.descriptor is not None
            or self.acquisition_in_progress
            or self.acquisition_uncertain
        ):
            raise MutationHostLifecycleError(
                "mutation-root descriptor acquisition was already attempted"
            )
        try:
            before = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise MutationHostQualificationError(
                "mutation-root path is unavailable"
            ) from exc
        if (
            not stat.S_ISDIR(before.st_mode)
            or _stable_directory_identity(before) != self.expected_identity
            or _path_mount_id(self.path) != self.expected_mount_id
        ):
            raise MutationHostQualificationError(
                "mutation-root path changed before descriptor binding"
            )
        self.acquisition_in_progress = True
        descriptor: int | None = None
        try:
            descriptor = os.open(
                self.path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
        except BaseException as exc:
            # open(2) returns its descriptor in a register.  A Python-level
            # asynchronous exception can arrive after the kernel effect but
            # before that integer is returned.  Never guess or close a reused
            # number: retain explicit uncertainty and refuse false settlement.
            self.acquisition_in_progress = False
            if descriptor is not None:
                self.descriptor = descriptor
            else:
                self.acquisition_uncertain = True
            raise MutationHostLifecycleError(
                "mutation-root descriptor acquisition outcome is uncertain"
            ) from exc
        self.descriptor = descriptor
        self.acquisition_in_progress = False
        self.revalidate(require_named=True)

    def revalidate(self, *, require_named: bool) -> None:
        descriptor = self.descriptor
        if descriptor is None or self.acquisition_uncertain:
            raise MutationHostLifecycleError(
                "mutation-root descriptor authority is incomplete"
            )
        try:
            if (
                _stable_directory_identity(os.fstat(descriptor))
                != self.expected_identity
                or _descriptor_mount_id(descriptor) != self.expected_mount_id
                or os.get_inheritable(descriptor)
            ):
                raise MutationHostQualificationError(
                    "mutation-root descriptor identity changed"
                )
            if require_named:
                named = os.stat(self.path, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(named.st_mode)
                    or _stable_directory_identity(named) != self.expected_identity
                    or _path_mount_id(self.path) != self.expected_mount_id
                ):
                    raise MutationHostQualificationError(
                        "mutation-root path was rebound"
                    )
        except (MutationHostLifecycleError, MutationHostQualificationError):
            raise
        except OSError as exc:
            raise MutationHostQualificationError(
                "mutation-root authority is unavailable"
            ) from exc

    def close(self) -> None:
        if (
            self.acquisition_in_progress
            or self.acquisition_uncertain
            or self.cleanup_acquisition_uncertain
        ):
            raise MutationHostLifecycleError(
                "mutation-root descriptor acquisition remains uncertain"
            )
        descriptor = self.descriptor
        if descriptor is None:
            return
        self.descriptor = None
        try:
            os.close(descriptor)
        except OSError as exc:
            raise MutationHostLifecycleError(
                "mutation-root descriptor cleanup failed"
            ) from exc


@dataclass(slots=True)
class _MutationRootAuthorities:
    roots: tuple[_MutationRootAuthority, ...]

    @classmethod
    def from_spec(cls, spec: IsolatedToolSetSpec) -> _MutationRootAuthorities:
        observed: dict[str, tuple[tuple[int, ...], int]] = {}
        for binding in spec.bindings:
            context = binding.context
            for path, identity, mount_id in (
                (
                    context.workspace_root,
                    context.workspace_root_identity,
                    context.workspace_root_mount_id,
                ),
                (
                    context.repo_root,
                    context.repo_root_identity,
                    context.repo_root_mount_id,
                ),
            ):
                candidate = (identity, mount_id)
                if path in observed and observed[path] != candidate:
                    raise MutationHostQualificationError(
                        "mutation-root specifications disagree on identity"
                    )
                observed[path] = candidate
        roots = tuple(
            _MutationRootAuthority(
                path=Path(path),
                expected_identity=identity,
                expected_mount_id=mount_id,
                child_descriptor=_CHILD_ROOT_DESCRIPTOR + index + 1,
            )
            for index, (path, (identity, mount_id)) in enumerate(
                sorted(observed.items()),
            )
        )
        if not roots or roots[-1].child_descriptor >= _SPAWN_PROJECTION_FD_MINIMUM:
            raise MutationHostQualificationError(
                "mutation-root descriptor projection is too large"
            )
        return cls(roots=roots)

    def acquire_all(self) -> None:
        for root in self.roots:
            root.acquire()

    def revalidate(self, *, require_named: bool) -> None:
        for root in self.roots:
            root.revalidate(require_named=require_named)

    @property
    def pass_descriptors(self) -> tuple[int, ...]:
        descriptors: list[int] = []
        for root in self.roots:
            if root.descriptor is None:
                raise MutationHostLifecycleError(
                    "mutation-root descriptor authority is incomplete"
                )
            descriptors.append(root.descriptor)
        return tuple(descriptors)

    def wire_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "path": str(root.path),
                "identity": list(root.expected_identity),
                "mount_id": root.expected_mount_id,
                "descriptor": root.child_descriptor,
            }
            for root in self.roots
        ]

    def close(self) -> None:
        failures: list[MutationHostLifecycleError] = []
        for root in reversed(self.roots):
            try:
                root.close()
            except MutationHostLifecycleError as exc:
                failures.append(exc)
        if failures:
            error = MutationHostLifecycleError(
                "mutation-root authority cleanup was incomplete"
            )
            for failure in failures[1:]:
                error.add_note(f"additional cleanup failure: {failure}")
            raise error from failures[0]
@dataclass
class _DeleteReceiptAuthority:
    """Process-live journal owner and its current exact CAS receipt."""

    journal: Any
    receipt: dict[str, object]
    durable: bool = False
    pending_bind: dict[str, object] | None = None
    parent_descriptor: int | None = None
    quarantine_descriptor: int | None = None
    target_identity: tuple[int, ...] | None = None
    parent_acquisition_in_progress: bool = False
    parent_acquisition_uncertain: bool = False
    quarantine_acquisition_in_progress: bool = False
    quarantine_acquisition_uncertain: bool = False
    reproof_descriptor: int | None = None
    reproof_acquisition_in_progress: bool = False
    reproof_acquisition_uncertain: bool = False
    reproof_close_in_progress: bool = False
    descriptor_close_in_progress: bool = False
    descriptor_settlement_uncertain: bool = False


@dataclass(frozen=True)
class _BoundRepositoryRoot:
    """Host-held repository identity used for descriptor-bound recovery."""

    descriptor: int
    device: int
    inode: int
    mount_id: int


def _snapshot_context(
    context: Any,
    *,
    authority_scopes: tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]
    | None = None,
) -> MutationContextSpec:
    from crucible.tools._context import ToolContext  # noqa: PLC0415

    if not isinstance(context, ToolContext):
        raise MutationHostQualificationError(
            "canonical source mutators require a concrete ToolContext"
        )
    repo_root = Path(context.repo_root).resolve(strict=True)
    workspace_root = Path(context.workspace_root).resolve(strict=True)
    repo_metadata = repo_root.stat()
    workspace_metadata = workspace_root.stat()
    if not stat.S_ISDIR(repo_metadata.st_mode) or not stat.S_ISDIR(
        workspace_metadata.st_mode
    ):
        raise MutationHostQualificationError(
            "canonical text writers require directory mutation roots"
        )
    repo_mount_id = _path_mount_id(repo_root)
    workspace_mount_id = _path_mount_id(workspace_root)
    if (
        _stable_directory_identity(repo_root.stat())
        != _stable_directory_identity(repo_metadata)
        or _stable_directory_identity(workspace_root.stat())
        != _stable_directory_identity(workspace_metadata)
        or _path_mount_id(repo_root) != repo_mount_id
        or _path_mount_id(workspace_root) != workspace_mount_id
    ):
        raise MutationHostQualificationError(
            "mutation roots changed while snapshotting authority"
        )
    resolver = context.path_resolver
    if resolver is not None:
        from crucible.workflows._worktree import (  # noqa: PLC0415
            WorktreeResolver,
        )

        if not isinstance(resolver, WorktreeResolver):
            raise MutationHostQualificationError(
                "custom write path resolvers cannot cross the isolated "
                "mutation boundary"
            )
        active_root = resolver.resolve_worktree_root(".")
        if active_root is None or Path(active_root).resolve() != repo_root:
            raise MutationHostQualificationError(
                "isolated mutation requires an already-materialized "
                "worktree matching ToolContext.repo_root"
            )
    if authority_scopes is None:
        authority_scopes = (
            _string_tuple(
                context.write_scope,
                setting="ToolContext.write_scope",
            ),
            _string_tuple(
                context.write_exact_scope,
                setting="ToolContext.write_exact_scope",
            ),
            _string_tuple(
                context.write_deny_scope,
                setting="ToolContext.write_deny_scope",
            ),
        )
    write_scope, write_exact_scope, write_deny_scope = authority_scopes
    journal_dir_value = getattr(context, "mutation_journal_dir", None)
    journal_transaction_id = getattr(
        context,
        "mutation_journal_transaction_id",
        None,
    )
    if (journal_dir_value is None) != (journal_transaction_id is None):
        raise MutationHostQualificationError(
            "ToolContext mutation journal identity is incomplete"
        )
    journal_dir: str | None = None
    if journal_dir_value is not None:
        journal_path = Path(os.path.abspath(os.fspath(journal_dir_value)))
        journal_transaction_id = _bounded_string(
            journal_transaction_id,
            setting="ToolContext mutation journal transaction identity",
            maximum=32,
        )
        if (
            not journal_path.is_absolute()
            or len(journal_transaction_id) != 32
            or any(
                character not in "0123456789abcdef"
                for character in journal_transaction_id
            )
        ):
            raise MutationHostQualificationError(
                "ToolContext mutation journal identity is malformed"
            )
        try:
            journal_metadata = journal_path.lstat()
        except OSError as exc:
            raise MutationHostQualificationError(
                "ToolContext mutation journal directory is unavailable"
            ) from exc
        if (
            not stat.S_ISDIR(journal_metadata.st_mode)
            or stat.S_ISLNK(journal_metadata.st_mode)
        ):
            raise MutationHostQualificationError(
                "ToolContext mutation journal directory must be real"
            )
        journal_dir = str(journal_path)
    return MutationContextSpec(
        workspace_root=str(workspace_root),
        workspace_root_identity=_stable_directory_identity(workspace_metadata),
        workspace_root_mount_id=workspace_mount_id,
        repo_root=str(repo_root),
        repo_root_identity=_stable_directory_identity(repo_metadata),
        repo_root_mount_id=repo_mount_id,
        workspace_relative_paths=bool(context.workspace_relative_paths),
        write_scope=write_scope,
        write_exact_scope=write_exact_scope,
        write_deny_scope=write_deny_scope,
        mutation_journal_dir=journal_dir,
        mutation_journal_transaction_id=journal_transaction_id,
        container_id=(
            None
            if context.container_id is None
            else _bounded_string(
                context.container_id,
                setting="ToolContext.container_id",
                maximum=1024,
            )
        ),
        container_workdir=_bounded_string(
            context.container_workdir,
            setting="ToolContext.container_workdir",
            maximum=4096,
        ),
    )


def _descriptor_by_name(toolset: Any) -> dict[str, dict[str, Any]]:
    descriptors = toolset.to_anthropic_tools()
    if not isinstance(descriptors, list):
        raise MutationHostQualificationError("ToolSet descriptors must be a list")
    result: dict[str, dict[str, Any]] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise MutationHostQualificationError("ToolSet descriptor must be an object")
        name = descriptor.get("name")
        if not isinstance(name, str) or name in result:
            raise MutationHostQualificationError(
                "ToolSet descriptors must have unique string names"
            )
        result[name] = descriptor
    return result


def _canonical_descriptors() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    from crucible.tools._context import ToolContext  # noqa: PLC0415
    from crucible.tools.write import build_exact_create_tools  # noqa: PLC0415
    from crucible.tools.write.handlers import builder  # noqa: PLC0415

    ordinary = _descriptor_by_name(builder.build(None))
    inert_context = ToolContext(
        workspace_root=Path("/"),
        repo_root=Path("/"),
        write_scope=("bound",),
        write_exact_scope=("bound",),
    )
    exact = _descriptor_by_name(
        build_exact_create_tools(
            inert_context,
            target_path="bound",
        )
    )["write_source"]
    return ordinary, exact


def build_isolated_toolset_spec(
    toolset: Any,
    *,
    expected_write_names: frozenset[str],
    authority_snapshot: Any | None = None,
) -> IsolatedToolSetSpec:
    """Qualify and snapshot canonical source mutators before inference."""

    from crucible.tools import BuiltToolSet, CompositeToolSet  # noqa: PLC0415

    canonical, canonical_exact = _canonical_descriptors()
    frozen_descriptors = (
        None if authority_snapshot is None else _descriptor_by_name(authority_snapshot)
    )
    if authority_snapshot is not None and (
        frozenset(authority_snapshot.write_names) != expected_write_names
    ):
        raise MutationHostQualificationError(
            "isolated mutation authority classification changed during snapshot"
        )
    bindings: dict[str, MutationToolBinding] = {}
    unsupported: set[str] = set()

    def visit(candidate: Any) -> None:
        from crucible.tools._dispatch_budget import (  # noqa: PLC0415
            ProgressBoundedMutatingToolSet,
        )

        if isinstance(candidate, ProgressBoundedMutatingToolSet):
            visit(candidate.isolated_mutation_inner_toolset())
            return
        if isinstance(candidate, CompositeToolSet):
            for member in candidate.members:
                visit(member)
            return
        write_names = {str(name) for name in getattr(candidate, "write_names", ())}
        if not write_names:
            return
        if not isinstance(candidate, BuiltToolSet):
            unsupported.update(write_names)
            return
        descriptors = (
            _descriptor_by_name(candidate)
            if frozen_descriptors is None
            else {
                name: frozen_descriptors[name]
                for name in candidate.tool_names
                if name in frozen_descriptors
            }
        )
        if authority_snapshot is None:
            context = _snapshot_context(candidate.context)
        else:
            try:
                bound_scopes = {
                    authority_snapshot.binding_for(name) for name in write_names
                }
            except KeyError as exc:
                raise MutationHostQualificationError(
                    "isolated mutation authority has an unbound writer"
                ) from exc
            if len(bound_scopes) != 1:
                raise MutationHostQualificationError(
                    "isolated mutation leaf writers have inconsistent authority scopes"
                )
            context = _snapshot_context(
                candidate.context,
                authority_scopes=next(iter(bound_scopes)),
            )
        if candidate.domain == "write":
            for name in write_names:
                if name not in {
                    "write_source",
                    "apply_patch",
                    "delete_source",
                }:
                    unsupported.add(name)
                    continue
                if descriptors.get(name) != canonical[name]:
                    unsupported.add(name)
                    continue
                bindings[name] = MutationToolBinding(
                    name=name,
                    kind=name,
                    context=context,
                )
            return
        if candidate.domain == "exact_create":
            bound_target = getattr(
                candidate,
                "isolated_exact_write_target",
                None,
            )
            if (
                write_names != {"write_source"}
                or descriptors.get("write_source") != canonical_exact
                or len(context.write_exact_scope) != 1
                or not isinstance(bound_target, str)
                or bound_target != context.write_exact_scope[0]
            ):
                unsupported.update(write_names)
                return
            bindings["write_source"] = MutationToolBinding(
                name="write_source",
                kind="exact_write_source",
                context=context,
                exact_target=bound_target,
            )
            return
        unsupported.update(write_names)

    visit(toolset)
    missing = set(expected_write_names) - set(bindings)
    extra = set(bindings) - set(expected_write_names)
    unsupported.update(missing)
    unsupported.update(extra)
    if unsupported:
        rendered = ", ".join(sorted(unsupported))
        raise MutationHostQualificationError(
            f"isolated mutation host does not support mutating tools: {rendered}"
        )
    if not bindings:
        raise MutationHostQualificationError(
            "isolated mutation host received no mutating tools"
        )
    return IsolatedToolSetSpec(
        bindings=tuple(bindings[name] for name in sorted(bindings))
    )


def _normalized_relative(relative: str) -> str | None:
    """Return one lexical repository-relative path with VCS roots denied."""

    if not isinstance(relative, str) or not relative or "\x00" in relative:
        return None
    candidate = PurePosixPath(relative.replace("\\", "/"))
    if candidate.is_absolute() or candidate.as_posix() in {"", "."}:
        return None
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    if any(part.casefold() in _VCS_METADATA_COMPONENTS for part in candidate.parts):
        return None
    return candidate.as_posix()


def _container_workdir_relative(
    context: MutationContextSpec,
    absolute: str,
) -> str | None:
    if (
        not absolute
        or "\x00" in absolute
        or "\\" in absolute
        or context.container_id is None
        or not context.container_workdir
        or "\x00" in context.container_workdir
        or "\\" in context.container_workdir
    ):
        return None
    container_root = PurePosixPath(context.container_workdir)
    if not container_root.is_absolute() or any(
        part in {"", ".", ".."} for part in container_root.parts[1:]
    ):
        return None
    candidate = PurePosixPath(absolute)
    if not candidate.is_absolute():
        return None
    try:
        relative = candidate.relative_to(container_root)
    except ValueError:
        return None
    return _normalized_relative(relative.as_posix())


def _context_relative(
    context: MutationContextSpec,
    relative: str,
) -> str | None:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        return None
    candidate = PurePosixPath(relative)
    normalized = (
        _container_workdir_relative(context, relative)
        if candidate.is_absolute()
        else _normalized_relative(relative)
    )
    if normalized is None:
        return None
    candidate = PurePosixPath(normalized)
    parts = candidate.parts
    repo_name = PurePosixPath(context.repo_root).name
    if (
        context.workspace_relative_paths
        and parts
        and parts[0] == repo_name
        and len(parts) > 1
    ):
        candidate = PurePosixPath(*parts[1:])
    return candidate.as_posix()


def _write_is_authorized(
    context: MutationContextSpec,
    relative: str,
) -> str | None:
    normalized = _context_relative(context, relative)
    if normalized is None:
        return None
    roots: list[str] = []
    for rendered in context.write_scope:
        root = _context_relative(context, rendered)
        if root is None:
            return None
        roots.append(root)
    if not roots or not any(
        normalized == root or normalized.startswith(root.rstrip("/") + "/")
        for root in roots
    ):
        return None
    exact_paths: list[str] = []
    for rendered in context.write_exact_scope:
        exact = _context_relative(context, rendered)
        if exact is None:
            return None
        exact_paths.append(exact)
    if exact_paths and normalized not in exact_paths:
        return None
    denied_roots: list[str] = []
    for rendered in context.write_deny_scope:
        denied = _context_relative(context, rendered)
        if denied is None:
            return None
        denied_roots.append(denied)
    if any(
        normalized == denied or normalized.startswith(denied.rstrip("/") + "/")
        for denied in denied_roots
    ):
        return None
    return normalized


@dataclass(frozen=True, slots=True)
class _ChildMutationRootAuthority:
    path: str
    descriptor: int
    identity: tuple[int, ...]
    mount_id: int

    def revalidate(self, *, require_named: bool) -> None:
        try:
            if (
                _stable_directory_identity(os.fstat(self.descriptor)) != self.identity
                or _descriptor_mount_id(self.descriptor) != self.mount_id
                or os.get_inheritable(self.descriptor)
            ):
                raise MutationHostQualificationError(
                    "mutation-root child descriptor changed"
                )
            if require_named:
                named = os.stat(self.path, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(named.st_mode)
                    or _stable_directory_identity(named) != self.identity
                    or _path_mount_id(Path(self.path)) != self.mount_id
                ):
                    raise MutationHostQualificationError(
                        "mutation-root path was rebound"
                    )
        except MutationHostQualificationError:
            raise
        except OSError as exc:
            raise MutationHostQualificationError(
                "mutation-root child authority is unavailable"
            ) from exc


_CHILD_MUTATION_ROOTS: dict[str, _ChildMutationRootAuthority] = {}


class _ChildAcquisitionUncertain(MutationHostLifecycleError):
    """A child-side acquisition may have taken effect before raising."""

    def __init__(self, operation: str, root_path: str, relative_path: str) -> None:
        super().__init__(
            f"{operation} acquisition outcome is uncertain for {relative_path}"
        )
        self.operation = operation
        self.root_path = root_path
        self.relative_path = relative_path


@dataclass(slots=True)
class _HeldRelativeParent:
    root: _ChildMutationRootAuthority
    descriptor: int
    owns_descriptor: bool
    relative_parent: str
    leaf: str

    def close(self) -> None:
        if not self.owns_descriptor:
            return
        descriptor = self.descriptor
        self.owns_descriptor = False
        try:
            os.close(descriptor)
        except OSError as exc:
            raise MutationHostLifecycleError(
                "mutation-path parent descriptor cleanup failed"
            ) from exc


def _child_root_for_context(
    context: MutationContextSpec,
) -> _ChildMutationRootAuthority:
    root = _CHILD_MUTATION_ROOTS.get(context.repo_root)
    if (
        root is None
        or root.identity != context.repo_root_identity
        or root.mount_id != context.repo_root_mount_id
    ):
        raise MutationHostQualificationError(
            "mutation context has no exact retained repo-root authority"
        )
    root.revalidate(require_named=True)
    return root


def _open_relative_parent(
    context: MutationContextSpec,
    normalized: str,
    *,
    create_parents: bool,
) -> _HeldRelativeParent:
    root = _child_root_for_context(context)
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise MutationHostQualificationError("mutation target is not normalized")
    descriptor = root.descriptor
    owns_descriptor = False
    traversed: list[str] = []
    try:
        for part in parts[:-1]:
            traversed.append(part)
            try:
                observed = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not create_parents:
                    raise
                relative_directory = PurePosixPath(*traversed).as_posix()
                try:
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                except BaseException as exc:
                    raise _ChildAcquisitionUncertain(
                        "mkdir",
                        root.path,
                        relative_directory,
                    ) from exc
                observed = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            if not stat.S_ISDIR(observed.st_mode):
                raise MutationHostQualificationError(
                    "mutation path parent is not a directory"
                )
            try:
                child_descriptor = os.open(
                    part,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except BaseException as exc:
                raise _ChildAcquisitionUncertain(
                    "open",
                    root.path,
                    PurePosixPath(*traversed).as_posix(),
                ) from exc
            if (
                _stable_directory_identity(os.fstat(child_descriptor))
                != _stable_directory_identity(observed)
                or _descriptor_mount_id(child_descriptor) != root.mount_id
            ):
                os.close(child_descriptor)
                raise MutationHostQualificationError(
                    "mutation parent changed identity or crossed a mount"
                )
            if owns_descriptor:
                prior = descriptor
                owns_descriptor = False
                os.close(prior)
            descriptor = child_descriptor
            owns_descriptor = True
        return _HeldRelativeParent(
            root=root,
            descriptor=descriptor,
            owns_descriptor=owns_descriptor,
            relative_parent=(
                "." if len(parts) == 1 else PurePosixPath(*parts[:-1]).as_posix()
            ),
            leaf=parts[-1],
        )
    except BaseException:
        if owns_descriptor:
            prior = descriptor
            owns_descriptor = False
            try:
                os.close(prior)
            except OSError as close_error:
                raise MutationHostLifecycleError(
                    "mutation-path descriptor cleanup failed"
                ) from close_error
        raise


def _reprove_child_named_parent(
    context: MutationContextSpec,
    normalized: str,
    held: _HeldRelativeParent,
) -> None:
    """Prove the live named chain still terminates at the held parent fd."""

    held.root.revalidate(require_named=True)
    current = _open_relative_parent(
        context,
        normalized,
        create_parents=False,
    )
    try:
        if (
            current.leaf != held.leaf
            or _stable_directory_identity(os.fstat(current.descriptor))
            != _stable_directory_identity(os.fstat(held.descriptor))
            or _descriptor_mount_id(current.descriptor) != held.root.mount_id
            or _descriptor_mount_id(held.descriptor) != held.root.mount_id
        ):
            raise MutationHostQualificationError(
                "delete named parent chain no longer reaches held authority"
            )
    finally:
        current.close()
    held.root.revalidate(require_named=True)


def _read_bound_bytes(
    context: MutationContextSpec,
    normalized: str,
) -> tuple[bytes, os.stat_result]:
    parent = _open_relative_parent(
        context,
        normalized,
        create_parents=False,
    )
    descriptor: int | None = None
    try:
        before = os.stat(
            parent.leaf,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise MutationHostQualificationError(
                "mutation read target is not one regular file"
            )
        try:
            descriptor = os.open(
                parent.leaf,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent.descriptor,
            )
        except BaseException as exc:
            raise _ChildAcquisitionUncertain(
                "open",
                parent.root.path,
                normalized,
            ) from exc
        if _stat_identity(os.fstat(descriptor)) != _stat_identity(before):
            raise MutationHostQualificationError(
                "mutation read target changed while opening"
            )
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(_SOURCE_HASH_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise MutationHostQualificationError(
                    "mutation read target ended before its exact size"
                )
            chunks.append(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, before.st_size) or _stat_identity(
            os.fstat(descriptor)
        ) != _stat_identity(before):
            raise MutationHostQualificationError(
                "mutation read target changed while reading"
            )
        return b"".join(chunks), before
    finally:
        close_failure: BaseException | None = None
        if descriptor is not None:
            acquired = descriptor
            descriptor = None
            try:
                os.close(acquired)
            except OSError as exc:
                close_failure = MutationHostLifecycleError(
                    "mutation file descriptor cleanup failed"
                )
                close_failure.__cause__ = exc
        try:
            parent.close()
        except MutationHostLifecycleError as exc:
            if close_failure is None:
                close_failure = exc
        if close_failure is not None:
            raise close_failure


def _resolve_for_read(context: MutationContextSpec, relative: str) -> str | None:
    return _context_relative(context, relative)


def _resolve_for_write(context: MutationContextSpec, relative: str) -> str | None:
    return _write_is_authorized(context, relative)


def _exact_declared_delete_target(
    context: MutationContextSpec,
    relative: str,
) -> str | None:
    """Authorize only one explicitly named, fully deletable scope root."""

    normalized = _context_relative(context, relative)
    if normalized is None:
        return None
    declared = tuple(
        _context_relative(context, rendered)
        for rendered in context.write_scope
    )
    if any(item is None for item in declared) or normalized not in declared:
        return None
    exact = tuple(
        _context_relative(context, rendered)
        for rendered in context.write_exact_scope
    )
    if any(item is None for item in exact) or (exact and normalized not in exact):
        return None
    denied = tuple(
        _context_relative(context, rendered)
        for rendered in context.write_deny_scope
    )
    if any(item is None for item in denied) or any(
        item == normalized or item.startswith(normalized + "/")
        for item in denied
        if item is not None
    ):
        return None
    return normalized


def _validate_publication_token(token: str) -> str:
    if (
        not isinstance(token, str)
        or len(token) != _PUBLICATION_TOKEN_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in token)
    ):
        raise ValueError("invalid atomic-publication token")
    return token


def _same_delete_entry(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    """Return whether two no-follow observations bind the same entry."""

    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and (stat.S_ISDIR(left.st_mode) or left.st_nlink == right.st_nlink)
    )


def _require_delete_mount(
    descriptor: int,
    expected_mount_id: int,
    *,
    label: str,
) -> None:
    if _descriptor_mount_id(descriptor) != expected_mount_id:
        raise ValueError(f"delete target crosses a mount boundary: {label}")


def _rename_noreplace_at(
    source_descriptor: int,
    source: str,
    destination_descriptor: int,
    destination: str,
) -> None:
    """Atomically move one descriptor-relative entry without replacement."""

    for name in (source, destination):
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
        source_descriptor,
        os.fsencode(source),
        destination_descriptor,
        os.fsencode(destination),
        0x1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == getattr(os, "EEXIST", 17):
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, f"renameat2: {os.strerror(error)}")


def _open_delete_directory_at(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    expected_mount_id: int,
) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    try:
        actual = os.fstat(descriptor)
        if not stat.S_ISDIR(actual.st_mode) or not _same_delete_entry(
            expected,
            actual,
        ):
            raise ValueError(f"delete directory changed while opening: {name}")
        _require_delete_mount(
            descriptor,
            expected_mount_id,
            label=name,
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_delete_regular_at(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    expected_mount_id: int,
) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=parent_descriptor,
    )
    try:
        actual = os.fstat(descriptor)
        if (
            not stat.S_ISREG(actual.st_mode)
            or actual.st_nlink != 1
            or not _same_delete_entry(expected, actual)
        ):
            raise ValueError(f"delete file changed while opening: {name}")
        _require_delete_mount(
            descriptor,
            expected_mount_id,
            label=name,
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _delete_directory_chain_identity(descriptor: int) -> tuple[int, ...]:
    """Return one rename-stable directory identity plus its mount ID."""

    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        raise MutationHostQualificationError(
            "delete parent-chain authority is not a directory"
        )
    return (*_stable_directory_identity(observed), _descriptor_mount_id(descriptor))


def _open_delete_parent_chain(
    root_descriptor: int,
    components: tuple[str, ...],
    *,
    expected_mount_id: int,
) -> tuple[int, tuple[tuple[int, ...], ...]]:
    """Open a chain with detach-before-close ownership for every known fd.

    The caller must register an acquisition-in-progress owner before entry.
    An exception from ``dup``/``open`` can then be treated as an unknown
    kernel effect, while every descriptor whose integer became visible here
    is removed from the local inventory before its sole close attempt.
    """

    owned: list[int] = []
    chain: list[tuple[int, ...]] = []
    try:
        descriptor = os.dup(root_descriptor)
        owned.append(descriptor)
        root_identity = _delete_directory_chain_identity(descriptor)
        if root_identity[-1] != expected_mount_id:
            raise MutationHostQualificationError(
                "delete parent-chain root crossed a mount boundary"
            )
        chain.append(root_identity)
        for component in components:
            if component in {"", ".", ".."} or "/" in component:
                raise MutationHostQualificationError(
                    "delete parent-chain component is unsafe"
                )
            before = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(before.st_mode):
                raise MutationHostQualificationError(
                    "delete parent-chain component is not a directory"
                )
            child = os.open(
                component,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            owned.append(child)
            try:
                opened = os.fstat(child)
                after = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                identity = _delete_directory_chain_identity(child)
                if (
                    _stable_directory_identity(before)
                    != _stable_directory_identity(opened)
                    or _stable_directory_identity(after)
                    != _stable_directory_identity(opened)
                    or identity[-1] != expected_mount_id
                ):
                    raise MutationHostQualificationError(
                        "delete parent chain changed or crossed a mount"
                    )
            except BaseException:
                raise
            owned.remove(descriptor)
            os.close(descriptor)
            descriptor = child
            chain.append(identity)
        owned.remove(descriptor)
        return descriptor, tuple(chain)
    except BaseException as primary:
        cleanup_failures: list[BaseException] = []
        while owned:
            descriptor = owned.pop()
            try:
                os.close(descriptor)
            except BaseException as exc:
                cleanup_failures.append(exc)
        if cleanup_failures:
            error = MutationHostLifecycleError(
                "delete parent-chain descriptor cleanup is indeterminate"
            )
            for failure in cleanup_failures[1:]:
                error.add_note(
                    "additional descriptor close failure: "
                    f"{type(failure).__name__}: {failure}"
                )
            raise error from primary
        raise


def _delete_entry_metadata(observed: os.stat_result) -> tuple[int, ...]:
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
    """Return root metadata whose value is stable across quarantine rename."""

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


@dataclass(frozen=True)
class _DeleteTreeEntry:
    name: str
    observed: os.stat_result
    children: tuple[_DeleteTreeEntry, ...] | None
    metadata: tuple[int, ...]
    digest: str | None = None
    symlink_target: str | None = None


def _digest_delete_regular(
    descriptor: int,
    expected: os.stat_result,
    *,
    label: str,
) -> str:
    before = os.fstat(descriptor)
    if not _same_delete_entry(expected, before):
        raise ValueError(f"delete file changed while hashing: {label}")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(
            descriptor,
            min(_SOURCE_HASH_CHUNK_BYTES, before.st_size - offset),
            offset,
        )
        if not chunk:
            raise ValueError(f"delete file changed while hashing: {label}")
        offset += len(chunk)
        if offset > _MAX_DELETE_REGULAR_BYTES:
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
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    expected_mount_id: int,
) -> str:
    descriptor = _open_delete_regular_at(
        parent_descriptor,
        name,
        expected,
        expected_mount_id=expected_mount_id,
    )
    try:
        return _digest_delete_regular(descriptor, expected, label=name)
    finally:
        os.close(descriptor)


def _bounded_delete_names(
    directory_descriptor: int,
    *,
    maximum: int,
    overflow_message: str,
    shared_count: list[int] | None = None,
) -> tuple[str, ...]:
    if maximum < 0 or (
        shared_count is not None
        and (len(shared_count) != 1 or shared_count[0] < 0)
    ):
        raise ValueError(overflow_message)
    names: list[str] = []
    with os.scandir(directory_descriptor) as entries:
        for entry in entries:
            names.append(entry.name)
            observed_count = len(names)
            if shared_count is not None:
                shared_count[0] += 1
                observed_count = shared_count[0]
            if observed_count > maximum:
                raise ValueError(overflow_message)
    return tuple(sorted(names))


def _delete_directory_is_empty(directory_descriptor: int) -> bool:
    with os.scandir(directory_descriptor) as entries:
        return next(iter(entries), None) is None


def _validate_delete_tree(
    directory_descriptor: int,
    *,
    root_device: int,
    root_mount_id: int,
    depth: int,
    count: list[int],
    regular_bytes: list[int],
) -> tuple[tuple[_DeleteTreeEntry, ...], int, int]:
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
    names = _bounded_delete_names(
        directory_descriptor,
        maximum=_MAX_DELETE_ENTRIES,
        overflow_message=entry_limit_message,
        shared_count=count,
    )
    for name in names:
        if name.casefold() in _VCS_METADATA_COMPONENTS:
            raise ValueError(
                f"recursive delete contains protected VCS metadata: {name}"
            )
        observed = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(observed.st_mode):
            child_descriptor = _open_delete_directory_at(
                directory_descriptor,
                name,
                observed,
                expected_mount_id=root_mount_id,
            )
            try:
                if os.fstat(child_descriptor).st_dev != root_device:
                    raise ValueError(
                        f"recursive delete crosses a filesystem boundary: {name}"
                    )
                children, child_files, child_directories = _validate_delete_tree(
                    child_descriptor,
                    root_device=root_device,
                    root_mount_id=root_mount_id,
                    depth=depth + 1,
                    count=count,
                    regular_bytes=regular_bytes,
                )
            finally:
                os.close(child_descriptor)
            files += child_files
            directories += child_directories + 1
            manifest.append(
                _DeleteTreeEntry(
                    name,
                    observed,
                    children,
                    _delete_entry_metadata(observed),
                )
            )
            continue
        if stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            if observed.st_nlink != 1:
                raise ValueError(
                    "recursive delete contains a multiply-linked entry: "
                    f"{name}"
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
                    directory_descriptor,
                    name,
                    observed,
                    expected_mount_id=root_mount_id,
                )
            else:
                symlink_target = os.readlink(
                    name,
                    dir_fd=directory_descriptor,
                )
            files += 1
            manifest.append(
                _DeleteTreeEntry(
                    name,
                    observed,
                    None,
                    _delete_entry_metadata(observed),
                    digest,
                    symlink_target,
                )
            )
            continue
        raise ValueError(
            f"recursive delete contains unsupported filesystem entry: {name}"
        )
    return tuple(manifest), files, directories


def _revalidate_delete_tree(
    directory_descriptor: int,
    *,
    manifest: tuple[_DeleteTreeEntry, ...],
    root_device: int,
    root_mount_id: int,
) -> None:
    expected_names = tuple(entry.name for entry in manifest)
    current_names = _bounded_delete_names(
        directory_descriptor,
        maximum=len(expected_names),
        overflow_message="delete tree changed after validation",
    )
    if current_names != expected_names:
        raise ValueError("delete tree changed after validation")
    for entry in manifest:
        observed = os.stat(
            entry.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if _delete_entry_metadata(observed) != entry.metadata:
            raise ValueError(
                "delete target metadata changed after validation: "
                f"{entry.name}"
            )
        if stat.S_ISDIR(observed.st_mode):
            child_descriptor = _open_delete_directory_at(
                directory_descriptor,
                entry.name,
                entry.observed,
                expected_mount_id=root_mount_id,
            )
            try:
                if os.fstat(child_descriptor).st_dev != root_device:
                    raise ValueError(
                        "recursive delete crosses a filesystem boundary: "
                        f"{entry.name}"
                    )
                _revalidate_delete_tree(
                    child_descriptor,
                    manifest=entry.children or (),
                    root_device=root_device,
                    root_mount_id=root_mount_id,
                )
            finally:
                os.close(child_descriptor)
            continue
        if stat.S_ISREG(observed.st_mode):
            digest = _snapshot_delete_regular_at(
                directory_descriptor,
                entry.name,
                entry.observed,
                expected_mount_id=root_mount_id,
            )
            if digest != entry.digest:
                raise ValueError(
                    "delete file content changed after validation: "
                    f"{entry.name}"
                )
            continue
        if (
            os.readlink(entry.name, dir_fd=directory_descriptor)
            != entry.symlink_target
        ):
            raise ValueError(
                "delete symlink target changed after validation: "
                f"{entry.name}"
            )


def _delete_tree_contents(
    directory_descriptor: int,
    *,
    manifest: tuple[_DeleteTreeEntry, ...],
    root_device: int,
    root_mount_id: int,
) -> tuple[int, int]:
    files = 0
    directories = 0
    expected_names = tuple(entry.name for entry in manifest)
    current_names = _bounded_delete_names(
        directory_descriptor,
        maximum=len(expected_names),
        overflow_message="delete tree changed after validation",
    )
    if current_names != expected_names:
        raise ValueError("delete tree changed after validation")
    for entry in manifest:
        name = entry.name
        observed = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if _delete_entry_metadata(observed) != entry.metadata:
            raise ValueError(f"delete target changed during traversal: {name}")
        if stat.S_ISDIR(observed.st_mode):
            child_descriptor = _open_delete_directory_at(
                directory_descriptor,
                name,
                entry.observed,
                expected_mount_id=root_mount_id,
            )
            try:
                if os.fstat(child_descriptor).st_dev != root_device:
                    raise ValueError(
                        f"recursive delete crosses a filesystem boundary: {name}"
                    )
                child_files, child_directories = _delete_tree_contents(
                    child_descriptor,
                    manifest=entry.children or (),
                    root_device=root_device,
                    root_mount_id=root_mount_id,
                )
                os.fsync(child_descriptor)
            finally:
                os.close(child_descriptor)
            current = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if not _same_delete_entry(entry.observed, current):
                raise ValueError(
                    f"delete target changed during traversal: {name}"
                )
            probe_descriptor = _open_delete_directory_at(
                directory_descriptor,
                name,
                current,
                expected_mount_id=root_mount_id,
            )
            os.close(probe_descriptor)
            os.rmdir(name, dir_fd=directory_descriptor)
            files += child_files
            directories += child_directories + 1
            continue
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not _same_delete_entry(entry.observed, current):
            raise ValueError(f"delete target changed during traversal: {name}")
        if stat.S_ISREG(observed.st_mode):
            descriptor = _open_delete_regular_at(
                directory_descriptor,
                name,
                current,
                expected_mount_id=root_mount_id,
            )
            os.close(descriptor)
        os.unlink(name, dir_fd=directory_descriptor)
        files += 1
    os.fsync(directory_descriptor)
    return files, directories


def _open_precreated_delete_quarantine(
    parent_descriptor: int,
    quarantine_name: str,
    *,
    expected_mount_id: int,
) -> tuple[int, os.stat_result]:
    descriptor = os.open(
        quarantine_name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    try:
        held = os.fstat(descriptor)
        named = os.stat(
            quarantine_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(held.st_mode)
            or not _same_delete_entry(held, named)
            or stat.S_IMODE(held.st_mode) != 0o700
            or held.st_uid != os.geteuid()
            or not _delete_directory_is_empty(descriptor)
        ):
            raise ValueError("precreated delete quarantine binding is unsafe")
        _require_delete_mount(
            descriptor,
            expected_mount_id,
            label=quarantine_name,
        )
        return descriptor, held
    except BaseException:
        os.close(descriptor)
        raise


def _discard_empty_delete_quarantine(
    parent_descriptor: int,
    quarantine_descriptor: int,
    quarantine_name: str,
    expected: os.stat_result,
) -> None:
    held = os.fstat(quarantine_descriptor)
    named = os.stat(
        quarantine_name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not _same_delete_entry(expected, held)
        or not _same_delete_entry(expected, named)
        or not _delete_directory_is_empty(quarantine_descriptor)
    ):
        raise ToolMutationIndeterminate(
            "delete quarantine changed before empty-directory settlement"
        )
    os.rmdir(quarantine_name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


def _restore_quarantined_delete(
    parent_descriptor: int,
    leaf: str,
    quarantine_descriptor: int,
    quarantine_name: str,
    quarantine_expected: os.stat_result,
    payload_expected: os.stat_result,
) -> None:
    try:
        payload = os.stat(
            _DELETE_QUARANTINE_PAYLOAD,
            dir_fd=quarantine_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise ToolMutationIndeterminate(
            "delete quarantine payload disappeared before restoration"
        ) from exc
    if not _same_delete_entry(payload_expected, payload):
        raise ToolMutationIndeterminate(
            "delete quarantine payload changed before restoration"
        )
    try:
        _rename_noreplace_at(
            quarantine_descriptor,
            _DELETE_QUARANTINE_PAYLOAD,
            parent_descriptor,
            leaf,
        )
    except FileExistsError as exc:
        raise ToolMutationIndeterminate(
            "delete target was recreated while its quarantined payload "
            "remained retained; both entries were preserved"
        ) from exc
    restored = os.stat(
        leaf,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if not _same_delete_entry(payload_expected, restored):
        raise ToolMutationIndeterminate(
            "restored delete target binding could not be proved"
        )
    os.fsync(quarantine_descriptor)
    os.fsync(parent_descriptor)
    _discard_empty_delete_quarantine(
        parent_descriptor,
        quarantine_descriptor,
        quarantine_name,
        quarantine_expected,
    )


def _require_public_delete_name_absent(
    parent_descriptor: int,
    leaf: str,
) -> None:
    try:
        os.stat(leaf, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ToolMutationIndeterminate(
        "delete target was recreated after quarantine; public and private "
        "entries were preserved"
    )


def _delete_source_from_spec(
    context: MutationContextSpec,
    *,
    file_path: str,
    recursive: bool,
    publication_token: str,
    publication_barrier: Callable[[], None],
    quarantine_barrier: Callable[[], None],
) -> str:
    """Delete one exact scope root through the child's retained repo fd.

    The parent first arms a durable receipt and creates/binds the empty
    quarantine. The child then atomically parks the exact validated payload,
    crosses the second parent barrier, re-proves its complete contents, and
    only then begins unlinking.
    """

    if not isinstance(recursive, bool):
        return "Error: recursive must be a boolean"
    normalized = _exact_declared_delete_target(context, file_path)
    if normalized is None:
        return (
            "Error: delete target must exactly match an explicit, permitted "
            f"write_scope entry: '{file_path}'"
        )
    try:
        token = _validate_publication_token(publication_token)
    except ValueError as exc:
        return f"Error deleting {file_path}: {exc}"

    parent: _HeldRelativeParent | None = None
    target_descriptor = -1
    quarantine_descriptor = -1
    quarantine_expected: os.stat_result | None = None
    payload_expected: os.stat_result | None = None
    quarantine_opened = False
    payload_moved = False
    destructive_started = False
    try:
        parent = _open_relative_parent(
            context,
            normalized,
            create_parents=False,
        )
        leaf = parent.leaf
        observed = os.stat(
            leaf,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError("final delete target cannot be a symlink")
        root_metadata = _delete_root_metadata(observed)
        root_mount_id = parent.root.mount_id
        _require_delete_mount(
            parent.descriptor,
            root_mount_id,
            label=parent.relative_parent,
        )
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
            if observed.st_dev != os.fstat(parent.descriptor).st_dev:
                raise ValueError(
                    "recursive delete root crosses a filesystem boundary"
                )
            target_descriptor = _open_delete_directory_at(
                parent.descriptor,
                leaf,
                observed,
                expected_mount_id=root_mount_id,
            )
            root_device = os.fstat(target_descriptor).st_dev
            manifest, files, directories = _validate_delete_tree(
                target_descriptor,
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
            target_descriptor = _open_delete_regular_at(
                parent.descriptor,
                leaf,
                observed,
                expected_mount_id=root_mount_id,
            )
            root_device = os.fstat(target_descriptor).st_dev
            root_digest = _digest_delete_regular(
                target_descriptor,
                observed,
                label=leaf,
            )
        else:
            raise ValueError("delete target must be a regular file or directory")

        # Barrier one: the parent has already armed the durable receipt. It now
        # creates, authenticates, and durably binds the empty qdir plus exact
        # child identity before allowing this public-name mutation.
        publication_barrier()
        _reprove_child_named_parent(context, normalized, parent)
        quarantine_name = (
            f".{leaf}.crucible-delete-{token}.quarantine"
        )
        quarantine_descriptor, quarantine_expected = (
            _open_precreated_delete_quarantine(
                parent.descriptor,
                quarantine_name,
                expected_mount_id=root_mount_id,
            )
        )
        quarantine_opened = True
        try:
            _rename_noreplace_at(
                parent.descriptor,
                leaf,
                quarantine_descriptor,
                _DELETE_QUARANTINE_PAYLOAD,
            )
            payload_moved = True
            os.fsync(quarantine_descriptor)
            os.fsync(parent.descriptor)
            moved = os.stat(
                _DELETE_QUARANTINE_PAYLOAD,
                dir_fd=quarantine_descriptor,
                follow_symlinks=False,
            )
            payload_expected = moved

            # Barrier two: the parent must still own the durable, child-bound
            # receipt after the rename. No unlink occurs before its ACK.
            quarantine_barrier()
            _reprove_child_named_parent(context, normalized, parent)
            if (
                not _same_delete_entry(observed, moved)
                or not _same_delete_entry(os.fstat(target_descriptor), moved)
                or _delete_root_metadata(moved) != root_metadata
            ):
                raise ValueError(
                    "delete target changed before atomic quarantine"
                )
            _require_public_delete_name_absent(parent.descriptor, leaf)
            if stat.S_ISDIR(moved.st_mode):
                _revalidate_delete_tree(
                    target_descriptor,
                    manifest=manifest,
                    root_device=root_device,
                    root_mount_id=root_mount_id,
                )
            else:
                moved_digest = _digest_delete_regular(
                    target_descriptor,
                    moved,
                    label=leaf,
                )
                if moved_digest != root_digest:
                    raise ValueError(
                        "delete file content changed before quarantine proof"
                    )
            _require_public_delete_name_absent(parent.descriptor, leaf)

            destructive_started = True
            if stat.S_ISDIR(moved.st_mode):
                deleted_files, deleted_directories = _delete_tree_contents(
                    target_descriptor,
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
                    dir_fd=quarantine_descriptor,
                )
                result = (
                    f"OK: recursively deleted {deleted_files} files and "
                    f"{deleted_directories + 1} directories from {file_path}"
                )
            else:
                os.unlink(
                    _DELETE_QUARANTINE_PAYLOAD,
                    dir_fd=quarantine_descriptor,
                )
                result = f"OK: deleted file {file_path}"
            payload_moved = False
            os.fsync(quarantine_descriptor)
            _require_public_delete_name_absent(parent.descriptor, leaf)
            _discard_empty_delete_quarantine(
                parent.descriptor,
                quarantine_descriptor,
                quarantine_name,
                quarantine_expected,
            )
            quarantine_opened = False
            return result
        except BaseException as exc:
            if payload_moved:
                if payload_expected is None or quarantine_expected is None:
                    raise ToolMutationIndeterminate(
                        "delete quarantine payload identity could not be "
                        "bound for restoration"
                    ) from exc
                try:
                    _restore_quarantined_delete(
                        parent.descriptor,
                        leaf,
                        quarantine_descriptor,
                        quarantine_name,
                        quarantine_expected,
                        payload_expected,
                    )
                except ToolMutationIndeterminate:
                    raise
                except BaseException as restore_exc:
                    raise ToolMutationIndeterminate(
                        "quarantined delete could not be restored exactly"
                    ) from restore_exc
                payload_moved = False
                quarantine_opened = False
                if destructive_started or not isinstance(
                    exc,
                    (ValueError, MutationHostQualificationError),
                ):
                    raise ToolMutationIndeterminate(
                        "delete failed after namespace mutation; host "
                        "settlement and journal rollback are required"
                    ) from exc
                return f"Error deleting {file_path}: {exc}"
            if quarantine_opened:
                if quarantine_expected is None:
                    raise ToolMutationIndeterminate(
                        "delete quarantine identity could not be bound for "
                        "empty-directory settlement"
                    ) from exc
                try:
                    _discard_empty_delete_quarantine(
                        parent.descriptor,
                        quarantine_descriptor,
                        quarantine_name,
                        quarantine_expected,
                    )
                except BaseException as cleanup_exc:
                    raise ToolMutationIndeterminate(
                        "empty delete quarantine could not be settled"
                    ) from cleanup_exc
                quarantine_opened = False
            if destructive_started:
                raise ToolMutationIndeterminate(
                    "delete failed after payload removal; host settlement "
                    "and journal rollback are required"
                ) from exc
            raise
    except ToolMutationIndeterminate:
        raise
    except FileNotFoundError:
        return f"Error: delete target not found: {file_path}"
    except (OSError, ValueError, MutationHostQualificationError) as exc:
        return f"Error deleting {file_path}: {exc}"
    finally:
        close_failure: BaseException | None = None
        for descriptor in (quarantine_descriptor, target_descriptor):
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except OSError as exc:
                close_failure = close_failure or MutationHostLifecycleError(
                    "delete descriptor cleanup failed"
                )
                close_failure.__cause__ = exc
        if parent is not None:
            try:
                parent.close()
            except MutationHostLifecycleError as exc:
                close_failure = close_failure or exc
        if close_failure is not None:
            raise close_failure


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write during atomic text publication")
        view = view[written:]


def _publish_text_atomic(
    context: MutationContextSpec,
    normalized: str,
    content: str,
    *,
    expected_current_sha256: str | None,
    publication_token: str,
    publication_barrier: Callable[[str, str | None], None],
) -> None:
    del context, normalized
    _validate_publication_token(publication_token)
    # The child is computation-only.  It never creates an ancestor, opens a
    # write descriptor, or publishes a name.  The trusted parent owns the
    # complete transaction and acknowledges only after exact publication.
    publication_barrier(content, expected_current_sha256)


def _write_source_from_spec(
    context: MutationContextSpec,
    *,
    file_path: str,
    content: str,
    expected_sha256: str,
    publication_token: str,
    publication_barrier: Callable[[str, str | None], None],
) -> str:
    normalized = _resolve_for_write(context, file_path)
    if normalized is None:
        return f"Error: cannot resolve write path for '{file_path}'"
    observed_sha256: str | None = None
    try:
        original_bytes, _ = _read_bound_bytes(context, normalized)
    except FileNotFoundError:
        original_bytes = None
    if original_bytes is not None:
        try:
            original = original_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
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
            context,
            normalized,
            content,
            expected_current_sha256=observed_sha256,
            publication_token=publication_token,
            publication_barrier=publication_barrier,
        )
    except (OSError, ValueError) as exc:
        return f"Error writing {file_path}: {exc}"
    n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"OK: wrote {n_lines} lines to {file_path}"


def _apply_patch_from_spec(
    context: MutationContextSpec,
    *,
    file_path: str,
    before: str,
    after: str,
    publication_token: str,
    publication_barrier: Callable[[str, str | None], None],
) -> str:
    if before == after:
        return (
            f"Error: replacement for {file_path} is identical to the "
            "matched text; apply_patch requires an actual edit."
        )
    normalized_read = _resolve_for_read(context, file_path)
    if normalized_read is None:
        return f"Error: file not found: {file_path}"
    try:
        original_bytes, _ = _read_bound_bytes(context, normalized_read)
        original = original_bytes.decode("utf-8")
    except FileNotFoundError:
        return f"Error: file not found: {file_path}"
    except (OSError, UnicodeDecodeError) as exc:
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
    normalized_write = _resolve_for_write(context, file_path)
    if normalized_write is None:
        return f"Error: cannot resolve write path for '{file_path}'"
    expected_current_sha256 = hashlib.sha256(original.encode("utf-8")).hexdigest()
    if normalized_write != normalized_read:
        try:
            original_wt_bytes, _ = _read_bound_bytes(context, normalized_write)
            original_wt = original_wt_bytes.decode("utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            return f"Error: file not found: {file_path}"
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
        expected_current_sha256 = hashlib.sha256(original_wt_bytes).hexdigest()
    try:
        _publish_text_atomic(
            context,
            normalized_write,
            patched,
            expected_current_sha256=expected_current_sha256,
            publication_token=publication_token,
            publication_barrier=publication_barrier,
        )
    except (OSError, ValueError) as exc:
        return f"Error writing patched {file_path}: {exc}"
    before_lines = before.count("\n") + 1
    after_lines = after.count("\n") + 1
    return f"OK: replaced {before_lines} lines with {after_lines} lines in {file_path}"


def _encode_frame(payload: Mapping[str, Any]) -> bytes:
    try:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MutationHostProtocolError(
            "mutation-host frame is not JSON serializable"
        ) from exc
    if len(encoded) > _MAX_FRAME_BYTES:
        raise MutationHostProtocolError(
            f"mutation-host frame exceeds {_MAX_FRAME_BYTES} bytes"
        )
    return encoded


def _decode_frame(encoded: bytes) -> dict[str, Any]:
    if not encoded or len(encoded) > _MAX_FRAME_BYTES:
        raise MutationHostProtocolError(
            "mutation-host child emitted an empty or oversized frame"
        )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MutationHostProtocolError(
            "mutation-host child emitted invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise MutationHostProtocolError("mutation-host child frame must be an object")
    return payload


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise MutationHostTimeoutError(
            "logical invocation deadline expired in isolated mutation host"
        )
    return value


def _read_kernel_identity(path: str, *, label: str) -> bytes:
    """Read one bounded kernel-owned procfs identity without aliases."""

    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
            | os.O_NONBLOCK,
        )
        payload = bytearray()
        while True:
            chunk = os.read(
                descriptor,
                min(1024, _MAX_PROC_IDENTITY_BYTES + 1 - len(payload)),
            )
            if not chunk:
                return bytes(payload)
            payload.extend(chunk)
            if len(payload) > _MAX_PROC_IDENTITY_BYTES:
                raise MutationHostLifecycleError(
                    f"{label} exceeds its byte ceiling"
                )
    except MutationHostLifecycleError:
        raise
    except OSError as exc:
        raise MutationHostLifecycleError(
            f"cannot read {label}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _child_start_ticks(pid: int) -> int:
    """Return the exact Linux process start tick for one live child."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise MutationHostLifecycleError(
            "isolated mutation child identity is malformed"
        )
    try:
        rendered = _read_kernel_identity(
            f"/proc/{pid}/stat",
            label="isolated mutation child process identity",
        ).decode("ascii")
        closing = rendered.rfind(") ")
        if closing < 0:
            raise ValueError
        fields = rendered[closing + 2:].split()
        state = fields[0]
        start_ticks = int(fields[19])
    except (IndexError, UnicodeError, ValueError) as exc:
        raise MutationHostLifecycleError(
            "isolated mutation child identity is malformed"
        ) from exc
    if len(state) != 1 or state in {"X", "Z"} or start_ticks <= 0:
        raise MutationHostLifecycleError(
            "isolated mutation child is not live"
        )
    return start_ticks


def _kernel_boot_id() -> str:
    """Return the canonical UUID-shaped Linux boot identity."""

    try:
        rendered = _read_kernel_identity(
            "/proc/sys/kernel/random/boot_id",
            label="kernel boot identity",
        ).decode("ascii").strip()
    except UnicodeError as exc:
        raise MutationHostLifecycleError(
            "kernel boot identity is malformed"
        ) from exc
    parts = rendered.split("-")
    if (
        tuple(map(len, parts)) != (8, 4, 4, 4, 12)
        or any(
            character not in "0123456789abcdef"
            for character in rendered.replace("-", "")
        )
    ):
        raise MutationHostLifecycleError(
            "kernel boot identity is malformed"
        )
    return rendered


class _PosixSpawnProcess:
    """Exact provisional ownership of a child and every parent-side pipe.

    The host attaches this object before pipe creation. ``posix_spawn`` writes
    its PID directly into the already-attached object before any parent close
    can fail. Partial setup is therefore ordinary retained lifecycle
    authority, not an unowned-child window. Parent communication remains on
    raw, nonblocking descriptors; ownership never crosses ``fdopen``.
    """

    _RAW_DESCRIPTOR_ATTRIBUTES = (
        "_request_read_descriptor",
        "_request_write_descriptor",
        "_response_read_descriptor",
        "_response_write_descriptor",
    )

    def __init__(self) -> None:
        # libc writes directly into this already-attached cell.  If an async
        # Python exception is delivered after the kernel spawn effect but
        # before the wrapper returns, PID authority is still observable here.
        self._spawn_pid_cell = ctypes.c_int(0)
        self._request_read_descriptor: int | None = None
        self._request_write_descriptor: int | None = None
        self._response_read_descriptor: int | None = None
        self._response_write_descriptor: int | None = None
        self.returncode: int | None = None
        self._wait_lock = threading.Lock()
        self._descriptor_lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        observed = int(self._spawn_pid_cell.value)
        return observed if observed > 0 else None

    def acquire_pipe_pair(self, first_attribute: str, second_attribute: str) -> None:
        """Acquire pipe descriptors through an already-attached result cell."""

        if (
            first_attribute not in self._RAW_DESCRIPTOR_ATTRIBUTES
            or second_attribute not in self._RAW_DESCRIPTOR_ATTRIBUTES
            or getattr(self, first_attribute) is not None
            or getattr(self, second_attribute) is not None
        ):
            raise MutationHostLifecycleError(
                "isolated mutation pipe acquisition authority is unsafe"
            )
        cell = (ctypes.c_int * 2)(-1, -1)
        primary_error: BaseException | None = None
        result: int | None = None
        try:
            result = _call_libc_pipe2(cell)
        except BaseException as exc:
            primary_error = exc
        # Publish both result-cell values before propagating any exception.
        # This is the exact ownership handoff for effect-then-error injection.
        first = int(cell[0])
        second = int(cell[1])
        if first >= 0:
            setattr(self, first_attribute, first)
        if second >= 0:
            setattr(self, second_attribute, second)
        if primary_error is not None:
            raise primary_error
        if result != 0 or first < 0 or second < 0 or first == second:
            raise MutationHostLifecycleError(
                "isolated mutation pipe acquisition failed closed"
            )

    @property
    def raw_descriptors(self) -> tuple[int, ...]:
        """Return the exact raw descriptors still owned by this process."""

        with self._descriptor_lock:
            return tuple(
                descriptor
                for attribute in self._RAW_DESCRIPTOR_ATTRIBUTES
                if (descriptor := getattr(self, attribute)) is not None
            )

    def prepare_parent_pipes(self) -> None:
        """Close child-only ends and make exact parent ends nonblocking."""

        failures: list[OSError] = []
        for attribute in (
            "_request_read_descriptor",
            "_response_write_descriptor",
        ):
            try:
                self._close_raw_descriptor(attribute)
            except OSError as exc:
                failures.append(exc)
        for attribute in (
            "_request_write_descriptor",
            "_response_read_descriptor",
        ):
            with self._descriptor_lock:
                descriptor = getattr(self, attribute)
                if descriptor is None:
                    raise MutationHostLifecycleError(
                        "isolated mutation parent pipe authority is missing"
                    )
                try:
                    os.set_blocking(descriptor, False)
                except OSError as exc:
                    failures.append(exc)
        if failures:
            raise MutationHostLifecycleError(
                "isolated mutation parent-pipe preparation failed"
            ) from failures[0]

    def _close_raw_descriptor(self, attribute: str) -> None:
        with self._descriptor_lock:
            descriptor = getattr(self, attribute)
            if descriptor is None:
                return
            # Atomically detach before the attempt. On Linux, EINTR and other
            # late errors can be reported after the integer was released and
            # reused. The same lock prevents in-flight I/O from retaining a
            # stale copy while this ownership transition occurs.
            setattr(self, attribute, None)
            os.close(descriptor)

    def close_parent_resources(self) -> None:
        """Consume each raw pipe descriptor in exactly one close attempt."""

        failures: list[OSError] = []
        for attribute in self._RAW_DESCRIPTOR_ATTRIBUTES:
            try:
                self._close_raw_descriptor(attribute)
            except OSError as exc:
                failures.append(exc)
        if failures:
            raise MutationHostLifecycleError(
                "isolated mutation pipe cleanup failed"
            ) from failures[0]

    def poll(self) -> int | None:
        with self._wait_lock:
            if self.returncode is not None:
                return self.returncode
            if self.pid is None:
                return None
            if self.pid <= 0:
                raise MutationHostLifecycleError(
                    "isolated mutation child PID authority is unsafe"
                )
            while True:
                try:
                    observed, status = os.waitpid(self.pid, os.WNOHANG)
                except InterruptedError:
                    # A signal handler that returns may expose EINTR instead
                    # of Python's usual transparent syscall restart.  Process
                    # death has not been observed yet, so keep the exact PID
                    # wait authority and retry rather than converting the
                    # interruption into false lifecycle evidence.
                    continue
                break
            if observed == 0:
                return None
            self.returncode = os.waitstatus_to_exitcode(status)
            return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.poll()
            if result is not None:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                assert timeout is not None
                raise subprocess.TimeoutExpired(
                    cmd="crucible isolated mutation host",
                    timeout=timeout,
                )
            time.sleep(0.005)

    def kill(self) -> None:
        if self.pid is None:
            return
        if self.pid <= 0:
            raise MutationHostLifecycleError(
                "isolated mutation child PID authority is unsafe"
            )
        with suppress(ProcessLookupError):
            os.kill(self.pid, signal.SIGKILL)


class _AlignedSpawnStorage(ctypes.Union):
    """Over-aligned storage for glibc's initialized opaque spawn records."""

    _fields_ = [
        ("alignment", ctypes.c_longdouble),
        ("payload", ctypes.c_ubyte * _SPAWN_STORAGE_BYTES),
    ]


class _AlignedSignalSet(ctypes.Union):
    """Over-aligned storage larger than Linux's opaque ``sigset_t``."""

    _fields_ = [
        ("alignment", ctypes.c_longdouble),
        ("payload", ctypes.c_ubyte * 256),
    ]


def _check_spawn_call(result: int, operation: str) -> None:
    if result:
        raise OSError(result, f"{operation}: {os.strerror(result)}")


def _call_libc_pipe2(result_cell: Any) -> int:
    """Write a CLOEXEC pipe pair into caller-owned result storage."""

    try:
        pipe2 = ctypes.CDLL(None, use_errno=True).pipe2
    except (AttributeError, OSError) as exc:
        raise MutationHostQualificationError(
            "atomic pipe2 acquisition is unavailable"
        ) from exc
    pipe2.restype = ctypes.c_int
    result = int(pipe2(result_cell, os.O_CLOEXEC))
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"pipe2: {os.strerror(error)}")
    return result


def _call_libc_posix_spawn(
    spawn: Any,
    spawned_pid: ctypes.c_int,
    executable: bytes,
    actions: _AlignedSpawnStorage,
    attributes: _AlignedSpawnStorage,
    argv: Any,
    envp: Any,
) -> int:
    """Return a spawned PID; split out to expose the live provisional window."""

    _check_spawn_call(
        spawn(
            ctypes.byref(spawned_pid),
            executable,
            ctypes.byref(actions),
            ctypes.byref(attributes),
            argv,
            envp,
        ),
        "posix_spawn",
    )
    return int(spawned_pid.value)


def _atomic_posix_spawn_closefrom(
    process: _PosixSpawnProcess,
    command: list[str],
    environment: dict[str, str],
    duplicate_actions: tuple[tuple[int, int], ...],
    *,
    closefrom: int,
) -> None:
    """Spawn with glibc's atomic close-from action.

    Python 3.12 does not expose ``POSIX_SPAWN_CLOSEFROM`` even when the host
    glibc supports ``posix_spawn_file_actions_addclosefrom_np``. Calling the
    same libc API through its stable initialized opaque records closes every
    ambient descriptor in the child-side spawn transaction. There is no
    inheritable-FD enumeration window for another thread to race.
    """

    if sys.platform != "linux":
        raise MutationHostQualificationError(
            "isolated mutation hosting requires Linux atomic close-from spawn"
        )
    try:
        libc = ctypes.CDLL(None)
        actions_init = libc.posix_spawn_file_actions_init
        actions_adddup2 = libc.posix_spawn_file_actions_adddup2
        actions_addclosefrom = libc.posix_spawn_file_actions_addclosefrom_np
        actions_destroy = libc.posix_spawn_file_actions_destroy
        attributes_init = libc.posix_spawnattr_init
        attributes_setflags = libc.posix_spawnattr_setflags
        attributes_setpgroup = libc.posix_spawnattr_setpgroup
        attributes_setsigmask = libc.posix_spawnattr_setsigmask
        attributes_setsigdefault = libc.posix_spawnattr_setsigdefault
        attributes_destroy = libc.posix_spawnattr_destroy
        sigemptyset = libc.sigemptyset
        sigaddset = libc.sigaddset
        spawn = libc.posix_spawn
    except (AttributeError, OSError) as exc:
        raise MutationHostQualificationError(
            "atomic POSIX close-from spawn is unavailable"
        ) from exc

    for function in (
        actions_init,
        actions_adddup2,
        actions_addclosefrom,
        actions_destroy,
        attributes_init,
        attributes_setflags,
        attributes_setpgroup,
        attributes_setsigmask,
        attributes_setsigdefault,
        attributes_destroy,
        sigemptyset,
        sigaddset,
        spawn,
    ):
        function.restype = ctypes.c_int

    actions = _AlignedSpawnStorage()
    attributes = _AlignedSpawnStorage()
    empty_signals = _AlignedSignalSet()
    default_signals = _AlignedSignalSet()
    actions_initialized = False
    attributes_initialized = False
    primary_error: BaseException | None = None
    try:
        _check_spawn_call(
            actions_init(ctypes.byref(actions)),
            "posix_spawn_file_actions_init",
        )
        actions_initialized = True
        for source, target in duplicate_actions:
            _check_spawn_call(
                actions_adddup2(ctypes.byref(actions), source, target),
                "posix_spawn_file_actions_adddup2",
            )
        _check_spawn_call(
            actions_addclosefrom(ctypes.byref(actions), closefrom),
            "posix_spawn_file_actions_addclosefrom_np",
        )

        _check_spawn_call(
            attributes_init(ctypes.byref(attributes)),
            "posix_spawnattr_init",
        )
        attributes_initialized = True
        _check_spawn_call(
            sigemptyset(ctypes.byref(empty_signals)),
            "sigemptyset",
        )
        _check_spawn_call(
            sigemptyset(ctypes.byref(default_signals)),
            "sigemptyset",
        )
        for signal_number in (signal.SIGINT, signal.SIGPIPE, signal.SIGTERM):
            _check_spawn_call(
                sigaddset(ctypes.byref(default_signals), signal_number),
                "sigaddset",
            )
        _check_spawn_call(
            attributes_setpgroup(ctypes.byref(attributes), 0),
            "posix_spawnattr_setpgroup",
        )
        _check_spawn_call(
            attributes_setsigmask(
                ctypes.byref(attributes),
                ctypes.byref(empty_signals),
            ),
            "posix_spawnattr_setsigmask",
        )
        _check_spawn_call(
            attributes_setsigdefault(
                ctypes.byref(attributes),
                ctypes.byref(default_signals),
            ),
            "posix_spawnattr_setsigdefault",
        )
        _check_spawn_call(
            attributes_setflags(
                ctypes.byref(attributes),
                _POSIX_SPAWN_SETPGROUP
                | _POSIX_SPAWN_SETSIGDEF
                | _POSIX_SPAWN_SETSIGMASK,
            ),
            "posix_spawnattr_setflags",
        )

        encoded_command = [os.fsencode(argument) for argument in command]
        encoded_environment = [
            os.fsencode(f"{key}={value}") for key, value in sorted(environment.items())
        ]
        argv = (ctypes.c_char_p * (len(encoded_command) + 1))(
            *encoded_command,
            None,
        )
        envp = (ctypes.c_char_p * (len(encoded_environment) + 1))(
            *encoded_environment,
            None,
        )
        spawned_pid = _call_libc_posix_spawn(
            spawn,
            process._spawn_pid_cell,
            encoded_command[0],
            actions,
            attributes,
            argv,
            envp,
        )
        if spawned_pid <= 0:
            raise MutationHostLifecycleError(
                "atomic POSIX spawn returned an unsafe child PID"
            )
        # libc populated process-owned storage at the syscall boundary.  The
        # return value is only a consistency check, never the authority handoff.
        if process.pid != spawned_pid:
            raise MutationHostLifecycleError(
                "atomic POSIX spawn PID result cell changed"
            )
    except BaseException as exc:
        primary_error = exc
    destroy_errors: list[OSError] = []
    if attributes_initialized:
        result = attributes_destroy(ctypes.byref(attributes))
        if result:
            destroy_errors.append(
                OSError(result, f"posix_spawnattr_destroy: {os.strerror(result)}")
            )
    if actions_initialized:
        result = actions_destroy(ctypes.byref(actions))
        if result:
            destroy_errors.append(
                OSError(
                    result,
                    f"posix_spawn_file_actions_destroy: {os.strerror(result)}",
                )
            )
    if primary_error is not None:
        raise primary_error
    if destroy_errors:
        raise MutationHostLifecycleError(
            "atomic POSIX spawn record cleanup failed with attached PID authority"
        ) from destroy_errors[0]


def _posix_spawn_child(
    process: _PosixSpawnProcess,
    command: list[str],
    environment: dict[str, str],
    stderr_descriptor: int,
    source_authority: _ChildSourceAuthority,
    private_root_authority: _PrivateRootAuthority,
    mutation_roots: _MutationRootAuthorities,
) -> None:
    """Populate attached process authority through atomic close-from spawn."""

    source_descriptor = source_authority.pass_descriptor
    root_descriptor = private_root_authority.child_descriptor
    child_source_descriptor = source_authority.child_source_descriptor
    if child_source_descriptor != _CHILD_SOURCE_DESCRIPTOR:
        raise MutationHostQualificationError(
            "isolated mutation child source descriptor mapping is unsafe"
        )
    if process.pid is not None or process.raw_descriptors:
        raise MutationHostLifecycleError(
            "isolated mutation provisional process authority is not empty"
        )
    process.acquire_pipe_pair(
        "_request_read_descriptor",
        "_request_write_descriptor",
    )
    process.acquire_pipe_pair(
        "_response_read_descriptor",
        "_response_write_descriptor",
    )
    request_read = process._request_read_descriptor
    request_write = process._request_write_descriptor
    response_read = process._response_read_descriptor
    response_write = process._response_write_descriptor
    assert request_read is not None
    assert request_write is not None
    assert response_read is not None
    assert response_write is not None
    sources = (
        request_read,
        response_write,
        stderr_descriptor,
        source_descriptor,
        root_descriptor,
        *mutation_roots.pass_descriptors,
    )
    final_targets = (
        0,
        1,
        2,
        child_source_descriptor,
        _CHILD_ROOT_DESCRIPTOR,
        *(root.child_descriptor for root in mutation_roots.roots),
    )
    projections = _spawn_projection_targets(
        frozenset({*sources, *final_targets}),
        count=len(sources),
    )
    duplicate_actions = tuple(zip(sources, projections, strict=True)) + tuple(
        zip(projections, final_targets, strict=True)
    )
    spawn_environment = dict(environment)
    spawn_environment[_CHILD_SOURCE_DESCRIPTOR_ENV] = str(child_source_descriptor)
    spawn_environment[_CHILD_SOURCE_SHA256_ENV] = source_authority.image.sha256
    spawn_environment[_CHILD_ROOT_DESCRIPTOR_ENV] = str(_CHILD_ROOT_DESCRIPTOR)
    spawn_environment[_CHILD_MUTATION_ROOTS_ENV] = json.dumps(
        mutation_roots.wire_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    )
    _atomic_posix_spawn_closefrom(
        process,
        command,
        spawn_environment,
        duplicate_actions,
        closefrom=mutation_roots.roots[-1].child_descriptor + 1,
    )


def _owned_entry_names(authority: _PrivateRootAuthority) -> tuple[str, ...]:
    """Find exact root inode names beneath the held original parent."""

    parent_descriptor = authority.parent_descriptor
    root_identity = authority.root_identity
    if parent_descriptor is None or root_identity is None:
        raise MutationHostLifecycleError(
            "private-root parent or inode identity is unavailable"
        )
    matches: list[str] = []
    try:
        entries = os.scandir(parent_descriptor)
    except OSError as exc:
        raise MutationHostLifecycleError(
            "could not inventory the held private-root parent"
        ) from exc
    with entries:
        for entry in entries:
            name = entry.name
            if not isinstance(name, str) or name in {"", ".", ".."}:
                raise MutationHostLifecycleError(
                    "private-root parent emitted an unsafe directory entry"
                )
            try:
                # The root was created on the held parent's mount. Its inode
                # must therefore appear in that parent's directory entries,
                # including after a rename. Avoid stat-ing unrelated entries:
                # a foreign FUSE mount can deny lookup even in a readable /tmp.
                # An unavailable inode hint still requires full verification;
                # no lookup error on a possible owned entry is suppressed.
                inode_hint = entry.inode()
                if inode_hint and inode_hint != root_identity[1]:
                    continue
                observed = os.stat(
                    name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if _stable_directory_identity(observed) == root_identity:
                matches.append(name)
    return tuple(matches)


def _clear_owned_directory(
    authority: _PrivateRootAuthority,
    descriptor: int,
    *,
    expected_mount_id: int,
) -> None:
    """Remove contents only through held, no-follow directory descriptors."""

    try:
        names = os.listdir(descriptor)
    except OSError as exc:
        raise MutationHostLifecycleError(
            "could not inventory an owned private-root directory"
        ) from exc
    for name in names:
        if not isinstance(name, str) or name in {"", ".", ".."}:
            raise MutationHostLifecycleError(
                "owned private root emitted an unsafe directory entry"
            )
        try:
            observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(observed.st_mode):
            try:
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except BaseException as exc:
                authority.cleanup_acquisition_uncertain = True
                raise MutationHostLifecycleError(
                    "private-root child acquisition outcome is uncertain"
                ) from exc
            close_error: OSError | None = None
            try:
                if (
                    _stable_directory_identity(os.fstat(child_descriptor))
                    != _stable_directory_identity(observed)
                    or _descriptor_mount_id(child_descriptor) != expected_mount_id
                ):
                    raise MutationHostLifecycleError(
                        "private-root child changed identity or crossed a mount"
                    )
                _clear_owned_directory(
                    authority,
                    child_descriptor,
                    expected_mount_id=expected_mount_id,
                )
                _quarantine_remove_bound_entry(
                    descriptor,
                    name,
                    child_descriptor,
                    directory=True,
                    token=secrets.token_hex(16),
                )
            finally:
                try:
                    os.close(child_descriptor)
                except OSError as exc:
                    close_error = exc
            if close_error is not None:
                raise MutationHostLifecycleError(
                    "private-root child descriptor close failed"
                ) from close_error
            continue

        try:
            leaf_descriptor = os.open(
                name,
                os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
        except BaseException as exc:
            authority.cleanup_acquisition_uncertain = True
            raise MutationHostLifecycleError(
                "private-root leaf acquisition outcome is uncertain"
            ) from exc
        close_error = None
        try:
            if _stat_identity(os.fstat(leaf_descriptor)) != _stat_identity(observed):
                raise MutationHostLifecycleError(
                    "private-root leaf changed while opening"
                )
            _quarantine_remove_bound_entry(
                descriptor,
                name,
                leaf_descriptor,
                directory=False,
                token=secrets.token_hex(16),
            )
        finally:
            try:
                os.close(leaf_descriptor)
            except OSError as exc:
                close_error = exc
        if close_error is not None:
            raise MutationHostLifecycleError(
                "private-root leaf descriptor close failed"
            ) from close_error


def _close_private_root_authority(authority: _PrivateRootAuthority) -> None:
    """Remove and close only the descriptor-bound root, even after rename."""

    if authority.parent_acquisition_uncertain:
        raise MutationHostLifecycleError(
            "private-root parent acquisition remains uncertain"
        )
    if authority.root_descriptor_acquisition_uncertain:
        raise MutationHostLifecycleError(
            "private-root descriptor acquisition remains uncertain"
        )
    if authority.cleanup_acquisition_uncertain:
        raise MutationHostLifecycleError(
            "private-root cleanup acquisition remains uncertain"
        )
    if authority.pending_creation is not None:
        pending = authority.pending_creation
        if (
            pending.descriptor is None
            and not pending.descriptor_settlement_uncertain
            and authority.root_descriptor is not None
        ):
            # The guarded ownership handoff completed and only publication of
            # the in-memory marker was interrupted.  The authority is now the
            # sole owner; continue through its normal exact cleanup path.
            authority.pending_creation = None
        else:
            pending.rollback()
            authority.pending_creation = None
            authority.root_name = None
            authority.root_path = None
            authority.root_removed = True
    if authority.root_creation_uncertain:
        if authority.parent_descriptor is None or authority.root_name is None:
            raise MutationHostLifecycleError(
                "private-root mkdir authority is incomplete"
            )
        try:
            os.stat(
                authority.root_name,
                dir_fd=authority.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            # The exact pending name is absent beneath the retained parent;
            # this is the only safe settlement of an ambiguous mkdir result.
            authority.root_creation_uncertain = False
            authority.root_name = None
            authority.root_path = None
            authority.root_removed = True
        else:
            raise MutationHostLifecycleError(
                "private-root mkdir effect remains uncertain"
            )

    if not authority.root_removed:
        if not authority.has_created_root:
            authority.root_removed = True
        else:
            root_descriptor = authority.root_descriptor
            if (
                root_descriptor is None
                or authority.root_identity is None
                or authority.root_mount_id is None
                or authority.parent_descriptor is None
                or authority.parent_identity is None
                or authority.parent_mount_id is None
            ):
                raise MutationHostLifecycleError(
                    "private-root acquisition failed before exact cleanup "
                    "authority was complete"
                )
            try:
                if (
                    _stable_directory_identity(os.fstat(root_descriptor))
                    != authority.root_identity
                    or _descriptor_mount_id(root_descriptor) != authority.root_mount_id
                    or _stable_directory_identity(os.fstat(authority.parent_descriptor))
                    != authority.parent_identity
                    or _descriptor_mount_id(authority.parent_descriptor)
                    != authority.parent_mount_id
                ):
                    raise MutationHostLifecycleError(
                        "private-root descriptor identity changed before cleanup"
                    )
                _clear_owned_directory(
                    authority,
                    root_descriptor,
                    expected_mount_id=authority.root_mount_id,
                )
                matches = _owned_entry_names(authority)
                if len(matches) != 1:
                    root_state = os.fstat(root_descriptor)
                    if not matches and root_state.st_nlink == 0:
                        authority.root_removal_unsynced = True
                    else:
                        raise MutationHostLifecycleError(
                            "owned private root was moved outside its held parent "
                            "or has ambiguous names"
                        )
                else:
                    owned_name = matches[0]
                    current = os.stat(
                        owned_name,
                        dir_fd=authority.parent_descriptor,
                        follow_symlinks=False,
                    )
                    if _stable_directory_identity(current) != authority.root_identity:
                        raise MutationHostLifecycleError(
                            "owned private-root name was rebound before removal"
                        )
                    _quarantine_remove_bound_entry(
                        authority.parent_descriptor,
                        owned_name,
                        root_descriptor,
                        directory=True,
                        token=secrets.token_hex(16),
                    )
                    if os.fstat(root_descriptor).st_nlink != 0:
                        raise MutationHostLifecycleError(
                            "owned private root survived descriptor-relative removal"
                        )
                    authority.root_removal_unsynced = True
                try:
                    os.fsync(authority.parent_descriptor)
                except BaseException as exc:
                    raise MutationHostLifecycleError(
                        "private-root removal fsync outcome is uncertain"
                    ) from exc
                if (
                    _stable_directory_identity(os.fstat(authority.parent_descriptor))
                    != authority.parent_identity
                    or _descriptor_mount_id(authority.parent_descriptor)
                    != authority.parent_mount_id
                    or _owned_entry_names(authority)
                    or os.fstat(root_descriptor).st_nlink != 0
                ):
                    raise MutationHostLifecycleError(
                        "private-root removal did not settle exact absence"
                    )
                authority.root_removal_unsynced = False
                authority.root_removed = True
            except MutationHostLifecycleError:
                raise
            except OSError as exc:
                raise MutationHostLifecycleError(
                    "descriptor-relative private-root cleanup failed"
                ) from exc

    failures: list[OSError] = []
    for attribute in ("root_descriptor", "parent_descriptor"):
        descriptor = getattr(authority, attribute)
        if descriptor is None:
            continue
        setattr(authority, attribute, None)
        try:
            os.close(descriptor)
        except OSError as exc:
            failures.append(exc)
    if failures:
        raise MutationHostLifecycleError(
            "private-root descriptor cleanup failed"
        ) from failures[0]


@dataclass(frozen=True, slots=True)
class _PendingStaging:
    root_path: str
    relative_path: str
    restore_target: Path | None = None
    repository_root: str | None = None
    parent_components: tuple[str, ...] = ()
    target_relative: str | None = None

    @property
    def path(self) -> Path:
        """Render the retained relative name without changing its authority."""

        return Path(self.root_path) / self.relative_path


@dataclass(slots=True)
class _PublicationAncestor:
    parent_descriptor: int
    name: str
    descriptor: int | None
    identity: tuple[int, ...]
    mount_id: int
    created: bool
    creation_unsynced: bool = False
    removed: bool = False
    removal_unsynced: bool = False
    removal_state: _QuarantineRemovalState | None = None


def _read_regular_descriptor(
    descriptor: int,
    *,
    expected_identity: tuple[int, ...] | None = None,
    allow_unlinked: bool = False,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or (not allow_unlinked and before.st_nlink < 1)
        or (
            expected_identity is not None
            and _stat_identity(before) != expected_identity
        )
    ):
        raise MutationHostQualificationError(
            "publication descriptor is not one retained regular file"
        )
    payload = bytearray()
    while len(payload) < before.st_size:
        chunk = os.pread(
            descriptor,
            min(_SOURCE_HASH_CHUNK_BYTES, before.st_size - len(payload)),
            len(payload),
        )
        if not chunk:
            raise MutationHostQualificationError(
                "publication descriptor ended before its exact size"
            )
        payload.extend(chunk)
    if os.pread(descriptor, 1, before.st_size) or _stat_identity(
        os.fstat(descriptor)
    ) != _stat_identity(before):
        raise MutationHostQualificationError(
            "publication descriptor changed while reading"
        )
    return bytes(payload), before


def _publication_inode_key(identity: tuple[int, ...]) -> tuple[int, ...]:
    return (
        identity[0],
        identity[1],
        stat.S_IFMT(identity[2]),
        identity[4],
        identity[5],
    )


def _link_descriptor_to_name(
    source_descriptor: int,
    parent_descriptor: int,
    name: str,
) -> None:
    # ``AT_EMPTY_PATH`` requires a capability on several supported kernels.
    # The proc descriptor symlink resolves the already-held open file
    # description; ``follow_symlinks=True`` links its inode, never a mutable
    # staging pathname.
    os.link(
        f"/proc/self/fd/{source_descriptor}",
        name,
        dst_dir_fd=parent_descriptor,
        follow_symlinks=True,
    )


def _rename_exchange(
    parent_descriptor: int,
    first: str,
    second: str,
) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise MutationHostQualificationError(
            "atomic publication exchange is unavailable"
        ) from exc
    renameat2.restype = ctypes.c_int
    result = int(
        renameat2(
            parent_descriptor,
            os.fsencode(first),
            parent_descriptor,
            os.fsencode(second),
            0x2,  # RENAME_EXCHANGE
        )
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"renameat2: {os.strerror(error)}")


@dataclass(slots=True)
class _ParentPublicationTransaction:
    """Parent-owned ancestor, stage, publication, and rollback authority."""

    root: _MutationRootAuthority
    normalized: str
    token: str
    ancestors: list[_PublicationAncestor]
    leaf: str
    final_parent_descriptor: int
    stage_name: str
    stage_descriptor: int | None = None
    stage_identity: tuple[int, ...] | None = None
    stage_sha256: str | None = None
    stage_mount_id: int | None = None
    stage_acquisition_uncertain: bool = False
    stage_removal_unsynced: bool = False
    stage_removal_state: _QuarantineRemovalState | None = None
    ancestor_acquisition_uncertain: bool = False
    pending_directory_creation: _BoundDirectoryCreation | None = None
    target_descriptor: int | None = None
    target_identity: tuple[int, ...] | None = None
    target_sha256: str | None = None
    target_mode: int | None = None
    target_acquisition_uncertain: bool = False
    target_removal_state: _QuarantineRemovalState | None = None
    continuity_recovery_pending: bool = False
    continuity_failure_observed: bool = False
    recovery_stage_identity: tuple[int, ...] | None = None
    recovery_target_identity: tuple[int, ...] | None = None
    publication_kind: Literal["none", "create", "replace"] = "none"
    phase: Literal[
        "attached",
        "staged",
        "create_attempted",
        "provisional_create",
        "commit_create_attempted",
        "replace_attempted",
        "provisional_replace",
        "rollback_replace_attempted",
        "replace_restored",
        "commit_replace_attempted",
        "committed_unsynced",
        "committed",
        "rolled_back_unsynced",
        "rolled_back",
        "closed",
    ] = "attached"
    descriptor_settlement_uncertain: bool = False

    @classmethod
    def prepare(
        cls,
        roots: _MutationRootAuthorities,
        context: MutationContextSpec,
        normalized: str,
        token: str,
        *,
        attach: Callable[[_ParentPublicationTransaction], None],
    ) -> _ParentPublicationTransaction:
        root = next(
            (
                candidate
                for candidate in roots.roots
                if str(candidate.path) == context.repo_root
                and candidate.expected_identity == context.repo_root_identity
                and candidate.expected_mount_id == context.repo_root_mount_id
            ),
            None,
        )
        if root is None or root.descriptor is None:
            raise MutationHostLifecycleError(
                "publication has no exact retained repository root"
            )
        root.revalidate(require_named=True)
        parts = PurePosixPath(normalized).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise MutationHostQualificationError("publication target is not normalized")
        transaction = cls(
            root=root,
            normalized=normalized,
            token=token,
            ancestors=[],
            leaf=parts[-1],
            final_parent_descriptor=root.descriptor,
            stage_name=f".{parts[-1]}.crucible-{token}.tmp",
        )
        attach(transaction)
        descriptor = root.descriptor
        for part in parts[:-1]:
            created = False
            try:
                observed = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pending = _BoundDirectoryCreation(
                    parent_descriptor=descriptor,
                    parent_identity=_stable_directory_identity(os.fstat(descriptor)),
                    expected_mount_id=root.expected_mount_id,
                    final_name=part,
                    private_name=(f".crucible-dir-{token}-{secrets.token_hex(16)}"),
                    mode=0o755,
                    token=token,
                )
                transaction.pending_directory_creation = pending
                pending.create()
                pending.publish_and_sync()
                child = pending.descriptor
                identity = pending.identity
                if child is None or identity is None:
                    raise MutationHostLifecycleError(
                        "created publication ancestor lost its authority"
                    ) from None
                created = True
                observed = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            if not stat.S_ISDIR(observed.st_mode):
                raise MutationHostQualificationError(
                    "publication ancestor is not a directory"
                )
            identity = _stable_directory_identity(observed)
            ancestor: _PublicationAncestor | None = None
            if not created:
                transaction.ancestor_acquisition_uncertain = True
                try:
                    child = os.open(
                        part,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                except BaseException as exc:
                    raise MutationHostLifecycleError(
                        "publication ancestor descriptor outcome is uncertain"
                    ) from exc
                ancestor = _PublicationAncestor(
                    parent_descriptor=descriptor,
                    name=part,
                    descriptor=child,
                    identity=identity,
                    mount_id=root.expected_mount_id,
                    created=False,
                )
                transaction.ancestors.append(ancestor)
                transaction.ancestor_acquisition_uncertain = False
            if child is None:
                raise MutationHostLifecycleError(
                    "publication ancestor descriptor is unavailable"
                )
            mount_id = _descriptor_mount_id(child)
            if (
                _stable_directory_identity(os.fstat(child)) != identity
                or mount_id != root.expected_mount_id
            ):
                raise MutationHostQualificationError(
                    "publication ancestor changed or crossed a mount"
                )
            if created:
                ancestor = _PublicationAncestor(
                    parent_descriptor=descriptor,
                    name=part,
                    descriptor=None,
                    identity=identity,
                    mount_id=mount_id,
                    created=True,
                )
                transaction.ancestors.append(ancestor)
                assert transaction.pending_directory_creation is pending
                try:
                    pending.descriptor = None
                    ancestor.descriptor = child
                except BaseException:
                    if ancestor.descriptor is None:
                        pending.descriptor = child
                    raise
                transaction.pending_directory_creation = None
            else:
                assert ancestor is not None
                ancestor.mount_id = mount_id
            transaction.final_parent_descriptor = child
            transaction.ancestor_acquisition_uncertain = False
            descriptor = child
        transaction.stage_acquisition_uncertain = True
        try:
            stage_descriptor = os.open(
                transaction.stage_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "publication stage descriptor outcome is uncertain"
            ) from exc
        transaction.stage_descriptor = stage_descriptor
        transaction.stage_acquisition_uncertain = False
        stage_metadata = os.fstat(stage_descriptor)
        if not stat.S_ISREG(stage_metadata.st_mode) or stage_metadata.st_nlink != 1:
            raise MutationHostQualificationError(
                "publication stage is not one regular file"
            )
        transaction.stage_identity = _stat_identity(stage_metadata)
        transaction.stage_mount_id = _descriptor_mount_id(stage_descriptor)
        if transaction.stage_mount_id != root.expected_mount_id:
            raise MutationHostQualificationError("publication stage crossed a mount")
        transaction.phase = "staged"
        return transaction

    def _named_stage_identity(self) -> tuple[int, ...] | None:
        try:
            observed = os.stat(
                self.stage_name,
                dir_fd=self.final_parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        return _stat_identity(observed)

    def revalidate(self) -> None:
        self.root.revalidate(require_named=True)
        for ancestor in self.ancestors:
            descriptor = ancestor.descriptor
            if descriptor is None:
                raise MutationHostLifecycleError(
                    "publication ancestor descriptor is unavailable"
                )
            observed = os.stat(
                ancestor.name,
                dir_fd=ancestor.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(observed.st_mode)
                or _stable_directory_identity(observed) != ancestor.identity
                or _stable_directory_identity(os.fstat(descriptor)) != ancestor.identity
                or _descriptor_mount_id(descriptor) != ancestor.mount_id
            ):
                raise MutationHostQualificationError(
                    "publication ancestor name or descriptor was rebound"
                )

    def _revalidate_held_publication_chain(self) -> None:
        """Validate retained directory authority without requiring public names."""

        self.root.revalidate(require_named=False)
        for ancestor in self.ancestors:
            if ancestor.removed:
                continue
            descriptor = ancestor.descriptor
            if descriptor is None:
                raise MutationHostLifecycleError(
                    "publication ancestor descriptor is unavailable"
                )
            if (
                _stable_directory_identity(os.fstat(descriptor)) != ancestor.identity
                or _descriptor_mount_id(descriptor) != ancestor.mount_id
            ):
                raise MutationHostQualificationError(
                    "held publication ancestor authority changed"
                )

    def _settle_unpublished_stage_removal(self) -> None:
        """Durably remove a staged inode that never entered publication."""

        descriptor = self.stage_descriptor
        expected = self.stage_identity
        if descriptor is None or expected is None:
            raise MutationHostLifecycleError(
                "unpublished stage removal authority is unavailable"
            )
        if self.phase != "staged":
            raise MutationHostLifecycleError(
                "unpublished stage removal phase is invalid"
            )
        # Set the durable-retry marker before the destructive transition.  An
        # exception after unlink but before fsync must force the next cleanup
        # attempt to sync the same retained parent even when the name is gone.
        self.stage_removal_unsynced = True
        self._unlink_exact_stage_name(expected)
        try:
            os.fsync(self.final_parent_descriptor)
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "unpublished stage removal fsync outcome is uncertain"
            ) from exc
        self._revalidate_held_publication_chain()
        observed = _stat_identity(os.fstat(descriptor))
        if (
            _publication_inode_key(observed) != _publication_inode_key(expected)
            or not stat.S_ISREG(observed[2])
            or observed[3] != 0
            or self._named_stage_identity() is not None
            or _descriptor_mount_id(descriptor) != self.stage_mount_id
        ):
            raise MutationHostLifecycleError(
                "unpublished stage removal did not settle exact absence"
            )
        self.stage_identity = observed
        self.stage_removal_unsynced = False

    def write_stage(self, content: str) -> str:
        descriptor = self.stage_descriptor
        if descriptor is None:
            raise MutationHostLifecycleError("publication stage is unavailable")
        payload = content.encode("utf-8")
        os.ftruncate(descriptor, 0)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        self.stage_identity = _stat_identity(metadata)
        self.stage_mount_id = _descriptor_mount_id(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or self.stage_mount_id != self.root.expected_mount_id
        ):
            raise MutationHostQualificationError(
                "publication stage authority changed while writing"
            )
        digest = hashlib.sha256(payload).hexdigest()
        self.stage_sha256 = digest
        return digest

    def _verify_stage(self, expected_sha256: str) -> None:
        descriptor = self.stage_descriptor
        identity = self.stage_identity
        if descriptor is None or identity is None:
            raise MutationHostLifecycleError("publication stage is incomplete")
        payload, observed = _read_regular_descriptor(
            descriptor,
            expected_identity=identity,
        )
        if (
            observed.st_nlink != 1
            or self._named_stage_identity() != identity
            or _descriptor_mount_id(descriptor) != self.stage_mount_id
            or hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise MutationHostQualificationError(
                "publication stage name, inode, mount, or content changed"
            )
        self.stage_sha256 = expected_sha256

    def _open_target(self) -> tuple[bytes, tuple[int, ...]]:
        observed = os.stat(
            self.leaf,
            dir_fd=self.final_parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise MutationHostQualificationError(
                "publication target is not one regular file"
            )
        self.target_acquisition_uncertain = True
        try:
            descriptor = os.open(
                self.leaf,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=self.final_parent_descriptor,
            )
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "publication target descriptor outcome is uncertain"
            ) from exc
        self.target_descriptor = descriptor
        self.target_acquisition_uncertain = False
        identity = _stat_identity(observed)
        payload, current = _read_regular_descriptor(
            descriptor,
            expected_identity=identity,
        )
        if self._target_named_identity() != _stat_identity(current):
            raise MutationHostQualificationError(
                "publication target changed while binding"
            )
        self.target_identity = _stat_identity(current)
        self.target_sha256 = hashlib.sha256(payload).hexdigest()
        self.target_mode = stat.S_IMODE(current.st_mode)
        return payload, identity

    def _prepare_replacement_mode(self) -> None:
        descriptor = self.stage_descriptor
        desired = self.target_mode
        if descriptor is None or desired is None or self.stage_sha256 is None:
            raise MutationHostLifecycleError(
                "replacement permission authority is unavailable"
            )
        chmod_error: BaseException | None = None
        try:
            os.fchmod(descriptor, desired)
        except BaseException as exc:
            chmod_error = exc
        metadata = os.fstat(descriptor)
        if stat.S_IMODE(metadata.st_mode) != desired:
            raise MutationHostLifecycleError(
                "replacement permission update outcome is uncertain"
            ) from chmod_error
        self.stage_identity = _stat_identity(metadata)
        self._verify_stage(self.stage_sha256)
        try:
            os.fsync(descriptor)
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "replacement permission fsync outcome is uncertain"
            ) from exc
        if self._stage_descriptor_identity() != self.stage_identity:
            raise MutationHostQualificationError(
                "replacement metadata changed during permission fsync"
            )
        self._verify_stage(self.stage_sha256)

    def _target_named_identity(self) -> tuple[int, ...] | None:
        try:
            return _stat_identity(
                os.stat(
                    self.leaf,
                    dir_fd=self.final_parent_descriptor,
                    follow_symlinks=False,
                )
            )
        except FileNotFoundError:
            return None

    def _unlink_exact_stage_name(self, expected: tuple[int, ...]) -> None:
        removal_state = self.stage_removal_state
        observed = self._named_stage_identity()
        if observed is None and not (
            removal_state is not None and removal_state.active
        ):
            return
        if observed is not None and _publication_inode_key(
            observed
        ) != _publication_inode_key(expected):
            raise MutationHostLifecycleError(
                "publication cleanup refused a rebound stage name"
            )
        descriptor = self._descriptor_for_identity(expected)
        if removal_state is None:
            removal_state = _QuarantineRemovalState()
            self.stage_removal_state = removal_state
        _quarantine_remove_bound_entry(
            self.final_parent_descriptor,
            self.stage_name,
            descriptor,
            directory=False,
            token=self.token,
            removal_state=removal_state,
        )

    def _unlink_exact_target_name(self, expected: tuple[int, ...]) -> None:
        removal_state = self.target_removal_state
        observed = self._target_named_identity()
        if observed is None and not (
            removal_state is not None and removal_state.active
        ):
            return
        if observed is not None and _publication_inode_key(
            observed
        ) != _publication_inode_key(expected):
            raise MutationHostLifecycleError(
                "publication cleanup refused a rebound target name"
            )
        descriptor = self._descriptor_for_identity(expected)
        if removal_state is None:
            removal_state = _QuarantineRemovalState()
            self.target_removal_state = removal_state
        _quarantine_remove_bound_entry(
            self.final_parent_descriptor,
            self.leaf,
            descriptor,
            directory=False,
            token=self.token,
            removal_state=removal_state,
        )

    def _descriptor_for_identity(self, expected: tuple[int, ...]) -> int:
        expected_key = _publication_inode_key(expected)
        for descriptor in (self.stage_descriptor, self.target_descriptor):
            if descriptor is None:
                continue
            if (
                _publication_inode_key(_stat_identity(os.fstat(descriptor)))
                == expected_key
            ):
                return descriptor
        raise MutationHostLifecycleError(
            "publication cleanup lost its retained inode descriptor"
        )

    @staticmethod
    def _same_inode(
        first: tuple[int, ...] | None,
        second: tuple[int, ...],
    ) -> bool:
        return first is not None and _publication_inode_key(
            first
        ) == _publication_inode_key(second)

    def _stage_descriptor_identity(self) -> tuple[int, ...]:
        if self.stage_descriptor is None:
            raise MutationHostLifecycleError("publication stage is unavailable")
        return _stat_identity(os.fstat(self.stage_descriptor))

    def _target_descriptor_identity(self) -> tuple[int, ...]:
        if self.target_descriptor is None:
            raise MutationHostLifecycleError("publication target is unavailable")
        return _stat_identity(os.fstat(self.target_descriptor))

    def _settle_create_attempt(self) -> bool:
        """Return whether the retained new inode is bound at the target name."""

        stage = self._stage_descriptor_identity()
        target_name = self._target_named_identity()
        stage_name = self._named_stage_identity()
        self.stage_identity = stage
        if self._same_inode(target_name, stage):
            self.phase = "provisional_create"
            if stage[3] != 2 or target_name != stage or stage_name != stage:
                raise MutationHostQualificationError(
                    "create publication acquired an unexpected inode alias"
                )
            return True
        if target_name is None and stage_name == stage and stage[3] == 1:
            self.phase = "staged"
            return False
        raise MutationHostLifecycleError(
            "create publication namespace outcome is uncertain"
        )

    def _replacement_orientation(self) -> Literal["before", "after"] | None:
        stage = self._stage_descriptor_identity()
        target = self._target_descriptor_identity()
        target_name = self._target_named_identity()
        stage_name = self._named_stage_identity()
        self.stage_identity = stage
        self.target_identity = target
        if (
            self._same_inode(target_name, target)
            and self._same_inode(stage_name, stage)
            and target_name == target
            and stage_name == stage
            and target[3] == 1
            and stage[3] == 1
        ):
            return "before"
        if (
            self._same_inode(target_name, stage)
            and self._same_inode(stage_name, target)
            and target_name == stage
            and stage_name == target
            and target[3] == 1
            and stage[3] == 1
        ):
            return "after"
        return None

    def _settle_replace_attempt(self) -> bool:
        """Return whether the exact replacement exchange took effect."""

        orientation = self._replacement_orientation()
        if orientation == "after":
            self.phase = "provisional_replace"
            return True
        if orientation == "before":
            self.phase = "staged"
            return False
        raise MutationHostLifecycleError(
            "replacement publication namespace outcome is uncertain"
        )

    def _verify_provisional_create(self) -> None:
        stage = self._stage_descriptor_identity()
        self.stage_identity = stage
        if (
            stage[3] != 2
            or self._target_named_identity() != stage
            or self._named_stage_identity() != stage
        ):
            raise MutationHostQualificationError(
                "provisional create publication authority changed"
            )
        self._verify_new_payload()

    def _verify_provisional_replace(self) -> None:
        if self._replacement_orientation() != "after":
            raise MutationHostQualificationError(
                "provisional replacement authority changed"
            )
        self._verify_new_payload()

    def _verify_new_payload(self) -> None:
        descriptor = self.stage_descriptor
        expected_sha256 = self.stage_sha256
        expected_identity = self.stage_identity
        if descriptor is None or expected_sha256 is None or expected_identity is None:
            raise MutationHostLifecycleError(
                "publication content authority is unavailable"
            )
        payload, observed = _read_regular_descriptor(
            descriptor,
            expected_identity=expected_identity,
        )
        if (
            _stat_identity(observed) != expected_identity
            or hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise MutationHostQualificationError(
                "publication content or metadata changed after staging"
            )

    def _verify_original_target_payload(self) -> None:
        descriptor = self.target_descriptor
        expected_sha256 = self.target_sha256
        expected_identity = self.target_identity
        if descriptor is None or expected_sha256 is None or expected_identity is None:
            raise MutationHostLifecycleError(
                "original publication target authority is unavailable"
            )
        payload, observed = _read_regular_descriptor(
            descriptor,
            expected_identity=expected_identity,
        )
        if (
            _stat_identity(observed) != expected_identity
            or hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise MutationHostQualificationError(
                "original publication target content or metadata changed"
            )

    def _capture_exact_publication_state(
        self,
        *,
        rolled_back: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        try:
            return self._capture_exact_publication_state_unchecked(
                rolled_back=rolled_back
            )
        except (MutationHostLifecycleError, MutationHostQualificationError):
            self.continuity_failure_observed = True
            raise

    def _capture_exact_publication_state_unchecked(
        self,
        *,
        rolled_back: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Capture one stable payload, metadata, namespace, and mount checkpoint."""

        stage_descriptor = self.stage_descriptor
        stage_sha256 = self.stage_sha256
        baseline_stage = self.stage_identity
        if stage_descriptor is None or stage_sha256 is None or baseline_stage is None:
            raise MutationHostLifecycleError(
                "publication checkpoint authority is unavailable"
            )
        stage_payload, stage_metadata = _read_regular_descriptor(
            stage_descriptor,
            allow_unlinked=True,
        )
        stage_identity = _stat_identity(stage_metadata)
        if self._stage_descriptor_identity() != stage_identity:
            raise MutationHostQualificationError(
                "publication stage changed during checkpoint capture"
            )
        expected_stage_mode = stat.S_IMODE(baseline_stage[2])
        if self.publication_kind == "replace":
            if self.target_mode is None:
                raise MutationHostLifecycleError(
                    "replacement permission authority is unavailable"
                )
            expected_stage_mode = self.target_mode
        if (
            hashlib.sha256(stage_payload).hexdigest() != stage_sha256
            or stat.S_IMODE(stage_metadata.st_mode) != expected_stage_mode
            or _descriptor_mount_id(stage_descriptor) != self.stage_mount_id
        ):
            raise MutationHostQualificationError(
                "publication content, mode, or mount changed"
            )

        target_name = self._target_named_identity()
        stage_name = self._named_stage_identity()
        if rolled_back:
            if stage_identity[3] != 0 or stage_name is not None:
                raise MutationHostLifecycleError(
                    "rolled-back publication retained an unexpected new-inode alias"
                )
            if self.publication_kind == "create":
                if target_name is not None:
                    raise MutationHostLifecycleError(
                        "rolled-back create retained its target name"
                    )
                return stage_identity, None
        else:
            if (
                stage_identity[3] != 1
                or target_name != stage_identity
                or stage_name is not None
            ):
                raise MutationHostQualificationError(
                    "committed publication acquired an unexpected inode alias"
                )
            if self.publication_kind == "create":
                return stage_identity, None

        if self.publication_kind != "replace":
            raise MutationHostLifecycleError(
                "publication kind is unavailable during exact validation"
            )
        target_descriptor = self.target_descriptor
        target_sha256 = self.target_sha256
        target_mode = self.target_mode
        if target_descriptor is None or target_sha256 is None or target_mode is None:
            raise MutationHostLifecycleError(
                "original publication target authority is unavailable"
            )
        target_payload, target_metadata = _read_regular_descriptor(
            target_descriptor,
            allow_unlinked=True,
        )
        target_identity = _stat_identity(target_metadata)
        if self._target_descriptor_identity() != target_identity:
            raise MutationHostQualificationError(
                "publication target changed during checkpoint capture"
            )
        if (
            hashlib.sha256(target_payload).hexdigest() != target_sha256
            or stat.S_IMODE(target_metadata.st_mode) != target_mode
            or _descriptor_mount_id(target_descriptor) != self.root.expected_mount_id
        ):
            raise MutationHostQualificationError(
                "original publication target content, mode, or mount changed"
            )
        if rolled_back:
            if target_identity[3] != 1 or target_name != target_identity:
                raise MutationHostLifecycleError(
                    "rolled-back replacement did not restore its exact target"
                )
        elif target_identity[3] != 0:
            raise MutationHostQualificationError(
                "committed replacement retained an unexpected old-target alias"
            )
        return stage_identity, target_identity

    def _expected_continuity_checkpoint(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        stage = (
            self.recovery_stage_identity
            if self.continuity_recovery_pending
            else self.stage_identity
        )
        target = (
            self.recovery_target_identity
            if self.continuity_recovery_pending
            else self.target_identity
        )
        if stage is None:
            raise MutationHostLifecycleError(
                "publication continuity checkpoint is unavailable"
            )
        return stage, target

    def _require_continuous_checkpoint(
        self,
        observed: tuple[tuple[int, ...], tuple[int, ...] | None],
    ) -> None:
        expected = self._expected_continuity_checkpoint()
        if observed == expected:
            return
        baseline_stage = self.stage_identity
        baseline_target = self.target_identity
        if (
            baseline_stage is None
            or _publication_inode_key(observed[0])
            != _publication_inode_key(baseline_stage)
            or (
                baseline_target is not None
                and (
                    observed[1] is None
                    or _publication_inode_key(observed[1])
                    != _publication_inode_key(baseline_target)
                )
            )
        ):
            raise MutationHostQualificationError("publication inode continuity changed")
        self.recovery_stage_identity = observed[0]
        self.recovery_target_identity = observed[1]
        self.continuity_recovery_pending = True
        if self.continuity_failure_observed:
            self.continuity_failure_observed = False
            return
        raise MutationHostQualificationError(
            "publication metadata continuity changed; an exact retry is required"
        )

    def _promote_continuity_checkpoint(self) -> None:
        if not self.continuity_recovery_pending:
            return
        if self.recovery_stage_identity is None:
            raise MutationHostLifecycleError(
                "publication recovery checkpoint is incomplete"
            )
        self.stage_identity = self.recovery_stage_identity
        self.target_identity = self.recovery_target_identity
        self.recovery_stage_identity = None
        self.recovery_target_identity = None
        self.continuity_recovery_pending = False
        self.continuity_failure_observed = False

    def publish(
        self,
        *,
        stage_sha256: str,
        expected_current_sha256: str | None,
    ) -> None:
        self.revalidate()
        self._verify_stage(stage_sha256)
        assert self.stage_descriptor is not None
        assert self.stage_identity is not None
        if expected_current_sha256 is None:
            if self._target_named_identity() is not None:
                raise ValueError(
                    f"destination appeared during publication: {self.leaf}"
                )
            self.publication_kind = "create"
            self.phase = "create_attempted"
            publication_error: BaseException | None = None
            try:
                _link_descriptor_to_name(
                    self.stage_descriptor,
                    self.final_parent_descriptor,
                    self.leaf,
                )
            except BaseException as exc:
                publication_error = exc
            took_effect = self._settle_create_attempt()
            if not took_effect:
                raise MutationHostLifecycleError(
                    "create publication did not take effect"
                ) from publication_error
            self._verify_provisional_create()
        else:
            current, target_identity = self._open_target()
            if hashlib.sha256(current).hexdigest() != expected_current_sha256:
                raise ValueError(f"stale file changed during publication: {self.leaf}")
            self.revalidate()
            self._verify_stage(stage_sha256)
            self._prepare_replacement_mode()
            if self._target_named_identity() != target_identity:
                raise MutationHostQualificationError(
                    "publication target changed before exchange"
                )
            self.publication_kind = "replace"
            self.phase = "replace_attempted"
            publication_error = None
            try:
                _rename_exchange(
                    self.final_parent_descriptor,
                    self.stage_name,
                    self.leaf,
                )
            except BaseException as exc:
                publication_error = exc
            took_effect = self._settle_replace_attempt()
            if not took_effect:
                raise MutationHostLifecycleError(
                    "replacement publication did not take effect"
                ) from publication_error
            self._verify_provisional_replace()
        self.revalidate()
        try:
            os.fsync(self.final_parent_descriptor)
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "provisional publication fsync outcome is uncertain"
            ) from exc
        self.revalidate()

    def _settle_create_commit_attempt(self) -> None:
        stage = self._stage_descriptor_identity()
        self.stage_identity = stage
        target_name = self._target_named_identity()
        stage_name = self._named_stage_identity()
        if target_name == stage and stage_name is None and stage[3] == 1:
            self.phase = "committed_unsynced"
            return
        if target_name == stage and stage_name == stage and stage[3] == 2:
            self.phase = "provisional_create"
            return
        raise MutationHostLifecycleError("create commit namespace outcome is uncertain")

    def _settle_replace_commit_attempt(self) -> None:
        stage = self._stage_descriptor_identity()
        target = self._target_descriptor_identity()
        self.stage_identity = stage
        self.target_identity = target
        target_name = self._target_named_identity()
        stage_name = self._named_stage_identity()
        if (
            target_name == stage
            and stage_name is None
            and stage[3] == 1
            and target[3] == 0
        ):
            self.phase = "committed_unsynced"
            return
        if self._replacement_orientation() == "after":
            self.phase = "provisional_replace"
            return
        raise MutationHostLifecycleError(
            "replacement commit namespace outcome is uncertain"
        )

    def _verify_committed_publication(self) -> None:
        self._require_continuous_checkpoint(
            self._capture_exact_publication_state(rolled_back=False)
        )
        self.revalidate()
        self._require_continuous_checkpoint(
            self._capture_exact_publication_state(rolled_back=False)
        )
        assert self.stage_descriptor is not None
        try:
            os.fsync(self.stage_descriptor)
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "committed publication file fsync outcome is uncertain"
            ) from exc
        self._require_continuous_checkpoint(
            self._capture_exact_publication_state(rolled_back=False)
        )
        self.revalidate()
        self._require_continuous_checkpoint(
            self._capture_exact_publication_state(rolled_back=False)
        )
        # This is intentionally the last authority-sensitive operation.  A
        # one-shot rebind immediately after an earlier chain check is caught
        # here rather than being accepted through held-parent file checks.
        self.revalidate()
        self._promote_continuity_checkpoint()

    def _finalize_publication(self) -> None:
        if self.phase == "commit_create_attempted":
            self._settle_create_commit_attempt()
        elif self.phase == "commit_replace_attempted":
            self._settle_replace_commit_attempt()
        if self.phase == "provisional_create":
            self._verify_provisional_create()
            self.revalidate()
            try:
                os.fsync(self.final_parent_descriptor)
            except BaseException as exc:
                raise MutationHostLifecycleError(
                    "create precommit fsync outcome is uncertain"
                ) from exc
            assert self.stage_identity is not None
            self.phase = "commit_create_attempted"
            self._unlink_exact_stage_name(self.stage_identity)
            self._settle_create_commit_attempt()
        elif self.phase == "provisional_replace":
            self._verify_provisional_replace()
            self.revalidate()
            try:
                os.fsync(self.final_parent_descriptor)
            except BaseException as exc:
                raise MutationHostLifecycleError(
                    "replacement precommit fsync outcome is uncertain"
                ) from exc
            assert self.target_identity is not None
            self.phase = "commit_replace_attempted"
            self._unlink_exact_stage_name(self.target_identity)
            self._settle_replace_commit_attempt()
        if self.phase == "committed_unsynced":
            try:
                os.fsync(self.final_parent_descriptor)
            except BaseException as exc:
                raise MutationHostLifecycleError(
                    "committed publication fsync outcome is uncertain"
                ) from exc
            self._verify_committed_publication()
            self.phase = "committed"
        if self.phase != "committed":
            raise MutationHostLifecycleError(
                "publication could not reach its exact commit state"
            )

    def _rollback_create(self) -> None:
        stage = self._stage_descriptor_identity()
        self.stage_identity = stage
        self._unlink_exact_target_name(stage)
        self._unlink_exact_stage_name(stage)
        self.stage_identity = self._stage_descriptor_identity()
        self.phase = "rolled_back_unsynced"
        self._finish_rollback()

    def _rollback_replace(self) -> None:
        if self.phase == "replace_restored":
            stage = self._stage_descriptor_identity()
            target = self._target_descriptor_identity()
            self.stage_identity = stage
            self.target_identity = target
            stage_name = self._named_stage_identity()
            target_name = self._target_named_identity()
            if stage_name is None:
                if stage[3] != 0 or target_name != target or target[3] != 1:
                    raise MutationHostLifecycleError(
                        "replacement rollback settlement is uncertain"
                    )
                self.phase = "rolled_back_unsynced"
                self._finish_rollback()
                return
        orientation = self._replacement_orientation()
        if orientation == "after":
            self.phase = "rollback_replace_attempted"
            exchange_error: BaseException | None = None
            try:
                _rename_exchange(
                    self.final_parent_descriptor,
                    self.stage_name,
                    self.leaf,
                )
            except BaseException as exc:
                exchange_error = exc
            orientation = self._replacement_orientation()
            if orientation != "before":
                raise MutationHostLifecycleError(
                    "replacement rollback exchange outcome is uncertain"
                ) from exchange_error
        if orientation != "before":
            raise MutationHostLifecycleError(
                "replacement rollback namespace is uncertain"
            )
        self.phase = "replace_restored"
        stage = self._stage_descriptor_identity()
        target = self._target_descriptor_identity()
        self.stage_identity = stage
        self.target_identity = target
        self._unlink_exact_stage_name(stage)
        self.stage_identity = self._stage_descriptor_identity()
        self.target_identity = self._target_descriptor_identity()
        self.phase = "rolled_back_unsynced"
        self._finish_rollback()

    def _verify_rolled_back_publication(self) -> None:
        self._require_continuous_checkpoint(
            self._capture_exact_publication_state(rolled_back=True)
        )
        self.revalidate()
        self._require_continuous_checkpoint(
            self._capture_exact_publication_state(rolled_back=True)
        )
        # Close the same revalidate-then-held-FD gap as commit validation.
        self.revalidate()
        self._promote_continuity_checkpoint()

    def _finish_rollback(self) -> None:
        if self.phase != "rolled_back_unsynced":
            raise MutationHostLifecycleError(
                "publication rollback sync phase is invalid"
            )
        # A failed rollback can deliberately retain an unexpected hard link
        # for operator settlement.  Its later removal is the missing rollback
        # transition, not a write/restore continuity event inside an fsync
        # checkpoint, so bind that exact transition before retrying durability.
        if self.stage_identity is not None and self.stage_identity[3] > 0:
            observed = self._capture_exact_publication_state(rolled_back=True)
            if _publication_inode_key(observed[0]) != _publication_inode_key(
                self.stage_identity
            ):
                raise MutationHostQualificationError(
                    "publication rollback alias settlement changed inode"
                )
            self.stage_identity, observed_target = observed
            if observed_target is not None:
                self.target_identity = observed_target
        try:
            os.fsync(self.final_parent_descriptor)
        except BaseException as exc:
            raise MutationHostLifecycleError(
                "publication rollback fsync outcome is uncertain"
            ) from exc
        self._verify_rolled_back_publication()
        if self.publication_kind == "replace":
            if self.target_descriptor is None:
                raise MutationHostLifecycleError(
                    "rolled-back target descriptor is unavailable"
                )
            try:
                os.fsync(self.target_descriptor)
            except BaseException as exc:
                raise MutationHostLifecycleError(
                    "rolled-back target file fsync outcome is uncertain"
                ) from exc
            self._verify_rolled_back_publication()
        self.phase = "rolled_back"

    def _rollback_publication(self) -> None:
        if self.phase == "create_attempted":
            self._settle_create_attempt()
        elif self.phase == "replace_attempted":
            self._settle_replace_attempt()
        elif self.phase == "rollback_replace_attempted":
            # Inspect the retained names before deciding whether another
            # exchange is necessary.
            pass
        elif self.phase in {
            "commit_create_attempted",
            "commit_replace_attempted",
            "committed_unsynced",
        }:
            self._finalize_publication()
            return
        elif self.phase == "rolled_back_unsynced":
            self._finish_rollback()
            return
        if self.phase == "provisional_create":
            self._rollback_create()
        elif self.phase in {
            "provisional_replace",
            "rollback_replace_attempted",
            "replace_restored",
        }:
            self._rollback_replace()

    def _cleanup_created_ancestors(self) -> None:
        for ancestor in reversed(self.ancestors):
            if not ancestor.created or ancestor.removed:
                continue
            descriptor = ancestor.descriptor
            if descriptor is None:
                raise MutationHostLifecycleError(
                    "created publication ancestor lost its descriptor"
                )
            removal_state = ancestor.removal_state
            if removal_state is not None and removal_state.active:
                _quarantine_remove_bound_entry(
                    ancestor.parent_descriptor,
                    ancestor.name,
                    descriptor,
                    directory=True,
                    token=self.token,
                    removal_state=removal_state,
                )
                if os.fstat(descriptor).st_nlink != 0:
                    raise MutationHostLifecycleError(
                        "publication ancestor has an unknown surviving name"
                    )
                ancestor.removal_unsynced = True
                observed = None
            else:
                observed = _named_entry_metadata(
                    ancestor.parent_descriptor,
                    ancestor.name,
                )
            if observed is None:
                if os.fstat(descriptor).st_nlink != 0:
                    raise MutationHostLifecycleError(
                        "created publication ancestor has an unknown surviving name"
                    )
                ancestor.removal_unsynced = True
            else:
                if _stable_directory_identity(observed) != ancestor.identity:
                    raise MutationHostLifecycleError(
                        "publication rollback refused a rebound ancestor"
                    )
                removal_state = _QuarantineRemovalState()
                ancestor.removal_state = removal_state
                _quarantine_remove_bound_entry(
                    ancestor.parent_descriptor,
                    ancestor.name,
                    descriptor,
                    directory=True,
                    token=self.token,
                    removal_state=removal_state,
                )
                if os.fstat(descriptor).st_nlink != 0:
                    raise MutationHostLifecycleError(
                        "publication ancestor has an unknown surviving name"
                    )
                ancestor.removal_unsynced = True
            try:
                os.fsync(ancestor.parent_descriptor)
            except BaseException as exc:
                raise MutationHostLifecycleError(
                    "publication ancestor removal fsync outcome is uncertain"
                ) from exc
            self._revalidate_remaining_ancestors(skip=ancestor)
            try:
                os.stat(
                    ancestor.name,
                    dir_fd=ancestor.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if os.fstat(descriptor).st_nlink != 0:
                    raise MutationHostLifecycleError(
                        "removed publication ancestor retained an unknown name"
                    ) from None
            else:
                raise MutationHostLifecycleError(
                    "removed publication ancestor name reappeared"
                )
            ancestor.removal_unsynced = False
            ancestor.creation_unsynced = False
            ancestor.removed = True

    def _revalidate_remaining_ancestors(
        self,
        *,
        skip: _PublicationAncestor | None = None,
    ) -> None:
        self.root.revalidate(require_named=True)
        for ancestor in self.ancestors:
            if ancestor is skip or ancestor.removed:
                continue
            descriptor = ancestor.descriptor
            if descriptor is None:
                raise MutationHostLifecycleError(
                    "publication ancestor descriptor is unavailable"
                )
            observed = os.stat(
                ancestor.name,
                dir_fd=ancestor.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(observed.st_mode)
                or _stable_directory_identity(observed) != ancestor.identity
                or _stable_directory_identity(os.fstat(descriptor)) != ancestor.identity
                or _descriptor_mount_id(descriptor) != ancestor.mount_id
            ):
                raise MutationHostQualificationError(
                    "publication ancestor changed during rollback cleanup"
                )

    def close(self, *, rollback: bool) -> None:
        if self.phase == "closed":
            return
        if self.descriptor_settlement_uncertain:
            raise MutationHostLifecycleError(
                "publication descriptor settlement remains uncertain"
            )
        if rollback:
            self._rollback_publication()
        else:
            self._finalize_publication()
        failures: list[BaseException] = []
        if rollback and self.phase not in {"committed", "committed_unsynced"}:
            if self.stage_acquisition_uncertain:
                failures.append(
                    MutationHostLifecycleError(
                        "publication stage descriptor acquisition remains uncertain"
                    )
                )
            elif self.stage_identity is not None and self.phase == "staged":
                try:
                    self._settle_unpublished_stage_removal()
                except (
                    MutationHostLifecycleError,
                    MutationHostQualificationError,
                ) as exc:
                    failures.append(exc)
            if self.stage_descriptor is not None and self.phase != "committed":
                try:
                    if os.fstat(self.stage_descriptor).st_nlink != 0:
                        failures.append(
                            MutationHostLifecycleError(
                                "publication stage has an unowned surviving link"
                            )
                        )
                except OSError as exc:
                    failures.append(exc)
        if self.pending_directory_creation is not None:
            pending = self.pending_directory_creation
            transferred = (
                pending.descriptor is None
                and not pending.descriptor_settlement_uncertain
                and any(
                    ancestor.created
                    and ancestor.descriptor is not None
                    and ancestor.parent_descriptor == pending.parent_descriptor
                    and ancestor.name == pending.final_name
                    and ancestor.identity == pending.identity
                    for ancestor in self.ancestors
                )
            )
            try:
                if not transferred:
                    pending.rollback()
                    self.ancestors[:] = [
                        ancestor
                        for ancestor in self.ancestors
                        if not (
                            ancestor.created
                            and ancestor.descriptor is None
                            and ancestor.parent_descriptor == pending.parent_descriptor
                            and ancestor.name == pending.final_name
                        )
                    ]
                self.pending_directory_creation = None
            except MutationHostLifecycleError as exc:
                failures.append(exc)
        if self.ancestor_acquisition_uncertain or self.target_acquisition_uncertain:
            failures.append(
                MutationHostLifecycleError(
                    "publication descriptor acquisition remains uncertain"
                )
            )
        if rollback and self.phase not in {"committed", "committed_unsynced"}:
            try:
                self._cleanup_created_ancestors()
            except MutationHostLifecycleError as exc:
                failures.append(exc)
        if failures:
            error = MutationHostLifecycleError(
                "parent-owned publication cleanup was incomplete"
            )
            for failure in failures[1:]:
                error.add_note(f"additional cleanup failure: {failure}")
            raise error from failures[0]
        close_failures: list[OSError] = []
        for attribute in ("target_descriptor", "stage_descriptor"):
            descriptor = getattr(self, attribute)
            if descriptor is None:
                continue
            setattr(self, attribute, None)
            try:
                os.close(descriptor)
            except OSError as exc:
                close_failures.append(exc)
        for ancestor in reversed(self.ancestors):
            descriptor = ancestor.descriptor
            if descriptor is None:
                continue
            ancestor.descriptor = None
            try:
                os.close(descriptor)
            except OSError as exc:
                close_failures.append(exc)
        if close_failures:
            self.descriptor_settlement_uncertain = True
            raise MutationHostLifecycleError(
                "publication descriptor settlement is uncertain"
            ) from close_failures[0]
        self.phase = "closed"


@dataclass(frozen=True, slots=True)
class _UncertainChildEffect:
    operation: str
    root_path: str
    relative_path: str


def _relative_entry_exists(
    roots: _MutationRootAuthorities,
    effect: _UncertainChildEffect,
) -> bool:
    root = next(
        (
            candidate
            for candidate in roots.roots
            if str(candidate.path) == effect.root_path
        ),
        None,
    )
    if root is None or root.descriptor is None:
        raise MutationHostLifecycleError(
            "uncertain child effect has no retained root authority"
        )
    root.revalidate(require_named=False)
    parts = PurePosixPath(effect.relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise MutationHostLifecycleError("uncertain child effect path is unsafe")
    descriptor = root.descriptor
    owns_descriptor = False
    try:
        for part in parts[:-1]:
            try:
                observed = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return False
            if not stat.S_ISDIR(observed.st_mode):
                return False
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except BaseException as exc:
                root.cleanup_acquisition_uncertain = True
                raise MutationHostLifecycleError(
                    "uncertain child-effect traversal is unresolved"
                ) from exc
            if (
                _stable_directory_identity(os.fstat(child))
                != _stable_directory_identity(observed)
                or _descriptor_mount_id(child) != root.expected_mount_id
            ):
                os.close(child)
                raise MutationHostLifecycleError(
                    "uncertain child-effect traversal changed identity or crossed a mount"
                )
            if owns_descriptor:
                prior = descriptor
                owns_descriptor = False
                os.close(prior)
            descriptor = child
            owns_descriptor = True
        try:
            os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        if owns_descriptor:
            acquired = descriptor
            owns_descriptor = False
            try:
                os.close(acquired)
            except OSError as exc:
                raise MutationHostLifecycleError(
                    "uncertain child-effect descriptor cleanup failed"
                ) from exc


def _unlink_pending_staging(
    roots: _MutationRootAuthorities,
    pending: _PendingStaging,
) -> None:
    root = next(
        (
            candidate
            for candidate in roots.roots
            if str(candidate.path) == pending.root_path
        ),
        None,
    )
    if root is None or root.descriptor is None:
        raise MutationHostLifecycleError(
            "pending staging has no retained root authority"
        )
    root.revalidate(require_named=False)
    parts = PurePosixPath(pending.relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise MutationHostLifecycleError("pending staging path is unsafe")
    descriptor = root.descriptor
    owns_descriptor = False
    try:
        for part in parts[:-1]:
            try:
                observed = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return
            if not stat.S_ISDIR(observed.st_mode):
                return
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except BaseException as exc:
                root.cleanup_acquisition_uncertain = True
                raise MutationHostLifecycleError(
                    "pending staging traversal acquisition is uncertain"
                ) from exc
            if (
                _stable_directory_identity(os.fstat(child))
                != _stable_directory_identity(observed)
                or _descriptor_mount_id(child) != root.expected_mount_id
            ):
                os.close(child)
                raise MutationHostLifecycleError(
                    "pending staging traversal changed identity or crossed a mount"
                )
            if owns_descriptor:
                prior = descriptor
                owns_descriptor = False
                os.close(prior)
            descriptor = child
            owns_descriptor = True
        try:
            observed = os.stat(
                parts[-1],
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise MutationHostLifecycleError(
                "pending staging target is not one regular file"
            )
        try:
            leaf_descriptor = os.open(
                parts[-1],
                os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
        except BaseException as exc:
            root.cleanup_acquisition_uncertain = True
            raise MutationHostLifecycleError(
                "pending staging leaf acquisition is uncertain"
            ) from exc
        try:
            if _stat_identity(os.fstat(leaf_descriptor)) != _stat_identity(observed):
                raise MutationHostLifecycleError(
                    "pending staging target changed while binding"
                )
            _quarantine_remove_bound_entry(
                descriptor,
                parts[-1],
                leaf_descriptor,
                directory=False,
                token=secrets.token_hex(16),
            )
        finally:
            try:
                os.close(leaf_descriptor)
            except OSError as exc:
                raise MutationHostLifecycleError(
                    "pending staging leaf descriptor close failed"
                ) from exc
        os.fsync(descriptor)
    finally:
        if owns_descriptor:
            acquired = descriptor
            owns_descriptor = False
            try:
                os.close(acquired)
            except OSError as exc:
                raise MutationHostLifecycleError(
                    "pending staging descriptor cleanup failed"
                ) from exc


class IsolatedMutationHost:
    """One externally reapable mutation process for a backend attempt."""

    def __init__(
        self,
        spec: IsolatedToolSetSpec,
        *,
        invocation_deadline: float,
        _before_start_ack: Callable[[], None] | None = None,
        _before_publish_ack: Callable[[], None] | None = None,
        _before_quarantine_ack: Callable[[], None] | None = None,
    ) -> None:
        if (
            isinstance(invocation_deadline, bool)
            or not isinstance(invocation_deadline, (int, float))
            or not math.isfinite(float(invocation_deadline))
        ):
            raise ValueError("invocation_deadline must be finite")
        self._spec = spec
        self._deadline = float(invocation_deadline)
        self._before_start_ack = _before_start_ack
        self._before_publish_ack = _before_publish_ack
        self._before_quarantine_ack = _before_quarantine_ack
        self._state_lock = threading.Lock()
        self._stop_lock = threading.RLock()
        self._delete_lifecycle_lock = threading.RLock()
        self._call_lock = threading.Lock()
        self._transaction_lock = threading.RLock()
        self._process: _PosixSpawnProcess | None = None
        self._private_root: Path | None = None
        self._private_root_authority: _PrivateRootAuthority | None = None
        self._stderr_descriptor: int | None = None
        self._stderr_acquisition_uncertain = False
        self._source_authority: _ChildSourceAuthority | None = None
        self._mutation_roots: _MutationRootAuthorities | None = None
        self._read_buffer = bytearray()
        self._active_call = False
        self._ever_started_write = False
        self._started_write_names: list[str] = []
        self._stopped = False
        self._settled = False
        self._started_once = False
        self._start_in_progress = False
        self._pending_staging: set[_PendingStaging] = set()
        self._uncertain_child_effects: set[_UncertainChildEffect] = set()
        self._active_transaction: _ParentPublicationTransaction | None = None
        self._deferred_interrupt: KeyboardInterrupt | None = None
        self._delete_receipts: dict[
            _PendingStaging,
            _DeleteReceiptAuthority,
        ] = {}
        self._repository_roots: dict[str, _BoundRepositoryRoot] = {}

    @property
    def ever_started_write(self) -> bool:
        with self._state_lock:
            return self._ever_started_write

    @property
    def tool_names(self) -> frozenset[str]:
        return self._spec.names

    @property
    def started_write_names(self) -> tuple[str, ...]:
        with self._state_lock:
            return tuple(self._started_write_names)

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            process = self._process
        return None if process is None else process.pid

    @property
    def is_reaped(self) -> bool:
        with self._state_lock:
            process = self._process
            if self._settled:
                return True
            if (
                self._start_in_progress
                or process is not None
                or self._private_root_authority is not None
                or self._source_authority is not None
                or self._mutation_roots is not None
                or self._stderr_descriptor is not None
                or self._stderr_acquisition_uncertain
                or self._uncertain_child_effects
                or self._active_transaction is not None
            ):
                return False
            return not self._started_once

    def start(self) -> None:
        # Serialize creation/attachment with stop(). A stop racing the
        # posix_spawn window must never observe "no process", mark the host
        # settled, and then let a newly spawned child escape ownership.
        with self._stop_lock:
            with self._state_lock:
                beginning = self._process is None and not self._stopped
                if beginning:
                    self._started_once = True
                    self._start_in_progress = True
            try:
                self._start_locked()
            finally:
                if beginning:
                    with self._state_lock:
                        self._start_in_progress = False

    def _start_locked(self) -> None:
        with self._state_lock:
            if self._stopped:
                raise MutationHostLifecycleError(
                    "isolated mutation host cannot restart after stop"
                )
            if self._process is not None:
                return
        _remaining(self._deadline)
        mutation_roots = _MutationRootAuthorities.from_spec(self._spec)
        with self._state_lock:
            self._mutation_roots = mutation_roots
        private_root_authority = _provisional_private_root_authority()
        with self._state_lock:
            self._private_root_authority = private_root_authority
        try:
            mutation_roots.acquire_all()
            mutation_roots.revalidate(require_named=True)
            self._bind_repository_roots()
            _open_private_root_authority(private_root_authority)
            private_root_authority.revalidate(require_named_root=True)
            private_root = private_root_authority.root_path
            root_descriptor = private_root_authority.root_descriptor
            if private_root is None or root_descriptor is None:
                raise MutationHostQualificationError(
                    "isolated mutation private-root acquisition was incomplete"
                )
            with self._state_lock:
                self._private_root = private_root
            # Bind only the immutable source captured when this already-loaded
            # parent module initialized.  No executable path is reopened here.
            source_authority = _provisional_child_source_authority()
            with self._state_lock:
                self._source_authority = source_authority
            _open_child_source_authority(source_authority)
            # The child executes only the already-loaded, descriptor-attested
            # host module. It has no package path on sys.path, and its canonical
            # writers/path resolver are defined below in this same source.
            stderr_descriptor: int | None = None
            try:
                stderr_descriptor = os.open(
                    ".child-stderr",
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=root_descriptor,
                )
            except BaseException as exc:
                with self._state_lock:
                    if stderr_descriptor is not None:
                        self._stderr_descriptor = stderr_descriptor
                    else:
                        self._stderr_acquisition_uncertain = True
                raise MutationHostLifecycleError(
                    "stderr descriptor acquisition outcome is uncertain"
                ) from exc
            assert stderr_descriptor is not None
            with self._state_lock:
                self._stderr_descriptor = stderr_descriptor
            # Index only explicit non-secret names. Iterating os.environ.items()
            # would retrieve every provider credential value before filtering
            # it, violating pre-inference credential non-observation.
            environment = {
                key: os.environ[key]
                for key in ("LANG", "LC_ALL", "LC_CTYPE", "PATH", "TZ")
                if key in os.environ
            }
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment[_CHILD_CWD_ENV] = str(private_root)
            command = [
                sys.executable,
                "-I",
                "-B",
                "-S",
                "-c",
                _CHILD_BOOTSTRAP,
                "--child",
            ]
            expected_sid = os.getsid(0)
            source_authority.revalidate()
            private_root_authority.revalidate(require_named_root=True)
            mutation_roots.revalidate(require_named=True)
            process = _PosixSpawnProcess()
            with self._state_lock:
                # Attach all raw pipe/PID authority before pipe creation and
                # before posix_spawn can publish a live process.  start/stop's
                # outer lock prevents a concurrent observer from settling this
                # deliberately provisional object midway through population.
                self._process = process
            assert _SOURCE_IMAGE_LOCK is not None
            with _SOURCE_IMAGE_LOCK:
                source_authority.revalidate()
                _posix_spawn_child(
                    process,
                    command,
                    environment,
                    stderr_descriptor,
                    source_authority,
                    private_root_authority,
                    mutation_roots,
                )
            spawned_pid = process.pid
            if spawned_pid is None:
                raise MutationHostLifecycleError(
                    "isolated mutation spawn returned without PID authority"
                )
            process.prepare_parent_pipes()
        except BaseException as primary_error:
            with self._state_lock:
                # Any failed acquisition is terminal for this host instance.
                # In particular, provisional descriptor/root authority may
                # still be attached for stop() to settle; a second start must
                # never overwrite and orphan that retained authority.
                self._stopped = True
                attached_process = self._process
            try:
                if attached_process is None:
                    self._close_resources(None)
                    with self._state_lock:
                        self._stopped = True
                        self._settled = True
                else:
                    self.stop()
            except MutationHostLifecycleError as cleanup_error:
                error = MutationHostLifecycleError(
                    "isolated mutation startup cleanup failed "
                    + (
                        "before process attachment"
                        if attached_process is None
                        else "with retained process authority"
                    )
                )
                error.add_note(f"cleanup failure: {cleanup_error}")
                raise error from primary_error
            if isinstance(primary_error, OSError):
                raise MutationHostQualificationError(
                    "isolated mutation startup authority is unavailable"
                ) from primary_error
            raise
        with self._state_lock:
            stopped_during_start = self._stopped
        if stopped_during_start:
            # Publish ownership before attempting settlement. If any OS
            # lifecycle operation fails, stop() retains this exact process,
            # its pipes, and its private root for a later retry.
            self.stop()
            raise MutationHostLifecycleError(
                "isolated mutation host was stopped during startup"
            )
        try:
            self._write_frame(
                {
                    "type": "initialize",
                    "spec": self._spec.to_wire(),
                }
            )
            try:
                ready = self._read_frame(
                    min(
                        self._deadline,
                        time.monotonic() + _START_GRACE_SECONDS,
                    )
                )
            except BaseException as ready_error:
                try:
                    source_authority.revalidate()
                except MutationHostQualificationError as source_error:
                    raise source_error from ready_error
                raise
            source_authority.revalidate()
            private_root_authority.revalidate(require_named_root=True)
            mutation_roots.revalidate(require_named=True)
            expected_ready = source_authority.expected_ready_frame(
                pid=spawned_pid,
                sid=expected_sid,
            )
            expected_ready.update(private_root_authority.ready_fields())
            expected_ready["mutation_roots"] = mutation_roots.wire_manifest()
            if ready != expected_ready:
                raise MutationHostProtocolError(
                    "isolated mutation child did not attest readiness"
                )
        except BaseException:
            self.stop()
            raise

    def _bind_repository_roots(self) -> None:
        """Alias each repository to the stronger retained mutation-root fd."""

        with self._state_lock:
            if self._repository_roots:
                return
            mutation_roots = self._mutation_roots
        if mutation_roots is None:
            raise MutationHostLifecycleError(
                "repository roots have no retained mutation-root authority"
            )
        mutation_roots.revalidate(require_named=True)
        authorities = {str(root.path): root for root in mutation_roots.roots}
        roots: dict[str, _BoundRepositoryRoot] = {}
        for rendered in sorted({
            binding.context.repo_root for binding in self._spec.bindings
        }):
            authority = authorities.get(rendered)
            if authority is None or authority.descriptor is None:
                raise MutationHostLifecycleError(
                    "repository root lacks retained descriptor authority"
                )
            observed = os.fstat(authority.descriptor)
            roots[rendered] = _BoundRepositoryRoot(
                descriptor=authority.descriptor,
                device=observed.st_dev,
                inode=observed.st_ino,
                mount_id=authority.expected_mount_id,
            )
        with self._state_lock:
            self._repository_roots = roots

    def _close_repository_roots(self) -> None:
        """Drop non-owning aliases after every delete receipt is settled."""

        with self._state_lock:
            if self._delete_receipts:
                raise MutationHostLifecycleError(
                    "repository-root aliases still own delete receipts"
                )
            self._repository_roots = {}

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute one canonical mutation and return its tool-result string."""

        if name not in self._spec.names:
            raise MutationHostProtocolError(
                f"isolated mutation dispatch rejected unknown tool {name!r}"
            )
        if not isinstance(arguments, dict):
            raise MutationHostProtocolError(
                "isolated mutation arguments must be an object"
            )
        binding = self._spec.binding_for(name)
        if binding.kind == "delete_source" and (
            binding.context.mutation_journal_dir is None
            or binding.context.mutation_journal_transaction_id is None
        ):
            return (
                "Error: delete_source requires active durable "
                "mutation-journal authority"
            )
        self.start()
        with self._call_lock:
            _remaining(self._deadline)
            with self._state_lock:
                if self._stopped:
                    raise MutationHostLifecycleError(
                        "isolated mutation host is stopped"
                    )
                mutation_roots = self._mutation_roots
            if mutation_roots is None:
                raise MutationHostLifecycleError(
                    "isolated mutation roots are unavailable"
                )
            mutation_roots.revalidate(require_named=True)
            with self._state_lock:
                self._active_call = True
            call_id = secrets.token_hex(16)
            publication_token = secrets.token_hex(_PUBLICATION_TOKEN_HEX_LENGTH // 2)
            binding = self._spec.binding_for(name)
            target = (
                binding.exact_target
                if binding.kind == "exact_write_source"
                else arguments.get("file_path")
            )
            normalized_target = (
                None
                if not isinstance(target, str) or binding.kind == "delete_source"
                else _resolve_for_write(binding.context, target)
            )
            staging = (
                self._staging_path(binding, arguments, publication_token)
                if binding.kind == "delete_source"
                else None
            )
            staging_registered = False
            terminal_received = False
            try:
                if staging is not None and binding.kind == "delete_source":
                    self._arm_delete_staging(
                        staging,
                        binding,
                        publication_token,
                    )
                    # _arm_delete_staging registers cleanup authority before
                    # its first durable CAS.  That ordering closes the
                    # arm-to-host-map interruption window.
                    staging_registered = True
                elif staging is not None:
                    with self._state_lock:
                        self._pending_staging.add(staging)
                    staging_registered = True
                self._write_frame(
                    {
                        "type": "call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                        "publication_token": publication_token,
                    }
                )
                accepted = self._read_frame(self._deadline)
                self._expect_call_frame(accepted, "accepted", call_id, name)
                self._write_frame(
                    {
                        "type": "execute",
                        "call_id": call_id,
                        "name": name,
                    }
                )
                started = self._read_frame(self._deadline)
                self._expect_call_frame(started, "started", call_id, name)
                with self._state_lock:
                    self._ever_started_write = True
                    self._started_write_names.append(name)
                if self._before_start_ack is not None:
                    self._before_start_ack()
                _remaining(self._deadline)
                self._write_frame(
                    {
                        "type": "ack_started",
                        "call_id": call_id,
                        "name": name,
                    }
                )
                terminal = self._read_frame(self._deadline)
                if (
                    terminal.get("type") == "publish_ready"
                    and binding.kind == "delete_source"
                ):
                    if set(terminal) != {"type", "call_id", "name"}:
                        raise MutationHostProtocolError(
                            "delete child emitted malformed publication authority"
                        )
                    self._expect_call_frame(
                        terminal,
                        "publish_ready",
                        call_id,
                        name,
                    )
                    if staging is None or not staging_registered:
                        raise MutationHostProtocolError(
                            "delete child requested publication without durable "
                            "staging authority"
                        )
                    self._prepare_delete_quarantine(staging)
                    if self._before_publish_ack is not None:
                        self._before_publish_ack()
                    self._reprove_delete_barrier(
                        staging,
                        expect_payload=False,
                    )
                    _remaining(self._deadline)
                    self._write_frame(
                        {
                            "type": "ack_publish",
                            "call_id": call_id,
                            "name": name,
                        }
                    )
                    terminal = self._read_frame(self._deadline)
                if terminal.get("type") == "publish_ready":
                    if (
                        set(terminal)
                        != {
                            "type",
                            "call_id",
                            "name",
                            "content",
                            "expected_current_sha256",
                        }
                        or terminal.get("call_id") != call_id
                        or terminal.get("name") != name
                        or not isinstance(terminal.get("content"), str)
                        or (
                            terminal.get("expected_current_sha256") is not None
                            and (
                                not isinstance(
                                    terminal.get("expected_current_sha256"),
                                    str,
                                )
                                or len(terminal["expected_current_sha256"]) != 64
                                or any(
                                    character not in "0123456789abcdef"
                                    for character in terminal["expected_current_sha256"]
                                )
                            )
                        )
                        or normalized_target is None
                    ):
                        raise MutationHostProtocolError(
                            "isolated mutation child emitted malformed publication authority"
                        )

                    def attach_transaction(
                        transaction: _ParentPublicationTransaction,
                    ) -> None:
                        with self._state_lock:
                            if self._active_transaction is not None:
                                raise MutationHostLifecycleError(
                                    "isolated mutation publication already exists"
                                )
                            self._active_transaction = transaction

                    with self._transaction_lock:
                        transaction = _ParentPublicationTransaction.prepare(
                            mutation_roots,
                            binding.context,
                            normalized_target,
                            publication_token,
                            attach=attach_transaction,
                        )
                        stage_sha256 = transaction.write_stage(terminal["content"])
                        if self._before_publish_ack is not None:
                            self._before_publish_ack()
                        mutation_roots.revalidate(require_named=True)
                        _remaining(self._deadline)
                        self._write_frame(
                            {
                                "type": "prepare_publish",
                                "call_id": call_id,
                                "name": name,
                            }
                        )
                        publish_alive = self._read_frame(self._deadline)
                        self._expect_call_frame(
                            publish_alive,
                            "publish_alive",
                            call_id,
                            name,
                        )
                        with self._state_lock:
                            live_process = self._process
                        if live_process is None or live_process.poll() is not None:
                            raise MutationHostProtocolError(
                                "isolated mutation child died before parent publication"
                            )
                        publication_error: str | None = None
                        try:
                            transaction.publish(
                                stage_sha256=stage_sha256,
                                expected_current_sha256=terminal[
                                    "expected_current_sha256"
                                ],
                            )
                        except ValueError as exc:
                            transaction.close(rollback=True)
                            with self._state_lock:
                                if self._active_transaction is transaction:
                                    self._active_transaction = None
                            prefix = (
                                "Error writing patched"
                                if binding.kind == "apply_patch"
                                else "Error writing"
                            )
                            publication_error = f"{prefix} {target}: {exc}"
                        self._write_frame(
                            {
                                "type": "ack_publish",
                                "call_id": call_id,
                                "name": name,
                            }
                        )
                        terminal = self._read_frame(self._deadline)
                        if publication_error is not None:
                            self._expect_call_frame(
                                terminal,
                                "result",
                                call_id,
                                name,
                            )
                            terminal = {
                                "type": "result",
                                "call_id": call_id,
                                "name": name,
                                "result": publication_error,
                            }
                if terminal.get("type") == "lifecycle_uncertain":
                    expected_fields = {
                        "type",
                        "call_id",
                        "name",
                        "operation",
                        "root_path",
                        "relative_path",
                    }
                    if (
                        set(terminal) != expected_fields
                        or terminal.get("call_id") != call_id
                        or terminal.get("name") != name
                        or terminal.get("operation") not in {"open", "mkdir"}
                        or not isinstance(terminal.get("root_path"), str)
                        or not isinstance(terminal.get("relative_path"), str)
                    ):
                        raise MutationHostProtocolError(
                            "isolated mutation child emitted malformed lifecycle uncertainty"
                        )
                    effect = _UncertainChildEffect(
                        operation=str(terminal["operation"]),
                        root_path=str(terminal["root_path"]),
                        relative_path=str(terminal["relative_path"]),
                    )
                    with self._state_lock:
                        self._uncertain_child_effects.add(effect)
                    raise MutationHostLifecycleError(
                        "isolated mutation child acquisition is uncertain"
                    )
                if terminal.get("type") == "quarantine_ready":
                    if binding.kind != "delete_source" or staging is None:
                        raise MutationHostProtocolError(
                            "unexpected delete quarantine barrier"
                        )
                    self._expect_call_frame(
                        terminal,
                        "quarantine_ready",
                        call_id,
                        name,
                    )
                    if self._before_quarantine_ack is not None:
                        self._before_quarantine_ack()
                    self._reprove_delete_barrier(
                        staging,
                        expect_payload=True,
                    )
                    _remaining(self._deadline)
                    self._write_frame(
                        {
                            "type": "ack_quarantine",
                            "call_id": call_id,
                            "name": name,
                        }
                    )
                    terminal = self._read_frame(self._deadline)
                if terminal.get("type") == "terminal_mutation":
                    self._expect_call_frame(
                        terminal,
                        "terminal_mutation",
                        call_id,
                        name,
                    )
                    message = terminal.get("message")
                    if not isinstance(message, str) or not message.strip():
                        raise MutationHostProtocolError(
                            "isolated mutation terminal omitted its message"
                        )
                    raise ToolMutationIndeterminate(message)
                self._expect_call_frame(terminal, "result", call_id, name)
                result = terminal.get("result")
                if not isinstance(result, str):
                    raise MutationHostProtocolError(
                        "isolated mutation result must be a string"
                    )
                with self._transaction_lock:
                    with self._state_lock:
                        active_transaction = self._active_transaction
                    if active_transaction is not None:
                        active_transaction.close(rollback=False)
                        with self._state_lock:
                            if self._active_transaction is active_transaction:
                                self._active_transaction = None
                terminal_received = True
                return result
            except ToolMutationIndeterminate:
                try:
                    self.stop()
                except BaseException as settlement_exc:
                    raise ToolMutationIndeterminate(
                        "indeterminate source mutation could not be settled; "
                        "provider continuation is forbidden"
                    ) from settlement_exc
                raise
            except BaseException:
                self.stop()
                raise
            finally:
                with self._state_lock:
                    self._active_call = False
                if staging is not None and staging_registered:
                    cleanup_error: BaseException | None = None
                    # Keep the eligibility proof and cleanup in the same
                    # receipt-lifecycle critical section as stop().  Otherwise
                    # stop could settle and close root authority between this
                    # check and the cleanup call.
                    with self._delete_lifecycle_lock:
                        with self._state_lock:
                            safe_to_clean = (
                                staging in self._pending_staging
                                and (terminal_received or self._settled)
                            )
                        if safe_to_clean:
                            try:
                                self._cleanup_staging_locked(staging)
                            except BaseException as exc:
                                cleanup_error = exc
                    if cleanup_error is not None:
                        # Respect the global stop->delete lock order. Cleanup
                        # can observe an asynchronous post-CAS exception, but
                        # stop() must be called only after releasing the
                        # delete lifecycle lock or an external stop can invert
                        # those locks and deadlock this dispatch.
                        try:
                            self.stop()
                        except BaseException as settlement:
                            raise settlement from cleanup_error
                        raise cleanup_error

    @staticmethod
    def _expect_call_frame(
        payload: Mapping[str, Any],
        kind: str,
        call_id: str,
        name: str,
    ) -> None:
        if (
            payload.get("type") != kind
            or payload.get("call_id") != call_id
            or payload.get("name") != name
        ):
            raise MutationHostProtocolError(
                f"isolated mutation child emitted mismatched {kind} frame"
            )

    def _staging_path(
        self,
        binding: MutationToolBinding,
        arguments: Mapping[str, Any],
        token: str,
    ) -> _PendingStaging | None:
        target = (
            binding.exact_target
            if binding.kind == "exact_write_source"
            else arguments.get("file_path")
        )
        if not isinstance(target, str):
            return None
        if binding.kind == "delete_source":
            normalized = _exact_declared_delete_target(binding.context, target)
            if normalized is None:
                return None
            _validate_publication_token(token)
            repository_root = binding.context.repo_root
            candidate = PurePosixPath(normalized)
            quarantine_name = (
                f".{candidate.name}.crucible-delete-{token}.quarantine"
            )
            quarantine_relative = (
                quarantine_name
                if candidate.parent.as_posix() == "."
                else f"{candidate.parent.as_posix()}/{quarantine_name}"
            )
            return _PendingStaging(
                root_path=repository_root,
                relative_path=quarantine_relative,
                restore_target=Path(repository_root) / normalized,
                repository_root=repository_root,
                parent_components=candidate.parts[:-1],
                target_relative=normalized,
            )
        normalized = _resolve_for_write(binding.context, target)
        if normalized is None:
            return None
        candidate = PurePosixPath(normalized)
        stage_name = f".{candidate.name}.crucible-{token}.tmp"
        stage_relative = (
            stage_name
            if candidate.parent.as_posix() == "."
            else f"{candidate.parent.as_posix()}/{stage_name}"
        )
        return _PendingStaging(
            root_path=binding.context.repo_root,
            relative_path=stage_relative,
        )

    def _bound_root_for_delete(
        self,
        pending: _PendingStaging,
    ) -> _BoundRepositoryRoot:
        repository_root = pending.repository_root
        if repository_root is None:
            raise MutationHostLifecycleError(
                "delete staging has no repository-root authority"
            )
        with self._state_lock:
            bound_root = self._repository_roots.get(repository_root)
        if bound_root is None:
            raise MutationHostLifecycleError(
                "delete staging lost its held repository root"
            )
        current = os.fstat(bound_root.descriptor)
        if (current.st_dev, current.st_ino) != (
            bound_root.device,
            bound_root.inode,
        ):
            raise MutationHostLifecycleError(
                "delete staging repository root changed identity"
            )
        return bound_root

    def _arm_delete_staging(
        self,
        pending: _PendingStaging,
        binding: MutationToolBinding,
        token: str,
    ) -> _DeleteReceiptAuthority:
        """Durably reserve the exact target/qdir mapping before child work."""

        with self._delete_lifecycle_lock:
            return self._arm_delete_staging_locked(
                pending,
                binding,
                token,
            )

    def _arm_delete_staging_locked(
        self,
        pending: _PendingStaging,
        binding: MutationToolBinding,
        token: str,
    ) -> _DeleteReceiptAuthority:
        """Arm a receipt while holding the complete delete lifecycle."""

        with self._state_lock:
            if self._stopped:
                raise MutationHostLifecycleError(
                    "isolated mutation host stopped before delete arm"
                )

        journal_dir = binding.context.mutation_journal_dir
        transaction_id = binding.context.mutation_journal_transaction_id
        target_relative = pending.target_relative
        if (
            journal_dir is None
            or transaction_id is None
            or target_relative is None
        ):
            raise MutationHostLifecycleError(
                "delete staging lacks durable mutation-journal authority"
            )
        bound_root = self._bound_root_for_delete(pending)
        try:
            from crucible.workflows._mutation_journal import (  # noqa: PLC0415
                resolve_live_mutation_journal,
            )

            journal = resolve_live_mutation_journal(
                journal_dir,
                transaction_id,
            )
        except Exception as exc:
            raise MutationHostLifecycleError(
                "delete receipt authority could not be resolved"
            ) from exc
        try:
            quarantine_relative = pending.path.relative_to(
                Path(pending.repository_root)
            ).as_posix()
        except ValueError as exc:
            raise MutationHostLifecycleError(
                "delete quarantine escaped its repository authority"
            ) from exc
        boot_id = _kernel_boot_id()
        expected_receipt: dict[str, object] = {
            "token": token,
            "target_relative": target_relative,
            "quarantine_relative": quarantine_relative,
            "parent_chain": [],
            "state": "reserved",
            "quarantine_device": None,
            "quarantine_inode": None,
            "quarantine_mode": None,
            "quarantine_mount_id": None,
            "child_pid": None,
            "child_start_ticks": None,
            "boot_id": boot_id,
        }
        authority = _DeleteReceiptAuthority(
            journal=journal,
            receipt=expected_receipt,
            parent_acquisition_in_progress=True,
        )
        # Register before dup/open.  An asynchronous exception can arrive
        # after the kernel allocates a descriptor but before Python returns
        # its integer; stop() must retain explicit uncertainty in that window.
        with self._state_lock:
            if (
                pending in self._delete_receipts
                or pending in self._pending_staging
            ):
                raise MutationHostLifecycleError(
                    "delete staging authority is already registered"
                )
            self._delete_receipts[pending] = authority
            self._pending_staging.add(pending)
        try:
            parent_descriptor, parent_chain = _open_delete_parent_chain(
                bound_root.descriptor,
                pending.parent_components,
                expected_mount_id=bound_root.mount_id,
            )
        except BaseException as exc:
            with self._state_lock:
                authority.parent_acquisition_in_progress = False
                authority.parent_acquisition_uncertain = True
            raise MutationHostLifecycleError(
                "delete receipt parent-chain acquisition is indeterminate"
            ) from exc
        with self._state_lock:
            if self._delete_receipts.get(pending) is not authority:
                authority.parent_acquisition_uncertain = True
                raise MutationHostLifecycleError(
                    "delete receipt owner changed during parent acquisition"
                )
            authority.parent_descriptor = parent_descriptor
            authority.receipt = {
                **expected_receipt,
                "parent_chain": [list(identity) for identity in parent_chain],
            }
            authority.parent_acquisition_in_progress = False
        try:
            self._ensure_delete_receipt_armed(pending, authority)
        except BaseException as primary:
            # The first CAS may have published before an asynchronous
            # exception became visible.  Exact-same reserved arm is
            # idempotent, so retry once here to settle that ambiguity.  If
            # this retry is interrupted too, stop() retains and retries the
            # same registered authority before releasing root descriptors.
            try:
                self._ensure_delete_receipt_armed(pending, authority)
            except BaseException as settlement:
                error = MutationHostLifecycleError(
                    "delete receipt arm could not be settled"
                )
                error.add_note(
                    "initial arm failed after "
                    f"{type(primary).__name__}: {primary}"
                )
                raise error from settlement
            raise
        return authority

    def _ensure_delete_receipt_armed(
        self,
        pending: _PendingStaging,
        authority: _DeleteReceiptAuthority,
    ) -> None:
        """Prove one pre-registered reserved receipt is durable and exact."""

        with self._delete_lifecycle_lock:
            self._ensure_delete_receipt_armed_locked(pending, authority)

    def _ensure_delete_receipt_armed_locked(
        self,
        pending: _PendingStaging,
        authority: _DeleteReceiptAuthority,
    ) -> None:
        """Arm one receipt under the host delete-lifecycle lock."""

        with self._state_lock:
            current = self._delete_receipts.get(pending)
            if current is not authority:
                raise MutationHostLifecycleError(
                    "delete receipt owner changed during arm"
                )
            if authority.durable:
                return
            expected = dict(authority.receipt)
        target_relative = expected.get("target_relative")
        token = expected.get("token")
        if not isinstance(target_relative, str) or not isinstance(token, str):
            raise MutationHostLifecycleError(
                "delete receipt arm authority is malformed"
            )
        bound_root = self._bound_root_for_delete(pending)
        receipt = authority.journal.arm_pending_delete_quarantine(
            token=token,
            target_relative=target_relative,
            parent_chain=expected.get("parent_chain"),
            boot_id=expected.get("boot_id"),
            workspace_root_descriptor=bound_root.descriptor,
        )
        if receipt != expected:
            raise MutationHostLifecycleError(
                "delete receipt differs from isolated staging authority"
            )
        with self._state_lock:
            current = self._delete_receipts.get(pending)
            if current is not authority:
                raise MutationHostLifecycleError(
                    "delete receipt owner changed during arm"
                )
            authority.receipt = receipt
            authority.durable = True

    def _prepare_delete_quarantine(
        self,
        pending: _PendingStaging,
    ) -> None:
        """Create and durably bind the empty qdir before child publication."""

        with self._delete_lifecycle_lock:
            self._prepare_delete_quarantine_locked(pending)

    def _reprove_delete_parent_chain_locked(
        self,
        pending: _PendingStaging,
        authority: _DeleteReceiptAuthority,
        bound_root: _BoundRepositoryRoot,
    ) -> None:
        """Rewalk one named chain with explicit provisional fd ownership."""

        with self._state_lock:
            roots = self._mutation_roots
            held_parent = authority.parent_descriptor
            if (
                roots is None
                or held_parent is None
                or authority.reproof_descriptor is not None
                or authority.reproof_acquisition_in_progress
                or authority.reproof_acquisition_uncertain
            ):
                raise MutationHostLifecycleError(
                    "delete parent reproof authority is incomplete"
                )
        roots.revalidate(require_named=True)
        with self._state_lock:
            authority.reproof_acquisition_in_progress = True
        try:
            current_parent, current_chain = _open_delete_parent_chain(
                bound_root.descriptor,
                pending.parent_components,
                expected_mount_id=bound_root.mount_id,
            )
        except BaseException as exc:
            with self._state_lock:
                authority.reproof_acquisition_in_progress = False
                authority.reproof_acquisition_uncertain = True
            raise MutationHostLifecycleError(
                "delete parent reproof acquisition is indeterminate"
            ) from exc
        with self._state_lock:
            if self._delete_receipts.get(pending) is not authority:
                authority.reproof_acquisition_uncertain = True
                raise MutationHostLifecycleError(
                    "delete receipt owner changed during parent reproof"
                )
            authority.reproof_descriptor = current_parent
            authority.reproof_acquisition_in_progress = False
        try:
            if authority.receipt.get("parent_chain") != [
                list(identity) for identity in current_chain
            ]:
                raise MutationHostLifecycleError(
                    "delete named parent chain changed"
                )
            if (
                _delete_directory_chain_identity(current_parent)
                != _delete_directory_chain_identity(held_parent)
            ):
                raise MutationHostLifecycleError(
                    "delete named chain no longer reaches retained parent"
                )
            roots.revalidate(require_named=True)
        finally:
            with self._state_lock:
                authority.reproof_close_in_progress = True
                if authority.reproof_descriptor == current_parent:
                    authority.reproof_descriptor = None
            try:
                os.close(current_parent)
            except BaseException as exc:
                with self._state_lock:
                    authority.reproof_close_in_progress = False
                    authority.reproof_acquisition_uncertain = True
                raise MutationHostLifecycleError(
                    "delete parent reproof descriptor close is indeterminate"
                ) from exc
            with self._state_lock:
                authority.reproof_close_in_progress = False

    def _prepare_delete_quarantine_locked(
        self,
        pending: _PendingStaging,
    ) -> None:
        """Create and bind a qdir under the complete delete lifecycle."""

        with self._state_lock:
            authority = self._delete_receipts.get(pending)
            process = self._process
            stopped = self._stopped
        if stopped:
            raise MutationHostLifecycleError(
                "isolated mutation host stopped before delete publication"
            )
        if authority is None or not authority.durable or process is None:
            raise MutationHostLifecycleError(
                "delete quarantine has no live receipt or child owner"
            )
        bound_root = self._bound_root_for_delete(pending)
        if pending.restore_target is None or (
            pending.path.parent != pending.restore_target.parent
        ):
            raise MutationHostLifecycleError(
                "delete quarantine is not adjacent to its public target"
            )

        try:
            self._reprove_delete_parent_chain_locked(
                pending,
                authority,
                bound_root,
            )
            held_parent = authority.parent_descriptor
            if held_parent is None:
                raise MutationHostLifecycleError(
                    "delete quarantine lost its retained parent"
                )
            target_name = pending.restore_target.name
            target = os.stat(
                target_name,
                dir_fd=held_parent,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(target.st_mode) or not (
                stat.S_ISREG(target.st_mode) or stat.S_ISDIR(target.st_mode)
            ):
                raise MutationHostLifecycleError(
                    "delete target changed before quarantine preparation"
                )
            with self._state_lock:
                current = self._delete_receipts.get(pending)
                if current is not authority:
                    raise MutationHostLifecycleError(
                        "delete parent descriptor owner changed during binding"
                    )
                authority.target_identity = _entry_ownership_key(target)
            os.mkdir(pending.path.name, mode=0o700, dir_fd=held_parent)
            with self._state_lock:
                if authority.quarantine_descriptor is not None:
                    raise MutationHostLifecycleError(
                        "delete qdir descriptor is already attached"
                    )
                authority.quarantine_acquisition_in_progress = True
            try:
                quarantine_fd = os.open(
                    pending.path.name,
                    os.O_RDONLY
                    | os.O_CLOEXEC
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW,
                    dir_fd=held_parent,
                )
            except BaseException as exc:
                with self._state_lock:
                    authority.quarantine_acquisition_in_progress = False
                    authority.quarantine_acquisition_uncertain = True
                raise MutationHostLifecycleError(
                    "delete qdir descriptor acquisition is indeterminate"
                ) from exc
            with self._state_lock:
                if self._delete_receipts.get(pending) is not authority:
                    authority.quarantine_acquisition_uncertain = True
                    raise MutationHostLifecycleError(
                        "delete receipt owner changed during qdir acquisition"
                    )
                authority.quarantine_descriptor = quarantine_fd
                authority.quarantine_acquisition_in_progress = False
            os.fchmod(quarantine_fd, 0o700)
            held = os.fstat(quarantine_fd)
            named = os.stat(
                pending.path.name,
                dir_fd=held_parent,
                follow_symlinks=False,
            )
            quarantine_mount_id = _descriptor_mount_id(quarantine_fd)
            if (
                not stat.S_ISDIR(held.st_mode)
                or not _same_delete_entry(held, named)
                or stat.S_IMODE(held.st_mode) != 0o700
                or held.st_uid != os.geteuid()
                or quarantine_mount_id != bound_root.mount_id
                or not _delete_directory_is_empty(quarantine_fd)
            ):
                raise MutationHostLifecycleError(
                    "precreated delete quarantine binding is unsafe"
                )
            os.fsync(quarantine_fd)
            os.fsync(held_parent)
            with self._state_lock:
                current = self._delete_receipts.get(pending)
                if (
                    current is not authority
                    or authority.quarantine_descriptor != quarantine_fd
                ):
                    raise MutationHostLifecycleError(
                        "delete receipt owner changed during binding"
                    )
                receipt_boot_id = authority.receipt.get("boot_id")
                if receipt_boot_id != _kernel_boot_id():
                    raise MutationHostLifecycleError(
                        "delete receipt belongs to another boot"
                    )
                authority.pending_bind = {
                    "expected": dict(authority.receipt),
                    "quarantine_device": held.st_dev,
                    "quarantine_inode": held.st_ino,
                    "quarantine_mode": held.st_mode,
                    "quarantine_mount_id": quarantine_mount_id,
                    "child_pid": process.pid,
                    "child_start_ticks": _child_start_ticks(process.pid),
                    "boot_id": receipt_boot_id,
                }
            try:
                self._ensure_delete_receipt_bound_locked(pending, authority)
            except BaseException as primary:
                # reserved->bound may already be durable.  Retry the exact
                # proposed bound while the child/qdir proof remains held.
                try:
                    self._ensure_delete_receipt_bound_locked(
                        pending,
                        authority,
                    )
                except BaseException as settlement:
                    error = MutationHostLifecycleError(
                        "delete receipt bind could not be settled"
                    )
                    error.add_note(
                        "initial bind failed after "
                        f"{type(primary).__name__}: {primary}"
                    )
                    raise error from settlement
                raise
        except MutationHostLifecycleError:
            raise
        except Exception as exc:
            # Keep any created qdir and durable receipt intact. stop() owns
            # exact descriptor-relative restoration/settlement.
            raise MutationHostLifecycleError(
                "delete quarantine could not be durably bound"
            ) from exc

    def _reprove_delete_barrier(
        self,
        pending: _PendingStaging,
        *,
        expect_payload: bool,
    ) -> None:
        """Reprove named and retained authority immediately before an ACK."""

        with self._delete_lifecycle_lock:
            with self._state_lock:
                authority = self._delete_receipts.get(pending)
                process = self._process
                roots = self._mutation_roots
                stopped = self._stopped
            if (
                stopped
                or authority is None
                or process is None
                or process.poll() is not None
                or roots is None
                or not authority.durable
                or authority.pending_bind is not None
                or authority.receipt.get("state") != "bound"
            ):
                raise MutationHostLifecycleError(
                    "delete barrier lost live durable authority"
                )
            parent_descriptor = authority.parent_descriptor
            quarantine_descriptor = authority.quarantine_descriptor
            target_identity = authority.target_identity
            if (
                parent_descriptor is None
                or quarantine_descriptor is None
                or target_identity is None
            ):
                raise MutationHostLifecycleError(
                    "delete barrier lost retained descriptor authority"
                )
            bound_root = self._bound_root_for_delete(pending)
            self._reprove_delete_parent_chain_locked(
                pending,
                authority,
                bound_root,
            )
            try:
                held_quarantine = os.fstat(quarantine_descriptor)
                named_quarantine = os.stat(
                    pending.path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                receipt = authority.receipt
                if (
                    not _same_delete_entry(held_quarantine, named_quarantine)
                    or held_quarantine.st_dev != receipt.get("quarantine_device")
                    or held_quarantine.st_ino != receipt.get("quarantine_inode")
                    or held_quarantine.st_mode != receipt.get("quarantine_mode")
                    or _descriptor_mount_id(quarantine_descriptor)
                    != receipt.get("quarantine_mount_id")
                    or receipt.get("boot_id") != _kernel_boot_id()
                    or receipt.get("child_pid") != process.pid
                    or receipt.get("child_start_ticks")
                    != _child_start_ticks(process.pid)
                ):
                    raise MutationHostLifecycleError(
                        "delete qdir or child differs from durable receipt"
                    )
                entries = _bounded_delete_names(
                    quarantine_descriptor,
                    maximum=2,
                    overflow_message="delete quarantine contains foreign entries",
                )
                expected_entries = (
                    (_DELETE_QUARANTINE_PAYLOAD,) if expect_payload else ()
                )
                if entries != expected_entries:
                    raise MutationHostLifecycleError(
                        "delete quarantine is not in the expected barrier state"
                    )
                target_name = pending.restore_target.name if (
                    pending.restore_target is not None
                ) else None
                if target_name is None:
                    raise MutationHostLifecycleError(
                        "delete barrier lost its restore target"
                    )
                if expect_payload:
                    try:
                        os.stat(
                            target_name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        raise MutationHostLifecycleError(
                            "delete public target reappeared before unlink ACK"
                        )
                    payload = os.stat(
                        _DELETE_QUARANTINE_PAYLOAD,
                        dir_fd=quarantine_descriptor,
                        follow_symlinks=False,
                    )
                    if _entry_ownership_key(payload) != target_identity:
                        raise MutationHostLifecycleError(
                            "delete quarantine payload changed before unlink ACK"
                        )
                else:
                    target = os.stat(
                        target_name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if _entry_ownership_key(target) != target_identity:
                        raise MutationHostLifecycleError(
                            "delete target changed before publication ACK"
                        )
                roots.revalidate(require_named=True)
            except MutationHostLifecycleError:
                raise
            except (OSError, ValueError, MutationHostQualificationError) as exc:
                raise MutationHostLifecycleError(
                    "delete barrier authority reproof failed"
                ) from exc

    def _ensure_delete_receipt_bound_locked(
        self,
        pending: _PendingStaging,
        authority: _DeleteReceiptAuthority,
    ) -> None:
        """Reconcile one exact reserved-to-bound receipt transition."""

        with self._state_lock:
            current = self._delete_receipts.get(pending)
            request = authority.pending_bind
            if current is not authority:
                raise MutationHostLifecycleError(
                    "delete receipt owner changed during binding"
                )
            if request is None:
                return
            request = dict(request)
        expected = request.pop("expected", None)
        if not isinstance(expected, dict):
            raise MutationHostLifecycleError(
                "delete receipt bind authority is malformed"
            )
        bound_root = self._bound_root_for_delete(pending)
        bound_receipt = authority.journal.bind_pending_delete_quarantine(
            expected,
            quarantine_device=int(request["quarantine_device"]),
            quarantine_inode=int(request["quarantine_inode"]),
            quarantine_mode=int(request["quarantine_mode"]),
            quarantine_mount_id=int(request["quarantine_mount_id"]),
            child_pid=int(request["child_pid"]),
            child_start_ticks=int(request["child_start_ticks"]),
            boot_id=str(request["boot_id"]),
            workspace_root_descriptor=bound_root.descriptor,
            parent_descriptor=authority.parent_descriptor,
        )
        proposed = {
            **expected,
            **request,
            "state": "bound",
        }
        if bound_receipt != proposed:
            raise MutationHostLifecycleError(
                "bound delete receipt differs from proposed authority"
            )
        with self._state_lock:
            current = self._delete_receipts.get(pending)
            if current is not authority:
                raise MutationHostLifecycleError(
                    "delete receipt owner changed during binding"
                )
            authority.receipt = bound_receipt
            authority.durable = True
            authority.pending_bind = None

    def _reconcile_delete_bind_after_child_death_locked(
        self,
        pending: _PendingStaging,
        authority: _DeleteReceiptAuthority,
    ) -> None:
        """Resolve an ambiguous bind after positively reaping its child."""

        try:
            # An exact already-bound retry needs only the durable qdir proof;
            # fresh reserved->bound still (correctly) rejects a dead child.
            self._ensure_delete_receipt_bound_locked(pending, authority)
            return
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            error = MutationHostLifecycleError(
                "delete bind state remains indeterminate after child death"
            )
            raise error from exc

    def _clear_delete_receipt(self, pending: _PendingStaging) -> None:
        """Clear one exact receipt only after qdir settlement was proved."""

        with self._delete_lifecycle_lock:
            self._clear_delete_receipt_locked(pending)

    def _clear_delete_receipt_locked(
        self,
        pending: _PendingStaging,
    ) -> None:
        """Clear one receipt under the complete delete lifecycle."""

        with self._state_lock:
            authority = self._delete_receipts.get(pending)
        if authority is None or not authority.durable:
            raise MutationHostLifecycleError(
                "delete settlement lost its durable receipt"
            )
        bound_root = self._bound_root_for_delete(pending)
        initial_failure: BaseException | None = None
        try:
            authority.journal.clear_pending_delete_quarantine(
                authority.receipt,
                workspace_root_descriptor=bound_root.descriptor,
                parent_descriptor=authority.parent_descriptor,
            )
        except BaseException as primary:
            initial_failure = primary
            with self._state_lock:
                current = self._delete_receipts.get(pending)
                locally_settled = (
                    current is None
                    and pending not in self._pending_staging
                )
            if locally_settled:
                raise
            if current is not authority:
                raise MutationHostLifecycleError(
                    "delete receipt owner changed during settlement"
                ) from primary
            try:
                authority.journal.clear_pending_delete_quarantine(
                    authority.receipt,
                    workspace_root_descriptor=bound_root.descriptor,
                    parent_descriptor=authority.parent_descriptor,
                )
            except BaseException as settlement:
                error = MutationHostLifecycleError(
                    "delete receipt clear could not be settled"
                )
                error.add_note(
                    "initial clear failed after "
                    f"{type(primary).__name__}: {primary}"
                )
                raise error from settlement
        self._close_delete_receipt_descriptors_locked(pending, authority)
        self._forget_delete_receipt_locked(pending, authority)
        if initial_failure is not None:
            raise initial_failure

    def _close_delete_receipt_descriptors_locked(
        self,
        pending: _PendingStaging,
        authority: _DeleteReceiptAuthority,
    ) -> None:
        """Detach retained fds before one-shot close and fail closed on doubt."""

        for attribute, label in (
            ("quarantine_descriptor", "quarantine"),
            ("parent_descriptor", "parent"),
        ):
            with self._state_lock:
                if self._delete_receipts.get(pending) is not authority:
                    raise MutationHostLifecycleError(
                        "delete receipt owner changed during descriptor close"
                    )
                descriptor = getattr(authority, attribute)
                if descriptor is None:
                    continue
                if authority.descriptor_close_in_progress:
                    raise MutationHostLifecycleError(
                        "delete descriptor close is already in progress"
                    )
                authority.descriptor_close_in_progress = True
                setattr(authority, attribute, None)
            try:
                os.close(descriptor)
            except BaseException as exc:
                with self._state_lock:
                    authority.descriptor_close_in_progress = False
                    authority.descriptor_settlement_uncertain = True
                error = MutationHostLifecycleError(
                    f"delete {label} descriptor close is indeterminate"
                )
                raise error from exc
            with self._state_lock:
                authority.descriptor_close_in_progress = False

    def _forget_delete_receipt_locked(
        self,
        pending: _PendingStaging,
        authority: _DeleteReceiptAuthority,
    ) -> None:
        """Drop local receipt state in an interruption-recoverable order."""

        with self._state_lock:
            current = self._delete_receipts.get(pending)
            if current is not authority:
                raise MutationHostLifecycleError(
                    "delete receipt owner changed during settlement"
                )
            if (
                authority.parent_descriptor is not None
                or authority.quarantine_descriptor is not None
                or authority.reproof_descriptor is not None
                or authority.reproof_close_in_progress
                or authority.descriptor_close_in_progress
                or authority.descriptor_settlement_uncertain
            ):
                raise MutationHostLifecycleError(
                    "delete receipt descriptors are not settled"
                )
            # The dictionary is stop()'s final inventory authority.  Remove
            # the redundant staging-set entry first, then the dictionary, so
            # interruption at either bytecode leaves a discoverable owner or
            # a completely settled local namespace.
            self._pending_staging.discard(pending)
            del self._delete_receipts[pending]

    def _write_frame(self, payload: Mapping[str, Any]) -> None:
        encoded = _encode_frame(payload)
        with self._state_lock:
            process = self._process
        if process is None or process.poll() is not None:
            raise MutationHostProtocolError("isolated mutation child is not running")
        selector = selectors.DefaultSelector()
        try:
            with process._descriptor_lock:
                descriptor = process._request_write_descriptor
                if descriptor is None:
                    raise MutationHostProtocolError(
                        "isolated mutation child has no request pipe"
                    )
                selector.register(descriptor, selectors.EVENT_WRITE)
            view = memoryview(encoded)
            while view:
                if not selector.select(_remaining(self._deadline)):
                    raise MutationHostTimeoutError(
                        "isolated mutation request pipe exceeded its deadline"
                    )
                with process._descriptor_lock:
                    if process._request_write_descriptor != descriptor:
                        raise MutationHostProtocolError(
                            "isolated mutation request pipe was detached"
                        )
                    try:
                        written = os.write(descriptor, view[:_PIPE_CHUNK_BYTES])
                    except BlockingIOError:
                        continue
                if written <= 0:
                    raise MutationHostProtocolError(
                        "isolated mutation child stopped reading requests"
                    )
                view = view[written:]
        except (BrokenPipeError, OSError) as exc:
            raise MutationHostProtocolError(
                "isolated mutation child closed its request pipe"
            ) from exc
        finally:
            selector.close()

    def _read_frame(self, deadline: float) -> dict[str, Any]:
        with self._state_lock:
            process = self._process
        if process is None:
            raise MutationHostProtocolError(
                "isolated mutation child has no response pipe"
            )
        selector = selectors.DefaultSelector()
        try:
            with process._descriptor_lock:
                descriptor = process._response_read_descriptor
                if descriptor is None:
                    raise MutationHostProtocolError(
                        "isolated mutation child has no response pipe"
                    )
                selector.register(descriptor, selectors.EVENT_READ)
            while True:
                newline = self._read_buffer.find(b"\n")
                if newline >= 0:
                    encoded = bytes(self._read_buffer[:newline])
                    del self._read_buffer[: newline + 1]
                    return _decode_frame(encoded)
                remaining = _remaining(deadline)
                if not selector.select(remaining):
                    raise MutationHostTimeoutError(
                        "isolated mutation child exceeded its logical deadline"
                    )
                with process._descriptor_lock:
                    if process._response_read_descriptor != descriptor:
                        raise MutationHostProtocolError(
                            "isolated mutation response pipe was detached"
                        )
                    try:
                        chunk = os.read(descriptor, _PIPE_CHUNK_BYTES)
                    except BlockingIOError:
                        continue
                if not chunk:
                    raise MutationHostProtocolError(
                        "isolated mutation child exited before a terminal frame"
                    )
                self._read_buffer.extend(chunk)
                if len(self._read_buffer) > _MAX_FRAME_BYTES:
                    raise MutationHostProtocolError(
                        "isolated mutation child emitted an oversized frame"
                    )
        finally:
            selector.close()

    def stop(self) -> None:
        """Settle all child authority before replaying an observed SIGINT.

        Python can raise :class:`KeyboardInterrupt` at any bytecode boundary,
        including between SIGKILL and ``waitpid`` or midway through resource
        cleanup. Every operation below is deliberately retryable. Retain the
        first interrupt, retry the exact owned settlement until it either
        positively completes or raises a real lifecycle failure, and only
        then replay cancellation to the caller.
        """

        interrupt_count = 0
        while True:
            try:
                self._stop_once()
            except KeyboardInterrupt as exc:
                interrupt_count += 1
                with self._state_lock:
                    self._deferred_interrupt = self._deferred_interrupt or exc
                if interrupt_count >= _MAX_SETTLEMENT_INTERRUPTS:
                    raise MutationHostLifecycleError(
                        "repeated SIGINT prevented isolated mutation "
                        "settlement; exact cleanup authority is retained"
                    ) from exc
                continue
            except BaseException as exc:
                with self._state_lock:
                    interrupted = self._deferred_interrupt is not None
                if interrupted:
                    exc.add_note(
                        "SIGINT was deferred while isolated mutation "
                        "settlement remained unproven"
                    )
                raise
            break
        with self._state_lock:
            deferred_interrupt = self._deferred_interrupt
            self._deferred_interrupt = None
        if deferred_interrupt is not None:
            raise deferred_interrupt

    def _stop_once(self) -> None:
        """Perform one retryable pass over exact host settlement state."""

        with self._stop_lock:
            # This lock is the receipt namespace lifecycle.  Holding it from
            # the stopped transition through child death, receipt settlement,
            # and root-descriptor close prevents an in-flight arm/prepare from
            # publishing after cleanup has proved the namespace empty.
            with self._delete_lifecycle_lock:
                with self._state_lock:
                    if self._settled:
                        return
                    self._stopped = True
                    process = self._process
                    active_call = self._active_call
                # A bind CAS can be durable before its return becomes visible
                # to dispatch. Reconcile every proposed bound while the exact
                # child identity is still live; only then may settlement kill
                # the child and recover its qdir.
                with self._state_lock:
                    pending_binds = tuple(
                        (pending, authority)
                        for pending, authority in self._delete_receipts.items()
                        if authority.pending_bind is not None
                    )
                bind_interrupt: KeyboardInterrupt | None = None
                for pending, authority in pending_binds:
                    try:
                        self._ensure_delete_receipt_bound_locked(
                            pending,
                            authority,
                        )
                    except KeyboardInterrupt as exc:
                        bind_interrupt = bind_interrupt or exc
                    except BaseException:
                        # Child settlement cannot depend on journal I/O. Once
                        # the child is dead, reconcile exact bound-vs-reserved
                        # durable state below while retaining receipt/qdir
                        # authority on any persistent failure.
                        pass
                if process is not None:
                    # Do not discard the process handle, pipes, private root,
                    # or exact staging paths unless group death was positively
                    # confirmed. A later close() call can then retry settlement
                    # instead of turning one failed kill into false success.
                    self._settle_process(process, graceful=not active_call)
                with self._state_lock:
                    pending_binds = tuple(
                        (pending, authority)
                        for pending, authority in self._delete_receipts.items()
                        if authority.pending_bind is not None
                    )
                for pending, authority in pending_binds:
                    self._reconcile_delete_bind_after_child_death_locked(
                        pending,
                        authority,
                    )
                with self._transaction_lock:
                    with self._state_lock:
                        transaction = self._active_transaction
                    if transaction is not None:
                        transaction.close(rollback=True)
                        with self._state_lock:
                            if self._active_transaction is transaction:
                                self._active_transaction = None
                with self._state_lock:
                    pending = tuple(
                        self._pending_staging.union(self._delete_receipts)
                    )
                for stage in pending:
                    try:
                        self._cleanup_staging(stage)
                    except OSError as exc:
                        raise MutationHostLifecycleError(
                            "isolated mutation staging cleanup failed"
                        ) from exc
                self._settle_child_effects()
                self._close_resources(process)
                with self._state_lock:
                    self._process = None
                    self._settled = True
                if bind_interrupt is not None:
                    raise bind_interrupt

    def _settle_child_effects(self) -> None:
        with self._state_lock:
            effects = tuple(self._uncertain_child_effects)
            roots = self._mutation_roots
        if not effects:
            return
        if roots is None:
            raise MutationHostLifecycleError(
                "uncertain child effects have no retained root authority"
            )
        unresolved: list[_UncertainChildEffect] = []
        for effect in effects:
            # Unknown child descriptors are settled by confirmed process
            # reaping. mkdir has a persistent filesystem effect and remains
            # uncertain until its exact descriptor-relative name is absent.
            if effect.operation == "mkdir" and _relative_entry_exists(roots, effect):
                unresolved.append(effect)
        if unresolved:
            raise MutationHostLifecycleError(
                "isolated mutation mkdir effects remain uncertain"
            )
        with self._state_lock:
            for effect in effects:
                self._uncertain_child_effects.discard(effect)

    def _settle_process(
        self,
        process: _PosixSpawnProcess,
        *,
        graceful: bool,
    ) -> None:
        try:
            self._settle_process_unchecked(process, graceful=graceful)
        except MutationHostLifecycleError:
            raise
        except OSError as exc:
            raise MutationHostLifecycleError(
                "isolated mutation settlement hit an OS lifecycle error"
            ) from exc

    def _settle_process_unchecked(
        self,
        process: _PosixSpawnProcess,
        *,
        graceful: bool,
    ) -> None:
        pgid = process.pid
        if pgid is None:
            return
        if pgid <= 0:
            raise MutationHostLifecycleError(
                "isolated mutation process-group authority is unsafe"
            )
        if graceful and process.poll() is None:
            with suppress(MutationHostProtocolError):
                self._write_frame({"type": "shutdown"})
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=_TERM_GRACE_SECONDS)
        if self._process_group_exists(pgid):
            self._signal_process_group(pgid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_TERM_GRACE_SECONDS)
        if self._process_group_exists(pgid):
            self._signal_process_group(pgid, signal.SIGKILL)
        elif process.poll() is None:
            try:
                process.kill()
            except OSError as exc:
                raise MutationHostLifecycleError(
                    "isolated mutation child could not be killed"
                ) from exc
        try:
            process.wait(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise MutationHostLifecycleError(
                "isolated mutation child could not be reaped after SIGKILL"
            ) from exc
        group_deadline = time.monotonic() + _KILL_GRACE_SECONDS
        while self._process_group_exists(pgid):
            if time.monotonic() >= group_deadline:
                raise MutationHostLifecycleError(
                    "isolated mutation process group survived SIGKILL"
                )
            time.sleep(0.01)

    @staticmethod
    def _process_group_exists(pgid: int) -> bool:
        if pgid <= 0:
            raise MutationHostLifecycleError(
                "isolated mutation process-group authority is unsafe"
            )
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:
            raise MutationHostLifecycleError(
                "could not inspect the isolated mutation process group"
            ) from exc
        return True

    @staticmethod
    def _signal_process_group(pgid: int, signal_number: int) -> bool:
        if pgid <= 0:
            raise MutationHostLifecycleError(
                "isolated mutation process-group authority is unsafe"
            )
        try:
            os.killpg(pgid, signal_number)
        except ProcessLookupError:
            return False
        except OSError as exc:
            raise MutationHostLifecycleError(
                "could not signal the isolated mutation process group"
            ) from exc
        return True

    def _cleanup_staging(self, pending: _PendingStaging) -> None:
        with self._delete_lifecycle_lock:
            self._cleanup_staging_locked(pending)

    def _cleanup_staging_locked(self, pending: _PendingStaging) -> None:
        """Settle one staging entry under the delete-lifecycle lock."""

        if pending.restore_target is not None:
            with self._state_lock:
                authority = self._delete_receipts.get(pending)
            if authority is None:
                raise MutationHostLifecycleError(
                    "delete staging lost its registered receipt authority"
                )
            if (
                authority.parent_acquisition_in_progress
                or authority.parent_acquisition_uncertain
                or authority.quarantine_acquisition_in_progress
                or authority.quarantine_acquisition_uncertain
                or authority.reproof_descriptor is not None
                or authority.reproof_acquisition_in_progress
                or authority.reproof_acquisition_uncertain
                or authority.reproof_close_in_progress
                or authority.descriptor_close_in_progress
                or authority.descriptor_settlement_uncertain
            ):
                raise MutationHostLifecycleError(
                    "delete descriptor lifecycle remains uncertain"
                )
            self._ensure_delete_receipt_armed(pending, authority)
            self._recover_delete_quarantine(pending)
            self._clear_delete_receipt(pending)
            return
        with self._state_lock:
            if pending not in self._pending_staging:
                return
            roots = self._mutation_roots
        if roots is None:
            raise MutationHostLifecycleError(
                "pending staging cleanup has no root authority"
            )
        _unlink_pending_staging(roots, pending)
        with self._state_lock:
            self._pending_staging.discard(pending)

    def _recover_delete_quarantine(self, pending: _PendingStaging) -> None:
        """Restore an opaque retained payload, or remove an empty qdir.

        The host owns only the exact tokenized quarantine-to-public mapping.
        It never recursively removes quarantine contents and never replaces a
        public target that appeared after quarantine. A collision therefore
        remains preserved for operator recovery and keeps host settlement
        incomplete.
        """

        with self._delete_lifecycle_lock:
            self._recover_delete_quarantine_locked(pending)

    def _recover_delete_quarantine_locked(
        self,
        pending: _PendingStaging,
    ) -> None:
        """Recover one qdir under the complete delete lifecycle."""

        stage = pending.path
        target = pending.restore_target
        repository_root = pending.repository_root
        if target is None or repository_root is None:
            raise MutationHostLifecycleError(
                "delete quarantine has incomplete restore authority"
            )
        if stage.parent != target.parent:
            raise MutationHostLifecycleError(
                "delete quarantine is not adjacent to its restore target"
            )
        try:
            with self._state_lock:
                receipt_authority = self._delete_receipts.get(pending)
            if receipt_authority is None or not receipt_authority.durable:
                raise MutationHostLifecycleError(
                    "delete recovery lost its durable receipt"
                )
            receipt = receipt_authority.receipt
            receipt_state = receipt.get("state")
            if receipt_state not in {"reserved", "bound"}:
                raise MutationHostLifecycleError(
                    "delete recovery receipt state is malformed"
                )
            with self._state_lock:
                bound_root = self._repository_roots.get(repository_root)
            if bound_root is None:
                raise MutationHostLifecycleError(
                    "delete recovery lost its held repository root"
                )
            current_root = os.fstat(bound_root.descriptor)
            if (current_root.st_dev, current_root.st_ino) != (
                bound_root.device,
                bound_root.inode,
            ) or _descriptor_mount_id(bound_root.descriptor) != (
                bound_root.mount_id
            ):
                raise MutationHostLifecycleError(
                    "delete recovery repository root identity changed"
                )

            parent_fd = receipt_authority.parent_descriptor
            if parent_fd is None:
                with self._state_lock:
                    receipt_authority.parent_acquisition_in_progress = True
                try:
                    opened_parent, parent_chain = _open_delete_parent_chain(
                        bound_root.descriptor,
                        pending.parent_components,
                        expected_mount_id=bound_root.mount_id,
                    )
                except BaseException:
                    with self._state_lock:
                        receipt_authority.parent_acquisition_in_progress = False
                        receipt_authority.parent_acquisition_uncertain = True
                    raise
                rendered_chain = [list(identity) for identity in parent_chain]
                if receipt.get("parent_chain") != rendered_chain:
                    try:
                        os.close(opened_parent)
                    except BaseException:
                        with self._state_lock:
                            receipt_authority.descriptor_settlement_uncertain = (
                                True
                            )
                    with self._state_lock:
                        receipt_authority.parent_acquisition_in_progress = False
                    raise MutationHostLifecycleError(
                        "delete recovery named parent chain differs from "
                        "its receipt"
                    )
                with self._state_lock:
                    if (
                        self._delete_receipts.get(pending)
                        is not receipt_authority
                        or receipt_authority.parent_descriptor is not None
                    ):
                        try:
                            os.close(opened_parent)
                        except BaseException:
                            receipt_authority.descriptor_settlement_uncertain = (
                                True
                            )
                        raise MutationHostLifecycleError(
                            "delete recovery parent owner changed"
                        )
                    receipt_authority.parent_descriptor = opened_parent
                    receipt_authority.parent_acquisition_in_progress = False
                    parent_fd = opened_parent
            expected_chain = receipt.get("parent_chain")
            if (
                not isinstance(expected_chain, list)
                or not expected_chain
                or list(_delete_directory_chain_identity(parent_fd))
                != expected_chain[-1]
                or _descriptor_mount_id(parent_fd) != bound_root.mount_id
            ):
                raise MutationHostLifecycleError(
                    "delete recovery retained parent differs from its receipt"
                )

            quarantine_fd = receipt_authority.quarantine_descriptor
            if quarantine_fd is None:
                try:
                    os.stat(
                        stage.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    return
                with self._state_lock:
                    receipt_authority.quarantine_acquisition_in_progress = True
                try:
                    opened_quarantine = os.open(
                        stage.name,
                        os.O_RDONLY
                        | os.O_CLOEXEC
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                except BaseException:
                    with self._state_lock:
                        receipt_authority.quarantine_acquisition_in_progress = (
                            False
                        )
                        receipt_authority.quarantine_acquisition_uncertain = True
                    raise
                with self._state_lock:
                    if (
                        self._delete_receipts.get(pending)
                        is not receipt_authority
                        or receipt_authority.quarantine_descriptor is not None
                    ):
                        try:
                            os.close(opened_quarantine)
                        except BaseException:
                            receipt_authority.descriptor_settlement_uncertain = (
                                True
                            )
                        raise MutationHostLifecycleError(
                            "delete recovery qdir owner changed"
                        )
                    receipt_authority.quarantine_descriptor = opened_quarantine
                    receipt_authority.quarantine_acquisition_in_progress = False
                    quarantine_fd = opened_quarantine

            held = os.fstat(quarantine_fd)
            try:
                named = os.stat(
                    stage.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if held.st_nlink != 0 or not _delete_directory_is_empty(
                    quarantine_fd
                ):
                    raise MutationHostLifecycleError(
                        "removed delete quarantine remains linked or nonempty"
                    ) from None
                return
            quarantine_mount_id = _descriptor_mount_id(quarantine_fd)
            if (
                not _same_delete_entry(held, named)
                or not stat.S_ISDIR(held.st_mode)
                or stat.S_IMODE(held.st_mode) != 0o700
                or held.st_uid != os.geteuid()
                or quarantine_mount_id != bound_root.mount_id
            ):
                raise MutationHostLifecycleError(
                    "delete quarantine binding could not be proved"
                )
            if receipt_state == "bound" and (
                held.st_dev != receipt.get("quarantine_device")
                or held.st_ino != receipt.get("quarantine_inode")
                or held.st_mode != receipt.get("quarantine_mode")
                or quarantine_mount_id
                != receipt.get("quarantine_mount_id")
            ):
                raise MutationHostLifecycleError(
                    "delete quarantine differs from its durable receipt"
                )
            entries = _bounded_delete_names(
                quarantine_fd,
                maximum=1,
                overflow_message=(
                    "delete quarantine contains unowned recovery entries"
                ),
            )
            if not entries:
                os.rmdir(stage.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                if (
                    os.fstat(quarantine_fd).st_nlink != 0
                    or not _delete_directory_is_empty(quarantine_fd)
                ):
                    raise MutationHostLifecycleError(
                        "delete quarantine removal could not be proved"
                    )
                return
            if receipt_state == "reserved":
                raise MutationHostLifecycleError(
                    "reserved delete quarantine unexpectedly has contents"
                )
            if entries != (_DELETE_QUARANTINE_PAYLOAD,):
                raise MutationHostLifecycleError(
                    "delete quarantine contains unowned recovery entries"
                )
            payload = os.stat(
                _DELETE_QUARANTINE_PAYLOAD,
                dir_fd=quarantine_fd,
                follow_symlinks=False,
            )
            if (
                receipt_authority.target_identity is None
                or stat.S_ISLNK(payload.st_mode)
                or not (
                    stat.S_ISREG(payload.st_mode)
                    or stat.S_ISDIR(payload.st_mode)
                )
                or _entry_ownership_key(payload)
                != receipt_authority.target_identity
            ):
                raise MutationHostLifecycleError(
                    "delete quarantine payload differs from retained target"
                )
            try:
                os.stat(
                    target.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise MutationHostLifecycleError(
                    "delete restore target is occupied; public and "
                    "quarantined entries were preserved"
                )
            try:
                _rename_noreplace_at(
                    quarantine_fd,
                    _DELETE_QUARANTINE_PAYLOAD,
                    parent_fd,
                    target.name,
                )
            except FileExistsError as exc:
                raise MutationHostLifecycleError(
                    "delete restore target appeared; public and quarantined "
                    "entries were preserved"
                ) from exc
            os.fsync(quarantine_fd)
            os.fsync(parent_fd)
            if not _delete_directory_is_empty(quarantine_fd):
                raise MutationHostLifecycleError(
                    "delete quarantine changed during restoration"
                )
            os.rmdir(stage.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            restored = os.stat(
                target.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                _entry_ownership_key(restored)
                != receipt_authority.target_identity
                or os.fstat(quarantine_fd).st_nlink != 0
            ):
                raise MutationHostLifecycleError(
                    "delete quarantine restoration could not be proved"
                )
        except MutationHostLifecycleError:
            raise
        except (OSError, ValueError, MutationHostQualificationError) as exc:
            raise MutationHostLifecycleError(
                "delete quarantine recovery failed"
            ) from exc

    def _close_resources(
        self,
        process: _PosixSpawnProcess | None,
    ) -> None:
        with self._state_lock:
            stderr_descriptor = self._stderr_descriptor
            stderr_acquisition_uncertain = self._stderr_acquisition_uncertain
            source_authority = self._source_authority
            mutation_roots = self._mutation_roots
            private_root_authority = self._private_root_authority
        failures: list[BaseException] = []
        if process is not None:
            try:
                process.close_parent_resources()
            except MutationHostLifecycleError as exc:
                failures.append(exc)
        if stderr_descriptor is not None:
            with self._state_lock:
                if self._stderr_descriptor == stderr_descriptor:
                    self._stderr_descriptor = None
            try:
                os.close(stderr_descriptor)
            except OSError as exc:
                failures.append(
                    MutationHostLifecycleError(
                        "isolated mutation stderr descriptor cleanup failed"
                    )
                )
                failures[-1].__cause__ = exc
        if stderr_acquisition_uncertain:
            failures.append(
                MutationHostLifecycleError(
                    "isolated mutation stderr acquisition remains uncertain"
                )
            )
        if source_authority is not None:
            try:
                source_authority.close()
            except MutationHostLifecycleError as exc:
                failures.append(exc)
            else:
                with self._state_lock:
                    if self._source_authority is source_authority:
                        self._source_authority = None
        if private_root_authority is not None:
            try:
                _close_private_root_authority(private_root_authority)
            except MutationHostLifecycleError as exc:
                failures.append(exc)
            with self._state_lock:
                if (
                    private_root_authority.root_removed
                    and private_root_authority.root_descriptor is None
                    and private_root_authority.parent_descriptor is None
                    and self._private_root_authority is private_root_authority
                ):
                    self._private_root_authority = None
                    self._private_root = None
        if mutation_roots is not None:
            try:
                mutation_roots.close()
            except MutationHostLifecycleError as exc:
                failures.append(exc)
            with self._state_lock:
                if (
                    all(root.descriptor is None for root in mutation_roots.roots)
                    and not any(
                        root.acquisition_in_progress or root.acquisition_uncertain
                        for root in mutation_roots.roots
                    )
                    and self._mutation_roots is mutation_roots
                ):
                    self._mutation_roots = None
        with self._state_lock:
            roots_closed = self._mutation_roots is None
        if roots_closed:
            try:
                self._close_repository_roots()
            except MutationHostLifecycleError as exc:
                failures.append(exc)
        if failures:
            error = MutationHostLifecycleError(
                "isolated mutation resource cleanup was incomplete"
            )
            for failure in failures[1:]:
                error.add_note(f"additional cleanup failure: {failure}")
            raise error from failures[0]

    def __enter__(self) -> IsolatedMutationHost:
        self.start()
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.stop()


class ProcessHostedToolSet:
    """Route a frozen ToolSet's mutators through an isolated process host.

    Backend adapters should freeze descriptors and classification first, bind
    provenance against that frozen view, then pass it here.  The wrapper does
    not alter catalogue order, schemas, or read dispatch.
    """

    def __init__(self, frozen_toolset: Any, host: IsolatedMutationHost) -> None:
        from crucible.tools._dispatch_budget import (  # noqa: PLC0415
            ProgressBoundedMutatingToolSet,
        )

        write_names = {str(name) for name in getattr(frozen_toolset, "write_names", ())}
        read_only_names = {
            str(name) for name in getattr(frozen_toolset, "read_only_names", ())
        }
        descriptors = frozen_toolset.to_anthropic_tools()
        if not isinstance(descriptors, list) or any(
            not isinstance(descriptor, Mapping)
            or not isinstance(descriptor.get("name"), str)
            for descriptor in descriptors
        ):
            raise MutationHostQualificationError(
                "hosted ToolSet descriptors must be named objects"
            )
        descriptor_names = {str(descriptor["name"]) for descriptor in descriptors}
        if (
            not write_names
            or write_names != set(host.tool_names)
            or write_names & read_only_names
            or write_names | read_only_names != descriptor_names
            or len(descriptor_names) != len(descriptors)
        ):
            raise MutationHostQualificationError(
                "hosted ToolSet catalogue/classification does not match the "
                "isolated mutation specification"
            )
        self._frozen = frozen_toolset
        self._host = host
        self._progress_wrapper = (
            frozen_toolset
            if isinstance(frozen_toolset, ProgressBoundedMutatingToolSet)
            else None
        )
        self.write_names = frozenset(write_names)
        self.read_only_names = frozenset(read_only_names)

    @property
    def tool_names(self) -> frozenset[str]:
        return self.write_names | self.read_only_names

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return self._frozen.to_anthropic_tools()

    def to_openai_tools(self) -> list[dict[str, Any]]:
        converter = getattr(self._frozen, "to_openai_tools", None)
        if callable(converter):
            return converter()
        return [
            {
                "type": "function",
                "function": {
                    "name": descriptor["name"],
                    "description": descriptor.get("description", ""),
                    "parameters": descriptor.get("input_schema") or {},
                },
            }
            for descriptor in self.to_anthropic_tools()
        ]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        if name in self.write_names:
            try:
                result = self._host.dispatch(name, arguments)
            except ToolMutationIndeterminate:
                raise
            except (
                MutationHostLifecycleError,
                MutationHostProtocolError,
                MutationHostTimeoutError,
            ) as exc:
                raise ToolMutationIndeterminate(
                    "isolated mutation dispatch did not reach a proved "
                    "terminal state; provider continuation is forbidden"
                ) from exc
            if self._progress_wrapper is not None:
                self._progress_wrapper.record_isolated_write_result(result)
            return result
        return self._frozen.dispatch(name, arguments)

    def mcp_config(self) -> Any:
        getter = getattr(self._frozen, "mcp_config", None)
        return None if not callable(getter) else getter()


def _context_from_spec(spec: MutationContextSpec) -> Any:
    from crucible.tools._context import ToolContext  # noqa: PLC0415

    return ToolContext(
        workspace_root=Path(spec.workspace_root),
        repo_root=Path(spec.repo_root),
        workspace_relative_paths=spec.workspace_relative_paths,
        write_scope=spec.write_scope,
        write_exact_scope=spec.write_exact_scope,
        write_deny_scope=spec.write_deny_scope,
        mutation_journal_dir=(
            None
            if spec.mutation_journal_dir is None
            else Path(spec.mutation_journal_dir)
        ),
        mutation_journal_transaction_id=(
            spec.mutation_journal_transaction_id
        ),
        container_id=spec.container_id,
        container_workdir=spec.container_workdir,
        path_resolver=None,
    )


def _child_read_frame() -> dict[str, Any] | None:
    encoded = sys.stdin.buffer.readline(_MAX_FRAME_BYTES + 2)
    if not encoded:
        return None
    if not encoded.endswith(b"\n") or len(encoded) > _MAX_FRAME_BYTES + 1:
        raise MutationHostProtocolError(
            "parent emitted an unterminated or oversized frame"
        )
    return _decode_frame(encoded[:-1])


def _child_emit(payload: Mapping[str, Any]) -> None:
    sys.stdout.buffer.write(_encode_frame(payload))
    sys.stdout.buffer.flush()


def _child_expect(
    payload: Mapping[str, Any] | None,
    kind: str,
    call_id: str,
    name: str,
) -> None:
    if (
        payload is None
        or payload.get("type") != kind
        or payload.get("call_id") != call_id
        or payload.get("name") != name
    ):
        raise MutationHostProtocolError(f"parent emitted a mismatched {kind} frame")


def _arm_parent_death_signal() -> None:
    """Kill the isolated writer if its exact parent process disappears."""

    parent_pid = os.getppid()
    if parent_pid <= 1:
        raise MutationHostLifecycleError(
            "isolated mutation child has no live parent"
        )
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (AttributeError, OSError) as exc:
        raise MutationHostLifecycleError(
            "isolated mutation child cannot arm parent-death settlement"
        ) from exc
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise MutationHostLifecycleError(
            "isolated mutation child cannot arm parent-death settlement: "
            f"{os.strerror(error)}"
        )
    # Close the race where the parent died between getppid() and prctl().
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
        raise MutationHostLifecycleError(
            "isolated mutation parent changed during child startup"
        )


def _execute_binding(
    binding: MutationToolBinding,
    arguments: Mapping[str, Any],
    publication_token: str,
    *,
    publication_barrier: Callable[[str, str | None], None],
    delete_publication_barrier: Callable[[], None],
    quarantine_barrier: Callable[[], None],
) -> str:
    try:
        if binding.kind == "write_source":
            return _write_source_from_spec(
                binding.context,
                file_path=arguments["file_path"],
                content=arguments["content"],
                expected_sha256=arguments.get("expected_sha256", ""),
                publication_token=publication_token,
                publication_barrier=publication_barrier,
            )
        if binding.kind == "exact_write_source":
            assert binding.exact_target is not None
            return _write_source_from_spec(
                binding.context,
                file_path=binding.exact_target,
                content=arguments["content"],
                expected_sha256="",
                publication_token=publication_token,
                publication_barrier=publication_barrier,
            )
        if binding.kind == "delete_source":
            return _delete_source_from_spec(
                binding.context,
                file_path=arguments["file_path"],
                recursive=arguments["recursive"],
                publication_token=publication_token,
                publication_barrier=delete_publication_barrier,
                quarantine_barrier=quarantine_barrier,
            )
        return _apply_patch_from_spec(
            binding.context,
            file_path=arguments["file_path"],
            before=arguments["before"],
            after=arguments["after"],
            publication_token=publication_token,
            publication_barrier=publication_barrier,
        )
    except (MutationHostLifecycleError, MutationHostQualificationError):
        raise
    except ToolMutationIndeterminate as exc:
        # A post-quarantine failure is terminal for this provider attempt.
        # Let the child exit so the parent must reap it, recover any retained
        # quarantine, and hand control to the outer workspace journal.
        raise ToolMutationIndeterminate(
            "isolated source deletion became indeterminate"
        ) from exc
    except Exception as exc:  # mirror BuiltToolSet.dispatch error semantics
        return f"ERROR: {type(exc).__name__}: {exc}"


def _child_source_ready_fields() -> dict[str, Any]:
    """Attest the exact sealed source image executed by this child."""

    raw_source_descriptor = os.environ.pop(
        _CHILD_SOURCE_DESCRIPTOR_ENV,
        "",
    )
    if not raw_source_descriptor.isascii() or not raw_source_descriptor.isdigit():
        raise MutationHostQualificationError(
            "isolated mutation child source descriptor is malformed"
        )
    source_descriptor = int(raw_source_descriptor)
    if source_descriptor <= 2:
        raise MutationHostQualificationError(
            "isolated mutation child source descriptor is unsafe"
        )
    os.set_inheritable(source_descriptor, False)
    source_metadata = os.fstat(source_descriptor)
    source_identity = _sealed_source_identity(source_metadata)
    seals = int(fcntl.fcntl(source_descriptor, _F_GET_SEALS))
    if (
        os.get_inheritable(source_descriptor)
        or seals & _SOURCE_IMAGE_REQUIRED_SEALS != _SOURCE_IMAGE_REQUIRED_SEALS
    ):
        raise MutationHostQualificationError(
            "isolated mutation child source projection changed"
        )
    return {
        "source_identity": list(source_identity),
        "source_sha256": _descriptor_sha256(
            source_descriptor,
            expected_identity=_stat_identity(source_metadata),
        ),
        "source_seals": seals,
        "closure": "sealed-loaded-source-v2",
    }


def _child_mutation_roots_ready_fields(
    spec: IsolatedToolSetSpec,
) -> dict[str, Any]:
    """Install and attest every retained descriptor-backed mutation root."""

    raw_manifest = os.environ.pop(_CHILD_MUTATION_ROOTS_ENV, "")
    try:
        manifest = json.loads(raw_manifest)
    except (TypeError, ValueError) as exc:
        raise MutationHostQualificationError(
            "isolated mutation root manifest is malformed"
        ) from exc
    if not isinstance(manifest, list) or not manifest:
        raise MutationHostQualificationError(
            "isolated mutation root manifest is malformed"
        )
    expected: dict[str, tuple[tuple[int, ...], int]] = {}
    for binding in spec.bindings:
        context = binding.context
        for path, identity, mount_id in (
            (
                context.workspace_root,
                context.workspace_root_identity,
                context.workspace_root_mount_id,
            ),
            (
                context.repo_root,
                context.repo_root_identity,
                context.repo_root_mount_id,
            ),
        ):
            candidate = (identity, mount_id)
            if path in expected and expected[path] != candidate:
                raise MutationHostQualificationError(
                    "isolated mutation root specifications disagree"
                )
            expected[path] = candidate
    installed: dict[str, _ChildMutationRootAuthority] = {}
    canonical: list[dict[str, Any]] = []
    for item in manifest:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "identity",
            "mount_id",
            "descriptor",
        }:
            raise MutationHostQualificationError(
                "isolated mutation root manifest is not closed"
            )
        path = _bounded_string(item["path"], setting="mutation root path")
        identity = _identity_tuple_from_wire(
            item["identity"],
            setting="mutation root identity",
        )
        mount_id = _positive_int(item["mount_id"], setting="mutation root mount_id")
        descriptor = _positive_int(
            item["descriptor"],
            setting="mutation root descriptor",
        )
        if path not in expected or expected[path] != (identity, mount_id):
            raise MutationHostQualificationError(
                "isolated mutation root manifest disagrees with the ToolSet"
            )
        if descriptor <= _CHILD_ROOT_DESCRIPTOR or descriptor in {
            root.descriptor for root in installed.values()
        }:
            raise MutationHostQualificationError(
                "isolated mutation root descriptor projection is unsafe"
            )
        os.set_inheritable(descriptor, False)
        authority = _ChildMutationRootAuthority(
            path=path,
            descriptor=descriptor,
            identity=identity,
            mount_id=mount_id,
        )
        authority.revalidate(require_named=True)
        installed[path] = authority
        canonical.append(
            {
                "path": path,
                "identity": list(identity),
                "mount_id": mount_id,
                "descriptor": descriptor,
            }
        )
    if set(installed) != set(expected):
        raise MutationHostQualificationError(
            "isolated mutation root manifest is incomplete"
        )
    if canonical != sorted(canonical, key=lambda item: str(item["path"])):
        raise MutationHostQualificationError(
            "isolated mutation root manifest is not canonical"
        )
    _CHILD_MUTATION_ROOTS.clear()
    _CHILD_MUTATION_ROOTS.update(installed)
    return {"mutation_roots": canonical}


def _child_private_root_ready_fields() -> dict[str, Any]:
    """Attest cwd, which bootstrap bound through the exact root descriptor."""

    raw_root_descriptor = os.environ.pop(_CHILD_ROOT_DESCRIPTOR_ENV, "")
    private_root = os.environ.pop(_CHILD_CWD_ENV, "")
    if (
        raw_root_descriptor != str(_CHILD_ROOT_DESCRIPTOR)
        or not private_root
        or not Path(private_root).is_absolute()
    ):
        raise MutationHostQualificationError(
            "isolated mutation child private-root authority is malformed"
        )
    # The bootstrap bound cwd with the inherited private-root descriptor.
    # Attest that kernel-held cwd directly instead of introducing another
    # child-side open acquisition whose effect could become ambiguous under
    # asynchronous exception delivery.
    cwd_metadata = os.stat(".", follow_symlinks=False)
    cwd_identity = _stable_directory_identity(cwd_metadata)
    cwd_mount_id = _path_mount_id(Path("."))
    if (
        _stable_directory_identity(os.stat(".", follow_symlinks=False)) != cwd_identity
        or _path_mount_id(Path(".")) != cwd_mount_id
    ):
        raise MutationHostQualificationError(
            "isolated mutation child cwd changed during attestation"
        )
    return {
        "private_root_identity": list(cwd_identity),
        "private_root_mount_id": cwd_mount_id,
    }


def _loaded_project_modules() -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in sys.modules
            if name == "crucible" or name.startswith("crucible.")
        )
    )


def _verify_child_project_import_seal(blocker: _ProjectImportBlocker) -> None:
    loaded = _loaded_project_modules()
    if loaded:
        raise MutationHostQualificationError(
            "isolated mutation child loaded project modules after sealing"
        )
    if not sys.meta_path or sys.meta_path[0] is not blocker:
        raise MutationHostQualificationError(
            "project import blocker retention is uncertain"
        )


def _seal_child_project_imports() -> _ProjectImportBlocker:
    """Deny every Crucible path import before acknowledging readiness."""

    loaded = _loaded_project_modules()
    if loaded:
        raise MutationHostQualificationError(
            "isolated mutation child loaded project modules outside its "
            "self-contained closure"
        )
    blocker = _ProjectImportBlocker()
    sys.meta_path.insert(0, blocker)
    _verify_child_project_import_seal(blocker)
    return blocker


def _child_main() -> int:
    _arm_parent_death_signal()
    initialize = _child_read_frame()
    if initialize is None or initialize.get("type") != "initialize":
        raise MutationHostProtocolError(
            "isolated mutation child requires one initialize frame"
        )
    spec = IsolatedToolSetSpec.from_wire(initialize.get("spec"))
    source_ready = _child_source_ready_fields()
    root_ready = _child_private_root_ready_fields()
    mutation_roots_ready = _child_mutation_roots_ready_fields(spec)
    project_import_blocker = _seal_child_project_imports()
    _child_emit(
        {
            "type": "ready",
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "sid": os.getsid(0),
            **source_ready,
            **root_ready,
            **mutation_roots_ready,
        }
    )
    while True:
        _verify_child_project_import_seal(project_import_blocker)
        request = _child_read_frame()
        if request is None or request.get("type") == "shutdown":
            return 0
        if (
            set(request)
            != {
                "type",
                "call_id",
                "name",
                "arguments",
                "publication_token",
            }
            or request.get("type") != "call"
        ):
            raise MutationHostProtocolError("invalid mutation call frame")
        call_id = _bounded_string(
            request["call_id"],
            setting="mutation call_id",
            maximum=128,
        )
        name = _bounded_string(
            request["name"],
            setting="mutation call name",
            maximum=128,
        )
        arguments = request["arguments"]
        publication_token = request["publication_token"]
        if (
            name not in spec.names
            or not isinstance(arguments, dict)
            or not isinstance(publication_token, str)
            or len(publication_token) != _PUBLICATION_TOKEN_HEX_LENGTH
            or any(
                character not in "0123456789abcdef" for character in publication_token
            )
        ):
            raise MutationHostProtocolError("invalid mutation call authority")
        _child_emit(
            {
                "type": "accepted",
                "call_id": call_id,
                "name": name,
            }
        )
        execute = _child_read_frame()
        _child_expect(execute, "execute", call_id, name)
        _child_emit(
            {
                "type": "started",
                "call_id": call_id,
                "name": name,
            }
        )
        ack = _child_read_frame()
        _child_expect(ack, "ack_started", call_id, name)

        def publication_barrier(
            content: str,
            expected_current_sha256: str | None,
            call_id: str = call_id,
            name: str = name,
        ) -> None:
            _child_emit(
                {
                    "type": "publish_ready",
                    "call_id": call_id,
                    "name": name,
                    "content": content,
                    "expected_current_sha256": expected_current_sha256,
                }
            )
            prepare = _child_read_frame()
            _child_expect(prepare, "prepare_publish", call_id, name)
            _child_emit(
                {
                    "type": "publish_alive",
                    "call_id": call_id,
                    "name": name,
                }
            )
            publish_ack = _child_read_frame()
            _child_expect(publish_ack, "ack_publish", call_id, name)

        def delete_publication_barrier(
            call_id: str = call_id,
            name: str = name,
        ) -> None:
            _child_emit(
                {
                    "type": "publish_ready",
                    "call_id": call_id,
                    "name": name,
                }
            )
            publish_ack = _child_read_frame()
            _child_expect(publish_ack, "ack_publish", call_id, name)

        def quarantine_barrier(
            call_id: str = call_id,
            name: str = name,
        ) -> None:
            _child_emit(
                {
                    "type": "quarantine_ready",
                    "call_id": call_id,
                    "name": name,
                }
            )
            quarantine_ack = _child_read_frame()
            _child_expect(
                quarantine_ack,
                "ack_quarantine",
                call_id,
                name,
            )

        try:
            result = _execute_binding(
                spec.binding_for(name),
                arguments,
                publication_token,
                publication_barrier=publication_barrier,
                delete_publication_barrier=delete_publication_barrier,
                quarantine_barrier=quarantine_barrier,
            )
            _verify_child_project_import_seal(project_import_blocker)
        except _ChildAcquisitionUncertain as exc:
            _child_emit(
                {
                    "type": "lifecycle_uncertain",
                    "call_id": call_id,
                    "name": name,
                    "operation": exc.operation,
                    "root_path": exc.root_path,
                    "relative_path": exc.relative_path,
                }
            )
            return 70
        except ToolMutationIndeterminate as exc:
            _verify_child_project_import_seal(project_import_blocker)
            _child_emit(
                {
                    "type": "terminal_mutation",
                    "call_id": call_id,
                    "name": name,
                    "message": str(exc)[:1024],
                }
            )
            return 2
        _child_emit(
            {
                "type": "result",
                "call_id": call_id,
                "name": name,
                "result": result,
            }
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args(argv)
    if not args.child:
        parser.error("this module is an internal --child entrypoint")
    try:
        return _child_main()
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IsolatedMutationHost",
    "IsolatedToolSetSpec",
    "MutationHostLifecycleError",
    "MutationHostProtocolError",
    "MutationHostQualificationError",
    "MutationHostTimeoutError",
    "ProcessHostedToolSet",
    "build_isolated_toolset_spec",
]
