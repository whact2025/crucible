# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Controller-side contract for the isolated Linux process supervisor."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


TARGET_ENV_SCHEMA = "crucible.runner-target-env.v1"
TARGET_ENV_MAX_BYTES = 256 * 1024
SUPERVISOR_SOURCE_MAX_BYTES = 256 * 1024
SUPERVISOR_ENV = MappingProxyType(
    {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
)
_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008
_F_GET_SEALS = 1034
_REQUIRED_SEALS = (
    _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE
)


class ProcessSupervisorContractError(OSError):
    """A private controller-to-supervisor boundary could not be created."""


def supervisor_environment() -> dict[str, str]:
    """Return a fresh copy of the immutable supervisor environment."""

    return dict(SUPERVISOR_ENV)


def supervisor_interpreter_path() -> str:
    """Return the kernel-bound interpreter identity for the supervisor.

    ``sys.executable`` is ordinary mutable process state and can also name a
    replaceable virtual-environment shim.  ``/proc/self/exe`` is resolved by
    the kernel against the calling process's executable inode, so the child
    exec cannot be redirected by same-uid pathname replacement.
    """

    if sys.platform != "linux":
        raise ProcessSupervisorContractError(
            "trusted supervisor interpreter requires Linux /proc"
        )
    path = "/proc/self/exe"
    flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProcessSupervisorContractError(
            "trusted supervisor interpreter identity is unavailable"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=True)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino)
            != (named.st_dev, named.st_ino)
        ):
            raise ProcessSupervisorContractError(
                "trusted supervisor interpreter identity is not a regular inode"
            )
    finally:
        os.close(descriptor)
    return path


def _create_memfd(name: str) -> int:
    native = getattr(os, "memfd_create", None)
    flags = _MFD_CLOEXEC | _MFD_ALLOW_SEALING
    if callable(native):
        return int(native(name, flags))
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        create = libc.memfd_create
    except (AttributeError, OSError) as exc:
        raise ProcessSupervisorContractError(
            "Linux libc does not expose memfd_create"
        ) from exc
    create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    create.restype = ctypes.c_int
    descriptor = int(create(name.encode("ascii", errors="strict"), flags))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise ProcessSupervisorContractError(
            error,
            os.strerror(error),
        )
    return descriptor


def _write_and_seal(payload: bytes, *, name: str) -> int:
    import fcntl  # noqa: PLC0415

    descriptor = _create_memfd(name)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProcessSupervisorContractError(
                    "sealed descriptor write made no progress"
                )
            view = view[written:]
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, _F_ADD_SEALS, _REQUIRED_SEALS)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_sealed_descriptor(
    descriptor: int,
    *,
    expected_size: int,
) -> None:
    import fcntl  # noqa: PLC0415

    try:
        metadata = os.fstat(descriptor)
        seals = fcntl.fcntl(descriptor, _F_GET_SEALS)
    except OSError as exc:
        raise ProcessSupervisorContractError(
            "trusted supervisor source descriptor is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != expected_size
        or seals & _REQUIRED_SEALS != _REQUIRED_SEALS
    ):
        raise ProcessSupervisorContractError(
            "trusted supervisor source descriptor lost sealed authority"
        )


@dataclass(frozen=True)
class TrustedSupervisorSource:
    """One import-time, sealed snapshot of the executable supervisor source."""

    descriptor: int
    sha256: str
    size: int

    def acquire(self) -> tuple[int, str]:
        """Return a private launch fd and its fd-bound interpreter pathname."""

        _require_sealed_descriptor(self.descriptor, expected_size=self.size)
        try:
            descriptor = os.dup(self.descriptor)
        except OSError as exc:
            raise ProcessSupervisorContractError(
                "trusted supervisor source could not be leased"
            ) from exc
        try:
            _require_sealed_descriptor(descriptor, expected_size=self.size)
            path = f"/proc/self/fd/{descriptor}"
            if not Path(path).exists():
                raise ProcessSupervisorContractError(
                    "fd-bound supervisor source path is unavailable"
                )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor, path


def _trusted_supervisor_source_from_payload(
    payload: bytes,
    *,
    logical_origin: str,
) -> TrustedSupervisorSource:
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > SUPERVISOR_SOURCE_MAX_BYTES
    ):
        raise ProcessSupervisorContractError(
            "supervisor source is not bounded exact bytes"
        )
    try:
        decoded = payload.decode("utf-8", errors="strict")
        compile(decoded, logical_origin, "exec", dont_inherit=True)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ProcessSupervisorContractError(
            "supervisor source is not valid trusted Python"
        ) from exc
    descriptor = _write_and_seal(
        payload,
        name="crucible-process-supervisor-source",
    )
    return TrustedSupervisorSource(
        descriptor=descriptor,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


def capture_trusted_supervisor_source(path: Path) -> TrustedSupervisorSource:
    """Open, authenticate, copy, compile, and seal one exact source snapshot."""

    if sys.platform != "linux":
        raise ProcessSupervisorContractError(
            "trusted supervisor source capture requires Linux"
        )
    parent = path.parent
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    source_descriptor = -1
    try:
        source_descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > SUPERVISOR_SOURCE_MAX_BYTES
            or before.st_uid != os.geteuid()
        ):
            raise ProcessSupervisorContractError(
                "supervisor source is not an owned bounded regular file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(source_descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise ProcessSupervisorContractError(
                    "supervisor source ended before its authenticated size"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(source_descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before:
            raise ProcessSupervisorContractError(
                "supervisor source changed during trusted capture"
            )
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        os.close(parent_descriptor)
    return _trusted_supervisor_source_from_payload(
        b"".join(chunks),
        logical_origin=str(path),
    )


def sealed_target_environment_descriptor(env: Mapping[str, str]) -> int:
    """Return one sealed, bounded descriptor carrying target-only env data."""

    if sys.platform != "linux":
        raise ProcessSupervisorContractError(
            "runner target environment requires a sealed Linux descriptor"
        )
    projected: dict[str, str] = {}
    for key, value in env.items():
        if (
            not isinstance(key, str)
            or not key
            or "\x00" in key
            or "=" in key
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ValueError(
                "runner target environment requires non-empty string keys, "
                "string values, and no NUL or '=' in keys"
            )
        projected[key] = value
    payload = json.dumps(
        {
            "schema_version": TARGET_ENV_SCHEMA,
            "env": projected,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > TARGET_ENV_MAX_BYTES:
        raise ValueError(
            "runner target environment exceeds the bounded transport size"
        )

    return _write_and_seal(payload, name="crucible-runner-target-env")


def open_target_working_directory(cwd: Path | str) -> int:
    """Pin one exact directory inode for supervisor and target execution."""

    try:
        resolved = Path(cwd).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProcessSupervisorContractError(
            "runner target working directory cannot be resolved"
        ) from exc
    flags = (
        getattr(os, "O_PATH", os.O_RDONLY)
        | os.O_DIRECTORY
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
    )
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ProcessSupervisorContractError(
            "runner target working directory cannot be pinned"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        named = os.stat(resolved, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ProcessSupervisorContractError(
                "runner target working directory changed during capture"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


TRUSTED_SUPERVISOR_SOURCE = capture_trusted_supervisor_source(
    Path(__file__).with_name("_process_supervisor.py")
)


def acquire_trusted_supervisor_source() -> tuple[int, str]:
    """Lease the import-time supervisor snapshot for one exact launch."""

    return TRUSTED_SUPERVISOR_SOURCE.acquire()


__all__ = [
    "ProcessSupervisorContractError",
    "SUPERVISOR_ENV",
    "SUPERVISOR_SOURCE_MAX_BYTES",
    "TARGET_ENV_MAX_BYTES",
    "TARGET_ENV_SCHEMA",
    "TRUSTED_SUPERVISOR_SOURCE",
    "TrustedSupervisorSource",
    "acquire_trusted_supervisor_source",
    "capture_trusted_supervisor_source",
    "open_target_working_directory",
    "sealed_target_environment_descriptor",
    "supervisor_environment",
    "supervisor_interpreter_path",
]
