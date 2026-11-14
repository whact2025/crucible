# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Assay-free held-data transport for the isolated process supervisor."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
from dataclasses import dataclass
from typing import Literal


TARGET_HELD_DATA_MAX_BYTES = 16 * 1024 * 1024
TARGET_HELD_DATA_AGGREGATE_MAX_BYTES = 32 * 1024 * 1024
TARGET_HELD_DATA_MAX_COUNT = 8
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


@dataclass(frozen=True, slots=True)
class TargetHeldDataDescriptor:
    """One controller-held descriptor explicitly inherited by the target."""

    descriptor: int
    kind: Literal["sealed_regular"]

    def __post_init__(self) -> None:
        if (
            isinstance(self.descriptor, bool)
            or not isinstance(self.descriptor, int)
            or self.descriptor < 3
            or self.kind != "sealed_regular"
        ):
            raise ValueError("target held-data descriptor is invalid")


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


def sealed_target_data_descriptor(payload: bytes, *, name: str) -> int:
    """Create one immutable data fd for a supervised target."""

    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > TARGET_HELD_DATA_MAX_BYTES
        or not name
        or len(name) > 96
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in name
        )
    ):
        raise ValueError("sealed target data payload or name is invalid")
    return _write_and_seal(payload, name=f"crucible-{name}")


def encode_target_held_data_descriptors(
    descriptors: tuple[TargetHeldDataDescriptor, ...],
) -> str:
    """Validate and encode the closed controller-to-supervisor fd contract."""

    import fcntl  # noqa: PLC0415

    if (
        not isinstance(descriptors, tuple)
        or len(descriptors) > TARGET_HELD_DATA_MAX_COUNT
        or any(
            not isinstance(item, TargetHeldDataDescriptor)
            for item in descriptors
        )
    ):
        raise ValueError("target held-data descriptor family is invalid")
    numbers = tuple(item.descriptor for item in descriptors)
    if len(set(numbers)) != len(numbers):
        raise ValueError("target held-data descriptors must be unique")
    aggregate_size = 0
    for item in descriptors:
        try:
            metadata = os.fstat(item.descriptor)
            status_flags = fcntl.fcntl(item.descriptor, fcntl.F_GETFL)
        except OSError as exc:
            raise ProcessSupervisorContractError(
                "target held-data descriptor is unavailable"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 0
            or metadata.st_size <= 0
            or metadata.st_size > TARGET_HELD_DATA_MAX_BYTES
            or status_flags & getattr(os, "O_PATH", 0) != 0
            or status_flags & os.O_ACCMODE == os.O_WRONLY
        ):
            raise ValueError(
                "target held-data descriptor is not anonymous readable bounded data"
            )
        aggregate_size += metadata.st_size
        if aggregate_size > TARGET_HELD_DATA_AGGREGATE_MAX_BYTES:
            raise ValueError(
                "target held-data descriptor family exceeds its byte cap"
            )
        try:
            seals = fcntl.fcntl(item.descriptor, _F_GET_SEALS)
        except OSError as exc:
            if exc.errno in {errno.EINVAL, errno.ENOTTY}:
                raise ValueError(
                    "target sealed held-data descriptor lost its seals"
                ) from exc
            raise ProcessSupervisorContractError(
                "target sealed held-data descriptor is unavailable"
            ) from exc
        if seals & _REQUIRED_SEALS != _REQUIRED_SEALS:
            raise ValueError(
                "target sealed held-data descriptor lost its seals"
            )
    return ",".join(
        f"{item.descriptor}:{item.kind}"
        for item in descriptors
    )


__all__ = [
    "ProcessSupervisorContractError",
    "TARGET_HELD_DATA_AGGREGATE_MAX_BYTES",
    "TARGET_HELD_DATA_MAX_BYTES",
    "TARGET_HELD_DATA_MAX_COUNT",
    "TargetHeldDataDescriptor",
    "encode_target_held_data_descriptors",
    "sealed_target_data_descriptor",
]
