# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Crucible tools — per-domain ``assay.ToolSet`` implementations.

Each subpackage under ``crucible.tools.<domain>/`` is a self-contained
collection of related tools for source navigation and bounded authoring
that exposes a builder and a ``build_<domain>_tools(context)`` factory. The
central
``CompositeToolSet`` in this package's ``_composite`` module merges N
of them when a workflow wants a specific combination.

Public API:
- ``ToolSetBuilder`` (in ``_builder``) — decorator-based registry for
  schema + handler pairs. Used by each subpackage.
- ``CompositeToolSet`` (in ``_composite``) — merges several ToolSets
  into one, with name-based dispatch routing.
- ``ToolContext`` (in ``_context``) — shared state passed as the first
  positional argument to every handler. Holds workspace_root,
  repo_root, runner configuration, and mutation scope. Each subpackage
  declares which fields it actually reads via type-narrowing.
- ``profiles`` — factories for role-scoped composite tool sets such as
  ``build_autonomous_implementer_profile`` and ``build_breach_defender_profile``.

The public surface is the domain factories plus explicit profile builders;
there is no monolithic compatibility wrapper.
"""

from __future__ import annotations

import ctypes as _ctypes
import hashlib as _hashlib
import importlib.util as _importlib_util
import os as _os
import stat as _stat
import sys as _sys
import threading as _threading
from pathlib import Path as _Path
from types import ModuleType as _ModuleType


# This package initializer is part of Crucible's outer frozen source closure.
# It is deliberately the enforcing loader for the mutation host: replaceable
# host-module bytes are never imported by Python's path loader.  Update this
# digest deliberately whenever ``_isolated_mutation_host.py`` changes.
_ISOLATED_MUTATION_HOST_SOURCE_SHA256 = (
    "301beae4ccd496b347e694c64c100b35398aa9e5a85138bdc5386cef3eba8585"
)
_ISOLATED_MUTATION_HOST_NAME = __name__ + "._isolated_mutation_host"
_ISOLATED_MUTATION_HOST_PATH = _Path(__file__).with_name("_isolated_mutation_host.py")
_ISOLATED_MUTATION_REQUIRED_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
_ISOLATED_MUTATION_F_ADD_SEALS = 1033
_ISOLATED_MUTATION_F_GET_SEALS = 1034
_ISOLATED_MUTATION_SOURCE_LOCK = _threading.RLock()
_ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN = False
_ISOLATED_MUTATION_HOST_DESCRIPTOR: int | None = None
_ISOLATED_MUTATION_HOST_IDENTITY: tuple[int, ...] | None = None
_ISOLATED_MUTATION_HOST_DIGEST: str | None = None
_ISOLATED_MUTATION_HOST_SEALS: int | None = None
_ISOLATED_MUTATION_HOST_PAYLOAD: bytes | None = None


def _isolated_source_identity(metadata: _os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
    )


def _read_exact_descriptor(descriptor: int, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        chunk = _os.pread(
            descriptor, min(1024 * 1024, size - len(payload)), len(payload)
        )
        if not chunk:
            raise ImportError("isolated mutation host source ended early")
        payload.extend(chunk)
    if _os.pread(descriptor, 1, size):
        raise ImportError("isolated mutation host source exceeded its identity")
    return bytes(payload)


def _snapshot_isolated_mutation_host_source() -> tuple[bytes, str]:
    """Bind trusted host bytes without importing any platform-only module."""

    global _ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN
    source_before = _os.stat(
        _ISOLATED_MUTATION_HOST_PATH,
        follow_symlinks=False,
    )
    if (
        not _stat.S_ISREG(source_before.st_mode)
        or source_before.st_nlink != 1
        or source_before.st_size <= 0
    ):
        raise ImportError("isolated mutation host source is not one regular file")
    flags = _os.O_RDONLY
    for flag_name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= int(getattr(_os, flag_name, 0))
    source_descriptor: int | None = None
    _ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN = True
    try:
        source_descriptor = _os.open(_ISOLATED_MUTATION_HOST_PATH, flags)
    except BaseException as exc:
        raise ImportError(
            "isolated mutation host source acquisition is uncertain"
        ) from exc
    primary_error: BaseException | None = None
    payload = b""
    digest = ""
    try:
        _os.set_inheritable(source_descriptor, False)
        source_identity = _isolated_source_identity(source_before)
        if _isolated_source_identity(_os.fstat(source_descriptor)) != source_identity:
            raise ImportError("isolated mutation host source changed while opening")
        payload = _read_exact_descriptor(source_descriptor, source_before.st_size)
        if _isolated_source_identity(_os.fstat(source_descriptor)) != source_identity:
            raise ImportError("isolated mutation host source changed while reading")
        digest = _hashlib.sha256(payload).hexdigest()
        if digest != _ISOLATED_MUTATION_HOST_SOURCE_SHA256:
            raise ImportError("isolated mutation host source failed its trusted digest")
    except BaseException as exc:
        primary_error = exc
    finally:
        acquired = source_descriptor
        source_descriptor = None
        try:
            _os.close(acquired)
        except BaseException as exc:
            raise ImportError(
                "isolated mutation host source descriptor settlement is uncertain"
            ) from exc
    _ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN = False
    if primary_error is not None:
        if isinstance(primary_error, ImportError):
            raise primary_error
        raise ImportError(
            "isolated mutation host source snapshot failed"
        ) from primary_error
    return payload, digest


def _prepare_isolated_mutation_host_source_locked() -> tuple[
    int,
    tuple[int, ...],
    str,
    int,
    bytes,
]:
    """Acquire, verify, and write-seal host bytes before their first import."""

    global _ISOLATED_MUTATION_HOST_DESCRIPTOR
    global _ISOLATED_MUTATION_HOST_DIGEST
    global _ISOLATED_MUTATION_HOST_IDENTITY
    global _ISOLATED_MUTATION_HOST_PAYLOAD
    global _ISOLATED_MUTATION_HOST_SEALS
    global _ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN
    if _ISOLATED_MUTATION_HOST_DESCRIPTOR is not None:
        if (
            _ISOLATED_MUTATION_HOST_IDENTITY is None
            or _ISOLATED_MUTATION_HOST_DIGEST is None
            or _ISOLATED_MUTATION_HOST_SEALS is None
            or _ISOLATED_MUTATION_HOST_PAYLOAD is None
        ):
            raise ImportError("isolated mutation host source state is incomplete")
        return (
            _ISOLATED_MUTATION_HOST_DESCRIPTOR,
            _ISOLATED_MUTATION_HOST_IDENTITY,
            _ISOLATED_MUTATION_HOST_DIGEST,
            _ISOLATED_MUTATION_HOST_SEALS,
            _ISOLATED_MUTATION_HOST_PAYLOAD,
        )
    if _ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN:
        raise ImportError("isolated mutation host source acquisition remains uncertain")
    payload = _ISOLATED_MUTATION_HOST_PAYLOAD
    digest = _ISOLATED_MUTATION_HOST_DIGEST
    if payload is None or digest is None:
        payload, digest = _snapshot_isolated_mutation_host_source()
        _ISOLATED_MUTATION_HOST_PAYLOAD = payload
        _ISOLATED_MUTATION_HOST_DIGEST = digest
    try:
        import fcntl as local_fcntl
    except ImportError as exc:
        raise ImportError(
            "isolated mutation host sealing is unavailable on this platform"
        ) from exc
    sealed_descriptor: int | None = None
    _ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN = True
    try:
        create = _ctypes.CDLL(None, use_errno=True).memfd_create
        create.restype = _ctypes.c_int
        sealed_descriptor = int(
            create(
                b"crucible-isolated-mutation-source",
                0x0001 | 0x0002,
            )
        )
        if sealed_descriptor < 0:
            error = _ctypes.get_errno()
            sealed_descriptor = None
            _ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN = False
            raise OSError(error, _os.strerror(error))
    except BaseException as exc:
        if not _ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN:
            raise ImportError(
                "isolated mutation host memfd acquisition failed"
            ) from exc
        raise ImportError(
            "isolated mutation host memfd acquisition is uncertain"
        ) from exc
    try:
        view = memoryview(payload)
        while view:
            written = _os.write(sealed_descriptor, view)
            if written <= 0:
                raise OSError("short write while sealing mutation host source")
            view = view[written:]
        _os.fsync(sealed_descriptor)
        local_fcntl.fcntl(
            sealed_descriptor,
            _ISOLATED_MUTATION_F_ADD_SEALS,
            _ISOLATED_MUTATION_REQUIRED_SEALS,
        )
        seals = int(
            local_fcntl.fcntl(
                sealed_descriptor,
                _ISOLATED_MUTATION_F_GET_SEALS,
            )
        )
        sealed_identity = _isolated_source_identity(_os.fstat(sealed_descriptor))
        if (
            seals & _ISOLATED_MUTATION_REQUIRED_SEALS
            != _ISOLATED_MUTATION_REQUIRED_SEALS
            or _os.get_inheritable(sealed_descriptor)
            or _hashlib.sha256(
                _read_exact_descriptor(sealed_descriptor, len(payload))
            ).hexdigest()
            != digest
        ):
            raise ImportError("isolated mutation host sealing failed")
    except BaseException as primary_error:
        acquired = sealed_descriptor
        sealed_descriptor = None
        try:
            _os.close(acquired)
        except BaseException as close_error:
            raise ImportError(
                "isolated mutation host memfd settlement is uncertain"
            ) from close_error
        _ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN = False
        raise primary_error
    _ISOLATED_MUTATION_HOST_DESCRIPTOR = sealed_descriptor
    _ISOLATED_MUTATION_HOST_IDENTITY = sealed_identity
    _ISOLATED_MUTATION_HOST_DIGEST = digest
    _ISOLATED_MUTATION_HOST_SEALS = seals
    _ISOLATED_MUTATION_HOST_PAYLOAD = payload
    sealed_descriptor = None
    _ISOLATED_MUTATION_SOURCE_ACQUISITION_UNCERTAIN = False
    return (
        _ISOLATED_MUTATION_HOST_DESCRIPTOR,
        _ISOLATED_MUTATION_HOST_IDENTITY,
        _ISOLATED_MUTATION_HOST_DIGEST,
        _ISOLATED_MUTATION_HOST_SEALS,
        _ISOLATED_MUTATION_HOST_PAYLOAD,
    )


def _prepare_isolated_mutation_host_source() -> tuple[
    int,
    tuple[int, ...],
    str,
    int,
    bytes,
]:
    with _ISOLATED_MUTATION_SOURCE_LOCK:
        return _prepare_isolated_mutation_host_source_locked()


class _SealedMutationHostLoader:
    """Only loader admitted for the already-verified mutation host image."""

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        del path, target
        if fullname != _ISOLATED_MUTATION_HOST_NAME:
            return None
        return _importlib_util.spec_from_loader(
            fullname,
            self,
            origin=str(_ISOLATED_MUTATION_HOST_PATH),
        )

    def create_module(self, spec):  # noqa: ANN001
        del spec
        return None

    def exec_module(self, module: _ModuleType) -> None:
        with _ISOLATED_MUTATION_SOURCE_LOCK:
            (
                descriptor,
                expected_identity,
                expected_digest,
                expected_seals,
                payload,
            ) = _prepare_isolated_mutation_host_source_locked()
            module.__dict__["__file__"] = str(_ISOLATED_MUTATION_HOST_PATH)
            module.__dict__["__cached__"] = None
            observed = _os.fstat(descriptor)
            try:
                import fcntl as local_fcntl
            except ImportError as exc:
                raise ImportError(
                    "isolated mutation host sealing is unavailable on this platform"
                ) from exc
            seals = int(
                local_fcntl.fcntl(
                    descriptor,
                    _ISOLATED_MUTATION_F_GET_SEALS,
                )
            )
            if (
                _isolated_source_identity(observed) != expected_identity
                or seals != expected_seals
                or _hashlib.sha256(
                    _read_exact_descriptor(
                        descriptor,
                        observed.st_size,
                    )
                ).hexdigest()
                != expected_digest
            ):
                raise ImportError("sealed isolated mutation host authority changed")
            module.__dict__["_PRESEALED_HOST_SOURCE_DESCRIPTOR"] = descriptor
            module.__dict__["_PRESEALED_HOST_SOURCE_IDENTITY"] = expected_identity
            module.__dict__["_PRESEALED_HOST_SOURCE_SHA256"] = expected_digest
            module.__dict__["_PRESEALED_HOST_SOURCE_SEALS"] = expected_seals
            module.__dict__["_PRESEALED_HOST_SOURCE_LOCK"] = (
                _ISOLATED_MUTATION_SOURCE_LOCK
            )
            module.__dict__["_PRESEALED_HOST_MODULE_LOADER"] = self
            code = compile(
                payload,
                str(_ISOLATED_MUTATION_HOST_PATH),
                "exec",
                dont_inherit=True,
            )
            exec(code, module.__dict__)


_ISOLATED_MUTATION_HOST_LOADER = _SealedMutationHostLoader()
_sys.meta_path.insert(0, _ISOLATED_MUTATION_HOST_LOADER)

from ._builder import ToolSetBuilder, BuiltToolSet, RegisteredTool
from ._context import ToolContext
from ._dispatch_budget import (
    BoundedReadOnlyToolSet,
    MutatingToolProgressSnapshot,
    ProgressBoundedMutatingToolSet,
    TerminalToolDispatchError,
    ToolDispatchBudgetExceeded,
    ToolDispatchBudgetSnapshot,
    ToolMutationIndeterminate,
)


def __getattr__(name: str):  # noqa: ANN202
    """Load the Assay-dependent composite surface only when requested."""

    if name == "CompositeToolSet":
        from ._composite import CompositeToolSet

        return CompositeToolSet
    if name == "composite":
        from ._composite import composite

        return composite
    raise AttributeError(name)


__all__ = [
    "ToolSetBuilder",
    "BuiltToolSet",
    "RegisteredTool",
    "CompositeToolSet",
    "BoundedReadOnlyToolSet",
    "MutatingToolProgressSnapshot",
    "ProgressBoundedMutatingToolSet",
    "TerminalToolDispatchError",
    "ToolDispatchBudgetExceeded",
    "ToolDispatchBudgetSnapshot",
    "ToolMutationIndeterminate",
    "composite",
    "ToolContext",
]
