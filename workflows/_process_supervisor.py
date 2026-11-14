#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Linux subreaper for benchmark authoring and verification processes.

This file intentionally imports only the standard library and supports
isolated ``python -I -S`` execution. The controller passes one private control
descriptor, one sealed target-environment descriptor, its sealed source
descriptor, a settlement grace, its expected controller PID, a monotonic target
deadline, an exact target working directory, an optional controller-held target
executable descriptor, an optional held auxiliary executable descriptor, and
an optional closed family of held target-data descriptors, and the target argv.
The auxiliary descriptor is admitted only beside a held target and only when
exactly one target argument names its ``/proc/self/fd`` alias.
Target stdout/stderr are inherited; the control descriptor carries the sole
terminal attestation.
"""

from __future__ import annotations

import ctypes

import errno

import fcntl

import hashlib

import json

import math

import os

import re

import select

import signal

import socket

import stat


import subprocess

import sys

import time


from pathlib import Path

from typing import Any, Callable

PROTOCOL_VERSION = "crucible-process-supervisor 9"

SCHEMA_VERSION = "crucible.process-supervisor.v1"

CGROUP_SETTLEMENT_SCHEMA_VERSION = "crucible.process-supervisor.v3"

TARGET_READY_SCHEMA_VERSION = "crucible.process-supervisor.target-ready.v1"

_TARGET_READY_FLAG = "--report-target-ready-v1"

_CGROUP_ADMISSION_SCHEMA = "crucible.process-supervisor-cgroup.v2"

_CGROUP2_SUPER_MAGIC = 0x63677270

HANDSHAKE_READY_SCHEMA = "crucible.process-supervisor.ready.v1"

HANDSHAKE_COMMAND_SCHEMA = "crucible.process-supervisor.command.v1"

HANDSHAKE_TERMINAL_SCHEMA = "crucible.process-supervisor.terminal.v1"

_PR_SET_PDEATHSIG = 1

_PR_SET_DUMPABLE = 4

_PR_SET_CHILD_SUBREAPER = 36

_MAX_CONTROL_BYTES = 4096

_MAX_CGROUP_CONTROL_BYTES = 1024

_TARGET_ENV_SCHEMA = "crucible.runner-target-env.v1"

_MAX_TARGET_ENV_BYTES = 256 * 1024

_MAX_HELD_DATA_BYTES = 16 * 1024 * 1024

_MAX_HELD_DATA_AGGREGATE_BYTES = 32 * 1024 * 1024

_MAX_HELD_DATA_COUNT = 8

_F_GET_SEALS = 1034

_F_SEAL_SEAL = 0x0001

_F_SEAL_SHRINK = 0x0002

_F_SEAL_GROW = 0x0004

_F_SEAL_WRITE = 0x0008

_REQUIRED_ENV_SEALS = _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE

_PORTABLE_ID = re.compile(r"[A-Za-z0-9._:-]{1,256}")

_HELD_DATA_ITEM = re.compile(r"([0-9]+):(sealed_regular)")

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")

_PIDFD_SYSCALLS = {
    "aarch64": (434, 424),
    "arm64": (434, 424),
    "i386": (434, 424),
    "i686": (434, 424),
    "riscv64": (434, 424),
    "x86_64": (434, 424),
}


class SupervisorError(OSError):
    """The kernel descendant boundary could not be proved."""


class _StatFs(ctypes.Structure):
    _fields_ = (
        ("f_type", ctypes.c_long),
        ("f_bsize", ctypes.c_long),
        ("f_blocks", ctypes.c_ulong),
        ("f_bfree", ctypes.c_ulong),
        ("f_bavail", ctypes.c_ulong),
        ("f_files", ctypes.c_ulong),
        ("f_ffree", ctypes.c_ulong),
        ("f_fsid", ctypes.c_int * 2),
        ("f_namelen", ctypes.c_long),
        ("f_frsize", ctypes.c_long),
        ("f_flags", ctypes.c_long),
        ("f_spare", ctypes.c_long * 4),
    )


def _parse_held_data_descriptors(raw: str) -> tuple[tuple[int, str], ...]:
    if not raw:
        raise ValueError("held target-data descriptor family is empty")
    values: list[tuple[int, str]] = []
    for encoded in raw.split(","):
        match = _HELD_DATA_ITEM.fullmatch(encoded)
        if match is None:
            raise ValueError("held target-data descriptor family is malformed")
        descriptor_text, kind = match.groups()
        descriptor = int(descriptor_text)
        if descriptor < 3:
            raise ValueError("held target-data descriptor family is malformed")
        values.append((descriptor, kind))
    if len(values) > _MAX_HELD_DATA_COUNT or len(
        {descriptor for descriptor, _kind in values}
    ) != len(values):
        raise ValueError("held target-data descriptor family is invalid")
    return tuple(values)


def _require_unique_inherited_descriptor_family(
    *,
    control_descriptor: int,
    target_env_descriptor: int,
    supervisor_source_descriptor: int,
    target_cwd_descriptor: int,
    target_executable_descriptor: int | None,
    auxiliary_executable_descriptor: int | None,
    held_data_descriptors: tuple[tuple[int, str], ...],
    cgroup_control_descriptor: int | None = None,
    target_cgroup_descriptor: int | None = None,
) -> None:
    """Reject every alias across the complete target-inherited fd family."""

    descriptors = [
        control_descriptor,
        target_env_descriptor,
        supervisor_source_descriptor,
        target_cwd_descriptor,
    ]
    if target_executable_descriptor is not None:
        descriptors.append(target_executable_descriptor)
    if auxiliary_executable_descriptor is not None:
        descriptors.append(auxiliary_executable_descriptor)
    if cgroup_control_descriptor is not None:
        descriptors.append(cgroup_control_descriptor)
    if target_cgroup_descriptor is not None:
        descriptors.append(target_cgroup_descriptor)
    descriptors.extend(descriptor for descriptor, _kind in held_data_descriptors)
    if min(descriptors) < 3 or len(set(descriptors)) != len(descriptors):
        raise ValueError("inherited process supervisor descriptors are aliased")


def _authenticate_held_data_descriptor(descriptor: int, kind: str) -> int:
    try:
        metadata = os.fstat(descriptor)
        status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    except OSError as exc:
        raise SupervisorError("held target-data descriptor is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 0
        or metadata.st_size <= 0
        or metadata.st_size > _MAX_HELD_DATA_BYTES
        or status_flags & getattr(os, "O_PATH", 0) != 0
        or status_flags & os.O_ACCMODE == os.O_WRONLY
    ):
        raise SupervisorError(
            "held target-data descriptor is not anonymous readable bounded data"
        )
    try:
        seals = fcntl.fcntl(descriptor, _F_GET_SEALS)
    except OSError as exc:
        raise SupervisorError(
            "held sealed target-data descriptor is unavailable"
        ) from exc
    if kind != "sealed_regular" or seals & _REQUIRED_ENV_SEALS != _REQUIRED_ENV_SEALS:
        raise SupervisorError("held sealed target-data descriptor lost its seals")
    return metadata.st_size


def _authenticate_control_descriptor(descriptor: int) -> None:
    """Reject legacy pipe/stream transports before target construction."""

    try:
        control = socket.socket(fileno=descriptor)
    except OSError as exc:
        raise SupervisorError(
            "process supervisor control transport is unavailable"
        ) from exc
    try:
        if control.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != (
            socket.SOCK_SEQPACKET
        ):
            raise SupervisorError(
                "process supervisor control transport is not seqpacket"
            )
    finally:
        control.detach()


def _require_cgroup2(descriptor: int) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fstatfs = libc.fstatfs
    except (AttributeError, OSError) as exc:
        raise SupervisorError("cgroup fstatfs is unavailable") from exc
    fstatfs.argtypes = (ctypes.c_int, ctypes.POINTER(_StatFs))
    fstatfs.restype = ctypes.c_int
    result = _StatFs()
    if fstatfs(descriptor, ctypes.byref(result)) != 0:
        raise SupervisorError("cannot authenticate target cgroup")
    if int(result.f_type) != _CGROUP2_SUPER_MAGIC:
        raise SupervisorError("target cgroup is not cgroup v2")


def _read_cgroup_file(descriptor: int, name: str) -> bytes:
    try:
        file_descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        try:
            payload = os.read(file_descriptor, _MAX_CONTROL_BYTES + 1)
            if len(payload) > _MAX_CONTROL_BYTES:
                raise SupervisorError("target cgroup control file is oversized")
            if os.read(file_descriptor, 1):
                raise SupervisorError("target cgroup control file is oversized")
            return payload
        finally:
            os.close(file_descriptor)
    except OSError as exc:
        raise SupervisorError("cannot read target cgroup control") from exc


def _write_cgroup_file(descriptor: int, name: str, payload: bytes) -> None:
    try:
        file_descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(file_descriptor, view)
                if written <= 0:
                    raise SupervisorError(
                        "target cgroup membership write made no progress"
                    )
                view = view[written:]
        finally:
            os.close(file_descriptor)
    except OSError as exc:
        raise SupervisorError("cannot enter target cgroup") from exc


def _canonical_cgroup_packet(record: dict[str, object]) -> bytes:
    payload = (
        json.dumps(
            record,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    if len(payload) > _MAX_CGROUP_CONTROL_BYTES:
        raise SupervisorError("cgroup admission packet is oversized")
    return payload


def _decode_cgroup_packet(payload: bytes) -> dict[str, object]:
    if not payload or len(payload) > _MAX_CGROUP_CONTROL_BYTES:
        raise SupervisorError("cgroup admission packet is missing or oversized")

    def reject_duplicates(pairs):
        record = {}
        for key, value in pairs:
            if key in record:
                raise ValueError("duplicate cgroup admission field")
            record[key] = value
        return record

    try:
        record = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SupervisorError("cgroup admission packet is malformed") from exc
    if type(record) is not dict or _canonical_cgroup_packet(record) != payload:
        raise SupervisorError("cgroup admission packet is not canonical")
    return record


def _enter_target_cgroup(
    target_descriptor: int,
    control_descriptor: int,
    *,
    deadline_monotonic_ns: int,
) -> str:
    """Join one descriptor-bound leaf and wait for controller admission."""

    try:
        metadata = os.fstat(target_descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077 != 0
        ):
            raise SupervisorError("target cgroup identity is invalid")
        _require_cgroup2(target_descriptor)
        if (
            _read_cgroup_file(target_descriptor, "cgroup.max.depth") != b"0\n"
            or _read_cgroup_file(target_descriptor, "cgroup.max.descendants") != b"0\n"
        ):
            raise SupervisorError("target cgroup descendant limits are invalid")
        _write_cgroup_file(target_descriptor, "cgroup.procs", b"0\n")
        process_id = os.getpid()
        start_token = _process_start_time(process_id)
        if (
            start_token is None
            or _read_cgroup_file(target_descriptor, "cgroup.procs")
            != f"{process_id}\n".encode("ascii")
            or _read_cgroup_file(target_descriptor, "cgroup.threads")
            != f"{process_id}\n".encode("ascii")
        ):
            raise SupervisorError("target cgroup membership is not exact")
        cgroup_identity = (
            f"{os.major(metadata.st_dev)}:{os.minor(metadata.st_dev)}:{metadata.st_ino}"
        )
        ready = _canonical_cgroup_packet(
            {
                "cgroup_identity": cgroup_identity,
                "phase": "ready",
                "process_id": process_id,
                "schema_version": _CGROUP_ADMISSION_SCHEMA,
                "start_token": start_token,
            }
        )
        control = socket.socket(fileno=control_descriptor)
        try:
            if control.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != (
                socket.SOCK_SEQPACKET
            ):
                raise SupervisorError("cgroup admission transport is not seqpacket")
            if control.send(ready) != len(ready):
                raise SupervisorError("cgroup readiness packet was truncated")
            remaining_ns = deadline_monotonic_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                raise SupervisorError("cgroup admission deadline expired")
            control.settimeout(remaining_ns / 1_000_000_000)
            acknowledgement = control.recv(_MAX_CGROUP_CONTROL_BYTES + 1)
            record = _decode_cgroup_packet(acknowledgement)
            admission_nonce = record.get("admission_nonce")
            if (
                set(record)
                != {
                    "admission_nonce",
                    "cgroup_identity",
                    "phase",
                    "process_id",
                    "schema_version",
                    "start_token",
                }
                or type(admission_nonce) is not str
                or re.fullmatch(r"[0-9a-f]{64}", admission_nonce) is None
                or record.get("cgroup_identity") != cgroup_identity
                or record.get("phase") != "admit"
                or record.get("process_id") != process_id
                or record.get("schema_version") != _CGROUP_ADMISSION_SCHEMA
                or record.get("start_token") != start_token
            ):
                raise SupervisorError("cgroup admission acknowledgement is invalid")
            return "sha256:" + hashlib.sha256(ready + acknowledgement).hexdigest()
        except OSError as exc:
            raise SupervisorError("cgroup admission transport failed") from exc
        finally:
            control.detach()
    except OSError as exc:
        if isinstance(exc, SupervisorError):
            raise
        raise SupervisorError("target cgroup admission failed") from exc


def _read_target_environment(descriptor: int) -> dict[str, str]:
    """Authenticate and parse one sealed controller-to-target environment."""

    try:
        metadata = os.fstat(descriptor)
        seals = fcntl.fcntl(descriptor, _F_GET_SEALS)
    except OSError as exc:
        raise SupervisorError("cannot authenticate target environment") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > _MAX_TARGET_ENV_BYTES
        or seals & _REQUIRED_ENV_SEALS != _REQUIRED_ENV_SEALS
    ):
        raise SupervisorError("target environment descriptor is not sealed")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise SupervisorError("target environment ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    except OSError as exc:
        raise SupervisorError("cannot read target environment") from exc

    def reject_duplicates(pairs):
        record = {}
        for key, value in pairs:
            if key in record:
                raise ValueError("duplicate target environment key")
            record[key] = value
        return record

    try:
        record = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SupervisorError("target environment record is malformed") from exc
    if (
        not isinstance(record, dict)
        or set(record) != {"schema_version", "env"}
        or record.get("schema_version") != _TARGET_ENV_SCHEMA
        or not isinstance(record.get("env"), dict)
    ):
        raise SupervisorError("target environment record has invalid schema")
    environment = record["env"]
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or not key
            or "\x00" in key
            or "=" in key
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise SupervisorError("target environment entry is invalid")
    return environment


def _enter_target_working_directory(descriptor: int) -> None:
    """Authenticate and enter the controller-pinned directory inode."""

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise SupervisorError("target working descriptor is not a directory")
        os.fchdir(descriptor)
        after = os.stat(".", follow_symlinks=False)
    except OSError as exc:
        raise SupervisorError("cannot enter target working directory") from exc
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise SupervisorError("target working directory identity changed")


def _establish_kernel_boundaries(expected_parent_id: int) -> None:
    if sys.platform != "linux":
        raise SupervisorError("descendant supervision requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    # Establish the control-fd confidentiality boundary first. Even a later
    # subreaper/PDEATHSIG setup failure must not leave a same-uid process able
    # to reopen the descriptor and forge the failure receipt through /proc.
    if prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise SupervisorError("cannot establish kernel supervision boundaries")
    if (
        prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0
        or prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0
    ):
        raise SupervisorError("cannot establish kernel supervision boundaries")
    # PR_SET_PDEATHSIG is not retroactive. Comparing the expected parent after
    # the prctl closes the fork-to-prctl race without ever spawning a target.
    if os.getppid() != expected_parent_id:
        raise SupervisorError("controller disappeared before supervision")


def _install_target_parent_death_boundary(expected_parent_id: int) -> None:
    """Fail-stop the direct target if its supervisor disappears around fork."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = (
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        )
        prctl.restype = ctypes.c_int
        if (
            prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0
            or os.getppid() != expected_parent_id
        ):
            os._exit(126)
    except BaseException:
        # This runs after fork and before exec in the supervisor's direct
        # child. Never raise through subprocess's error pipe or run Python
        # cleanup with an unproved parent-death boundary.
        os._exit(126)


def _process_identity_state(process_id: int) -> tuple[int, str] | None:
    """Return one start-time-bound process state from a single procfs read."""

    try:
        payload = Path(f"/proc/{process_id}/stat").read_text(
            encoding="utf-8",
            errors="strict",
        )
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise SupervisorError("cannot inspect supervised process identity") from exc
    close = payload.rfind(")")
    fields = payload[close + 2 :].split() if close >= 0 else []
    try:
        state = fields[0]
        start_time = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise SupervisorError("supervised process identity is malformed") from exc
    if len(state) != 1:
        raise SupervisorError("supervised process identity is malformed")
    return start_time, state


def _process_start_time(process_id: int) -> int | None:
    observed = _process_identity_state(process_id)
    return None if observed is None or observed[1] in {"Z", "X", "x"} else observed[0]


def _establish_generic_child_wait_authority() -> None:
    """Require and re-establish the default waitable-child disposition."""

    try:
        incoming = signal.getsignal(signal.SIGCHLD)
    except (OSError, ValueError) as exc:
        raise SupervisorError("generic target wait authority is unavailable") from exc
    if incoming is not signal.SIG_DFL:
        raise SupervisorError("generic target SIGCHLD disposition differs")
    try:
        # Reinstall DFL even when it is already observable.  This clears an
        # inherited SA_NOCLDWAIT flag that Python's getsignal cannot expose.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        observed = signal.getsignal(signal.SIGCHLD)
    except (OSError, ValueError) as exc:
        raise SupervisorError("generic target wait authority is unavailable") from exc
    if observed is not signal.SIG_DFL:
        raise SupervisorError("generic target SIGCHLD disposition differs")


def _unreaped_child_start_time(process_id: int) -> int | None:
    """Return the stable identity of this Supervisor's direct child.

    A very short-lived target can become a zombie before ``Popen`` returns.
    Its procfs start time remains immutable and its PID cannot be reused while
    this Supervisor retains sole wait authority, so that identity is still
    authoritative for terminal settlement.  General liveness checks continue
    to reject zombies through ``_process_start_time``.
    """

    observed = _process_identity_state(process_id)
    if observed is None or observed[0] <= 0 or observed[1] in {"X", "x"}:
        return None
    try:
        wait_status = os.waitid(
            os.P_PID,
            process_id,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError:
        return None
    except OSError as exc:
        raise SupervisorError("cannot verify generic target wait authority") from exc
    if wait_status is not None and wait_status.si_pid != process_id:
        raise SupervisorError("generic target wait identity differs")
    if observed[1] == "Z" and wait_status is None:
        return None
    return observed[0]


def _direct_process_children(process_id: int) -> tuple[int, ...]:
    task_root = Path(f"/proc/{process_id}/task")
    try:
        task_ids = tuple(os.listdir(task_root))
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise SupervisorError("cannot census supervised process tasks") from exc
    children: set[int] = set()
    for task_id in task_ids:
        if not task_id.isdigit():
            continue
        try:
            raw = (task_root / task_id / "children").read_text(encoding="ascii")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError) as exc:
            raise SupervisorError("cannot census supervised process children") from exc
        try:
            children.update(int(item) for item in raw.split())
        except ValueError as exc:
            raise SupervisorError("supervised child census is malformed") from exc
    return tuple(sorted(children))


def _supervised_descendants() -> tuple[tuple[tuple[int, int], str], ...]:
    """Census descendants with state bound to the same read as start time."""

    supervisor_id = os.getpid()
    pending = [supervisor_id]
    seen = {supervisor_id}
    observations: dict[int, tuple[int, str]] = {}
    while pending:
        parent = pending.pop()
        for process_id in _direct_process_children(parent):
            if process_id in seen:
                continue
            seen.add(process_id)
            observed = _process_identity_state(process_id)
            if observed is None:
                continue
            observations[process_id] = observed
            pending.append(process_id)
    return tuple(
        ((process_id, start_time), state)
        for process_id, (start_time, state) in sorted(observations.items())
    )


def _open_pidfd(process_id: int) -> int | None:
    """Pin one process identity or fail closed when pidfds are unavailable."""

    syscall_numbers = _PIDFD_SYSCALLS.get(os.uname().machine.lower())
    if syscall_numbers is None:
        raise SupervisorError("pidfd process signaling is unsupported")
    pidfd_open_number, _pidfd_send_signal_number = syscall_numbers
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    descriptor = syscall(
        ctypes.c_long(pidfd_open_number),
        ctypes.c_int(process_id),
        ctypes.c_uint(0),
    )
    if descriptor >= 0:
        return int(descriptor)
    error_number = ctypes.get_errno()
    if error_number == errno.ESRCH:
        return None
    raise SupervisorError("cannot pin supervised process identity with pidfd")


def _pidfd_send_signal(descriptor: int, signal_number: int) -> bool:
    """Signal only the kernel object pinned by ``descriptor``."""

    syscall_numbers = _PIDFD_SYSCALLS.get(os.uname().machine.lower())
    if syscall_numbers is None:
        raise SupervisorError("pidfd process signaling is unsupported")
    _pidfd_open_number, pidfd_send_signal_number = syscall_numbers
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    result = syscall(
        ctypes.c_long(pidfd_send_signal_number),
        ctypes.c_int(descriptor),
        ctypes.c_int(signal_number),
        ctypes.c_void_p(),
        ctypes.c_uint(0),
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number == errno.ESRCH:
        return False
    raise SupervisorError("cannot signal pinned supervised process identity")


def _signal_identity(identity: tuple[int, int], signal_number: int) -> None:
    process_id, start_time = identity
    descriptor = _open_pidfd(process_id)
    if descriptor is None:
        return
    try:
        # Opening the pidfd first pins the kernel process object. The
        # diagnostic start-time reread rejects a census identity that was
        # already recycled; any exit/reuse after this point cannot redirect
        # pidfd_send_signal to the replacement PID.
        if _process_start_time(process_id) != start_time:
            return
        _pidfd_send_signal(descriptor, signal_number)
    finally:
        os.close(descriptor)


def _reap_adopted_children() -> bool:
    while True:
        try:
            process_id, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return False
        if process_id == 0:
            return True


def _reap_adopted_children_except(excluded_process_id: int) -> None:
    """Reap exited adopted children without racing the direct target owner."""

    if excluded_process_id <= 1:
        raise SupervisorError("supervised target identity is invalid")
    for process_id in _direct_process_children(os.getpid()):
        if process_id == excluded_process_id:
            continue
        try:
            reaped, _status = os.waitpid(process_id, os.WNOHANG)
        except ChildProcessError:
            # The child may have been reparented or reaped between procfs
            # census and waitpid.  It cannot redirect waitpid to a reused PID
            # that is not our child.
            continue
        if reaped not in {0, process_id}:
            raise SupervisorError("adopted child reap identity differs")


def _settle_descendants(
    target: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
    observe_control: Callable[[], None] | None = None,
) -> tuple[bool, bool]:
    """Settle the full adopted tree and report normal-exit background work."""

    started = time.monotonic()
    term_deadline = started + grace_seconds * 0.25
    kill_deadline = started + grace_seconds * 0.65
    signal_number = signal.SIGTERM
    found_background_descendant = False
    while True:
        if observe_control is not None:
            observe_control()
        target.poll()
        if target.returncode is not None:
            # Dead adopted grandchildren are direct zombies of this
            # subreaper.  Reap them before inventory so a completed process is
            # never classified as background work merely because its parent
            # exited first.
            children_exist = _reap_adopted_children()
        else:
            children_exist = True
        observations = _supervised_descendants()
        identities = tuple(identity for identity, _state in observations)
        if target.returncode is not None and any(
            state not in {"Z", "X", "x"} for _identity, state in observations
        ):
            # Once the direct target has been reaped, every remaining
            # *executing* adopted process is background work that outlived the
            # runner contract. A pre-exited zombie is still retained below for
            # mandatory reap settlement, but it cannot execute late work and
            # must not turn an otherwise successful runner into a false
            # background-descendant failure.
            found_background_descendant = True
        for identity in identities:
            _signal_identity(identity, signal_number)
        target.poll()
        if target.returncode is not None:
            children_exist = _reap_adopted_children()
        remaining = _supervised_descendants()
        if target.returncode is not None and not children_exist and not remaining:
            if observe_control is not None:
                observe_control()
            return True, found_background_descendant
        now = time.monotonic()
        if now >= term_deadline:
            signal_number = signal.SIGKILL
        if now >= kill_deadline:
            return False, found_background_descendant
        time.sleep(0.005)


def _write_control(
    descriptor: int,
    *,
    returncode: int | None,
    settled: bool,
    error: str | None,
    cgroup_bound: bool,
    cgroup_admission_sha256: str | None,
    target_pid: int | None,
    target_start_token: int | None,
) -> None:
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "returncode": returncode,
        "settled": settled,
        "error": error,
    }
    if cgroup_bound:
        if (
            type(cgroup_admission_sha256) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", cgroup_admission_sha256) is None
            or not (
                (target_pid is None and target_start_token is None)
                or (
                    type(target_pid) is int
                    and target_pid > 1
                    and type(target_start_token) is int
                    and target_start_token > 0
                )
            )
        ):
            raise SupervisorError(
                "cgroup or target identity is unavailable for settlement"
            )
        supervisor_pid = os.getpid()
        supervisor_start_token = _process_start_time(supervisor_pid)
        if supervisor_start_token is None or supervisor_start_token <= 0:
            raise SupervisorError(
                "process supervisor identity is unavailable for settlement"
            )
        record.update(
            {
                "cgroup_admission_sha256": cgroup_admission_sha256,
                "schema_version": CGROUP_SETTLEMENT_SCHEMA_VERSION,
                "supervisor_pid": supervisor_pid,
                "supervisor_start_token": supervisor_start_token,
                "target_pid": target_pid,
                "target_start_token": target_start_token,
            }
        )
    payload = (
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_CONTROL_BYTES:
        raise SupervisorError("process supervisor control record is too large")
    control = socket.socket(fileno=descriptor)
    try:
        if control.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != (
            socket.SOCK_SEQPACKET
        ):
            raise SupervisorError(
                "process supervisor control transport is not seqpacket"
            )
        written = control.send(payload)
        if written != len(payload):
            raise SupervisorError("process supervisor control packet was truncated")
    except OSError as exc:
        raise SupervisorError(
            "cannot publish process supervisor control packet"
        ) from exc
    finally:
        control.detach()


def _write_target_ready(
    descriptor: int,
    *,
    cgroup_admission_sha256: str,
    target_pid: int,
    target_start_token: int,
) -> None:
    """Publish one exact post-spawn identity on the private control channel."""

    supervisor_pid = os.getpid()
    supervisor_start_token = _process_start_time(supervisor_pid)
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", cgroup_admission_sha256) is None
        or type(target_pid) is not int
        or target_pid <= 1
        or type(target_start_token) is not int
        or target_start_token <= 0
        or supervisor_start_token is None
        or supervisor_start_token <= 0
    ):
        raise SupervisorError("target-ready identity is unavailable")
    _send_packet(
        descriptor,
        {
            "schema_version": TARGET_READY_SCHEMA_VERSION,
            "phase": "target_ready",
            "cgroup_admission_sha256": cgroup_admission_sha256,
            "supervisor_pid": supervisor_pid,
            "supervisor_start_token": supervisor_start_token,
            "target_pid": target_pid,
            "target_start_token": target_start_token,
        },
    )


def _send_packet(descriptor: int, record: dict[str, object]) -> None:
    """Send one canonical packet on the private sequenced control channel."""

    payload = json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    if not payload or len(payload) > _MAX_CONTROL_BYTES:
        raise SupervisorError("process supervisor control record is too large")
    control = socket.socket(fileno=descriptor)
    try:
        if control.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != (
            socket.SOCK_SEQPACKET
        ):
            raise SupervisorError(
                "process supervisor control transport is not seqpacket"
            )
        written = control.send(payload)
        if written != len(payload):
            raise SupervisorError("process supervisor control packet was truncated")
    except OSError as exc:
        raise SupervisorError(
            "cannot publish process supervisor control packet"
        ) from exc
    finally:
        control.detach()


def _valid_epoch_binding(value: object) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, dict)
        and set(value) == {"epoch_id", "generation", "family_digest"}
        and isinstance(value.get("epoch_id"), str)
        and _PORTABLE_ID.fullmatch(value["epoch_id"]) is not None
        and type(value.get("generation")) is int
        and value["generation"] > 0
        and isinstance(value.get("family_digest"), str)
        and _SHA256.fullmatch(value["family_digest"]) is not None
    )


def _read_handshake_command(
    descriptor: int,
    *,
    launch_id: str,
    deadline_monotonic_ns: int | None,
    stop_requested: Callable[[], bool],
    expected_epoch: object = ...,
    schema_version: str = HANDSHAKE_COMMAND_SCHEMA,
) -> tuple[str, dict[str, object] | None]:
    """Read one authenticated START/CANCEL command with a bounded wait."""

    control = socket.socket(fileno=descriptor)
    try:
        if control.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != (
            socket.SOCK_SEQPACKET
        ):
            raise SupervisorError(
                "process supervisor control transport is not seqpacket"
            )
        while True:
            if stop_requested():
                # A direct supervisor signal is physical teardown only. It is
                # never authenticated controller authority and therefore
                # cannot synthesize a trusted CANCEL terminal, before or after
                # epoch binding.
                raise SupervisorError(
                    "process supervisor stopped without controller authority"
                )
            remaining_ns = (
                None
                if deadline_monotonic_ns is None
                else deadline_monotonic_ns - time.monotonic_ns()
            )
            if remaining_ns is not None and remaining_ns <= 0:
                return "timeout", None
            readable, _writable, _exceptional = select.select(
                (control,),
                (),
                (),
                (
                    0.05
                    if remaining_ns is None
                    else min(0.05, remaining_ns / 1_000_000_000)
                ),
            )
            if readable:
                payload = control.recv(_MAX_CONTROL_BYTES + 1)
                break
    finally:
        control.detach()
    if not payload or len(payload) > _MAX_CONTROL_BYTES:
        raise SupervisorError(
            "process supervisor handshake command is unavailable or oversized"
        )

    def reject_duplicates(pairs):
        record = {}
        for key, value in pairs:
            if key in record:
                raise ValueError("duplicate process supervisor command field")
            record[key] = value
        return record

    try:
        record = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SupervisorError(
            "process supervisor handshake command is malformed"
        ) from exc
    epoch = record.get("epoch") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or set(record) != {"schema_version", "command", "launch_id", "epoch"}
        or record.get("schema_version") != schema_version
        or record.get("command") not in {"start", "cancel"}
        or record.get("launch_id") != launch_id
        or not _valid_epoch_binding(epoch)
        or (expected_epoch is not ... and epoch != expected_epoch)
    ):
        raise SupervisorError("process supervisor handshake command changed authority")
    return str(record["command"]), epoch


def _write_handshake_terminal(
    descriptor: int,
    *,
    launch_id: str,
    epoch: dict[str, object] | None,
    outcome: str,
    target_started: bool,
    returncode: int,
    settled: bool,
    error: str | None,
) -> None:
    _send_packet(
        descriptor,
        {
            "schema_version": HANDSHAKE_TERMINAL_SCHEMA,
            "phase": "terminal",
            "launch_id": launch_id,
            "epoch": epoch,
            "outcome": outcome,
            "target_started": target_started,
            "returncode": returncode,
            "settled": settled,
            "error": error,
        },
    )


def _handshake_command_is_pending(descriptor: int) -> bool:
    """Return whether another controller transition is already queued."""

    control = socket.socket(fileno=descriptor)
    try:
        readable, _writable, _exceptional = select.select((control,), (), (), 0.0)
        return bool(readable)
    finally:
        control.detach()


def _run() -> None:
    stop_requested = False

    def request_stop(_signal_number: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    try:
        if len(sys.argv) < 10:
            raise ValueError("invalid process supervisor argv")
        control_descriptor = int(sys.argv[1])
        target_env_descriptor = int(sys.argv[2])
        supervisor_source_descriptor = int(sys.argv[3])
        target_cwd_descriptor = int(sys.argv[4])
        grace_seconds = float(sys.argv[5])
        expected_parent_id = int(sys.argv[6])
        deadline_monotonic_ns = int(sys.argv[7])
        suffix = tuple(sys.argv[8:])
        cgroup_control_descriptor: int | None = None
        target_cgroup_descriptor: int | None = None
        cgroup_admission_sha256: str | None = None
        if suffix and suffix[0] == "--held-cgroup-fds":
            if len(suffix) < 5:
                raise ValueError("invalid process supervisor cgroup argv")
            cgroup_control_descriptor = int(suffix[1])
            target_cgroup_descriptor = int(suffix[2])
            suffix = suffix[3:]
        report_target_ready = bool(suffix and suffix[0] == _TARGET_READY_FLAG)
        if report_target_ready:
            suffix = suffix[1:]
        target_executable_descriptor: int | None
        auxiliary_executable_descriptor: int | None
        held_data_descriptors: tuple[tuple[int, str], ...] = ()
        handshake_launch_id: str | None = None
        target_argv: tuple[str, ...] = ()
        if len(suffix) >= 4 and suffix[0] == "--controller-handshake-v1":
            target_executable_descriptor = None
            auxiliary_executable_descriptor = None
            handshake_launch_id = suffix[1]
            if suffix[2] == "--":
                target_argv = suffix[3:]
            elif (
                len(suffix) >= 6
                and suffix[2] == "--held-data-fds-v1"
                and (suffix[4] == "--")
            ):
                held_data_descriptors = _parse_held_data_descriptors(suffix[3])
                target_argv = suffix[5:]
            else:
                raise ValueError("invalid process supervisor argv")
        elif suffix and suffix[0] == "--":
            target_executable_descriptor = None
            auxiliary_executable_descriptor = None
            target_argv = suffix[1:]
        if len(suffix) >= 5 and suffix[0] == "--held-executable-fd":
            if len(suffix) < 5 or suffix[0] != "--held-executable-fd":
                raise ValueError("invalid held executable argv")
            target_executable_descriptor = int(suffix[1])
            raw_target_argv0 = suffix[2]
            target_argv0 = Path(raw_target_argv0)
            if suffix[3] == "--":
                auxiliary_executable_descriptor = None
                target_argv = suffix[4:]
            elif (
                len(suffix) >= 7
                and suffix[3] == "--held-auxiliary-executable-fd"
                and (suffix[5] == "--")
            ):
                auxiliary_executable_descriptor = int(suffix[4])
                target_argv = suffix[6:]
            else:
                raise ValueError("invalid process supervisor argv")
            reserved_descriptors = {
                control_descriptor,
                target_env_descriptor,
                supervisor_source_descriptor,
                target_cwd_descriptor,
                *(
                    ()
                    if cgroup_control_descriptor is None
                    else (cgroup_control_descriptor, target_cgroup_descriptor)
                ),
            }
            if (
                target_executable_descriptor < 3
                or target_executable_descriptor in reserved_descriptors
                or (not target_argv0.is_absolute())
                or (".." in target_argv0.parts)
                or ("\x00" in raw_target_argv0)
                or (str(target_argv0) != raw_target_argv0)
                or (not target_argv)
                or (target_argv[0] != raw_target_argv0)
            ):
                raise ValueError("invalid held executable contract")
            if auxiliary_executable_descriptor is not None:
                auxiliary_alias = f"/proc/self/fd/{auxiliary_executable_descriptor}"
                if (
                    auxiliary_executable_descriptor < 3
                    or auxiliary_executable_descriptor
                    in reserved_descriptors | {target_executable_descriptor}
                    or target_argv.count(auxiliary_alias) != 1
                    or (target_argv[0] == auxiliary_alias)
                ):
                    raise ValueError("invalid held auxiliary executable contract")
        elif not target_argv:
            raise ValueError("invalid process supervisor argv")
        authority_descriptors = (
            control_descriptor,
            target_env_descriptor,
            supervisor_source_descriptor,
            target_cwd_descriptor,
            *(
                ()
                if target_executable_descriptor is None
                else (target_executable_descriptor,)
            ),
            *(
                ()
                if auxiliary_executable_descriptor is None
                else (auxiliary_executable_descriptor,)
            ),
            *(descriptor for descriptor, _kind in held_data_descriptors),
            *(
                ()
                if cgroup_control_descriptor is None
                else (cgroup_control_descriptor, target_cgroup_descriptor)
            ),
        )
        _require_unique_inherited_descriptor_family(
            control_descriptor=control_descriptor,
            target_env_descriptor=target_env_descriptor,
            supervisor_source_descriptor=supervisor_source_descriptor,
            target_cwd_descriptor=target_cwd_descriptor,
            target_executable_descriptor=target_executable_descriptor,
            auxiliary_executable_descriptor=auxiliary_executable_descriptor,
            held_data_descriptors=held_data_descriptors,
            cgroup_control_descriptor=cgroup_control_descriptor,
            target_cgroup_descriptor=target_cgroup_descriptor,
        )
        handshake_target_timeout_ns = (
            deadline_monotonic_ns if handshake_launch_id is not None else None
        )
        if (
            (cgroup_control_descriptor is None) != (target_cgroup_descriptor is None)
            or any(
                (type(descriptor) is not int for descriptor in authority_descriptors)
            )
            or min(authority_descriptors) < 3
            or (len(set(authority_descriptors)) != len(authority_descriptors))
            or (not math.isfinite(grace_seconds))
            or (not 0 < grace_seconds <= 10)
            or (expected_parent_id <= 1)
            or (deadline_monotonic_ns <= 0)
            or (
                handshake_launch_id is not None
                and _PORTABLE_ID.fullmatch(handshake_launch_id) is None
            )
            or (not target_argv)
            or any((not item or "\x00" in item for item in target_argv))
            or (report_target_ready and cgroup_control_descriptor is None)
        ):
            raise ValueError("invalid process supervisor contract")

        def write_control(
            *,
            returncode: int | None,
            settled: bool,
            error: str | None,
            target_pid: int | None = None,
            target_start_token: int | None = None,
        ) -> None:
            _write_control(
                control_descriptor,
                returncode=returncode,
                settled=settled,
                error=error,
                cgroup_bound=cgroup_admission_sha256 is not None,
                cgroup_admission_sha256=cgroup_admission_sha256,
                target_pid=target_pid,
                target_start_token=target_start_token,
            )

        def write_handshake_terminal(
            descriptor: int,
            *,
            launch_id: str,
            epoch: dict[str, object] | None,
            outcome: str,
            target_started: bool,
            returncode: int,
            settled: bool,
            error: str | None,
            target_pid: int | None = None,
            target_start_token: int | None = None,
        ) -> None:
            if (
                handshake_launch_id is None
                or descriptor != control_descriptor
                or launch_id != handshake_launch_id
            ):
                raise SupervisorError("handshake terminal lacks launch authority")
            _write_handshake_terminal(
                control_descriptor,
                launch_id=handshake_launch_id,
                epoch=epoch,
                outcome=outcome,
                target_started=target_started,
                returncode=returncode,
                settled=settled,
                error=error,
            )

        for signal_number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signal_number, request_stop)
        try:
            _establish_kernel_boundaries(expected_parent_id)
        except SupervisorError:
            write_control(returncode=None, settled=True, error="kernel_boundary")
            return
        _authenticate_control_descriptor(control_descriptor)
        if cgroup_control_descriptor is not None:
            _authenticate_control_descriptor(cgroup_control_descriptor)
        if target_executable_descriptor is not None:
            target_metadata = os.fstat(target_executable_descriptor)
            if (
                not stat.S_ISREG(target_metadata.st_mode)
                or target_metadata.st_mode & 73 == 0
            ):
                raise ValueError("invalid held executable contract")
        if auxiliary_executable_descriptor is not None:
            auxiliary_metadata = os.fstat(auxiliary_executable_descriptor)
            if (
                not stat.S_ISREG(auxiliary_metadata.st_mode)
                or auxiliary_metadata.st_mode & 73 == 0
            ):
                raise ValueError("invalid held auxiliary executable contract")
        held_data_total = sum(
            (
                _authenticate_held_data_descriptor(descriptor, kind)
                for descriptor, kind in held_data_descriptors
            )
        )
        if held_data_total > _MAX_HELD_DATA_AGGREGATE_BYTES:
            raise ValueError("held target-data descriptor family exceeds its byte cap")
        os.close(supervisor_source_descriptor)
        try:
            target_env = _read_target_environment(target_env_descriptor)
        finally:
            os.close(target_env_descriptor)
        try:
            _enter_target_working_directory(target_cwd_descriptor)
        except BaseException:
            os.close(target_cwd_descriptor)
            raise
        os.close(target_cwd_descriptor)
    except (TypeError, ValueError, OSError):
        return
    handshake_epoch: dict[str, object] | None = None
    if handshake_launch_id is not None:
        _send_packet(
            control_descriptor,
            {
                "schema_version": HANDSHAKE_READY_SCHEMA,
                "phase": "ready",
                "launch_id": handshake_launch_id,
            },
        )
        try:
            command, handshake_epoch = _read_handshake_command(
                control_descriptor,
                launch_id=handshake_launch_id,
                deadline_monotonic_ns=None,
                stop_requested=lambda: stop_requested,
                schema_version=HANDSHAKE_COMMAND_SCHEMA,
            )
        except SupervisorError:
            return
        if command == "cancel":
            write_handshake_terminal(
                control_descriptor,
                launch_id=handshake_launch_id,
                epoch=handshake_epoch,
                outcome="cancelled",
                target_started=False,
                returncode=130,
                settled=True,
                error=None,
            )
            return
        assert command == "start"
        assert handshake_target_timeout_ns is not None
        deadline_monotonic_ns = time.monotonic_ns() + handshake_target_timeout_ns
    elif stop_requested or os.getppid() != expected_parent_id:
        write_control(returncode=None, settled=True, error="parent")
        return
    if time.monotonic_ns() >= deadline_monotonic_ns:
        if handshake_launch_id is not None:
            write_handshake_terminal(
                control_descriptor,
                launch_id=handshake_launch_id,
                epoch=handshake_epoch,
                outcome="timed_out",
                target_started=False,
                returncode=-124,
                settled=True,
                error=None,
            )
        else:
            write_control(returncode=-124, settled=True, error=None)
        return
    if cgroup_control_descriptor is not None:
        if target_cgroup_descriptor is None:
            return
        try:
            cgroup_admission_sha256 = _enter_target_cgroup(
                target_cgroup_descriptor,
                cgroup_control_descriptor,
                deadline_monotonic_ns=deadline_monotonic_ns,
            )
        except SupervisorError:
            if handshake_launch_id is not None:
                write_handshake_terminal(
                    control_descriptor,
                    launch_id=handshake_launch_id,
                    epoch=handshake_epoch,
                    outcome="protocol_failed",
                    target_started=False,
                    returncode=125,
                    settled=True,
                    error="cgroup_boundary",
                )
            else:
                write_control(returncode=None, settled=True, error="cgroup_boundary")
            return
        finally:
            os.close(cgroup_control_descriptor)
            os.close(target_cgroup_descriptor)
    if stop_requested or os.getppid() != expected_parent_id:
        if handshake_launch_id is not None:
            write_handshake_terminal(
                control_descriptor,
                launch_id=handshake_launch_id,
                epoch=handshake_epoch,
                outcome="protocol_failed",
                target_started=False,
                returncode=125,
                settled=True,
                error="parent",
            )
        else:
            write_control(returncode=None, settled=True, error="parent")
        return
    if time.monotonic_ns() >= deadline_monotonic_ns:
        if handshake_launch_id is not None:
            write_handshake_terminal(
                control_descriptor,
                launch_id=handshake_launch_id,
                epoch=handshake_epoch,
                outcome="timed_out",
                target_started=False,
                returncode=-124,
                settled=True,
                error=None,
            )
        else:
            write_control(returncode=-124, settled=True, error=None)
        return
    target_start_token: int | None = None
    try:
        supervisor_id = os.getpid()
        _establish_generic_child_wait_authority()
        held_target_fds = tuple(
            (descriptor for descriptor, _kind in held_data_descriptors)
        )
        try:
            if target_executable_descriptor is None:
                target = subprocess.Popen(
                    target_argv,
                    env=target_env,
                    cwd=".",
                    stdin=None,
                    close_fds=True,
                    start_new_session=True,
                    pass_fds=held_target_fds,
                    preexec_fn=lambda: _install_target_parent_death_boundary(
                        supervisor_id
                    ),
                )
            else:
                target = subprocess.Popen(
                    target_argv,
                    executable=f"/proc/self/fd/{target_executable_descriptor}",
                    env=target_env,
                    cwd=".",
                    stdin=None,
                    close_fds=True,
                    start_new_session=True,
                    pass_fds=(
                        target_executable_descriptor,
                        *(
                            ()
                            if auxiliary_executable_descriptor is None
                            else (auxiliary_executable_descriptor,)
                        ),
                        *held_target_fds,
                        *(),
                        *(),
                    ),
                    preexec_fn=lambda: _install_target_parent_death_boundary(
                        supervisor_id
                    ),
                )
        finally:
            for descriptor in held_target_fds:
                os.close(descriptor)
    except OSError:
        if handshake_launch_id is not None:
            write_handshake_terminal(
                control_descriptor,
                launch_id=handshake_launch_id,
                epoch=handshake_epoch,
                outcome="spawn_failed",
                target_started=False,
                returncode=126,
                settled=True,
                error=None,
            )
        else:
            write_control(returncode=None, settled=True, error="spawn")
        return
    if cgroup_admission_sha256 is not None:
        target_start_token = (
            _process_start_time(target.pid)
            if report_target_ready
            else _unreaped_child_start_time(target.pid)
        )
        if target_start_token is None or target_start_token <= 0:
            try:
                target.kill()
            except ProcessLookupError:
                pass
            _settle_descendants(target, grace_seconds=grace_seconds)
            raise SupervisorError("generic target process identity is unavailable")
        if report_target_ready:
            assert cgroup_admission_sha256 is not None
            _write_target_ready(
                control_descriptor,
                cgroup_admission_sha256=cgroup_admission_sha256,
                target_pid=target.pid,
                target_start_token=target_start_token,
            )
    deadline_expired = False
    cancellation_requested = False
    command_protocol_failed = False

    def observe_queued_commands() -> None:
        """Consume every queued post-START transition exactly once."""
        nonlocal cancellation_requested, command_protocol_failed
        if handshake_launch_id is None:
            command_protocol_failed = True
            return
        launch_id = handshake_launch_id
        while not command_protocol_failed and _handshake_command_is_pending(
            control_descriptor
        ):
            try:
                command, _observed_epoch = _read_handshake_command(
                    control_descriptor,
                    launch_id=launch_id,
                    deadline_monotonic_ns=time.monotonic_ns() + 1000000000,
                    stop_requested=lambda: False,
                    expected_epoch=handshake_epoch,
                    schema_version=HANDSHAKE_COMMAND_SCHEMA,
                )
            except SupervisorError:
                command_protocol_failed = True
                return
            if command != "cancel" or cancellation_requested:
                command_protocol_failed = True
                return
            cancellation_requested = True

    while target.poll() is None and (not stop_requested):
        _reap_adopted_children_except(target.pid)
        remaining_ns = deadline_monotonic_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            deadline_expired = True
            break
        if handshake_launch_id is None:
            time.sleep(min(0.005, remaining_ns / 1000000000))
            continue
        try:
            command, _observed_epoch = _read_handshake_command(
                control_descriptor,
                launch_id=handshake_launch_id,
                deadline_monotonic_ns=min(
                    deadline_monotonic_ns, time.monotonic_ns() + 5000000
                ),
                stop_requested=lambda: stop_requested,
                expected_epoch=handshake_epoch,
                schema_version=HANDSHAKE_COMMAND_SCHEMA,
            )
        except SupervisorError:
            command_protocol_failed = True
            break
        if command == "cancel":
            cancellation_requested = True
            break
        if command == "start":
            command_protocol_failed = True
            break
        assert command == "timeout"
    if handshake_launch_id is not None and stop_requested:
        command_protocol_failed = True
    try:
        settled, found_background_descendant = _settle_descendants(
            target,
            grace_seconds=grace_seconds,
            observe_control=observe_queued_commands
            if handshake_launch_id is not None
            else None,
        )
    except SupervisorError:
        settled = False
        found_background_descendant = False
    if handshake_launch_id is not None:
        observe_queued_commands()
    reported_returncode = -124 if deadline_expired and settled else target.returncode
    if reported_returncode is None:
        reported_returncode = 125
    authenticated_settled = settled and (not command_protocol_failed)
    settlement_error = None
    if not authenticated_settled:
        settlement_error = "settlement"
    elif (
        found_background_descendant
        and (not deadline_expired)
        and (not cancellation_requested)
    ):
        settlement_error = "background_descendants"
    if handshake_launch_id is not None:
        outcome = (
            "protocol_failed"
            if command_protocol_failed
            else "timed_out"
            if deadline_expired
            else "cancelled"
            if cancellation_requested
            else "completed"
        )
        if outcome == "timed_out":
            reported_returncode = -124
        elif outcome == "cancelled" and reported_returncode in {0, -124}:
            reported_returncode = 130
        elif outcome == "protocol_failed" and reported_returncode in {0, -124}:
            reported_returncode = 125
        write_handshake_terminal(
            control_descriptor,
            launch_id=handshake_launch_id,
            epoch=handshake_epoch,
            outcome=outcome,
            target_started=True,
            target_pid=None,
            target_start_token=None,
            returncode=reported_returncode,
            settled=authenticated_settled,
            error=settlement_error,
        )
    else:
        write_control(
            returncode=reported_returncode,
            settled=settled,
            error=settlement_error,
            target_pid=target.pid,
            target_start_token=target_start_token,
        )


def main() -> None:
    if sys.argv[1:] == ["--version"]:
        print(PROTOCOL_VERSION)
        return
    _run()


if __name__ == "__main__":
    main()
