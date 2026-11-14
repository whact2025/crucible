# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Descriptor-bound cgroup-v2 containment for one supervised process tree.

The caller must itself be the sole process in a dedicated, delegated cgroup;
every one of its threads is checked against ``cgroup.threads``.  It retains
that controller cgroup while the process supervisor joins a newly-created
child.  Every target descendant therefore inherits the child cgroup,
irrespective of process group, session, reparenting, or the lifetime of the
supervisor.

This module is controller-side code.  The isolated process supervisor only
receives a duplicate of :attr:`DelegatedCgroup.target_descriptor`; it never
receives the controller directory descriptor or the target leaf name.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import secrets
import socket
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from crucible.common.process_settlement import CGROUP_SETTLEMENT_SCHEMA


CGROUP2_SUPER_MAGIC = 0x63677270
CGROUP_ADMISSION_SCHEMA = "crucible.process-supervisor-cgroup.v2"
CGROUP_MOUNT_PATH = Path("/sys/fs/cgroup")
_MAX_CONTROL_BYTES = 16 * 1024
_MAX_HANDSHAKE_BYTES = 1024
_MAX_PROCESSES = 1 << 20
_F_DUPFD_CLOEXEC = getattr(fcntl, "F_DUPFD_CLOEXEC", 1030)


class CgroupContainmentError(OSError):
    """A delegated cgroup could not prove complete process settlement."""


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


def _require_cgroup2(descriptor: int) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fstatfs = libc.fstatfs
    except (AttributeError, OSError) as exc:
        raise CgroupContainmentError("fstatfs is unavailable") from exc
    fstatfs.argtypes = (ctypes.c_int, ctypes.POINTER(_StatFs))
    fstatfs.restype = ctypes.c_int
    result = _StatFs()
    if fstatfs(descriptor, ctypes.byref(result)) != 0:
        error_number = ctypes.get_errno()
        raise CgroupContainmentError(
            error_number,
            os.strerror(error_number),
        )
    if int(result.f_type) != CGROUP2_SUPER_MAGIC:
        raise CgroupContainmentError("containment authority is not cgroup v2")


def _read_bounded_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum_bytes: int = _MAX_CONTROL_BYTES,
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_descriptor,
    )
    try:
        payload = bytearray()
        while len(payload) <= maximum_bytes:
            chunk = os.read(
                descriptor,
                min(4096, maximum_bytes + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
    finally:
        os.close(descriptor)
    if len(payload) > maximum_bytes:
        raise CgroupContainmentError(f"{name} exceeds its fixed byte bound")
    return bytes(payload)


def _write_exact_at(directory_descriptor: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_descriptor,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CgroupContainmentError(
                    f"{name} write made no progress"
                )
            view = view[written:]
    finally:
        os.close(descriptor)


def _parse_canonical_decimal_lines(payload: bytes, *, label: str) -> tuple[int, ...]:
    if not payload:
        return ()
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise CgroupContainmentError(f"{label} is not ASCII") from exc
    if not text.endswith("\n"):
        raise CgroupContainmentError(f"{label} is not newline terminated")
    values: list[int] = []
    for raw in text.splitlines():
        if not raw or not raw.isascii() or not raw.isdecimal():
            raise CgroupContainmentError(f"{label} is malformed")
        value = int(raw)
        if value <= 0 or str(value) != raw:
            raise CgroupContainmentError(f"{label} is not canonical")
        values.append(value)
        if len(values) > _MAX_PROCESSES:
            raise CgroupContainmentError(f"{label} exceeds its fixed bound")
    if len(set(values)) != len(values):
        raise CgroupContainmentError(f"{label} repeats an identity")
    return tuple(values)


def _parse_key_value_lines(payload: bytes, *, label: str) -> dict[str, int]:
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise CgroupContainmentError(f"{label} is not ASCII") from exc
    if not text or not text.endswith("\n"):
        raise CgroupContainmentError(f"{label} is malformed")
    parsed: dict[str, int] = {}
    for line in text.splitlines():
        pieces = line.split(" ")
        if len(pieces) != 2:
            raise CgroupContainmentError(f"{label} is malformed")
        key, raw_value = pieces
        if (
            not key
            or not key.isascii()
            or not all(character.isalnum() or character == "_" for character in key)
            or not raw_value.isascii()
            or not raw_value.isdecimal()
            or str(int(raw_value)) != raw_value
            or key in parsed
        ):
            raise CgroupContainmentError(f"{label} is malformed")
        parsed[key] = int(raw_value)
    return parsed


def _decode_canonical_object(payload: bytes, *, label: str) -> dict[str, object]:
    if not payload or len(payload) > _MAX_HANDSHAKE_BYTES:
        raise CgroupContainmentError(f"{label} is missing or oversized")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        record: dict[str, object] = {}
        for key, value in pairs:
            if key in record:
                raise ValueError("duplicate key")
            record[key] = value
        return record

    try:
        decoded = payload.decode("utf-8", errors="strict")
        record = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CgroupContainmentError(f"{label} is malformed") from exc
    if type(record) is not dict:
        raise CgroupContainmentError(f"{label} is not an exact object")
    canonical = (
        json.dumps(
            record,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    if payload != canonical:
        raise CgroupContainmentError(f"{label} is not canonical")
    return record


def _current_cgroup_relative_path() -> PurePosixPath:
    try:
        payload = Path("/proc/self/cgroup").read_bytes()
    except OSError as exc:
        raise CgroupContainmentError("current cgroup identity is unavailable") from exc
    if len(payload) > _MAX_CONTROL_BYTES:
        raise CgroupContainmentError("current cgroup identity is oversized")
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise CgroupContainmentError("current cgroup identity is not ASCII") from exc
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0].startswith("0::/"):
        raise CgroupContainmentError("current unified cgroup identity is malformed")
    raw_path = lines[0][3:]
    path = PurePosixPath(raw_path)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or ".." in path.parts
        or "." in path.parts
        or str(path) != raw_path
    ):
        raise CgroupContainmentError("current cgroup path is not confined")
    return path.relative_to("/")


def _open_directory_walk(root: Path, relative: PurePosixPath) -> tuple[int, int]:
    mount_descriptor = os.open(
        root,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    current_descriptor = mount_descriptor
    try:
        _require_cgroup2(mount_descriptor)
        for component in relative.parts:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_descriptor,
            )
            if current_descriptor != mount_descriptor:
                os.close(current_descriptor)
            current_descriptor = next_descriptor
        if current_descriptor == mount_descriptor:
            current_descriptor = os.dup(mount_descriptor)
        return mount_descriptor, current_descriptor
    except BaseException:
        if current_descriptor != mount_descriptor:
            os.close(current_descriptor)
        os.close(mount_descriptor)
        raise


def _process_start_token(process_id: int) -> int:
    try:
        payload = Path(f"/proc/{process_id}/stat").read_bytes()
    except OSError as exc:
        raise CgroupContainmentError("controller process identity is unavailable") from exc
    if len(payload) > _MAX_CONTROL_BYTES:
        raise CgroupContainmentError("controller process identity is oversized")
    closing = payload.rfind(b") ")
    fields = payload[closing + 2 :].split() if closing >= 0 else []
    try:
        token = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise CgroupContainmentError("controller process identity is malformed") from exc
    if token <= 0:
        raise CgroupContainmentError("controller process start token is invalid")
    return token


def _directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _identity_text(metadata: os.stat_result) -> str:
    return f"{os.major(metadata.st_dev)}:{os.minor(metadata.st_dev)}:{metadata.st_ino}"


@dataclass(frozen=True)
class CgroupSettlementProof:
    """Canonical evidence that one exact cgroup was killed, emptied, and removed."""

    canonical_bytes: bytes

    @property
    def sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_bytes).hexdigest()


class DelegatedCgroup:
    """One descriptor-pinned controller/target cgroup pair."""

    def __init__(
        self,
        *,
        mount_descriptor: int,
        controller_descriptor: int,
        target_descriptor: int,
        controller_path: PurePosixPath,
        target_name: str,
        controller_identity: tuple[int, int],
        target_identity: tuple[int, int],
        controller_pid: int,
        controller_start_token: int,
    ) -> None:
        self._mount_descriptor = mount_descriptor
        self._controller_descriptor = controller_descriptor
        self._target_descriptor = target_descriptor
        self._controller_path = controller_path
        self._target_name = target_name
        self._controller_identity = controller_identity
        self._target_identity = target_identity
        self._controller_pid = controller_pid
        self._controller_start_token = controller_start_token
        self._supervisor_pid: int | None = None
        self._supervisor_start_token: int | None = None
        self._admission_outcome = "not_attempted"
        self._admission_sha256: str | None = None
        self._settlement_phase = "active"
        self._settlement_target_identity: str | None = None
        self._proof: CgroupSettlementProof | None = None
        self._closed = False
        self._close_failed = False

    @classmethod
    def create(cls) -> DelegatedCgroup:
        """Create a target leaf below the caller's dedicated delegated cgroup."""

        if sys.platform != "linux":
            raise CgroupContainmentError("cgroup containment requires Linux")
        controller_pid = os.getpid()
        task_root = Path("/proc/self/task")
        try:
            task_ids = {
                int(name)
                for name in os.listdir(task_root)
                if name.isascii() and name.isdecimal()
            }
        except OSError as exc:
            raise CgroupContainmentError("controller thread census failed") from exc
        if controller_pid not in task_ids or not task_ids:
            raise CgroupContainmentError("controller thread census is malformed")
        controller_path = _current_cgroup_relative_path()
        mount_descriptor, controller_descriptor = _open_directory_walk(
            CGROUP_MOUNT_PATH,
            controller_path,
        )
        target_descriptor: int | None = None
        target_name: str | None = None
        try:
            controller_metadata = os.fstat(controller_descriptor)
            if (
                not stat.S_ISDIR(controller_metadata.st_mode)
                or controller_metadata.st_uid != os.geteuid()
                or controller_metadata.st_mode & 0o022 != 0
            ):
                raise CgroupContainmentError(
                    "controller cgroup is not privately delegated"
                )
            controller_members = _parse_canonical_decimal_lines(
                _read_bounded_at(controller_descriptor, "cgroup.procs"),
                label="controller cgroup.procs",
            )
            if controller_members != (controller_pid,):
                raise CgroupContainmentError(
                    "controller cgroup does not contain exactly the controller"
                )
            controller_threads = set(
                _parse_canonical_decimal_lines(
                    _read_bounded_at(controller_descriptor, "cgroup.threads"),
                    label="controller cgroup.threads",
                )
            )
            if controller_threads != task_ids:
                raise CgroupContainmentError(
                    "controller cgroup thread membership is not exact"
                )
            controller_start_token = _process_start_token(controller_pid)
            target_name = f"crucible-target-{secrets.token_hex(16)}"
            os.mkdir(target_name, mode=0o700, dir_fd=controller_descriptor)
            target_descriptor = os.open(
                target_name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=controller_descriptor,
            )
            _require_cgroup2(target_descriptor)
            target_metadata = os.fstat(target_descriptor)
            named_metadata = os.stat(
                target_name,
                dir_fd=controller_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(target_metadata.st_mode)
                or target_metadata.st_uid != os.geteuid()
                or target_metadata.st_mode & 0o077 != 0
                or _directory_identity(target_metadata)
                != _directory_identity(named_metadata)
            ):
                raise CgroupContainmentError(
                    "target cgroup did not retain its private identity"
                )
            _write_exact_at(target_descriptor, "cgroup.max.depth", b"0\n")
            _write_exact_at(target_descriptor, "cgroup.max.descendants", b"0\n")
            if (
                _read_bounded_at(target_descriptor, "cgroup.max.depth") != b"0\n"
                or _read_bounded_at(target_descriptor, "cgroup.max.descendants")
                != b"0\n"
                or _parse_canonical_decimal_lines(
                    _read_bounded_at(target_descriptor, "cgroup.procs"),
                    label="target cgroup.procs",
                )
            ):
                raise CgroupContainmentError(
                    "target cgroup limits or initial membership are invalid"
                )
            return cls(
                mount_descriptor=mount_descriptor,
                controller_descriptor=controller_descriptor,
                target_descriptor=target_descriptor,
                controller_path=controller_path,
                target_name=target_name,
                controller_identity=_directory_identity(controller_metadata),
                target_identity=_directory_identity(target_metadata),
                controller_pid=controller_pid,
                controller_start_token=controller_start_token,
            )
        except BaseException as exc:
            cleanup_error: BaseException | None = None
            if target_descriptor is not None:
                try:
                    _write_exact_at(target_descriptor, "cgroup.kill", b"1\n")
                except BaseException as cleanup_exc:
                    if not (
                        isinstance(cleanup_exc, OSError)
                        and cleanup_exc.errno in {errno.ENOENT, errno.ENODEV}
                    ):
                        cleanup_error = cleanup_exc
                try:
                    os.close(target_descriptor)
                except BaseException as cleanup_exc:
                    cleanup_error = cleanup_error or cleanup_exc
            if target_name is not None:
                try:
                    os.rmdir(target_name, dir_fd=controller_descriptor)
                except FileNotFoundError:
                    pass
                except BaseException as cleanup_exc:
                    cleanup_error = cleanup_error or cleanup_exc
            os.close(controller_descriptor)
            os.close(mount_descriptor)
            if cleanup_error is not None:
                raise CgroupContainmentError(
                    "failed cgroup construction could not be cleaned"
                ) from cleanup_error
            raise exc

    @property
    def target_identity(self) -> str:
        self._require_open()
        return _identity_text(os.fstat(self._target_descriptor))

    def lease_target_descriptor(self) -> int:
        """Return a private CLOEXEC descriptor for the process supervisor."""

        self._require_open()
        self._revalidate()
        try:
            descriptor = fcntl.fcntl(
                self._target_descriptor,
                _F_DUPFD_CLOEXEC,
                3,
            )
        except OSError as exc:
            raise CgroupContainmentError(
                "target cgroup descriptor cannot be leased"
            ) from exc
        return int(descriptor)

    def admit_supervisor(self, process_id: int, start_token: int) -> None:
        """Bind a pre-target supervisor after its cgroup-entry handshake.

        The supervisor must send its PID and procfs start token before it
        spawns the target.  The controller calls this method before returning
        the handshake acknowledgement.  Exact single-process membership
        closes the fork-to-cgroup migration race.
        """

        self._require_open()
        if self._proof is not None or self._supervisor_pid is not None:
            raise CgroupContainmentError(
                "target cgroup supervisor admission is not fresh"
            )
        if (
            type(process_id) is not int
            or process_id <= 1
            or type(start_token) is not int
            or start_token <= 0
            or _process_start_token(process_id) != start_token
        ):
            raise CgroupContainmentError("supervisor process identity is invalid")
        self._revalidate()
        processes = _parse_canonical_decimal_lines(
            _read_bounded_at(self._target_descriptor, "cgroup.procs"),
            label="target cgroup.procs",
        )
        threads = _parse_canonical_decimal_lines(
            _read_bounded_at(self._target_descriptor, "cgroup.threads"),
            label="target cgroup.threads",
        )
        if processes != (process_id,) or threads != (process_id,):
            raise CgroupContainmentError(
                "target cgroup does not contain exactly the supervisor"
            )
        self._supervisor_pid = process_id
        self._supervisor_start_token = start_token
        self._admission_outcome = "membership_only"

    def admit_supervisor_handshake(
        self,
        descriptor: int,
        *,
        expected_process_id: int,
        deadline: float,
    ) -> str:
        """Authenticate cgroup entry and acknowledge target-spawn authority.

        ``descriptor`` remains caller-owned.  The supervisor must block before
        target spawn until the exact acknowledgement is received.  An
        acknowledgement send error is retained as uncertain because the
        packet may nevertheless have reached the supervisor.
        """

        self._require_open()
        if (
            type(descriptor) is not int
            or descriptor < 3
            or os.get_inheritable(descriptor)
            or type(expected_process_id) is not int
            or expected_process_id <= 1
            or type(deadline) not in {int, float}
            or not float(deadline) > time.monotonic()
            or self._admission_outcome != "not_attempted"
        ):
            raise CgroupContainmentError("cgroup admission contract is invalid")
        control = socket.socket(fileno=descriptor)
        try:
            if control.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != (
                socket.SOCK_SEQPACKET
            ):
                raise CgroupContainmentError(
                    "cgroup admission transport is not seqpacket"
                )
            control.settimeout(max(0.0, float(deadline) - time.monotonic()))
            payload = control.recv(_MAX_HANDSHAKE_BYTES + 1)
            if len(payload) > _MAX_HANDSHAKE_BYTES:
                raise CgroupContainmentError(
                    "cgroup admission packet exceeds its fixed bound"
                )
            record = _decode_canonical_object(
                payload,
                label="cgroup admission readiness",
            )
            ready_start_token = record.get("start_token")
            if (
                set(record)
                != {
                    "cgroup_identity",
                    "phase",
                    "process_id",
                    "schema_version",
                    "start_token",
                }
                or record.get("schema_version") != CGROUP_ADMISSION_SCHEMA
                or record.get("phase") != "ready"
                or type(record.get("process_id")) is not int
                or record.get("process_id") != expected_process_id
                or type(ready_start_token) is not int
                or type(record.get("cgroup_identity")) is not str
                or record.get("cgroup_identity") != self.target_identity
            ):
                raise CgroupContainmentError(
                    "cgroup admission readiness does not match authority"
                )
            start_token = ready_start_token
            self.admit_supervisor(expected_process_id, start_token)
            admission_nonce = secrets.token_hex(32)
            acknowledgement = (
                json.dumps(
                    {
                        "admission_nonce": admission_nonce,
                        "cgroup_identity": self.target_identity,
                        "phase": "admit",
                        "process_id": expected_process_id,
                        "schema_version": CGROUP_ADMISSION_SCHEMA,
                        "start_token": start_token,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
            self._admission_outcome = "acknowledgement_uncertain"
            written = control.send(acknowledgement)
            if written != len(acknowledgement):
                raise CgroupContainmentError(
                    "cgroup admission acknowledgement was truncated"
                )
            self._admission_sha256 = "sha256:" + hashlib.sha256(
                payload + acknowledgement
            ).hexdigest()
            self._admission_outcome = "committed"
            return self._admission_sha256
        except BaseException:
            if self._admission_outcome == "membership_only":
                self._admission_outcome = "precommit_failed"
            raise
        finally:
            control.detach()

    def _require_open(self) -> None:
        if self._closed:
            raise CgroupContainmentError("cgroup authority is closed")

    def _revalidate(self, *, require_named_target: bool = True) -> None:
        if os.getpid() != self._controller_pid or (
            _process_start_token(self._controller_pid)
            != self._controller_start_token
        ):
            raise CgroupContainmentError("cgroup controller identity changed")
        controller_metadata = os.fstat(self._controller_descriptor)
        target_metadata = os.fstat(self._target_descriptor)
        if (
            _directory_identity(controller_metadata) != self._controller_identity
            or _directory_identity(target_metadata) != self._target_identity
        ):
            raise CgroupContainmentError("cgroup authority identity changed")
        if require_named_target:
            named_target = os.stat(
                self._target_name,
                dir_fd=self._controller_descriptor,
                follow_symlinks=False,
            )
            if _directory_identity(named_target) != self._target_identity:
                raise CgroupContainmentError("cgroup authority identity changed")
        controller_members = _parse_canonical_decimal_lines(
            _read_bounded_at(self._controller_descriptor, "cgroup.procs"),
            label="controller cgroup.procs",
        )
        if controller_members != (self._controller_pid,):
            raise CgroupContainmentError(
                "controller cgroup process membership changed"
            )
        try:
            current_tasks = {
                int(name)
                for name in os.listdir("/proc/self/task")
                if name.isascii() and name.isdecimal()
            }
        except OSError as exc:
            raise CgroupContainmentError("controller thread census failed") from exc
        controller_threads = set(
            _parse_canonical_decimal_lines(
                _read_bounded_at(self._controller_descriptor, "cgroup.threads"),
                label="controller cgroup.threads",
            )
        )
        if not current_tasks or controller_threads != current_tasks:
            raise CgroupContainmentError(
                "controller cgroup thread membership changed"
            )

    def settle(self, *, deadline: float) -> CgroupSettlementProof:
        """Kill the exact target cgroup and prove it empty before removal."""

        self._require_open()
        if self._proof is not None:
            return self._proof
        if self._settlement_phase in {"target_closing", "failed"}:
            raise CgroupContainmentError(
                "cgroup settlement crossed an uncertain irreversible boundary"
            )
        if type(deadline) not in {int, float} or not float(deadline) > time.monotonic():
            raise CgroupContainmentError("cgroup settlement deadline is invalid")
        removal_effect_error: BaseException | None = None
        if self._settlement_phase == "active":
            self._revalidate()
            target_metadata = os.fstat(self._target_descriptor)
            self._settlement_target_identity = _identity_text(target_metadata)
            _write_exact_at(self._target_descriptor, "cgroup.kill", b"1\n")
            while True:
                self._revalidate()
                events = _parse_key_value_lines(
                    _read_bounded_at(self._target_descriptor, "cgroup.events"),
                    label="target cgroup.events",
                )
                processes = _parse_canonical_decimal_lines(
                    _read_bounded_at(self._target_descriptor, "cgroup.procs"),
                    label="target cgroup.procs",
                )
                threads = _parse_canonical_decimal_lines(
                    _read_bounded_at(self._target_descriptor, "cgroup.threads"),
                    label="target cgroup.threads",
                )
                if events.get("populated") == 0 and not processes and not threads:
                    break
                now = time.monotonic()
                if now >= float(deadline):
                    raise CgroupContainmentError(
                        "target cgroup did not become empty before its deadline"
                    )
                time.sleep(min(0.005, float(deadline) - now))
            statistics = _parse_key_value_lines(
                _read_bounded_at(self._target_descriptor, "cgroup.stat"),
                label="target cgroup.stat",
            )
            if statistics.get("nr_descendants") != 0 or statistics.get(
                "nr_dying_descendants"
            ) != 0:
                raise CgroupContainmentError("target cgroup retained descendants")
            self._settlement_phase = "empty"
        if self._settlement_phase == "empty":
            self._revalidate()
            self._settlement_phase = "removing"
        if self._settlement_phase == "removing":
            self._revalidate(require_named_target=False)
            try:
                named_target = os.stat(
                    self._target_name,
                    dir_fd=self._controller_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                named_target = None
            if named_target is not None:
                if _directory_identity(named_target) != self._target_identity:
                    self._settlement_phase = "failed"
                    raise CgroupContainmentError(
                        "target cgroup was replaced before removal"
                    )
                try:
                    os.rmdir(self._target_name, dir_fd=self._controller_descriptor)
                except BaseException as exc:
                    try:
                        post_error = os.stat(
                            self._target_name,
                            dir_fd=self._controller_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        post_error = None
                    if post_error is not None:
                        if _directory_identity(post_error) != self._target_identity:
                            self._settlement_phase = "failed"
                        raise
                    removal_effect_error = exc
            try:
                os.stat(
                    self._target_name,
                    dir_fd=self._controller_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                self._settlement_phase = "removed"
            else:
                raise CgroupContainmentError(
                    "target cgroup remained after removal"
                )
        if self._settlement_phase == "removed":
            target_descriptor = self._target_descriptor
            self._target_descriptor = -1
            self._settlement_phase = "target_closing"
            try:
                os.close(target_descriptor)
            except BaseException:
                self._settlement_phase = "failed"
                raise
            self._settlement_phase = "target_closed"
        if self._settlement_phase != "target_closed":
            raise CgroupContainmentError("cgroup settlement state is invalid")
        target_identity_text = self._settlement_target_identity
        if target_identity_text is None:
            self._settlement_phase = "failed"
            raise CgroupContainmentError("target cgroup identity was not retained")
        record = {
            "controller_cgroup": "/" + str(self._controller_path),
            "controller_identity": _identity_text(
                os.fstat(self._controller_descriptor)
            ),
            "controller_pid": self._controller_pid,
            "controller_start_token": self._controller_start_token,
            "admission_outcome": self._admission_outcome,
            "admission_sha256": self._admission_sha256,
            "kill_requested": True,
            "populated": 0,
            "removed": True,
            "schema_version": CGROUP_SETTLEMENT_SCHEMA,
            "supervisor_pid": self._supervisor_pid,
            "supervisor_start_token": self._supervisor_start_token,
            "target_identity": target_identity_text,
        }
        canonical = (
            json.dumps(
                record,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        self._proof = CgroupSettlementProof(canonical)
        if removal_effect_error is not None:
            raise removal_effect_error
        return self._proof

    def close(self) -> None:
        """Close an already-settled authority without weakening its proof."""

        if self._closed:
            return
        if self._close_failed:
            raise CgroupContainmentError(
                "cgroup authority descriptor cleanup is uncertain"
            )
        if self._proof is None:
            raise CgroupContainmentError(
                "unsettled cgroup authority cannot be closed"
            )
        descriptors = (self._controller_descriptor, self._mount_descriptor)
        self._controller_descriptor = -1
        self._mount_descriptor = -1
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except BaseException:
                self._close_failed = True
        if self._close_failed:
            raise CgroupContainmentError(
                "cgroup authority descriptor cleanup is uncertain"
            )
        self._closed = True


__all__ = [
    "CGROUP2_SUPER_MAGIC",
    "CGROUP_ADMISSION_SCHEMA",
    "CGROUP_MOUNT_PATH",
    "CGROUP_SETTLEMENT_SCHEMA",
    "CgroupContainmentError",
    "CgroupSettlementProof",
    "DelegatedCgroup",
]
