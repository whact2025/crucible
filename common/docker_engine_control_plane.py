# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Trusted Unix Docker Engine control plane for daemon-side settlement.

The adapter speaks the Docker Engine HTTP API directly over an ``AF_UNIX``
socket.  It never derives authority from human CLI output.  Every public
operation and every socket read/write shares the caller's finite absolute
monotonic deadline.

Daemon identity deliberately includes a restart-sensitive generation.  The
persistent Docker Engine ID is not restart-sensitive, so the default local
probe binds the authenticated Unix peer PID to Linux boot identity and process
start time.  It supports both a direct dockerd peer and systemd socket
activation, where PID 1 plus a stable root-owned dockerd PID file form the
authority. Other socket proxies must inject an equally restart-sensitive
generation probe; the local probe fails closed for unknown topologies.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import socket
import stat
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urlencode

from crucible.common.benchmark_runtime import canonical_fingerprint
from crucible.common.docker_settlement import (
    DockerContainerInspection,
    DockerDaemonIdentity,
)


_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_API_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+$")
_BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
_HTTP_STATUS_RE = re.compile(rb"^HTTP/1\.[01] ([0-9]{3})(?: .*)?$")
_HEADER_NAME_RE = re.compile(rb"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_MAX_HEADER_BYTES = 64 * 1024
_MAX_PROC_BYTES = 16 * 1024


class DockerEngineControlPlaneError(RuntimeError):
    """Base error for unavailable or invalid Docker Engine authority."""


class DockerEngineDeadlineError(DockerEngineControlPlaneError, TimeoutError):
    """The caller's absolute monotonic deadline is invalid or expired."""


class DockerEngineQueryError(DockerEngineControlPlaneError):
    """A Docker Engine request failed without an authoritative observation."""


class DockerEngineProtocolError(DockerEngineControlPlaneError):
    """The Docker Engine returned a malformed or contract-incompatible reply."""


class DockerDaemonGenerationError(DockerEngineControlPlaneError):
    """Restart-sensitive local daemon generation could not be proved."""


@dataclasses.dataclass(frozen=True)
class DockerPeerCredentials:
    """Linux credentials of the process serving one Unix socket connection."""

    pid: int
    uid: int
    gid: int

    def __post_init__(self) -> None:
        if isinstance(self.pid, bool) or self.pid <= 0:
            raise ValueError("Docker Unix peer PID must be positive")
        if isinstance(self.uid, bool) or self.uid < 0:
            raise ValueError("Docker Unix peer UID must be non-negative")
        if isinstance(self.gid, bool) or self.gid < 0:
            raise ValueError("Docker Unix peer GID must be non-negative")


@dataclasses.dataclass(frozen=True)
class DockerEngineHTTPResponse:
    """Bounded typed response returned by a Docker Engine transport."""

    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    peer: DockerPeerCredentials

    def __post_init__(self) -> None:
        if isinstance(self.status, bool) or not 100 <= self.status <= 599:
            raise ValueError("Docker HTTP status must be in [100, 599]")
        if not isinstance(self.body, bytes):
            raise TypeError("Docker HTTP response body must be bytes")
        normalized: list[tuple[str, str]] = []
        for name, value in self.headers:
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError("Docker HTTP headers must be string pairs")
            normalized.append((name.lower(), value))
        object.__setattr__(self, "headers", tuple(normalized))


class DockerEngineTransport(Protocol):
    """Provider-free transport boundary used by the control-plane adapter."""

    def request(
        self,
        method: str,
        target: str,
        *,
        deadline_monotonic: float,
    ) -> DockerEngineHTTPResponse: ...


class DockerDaemonGenerationProbe(Protocol):
    """Return a restart-sensitive generation for an authenticated Unix peer."""

    def __call__(self, peer: DockerPeerCredentials) -> str: ...


def _remaining(
    deadline_monotonic: float,
    *,
    monotonic: Callable[[], float],
) -> float:
    if (
        isinstance(deadline_monotonic, bool)
        or not isinstance(deadline_monotonic, (int, float))
        or not math.isfinite(deadline_monotonic)
    ):
        raise DockerEngineDeadlineError(
            "Docker Engine deadline must be a finite monotonic timestamp"
        )
    remaining = float(deadline_monotonic) - monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise DockerEngineDeadlineError("Docker Engine deadline expired")
    return remaining


def _read_bounded(path: Path, *, limit: int = _MAX_PROC_BYTES) -> bytes:
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except (OSError, ValueError) as exc:
        raise DockerDaemonGenerationError(
            f"Docker daemon generation source is unavailable: {path}"
        ) from exc
    if len(payload) > limit:
        raise DockerDaemonGenerationError(
            f"Docker daemon generation source is oversized: {path}"
        )
    return payload


def _parse_proc_stat(payload: bytes, *, expected_pid: int) -> int:
    """Return Linux ``/proc/<pid>/stat`` field 22 (starttime)."""

    marker = payload.rfind(b") ")
    opening = payload.find(b"(")
    if opening <= 0 or marker <= opening:
        raise DockerDaemonGenerationError("dockerd process stat is malformed")
    try:
        observed_pid = int(payload[:opening].strip())
        fields_after_comm = payload[marker + 2 :].split()
        # The first token after comm is field 3. starttime is field 22.
        starttime = int(fields_after_comm[19])
    except (ValueError, IndexError) as exc:
        raise DockerDaemonGenerationError(
            "dockerd process stat lacks a valid start time"
        ) from exc
    if observed_pid != expected_pid or starttime <= 0:
        raise DockerDaemonGenerationError(
            "dockerd process identity changed while reading generation"
        )
    return starttime


def _parse_proc_parent_pid(payload: bytes, *, expected_pid: int) -> int:
    marker = payload.rfind(b") ")
    opening = payload.find(b"(")
    if opening <= 0 or marker <= opening:
        raise DockerDaemonGenerationError("dockerd process stat is malformed")
    try:
        observed_pid = int(payload[:opening].strip())
        fields_after_comm = payload[marker + 2 :].split()
        parent_pid = int(fields_after_comm[1])
    except (ValueError, IndexError) as exc:
        raise DockerDaemonGenerationError(
            "dockerd process stat lacks a valid parent PID"
        ) from exc
    if observed_pid != expected_pid or parent_pid < 0:
        raise DockerDaemonGenerationError(
            "dockerd process identity changed while reading its parent"
        )
    return parent_pid


def _read_root_owned_pid_file(path: Path, *, required_uid: int = 0) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DockerDaemonGenerationError(
            "Docker daemon PID file is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        payload = os.read(descriptor, 65)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise DockerDaemonGenerationError(
            "Docker daemon PID file could not be read"
        ) from exc
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != required_uid
        or before.st_mode & 0o022
        or (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or not 0 < len(payload) <= 64
    ):
        raise DockerDaemonGenerationError(
            "Docker daemon PID file is not a stable root-owned regular file"
        )
    stripped = payload.rstrip(b"\n")
    if (
        payload not in {stripped, stripped + b"\n"}
        or not stripped.isdigit()
        or stripped.startswith(b"0")
    ):
        raise DockerDaemonGenerationError("Docker daemon PID file is malformed")
    pid = int(stripped)
    if pid <= 1:
        raise DockerDaemonGenerationError("Docker daemon PID is invalid")
    return pid


def _read_process_executable(process_root: Path, *, expected_name: str) -> str:
    try:
        executable = os.readlink(process_root / "exe")
    except PermissionError:
        try:
            raw = _read_bounded(process_root / "cmdline", limit=4_096)
            executable = raw.split(b"\x00", 1)[0].decode(
                "utf-8",
                errors="strict",
            )
        except UnicodeDecodeError as exc:
            raise DockerDaemonGenerationError(
                f"{expected_name} command identity is not valid UTF-8"
            ) from exc
    except OSError as exc:
        raise DockerDaemonGenerationError(
            f"{expected_name} executable identity is unavailable"
        ) from exc
    if (
        not executable.startswith("/")
        or executable.endswith(" (deleted)")
        or Path(executable).name != expected_name
    ):
        raise DockerDaemonGenerationError(
            f"Docker Unix socket authority is not validated {expected_name}"
        )
    return executable


class ProcDockerDaemonGenerationProbe:
    """Prove one local dockerd generation from authenticated Linux sources."""

    def __init__(self, proc_root: str | os.PathLike[str] = "/proc") -> None:
        root = Path(proc_root)
        if not root.is_absolute():
            raise ValueError("proc_root must be absolute")
        self._proc_root = root

    def __call__(self, peer: DockerPeerCredentials) -> str:
        process_root = self._proc_root / str(peer.pid)
        stat_path = process_root / "stat"
        stat_before = _read_bounded(stat_path)
        starttime_before = _parse_proc_stat(
            stat_before,
            expected_pid=peer.pid,
        )

        try:
            comm = (
                _read_bounded(process_root / "comm", limit=256)
                .decode(
                    "utf-8",
                    errors="strict",
                )
                .strip()
            )
        except UnicodeDecodeError as exc:
            raise DockerDaemonGenerationError(
                "dockerd process name is not valid UTF-8"
            ) from exc
        if comm != "dockerd":
            raise DockerDaemonGenerationError(
                "Docker Unix socket peer is not a validated dockerd process"
            )

        executable = _read_process_executable(
            process_root,
            expected_name="dockerd",
        )

        try:
            boot_id = (
                _read_bounded(
                    self._proc_root / "sys/kernel/random/boot_id",
                    limit=128,
                )
                .decode("ascii", errors="strict")
                .strip()
                .lower()
            )
        except UnicodeDecodeError as exc:
            raise DockerDaemonGenerationError(
                "kernel boot identity is not ASCII"
            ) from exc
        if not _BOOT_ID_RE.fullmatch(boot_id):
            raise DockerDaemonGenerationError("kernel boot identity is invalid")

        stat_after = _read_bounded(stat_path)
        starttime_after = _parse_proc_stat(stat_after, expected_pid=peer.pid)
        if starttime_before != starttime_after:
            raise DockerDaemonGenerationError(
                "dockerd process identity changed while probing generation"
            )
        return canonical_fingerprint(
            {
                "boot_id": boot_id,
                "executable": executable,
                "gid": peer.gid,
                "pid": peer.pid,
                "starttime_ticks": starttime_before,
                "uid": peer.uid,
            }
        )


class SystemdSocketActivatedDockerDaemonGenerationProbe:
    """Bind a systemd-owned Docker socket to its root-owned dockerd PID file.

    With ``docker.socket`` activation, Linux reports PID 1 as the Unix peer
    even though the inherited connection is served by ``docker.service``.
    PID 1 is not restart-sensitive for dockerd, so this probe validates both
    the root-owned systemd peer and the root-owned dockerd PID file, then binds
    the daemon PID/start time beneath the current kernel boot identity.
    """

    def __init__(
        self,
        *,
        proc_root: str | os.PathLike[str] = "/proc",
        pid_file: str | os.PathLike[str] = "/run/docker.pid",
        required_uid: int = 0,
    ) -> None:
        root = Path(proc_root)
        pid_path = Path(pid_file)
        if (
            not root.is_absolute()
            or not pid_path.is_absolute()
            or isinstance(required_uid, bool)
            or not isinstance(required_uid, int)
            or required_uid < 0
        ):
            raise ValueError(
                "proc_root/pid_file must be absolute and required_uid non-negative"
            )
        self._proc_root = root
        self._pid_file = pid_path
        self._required_uid = required_uid

    def _process_identity(
        self,
        pid: int,
        *,
        expected_name: str,
    ) -> tuple[int, int, str]:
        process_root = self._proc_root / str(pid)
        try:
            owner = process_root.stat().st_uid
        except OSError as exc:
            raise DockerDaemonGenerationError(
                f"{expected_name} process identity is unavailable"
            ) from exc
        if owner != self._required_uid:
            raise DockerDaemonGenerationError(
                f"{expected_name} process is not root-owned"
            )
        stat_payload = _read_bounded(process_root / "stat")
        starttime = _parse_proc_stat(stat_payload, expected_pid=pid)
        parent_pid = _parse_proc_parent_pid(stat_payload, expected_pid=pid)
        try:
            comm = (
                _read_bounded(process_root / "comm", limit=256)
                .decode("utf-8", errors="strict")
                .strip()
            )
        except UnicodeDecodeError as exc:
            raise DockerDaemonGenerationError(
                f"{expected_name} process name is not valid UTF-8"
            ) from exc
        if comm != expected_name:
            raise DockerDaemonGenerationError(
                f"Docker socket authority is not validated {expected_name}"
            )
        executable = _read_process_executable(
            process_root,
            expected_name=expected_name,
        )
        return starttime, parent_pid, executable

    def __call__(self, peer: DockerPeerCredentials) -> str:
        if peer.pid != 1 or peer.uid != self._required_uid:
            raise DockerDaemonGenerationError(
                "Docker socket activation peer is not root systemd PID 1"
            )
        systemd_start, systemd_parent, systemd_executable = self._process_identity(
            peer.pid,
            expected_name="systemd",
        )
        if systemd_parent != 0:
            raise DockerDaemonGenerationError(
                "Docker socket activation peer has an invalid parent"
            )

        daemon_pid = _read_root_owned_pid_file(
            self._pid_file,
            required_uid=self._required_uid,
        )
        daemon_start, daemon_parent, daemon_executable = self._process_identity(
            daemon_pid,
            expected_name="dockerd",
        )
        if daemon_parent != 1:
            raise DockerDaemonGenerationError(
                "socket-activated dockerd is not a direct systemd child"
            )
        try:
            boot_id = (
                _read_bounded(
                    self._proc_root / "sys/kernel/random/boot_id",
                    limit=128,
                )
                .decode("ascii", errors="strict")
                .strip()
                .lower()
            )
        except UnicodeDecodeError as exc:
            raise DockerDaemonGenerationError(
                "kernel boot identity is not ASCII"
            ) from exc
        if not _BOOT_ID_RE.fullmatch(boot_id):
            raise DockerDaemonGenerationError("kernel boot identity is invalid")

        daemon_stat_after = _read_bounded(
            self._proc_root / str(daemon_pid) / "stat"
        )
        systemd_stat_after = _read_bounded(self._proc_root / "1/stat")
        if (
            _read_root_owned_pid_file(
                self._pid_file,
                required_uid=self._required_uid,
            )
            != daemon_pid
            or _parse_proc_stat(
                daemon_stat_after,
                expected_pid=daemon_pid,
            )
            != daemon_start
            or _parse_proc_stat(systemd_stat_after, expected_pid=1)
            != systemd_start
        ):
            raise DockerDaemonGenerationError(
                "Docker daemon generation changed while it was probed"
            )
        return canonical_fingerprint({
            "boot_id": boot_id,
            "authority_uid": self._required_uid,
            "dockerd": {
                "executable": daemon_executable,
                "parent_pid": daemon_parent,
                "pid": daemon_pid,
                "starttime_ticks": daemon_start,
            },
            "socket_activator": {
                "executable": systemd_executable,
                "gid": peer.gid,
                "pid": peer.pid,
                "starttime_ticks": systemd_start,
                "uid": peer.uid,
            },
        })


class LocalDockerDaemonGenerationProbe:
    """Select the exact supported local Unix-socket ownership topology."""

    def __init__(
        self,
        *,
        proc_root: str | os.PathLike[str] = "/proc",
        pid_file: str | os.PathLike[str] = "/run/docker.pid",
    ) -> None:
        self._proc_root = Path(proc_root)
        self._direct = ProcDockerDaemonGenerationProbe(proc_root)
        self._socket_activated = SystemdSocketActivatedDockerDaemonGenerationProbe(
            proc_root=proc_root,
            pid_file=pid_file,
        )

    def __call__(self, peer: DockerPeerCredentials) -> str:
        try:
            comm = (
                _read_bounded(
                    self._proc_root / str(peer.pid) / "comm",
                    limit=256,
                )
                .decode("utf-8", errors="strict")
                .strip()
            )
        except UnicodeDecodeError as exc:
            raise DockerDaemonGenerationError(
                "Docker Unix peer process name is not valid UTF-8"
            ) from exc
        if comm == "dockerd":
            return self._direct(peer)
        if comm == "systemd" and peer.pid == 1:
            return self._socket_activated(peer)
        raise DockerDaemonGenerationError(
            "Docker Unix socket peer topology is unsupported"
        )


class _DeadlineSocketReader:
    def __init__(
        self,
        sock: socket.socket,
        *,
        deadline_monotonic: float,
        monotonic: Callable[[], float],
    ) -> None:
        self._sock = sock
        self._deadline = deadline_monotonic
        self._monotonic = monotonic
        self._buffer = bytearray()

    def _recv(self) -> bytes:
        self._sock.settimeout(
            _remaining(
                self._deadline,
                monotonic=self._monotonic,
            )
        )
        try:
            return self._sock.recv(64 * 1024)
        except socket.timeout as exc:
            raise DockerEngineDeadlineError(
                "Docker Engine deadline expired while reading response"
            ) from exc
        except OSError as exc:
            raise DockerEngineQueryError("Docker Engine response read failed") from exc

    def read_until(self, delimiter: bytes, *, limit: int) -> bytes:
        while True:
            index = self._buffer.find(delimiter)
            if index >= 0:
                end = index + len(delimiter)
                payload = bytes(self._buffer[:end])
                del self._buffer[:end]
                return payload
            if len(self._buffer) > limit:
                raise DockerEngineProtocolError(
                    "Docker Engine response section exceeded its bound"
                )
            chunk = self._recv()
            if not chunk:
                raise DockerEngineProtocolError(
                    "Docker Engine response ended before its delimiter"
                )
            self._buffer.extend(chunk)
            if len(self._buffer) > limit:
                raise DockerEngineProtocolError(
                    "Docker Engine response section exceeded its bound"
                )

    def read_exact(self, size: int) -> bytes:
        while len(self._buffer) < size:
            chunk = self._recv()
            if not chunk:
                raise DockerEngineProtocolError(
                    "Docker Engine response body ended early"
                )
            self._buffer.extend(chunk)
        payload = bytes(self._buffer[:size])
        del self._buffer[:size]
        return payload

    def read_to_eof(self, *, limit: int) -> bytes:
        while True:
            if len(self._buffer) > limit:
                raise DockerEngineProtocolError(
                    "Docker Engine response body exceeded its bound"
                )
            chunk = self._recv()
            if not chunk:
                return bytes(self._buffer)
            self._buffer.extend(chunk)


def _parse_headers(payload: bytes) -> tuple[int, dict[str, list[str]]]:
    lines = payload.removesuffix(b"\r\n\r\n").split(b"\r\n")
    if not lines:
        raise DockerEngineProtocolError("Docker Engine response is empty")
    status_match = _HTTP_STATUS_RE.fullmatch(lines[0])
    if status_match is None:
        raise DockerEngineProtocolError("Docker Engine HTTP status is malformed")
    status_code = int(status_match.group(1))
    headers: dict[str, list[str]] = {}
    for line in lines[1:]:
        if line[:1] in {b" ", b"\t"} or b":" not in line:
            raise DockerEngineProtocolError("Docker Engine HTTP header is malformed")
        name_raw, value_raw = line.split(b":", 1)
        if not _HEADER_NAME_RE.fullmatch(name_raw):
            raise DockerEngineProtocolError(
                "Docker Engine HTTP header name is malformed"
            )
        try:
            name = name_raw.decode("ascii").lower()
            value = value_raw.decode("latin-1").strip()
        except UnicodeDecodeError as exc:
            raise DockerEngineProtocolError(
                "Docker Engine HTTP header encoding is invalid"
            ) from exc
        headers.setdefault(name, []).append(value)
    return status_code, headers


def _read_chunked_body(
    reader: _DeadlineSocketReader,
    *,
    limit: int,
) -> bytes:
    body = bytearray()
    while True:
        line = reader.read_until(b"\r\n", limit=4_096).removesuffix(b"\r\n")
        size_token = line.split(b";", 1)[0].strip()
        try:
            size = int(size_token, 16)
        except ValueError as exc:
            raise DockerEngineProtocolError(
                "Docker Engine chunk size is malformed"
            ) from exc
        if size < 0 or len(body) + size > limit:
            raise DockerEngineProtocolError(
                "Docker Engine response body exceeded its bound"
            )
        if size == 0:
            # Consume a bounded trailer section. No trailer has authority.
            while reader.read_until(b"\r\n", limit=_MAX_HEADER_BYTES) != b"\r\n":
                pass
            return bytes(body)
        body.extend(reader.read_exact(size))
        if reader.read_exact(2) != b"\r\n":
            raise DockerEngineProtocolError(
                "Docker Engine chunk terminator is malformed"
            )


class UnixDockerEngineTransport:
    """Minimal bounded HTTP/1.1 transport over one local Unix socket."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        *,
        max_response_bytes: int = 8 * 1024 * 1024,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        path = os.path.realpath(os.path.abspath(os.fspath(socket_path)))
        if not path.startswith("/") or "\x00" in path:
            raise ValueError("Docker socket path must be an absolute Unix path")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.socket_path = path
        self._max_response_bytes = max_response_bytes
        self._monotonic = monotonic

    def _peer(self, sock: socket.socket) -> DockerPeerCredentials:
        option = getattr(socket, "SO_PEERCRED", None)
        if option is None:
            raise DockerEngineQueryError(
                "SO_PEERCRED is required for trusted Docker Unix authority"
            )
        try:
            raw = sock.getsockopt(socket.SOL_SOCKET, option, struct.calcsize("3i"))
            pid, uid, gid = struct.unpack("3i", raw)
            return DockerPeerCredentials(pid=pid, uid=uid, gid=gid)
        except (OSError, struct.error, ValueError) as exc:
            raise DockerEngineQueryError(
                "Docker Unix peer credentials are unavailable"
            ) from exc

    def request(
        self,
        method: str,
        target: str,
        *,
        deadline_monotonic: float,
    ) -> DockerEngineHTTPResponse:
        if method not in {"DELETE", "GET"}:
            raise ValueError("unsupported Docker Engine HTTP method")
        if not target.startswith("/") or any(char in target for char in "\r\n\x00"):
            raise ValueError("Docker Engine request target is invalid")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(
                _remaining(
                    deadline_monotonic,
                    monotonic=self._monotonic,
                )
            )
            try:
                sock.connect(self.socket_path)
            except socket.timeout as exc:
                raise DockerEngineDeadlineError(
                    "Docker Engine deadline expired while connecting"
                ) from exc
            except OSError as exc:
                raise DockerEngineQueryError(
                    "Docker Engine Unix socket connection failed"
                ) from exc
            peer = self._peer(sock)
            request = (
                f"{method} {target} HTTP/1.1\r\n"
                "Host: docker\r\n"
                "Accept: application/json\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            view = memoryview(request)
            while view:
                sock.settimeout(
                    _remaining(
                        deadline_monotonic,
                        monotonic=self._monotonic,
                    )
                )
                try:
                    sent = sock.send(view)
                except socket.timeout as exc:
                    raise DockerEngineDeadlineError(
                        "Docker Engine deadline expired while writing request"
                    ) from exc
                except OSError as exc:
                    raise DockerEngineQueryError(
                        "Docker Engine request write failed"
                    ) from exc
                if sent <= 0:
                    raise DockerEngineQueryError(
                        "Docker Engine request write made no progress"
                    )
                view = view[sent:]

            reader = _DeadlineSocketReader(
                sock,
                deadline_monotonic=deadline_monotonic,
                monotonic=self._monotonic,
            )
            header_payload = reader.read_until(
                b"\r\n\r\n",
                limit=_MAX_HEADER_BYTES,
            )
            status_code, headers = _parse_headers(header_payload)
            transfer_encoding = headers.get("transfer-encoding", [])
            content_length = headers.get("content-length", [])
            if transfer_encoding and content_length:
                raise DockerEngineProtocolError(
                    "Docker Engine response has conflicting body framing"
                )
            if len(transfer_encoding) > 1 or len(content_length) > 1:
                raise DockerEngineProtocolError(
                    "Docker Engine response has duplicate body framing"
                )
            if transfer_encoding:
                if transfer_encoding[0].lower() != "chunked":
                    raise DockerEngineProtocolError(
                        "Docker Engine response transfer encoding is unsupported"
                    )
                body = _read_chunked_body(
                    reader,
                    limit=self._max_response_bytes,
                )
            elif content_length:
                try:
                    length = int(content_length[0])
                except ValueError as exc:
                    raise DockerEngineProtocolError(
                        "Docker Engine Content-Length is malformed"
                    ) from exc
                if length < 0 or length > self._max_response_bytes:
                    raise DockerEngineProtocolError(
                        "Docker Engine response body exceeded its bound"
                    )
                body = reader.read_exact(length)
            else:
                body = reader.read_to_eof(limit=self._max_response_bytes)
            _remaining(deadline_monotonic, monotonic=self._monotonic)
            return DockerEngineHTTPResponse(
                status=status_code,
                headers=tuple(
                    (name, value)
                    for name, values in sorted(headers.items())
                    for value in values
                ),
                body=body,
                peer=peer,
            )
        finally:
            sock.close()


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DockerEngineProtocolError(
                f"Docker Engine JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _decode_json(body: bytes, *, operation: str) -> object:
    try:
        text = body.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, DockerEngineProtocolError):
            raise
        raise DockerEngineProtocolError(
            f"Docker Engine {operation} response is not strict JSON"
        ) from exc


def _require_string(
    source: Mapping[str, object],
    field: str,
    *,
    operation: str,
) -> str:
    value = source.get(field)
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise DockerEngineProtocolError(
            f"Docker Engine {operation} response lacks valid {field}"
        )
    return value


def _parse_api_version(value: str) -> tuple[int, int]:
    if not _API_VERSION_RE.fullmatch(value):
        raise DockerEngineProtocolError("Docker API version is malformed")
    major_raw, minor_raw = value.split(".", 1)
    return int(major_raw), int(minor_raw)


_STABLE_INFO_FIELDS = (
    "Architecture",
    "CgroupDriver",
    "CgroupVersion",
    "DockerRootDir",
    "Driver",
    "ExperimentalBuild",
    "InitBinary",
    "Isolation",
    "KernelVersion",
    "LiveRestoreEnabled",
    "Name",
    "NCPU",
    "OSType",
    "OSVersion",
    "OperatingSystem",
    "ServerVersion",
)


def _stable_configuration(
    *,
    info: Mapping[str, object],
    version: Mapping[str, object],
    configured_api_version: str,
    peer: DockerPeerCredentials,
) -> dict[str, object]:
    configuration: dict[str, object] = {
        "configured_api_version": configured_api_version,
        "peer_gid": peer.gid,
        "peer_uid": peer.uid,
        "server_api_version": _require_string(
            version,
            "ApiVersion",
            operation="version",
        ),
        "server_min_api_version": _require_string(
            version,
            "MinAPIVersion",
            operation="version",
        ),
        "server_version": _require_string(
            version,
            "Version",
            operation="version",
        ),
    }
    for field in _STABLE_INFO_FIELDS:
        value = info.get(field)
        if value is None or isinstance(value, (str, int, bool)):
            configuration[field] = value
        else:
            raise DockerEngineProtocolError(
                f"Docker Engine info field {field} has an invalid type"
            )
    security_options = info.get("SecurityOptions", [])
    if not isinstance(security_options, list) or any(
        not isinstance(item, str) for item in security_options
    ):
        raise DockerEngineProtocolError(
            "Docker Engine SecurityOptions has an invalid type"
        )
    configuration["SecurityOptions"] = sorted(security_options)
    return configuration


class UnixDockerEngineControlPlane:
    """Implement ``DockerSettlementControlPlane`` using the Engine HTTP API."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str] = "/var/run/docker.sock",
        *,
        api_version: str = "1.41",
        transport: DockerEngineTransport | None = None,
        generation_probe: DockerDaemonGenerationProbe | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        max_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        path = os.path.realpath(os.path.abspath(os.fspath(socket_path)))
        if not path.startswith("/") or "\x00" in path:
            raise ValueError("Docker socket path must be an absolute Unix path")
        if not _API_VERSION_RE.fullmatch(api_version):
            raise ValueError("api_version must have major.minor form")
        self._socket_path = path
        self._endpoint = f"unix://{path}"
        self._api_version = api_version
        self._monotonic = monotonic
        self._transport = transport or UnixDockerEngineTransport(
            path,
            max_response_bytes=max_response_bytes,
            monotonic=monotonic,
        )
        self._generation_probe = generation_probe or LocalDockerDaemonGenerationProbe()

    def _request(
        self,
        method: str,
        target: str,
        *,
        deadline_monotonic: float,
    ) -> DockerEngineHTTPResponse:
        _remaining(deadline_monotonic, monotonic=self._monotonic)
        try:
            response = self._transport.request(
                method,
                target,
                deadline_monotonic=deadline_monotonic,
            )
        except DockerEngineControlPlaneError:
            raise
        except Exception as exc:
            raise DockerEngineQueryError(
                "Docker Engine transport query failed"
            ) from exc
        _remaining(deadline_monotonic, monotonic=self._monotonic)
        if not isinstance(response, DockerEngineHTTPResponse):
            raise DockerEngineProtocolError(
                "Docker Engine transport returned an untyped response"
            )
        return response

    @staticmethod
    def _expect_status(
        response: DockerEngineHTTPResponse,
        expected: int,
        *,
        operation: str,
    ) -> None:
        if response.status != expected:
            raise DockerEngineQueryError(
                f"Docker Engine {operation} returned HTTP {response.status}"
            )

    def daemon_identity(
        self,
        *,
        deadline_monotonic: float,
    ) -> DockerDaemonIdentity:
        version_response = self._request(
            "GET",
            "/version",
            deadline_monotonic=deadline_monotonic,
        )
        self._expect_status(version_response, 200, operation="version")
        version_raw = _decode_json(version_response.body, operation="version")
        if not isinstance(version_raw, dict):
            raise DockerEngineProtocolError(
                "Docker Engine version response must be an object"
            )
        server_api = _require_string(
            version_raw,
            "ApiVersion",
            operation="version",
        )
        server_min_api = _require_string(
            version_raw,
            "MinAPIVersion",
            operation="version",
        )
        configured = _parse_api_version(self._api_version)
        if not (
            _parse_api_version(server_min_api)
            <= configured
            <= _parse_api_version(server_api)
        ):
            raise DockerEngineProtocolError(
                "configured Docker API version is outside the daemon range"
            )

        info_response = self._request(
            "GET",
            f"/v{self._api_version}/info",
            deadline_monotonic=deadline_monotonic,
        )
        self._expect_status(info_response, 200, operation="info")
        if info_response.peer != version_response.peer:
            raise DockerEngineQueryError(
                "Docker Unix peer changed during daemon identity query"
            )
        info_raw = _decode_json(info_response.body, operation="info")
        if not isinstance(info_raw, dict):
            raise DockerEngineProtocolError(
                "Docker Engine info response must be an object"
            )
        engine_id = _require_string(info_raw, "ID", operation="info")
        info_server_version = _require_string(
            info_raw,
            "ServerVersion",
            operation="info",
        )
        if info_server_version != _require_string(
            version_raw,
            "Version",
            operation="version",
        ):
            raise DockerEngineProtocolError(
                "Docker Engine version and info observations disagree"
            )
        _remaining(deadline_monotonic, monotonic=self._monotonic)
        try:
            generation = self._generation_probe(info_response.peer)
        except DockerEngineControlPlaneError:
            raise
        except Exception as exc:
            raise DockerDaemonGenerationError(
                "Docker daemon generation probe failed"
            ) from exc
        _remaining(deadline_monotonic, monotonic=self._monotonic)
        if (
            not isinstance(generation, str)
            or not generation.strip()
            or generation != generation.strip()
            or len(generation) > 256
        ):
            raise DockerDaemonGenerationError(
                "Docker daemon generation probe returned an invalid value"
            )
        configuration = _stable_configuration(
            info=info_raw,
            version=version_raw,
            configured_api_version=self._api_version,
            peer=info_response.peer,
        )
        return DockerDaemonIdentity.from_components(
            endpoint=self._endpoint,
            engine_id=engine_id,
            generation=generation,
            configuration=configuration,
        )

    def list_container_ids(
        self,
        *,
        labels: Mapping[str, str],
        limit: int,
        deadline_monotonic: float,
    ) -> Sequence[str]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Docker container list limit must be positive")
        filters: list[str] = []
        for key, value in sorted(labels.items()):
            if (
                not isinstance(key, str)
                or not key
                or "\x00" in key
                or not isinstance(value, str)
                or "\x00" in value
            ):
                raise ValueError("Docker label filters must be non-NUL strings")
            filters.append(f"{key}={value}")
        query = urlencode(
            {
                "all": "1",
                "filters": json.dumps(
                    {"label": filters},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "limit": str(limit),
            }
        )
        response = self._request(
            "GET",
            f"/v{self._api_version}/containers/json?{query}",
            deadline_monotonic=deadline_monotonic,
        )
        self._expect_status(response, 200, operation="container list")
        raw = _decode_json(response.body, operation="container list")
        if not isinstance(raw, list):
            raise DockerEngineProtocolError(
                "Docker Engine container list response must be an array"
            )
        ids: list[str] = []
        for item in raw:
            if not isinstance(item, dict):
                raise DockerEngineProtocolError(
                    "Docker Engine container list item must be an object"
                )
            container_id = item.get("Id")
            if not isinstance(container_id, str) or not _CONTAINER_ID_RE.fullmatch(
                container_id
            ):
                raise DockerEngineProtocolError(
                    "Docker Engine list returned a non-canonical container ID"
                )
            ids.append(container_id)
        if len(ids) > limit:
            raise DockerEngineProtocolError(
                "Docker Engine returned more container IDs than requested"
            )
        return tuple(ids)

    def inspect_container(
        self,
        reference: str,
        *,
        deadline_monotonic: float,
    ) -> DockerContainerInspection | None:
        if (
            not isinstance(reference, str)
            or not reference
            or "\x00" in reference
            or len(reference) > 4_096
        ):
            raise ValueError("Docker container reference is invalid")
        response = self._request(
            "GET",
            f"/v{self._api_version}/containers/{quote(reference, safe='')}/json",
            deadline_monotonic=deadline_monotonic,
        )
        if response.status == 404:
            return None
        self._expect_status(response, 200, operation="container inspect")
        raw = _decode_json(response.body, operation="container inspect")
        if not isinstance(raw, dict):
            raise DockerEngineProtocolError(
                "Docker Engine inspect response must be an object"
            )
        container_id = raw.get("Id")
        if not isinstance(container_id, str) or not _CONTAINER_ID_RE.fullmatch(
            container_id
        ):
            raise DockerEngineProtocolError(
                "Docker Engine inspect returned a non-canonical container ID"
            )
        config = raw.get("Config")
        if not isinstance(config, dict):
            raise DockerEngineProtocolError(
                "Docker Engine inspect response lacks Config"
            )
        labels_raw = config.get("Labels")
        if labels_raw is None:
            labels: dict[str, str] = {}
        elif isinstance(labels_raw, dict) and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in labels_raw.items()
        ):
            labels = labels_raw
        else:
            raise DockerEngineProtocolError(
                "Docker Engine inspect labels are malformed"
            )
        return DockerContainerInspection.from_parts(container_id, labels)

    def remove_container(
        self,
        container_id: str,
        *,
        deadline_monotonic: float,
    ) -> None:
        self.remove_container_with_result(
            container_id,
            deadline_monotonic=deadline_monotonic,
        )

    def remove_container_with_result(
        self,
        container_id: str,
        *,
        deadline_monotonic: float,
    ) -> str:
        """Remove one exact ID and retain whether Docker returned 204 or 404."""

        if not isinstance(container_id, str) or not _CONTAINER_ID_RE.fullmatch(
            container_id
        ):
            raise ValueError(
                "Docker removal requires a full canonical 64-hex container ID"
            )
        query = urlencode({"force": "1", "v": "1"})
        response = self._request(
            "DELETE",
            f"/v{self._api_version}/containers/{container_id}?{query}",
            deadline_monotonic=deadline_monotonic,
        )
        # A concurrent authoritative absence is already the requested state.
        if response.status not in {204, 404}:
            raise DockerEngineQueryError(
                f"Docker Engine container remove returned HTTP {response.status}"
            )
        return "REMOVED_HTTP_204" if response.status == 204 else "ABSENT_HTTP_404"


__all__ = [
    "DockerDaemonGenerationError",
    "DockerDaemonGenerationProbe",
    "DockerEngineControlPlaneError",
    "DockerEngineDeadlineError",
    "DockerEngineHTTPResponse",
    "DockerEngineProtocolError",
    "DockerEngineQueryError",
    "DockerEngineTransport",
    "DockerPeerCredentials",
    "LocalDockerDaemonGenerationProbe",
    "ProcDockerDaemonGenerationProbe",
    "SystemdSocketActivatedDockerDaemonGenerationProbe",
    "UnixDockerEngineControlPlane",
    "UnixDockerEngineTransport",
]
