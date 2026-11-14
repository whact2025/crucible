#!/usr/bin/python3
# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Privileged ext4 project-quota helper for Crucible sandboxes.

The command is intentionally small and self-contained so operators can copy it
to a root-owned location without installing Crucible as root.  It implements
the protocol consumed by :class:`crucible.common.storage_quota.KernelQuotaBackend`::

    crucible-quota-helper reconcile --owner-host HOST
    crucible-quota-helper reserve --owner-host HOST --owner-pid PID \
        --owner-start-token TOKEN --limit-bytes BYTES --path PATH [...]
    crucible-quota-helper release --token TOKEN
    crucible-quota-helper retire --token TOKEN

Reservations use one ext4 project ID for all supplied paths, so the hard limit
is aggregate.  Releasing a lease does *not* remove its quota while any leased
path remains.  This matters because Crucible can reach its Python ``finally``
block before a container has completely stopped.  Released/dead leases become
tombstones; they are retired only after all paths disappear and the kernel
reports zero project usage.

``release`` is an idempotent request to begin that cleanup and can succeed
while the tombstone remains.  ``retire`` is the exact proof operation: it
fails unless the known lease is already RELEASING (or RETIRED), every leased
path is absent, charged bytes and inodes are zero, the exact original nonzero
hard limits remain enforced, and RETIRED is durably visible in the ledger.
Retired project IDs and their protective caps are quarantined forever; neither
is cleared or reused.  Retirement is mutation-free.  Startup reconciliation
may establish the original protective cap only when complete kernel evidence
shows that every block and inode limit is still unset; partial or nonzero
limits are never overwritten.

Each new lease also persists the canonical root, root inode/type/device,
descriptor-derived filesystem ID, full mountinfo identity (including its root),
and ext4 UUID captured from a held ``O_NOFOLLOW`` root descriptor.  Legacy rows
are schema-migrated without inventing those fields and therefore cannot retire.

Default configuration (overridable by a root-owned JSON file) is::

    {
      "allowed_root": "/data/crucible-swebench",
      "ledger_path": "/var/lib/crucible-quota-helper/leases.sqlite3",
      "project_id_min": 100000,
      "project_id_max": 999999,
      "inode_hard_limit": 1000000
    }

The allowed root must itself be an ext4 mount with the ``project`` feature and
``prjquota`` mount option.  Lease paths must be strict descendants on that
same mount.  Project IDs are monotonically allocated and never reused.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import fcntl
import json
import os
import re
import secrets
import socket
import sqlite3
import stat
import struct
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator, Literal, Sequence


DEFAULT_CONFIG_PATH = Path("/etc/crucible/quota-helper.json")
DEFAULT_ALLOWED_ROOT = Path("/data/crucible-swebench")
DEFAULT_LEDGER_PATH = Path(
    "/var/lib/crucible-quota-helper/leases.sqlite3"
)
DEFAULT_PROJECT_ID_MIN = 100_000
DEFAULT_PROJECT_ID_MAX = 999_999
DEFAULT_INODE_HARD_LIMIT = 1_000_000

LEASE_PREPARING = "PREPARING"
LEASE_ACTIVE = "ACTIVE"
LEASE_RELEASING = "RELEASING"
LEASE_RETIRED = "RETIRED"
NON_RETIRED_STATES = (LEASE_PREPARING, LEASE_ACTIVE, LEASE_RELEASING)
_LINUX_FD_ALIAS_RE = re.compile(
    r"(?P<root>/proc/self/fd|/dev/fd)/(?P<descriptor>[1-9][0-9]*)\Z"
)
_LEASE_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_EXT4_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_LEDGER_SCHEMA_VERSION = "2"
_AUTHORITY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("authority_root", "TEXT"),
    ("root_device_major", "INTEGER"),
    ("root_device_minor", "INTEGER"),
    ("root_inode", "TEXT"),
    ("root_type", "INTEGER"),
    ("root_fsid", "TEXT"),
    ("mount_device", "TEXT"),
    ("mountpoint", "TEXT"),
    ("mount_root", "TEXT"),
    ("mount_filesystem", "TEXT"),
    ("filesystem_uuid", "TEXT"),
)

# linux/fs.h and linux/quota.h constants.  fsxattr's ABI is stable and is the
# interface used by lsattr/chattr.  Direct ioctls let us verify the exact inode
# opened with O_NOFOLLOW instead of parsing path-bearing command output.
FS_XFLAG_PROJINHERIT = 0x00000200
FS_IOC_FSGETXATTR = 0x801C581F
FS_IOC_FSSETXATTR = 0x401C5820
FSXATTR = struct.Struct("=IIIII8s")
PRJQUOTA = 2
Q_GETQUOTA = (0x800007 << 8) | PRJQUOTA
QIF_BLIMITS = 1
QIF_SPACE = 2
QIF_ILIMITS = 4
QIF_INODES = 8


class QuotaHelperError(RuntimeError):
    """An operation could not be completed without weakening enforcement."""


@dataclass(frozen=True)
class Config:
    allowed_root: Path = DEFAULT_ALLOWED_ROOT
    ledger_path: Path = DEFAULT_LEDGER_PATH
    project_id_min: int = DEFAULT_PROJECT_ID_MIN
    project_id_max: int = DEFAULT_PROJECT_ID_MAX
    inode_hard_limit: int = DEFAULT_INODE_HARD_LIMIT

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.is_absolute():
            raise QuotaHelperError("configuration path must be absolute")
        if not path.exists():
            if path != DEFAULT_CONFIG_PATH:
                raise QuotaHelperError(f"configuration does not exist: {path}")
            config = cls()
            config.validate()
            return config

        with _open_secure_regular(path, expected_uid=os.geteuid()) as stream:
            try:
                raw = json.load(stream)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise QuotaHelperError(f"invalid configuration: {exc}") from exc
        if not isinstance(raw, dict):
            raise QuotaHelperError("configuration must be a JSON object")
        allowed_keys = {
            "allowed_root",
            "ledger_path",
            "project_id_min",
            "project_id_max",
            "inode_hard_limit",
        }
        unknown = sorted(set(raw) - allowed_keys)
        if unknown:
            raise QuotaHelperError(
                f"unknown configuration keys: {', '.join(unknown)}"
            )
        try:
            config = cls(
                allowed_root=Path(raw.get("allowed_root", DEFAULT_ALLOWED_ROOT)),
                ledger_path=Path(raw.get("ledger_path", DEFAULT_LEDGER_PATH)),
                project_id_min=_plain_int(
                    raw.get("project_id_min", DEFAULT_PROJECT_ID_MIN),
                    "project_id_min",
                ),
                project_id_max=_plain_int(
                    raw.get("project_id_max", DEFAULT_PROJECT_ID_MAX),
                    "project_id_max",
                ),
                inode_hard_limit=_plain_int(
                    raw.get("inode_hard_limit", DEFAULT_INODE_HARD_LIMIT),
                    "inode_hard_limit",
                ),
            )
        except (TypeError, ValueError) as exc:
            raise QuotaHelperError(f"invalid configuration: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not self.allowed_root.is_absolute():
            raise QuotaHelperError("allowed_root must be absolute")
        if not self.ledger_path.is_absolute():
            raise QuotaHelperError("ledger_path must be absolute")
        if not (1 <= self.project_id_min <= self.project_id_max < 2**32 - 1):
            raise QuotaHelperError("invalid project ID range")
        if not (1 <= self.inode_hard_limit < 2**63):
            raise QuotaHelperError("inode_hard_limit must be positive")
        if _is_relative_to(self.ledger_path, self.allowed_root):
            raise QuotaHelperError("ledger_path must be outside allowed_root")


@dataclass(frozen=True)
class MountInfo:
    mount_id: int
    device: str
    root: str
    mountpoint: Path
    filesystem: str
    source: str
    options: frozenset[str]


@dataclass(frozen=True)
class Tools:
    setquota: str
    tune2fs: str

    @classmethod
    def discover(cls) -> "Tools":
        return cls(
            setquota=_trusted_executable("setquota"),
            tune2fs=_trusted_executable("tune2fs"),
        )


@dataclass(frozen=True)
class Node:
    path: Path
    device: int
    inode: int
    mode: int
    links: int

    @property
    def is_dir(self) -> bool:
        return stat.S_ISDIR(self.mode)

    @property
    def is_regular(self) -> bool:
        return stat.S_ISREG(self.mode)

    @property
    def is_symlink(self) -> bool:
        return stat.S_ISLNK(self.mode)

    @property
    def signature(self) -> tuple[str, int, int, int]:
        return (str(self.path), self.device, self.inode, stat.S_IFMT(self.mode))


@dataclass(frozen=True)
class FsXattr:
    xflags: int
    extsize: int
    nextents: int
    project_id: int
    cowextsize: int
    pad: bytes

    @classmethod
    def unpack(cls, value: bytes) -> "FsXattr":
        return cls(*FSXATTR.unpack(value))

    def pack(self) -> bytes:
        return FSXATTR.pack(
            self.xflags,
            self.extsize,
            self.nextents,
            self.project_id,
            self.cowextsize,
            self.pad,
        )


class IfDqblk(ctypes.Structure):
    _fields_ = [
        ("bhardlimit", ctypes.c_uint64),
        ("bsoftlimit", ctypes.c_uint64),
        ("curspace", ctypes.c_uint64),
        ("ihardlimit", ctypes.c_uint64),
        ("isoftlimit", ctypes.c_uint64),
        ("curinodes", ctypes.c_uint64),
        ("btime", ctypes.c_uint64),
        ("itime", ctypes.c_uint64),
        ("valid", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class Quota:
    hard_blocks: int
    soft_blocks: int
    current_bytes: int
    hard_inodes: int
    soft_inodes: int
    current_inodes: int
    valid: int


@dataclass(frozen=True)
class FilesystemAuthority:
    canonical_root: Path
    root_device_major: int
    root_device_minor: int
    root_inode: str
    root_type: int
    root_fsid: str
    mount_id: int
    mount_device: str
    mountpoint: Path
    mount_root: str
    mount_filesystem: str
    mount_source: str
    filesystem_uuid: str


@dataclass(frozen=True)
class RetirementEvidence:
    token: str
    project_id: int
    state: str
    filesystem_authority: FilesystemAuthority
    limit_bytes: int
    hard_blocks: int
    hard_inodes: int
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class QuarantineRepairResult:
    project_id: int
    status: Literal["already_established", "established"]


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_root() -> None:
    if os.geteuid() != 0:
        raise QuotaHelperError("quota helper must run as root")


def _assert_secure_script() -> None:
    script_text = __file__
    if not isinstance(script_text, str):
        raise QuotaHelperError("helper executable identity is not textual")
    descriptor_match = _LINUX_FD_ALIAS_RE.fullmatch(script_text)
    if descriptor_match is None:
        script = Path(script_text)
        try:
            info = script.lstat()
        except OSError as exc:
            raise QuotaHelperError(
                f"cannot inspect helper executable: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise QuotaHelperError(
                "helper executable must be a regular file, not a symlink"
            )
    else:
        alias_root = descriptor_match.group("root")
        descriptor_text = descriptor_match.group("descriptor")
        descriptor = int(descriptor_text)
        if descriptor < 3 or str(descriptor) != descriptor_text:
            raise QuotaHelperError(
                "helper executable descriptor alias is not canonical"
            )
        try:
            alias_root_info = os.lstat(alias_root)
            followed_root_info = os.stat(alias_root, follow_symlinks=True)
            proc_root_info = os.stat("/proc/self/fd", follow_symlinks=True)
            alias_info = os.lstat(script_text)
            info = os.fstat(descriptor)
            followed_info = os.stat(script_text, follow_symlinks=True)
        except OSError as exc:
            raise QuotaHelperError(
                f"cannot inspect helper executable descriptor: {exc}"
            ) from exc
        alias_root_type_is_valid = (
            alias_root == "/proc/self/fd"
            and stat.S_ISDIR(alias_root_info.st_mode)
        ) or (alias_root == "/dev/fd" and stat.S_ISLNK(alias_root_info.st_mode))
        if (
            not alias_root_type_is_valid
            or (
                followed_root_info.st_dev,
                followed_root_info.st_ino,
                followed_root_info.st_mode,
            )
            != (
                proc_root_info.st_dev,
                proc_root_info.st_ino,
                proc_root_info.st_mode,
            )
            or not stat.S_ISLNK(alias_info.st_mode)
            or (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_uid,
                info.st_nlink,
            )
            != (
                followed_info.st_dev,
                followed_info.st_ino,
                followed_info.st_mode,
                followed_info.st_uid,
                followed_info.st_nlink,
            )
        ):
            raise QuotaHelperError(
                "helper executable descriptor alias changed during inspection"
            )
        if not stat.S_ISREG(info.st_mode):
            raise QuotaHelperError(
                "helper executable descriptor must hold a regular file"
            )
    if info.st_uid != 0 or info.st_mode & 0o022:
        raise QuotaHelperError(
            "helper executable must be root-owned and not group/world writable"
        )


@contextlib.contextmanager
def _open_secure_regular(path: Path, *, expected_uid: int) -> Iterator[IO[str]]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QuotaHelperError(f"cannot securely open {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise QuotaHelperError(f"not a regular file: {path}")
        if info.st_uid != expected_uid or info.st_mode & 0o022:
            raise QuotaHelperError(
                f"{path} must be owned by uid {expected_uid} and not "
                "group/world writable"
            )
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
            yield stream
    finally:
        os.close(descriptor)


def _trusted_executable(name: str) -> str:
    for directory in ("/usr/sbin", "/usr/bin", "/sbin", "/bin"):
        candidate = Path(directory, name)
        try:
            info = candidate.stat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISREG(info.st_mode)
            and info.st_uid == 0
            and not info.st_mode & 0o022
            and os.access(candidate, os.X_OK)
        ):
            return str(candidate)
    raise QuotaHelperError(f"required trusted executable not found: {name}")


def _run(command: Sequence[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise QuotaHelperError(f"command failed to execute: {command[0]}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip()[-1000:] or result.stdout.strip()[-1000:]
        raise QuotaHelperError(
            f"command failed ({result.returncode}): {command[0]}: {detail}"
        )
    return result


def _unescape_mount_field(value: str) -> str:
    replacements = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
    for encoded, decoded in replacements.items():
        value = value.replace(encoded, decoded)
    return value


def _read_mountinfo(path: Path = Path("/proc/self/mountinfo")) -> tuple[MountInfo, ...]:
    mounts: list[MountInfo] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise QuotaHelperError(f"cannot read mount table: {exc}") from exc
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            mounts.append(
                MountInfo(
                    mount_id=int(fields[0]),
                    device=fields[2],
                    root=_unescape_mount_field(fields[3]),
                    mountpoint=Path(_unescape_mount_field(fields[4])),
                    filesystem=fields[separator + 1],
                    source=_unescape_mount_field(fields[separator + 2]),
                    options=frozenset(
                        fields[5].split(",")
                        + fields[separator + 3].split(",")
                    ),
                )
            )
        except (IndexError, ValueError) as exc:
            raise QuotaHelperError("malformed /proc/self/mountinfo") from exc
    return tuple(mounts)


def _mount_for(path: Path, mounts: Sequence[MountInfo]) -> MountInfo:
    candidates = [mount for mount in mounts if _is_relative_to(path, mount.mountpoint)]
    if not candidates:
        raise QuotaHelperError(f"cannot identify mount for {path}")
    return max(candidates, key=lambda item: len(item.mountpoint.parts))


def _canonical_no_symlinks(path: Path, *, require_directory: bool) -> Path:
    if not path.is_absolute():
        raise QuotaHelperError(f"path must be absolute: {path}")
    if any(ord(character) < 32 for character in str(path)):
        raise QuotaHelperError(f"path contains control characters: {path!s}")
    try:
        canonical = Path(os.path.realpath(path, strict=True))
    except OSError as exc:
        raise QuotaHelperError(f"path does not resolve: {path}: {exc}") from exc
    if canonical != path:
        raise QuotaHelperError(f"path must already be canonical: {path}")

    current = Path("/")
    for component in canonical.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except OSError as exc:
            raise QuotaHelperError(f"cannot inspect path component {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise QuotaHelperError(f"symlink path component is forbidden: {current}")
    if require_directory and not canonical.is_dir():
        raise QuotaHelperError(f"path is not a directory: {canonical}")
    return canonical


@contextlib.contextmanager
def _held_root_descriptor(root: Path) -> Iterator[int]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise QuotaHelperError(f"cannot securely open quota root {root}: {exc}") from exc
    try:
        yield descriptor
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise QuotaHelperError(
                f"cannot close held quota-root descriptor: {exc}"
            ) from exc


def _ext4_uuid_and_features(tools: Tools, mount: MountInfo) -> tuple[str, set[str]]:
    result = _run([tools.tune2fs, "-l", mount.source])
    feature_lines = [
        line.partition(":")[2].split()
        for line in result.stdout.splitlines()
        if line.startswith("Filesystem features:")
    ]
    uuid_lines = [
        line.partition(":")[2].strip().lower()
        for line in result.stdout.splitlines()
        if line.startswith("Filesystem UUID:")
    ]
    if len(feature_lines) != 1:
        raise QuotaHelperError("tune2fs returned ambiguous filesystem features")
    if len(uuid_lines) != 1 or _EXT4_UUID_RE.fullmatch(uuid_lines[0]) is None:
        raise QuotaHelperError("tune2fs returned no exact ext4 filesystem UUID")
    return uuid_lines[0], set(feature_lines[0])


def _filesystem_context(
    config: Config,
    tools: Tools,
) -> tuple[Path, MountInfo, tuple[MountInfo, ...], FilesystemAuthority]:
    root = _canonical_no_symlinks(config.allowed_root, require_directory=True)
    with _held_root_descriptor(root) as descriptor:
        try:
            root_info = os.fstat(descriptor)
            filesystem_info = os.fstatvfs(descriptor)
        except OSError as exc:
            raise QuotaHelperError(
                f"cannot inspect held quota-root descriptor: {exc}"
            ) from exc
        mounts = _read_mountinfo()
        mount = _mount_for(root, mounts)
        if mount.mountpoint != root:
            raise QuotaHelperError("allowed_root must be a dedicated mount point")
        if mount.filesystem != "ext4":
            raise QuotaHelperError(
                f"allowed_root must use ext4, found {mount.filesystem!r}"
            )
        if "prjquota" not in mount.options:
            raise QuotaHelperError("allowed_root mount is missing the prjquota option")
        if not mount.root.startswith("/") or any(
            ord(character) < 32 for character in mount.root
        ):
            raise QuotaHelperError("allowed_root mount has malformed mountinfo root")
        match = re.fullmatch(r"([0-9]+):([0-9]+)", mount.device)
        if match is None:
            raise QuotaHelperError("allowed_root mount has malformed device identity")
        device_major = os.major(root_info.st_dev)
        device_minor = os.minor(root_info.st_dev)
        if (
            int(match.group(1)) != device_major
            or int(match.group(2)) != device_minor
            or not stat.S_ISDIR(root_info.st_mode)
        ):
            raise QuotaHelperError(
                "held quota-root descriptor does not match its ext4 mount"
            )
        filesystem_uuid, features = _ext4_uuid_and_features(tools, mount)
        if "project" not in features:
            raise QuotaHelperError("ext4 filesystem is missing the project feature")
        fsid = getattr(filesystem_info, "f_fsid", None)
        if type(fsid) is not int:
            raise QuotaHelperError("held quota root has no exact f_fsid identity")
        authority = FilesystemAuthority(
            canonical_root=root,
            root_device_major=device_major,
            root_device_minor=device_minor,
            root_inode=str(root_info.st_ino),
            root_type=stat.S_IFMT(root_info.st_mode),
            root_fsid=str(fsid),
            mount_id=mount.mount_id,
            mount_device=mount.device,
            mountpoint=mount.mountpoint,
            mount_root=mount.root,
            mount_filesystem=mount.filesystem,
            mount_source=mount.source,
            filesystem_uuid=filesystem_uuid,
        )
    return root, mount, mounts, authority


def _validate_roots(
    requested: Sequence[Path],
    *,
    root: Path,
    mount: MountInfo,
    mounts: Sequence[MountInfo],
) -> tuple[Path, ...]:
    if not requested:
        raise QuotaHelperError("at least one --path is required")
    roots = tuple(
        _canonical_no_symlinks(path, require_directory=True) for path in requested
    )
    if len(set(roots)) != len(roots):
        raise QuotaHelperError("duplicate lease path")
    for path in roots:
        if path == root or not _is_relative_to(path, root):
            raise QuotaHelperError(
                f"lease path must be a strict descendant of {root}: {path}"
            )
        if _mount_for(path, mounts).mount_id != mount.mount_id:
            raise QuotaHelperError(f"lease path crosses a mount boundary: {path}")
    for index, path in enumerate(roots):
        for other in roots[index + 1 :]:
            if _is_relative_to(path, other) or _is_relative_to(other, path):
                raise QuotaHelperError("lease paths may not overlap")
    for nested in mounts:
        if nested.mount_id == mount.mount_id:
            continue
        if any(_is_relative_to(nested.mountpoint, path) for path in roots):
            raise QuotaHelperError(
                f"nested mount is forbidden in lease path: {nested.mountpoint}"
            )
    return roots


def _scan_tree(roots: Sequence[Path], *, expected_device: int) -> tuple[Node, ...]:
    nodes: list[Node] = []
    stack = list(reversed(roots))
    while stack:
        path = stack.pop()
        try:
            info = path.lstat()
        except OSError as exc:
            raise QuotaHelperError(f"cannot inspect lease tree {path}: {exc}") from exc
        node = Node(path, info.st_dev, info.st_ino, info.st_mode, info.st_nlink)
        if node.device != expected_device:
            raise QuotaHelperError(f"cross-device inode in lease tree: {path}")
        if not (node.is_dir or node.is_regular or node.is_symlink):
            raise QuotaHelperError(f"unsupported special file in lease tree: {path}")
        nodes.append(node)
        if not node.is_dir:
            continue
        try:
            with os.scandir(path) as entries:
                children = sorted((Path(entry.path) for entry in entries), reverse=True)
        except OSError as exc:
            raise QuotaHelperError(f"cannot enumerate lease tree {path}: {exc}") from exc
        stack.extend(children)

    link_counts = Counter(
        (node.device, node.inode) for node in nodes if node.is_regular
    )
    for node in nodes:
        if node.is_regular and link_counts[(node.device, node.inode)] != node.links:
            raise QuotaHelperError(
                f"file has a hard link outside the leased roots: {node.path}"
            )
    return tuple(nodes)


@contextlib.contextmanager
def _open_node(node: Node) -> Iterator[int]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    if node.is_dir:
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(node.path, flags)
    except OSError as exc:
        raise QuotaHelperError(f"cannot securely open inode {node.path}: {exc}") from exc
    try:
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino, stat.S_IFMT(current.st_mode)) != (
            node.device,
            node.inode,
            stat.S_IFMT(node.mode),
        ):
            raise QuotaHelperError(f"inode changed during reservation: {node.path}")
        yield descriptor
    finally:
        os.close(descriptor)


def _get_fsxattr(node: Node) -> FsXattr:
    if node.is_symlink:
        raise QuotaHelperError("internal error: fsxattr requested for symlink")
    with _open_node(node) as descriptor:
        buffer = bytearray(FSXATTR.size)
        try:
            fcntl.ioctl(descriptor, FS_IOC_FSGETXATTR, buffer, True)
        except OSError as exc:
            raise QuotaHelperError(f"cannot read project ID for {node.path}: {exc}") from exc
    return FsXattr.unpack(bytes(buffer))


def _set_project(node: Node, project_id: int) -> None:
    if node.is_symlink:
        return
    with _open_node(node) as descriptor:
        buffer = bytearray(FSXATTR.size)
        try:
            fcntl.ioctl(descriptor, FS_IOC_FSGETXATTR, buffer, True)
            current = FsXattr.unpack(bytes(buffer))
            updated = FsXattr(
                xflags=(
                    current.xflags | FS_XFLAG_PROJINHERIT
                    if node.is_dir
                    else current.xflags
                ),
                extsize=current.extsize,
                nextents=current.nextents,
                project_id=project_id,
                cowextsize=current.cowextsize,
                pad=current.pad,
            )
            fcntl.ioctl(descriptor, FS_IOC_FSSETXATTR, updated.pack())
        except OSError as exc:
            raise QuotaHelperError(f"cannot tag project inode {node.path}: {exc}") from exc


def _validate_fresh(nodes: Sequence[Node]) -> None:
    for node in nodes:
        if node.is_symlink:
            continue
        attributes = _get_fsxattr(node)
        if attributes.project_id:
            raise QuotaHelperError(
                f"existing nonzero project ID on {node.path}: {attributes.project_id}"
            )
        if node.is_dir and attributes.xflags & FS_XFLAG_PROJINHERIT:
            raise QuotaHelperError(f"existing project inheritance flag on {node.path}")


def _verify_project(nodes: Sequence[Node], project_id: int) -> None:
    for node in nodes:
        if node.is_symlink:
            continue
        attributes = _get_fsxattr(node)
        if attributes.project_id != project_id:
            raise QuotaHelperError(
                f"project ID verification failed for {node.path}: "
                f"expected {project_id}, found {attributes.project_id}"
            )
        if node.is_dir and not attributes.xflags & FS_XFLAG_PROJINHERIT:
            raise QuotaHelperError(
                f"project inheritance verification failed for {node.path}"
            )


def _set_quota(
    tools: Tools,
    mount: MountInfo,
    project_id: int,
    hard_blocks: int,
    hard_inodes: int,
) -> None:
    _run(
        [
            tools.setquota,
            "-P",
            str(project_id),
            "0",
            str(hard_blocks),
            "0",
            str(hard_inodes),
            str(mount.mountpoint),
        ]
    )


def _get_quota(mount: MountInfo, project_id: int) -> Quota:
    libc = ctypes.CDLL(None, use_errno=True)
    quotactl = libc.quotactl
    quotactl.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
    quotactl.restype = ctypes.c_int
    value = IfDqblk()
    result = quotactl(
        Q_GETQUOTA,
        os.fsencode(mount.source),
        project_id,
        ctypes.byref(value),
    )
    if result:
        error = ctypes.get_errno()
        raise QuotaHelperError(
            f"cannot query project quota {project_id}: {os.strerror(error)}"
        )
    return Quota(
        hard_blocks=value.bhardlimit,
        soft_blocks=value.bsoftlimit,
        current_bytes=value.curspace,
        hard_inodes=value.ihardlimit,
        soft_inodes=value.isoftlimit,
        current_inodes=value.curinodes,
        valid=value.valid,
    )


def _verify_hard_quota(
    mount: MountInfo,
    project_id: int,
    hard_blocks: int,
    hard_inodes: int,
) -> None:
    quota = _get_quota(mount, project_id)
    required = QIF_BLIMITS | QIF_ILIMITS
    if (
        quota.valid & required != required
        or quota.hard_blocks != hard_blocks
        or quota.hard_inodes != hard_inodes
    ):
        raise QuotaHelperError(
            f"hard quota verification failed for project {project_id}"
        )


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except OSError as exc:
        raise QuotaHelperError(f"cannot read kernel boot ID: {exc}") from exc
    if not value:
        raise QuotaHelperError("kernel boot ID is empty")
    return value


def _process_identity(pid: int) -> tuple[str, str] | None:
    if pid <= 0:
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    command_end = raw.rfind(")")
    if command_end < 0:
        return None
    fields = raw[command_end + 1 :].split()
    try:
        state = fields[0]
        start_token = fields[19]
        int(start_token)
    except (IndexError, ValueError):
        return None
    if state in {"Z", "X", "x"}:
        return None
    return start_token, state


def _owner_is_alive(
    *, owner_host: str, owner_pid: int, owner_start_token: str, owner_boot_id: str
) -> bool:
    if owner_host != socket.gethostname() or owner_boot_id != _boot_id():
        return False
    identity = _process_identity(owner_pid)
    return identity is not None and identity[0] == owner_start_token


class Ledger:
    """Root-owned SQLite ledger serialized by an adjacent advisory lock."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.lock_path = config.ledger_path.with_suffix(
            config.ledger_path.suffix + ".lock"
        )
        self.lock_descriptor: int | None = None
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> "Ledger":
        parent = self.config.ledger_path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = parent.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise QuotaHelperError(
                f"ledger directory must be private and owned by uid {os.geteuid()}: {parent}"
            )
        self.lock_descriptor = self._open_state_file(self.lock_path)
        fcntl.flock(self.lock_descriptor, fcntl.LOCK_EX)
        ledger_descriptor = self._open_state_file(self.config.ledger_path)
        os.close(ledger_descriptor)
        try:
            connection = sqlite3.connect(
                self.config.ledger_path,
                timeout=30,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            self.connection = connection
            self._initialize()
        except Exception:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.lock_descriptor is not None:
            fcntl.flock(self.lock_descriptor, fcntl.LOCK_UN)
            os.close(self.lock_descriptor)
            self.lock_descriptor = None

    @staticmethod
    def _open_state_file(path: Path) -> int:
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise QuotaHelperError(f"cannot securely open state file {path}: {exc}") from exc
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            os.close(descriptor)
            raise QuotaHelperError(
                f"state file must be private and owned by uid {os.geteuid()}: {path}"
            )
        return descriptor

    @property
    def db(self) -> sqlite3.Connection:
        if self.connection is None:
            raise QuotaHelperError("ledger is not open")
        return self.connection

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
        except Exception:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _initialize(self) -> None:
        with self.transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leases (
                    token TEXT PRIMARY KEY,
                    project_id INTEGER NOT NULL UNIQUE,
                    state TEXT NOT NULL CHECK (
                        state IN ('PREPARING','ACTIVE','RELEASING','RETIRED')
                    ),
                    owner_host TEXT NOT NULL,
                    owner_pid INTEGER NOT NULL,
                    owner_start_token TEXT NOT NULL,
                    owner_boot_id TEXT NOT NULL,
                    limit_bytes INTEGER NOT NULL,
                    hard_blocks INTEGER NOT NULL,
                    hard_inodes INTEGER NOT NULL,
                    mount_id INTEGER NOT NULL,
                    mount_source TEXT NOT NULL,
                    authority_root TEXT,
                    root_device_major INTEGER,
                    root_device_minor INTEGER,
                    root_inode TEXT,
                    root_type INTEGER,
                    root_fsid TEXT,
                    mount_device TEXT,
                    mountpoint TEXT,
                    mount_root TEXT,
                    mount_filesystem TEXT,
                    filesystem_uuid TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS lease_paths (
                    token TEXT NOT NULL REFERENCES leases(token),
                    path TEXT NOT NULL,
                    PRIMARY KEY (token, path)
                );
                """
            )
            schema_row = db.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(leases)")
            }
            if schema_row is None:
                lease_count = db.execute("SELECT count(*) FROM leases").fetchone()[0]
                if lease_count:
                    raise QuotaHelperError(
                        "ledger schema identity is missing for existing leases"
                    )
                db.execute(
                    "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                    (_LEDGER_SCHEMA_VERSION,),
                )
            elif schema_row["value"] == "1":
                for name, column_type in _AUTHORITY_COLUMNS:
                    if name not in columns:
                        db.execute(
                            f"ALTER TABLE leases ADD COLUMN {name} {column_type}"
                        )
                db.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                    (_LEDGER_SCHEMA_VERSION,),
                )
            elif schema_row["value"] != _LEDGER_SCHEMA_VERSION:
                raise QuotaHelperError(
                    "unsupported ledger schema version: "
                    f"{schema_row['value']!r}"
                )
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(leases)")
            }
            missing_columns = [
                name for name, _column_type in _AUTHORITY_COLUMNS if name not in columns
            ]
            if missing_columns:
                raise QuotaHelperError(
                    "ledger schema is missing filesystem authority columns: "
                    + ", ".join(missing_columns)
                )
            expected = {
                "project_id_min": str(self.config.project_id_min),
                "project_id_max": str(self.config.project_id_max),
                "inode_hard_limit": str(self.config.inode_hard_limit),
            }
            for key, value in expected.items():
                row = db.execute(
                    "SELECT value FROM metadata WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    db.execute(
                        "INSERT INTO metadata(key, value) VALUES (?, ?)",
                        (key, value),
                    )
                elif row["value"] != value:
                    raise QuotaHelperError(
                        f"ledger metadata mismatch for {key}: {row['value']!r}"
                    )
            db.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES "
                "('next_project_id', ?)",
                (str(self.config.project_id_min),),
            )

    def create_preparing(
        self,
        *,
        token: str,
        owner_host: str,
        owner_pid: int,
        owner_start_token: str,
        owner_boot_id: str,
        limit_bytes: int,
        hard_blocks: int,
        hard_inodes: int | None = None,
        filesystem_authority: FilesystemAuthority,
        paths: Sequence[Path],
    ) -> int:
        with self.transaction() as db:
            row = db.execute(
                "SELECT value FROM metadata WHERE key = 'next_project_id'"
            ).fetchone()
            if row is None:
                raise QuotaHelperError("ledger lost project allocator state")
            project_id = int(row["value"])
            if project_id > self.config.project_id_max:
                raise QuotaHelperError("project ID range is exhausted")
            if hard_inodes is None:
                hard_inodes = self.config.inode_hard_limit
            db.execute(
                "UPDATE metadata SET value = ? WHERE key = 'next_project_id'",
                (str(project_id + 1),),
            )
            now = time.time()
            db.execute(
                """
                INSERT INTO leases(
                    token, project_id, state, owner_host, owner_pid,
                    owner_start_token, owner_boot_id, limit_bytes, hard_blocks,
                    hard_inodes,
                    mount_id, mount_source, authority_root, root_device_major,
                    root_device_minor, root_inode, root_type, root_fsid,
                    mount_device, mountpoint, mount_root, mount_filesystem,
                    filesystem_uuid, created_at, updated_at, last_error
                ) VALUES (
                    ?, ?, 'PREPARING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL
                )
                """,
                (
                    token,
                    project_id,
                    owner_host,
                    owner_pid,
                    owner_start_token,
                    owner_boot_id,
                    limit_bytes,
                    hard_blocks,
                    hard_inodes,
                    filesystem_authority.mount_id,
                    filesystem_authority.mount_source,
                    str(filesystem_authority.canonical_root),
                    filesystem_authority.root_device_major,
                    filesystem_authority.root_device_minor,
                    filesystem_authority.root_inode,
                    filesystem_authority.root_type,
                    filesystem_authority.root_fsid,
                    filesystem_authority.mount_device,
                    str(filesystem_authority.mountpoint),
                    filesystem_authority.mount_root,
                    filesystem_authority.mount_filesystem,
                    filesystem_authority.filesystem_uuid,
                    now,
                    now,
                ),
            )
            db.executemany(
                "INSERT INTO lease_paths(token, path) VALUES (?, ?)",
                ((token, str(path)) for path in paths),
            )
        return project_id

    def set_state(self, token: str, state: str, error: str | None = None) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE leases SET state = ?, updated_at = ?, last_error = ? "
                "WHERE token = ?",
                (state, time.time(), error[-2000:] if error else None, token),
            )

    def rows(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM leases ORDER BY project_id"))

    def row(self, token: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM leases WHERE token = ?", (token,)
        ).fetchone()

    def paths(self, token: str) -> tuple[Path, ...]:
        return tuple(
            Path(row["path"])
            for row in self.db.execute(
                "SELECT path FROM lease_paths WHERE token = ? ORDER BY path",
                (token,),
            )
        )

    def ensure_no_overlap(self, paths: Sequence[Path]) -> None:
        rows = self.db.execute(
            """
            SELECT lease_paths.path, leases.project_id
            FROM lease_paths JOIN leases USING(token)
            WHERE leases.state != 'RETIRED'
            """
        )
        for row in rows:
            existing = Path(row["path"])
            for candidate in paths:
                if _is_relative_to(candidate, existing) or _is_relative_to(
                    existing, candidate
                ):
                    raise QuotaHelperError(
                        f"lease path overlaps project {row['project_id']}: {candidate}"
                    )


class QuotaManager:
    def __init__(
        self,
        config: Config,
        ledger: Ledger,
        tools: Tools,
        root: Path,
        mount: MountInfo,
        mounts: Sequence[MountInfo],
        filesystem_authority: FilesystemAuthority,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.tools = tools
        self.root = root
        self.mount = mount
        self.mounts = tuple(mounts)
        self.filesystem_authority = filesystem_authority
        if (
            filesystem_authority.canonical_root != root
            or filesystem_authority.mount_id != mount.mount_id
            or filesystem_authority.mount_device != mount.device
            or filesystem_authority.mountpoint != mount.mountpoint
            or filesystem_authority.mount_root != mount.root
            or filesystem_authority.mount_filesystem != mount.filesystem
            or filesystem_authority.mount_source != mount.source
        ):
            raise QuotaHelperError(
                "initial filesystem authority does not match the held mount"
            )
        self.device = os.makedev(
            filesystem_authority.root_device_major,
            filesystem_authority.root_device_minor,
        )

    def _durable_filesystem_authority(
        self,
        row: sqlite3.Row,
    ) -> FilesystemAuthority:
        """Parse exact reserve-time authority without touching its filesystem."""

        try:
            mount_id = row["mount_id"]
            mount_source = row["mount_source"]
            authority_root = row["authority_root"]
            root_device_major = row["root_device_major"]
            root_device_minor = row["root_device_minor"]
            root_inode = row["root_inode"]
            root_type = row["root_type"]
            root_fsid = row["root_fsid"]
            mount_device = row["mount_device"]
            mountpoint = row["mountpoint"]
            mount_root = row["mount_root"]
            mount_filesystem = row["mount_filesystem"]
            filesystem_uuid = row["filesystem_uuid"]
        except (IndexError, KeyError) as exc:
            raise QuotaHelperError(
                "lease has no durable filesystem authority (legacy evidence rejected)"
            ) from exc
        if any(
            value is None
            for value in (
                authority_root,
                root_device_major,
                root_device_minor,
                root_inode,
                root_type,
                root_fsid,
                mount_device,
                mountpoint,
                mount_root,
                mount_filesystem,
                filesystem_uuid,
            )
        ):
            raise QuotaHelperError(
                "lease has incomplete durable filesystem authority "
                "(legacy evidence rejected)"
            )
        if type(mount_id) is not int or mount_id <= 0:
            raise QuotaHelperError("lease has malformed durable mount ID evidence")
        if type(mount_source) is not str or not mount_source:
            raise QuotaHelperError("lease has malformed durable mount source evidence")
        if any(ord(character) < 32 for character in mount_source):
            raise QuotaHelperError("lease has malformed durable mount source evidence")
        if (
            type(authority_root) is not str
            or not authority_root
            or any(ord(character) < 32 for character in authority_root)
        ):
            raise QuotaHelperError("lease has malformed durable root authority")
        canonical_root = Path(authority_root)
        if (
            not canonical_root.is_absolute()
            or ".." in canonical_root.parts
            or str(canonical_root) != authority_root
        ):
            raise QuotaHelperError("lease has malformed durable root authority")
        if (
            type(root_device_major) is not int
            or root_device_major < 0
            or type(root_device_minor) is not int
            or root_device_minor < 0
            or type(root_type) is not int
            or root_type != stat.S_IFDIR
        ):
            raise QuotaHelperError("lease has malformed durable root device evidence")
        if (
            type(root_inode) is not str
            or re.fullmatch(r"[1-9][0-9]*", root_inode) is None
            or type(root_fsid) is not str
            or re.fullmatch(r"-?(?:0|[1-9][0-9]*)", root_fsid) is None
            or str(int(root_fsid)) != root_fsid
        ):
            raise QuotaHelperError("lease has malformed durable root identity evidence")
        expected_device = f"{root_device_major}:{root_device_minor}"
        if type(mount_device) is not str or mount_device != expected_device:
            raise QuotaHelperError("lease has malformed durable mount device evidence")
        if (
            type(mountpoint) is not str
            or any(ord(character) < 32 for character in mountpoint)
            or str(Path(mountpoint)) != mountpoint
            or Path(mountpoint) != canonical_root
        ):
            raise QuotaHelperError("lease has malformed durable mountpoint evidence")
        if (
            type(mount_root) is not str
            or not mount_root.startswith("/")
            or any(ord(character) < 32 for character in mount_root)
        ):
            raise QuotaHelperError("lease has malformed durable mount-root evidence")
        if mount_filesystem != "ext4":
            raise QuotaHelperError("lease has malformed durable filesystem evidence")
        if (
            type(filesystem_uuid) is not str
            or _EXT4_UUID_RE.fullmatch(filesystem_uuid) is None
        ):
            raise QuotaHelperError("lease has malformed durable filesystem UUID")
        authority = FilesystemAuthority(
            canonical_root=canonical_root,
            root_device_major=root_device_major,
            root_device_minor=root_device_minor,
            root_inode=root_inode,
            root_type=root_type,
            root_fsid=root_fsid,
            mount_id=mount_id,
            mount_device=mount_device,
            mountpoint=Path(mountpoint),
            mount_root=mount_root,
            mount_filesystem=mount_filesystem,
            mount_source=mount_source,
            filesystem_uuid=filesystem_uuid,
        )
        if authority != self.filesystem_authority:
            raise QuotaHelperError(
                "lease durable filesystem authority does not match held authority"
            )
        return authority

    def _revalidate_current_filesystem(
        self,
        expected: FilesystemAuthority,
    ) -> FilesystemAuthority:
        """Re-open the root and require its full current identity."""

        root, _mount, _mounts, current = _filesystem_context(
            self.config,
            self.tools,
        )
        if (
            root != self.root
            or current != expected
            or current != self.filesystem_authority
        ):
            raise QuotaHelperError("current filesystem authority changed")
        return current

    def _validate_row_filesystem(self, row: sqlite3.Row) -> FilesystemAuthority:
        authority = self._durable_filesystem_authority(row)
        self._revalidate_current_filesystem(authority)
        return authority

    def _retirement_evidence(self, row: sqlite3.Row) -> RetirementEvidence:
        """Parse one exact durable lease after binding its mount identity."""

        filesystem_authority = self._validate_row_filesystem(row)
        try:
            token = row["token"]
            project_id = row["project_id"]
            state = row["state"]
            limit_bytes = row["limit_bytes"]
            hard_blocks = row["hard_blocks"]
            hard_inodes = row["hard_inodes"]
        except (IndexError, KeyError) as exc:
            raise QuotaHelperError(
                "lease has incomplete durable retirement evidence"
            ) from exc
        if type(token) is not str or _LEASE_TOKEN_RE.fullmatch(token) is None:
            raise QuotaHelperError("lease has malformed durable token evidence")
        if (
            type(project_id) is not int
            or not self.config.project_id_min
            <= project_id
            <= self.config.project_id_max
        ):
            raise QuotaHelperError("lease has malformed durable project ID evidence")
        if type(state) is not str or state not in {
            LEASE_PREPARING,
            LEASE_ACTIVE,
            LEASE_RELEASING,
            LEASE_RETIRED,
        }:
            raise QuotaHelperError("lease has malformed durable state evidence")
        if (
            type(limit_bytes) is not int
            or limit_bytes < 1024
            or type(hard_blocks) is not int
            or hard_blocks <= 0
            or hard_blocks != limit_bytes // 1024
            or type(hard_inodes) is not int
            or hard_inodes <= 0
            or hard_inodes != self.config.inode_hard_limit
        ):
            raise QuotaHelperError("lease has malformed durable quota-cap evidence")
        paths = self.ledger.paths(token)
        if not paths or len(set(paths)) != len(paths):
            raise QuotaHelperError(
                "lease has missing or duplicate durable path evidence"
            )
        for path in paths:
            if (
                not path.is_absolute()
                or path == self.root
                or not _is_relative_to(path, self.root)
                or ".." in path.parts
                or any(ord(character) < 32 for character in str(path))
            ):
                raise QuotaHelperError("lease has malformed durable root/path evidence")
        for index, path in enumerate(paths):
            for other in paths[index + 1 :]:
                if _is_relative_to(path, other) or _is_relative_to(other, path):
                    raise QuotaHelperError("lease durable paths overlap")
        return RetirementEvidence(
            token=token,
            project_id=project_id,
            state=state,
            filesystem_authority=filesystem_authority,
            limit_bytes=limit_bytes,
            hard_blocks=hard_blocks,
            hard_inodes=hard_inodes,
            paths=paths,
        )

    @staticmethod
    def _paths_absent(paths: Sequence[Path]) -> bool:
        # Point-in-time lexists semantics via lstat: a symlink or hostile
        # replacement at the old name is still present.
        for path in paths:
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise QuotaHelperError(
                    f"cannot inspect tombstone path {path}: {exc}"
                ) from exc
            return False
        return True

    def _read_complete_quota(
        self,
        evidence: RetirementEvidence,
    ) -> Quota:
        self._revalidate_current_filesystem(evidence.filesystem_authority)
        quota = _get_quota(self.mount, evidence.project_id)
        required = QIF_BLIMITS | QIF_SPACE | QIF_ILIMITS | QIF_INODES
        if quota.valid & required != required:
            raise QuotaHelperError(
                f"kernel did not return complete quota evidence for project "
                f"{evidence.project_id}"
            )
        return quota

    def _quota_is_quarantined(
        self,
        evidence: RetirementEvidence,
        *,
        allow_pending_usage: bool,
    ) -> bool:
        quota = self._read_complete_quota(evidence)
        if (
            quota.hard_blocks != evidence.hard_blocks
            or quota.hard_inodes != evidence.hard_inodes
        ):
            raise QuotaHelperError(
                f"protective quota cap drifted for project {evidence.project_id}"
            )
        if quota.current_bytes or quota.current_inodes:
            if allow_pending_usage:
                return False
            raise QuotaHelperError(
                f"project usage reappeared after retirement for "
                f"project {evidence.project_id}"
            )
        return True

    def _establish_quarantine(
        self,
        evidence: RetirementEvidence,
    ) -> QuarantineRepairResult:
        """Monotonically repair only an entirely absent protective cap.

        This is reconciliation authority, never retirement authority.  The
        durable original limits are the sole write values.  Any partial or
        nonzero mismatch is retained for operator diagnosis rather than being
        overwritten or loosened.
        """

        if evidence.state not in {LEASE_RELEASING, LEASE_RETIRED}:
            raise QuotaHelperError(
                "quarantine repair requires a RELEASING or RETIRED tombstone"
            )
        quota = self._read_complete_quota(evidence)
        if (
            quota.hard_blocks == evidence.hard_blocks
            and quota.hard_inodes == evidence.hard_inodes
        ):
            if quota.soft_blocks or quota.soft_inodes:
                raise QuotaHelperError(
                    f"refusing to overwrite partial or nonzero quota limits for "
                    f"project {evidence.project_id}"
                )
            return QuarantineRepairResult(
                project_id=evidence.project_id,
                status="already_established",
            )
        if any(
            (
                quota.hard_blocks,
                quota.soft_blocks,
                quota.hard_inodes,
                quota.soft_inodes,
            )
        ):
            raise QuotaHelperError(
                f"refusing to overwrite partial or nonzero quota limits for "
                f"project {evidence.project_id}"
            )
        # Narrow the observation-to-mutation window: after the first
        # eligibility decision, re-open/revalidate the filesystem and require
        # one final complete all-zero limit observation immediately before the
        # exact durable cap is written.  The quota interface has no compare-
        # and-swap primitive, so this remains a point-in-time proof against
        # trusted external quota mutation rather than an atomic exclusion.
        current = self._read_complete_quota(evidence)
        if any(
            (
                current.hard_blocks,
                current.soft_blocks,
                current.hard_inodes,
                current.soft_inodes,
            )
        ):
            raise QuotaHelperError(
                f"refusing to overwrite partial or nonzero quota limits for "
                f"project {evidence.project_id}"
            )
        _set_quota(
            self.tools,
            self.mount,
            evidence.project_id,
            evidence.hard_blocks,
            evidence.hard_inodes,
        )
        repaired = self._read_complete_quota(evidence)
        if (
            repaired.hard_blocks != evidence.hard_blocks
            or repaired.soft_blocks != 0
            or repaired.hard_inodes != evidence.hard_inodes
            or repaired.soft_inodes != 0
        ):
            raise QuotaHelperError(
                f"quarantine repair verification failed for project "
                f"{evidence.project_id}"
            )
        return QuarantineRepairResult(
            project_id=evidence.project_id,
            status="established",
        )

    def reserve(
        self,
        *,
        owner_host: str,
        owner_pid: int,
        owner_start_token: str,
        limit_bytes: int,
        requested_paths: Sequence[Path],
    ) -> str:
        if owner_host != socket.gethostname():
            raise QuotaHelperError("owner-host must exactly match the local hostname")
        if limit_bytes < 1024:
            raise QuotaHelperError("limit-bytes must be at least 1024")
        if not owner_start_token.isdecimal():
            raise QuotaHelperError("owner-start-token must be a decimal process token")
        identity = _process_identity(owner_pid)
        if identity is None or identity[0] != owner_start_token:
            raise QuotaHelperError("owner PID/start token is not currently alive")

        self._revalidate_current_filesystem(self.filesystem_authority)
        roots = _validate_roots(
            requested_paths,
            root=self.root,
            mount=self.mount,
            mounts=self.mounts,
        )
        initial_nodes = _scan_tree(roots, expected_device=self.device)
        _validate_fresh(initial_nodes)
        self.ledger.ensure_no_overlap(roots)
        reserve_authority = self._revalidate_current_filesystem(
            self.filesystem_authority
        )

        token = secrets.token_urlsafe(32)
        # setquota's block limits are KiB.  Rounding down ensures the kernel
        # cap never permits more space than Crucible requested.
        hard_blocks = limit_bytes // 1024
        hard_inodes = self.config.inode_hard_limit
        project_id = self.ledger.create_preparing(
            token=token,
            owner_host=owner_host,
            owner_pid=owner_pid,
            owner_start_token=owner_start_token,
            owner_boot_id=_boot_id(),
            limit_bytes=limit_bytes,
            hard_blocks=hard_blocks,
            hard_inodes=hard_inodes,
            filesystem_authority=reserve_authority,
            paths=roots,
        )
        try:
            # The hard cap is installed before any inode is charged to the
            # project.  Directories are processed first so concurrent creates
            # inherit the project while the remainder of the tree is tagged.
            self._revalidate_current_filesystem(reserve_authority)
            _set_quota(
                self.tools,
                self.mount,
                project_id,
                hard_blocks,
                hard_inodes,
            )
            _verify_hard_quota(
                self.mount,
                project_id,
                hard_blocks,
                hard_inodes,
            )
            ordered = sorted(initial_nodes, key=lambda node: (not node.is_dir, len(node.path.parts)))
            for node in ordered:
                _set_project(node, project_id)
            final_nodes = _scan_tree(roots, expected_device=self.device)
            _verify_project(final_nodes, project_id)
            if {node.signature for node in initial_nodes} != {
                node.signature for node in final_nodes
            }:
                raise QuotaHelperError("lease tree changed while quota was being applied")
            _verify_hard_quota(
                self.mount,
                project_id,
                hard_blocks,
                hard_inodes,
            )
            self.ledger.set_state(token, LEASE_ACTIVE)
        except Exception as exc:
            # Persist the tombstone before returning failure.  Its ID is never
            # reused and reconciliation retains any partially installed cap.
            try:
                self.ledger.set_state(token, LEASE_RELEASING, str(exc))
            except Exception as state_exc:
                raise QuotaHelperError(
                    f"reservation failed and tombstone persistence failed: {state_exc}"
                ) from exc
            raise
        return token

    def release(self, token: str) -> None:
        if not token or len(token) > 256:
            # Unknown tokens are idempotent by protocol, including malformed
            # values; no state is weakened.
            return
        row = self.ledger.row(token)
        if row is None:
            return
        self._validate_row_filesystem(row)
        if row["state"] != LEASE_RELEASING:
            if row["state"] != LEASE_RETIRED:
                self.ledger.set_state(token, LEASE_RELEASING)
        self._try_retire(token)

    def retire(self, token: str) -> None:
        """Prove one known tombstone has durably reached RETIRED.

        Unlike ``release``, this operation is deliberately not idempotent for
        unknown or malformed tokens: a successful process exit is authority
        for the unprivileged broker to forget one exact issued lease.
        """

        if _LEASE_TOKEN_RE.fullmatch(token) is None:
            raise QuotaHelperError("retire token does not match the exact shape")
        row = self.ledger.row(token)
        if row is None:
            raise QuotaHelperError("retire token does not identify a known lease")
        self._validate_row_filesystem(row)
        if row["state"] not in {LEASE_RELEASING, LEASE_RETIRED}:
            raise QuotaHelperError("lease must be RELEASING before retirement proof")
        if not self._try_retire(token):
            raise QuotaHelperError("lease retirement proof is not yet complete")
        confirmed = self.ledger.row(token)
        if confirmed is None or confirmed["state"] != LEASE_RETIRED:
            raise QuotaHelperError("durable RETIRED state verification failed")

    def reconcile(self, owner_host: str) -> None:
        if owner_host != socket.gethostname():
            raise QuotaHelperError("owner-host must exactly match the local hostname")
        candidates: list[str] = []
        # First durably transition every cleanup candidate.  The separately
        # typed monotonic repair and mutation-free retirement proof run only
        # after this transition commits.
        for row in self.ledger.rows():
            self._validate_row_filesystem(row)
            state = row["state"]
            token = row["token"]
            if state == LEASE_RETIRED:
                candidates.append(token)
                continue
            should_release = state in {LEASE_PREPARING, LEASE_RELEASING}
            if state == LEASE_ACTIVE:
                should_release = not _owner_is_alive(
                    owner_host=row["owner_host"],
                    owner_pid=row["owner_pid"],
                    owner_start_token=row["owner_start_token"],
                    owner_boot_id=row["owner_boot_id"],
                )
            if should_release:
                if state != LEASE_RELEASING:
                    self.ledger.set_state(token, LEASE_RELEASING)
                candidates.append(token)
        failures: list[str] = []
        for token in candidates:
            try:
                repair_row = self.ledger.row(token)
                if repair_row is None:
                    raise QuotaHelperError("quarantine repair lost its lease row")
                repair_evidence = self._retirement_evidence(repair_row)
                self._establish_quarantine(repair_evidence)
                self._try_retire(token)
            except QuotaHelperError as exc:
                failures.append(f"{token[:8]}: {exc}")
        if failures:
            raise QuotaHelperError("reconciliation cleanup failed: " + "; ".join(failures))

    def _try_retire(self, token: str) -> bool:
        row = self.ledger.row(token)
        if row is None:
            return False
        evidence = self._retirement_evidence(row)
        if evidence.state not in {LEASE_RELEASING, LEASE_RETIRED}:
            raise QuotaHelperError("lease must be RELEASING before retirement proof")
        self._revalidate_current_filesystem(evidence.filesystem_authority)
        if not self._paths_absent(evidence.paths):
            return False
        if not self._quota_is_quarantined(evidence, allow_pending_usage=True):
            return False
        if evidence.state != LEASE_RETIRED:
            self._revalidate_current_filesystem(evidence.filesystem_authority)
            self.ledger.set_state(token, LEASE_RETIRED)

        # Retirement crosses a durable state-commit boundary.  Re-read every
        # authority and effect after it.  Already-RETIRED rows take this same
        # path; state alone is never accepted as proof.
        confirmed_row = self.ledger.row(token)
        if confirmed_row is None or confirmed_row["state"] != LEASE_RETIRED:
            raise QuotaHelperError(
                f"durable RETIRED state verification failed for project "
                f"{evidence.project_id}"
            )
        confirmed = self._retirement_evidence(confirmed_row)
        if confirmed != RetirementEvidence(
            token=evidence.token,
            project_id=evidence.project_id,
            state=LEASE_RETIRED,
            filesystem_authority=evidence.filesystem_authority,
            limit_bytes=evidence.limit_bytes,
            hard_blocks=evidence.hard_blocks,
            hard_inodes=evidence.hard_inodes,
            paths=evidence.paths,
        ):
            raise QuotaHelperError("durable retirement evidence changed after commit")
        self._revalidate_current_filesystem(confirmed.filesystem_authority)
        if not self._paths_absent(confirmed.paths):
            raise QuotaHelperError("lease path reappeared after retirement commit")
        self._quota_is_quarantined(confirmed, allow_pending_usage=False)
        return True

    def status(self) -> dict[str, object]:
        leases = []
        for row in self.ledger.rows():
            leases.append(
                {
                    "project_id": row["project_id"],
                    "state": row["state"],
                    "owner_host": row["owner_host"],
                    "owner_pid": row["owner_pid"],
                    "limit_bytes": row["limit_bytes"],
                    "paths": [str(path) for path in self.ledger.paths(row["token"])],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "last_error": row["last_error"],
                }
            )
        return {
            "allowed_root": str(self.root),
            "mount_source": self.mount.source,
            "mount_id": self.mount.mount_id,
            "mount_root": self.mount.root,
            "filesystem": self.mount.filesystem,
            "filesystem_uuid": self.filesystem_authority.filesystem_uuid,
            "leases": leases,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"root-owned JSON configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    reserve = subparsers.add_parser("reserve")
    reserve.add_argument("--owner-host", required=True)
    reserve.add_argument("--owner-pid", type=int, required=True)
    reserve.add_argument("--owner-start-token", required=True)
    reserve.add_argument("--limit-bytes", type=int, required=True)
    reserve.add_argument("--path", type=Path, action="append", required=True)

    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("--owner-host", required=True)

    release = subparsers.add_parser("release")
    release.add_argument("--token", required=True)

    retire = subparsers.add_parser("retire")
    retire.add_argument("--token", required=True)

    subparsers.add_parser("probe")
    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        _assert_root()
        _assert_secure_script()
        config = Config.load(arguments.config)
        tools = Tools.discover()
        root, mount, mounts, filesystem_authority = _filesystem_context(
            config,
            tools,
        )
        with Ledger(config) as ledger:
            manager = QuotaManager(
                config,
                ledger,
                tools,
                root,
                mount,
                mounts,
                filesystem_authority,
            )
            if arguments.operation == "reserve":
                token = manager.reserve(
                    owner_host=arguments.owner_host,
                    owner_pid=arguments.owner_pid,
                    owner_start_token=arguments.owner_start_token,
                    limit_bytes=arguments.limit_bytes,
                    requested_paths=arguments.path,
                )
                print(token, flush=True)
            elif arguments.operation == "reconcile":
                manager.reconcile(arguments.owner_host)
            elif arguments.operation == "release":
                manager.release(arguments.token)
            elif arguments.operation == "retire":
                manager.retire(arguments.token)
            elif arguments.operation == "probe":
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "allowed_root": str(root),
                            "mount_source": mount.source,
                            "mount_root": mount.root,
                            "filesystem_uuid": filesystem_authority.filesystem_uuid,
                            "project_id_range": [
                                config.project_id_min,
                                config.project_id_max,
                            ],
                            "inode_hard_limit": config.inode_hard_limit,
                        },
                        sort_keys=True,
                    )
                )
            elif arguments.operation == "status":
                print(json.dumps(manager.status(), indent=2, sort_keys=True))
            else:  # pragma: no cover - argparse enforces the choices
                raise QuotaHelperError("unknown operation")
        return 0
    except QuotaHelperError as exc:
        print(f"crucible-quota-helper: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
