# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Descriptor-retained authority for an exact immutable input tree.

Product policy supplies one fixed source root, destination, contract
fingerprint, manifest identity, and manifest decoder.  This module owns only
the reusable transport boundary: no-follow descriptor acquisition, exact
tree inventory, byte hashing, stability revalidation, and deterministic
descriptor cleanup.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar


_MAX_DEPTH = 32
_MAX_PATH_BYTES = 8 * 1024


class RetainedInputTreeFailure(str, Enum):
    """Stable reason codes for product-owned error translation."""

    MEMBER_BYTES_UNAVAILABLE = "member bytes are unavailable"
    MEMBER_ENDED_EARLY = "member ended before its exact size"
    MEMBER_EXCEEDS_SIZE = "member exceeds its exact size"
    MANIFEST_NOT_STRICT_JSON = "manifest is not strict JSON"
    MANIFEST_NOT_OBJECT = "manifest is not an object"
    BINDING_INVALID = "input binding is invalid"
    BINDING_UNAVAILABLE = "input binding is unavailable"
    BINDING_CHANGED = "input binding changed identity"
    ANCESTRY_NOT_RETAINED = "source-root ancestry is no longer retained"
    ANCESTRY_CHANGED = "source-root ancestry changed"
    DIRECTORY_NOT_RETAINED = "directory is no longer retained"
    DIRECTORY_CHANGED = "directory identity or inventory changed"
    MEMBER_NOT_RETAINED = "member is no longer retained"
    MEMBER_METADATA_CHANGED = "member identity or metadata changed"
    MEMBER_BYTES_CHANGED = "member bytes changed"
    AUTHORITY_CLOSED = "input authority is closed"
    DESCRIPTOR_CLOSE_FAILED = "input descriptors did not close cleanly"
    INVENTORY_PATH_INVALID = "inventory path is invalid"
    INVENTORY_PATH_NOT_CANONICAL = "inventory path is not canonical"
    INVENTORY_MEMBER_INVALID = "inventory member is invalid"
    INVENTORY_INVALID = "inventory is invalid"
    DIRECTORY_CANNOT_BE_RETAINED = "directory cannot be retained"
    DIRECTORY_IDENTITY_DIFFERS = "directory identity or inventory differs"
    ANCESTRY_PARENT_NOT_CANONICAL = "source-root parent is not canonical"
    ANCESTRY_DIFFERS = "source-root ancestry differs"
    ANCESTRY_CANNOT_BE_RETAINED = "source-root ancestry cannot be retained"
    MEMBER_CANNOT_BE_RETAINED = "member cannot be retained"
    MEMBER_IDENTITY_DIFFERS = "member identity or bytes differ"
    MEMBERS_SHARE_INODE = "members share an inode"
    MANIFEST_BYTES_UNAVAILABLE = "manifest bytes are unavailable"
    MANIFEST_IDENTITY_DIFFERS = "manifest identity differs"


class RetainedInputTreeError(RuntimeError):
    """A descriptor-retained input tree failed its closed contract."""

    def __init__(self, failure: RetainedInputTreeFailure) -> None:
        self.failure = failure
        super().__init__(failure.value)


def canonical_json_bytes(value: object) -> bytes:
    """Return the exact ASCII canonical JSON form used by fingerprints."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    """Return an untagged lowercase SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def strict_json_object(payload: bytes) -> dict[str, Any]:
    """Decode one UTF-8 JSON object while rejecting duplicate fields."""

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate field {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RetainedInputTreeError(
            RetainedInputTreeFailure.MANIFEST_NOT_STRICT_JSON
        ) from exc
    if not isinstance(decoded, dict):
        raise RetainedInputTreeError(RetainedInputTreeFailure.MANIFEST_NOT_OBJECT)
    return decoded


def canonical_relative_path(value: object) -> PurePosixPath:
    """Validate and return one bounded canonical relative inventory path."""

    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > _MAX_PATH_BYTES
    ):
        raise RetainedInputTreeError(RetainedInputTreeFailure.INVENTORY_PATH_INVALID)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or len(path.parts) > _MAX_DEPTH
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RetainedInputTreeError(
            RetainedInputTreeFailure.INVENTORY_PATH_NOT_CANONICAL
        )
    return path


def _sha256_descriptor(descriptor: int, *, expected_size: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = expected_size
    while remaining:
        try:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        except OSError as exc:
            raise RetainedInputTreeError(
                RetainedInputTreeFailure.MEMBER_BYTES_UNAVAILABLE
            ) from exc
        if not chunk:
            raise RetainedInputTreeError(RetainedInputTreeFailure.MEMBER_ENDED_EARLY)
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise RetainedInputTreeError(RetainedInputTreeFailure.MEMBER_EXCEEDS_SIZE)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _metadata_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _path_directory_identity(value: os.stat_result) -> tuple[int, ...]:
    """Bind ancestry identity without coupling it to unrelated siblings."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
    )


@dataclass(frozen=True, slots=True)
class RetainedInputTreeMember:
    """One exact regular file admitted by product-owned inventory policy."""

    relative: PurePosixPath
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.relative, PurePosixPath)
            or self.relative.is_absolute()
            or str(self.relative) in {"", "."}
            or len(self.relative.parts) > _MAX_DEPTH
            or len(str(self.relative).encode("utf-8")) > _MAX_PATH_BYTES
            or any(part in {"", ".", ".."} for part in self.relative.parts)
            or isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
            or not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or set(self.sha256) - set("0123456789abcdef")
        ):
            raise RetainedInputTreeError(
                RetainedInputTreeFailure.INVENTORY_MEMBER_INVALID
            )


@dataclass(frozen=True, slots=True)
class RetainedInputTreeBinding:
    """The named mount spelling and retained identity of one input root."""

    fingerprint_schema: ClassVar[str] = "crucible.retained-input-tree-binding.v1"

    source_root: Path
    destination: PurePosixPath
    contract_fingerprint: str
    root_device: int
    root_inode: int
    root_uid: int
    root_gid: int
    root_mode: int

    def __post_init__(self) -> None:
        if (
            not self.source_root.is_absolute()
            or self.source_root == Path("/")
            or any(part in {"", ".", ".."} for part in self.source_root.parts[1:])
            or not self.destination.is_absolute()
            or self.destination == PurePosixPath("/")
            or any(part in {"", ".", ".."} for part in self.destination.parts[1:])
            or not isinstance(self.contract_fingerprint, str)
            or len(self.contract_fingerprint) != 71
            or not self.contract_fingerprint.startswith("sha256:")
            or set(self.contract_fingerprint[7:]) - set("0123456789abcdef")
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    self.root_device,
                    self.root_uid,
                    self.root_gid,
                    self.root_mode,
                )
            )
            or isinstance(self.root_inode, bool)
            or not isinstance(self.root_inode, int)
            or self.root_inode <= 0
            or self.root_mode > 0o7777
            or not isinstance(self.fingerprint_schema, str)
            or not self.fingerprint_schema
        ):
            raise RetainedInputTreeError(RetainedInputTreeFailure.BINDING_INVALID)

    @property
    def fingerprint(self) -> str:
        payload = {
            "contract_fingerprint": self.contract_fingerprint,
            "destination": str(self.destination),
            "root_device": self.root_device,
            "root_gid": self.root_gid,
            "root_inode": self.root_inode,
            "root_mode": self.root_mode,
            "root_uid": self.root_uid,
            "schema_version": self.fingerprint_schema,
            "source_root": str(self.source_root),
        }
        return "sha256:" + sha256_bytes(canonical_json_bytes(payload))

    @property
    def root_stat_identity(self) -> tuple[int, int, int, int, int]:
        """Return the exact ``stat`` projection used for mount attestation."""

        return (
            self.root_device,
            self.root_inode,
            self.root_uid,
            self.root_gid,
            self.root_mode,
        )

    def matches_root_stat_identity(
        self,
        observed: tuple[int, int, int, int, int],
    ) -> bool:
        """Compare one parsed live-mount identity with the retained root."""

        return observed == self.root_stat_identity

    def assert_named(self) -> None:
        """Re-prove that the mount spelling still names this retained root."""

        try:
            linked = self.source_root.lstat()
        except OSError as exc:
            raise RetainedInputTreeError(
                RetainedInputTreeFailure.BINDING_UNAVAILABLE
            ) from exc
        if (
            not stat.S_ISDIR(linked.st_mode)
            or stat.S_ISLNK(linked.st_mode)
            or linked.st_dev != self.root_device
            or linked.st_ino != self.root_inode
            or linked.st_uid != self.root_uid
            or linked.st_gid != self.root_gid
            or stat.S_IMODE(linked.st_mode) != self.root_mode
        ):
            raise RetainedInputTreeError(RetainedInputTreeFailure.BINDING_CHANGED)


@dataclass(slots=True)
class RetainedPathDirectory:
    """One retained component in the absolute source-root ancestry."""

    path: Path
    parent_descriptor: int | None
    name: str | None
    descriptor: int
    metadata: tuple[int, ...]

    def assert_stable(self) -> None:
        try:
            opened = os.fstat(self.descriptor)
            linked = (
                os.stat(
                    self.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
                if self.parent_descriptor is not None and self.name is not None
                else self.path.lstat()
            )
        except OSError as exc:
            raise RetainedInputTreeError(
                RetainedInputTreeFailure.ANCESTRY_NOT_RETAINED
            ) from exc
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(linked.st_mode)
            or _path_directory_identity(opened) != self.metadata
            or _path_directory_identity(linked) != self.metadata
        ):
            raise RetainedInputTreeError(RetainedInputTreeFailure.ANCESTRY_CHANGED)


@dataclass(slots=True)
class RetainedInputDirectory:
    relative: PurePosixPath
    parent_descriptor: int | None
    name: str | None
    descriptor: int
    metadata: tuple[int, ...]
    expected_names: tuple[str, ...] | None

    def assert_stable(self) -> None:
        try:
            opened = os.fstat(self.descriptor)
            names = tuple(sorted(os.listdir(self.descriptor), key=os.fsencode))
            linked = (
                os.stat(
                    self.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
                if self.parent_descriptor is not None and self.name is not None
                else opened
            )
        except OSError as exc:
            raise RetainedInputTreeError(
                RetainedInputTreeFailure.DIRECTORY_NOT_RETAINED
            ) from exc
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _metadata_identity(opened) != self.metadata
            or _metadata_identity(linked) != self.metadata
            or (self.expected_names is not None and names != self.expected_names)
        ):
            raise RetainedInputTreeError(RetainedInputTreeFailure.DIRECTORY_CHANGED)


@dataclass(slots=True)
class RetainedInputFile:
    relative: PurePosixPath
    parent_descriptor: int
    name: str
    descriptor: int
    metadata: tuple[int, ...]
    sha256: str

    def assert_stable(self, *, full: bool) -> None:
        try:
            opened = os.fstat(self.descriptor)
            linked = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RetainedInputTreeError(
                RetainedInputTreeFailure.MEMBER_NOT_RETAINED
            ) from exc
        if (
            not stat.S_ISREG(opened.st_mode)
            or _metadata_identity(opened) != self.metadata
            or _metadata_identity(linked) != self.metadata
        ):
            raise RetainedInputTreeError(
                RetainedInputTreeFailure.MEMBER_METADATA_CHANGED
            )
        if (
            full
            and _sha256_descriptor(
                self.descriptor,
                expected_size=opened.st_size,
            )
            != self.sha256
        ):
            raise RetainedInputTreeError(RetainedInputTreeFailure.MEMBER_BYTES_CHANGED)


@dataclass(slots=True)
class TrustedRetainedInputTree:
    """A complete exact tree retained by ancestry, directory, and file FDs."""

    binding: RetainedInputTreeBinding
    path_anchors: tuple[RetainedPathDirectory, ...]
    directories: tuple[RetainedInputDirectory, ...]
    files: tuple[RetainedInputFile, ...]
    _closed: bool = False

    def assert_stable(self, *, full: bool = False) -> None:
        if self._closed:
            raise RetainedInputTreeError(RetainedInputTreeFailure.AUTHORITY_CLOSED)
        for anchor in self.path_anchors:
            anchor.assert_stable()
        for directory in self.directories:
            directory.assert_stable()
        for member in self.files:
            member.assert_stable(full=full)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[OSError] = []
        for retained_group in (
            reversed(self.files),
            reversed(self.directories),
            reversed(self.path_anchors),
        ):
            for retained in retained_group:
                try:
                    os.close(retained.descriptor)
                except OSError as exc:
                    errors.append(exc)
        if errors:
            raise RetainedInputTreeError(
                RetainedInputTreeFailure.DESCRIPTOR_CLOSE_FAILED
            ) from errors[0]

    def __enter__(self) -> TrustedRetainedInputTree:
        self.assert_stable()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _open_directory(
    *,
    relative: PurePosixPath,
    parent_descriptor: int | None,
    name: str | None,
    root: Path,
    expected_names: tuple[str, ...] | None,
    root_device: int | None,
) -> RetainedInputDirectory:
    descriptor: int | None = None
    try:
        if parent_descriptor is None:
            before = root.lstat()
            descriptor = os.open(
                root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
        else:
            if name is None:  # pragma: no cover - private caller invariant
                raise AssertionError("directory name is unavailable")
            before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
        opened = os.fstat(descriptor)
        if parent_descriptor is None:
            after = root.lstat()
        else:
            if name is None:  # pragma: no cover - private caller invariant
                raise AssertionError("directory name is unavailable")
            after = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        names = tuple(sorted(os.listdir(descriptor), key=os.fsencode))
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise RetainedInputTreeError(
            RetainedInputTreeFailure.DIRECTORY_CANNOT_BE_RETAINED
        ) from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or _metadata_identity(before) != _metadata_identity(opened)
        or _metadata_identity(after) != _metadata_identity(opened)
        or opened.st_uid != os.geteuid()
        or opened.st_gid != os.getegid()
        or opened.st_mode & stat.S_IWOTH
        or (root_device is not None and opened.st_dev != root_device)
        or (expected_names is not None and names != expected_names)
    ):
        os.close(descriptor)
        raise RetainedInputTreeError(
            RetainedInputTreeFailure.DIRECTORY_IDENTITY_DIFFERS
        )
    return RetainedInputDirectory(
        relative=relative,
        parent_descriptor=parent_descriptor,
        name=name,
        descriptor=descriptor,
        metadata=_metadata_identity(opened),
        expected_names=expected_names,
    )


def _open_source_parent_anchors(root: Path) -> tuple[RetainedPathDirectory, ...]:
    """Walk one fixed absolute parent component at a time without links."""

    parent = root.parent
    if (
        not parent.is_absolute()
        or not parent.parts
        or parent.parts[0] != "/"
        or any(part in {"", ".", ".."} for part in parent.parts[1:])
    ):
        raise RetainedInputTreeError(
            RetainedInputTreeFailure.ANCESTRY_PARENT_NOT_CANONICAL
        )
    anchors: list[RetainedPathDirectory] = []
    descriptor = -1
    try:
        descriptor = os.open(
            "/",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        opened = os.fstat(descriptor)
        root_anchor = RetainedPathDirectory(
            path=Path("/"),
            parent_descriptor=None,
            name=None,
            descriptor=descriptor,
            metadata=_path_directory_identity(opened),
        )
        root_anchor.assert_stable()
        anchors.append(root_anchor)
        descriptor = -1

        current = Path("/")
        for part in parent.parts[1:]:
            prior = anchors[-1]
            linked_before = os.stat(
                part,
                dir_fd=prior.descriptor,
                follow_symlinks=False,
            )
            descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=prior.descriptor,
            )
            opened = os.fstat(descriptor)
            linked_after = os.stat(
                part,
                dir_fd=prior.descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(opened.st_mode)
                or stat.S_ISLNK(linked_before.st_mode)
                or (opened.st_mode & stat.S_IWOTH and not opened.st_mode & stat.S_ISVTX)
                or _path_directory_identity(linked_before)
                != _path_directory_identity(opened)
                or _path_directory_identity(linked_after)
                != _path_directory_identity(opened)
            ):
                raise RetainedInputTreeError(RetainedInputTreeFailure.ANCESTRY_DIFFERS)
            current /= part
            anchor = RetainedPathDirectory(
                path=current,
                parent_descriptor=prior.descriptor,
                name=part,
                descriptor=descriptor,
                metadata=_path_directory_identity(opened),
            )
            anchor.assert_stable()
            anchors.append(anchor)
            descriptor = -1
        return tuple(anchors)
    except (OSError, RetainedInputTreeError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        for retained in reversed(anchors):
            try:
                os.close(retained.descriptor)
            except OSError:
                pass
        if isinstance(exc, RetainedInputTreeError):
            raise
        raise RetainedInputTreeError(
            RetainedInputTreeFailure.ANCESTRY_CANNOT_BE_RETAINED
        ) from exc


def _open_file(
    *,
    member: RetainedInputTreeMember,
    parent: RetainedInputDirectory,
    root_device: int,
) -> RetainedInputFile:
    descriptor = -1
    try:
        linked_before = os.stat(
            member.relative.name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            member.relative.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent.descriptor,
        )
        opened = os.fstat(descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        observed_sha256 = _sha256_descriptor(
            descriptor,
            expected_size=member.size,
        )
        linked_after = os.stat(
            member.relative.name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
    except (OSError, RetainedInputTreeError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise RetainedInputTreeError(
            RetainedInputTreeFailure.MEMBER_CANNOT_BE_RETAINED
        ) from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or stat.S_ISLNK(linked_before.st_mode)
        or opened.st_nlink != 1
        or opened.st_dev != root_device
        or opened.st_uid != os.geteuid()
        or opened.st_gid != os.getegid()
        or opened.st_size != member.size
        or _metadata_identity(linked_before) != _metadata_identity(opened)
        or _metadata_identity(linked_after) != _metadata_identity(opened)
        or observed_sha256 != member.sha256
    ):
        os.close(descriptor)
        raise RetainedInputTreeError(RetainedInputTreeFailure.MEMBER_IDENTITY_DIFFERS)
    return RetainedInputFile(
        relative=member.relative,
        parent_descriptor=parent.descriptor,
        name=member.relative.name,
        descriptor=descriptor,
        metadata=_metadata_identity(opened),
        sha256=member.sha256,
    )


def open_retained_input_tree(
    *,
    source_root: Path,
    destination: PurePosixPath,
    contract_fingerprint: str,
    manifest: RetainedInputTreeMember,
    inventory_loader: Callable[[bytes], tuple[RetainedInputTreeMember, ...]],
    binding_type: type[RetainedInputTreeBinding] = RetainedInputTreeBinding,
) -> TrustedRetainedInputTree:
    """Retain and exact-validate one product-described immutable tree."""

    path_anchors: tuple[RetainedPathDirectory, ...] = ()
    retained_directories: list[RetainedInputDirectory] = []
    retained_files: list[RetainedInputFile] = []
    directory_by_path: dict[PurePosixPath, RetainedInputDirectory] = {}
    try:
        path_anchors = _open_source_parent_anchors(source_root)
        retained_root = _open_directory(
            relative=PurePosixPath("."),
            parent_descriptor=path_anchors[-1].descriptor,
            name=source_root.name,
            root=source_root,
            expected_names=None,
            root_device=None,
        )
        retained_directories.append(retained_root)
        directory_by_path[PurePosixPath(".")] = retained_root
        root_device = retained_root.metadata[0]

        retained_manifest = _open_file(
            member=manifest,
            parent=retained_root,
            root_device=root_device,
        )
        retained_files.append(retained_manifest)
        try:
            manifest_payload = os.pread(
                retained_manifest.descriptor,
                manifest.size + 1,
                0,
            )
        except OSError as exc:
            raise RetainedInputTreeError(
                RetainedInputTreeFailure.MANIFEST_BYTES_UNAVAILABLE
            ) from exc
        if (
            len(manifest_payload) != manifest.size
            or sha256_bytes(manifest_payload) != manifest.sha256
        ):
            raise RetainedInputTreeError(
                RetainedInputTreeFailure.MANIFEST_IDENTITY_DIFFERS
            )
        inventory = inventory_loader(manifest_payload)
        if not isinstance(inventory, tuple) or any(
            not isinstance(item, RetainedInputTreeMember) for item in inventory
        ):
            raise RetainedInputTreeError(RetainedInputTreeFailure.INVENTORY_INVALID)
        members = (*inventory, manifest)
        paths = tuple(item.relative for item in members)
        if len(set(paths)) != len(paths):
            raise RetainedInputTreeError(RetainedInputTreeFailure.INVENTORY_INVALID)
        directories: set[PurePosixPath] = {PurePosixPath(".")}
        expected_children: dict[PurePosixPath, set[str]] = {PurePosixPath("."): set()}
        file_records = {member.relative: member for member in members}
        for relative in file_records:
            parent = relative.parent
            expected_children.setdefault(parent, set()).add(relative.name)
            while parent != PurePosixPath("."):
                directories.add(parent)
                grandparent = parent.parent
                expected_children.setdefault(grandparent, set()).add(parent.name)
                expected_children.setdefault(parent, set())
                parent = grandparent

        retained_root.expected_names = tuple(
            sorted(expected_children[PurePosixPath(".")], key=os.fsencode)
        )
        retained_root.assert_stable()
        for relative in sorted(
            directories - {PurePosixPath(".")},
            key=lambda item: (len(item.parts), os.fsencode(str(item))),
        ):
            parent_path = relative.parent
            parent_directory = directory_by_path.get(parent_path)
            if (
                parent_directory is None
            ):  # pragma: no cover - constructed tree invariant
                raise AssertionError("directory parent is unavailable")
            retained_directory = _open_directory(
                relative=relative,
                parent_descriptor=parent_directory.descriptor,
                name=relative.name,
                root=source_root,
                expected_names=tuple(
                    sorted(expected_children[relative], key=os.fsencode)
                ),
                root_device=root_device,
            )
            retained_directories.append(retained_directory)
            directory_by_path[relative] = retained_directory
        observed_inodes: set[tuple[int, int]] = {
            (retained_manifest.metadata[0], retained_manifest.metadata[1])
        }
        for relative, member in sorted(
            file_records.items(), key=lambda item: os.fsencode(str(item[0]))
        ):
            if relative == manifest.relative:
                continue
            retained_file = _open_file(
                member=member,
                parent=directory_by_path[relative.parent],
                root_device=root_device,
            )
            inode = retained_file.metadata[0], retained_file.metadata[1]
            if inode in observed_inodes:
                os.close(retained_file.descriptor)
                raise RetainedInputTreeError(
                    RetainedInputTreeFailure.MEMBERS_SHARE_INODE
                )
            observed_inodes.add(inode)
            retained_files.append(retained_file)
        root_metadata = retained_directories[0].metadata
        binding = binding_type(
            source_root=source_root,
            destination=destination,
            contract_fingerprint=contract_fingerprint,
            root_device=root_metadata[0],
            root_inode=root_metadata[1],
            root_uid=root_metadata[4],
            root_gid=root_metadata[5],
            root_mode=stat.S_IMODE(root_metadata[2]),
        )
        authority = TrustedRetainedInputTree(
            binding=binding,
            path_anchors=path_anchors,
            directories=tuple(retained_directories),
            files=tuple(retained_files),
        )
        authority.assert_stable(full=True)
        return authority
    except BaseException:
        for retained_group in (
            reversed(retained_files),
            reversed(retained_directories),
            reversed(path_anchors),
        ):
            for retained in retained_group:
                try:
                    os.close(retained.descriptor)
                except OSError:
                    pass
        raise


__all__ = [
    "RetainedInputDirectory",
    "RetainedInputFile",
    "RetainedInputTreeBinding",
    "RetainedInputTreeError",
    "RetainedInputTreeFailure",
    "RetainedInputTreeMember",
    "RetainedPathDirectory",
    "TrustedRetainedInputTree",
    "canonical_json_bytes",
    "canonical_relative_path",
    "open_retained_input_tree",
    "sha256_bytes",
    "strict_json_object",
]
