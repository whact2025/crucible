# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Race-aware reads for routing, dataset, and backend identity files."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import TypeAlias

StatIdentity: TypeAlias = tuple[int, int, int, int, int, int, int, int, int]
DEFAULT_MAXIMUM_FILE_BYTES = 16 * 1024 * 1024


class AuthorityIOError(RuntimeError):
    """A filesystem authority is not exactly admissible."""


def stat_identity(value: os.stat_result) -> StatIdentity:
    """Return metadata sufficient to detect replacement or in-place mutation."""

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


def _require_absolute_normalized(path: Path, *, label: str) -> None:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a Path")
    if not path.is_absolute() or ".." in path.parts or "\x00" in str(path):
        raise AuthorityIOError(f"{label} must be absolute and normalized")


def _require_same(
    left: os.stat_result,
    right: os.stat_result,
    *,
    label: str,
) -> None:
    if stat_identity(left) != stat_identity(right):
        raise AuthorityIOError(f"{label} changed while being measured")


def _stable_directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def descriptor_mount_id(descriptor: int) -> int:
    """Return Linux's mount authority for one held descriptor."""

    try:
        with open(
            f"/proc/self/fdinfo/{descriptor}",
            "r",
            encoding="ascii",
            errors="strict",
        ) as stream:
            values = [
                int(line.split(":", 1)[1].strip())
                for line in stream
                if line.startswith("mnt_id:")
            ]
    except (OSError, UnicodeError, ValueError) as exc:
        raise AuthorityIOError("descriptor mount identity is unavailable") from exc
    if len(values) != 1 or values[0] <= 0:
        raise AuthorityIOError("descriptor mount identity is malformed")
    return values[0]


def _regular_file_flags() -> int:
    nonblock = getattr(os, "O_NONBLOCK", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nonblock, int) or not nonblock:
        raise AuthorityIOError("non-blocking authority-file opens are unavailable")
    if not isinstance(nofollow, int) or not nofollow:
        raise AuthorityIOError("no-follow authority-file opens are unavailable")
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nonblock | nofollow


def _directory_flags() -> int:
    directory = getattr(os, "O_DIRECTORY", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(directory, int) or not directory:
        raise AuthorityIOError("directory-only authority opens are unavailable")
    if not isinstance(nofollow, int) or not nofollow:
        raise AuthorityIOError("no-follow authority-directory opens are unavailable")
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | directory | nofollow


def _require_regular_policy(
    *,
    maximum: int,
    expected_mode: int | None,
    require_single_link: bool,
    allow_empty: bool,
) -> None:
    if type(maximum) is not int or maximum < 0:
        raise ValueError("file byte bound must be an exact non-negative integer")
    if expected_mode is not None and (
        type(expected_mode) is not int or not 0 <= expected_mode <= 0o7777
    ):
        raise ValueError("expected mode is invalid")
    if type(require_single_link) is not bool or type(allow_empty) is not bool:
        raise TypeError("file-read policy flags must be exact booleans")


def _require_regular_metadata(
    metadata: os.stat_result,
    *,
    label: str,
    maximum: int,
    expected_mode: int | None,
    require_single_link: bool,
    allow_empty: bool,
) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise AuthorityIOError(f"authority path is not a regular file: {label}")
    if require_single_link and metadata.st_nlink != 1:
        raise AuthorityIOError(f"authority file has an unbound hard link: {label}")
    if metadata.st_size < 0 or metadata.st_size > maximum:
        raise AuthorityIOError(f"authority file exceeds {maximum} bytes: {label}")
    if not allow_empty and metadata.st_size == 0:
        raise AuthorityIOError(f"authority file is empty: {label}")
    if expected_mode is not None and stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise AuthorityIOError(f"authority file mode is not exact: {label}")


def open_exact_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum: int = DEFAULT_MAXIMUM_FILE_BYTES,
    expected_mode: int | None = None,
    require_single_link: bool = True,
    allow_empty: bool = True,
) -> tuple[int, bytes, os.stat_result]:
    """Open and read one exact regular leaf beneath a held directory.

    The returned descriptor remains positioned at byte zero.  Admission uses
    non-blocking, no-follow opens and proves the parent descriptor, the file
    descriptor, the directory entry, and a second rebound descriptor retain
    identical inode and mount authority throughout the bounded read.
    """

    if type(directory_descriptor) is not int or directory_descriptor < 0:
        raise TypeError("directory descriptor must be an exact non-negative integer")
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
        or "\x00" in name
    ):
        raise AuthorityIOError("authority filename must be one exact leaf")
    _require_regular_policy(
        maximum=maximum,
        expected_mode=expected_mode,
        require_single_link=require_single_link,
        allow_empty=allow_empty,
    )

    descriptor: int | None = None
    rebound_descriptor: int | None = None
    succeeded = False
    label = name
    try:
        directory_before = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(directory_before.st_mode):
            raise AuthorityIOError("authority parent descriptor is not a directory")
        directory_mount_before = descriptor_mount_id(directory_descriptor)
        before = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        _require_regular_metadata(
            before,
            label=label,
            maximum=maximum,
            expected_mode=expected_mode,
            require_single_link=require_single_link,
            allow_empty=allow_empty,
        )
        descriptor = os.open(
            name,
            _regular_file_flags(),
            dir_fd=directory_descriptor,
        )
        opened = os.fstat(descriptor)
        opened_mount = descriptor_mount_id(descriptor)
        _require_same(before, opened, label=label)
        _require_regular_metadata(
            opened,
            label=label,
            maximum=maximum,
            expected_mode=expected_mode,
            require_single_link=require_single_link,
            allow_empty=allow_empty,
        )

        chunks: list[bytes] = []
        count = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum + 1 - count),
            )
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > maximum:
                raise AuthorityIOError(
                    f"authority file exceeded {maximum} bytes while reading: {label}"
                )

        after_descriptor = os.fstat(descriptor)
        after_path = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        rebound_descriptor = os.open(
            name,
            _regular_file_flags(),
            dir_fd=directory_descriptor,
        )
        rebound = os.fstat(rebound_descriptor)
        rebound_mount = descriptor_mount_id(rebound_descriptor)
        directory_after = os.fstat(directory_descriptor)
        directory_mount_after = descriptor_mount_id(directory_descriptor)
        _require_same(opened, after_descriptor, label=label)
        _require_same(opened, after_path, label=label)
        _require_same(opened, rebound, label=label)
        if opened_mount != rebound_mount:
            raise AuthorityIOError(
                f"authority file mount changed while being measured: {label}"
            )
        if (
            _stable_directory_identity(directory_before)
            != _stable_directory_identity(directory_after)
            or directory_mount_before != directory_mount_after
        ):
            raise AuthorityIOError(
                "authority parent descriptor changed while being measured"
            )
        payload = b"".join(chunks)
        if len(payload) != opened.st_size:
            raise AuthorityIOError(
                f"authority file size changed while reading: {label}"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        succeeded = True
        return descriptor, payload, opened
    except AuthorityIOError:
        raise
    except OSError as exc:
        raise AuthorityIOError(f"cannot read authority file: {label}") from exc
    finally:
        close_error: OSError | None = None
        if rebound_descriptor is not None:
            try:
                os.close(rebound_descriptor)
            except OSError as exc:
                close_error = exc
        if descriptor is not None and (not succeeded or close_error is not None):
            try:
                os.close(descriptor)
            except OSError as exc:
                if close_error is None:
                    close_error = exc
        if close_error is not None:
            raise AuthorityIOError(
                "cannot close authority file descriptor"
            ) from close_error


def read_exact_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum: int = DEFAULT_MAXIMUM_FILE_BYTES,
    expected_mode: int | None = None,
    require_single_link: bool = True,
    allow_empty: bool = True,
) -> tuple[bytes, os.stat_result]:
    """Read one exact regular leaf beneath a held directory."""

    descriptor, payload, metadata = open_exact_regular_at(
        directory_descriptor,
        name,
        maximum=maximum,
        expected_mode=expected_mode,
        require_single_link=require_single_link,
        allow_empty=allow_empty,
    )
    try:
        os.close(descriptor)
    except OSError as exc:
        raise AuthorityIOError("cannot close authority file descriptor") from exc
    return payload, metadata


def _open_exact_directory(path: Path) -> int:
    """Open an absolute directory through a no-symlink component chain."""

    _require_absolute_normalized(path, label="directory path")
    descriptor: int | None = os.open(os.sep, _directory_flags())
    try:
        for component in path.parts[1:]:
            assert descriptor is not None
            child = os.open(component, _directory_flags(), dir_fd=descriptor)
            try:
                os.close(descriptor)
            except OSError as exc:
                descriptor = None
                try:
                    os.close(child)
                except OSError as child_exc:
                    raise AuthorityIOError(
                        "cannot close authority-directory descriptors"
                    ) from child_exc
                raise AuthorityIOError(
                    "cannot close authority-directory descriptor"
                ) from exc
            descriptor = child
        assert descriptor is not None
        return descriptor
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise AuthorityIOError(
                    "cannot close authority-directory descriptor"
                ) from exc
        raise


def open_exact_regular(
    path: Path,
    *,
    maximum: int = DEFAULT_MAXIMUM_FILE_BYTES,
    expected_mode: int | None = None,
    require_single_link: bool = True,
    allow_empty: bool = True,
) -> tuple[int, bytes, os.stat_result]:
    """Open and read an exact absolute regular file without pathname races."""

    _require_absolute_normalized(path, label="file path")
    parent_descriptor: int | None = None
    rebound_parent: int | None = None
    descriptor: int | None = None
    succeeded = False
    try:
        parent_descriptor = _open_exact_directory(path.parent)
        parent_before = os.fstat(parent_descriptor)
        parent_mount_before = descriptor_mount_id(parent_descriptor)
        descriptor, payload, metadata = open_exact_regular_at(
            parent_descriptor,
            path.name,
            maximum=maximum,
            expected_mode=expected_mode,
            require_single_link=require_single_link,
            allow_empty=allow_empty,
        )
        rebound_parent = _open_exact_directory(path.parent)
        parent_after = os.fstat(rebound_parent)
        parent_mount_after = descriptor_mount_id(rebound_parent)
        rebound_path = os.stat(
            path.name,
            dir_fd=rebound_parent,
            follow_symlinks=False,
        )
        if (
            _stable_directory_identity(parent_before)
            != _stable_directory_identity(parent_after)
            or parent_mount_before != parent_mount_after
        ):
            raise AuthorityIOError(
                f"authority parent path changed while being measured: {path.parent}"
            )
        _require_same(metadata, rebound_path, label=str(path))
        succeeded = True
        return descriptor, payload, metadata
    except AuthorityIOError:
        raise
    except OSError as exc:
        raise AuthorityIOError(
            f"authority path is not an exact regular file: {path}"
        ) from exc
    finally:
        close_error: OSError | None = None
        if rebound_parent is not None:
            try:
                os.close(rebound_parent)
            except OSError as exc:
                close_error = exc
        if parent_descriptor is not None:
            try:
                os.close(parent_descriptor)
            except OSError as exc:
                if close_error is None:
                    close_error = exc
        if descriptor is not None and (not succeeded or close_error is not None):
            try:
                os.close(descriptor)
            except OSError as exc:
                if close_error is None:
                    close_error = exc
        if close_error is not None:
            raise AuthorityIOError(
                "cannot close authority path descriptor"
            ) from close_error


def read_exact_regular(
    path: Path,
    *,
    maximum: int = DEFAULT_MAXIMUM_FILE_BYTES,
    expected_mode: int | None = None,
    require_single_link: bool = True,
    allow_empty: bool = True,
) -> tuple[bytes, os.stat_result]:
    """Read an exact regular file without following its final symlink.

    The lexical path, opened descriptor, descriptor after the read, and lexical
    path after the read must all identify the same object.  Single-link files
    are the default because an unbound alias would provide a second mutation
    authority outside the declared path.
    """

    descriptor, payload, metadata = open_exact_regular(
        path,
        maximum=maximum,
        expected_mode=expected_mode,
        require_single_link=require_single_link,
        allow_empty=allow_empty,
    )
    try:
        os.close(descriptor)
    except OSError as exc:
        raise AuthorityIOError(
            f"cannot close authority file descriptor: {path}"
        ) from exc
    return payload, metadata


__all__ = [
    "AuthorityIOError",
    "DEFAULT_MAXIMUM_FILE_BYTES",
    "StatIdentity",
    "descriptor_mount_id",
    "open_exact_regular",
    "open_exact_regular_at",
    "read_exact_regular",
    "read_exact_regular_at",
    "stat_identity",
]
