# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Small locked, crash-atomic byte-record persistence substrate.

Conversation continuity and workflow control state use the same primitive:
one SHA-256-keyed regular file, one per-key ``flock``, a fixed staged image,
``fsync`` before and after ``os.replace``, and fail-closed recovery that
discards an uncommitted staged image after process death.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import stat
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path


class DurableStateError(RuntimeError):
    """A durable byte record is unavailable, unsafe, or cannot be published."""


class AtomicStateStore:
    """Per-key atomic byte records under one real directory."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_record_bytes: int = 1024 * 1024,
    ) -> None:
        if type(max_record_bytes) is not int or max_record_bytes < 1:
            raise ValueError("max_record_bytes must be an exact integer >= 1")
        self.root = Path(root).absolute()
        self.max_record_bytes = max_record_bytes
        self._root_descriptor = -1
        self._closed = False
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            root_stat = self.root.lstat()
        except OSError as exc:
            raise DurableStateError(
                f"durable state root is unavailable: {exc}"
            ) from exc
        if not stat.S_ISDIR(root_stat.st_mode) or self.root.is_symlink():
            raise DurableStateError(
                "durable state root must be a real directory"
            )
        self._root_descriptor = self._open_root_without_symlinks()
        try:
            bound = os.fstat(self._root_descriptor)
            self.root_identity = (
                bound.st_dev,
                bound.st_ino,
                bound.st_uid,
                stat.S_IMODE(bound.st_mode),
            )
            self._verify_root_binding()
        except BaseException:
            os.close(self._root_descriptor)
            self._root_descriptor = -1
            self._closed = True
            raise
        self._finalizer = weakref.finalize(
            self,
            os.close,
            self._root_descriptor,
        )

    @property
    def closed(self) -> bool:
        """Whether this store has released its retained root descriptor."""

        return self._closed

    def close(self) -> None:
        """Release the retained root descriptor exactly once."""

        if self._closed:
            return
        self._closed = True
        self._root_descriptor = -1
        try:
            self._finalizer()
        except OSError as exc:
            raise DurableStateError(
                f"cannot close durable state root descriptor: {exc}"
            ) from exc

    def __enter__(self) -> AtomicStateStore:
        self._verify_root_binding()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _open_root_without_symlinks(self) -> int:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(os.sep, flags)
        except OSError as exc:
            raise DurableStateError(
                f"cannot open filesystem root for durable state: {exc}"
            ) from exc
        try:
            for component in self.root.parts[1:]:
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except OSError as exc:
            os.close(descriptor)
            raise DurableStateError(
                "durable state root has an unavailable or symlinked path component"
            ) from exc

    @property
    def root_descriptor(self) -> int:
        """Borrow the held root descriptor; callers must not close it."""

        self._verify_root_binding()
        return self._root_descriptor

    def _verify_root_binding(self) -> None:
        if self._closed or self._root_descriptor < 0:
            raise DurableStateError("durable state store is closed")
        try:
            held = os.fstat(self._root_descriptor)
            observed_descriptor = self._open_root_without_symlinks()
        except (OSError, DurableStateError) as exc:
            raise DurableStateError(
                "durable state root pathname is unavailable or replaced"
            ) from exc
        try:
            observed = os.fstat(observed_descriptor)
        finally:
            os.close(observed_descriptor)
        held_identity = (
            held.st_dev,
            held.st_ino,
            held.st_uid,
            stat.S_IMODE(held.st_mode),
        )
        observed_identity = (
            observed.st_dev,
            observed.st_ino,
            observed.st_uid,
            stat.S_IMODE(observed.st_mode),
        )
        if held_identity != self.root_identity or observed_identity != self.root_identity:
            raise DurableStateError(
                "durable state root pathname or inode identity changed"
            )

    @staticmethod
    def _key_suffix(key_fingerprint: str) -> str:
        if (
            not isinstance(key_fingerprint, str)
            or len(key_fingerprint) != 71
            or not key_fingerprint.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in key_fingerprint[7:]
            )
        ):
            raise ValueError(
                "state key must be 'sha256:' followed by 64 lowercase "
                "hexadecimal digits"
            )
        return key_fingerprint[7:]

    def paths(
        self,
        key_fingerprint: str,
    ) -> tuple[Path, Path, Path]:
        suffix = self._key_suffix(key_fingerprint)
        return (
            self.root / f"{suffix}.json",
            self.root / f".{suffix}.lock",
            self.root / f".{suffix}.json.staged",
        )

    def _reject_unsafe_existing(self, name: str, *, kind: str) -> None:
        try:
            path_stat = os.stat(
                name,
                dir_fd=self._root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DurableStateError(
                f"cannot inspect durable {kind}: {exc}"
            ) from exc
        if not stat.S_ISREG(path_stat.st_mode):
            raise DurableStateError(
                f"durable {kind} must be a regular file: {self.root / name}"
            )

    def _fsync_root(self) -> None:
        self._verify_root_binding()
        os.fsync(self._root_descriptor)

    @contextlib.contextmanager
    def _locked(
        self,
        key_fingerprint: str,
    ) -> Iterator[tuple[str, str]]:
        record_path, lock_path, staged_path = self.paths(key_fingerprint)
        self._verify_root_binding()
        record_name = record_path.name
        lock_name = lock_path.name
        staged_name = staged_path.name
        self._reject_unsafe_existing(lock_name, kind="lock")
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                lock_name,
                flags,
                0o600,
                dir_fd=self._root_descriptor,
            )
        except OSError as exc:
            raise DurableStateError(
                f"cannot open durable state lock: {exc}"
            ) from exc
        try:
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise DurableStateError("durable state lock must be a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._verify_root_binding()
            self._recover_staged(staged_name)
            yield record_name, staged_name
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _recover_staged(self, staged_name: str) -> None:
        try:
            staged_stat = os.stat(
                staged_name,
                dir_fd=self._root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DurableStateError(
                f"cannot inspect staged durable record: {exc}"
            ) from exc
        if not stat.S_ISREG(staged_stat.st_mode):
            raise DurableStateError(
                "staged durable record must be a regular file"
            )
        try:
            os.unlink(staged_name, dir_fd=self._root_descriptor)
            self._fsync_root()
        except OSError as exc:
            raise DurableStateError(
                f"cannot recover staged durable record: {exc}"
            ) from exc

    def _read_unlocked(self, record_name: str) -> bytes | None:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                record_name,
                flags,
                dir_fd=self._root_descriptor,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise DurableStateError(
                f"cannot open durable state record: {exc}"
            ) from exc
        try:
            record_stat = os.fstat(descriptor)
            if not stat.S_ISREG(record_stat.st_mode):
                raise DurableStateError(
                    "durable record must be a regular file"
                )
            chunks: list[bytes] = []
            remaining = self.max_record_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
        except OSError as exc:
            raise DurableStateError(
                f"cannot read durable state record: {exc}"
            ) from exc
        finally:
            os.close(descriptor)
        if len(payload) > self.max_record_bytes:
            raise DurableStateError(
                f"durable state record exceeds {self.max_record_bytes} bytes"
            )
        return payload

    def _write_unlocked(
        self,
        staged_name: str,
        record_name: str,
        payload: bytes,
    ) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("durable state payload must be bytes")
        if len(payload) > self.max_record_bytes:
            raise DurableStateError(
                f"durable state record exceeds {self.max_record_bytes} bytes"
            )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor: int | None = None
        published = False
        try:
            descriptor = os.open(
                staged_name,
                flags | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=self._root_descriptor,
            )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write to staged durable record")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(
                staged_name,
                record_name,
                src_dir_fd=self._root_descriptor,
                dst_dir_fd=self._root_descriptor,
            )
            published = True
            self._fsync_root()
        except OSError as exc:
            raise DurableStateError(
                f"cannot publish durable state record atomically: {exc}"
            ) from exc
        finally:
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            if not published:
                with contextlib.suppress(OSError):
                    os.unlink(staged_name, dir_fd=self._root_descriptor)

    def load(
        self,
        key_fingerprint: str,
        *,
        validator: Callable[[bytes | None], None] | None = None,
    ) -> bytes | None:
        if validator is not None and not callable(validator):
            raise TypeError("durable state validator must be callable")
        with self._locked(key_fingerprint) as (
            record_path,
            _staged_path,
        ):
            payload = self._read_unlocked(record_path)
            if validator is not None:
                validator(payload)
            return payload

    def update(
        self,
        key_fingerprint: str,
        transform: Callable[[bytes | None], bytes],
        *,
        after_publish: Callable[[bytes], None] | None = None,
    ) -> bytes:
        if not callable(transform):
            raise TypeError("durable state transform must be callable")
        if after_publish is not None and not callable(after_publish):
            raise TypeError("durable state after_publish must be callable")
        with self._locked(key_fingerprint) as (record_path, staged_path):
            current = self._read_unlocked(record_path)
            updated = transform(current)
            if not isinstance(updated, bytes):
                raise TypeError("durable state transform must return bytes")
            if updated == current:
                if after_publish is not None:
                    after_publish(updated)
                return updated
            self._write_unlocked(staged_path, record_path, updated)
            if after_publish is not None:
                after_publish(updated)
            return updated

    @contextlib.contextmanager
    def nonblocking_lease(self, name: str) -> Iterator[None]:
        """Hold one root-bound process lease until the caller exits."""

        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or os.sep in name
        ):
            raise ValueError("durable lease name must be one filename")
        self._verify_root_binding()
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                name,
                flags,
                0o600,
                dir_fd=self._root_descriptor,
            )
        except OSError as exc:
            raise DurableStateError("cannot open durable execution lease") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise DurableStateError("durable execution lease is not regular")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise DurableStateError(
                        "durable execution lease is already held"
                    ) from exc
                raise
            self._verify_root_binding()
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


__all__ = ["AtomicStateStore", "DurableStateError"]
