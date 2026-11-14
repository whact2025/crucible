# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Race and topology failure tests for benchmark authority-file reads."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from crucible.common import authority_io as canonical_module
from crucible.common.authority_io import (
    AuthorityIOError,
    open_exact_regular_at,
    read_exact_regular,
    read_exact_regular_at,
)


def test_exact_regular_read_rejects_symlink_hardlink_directory_and_oversize(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"payload")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target)
    hardlink = tmp_path / "hardlink"
    os.link(target, hardlink)

    with pytest.raises(AuthorityIOError, match="regular file"):
        read_exact_regular(symlink)
    with pytest.raises(AuthorityIOError, match="hard link"):
        read_exact_regular(target)
    with pytest.raises(AuthorityIOError, match="regular file"):
        read_exact_regular(tmp_path)

    hardlink.unlink()
    with pytest.raises(AuthorityIOError, match="exceeds"):
        read_exact_regular(target, maximum=3)
    with pytest.raises(AuthorityIOError, match="mode"):
        read_exact_regular(target, expected_mode=0o700)

def test_exact_regular_read_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "record").write_bytes(b"payload")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(AuthorityIOError, match="regular file"):
        read_exact_regular(alias / "record")

def test_exact_regular_read_rejects_parent_path_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority-root"
    parent.mkdir()
    path = parent / "record"
    path.write_bytes(b"payload")
    displaced = tmp_path / "displaced-authority-root"
    real_open_directory = canonical_module._open_exact_directory
    opens = 0

    def racing_open_directory(candidate: Path) -> int:
        nonlocal opens
        if candidate == parent:
            opens += 1
            if opens == 2:
                parent.rename(displaced)
                parent.mkdir()
                (parent / path.name).write_bytes(b"payload")
        return real_open_directory(candidate)

    monkeypatch.setattr(
        canonical_module,
        "_open_exact_directory",
        racing_open_directory,
    )
    with pytest.raises(AuthorityIOError, match="parent path changed"):
        read_exact_regular(path)
    assert opens == 2

def test_exact_regular_read_detects_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "changing"
    path.write_bytes(b"initial")
    original_read = os.read
    mutated = False

    def changing_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if not mutated:
            mutated = True
            with path.open("ab") as stream:
                stream.write(b"-changed")
                stream.flush()
                os.fsync(stream.fileno())
        return chunk

    monkeypatch.setattr(os, "read", changing_read)
    with pytest.raises(AuthorityIOError, match="changed|exceeded"):
        read_exact_regular(path, maximum=64)

def test_exact_regular_read_rejects_fifo_swap_before_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "authority.json"
    path.write_bytes(b"payload")
    displaced = tmp_path / "displaced-authority.json"
    real_open = canonical_module.os.open
    swapped = False

    def racing_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if (
            not swapped
            and dir_fd is not None
            and os.fsdecode(candidate) == path.name
            and flags & os.O_ACCMODE == os.O_RDONLY
            and not flags & getattr(os, "O_DIRECTORY", 0)
        ):
            assert flags & os.O_NONBLOCK
            path.rename(displaced)
            os.mkfifo(path, mode=0o600)
            swapped = True
        return real_open(candidate, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(canonical_module.os, "open", racing_open)
    with pytest.raises(AuthorityIOError, match="changed"):
        read_exact_regular(path)
    assert swapped is True

def test_exact_regular_read_rejects_preexisting_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "authority.fifo"
    os.mkfifo(fifo, mode=0o600)

    with pytest.raises(AuthorityIOError, match="regular file"):
        read_exact_regular(fifo)

    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(AuthorityIOError, match="regular file"):
            read_exact_regular_at(parent, fifo.name)
    finally:
        os.close(parent)

def test_exact_regular_open_turns_rebound_close_uncertainty_into_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "authority.json"
    path.write_bytes(b"payload")
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_open = canonical_module.os.open
    real_close = canonical_module.os.close
    file_descriptors: list[int] = []
    injected = False

    def tracking_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(candidate, flags, mode, dir_fd=dir_fd)
        if (
            dir_fd == parent
            and os.fsdecode(candidate) == path.name
            and not flags & os.O_DIRECTORY
        ):
            file_descriptors.append(descriptor)
        return descriptor

    def uncertain_close(descriptor: int) -> None:
        nonlocal injected
        if (
            not injected
            and len(file_descriptors) == 2
            and descriptor == file_descriptors[1]
        ):
            injected = True
            real_close(descriptor)
            raise OSError("injected rebound close uncertainty")
        real_close(descriptor)

    monkeypatch.setattr(canonical_module.os, "open", tracking_open)
    monkeypatch.setattr(canonical_module.os, "close", uncertain_close)
    try:
        with pytest.raises(AuthorityIOError, match="close"):
            open_exact_regular_at(parent, path.name)
        assert injected is True
        assert len(file_descriptors) == 2
        with pytest.raises(OSError):
            os.fstat(file_descriptors[0])
    finally:
        real_close(parent)
