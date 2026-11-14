# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Foreign temporary mounts must not become owned-root cleanup authority."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from crucible.tools import _isolated_mutation_host as host


def _authority(parent: Path):
    authority = host._PrivateRootAuthority(parent_path=parent)
    host._open_private_root_authority(authority)
    return authority


def test_unreadable_unrelated_mount_does_not_block_owned_root_cleanup(
    tmp_path, monkeypatch,
):
    authority = _authority(tmp_path)
    root = authority.root_path
    foreign = tmp_path / ".mount-unrelated"
    foreign.mkdir()
    sentinel = foreign / "unrelated.txt"
    sentinel.write_text("preserve unrelated mount contents\n")
    real_stat = os.stat

    def inaccessible_mount(path, *args, **kwargs):
        if path == foreign.name and kwargs.get("dir_fd") == authority.parent_descriptor:
            raise PermissionError(errno.EACCES, "unrelated mount denies lookup")
        return real_stat(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(host.os, "stat", inaccessible_mount)
            host._close_private_root_authority(authority)
        assert authority.root_removed
        assert authority.root_descriptor is None
        assert authority.parent_descriptor is None
        assert not root.exists()
        assert sentinel.read_text() == "preserve unrelated mount contents\n"
    finally:
        host._close_private_root_authority(authority)


@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
def test_possible_owned_entry_lookup_failure_preserves_exact_authority(
    tmp_path, monkeypatch, renamed, error_number,
):
    authority = _authority(tmp_path)
    owned = authority.root_path
    if renamed:
        moved = owned.with_name("renamed-owned-root")
        owned.rename(moved)
        owned = moved
    real_stat = os.stat

    def inaccessible_owned(path, *args, **kwargs):
        if path == owned.name and kwargs.get("dir_fd") == authority.parent_descriptor:
            raise OSError(error_number, "owned metadata is unavailable")
        return real_stat(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(host.os, "stat", inaccessible_owned)
            with pytest.raises(host.MutationHostLifecycleError):
                host._close_private_root_authority(authority)
        assert owned.exists()
        assert not authority.root_removed
        assert authority.root_descriptor is not None
        assert authority.parent_descriptor is not None
        host._close_private_root_authority(authority)
        assert authority.root_removed
        assert not owned.exists()
    finally:
        host._close_private_root_authority(authority)


def test_inaccessible_parent_inventory_preserves_owned_root(tmp_path, monkeypatch):
    authority = _authority(tmp_path)
    root = authority.root_path
    real_scandir = os.scandir

    def inaccessible_parent(path):
        if path == authority.parent_descriptor:
            raise PermissionError(errno.EACCES, "held parent denies inventory")
        return real_scandir(path)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(host.os, "scandir", inaccessible_parent)
            with pytest.raises(host.MutationHostLifecycleError, match="inventory"):
                host._close_private_root_authority(authority)
        assert root.exists()
        assert not authority.root_removed
    finally:
        host._close_private_root_authority(authority)


def test_interrupted_parent_inventory_preserves_owned_root(tmp_path, monkeypatch):
    authority = _authority(tmp_path)
    root = authority.root_path
    real_scandir = os.scandir

    class InterruptedInventory:
        def __init__(self, entries):
            self.entries = entries

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.entries.close()

        def __iter__(self):
            yield next(self.entries)
            raise OSError(errno.EIO, "parent inventory interrupted")

    def interrupted_parent(path):
        entries = real_scandir(path)
        if path == authority.parent_descriptor:
            return InterruptedInventory(entries)
        return entries

    try:
        with monkeypatch.context() as patch:
            patch.setattr(host.os, "scandir", interrupted_parent)
            with pytest.raises(host.MutationHostLifecycleError):
                host._close_private_root_authority(authority)
        assert root.exists()
        assert not authority.root_removed
        assert authority.root_descriptor is not None
        assert authority.parent_descriptor is not None
    finally:
        host._close_private_root_authority(authority)


def test_missing_inode_hint_requires_metadata_verification(tmp_path, monkeypatch):
    authority = _authority(tmp_path)
    root = authority.root_path
    foreign = tmp_path / ".mount-unknown-inode"
    foreign.mkdir()
    real_scandir = os.scandir
    real_stat = os.stat

    class UnknownInodeEntry:
        name = foreign.name

        def inode(self):
            return 0

    class UnknownInodeInventory:
        def __init__(self, entries):
            self.entries = entries

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.entries.close()

        def __iter__(self):
            for entry in self.entries:
                yield UnknownInodeEntry() if entry.name == foreign.name else entry

    def unknown_inode_parent(path):
        entries = real_scandir(path)
        if path == authority.parent_descriptor:
            return UnknownInodeInventory(entries)
        return entries

    def inaccessible_unknown_entry(path, *args, **kwargs):
        if path == foreign.name and kwargs.get("dir_fd") == authority.parent_descriptor:
            raise PermissionError(errno.EACCES, "unknown entry denies lookup")
        return real_stat(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(host.os, "scandir", unknown_inode_parent)
            patch.setattr(host.os, "stat", inaccessible_unknown_entry)
            with pytest.raises(host.MutationHostLifecycleError):
                host._close_private_root_authority(authority)
        assert root.exists()
        assert foreign.exists()
        assert not authority.root_removed
        assert authority.root_descriptor is not None
        assert authority.parent_descriptor is not None
    finally:
        host._close_private_root_authority(authority)
