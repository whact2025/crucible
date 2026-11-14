# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.tools._buildlock`` — the portable build lock.

Pins the 2026-06-10 audit fix: the module previously did a top-level
``import fcntl`` (POSIX-only), so on Windows the entire
``tools.build`` → ``workflows.{tier,architect}`` import chain failed
at *collection* time. The lock now dispatches per platform —
``fcntl.flock`` on POSIX (contract unchanged), best-effort
``msvcrt.locking`` on Windows, logged no-op fallback when neither
primitive exists.

Every test here is platform-portable and exercises the real OS
primitive of the host (byte-range locks and flock both conflict
across handles within one process, so contention needs no
subprocess).
"""

from __future__ import annotations

import logging
import time

from crucible.tools import _buildlock
from crucible.tools._buildlock import build_lock

_LOGGER = _buildlock.__name__


# ----------------------------------------------------------------------
# Import-chain regression + platform coverage
# ----------------------------------------------------------------------


def test_tools_build_chain_imports_without_fcntl_at_top_level():
    """The audit regression: ``tools.build`` (→ handlers → _buildlock)
    must import on every platform, not just POSIX."""
    import crucible.tools.build  # noqa: F401  (import is the assertion)


def test_host_platform_has_a_locking_primitive():
    """CI/dev hosts are POSIX or Windows; one primitive must resolve.
    (The no-primitive fallback is covered separately via monkeypatch.)"""
    assert _buildlock.fcntl is not None or _buildlock.msvcrt is not None


# ----------------------------------------------------------------------
# Uncontended paths
# ----------------------------------------------------------------------


def test_acquire_release_roundtrip(tmp_path):
    lock_path = tmp_path / ".crucible_build.lock"
    with build_lock(tmp_path):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_sequential_reacquire(tmp_path):
    for _ in range(2):
        with build_lock(tmp_path):
            pass


def test_creates_missing_build_root(tmp_path):
    root = tmp_path / "deep" / "nested"
    with build_lock(root):
        assert (root / ".crucible_build.lock").exists()


# ----------------------------------------------------------------------
# Contention — the lock must actually lock, then fall through unlocked
# ----------------------------------------------------------------------


def test_contended_lock_blocks_then_proceeds_unlocked(tmp_path, caplog):
    """A second pipeline on the same build root must (a) really block
    on the held lock — mutual exclusion works on this platform — and
    (b) yield unlocked after the timeout rather than deadlock."""
    holder = (tmp_path / ".crucible_build.lock").open("w")
    try:
        assert _buildlock._acquire(holder, timeout=10)
        start = time.monotonic()
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            with build_lock(tmp_path, timeout=2):
                waited = time.monotonic() - start
        assert waited >= 1.5  # blocked for ~the full timeout
        assert "proceeding unlocked" in caplog.text
    finally:
        holder.close()  # closing the handle releases any held lock


def test_lock_file_reuse_after_contention(tmp_path, caplog):
    """On Windows the lock file survives a contended run (unlink fails
    while the holder keeps it open); a later pipeline must reuse it and
    acquire cleanly once the holder releases."""
    holder = (tmp_path / ".crucible_build.lock").open("w")
    try:
        assert _buildlock._acquire(holder, timeout=10)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            with build_lock(tmp_path, timeout=1):
                pass
        assert "proceeding unlocked" in caplog.text

        caplog.clear()
        _buildlock._release(holder)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            with build_lock(tmp_path, timeout=30):
                pass
        assert "proceeding unlocked" not in caplog.text
    finally:
        holder.close()


# ----------------------------------------------------------------------
# No-primitive fallback (NFS / exotic platforms)
# ----------------------------------------------------------------------


def test_no_primitive_falls_back_to_noop(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(_buildlock, "fcntl", None)
    monkeypatch.setattr(_buildlock, "msvcrt", None)
    entered = False
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        with build_lock(tmp_path):
            entered = True
    assert entered
    assert "Cannot acquire build lock" in caplog.text
