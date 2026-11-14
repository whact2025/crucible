# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Cross-process exclusive build lock.

Concurrent crucible pipelines targeting the **same** worktree need to
serialize their builds; pipelines on **different** worktrees should
proceed in parallel. A file-system lock rooted at the build
directory does exactly that.

Platform dispatch: ``fcntl.flock`` on POSIX and a best-effort
``msvcrt.locking`` byte-range lock on Windows. Falls back
to no-op if no primitive exists or the lock cannot be acquired (NFS /
exotic filesystems). The POSIX acquire runs in a background thread
because ``fcntl.flock`` blocks with no timeout and ``signal.alarm``
only works from the main thread; ``msvcrt`` has a non-blocking variant,
so the Windows acquire polls from the calling thread instead.

Leaf module: stdlib only.
"""

from __future__ import annotations

import errno
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

try:  # POSIX
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt
except ImportError:
    msvcrt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Longer than the 900s build timeout; gives a runaway build time to
# release before the next pipeline gives up and proceeds unlocked.
BUILD_LOCK_TIMEOUT = 1200

# Windows acquire polls LK_NBLCK at this interval. (msvcrt has no
# blocking-with-timeout mode; LK_LOCK's built-in retry is a fixed
# ~10s burst — far too coarse for short timeouts.)
_WINDOWS_POLL_INTERVAL_S = 1.0

# _locking() errnos that mean "held by someone else, keep waiting".
# Anything else (EBADF, EINVAL, ...) is a broken descriptor and must
# surface so the caller falls through to the unlocked path at once.
_WINDOWS_CONTENDED_ERRNOS = (errno.EACCES, errno.EDEADLK)


def _acquire(fd: IO[str], timeout: float) -> bool:
    """Block until *fd* holds the exclusive lock; ``False`` on timeout.

    Raises ``OSError`` when this platform has no locking primitive at
    all (the caller logs and proceeds unlocked).
    """
    if fcntl is not None:
        acquired = threading.Event()

        def _try_lock() -> None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                acquired.set()
            except (OSError, ValueError):
                pass

        t = threading.Thread(target=_try_lock, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return acquired.is_set()

    if msvcrt is not None:
        # Lock byte 0 (range locks may extend past EOF, so the empty
        # lock file is fine); release must cover the same range, hence
        # the seek(0) here and in _release.
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError as exc:
                if exc.errno not in _WINDOWS_CONTENDED_ERRNOS:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(_WINDOWS_POLL_INTERVAL_S, remaining))

    raise OSError("no file-locking primitive on this platform")


def _release(fd: IO[str]) -> None:
    """Release *fd*'s lock. Unheld: no-op on POSIX, OSError on Windows."""
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:
        fd.seek(0)
        msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def build_lock(build_root: Path, timeout: int = BUILD_LOCK_TIMEOUT) -> Iterator[None]:
    """Cross-process exclusive lock scoped to *build_root*.

    Yields once the lock is held (or the timeout elapses unlocked).
    """
    lock_path = build_root / ".crucible_build.lock"
    fd = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = lock_path.open("w")
        try:
            if not _acquire(fd, timeout):
                logger.warning(
                    "Build lock timeout (%ds) for %s, proceeding unlocked",
                    timeout, lock_path,
                )
        except (OSError, OverflowError, ValueError):
            logger.warning(
                "Cannot acquire build lock %s, proceeding unlocked", lock_path,
            )
        yield
    finally:
        if fd is not None:
            try:
                _release(fd)
            except OSError:
                pass
            fd.close()
            try:
                lock_path.unlink()
            except OSError:
                pass


__all__ = ["build_lock", "BUILD_LOCK_TIMEOUT"]
