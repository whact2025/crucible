# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Freeze the current Git-visible project files for independent plan drafts.

Draft tools share this read-only copy, never the live checkout or its Corpus.
The caller still validates source identity and every selected DAG against the
original project. Copies retain exact bytes and executable/read permissions;
write permissions are removed. Capture fails closed on aliases, races, and
the existing two-MiB source-file bound, with a 128-MiB/20,000-file total bound.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import selectors
import shutil
import stat
import subprocess
import time

from crucible.common.read_only_sources import (
    _DENIED_COMPONENTS,
    _read_file,
    SourceReferenceError,
)

from ._mutation_journal import (
    _git_argv,
    _reject_git_redirect_env,
    _sanitized_git_env,
    MutationJournalError,
)


_MAX_FILES = 20_000
_MAX_LIST_BYTES = 8 * 1024 * 1024
_MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
_DENIED = _DENIED_COMPONENTS | {".crucible", ".agents"}


class PlanSourceSnapshotError(ValueError):
    """No independent source snapshot can be safely admitted."""


def _git_paths(root: Path, *options: str) -> set[str]:
    process = subprocess.Popen(
        _git_argv("ls-files", *options, "-z", "--"),
        cwd=root,
        env=_sanitized_git_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    output = bytearray()
    deadline = time.monotonic() + 15
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not ready.select(remaining):
                    raise PlanSourceSnapshotError("source inventory timed out")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > _MAX_LIST_BYTES:
                    raise PlanSourceSnapshotError("source path inventory exceeds its byte limit")
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
    if process.returncode:
        raise PlanSourceSnapshotError("source snapshot requires a Git project")
    try:
        return {raw.decode("utf-8") for raw in output.split(b"\0") if raw}
    except UnicodeDecodeError as exc:
        raise PlanSourceSnapshotError("source paths must be UTF-8") from exc


def _source_paths(root: Path, excluded: tuple[PurePosixPath, ...]) -> tuple[str, ...]:
    paths = _git_paths(root, "--cached", "--others", "--exclude-standard")
    # A tracked deletion is current project state, not a file to resurrect.
    paths.difference_update(_git_paths(root, "--deleted"))
    selected = []
    for relative in sorted(paths):
        path = PurePosixPath(relative)
        if (
            path.is_absolute() or path.as_posix() != relative
            or ".." in path.parts or "\\" in relative
            or any(ord(char) < 32 or ord(char) == 127 for char in relative)
        ):
            raise PlanSourceSnapshotError("source inventory contains an unsafe path")
        if any(
            part.casefold() in _DENIED or part.casefold().startswith(".env.")
            for part in path.parts
        ) or any(path == deny or deny in path.parents for deny in excluded):
            continue
        selected.append(relative)
    if len(selected) > _MAX_FILES:
        raise PlanSourceSnapshotError("source snapshot exceeds its file limit")
    return tuple(selected)


def _identity(path: Path) -> tuple[int, int, int, int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PlanSourceSnapshotError("snapshot source must be a regular unaliased file")
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def _remove_partial(destination: Path) -> None:
    # This directory was created exclusively by this capture, never reused.
    for current, _dirs, _files in os.walk(destination, topdown=True, followlinks=False):
        Path(current).chmod(0o700)
    shutil.rmtree(destination)


def create_plan_source_snapshot(
    source_root: Path,
    destination: Path,
    *,
    excluded_roots: tuple[Path, ...],
) -> Path:
    """Copy only this project's tracked and nonignored untracked source.

    ``destination`` must not exist and its parent must already exist. Excluded
    Corpus/run/control/debug roots remain absent even if tracked. Existing
    source-reference policy excludes credentials and ambient instructions;
    ordinary README and design documentation remain visible. The caller owns
    the returned directory and must keep it outside candidate mutation scope.
    """
    root = Path(source_root).resolve(strict=True)
    if not root.is_dir():
        raise PlanSourceSnapshotError("source root is not a directory")
    output = Path(destination).absolute()
    if output.exists() or output.is_symlink():
        raise PlanSourceSnapshotError("snapshot destination already exists")
    output = output.parent.resolve(strict=True) / output.name
    denied = []
    for candidate in (*excluded_roots, output):
        absolute = Path(candidate).resolve()
        if absolute == root or absolute in root.parents:
            raise PlanSourceSnapshotError("excluded root covers the source project")
        if root in absolute.parents:
            denied.append(PurePosixPath(absolute.relative_to(root).as_posix()))
    exclusions = tuple(denied)
    try:
        _reject_git_redirect_env()
        paths = _source_paths(root, exclusions)
    except (MutationJournalError, OSError, subprocess.SubprocessError) as exc:
        raise PlanSourceSnapshotError(f"source inventory is unavailable: {exc}") from exc

    output.mkdir(mode=0o700)
    try:
        copied = {}
        total_bytes = 0
        for relative in paths:
            original = root / relative
            before = _identity(original)
            blob = _read_file(root, relative, None)
            if _identity(original) != before:
                raise PlanSourceSnapshotError("source changed during snapshot capture")
            total_bytes += len(blob)
            if total_bytes > _MAX_SNAPSHOT_BYTES:
                raise PlanSourceSnapshotError("source snapshot exceeds its total byte limit")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(blob)
            target.chmod(stat.S_IMODE(before[2]) & ~0o222)
            copied[relative] = before
        if _source_paths(root, exclusions) != paths:
            raise PlanSourceSnapshotError("source inventory changed during snapshot capture")
        for relative, before in copied.items():
            if (
                _identity(root / relative) != before
                or _read_file(root, relative, None) != (output / relative).read_bytes()
            ):
                raise PlanSourceSnapshotError("source changed during snapshot capture")
        for current, _dirs, _files in os.walk(output, topdown=False):
            Path(current).chmod(0o555)
    except (OSError, SourceReferenceError, subprocess.SubprocessError) as exc:
        _remove_partial(output)
        raise PlanSourceSnapshotError(f"source snapshot capture failed: {exc}") from exc
    except BaseException:
        _remove_partial(output)
        raise
    return output
