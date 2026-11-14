# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Per-slice rollback via ``git apply -R`` on a captured ``diff_ref``
plus cleanup of any ``scope_escapes`` recorded against the slice.

When a slice halts with a ``hard_fail_*`` classification (or a
transport-failure mid-stream), the implementer has typically written
partial / broken state to the shared checkout. Without rollback that
state persists into the next workflow run, contaminating both
postmortem (the diff captured for the failed slice and the dirty tree
don't agree) and any retry attempt — and in fan-out mode it can
contaminate subsequent siblings in the same tier.

This module exposes ``rollback_slice(corpus, impl, cwd)`` which
resolves ``impl.diff_ref`` via the Corpus, runs ``git apply -R`` in
``cwd``, additionally reverts/deletes any ``scope_escapes`` files
recorded on the Implementation, and returns ``True`` on success.

Path-scoped revert against the shared checkout. The rollback is
per-slice; prior successful slices in the same tier are kept on
disk so postmortem can see what landed.

**Scope-escape semantics.** Files the implementer wrote outside its
declared ``files_in_scope ∪ creates_paths`` are captured in
``Implementation.scope_escapes`` (see ``assay.primitives.implement``).
Rollback cleans those escapes before the slice cascade can continue:
tracked files are reverted to HEAD in the worktree and untracked files are
deleted. The cross-run reaper handles state retained after process crashes.

**Preconditions:**

- The caller must settle mutating children and hold an exclusive write
  window for ``cwd`` before invoking ``rollback_slice``. Tier stages shared
  checkout mutations serially. Parallel writer lanes require isolated
  checkout roots and joined child work before integration or rollback.
"""

from __future__ import annotations

import hashlib
import logging
import os
import stat
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

from assay.corpus.store import Corpus
from assay.types.artifacts import Implementation

from ._mutation_journal import (
    MutationJournalError,
    _git_argv,
    _reject_git_redirect_env,
    _sanitized_git_env,
    normalize_workspace_path,
)

logger = logging.getLogger(__name__)
_UNBOUND_ENTRY = object()
_WORKTREE_TRANSFORM_ATTRIBUTES = frozenset({
    b"crlf",
    b"eol",
    b"filter",
    b"ident",
    b"text",
    b"working-tree-encoding",
})
_ATTRIBUTE_SOURCE_MAX_BYTES = 1024 * 1024
_ATTRIBUTE_INSPECTION_TIMEOUT_S = 5.0


def _normalize_rollback_path(cwd: Path, rendered: object) -> str:
    """Normalize one artifact path and typed-fail on process-hostile values."""

    if not isinstance(rendered, str):
        raise MutationJournalError(
            f"rollback workspace path must be a string: {rendered!r}"
        )
    if "\0" in rendered:
        raise MutationJournalError(
            f"rollback workspace path contains an embedded NUL: {rendered!r}"
        )
    try:
        return normalize_workspace_path(cwd, rendered)
    except MutationJournalError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot normalize rollback workspace path {rendered!r}: {exc}"
        ) from exc


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
    )


def _open_checkout_root(cwd: str | Path) -> tuple[Path, int]:
    """Open and identity-bind the checkout before any classification."""

    _reject_git_redirect_env()
    missing_flags = [
        name for name in ("O_DIRECTORY", "O_NOFOLLOW")
        if not hasattr(os, name)
    ]
    required_dir_fd = (
        os.open,
        os.stat,
        os.unlink,
        os.rmdir,
        os.readlink,
        os.symlink,
    )
    unsupported = [
        function.__name__
        for function in required_dir_fd
        if function not in os.supports_dir_fd
    ]
    if missing_flags or unsupported:
        raise MutationJournalError(
            "platform cannot safely bind rollback to a checkout without "
            "following aliases: "
            f"missing_flags={missing_flags!r}, missing_dir_fd={unsupported!r}"
        )
    try:
        cwd_path = Path(cwd).resolve(strict=True)
        before = cwd_path.lstat()
    except (OSError, RuntimeError) as exc:
        raise MutationJournalError(
            f"cannot resolve rollback checkout {cwd!r}: {exc}"
        ) from exc
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise MutationJournalError(
            f"rollback checkout is not a real directory: {cwd_path}"
        )
    try:
        descriptor = os.open(cwd_path, _directory_open_flags())
    except OSError as exc:
        raise MutationJournalError(
            f"cannot safely open rollback checkout {cwd_path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise MutationJournalError(
            f"cannot inspect held rollback checkout {cwd_path}: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
    ):
        os.close(descriptor)
        raise MutationJournalError(
            f"rollback checkout changed identity while opening: {cwd_path}"
        )
    try:
        _reject_config_includes(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    return cwd_path, descriptor


def _assert_checkout_root_binding(cwd: Path, root_fd: int) -> None:
    """Require the current checkout pathname to name the held root inode."""

    try:
        current = cwd.lstat()
        held = os.fstat(root_fd)
    except OSError as exc:
        raise MutationJournalError(
            f"cannot revalidate held rollback checkout {cwd}: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(held.st_mode)
        or current.st_dev != held.st_dev
        or current.st_ino != held.st_ino
    ):
        raise MutationJournalError(
            f"rollback checkout {cwd} changed identity during verification"
        )


def _assert_held_checkout_root(root_fd: int) -> None:
    """Require the admitted checkout descriptor to remain a directory.

    Once rollback has opened and authenticated the checkout, a concurrent
    rename of its public pathname must not redirect cleanup into a replacement
    tree.  Later operations are descriptor-relative, so their continuing
    authority is the held inode itself rather than a pathname rebind.
    """

    try:
        held = os.fstat(root_fd)
    except OSError as exc:
        raise MutationJournalError(
            "cannot inspect held rollback checkout descriptor"
        ) from exc
    if not stat.S_ISDIR(held.st_mode):
        raise MutationJournalError(
            "held rollback checkout is no longer a directory"
        )


def _descriptor_git_cwd(root_fd: int) -> str:
    """Return a subprocess cwd alias bound to the already-open checkout."""

    try:
        expected = os.fstat(root_fd)
    except OSError as exc:
        raise MutationJournalError(
            f"cannot inspect held rollback checkout descriptor: {exc}"
        ) from exc
    for root in ("/proc/self/fd", "/dev/fd"):
        candidate = f"{root}/{root_fd}"
        try:
            observed = os.stat(candidate)
        except OSError:
            continue
        if observed.st_dev == expected.st_dev and observed.st_ino == expected.st_ino:
            return candidate
    raise MutationJournalError(
        "platform cannot anchor Git rollback commands to the held checkout "
        "descriptor"
    )


def _run_bound_git(
    root_fd: int,
    *args: str,
    input_bytes: bytes | None = None,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run one sanitized Git command from the descriptor-bound checkout."""

    try:
        return subprocess.run(
            _git_argv(*args),
            cwd=_descriptor_git_cwd(root_fd),
            pass_fds=(root_fd,),
            env=_sanitized_git_env(
                GIT_NO_LAZY_FETCH="1",
                GIT_NO_REPLACE_OBJECTS="1",
            ),
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise MutationJournalError(f"cannot run descriptor-bound Git: {exc}") from exc


def _run_bound_git_with_ambient_attributes(
    root_fd: int,
    *args: str,
    input_bytes: bytes | None = None,
    timeout_s: float | None = _ATTRIBUTE_INSPECTION_TIMEOUT_S,
) -> subprocess.CompletedProcess[bytes]:
    """Run ``check-attr`` with operator global/system attributes enabled.

    This runner is deliberately private to read-only attribute inspection.
    It scrubs ambient ``GIT_*`` redirects, disables hooks/transports, and does
    not execute clean/smudge/process drivers. Unlike the normal Git runner it
    does not replace the global/system attribute sources with ``/dev/null``.
    """

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    env.update({
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
    })
    argv = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "protocol.ext.allow=never",
        *args,
    ]
    try:
        return subprocess.run(
            argv,
            cwd=_descriptor_git_cwd(root_fd),
            pass_fds=(root_fd,),
            env=env,
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot inspect ambient Git attributes: {exc}"
        ) from exc


def _reject_config_includes(root_fd: int) -> None:
    """Reject config includes before any Git command may follow them.

    Even ``git config --no-includes`` can block while Git performs its early
    repository-config load.  Inspect each primary config file directly via
    no-follow bounded descriptors instead, including linked-worktree common
    and per-worktree configs plus operator global/system sources.
    """

    configs: list[tuple[str, Path, int | None, bool]] = []
    try:
        dotgit = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot inspect checkout Git control path: {exc}"
        ) from exc
    if stat.S_ISDIR(dotgit.st_mode):
        configs.extend((
            ("repository config", Path(".git/config"), root_fd, True),
            (
                "worktree config",
                Path(".git/config.worktree"),
                root_fd,
                False,
            ),
        ))
    elif stat.S_ISREG(dotgit.st_mode):
        control = _bounded_attribute_source_payload(
            Path(".git"),
            source="Git control file",
            anchor_fd=root_fd,
        )
        if control is None or b"\0" in control:
            raise MutationJournalError("checkout .git control file is malformed")
        lines = control.splitlines()
        if len(lines) != 1 or not lines[0].lower().startswith(b"gitdir: "):
            raise MutationJournalError("checkout .git control file is malformed")
        rendered_gitdir = os.fsdecode(lines[0][8:])
        gitdir = Path(os.path.normpath(rendered_gitdir))
        if not gitdir.is_absolute():
            raise MutationJournalError(
                "relative linked-worktree gitdir is unsupported during recovery"
            )
        commondir_payload = _bounded_attribute_source_payload(
            gitdir / "commondir",
            source="Git commondir control file",
        )
        if commondir_payload is None:
            common_dir = gitdir
        else:
            if b"\0" in commondir_payload:
                raise MutationJournalError("Git commondir file is malformed")
            commondir_lines = commondir_payload.splitlines()
            if len(commondir_lines) != 1 or not commondir_lines[0]:
                raise MutationJournalError("Git commondir file is malformed")
            rendered_common = Path(os.fsdecode(commondir_lines[0]))
            common_dir = Path(os.path.normpath(
                os.fspath(
                    rendered_common
                    if rendered_common.is_absolute()
                    else gitdir / rendered_common
                )
            ))
            if not common_dir.is_absolute():
                raise MutationJournalError("Git commondir is not absolute")
        configs.extend((
            ("repository config", common_dir / "config", None, True),
            (
                "worktree config",
                gitdir / "config.worktree",
                None,
                False,
            ),
        ))
    else:
        raise MutationJournalError(
            "checkout .git control path is not a regular file or directory"
        )

    configs.append(("system config", Path("/etc/gitconfig"), None, False))
    home = os.environ.get("HOME")
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if home:
        configs.append(("global config", Path(home) / ".gitconfig", None, False))
        configs.append((
            "global config",
            Path(home) / ".config" / "git" / "config",
            None,
            False,
        ))
    if xdg:
        configs.append((
            "global config",
            Path(xdg) / "git" / "config",
            None,
            False,
        ))

    seen: set[tuple[str, int | None]] = set()
    for source, candidate, anchor_fd, required in configs:
        identity = (os.fspath(candidate), anchor_fd)
        if identity in seen:
            continue
        seen.add(identity)
        payload = _bounded_attribute_source_payload(
            candidate,
            source=source,
            anchor_fd=anchor_fd,
        )
        if payload is None:
            if required:
                raise MutationJournalError(
                    f"required {source} is missing: {candidate}"
                )
            continue
        if any(
            line.lstrip().lower().startswith(b"[include")
            for line in payload.splitlines()
        ):
            raise MutationJournalError(
                "Git config include/includeIf directives are unsupported "
                "during descriptor-bound recovery"
            )


def _head_commit_oid(root_fd: int) -> str:
    result = _run_bound_git(root_fd, "rev-parse", "--verify", "HEAD^{commit}")
    if result.returncode != 0:
        raise MutationJournalError(
            "cannot resolve immutable HEAD for scope cleanup: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    oid = result.stdout.decode("ascii", errors="strict").strip()
    _new_git_object_hash(oid)
    return oid


def _transform_attribute_state(
    root_fd: int,
    path: str,
    *,
    selector: tuple[str, ...] = (),
    ambient: bool = False,
) -> tuple[tuple[str, str], ...]:
    """Return exact effective transform attributes without running drivers."""

    runner = (
        _run_bound_git_with_ambient_attributes
        if ambient
        else _run_bound_git
    )
    result = runner(
        root_fd,
        "check-attr",
        "-z",
        "--all",
        *selector,
        "--stdin",
        input_bytes=os.fsencode(path) + b"\0",
        timeout_s=_ATTRIBUTE_INSPECTION_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise MutationJournalError(
            f"cannot inspect transforming attributes for {path!r}: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    fields = result.stdout.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 3:
        raise MutationJournalError(
            f"malformed git check-attr output for {path!r}"
        )
    state: dict[bytes, bytes] = {}
    encoded_path = os.fsencode(path)
    for offset in range(0, len(fields), 3):
        observed_path, attribute, value = fields[offset:offset + 3]
        normalized = attribute.lower()
        if observed_path != encoded_path:
            raise MutationJournalError(
                f"git check-attr returned an unexpected path for {path!r}"
            )
        if normalized not in _WORKTREE_TRANSFORM_ATTRIBUTES:
            continue
        if normalized in state:
            raise MutationJournalError(
                f"git check-attr repeated {os.fsdecode(attribute)!r} for "
                f"{path!r}"
            )
        state[normalized] = value
    return tuple(sorted(
        (os.fsdecode(attribute), os.fsdecode(value))
        for attribute, value in state.items()
    ))


def _bound_config_values(
    root_fd: int,
    key: str,
    *,
    ambient: bool,
) -> tuple[str, ...]:
    runner = (
        _run_bound_git_with_ambient_attributes
        if ambient
        else _run_bound_git
    )
    if ambient:
        # ``--get-all`` is authority-relevant here: a lower-precedence
        # global/system transform may have materialized the checkout before
        # failed work added a masking local value.  The effective value alone
        # would lose that provenance and make a raw blob restore unsafe.
        result = runner(
            root_fd,
            "config",
            "--includes",
            "--show-scope",
            "--null",
            "--get-all",
            key,
            timeout_s=_ATTRIBUTE_INSPECTION_TIMEOUT_S,
        )
    else:
        result = runner(
            root_fd,
            "config",
            "--local",
            "--no-includes",
            "--null",
            "--get-all",
            key,
            timeout_s=_ATTRIBUTE_INSPECTION_TIMEOUT_S,
        )
    if result.returncode == 1:
        return ()
    if result.returncode != 0:
        raise MutationJournalError(
            f"cannot inspect Git config {key}: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    fields = result.stdout.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if ambient:
        if len(fields) % 2:
            raise MutationJournalError(
                f"Git returned malformed scoped config for {key}"
            )
        values: list[str] = []
        valid_scopes = {b"system", b"global", b"local", b"worktree", b"command"}
        for offset in range(0, len(fields), 2):
            scope, value = fields[offset:offset + 2]
            if scope not in valid_scopes:
                raise MutationJournalError(
                    f"Git returned unknown config scope for {key}: "
                    f"{os.fsdecode(scope)!r}"
                )
            values.append(value.decode("utf-8", errors="surrogateescape"))
        return tuple(values)
    return tuple(
        value.decode("utf-8", errors="surrogateescape")
        for value in fields
    )


def _transform_config_state(
    root_fd: int,
    *,
    ambient: bool,
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for key in ("core.autocrlf", "core.eol", "core.attributesFile"):
        for value in _bound_config_values(root_fd, key, ambient=ambient):
            result.append((key.lower(), value))
    return tuple(result)


def _bounded_attribute_source_payload(
    candidate: Path,
    *,
    source: str,
    anchor_fd: int | None = None,
) -> bytes | None:
    """Read one external attribute source without aliases or blocking files."""

    if anchor_fd is None:
        if not candidate.is_absolute():
            raise MutationJournalError(
                f"{source} source must be absolute: {candidate}"
            )
        relative = candidate.as_posix().lstrip("/")
    else:
        if candidate.is_absolute():
            raise MutationJournalError(
                f"bound {source} source must be relative: {candidate}"
            )
        relative = candidate.as_posix()
    if not relative:
        raise MutationJournalError(f"invalid {source} source: {candidate}")
    try:
        root_fd = (
            os.open(Path("/"), _directory_open_flags())
            if anchor_fd is None
            else os.dup(anchor_fd)
        )
    except OSError as exc:
        raise MutationJournalError(
            f"cannot bind filesystem root for {source} source {candidate}: {exc}"
        ) from exc
    try:
        with _open_workspace_parent_if_present(
            root_fd,
            relative,
        ) as parent_binding:
            if parent_binding is None:
                # Rewalk once before accepting absence.  A source that appears
                # during authority capture is ambiguity, never an empty file.
                with _open_workspace_parent_if_present(
                    root_fd,
                    relative,
                ) as current_binding:
                    if current_binding is not None:
                        raise MutationJournalError(
                            f"parent of {source} source {candidate} appeared "
                            "during inspection"
                        )
                return None
            parent_fd, name = parent_binding
            _assert_workspace_parent_binding(root_fd, relative, parent_fd)
            try:
                before = os.stat(
                    name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                try:
                    os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    _assert_workspace_parent_binding(
                        root_fd,
                        relative,
                        parent_fd,
                    )
                    return None
                raise MutationJournalError(
                    f"{source} source {candidate} appeared during inspection"
                ) from None
            except (OSError, TypeError, ValueError) as exc:
                raise MutationJournalError(
                    f"cannot inspect {source} source {candidate}: {exc}"
                ) from exc
            if not stat.S_ISREG(before.st_mode):
                raise MutationJournalError(
                    f"{source} source {candidate} is not a regular file"
                )
            if before.st_nlink != 1:
                raise MutationJournalError(
                    f"{source} source {candidate} is hard-linked"
                )
            if before.st_size > _ATTRIBUTE_SOURCE_MAX_BYTES:
                raise MutationJournalError(
                    f"{source} source {candidate} exceeds "
                    f"{_ATTRIBUTE_SOURCE_MAX_BYTES} bytes"
                )
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | os.O_NOFOLLOW
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                file_fd = os.open(name, flags, dir_fd=parent_fd)
            except (OSError, TypeError, ValueError) as exc:
                raise MutationJournalError(
                    f"cannot safely open {source} source {candidate}: {exc}"
                ) from exc
            try:
                opened = os.fstat(file_fd)
                stable_fields = (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or any(
                        getattr(before, field) != getattr(opened, field)
                        for field in stable_fields
                    )
                ):
                    raise MutationJournalError(
                        f"{source} source {candidate} changed while opening"
                    )
                chunks: list[bytes] = []
                observed_bytes = 0
                while True:
                    chunk = os.read(file_fd, 64 * 1024)
                    if not chunk:
                        break
                    observed_bytes += len(chunk)
                    if observed_bytes > _ATTRIBUTE_SOURCE_MAX_BYTES:
                        raise MutationJournalError(
                            f"{source} source {candidate} grew beyond "
                            f"{_ATTRIBUTE_SOURCE_MAX_BYTES} bytes"
                        )
                    chunks.append(chunk)
                finished = os.fstat(file_fd)
                if any(
                    getattr(opened, field) != getattr(finished, field)
                    for field in stable_fields
                ):
                    raise MutationJournalError(
                        f"{source} source {candidate} changed during inspection"
                    )
                current = os.stat(
                    name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if any(
                    getattr(opened, field) != getattr(current, field)
                    for field in stable_fields
                ):
                    raise MutationJournalError(
                        f"{source} source {candidate} changed identity"
                    )
                _assert_workspace_parent_binding(root_fd, relative, parent_fd)
                return b"".join(chunks)
            except MutationJournalError:
                raise
            except (OSError, TypeError, ValueError) as exc:
                raise MutationJournalError(
                    f"cannot read {source} source {candidate}: {exc}"
                ) from exc
            finally:
                os.close(file_fd)
    finally:
        os.close(root_fd)


def _nonempty_attribute_source_state(
    root_fd: int,
) -> tuple[tuple[str, str], ...]:
    """Conservatively reject attribute sources hidden by precedence."""

    candidates: list[tuple[str, Path]] = []
    info_result = _run_bound_git(
        root_fd,
        "rev-parse",
        "--path-format=absolute",
        "--git-path",
        "info/attributes",
    )
    if info_result.returncode != 0:
        raise MutationJournalError(
            "cannot resolve repository info/attributes: "
            + info_result.stderr.decode("utf-8", errors="replace").strip()
        )
    info_path = os.fsdecode(info_result.stdout).strip()
    if info_path:
        candidates.append(("info/attributes", Path(info_path)))
    home = os.environ.get("HOME")
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        candidates.append((
            "global attributes",
            Path(xdg) / "git" / "attributes",
        ))
    elif home:
        candidates.append((
            "global attributes",
            Path(home) / ".config" / "git" / "attributes",
        ))
    if home:
        candidates.append((
            "legacy global attributes",
            Path(home) / ".gitattributes",
        ))
    candidates.append(("system attributes", Path("/etc/gitattributes")))

    result: list[tuple[str, str]] = []
    for source, candidate in candidates:
        payload = _bounded_attribute_source_payload(
            candidate,
            source=source,
        )
        if payload is not None and payload.strip():
            result.append((
                "attribute-source",
                f"{source}:{hashlib.sha256(payload).hexdigest()}",
            ))
    return tuple(result)


def _scope_escape_transform_authority(
    _cwd: Path,
    root_fd: int,
    path: str,
) -> tuple[str, tuple[tuple[str, tuple[tuple[str, str], ...]], ...]]:
    """Bind current/index/HEAD/ambient transform authority for one path."""

    _assert_held_checkout_root(root_fd)
    head_oid = _head_commit_oid(root_fd)
    states: list[tuple[str, tuple[tuple[str, str], ...]]] = [
        ("config-local", _transform_config_state(root_fd, ambient=False)),
        ("config-ambient", _transform_config_state(root_fd, ambient=True)),
        ("attribute-sources", _nonempty_attribute_source_state(root_fd)),
    ]
    # Configuration and external-source authority must be rejected before
    # ``check-attr``.  A configured FIFO/device attributesFile (or config
    # include) must time out/fail here rather than hold the mutation lock or
    # make Git consult it while resolving path attributes.
    _reject_active_scope_escape_transforms(path, (head_oid, tuple(states)))
    for name, selector, ambient in (
        ("worktree", (), False),
        ("index", ("--cached",), False),
        ("head", (f"--source={head_oid}",), False),
        ("ambient-worktree", (), True),
        ("ambient-index", ("--cached",), True),
        ("ambient-head", (f"--source={head_oid}",), True),
    ):
        _assert_held_checkout_root(root_fd)
        states.append((
            name,
            _transform_attribute_state(
                root_fd,
                path,
                selector=selector,
                ambient=ambient,
            ),
        ))
        _assert_held_checkout_root(root_fd)
    _assert_held_checkout_root(root_fd)
    return head_oid, tuple(states)


def _reject_active_scope_escape_transforms(
    path: str,
    authority: tuple[
        str,
        tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    ],
) -> None:
    inactive = {"unset", "unspecified"}
    active: list[tuple[str, str, str]] = []
    for source, state in authority[1]:
        for attribute, value in state:
            normalized_attribute = attribute.lower()
            normalized_value = value.lower()
            if normalized_attribute == "core.autocrlf":
                enabled = normalized_value != "false"
            elif normalized_attribute == "core.eol":
                enabled = normalized_value != "native"
            elif normalized_attribute in {
                "core.attributesfile",
                "attribute-source",
            }:
                enabled = True
            else:
                enabled = normalized_value not in inactive
            if enabled:
                active.append((source, attribute, value))
    if active:
        source, attribute, value = active[0]
        raise MutationJournalError(
            f"tracked scope_escape {path!r} has transforming Git attribute "
            f"{attribute}={value!r} from {source}; raw restore is unsafe"
        )


@contextmanager
def _open_workspace_parent(root_fd: int, path: str) -> Iterator[tuple[int, str]]:
    """Yield the held parent descriptor and final name for a normalized path."""

    components = tuple(path.split("/"))
    if not components or any(
        component in {"", ".", ".."} for component in components
    ):
        raise MutationJournalError(
            f"cannot safely access non-normalized workspace path {path!r}"
        )
    try:
        descriptor = os.dup(root_fd)
    except OSError as exc:
        raise MutationJournalError(
            f"cannot duplicate held rollback checkout descriptor: {exc}"
        ) from exc
    try:
        for component in components[:-1]:
            try:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise MutationJournalError(
                    "cannot safely open parent of workspace path "
                    f"{path!r}: {exc}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        yield descriptor, components[-1]
    finally:
        os.close(descriptor)


@contextmanager
def _open_workspace_parent_if_present(
    root_fd: int,
    path: str,
) -> Iterator[tuple[int, str] | None]:
    """Bind a parent, yielding ``None`` only for a missing ancestor.

    Read-only tree comparison needs to distinguish a safely absent path from
    an unsafe alias or inspection failure.  The mutating parent helper above
    intentionally treats both as errors; this variant preserves the same
    descriptor/O_NOFOLLOW traversal while making only ``ENOENT`` a typed
    absence result.
    """

    components = tuple(path.split("/"))
    if not components or any(
        component in {"", ".", ".."} for component in components
    ):
        raise MutationJournalError(
            f"cannot safely access non-normalized workspace path {path!r}"
        )
    try:
        descriptor = os.dup(root_fd)
    except OSError as exc:
        raise MutationJournalError(
            f"cannot duplicate held rollback checkout descriptor: {exc}"
        ) from exc
    try:
        for component in components[:-1]:
            try:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                yield None
                return
            except (OSError, TypeError, ValueError) as exc:
                raise MutationJournalError(
                    "cannot safely open parent of workspace path "
                    f"{path!r}: {exc}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        yield descriptor, components[-1]
    finally:
        os.close(descriptor)


def _assert_workspace_parent_binding(
    root_fd: int,
    path: str,
    held_parent_fd: int,
) -> None:
    """Require the held parent to remain the root-relative path's parent."""

    try:
        held = os.fstat(held_parent_fd)
    except OSError as exc:
        raise MutationJournalError(
            f"cannot inspect held parent of workspace path {path!r}: {exc}"
        ) from exc
    with _open_workspace_parent(root_fd, path) as (current_parent_fd, _name):
        try:
            current = os.fstat(current_parent_fd)
        except OSError as exc:
            raise MutationJournalError(
                f"cannot inspect current parent of workspace path {path!r}: {exc}"
            ) from exc
    if current.st_dev != held.st_dev or current.st_ino != held.st_ino:
        raise MutationJournalError(
            f"parent of workspace path {path!r} changed identity after "
            "classification"
        )


def _same_entry_identity(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _assert_workspace_entry_binding(
    parent_fd: int,
    name: str,
    path: str,
    expected: os.stat_result | None,
) -> None:
    """Require a final entry to retain its classified identity/absence."""

    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if expected is None:
            return
        raise MutationJournalError(
            f"workspace path {path!r} disappeared during verification"
        ) from None
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot revalidate workspace path {path!r}: {exc}"
        ) from exc
    if expected is None:
        raise MutationJournalError(
            f"workspace path {path!r} appeared during verification"
        )
    if not _same_entry_identity(expected, current):
        raise MutationJournalError(
            f"workspace path {path!r} changed identity during verification"
        )


def _snapshot_workspace_entry(
    parent_fd: int,
    name: str,
    path: str,
) -> tuple[os.stat_result | None, int | None]:
    """Snapshot the final entry before classification, holding regular files."""

    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None, None
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot snapshot workspace path {path!r}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        return metadata, None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot bind regular workspace path {path!r}: {exc}"
        ) from exc
    if not stat.S_ISREG(opened.st_mode) or not _same_entry_identity(
        metadata,
        opened,
    ):
        os.close(descriptor)
        raise MutationJournalError(
            f"workspace path {path!r} changed identity while binding"
        )
    return metadata, descriptor


def _git_index_mode(root_fd: int, path: str) -> str | None:
    """Return an exact literal path's stage-zero mode, or ``None`` if untracked.

    ``git ls-files --error-unmatch`` reserves exit status 1 for a path that is
    not present in the index.  Any other non-zero status is an operational
    failure, not evidence that deleting the path is safe.  Parsing ``--stage``
    output also distinguishes a regular tracked path from a ``160000`` gitlink
    and prevents a directory pathspec from being mistaken for an exact entry.
    """

    result = _run_bound_git(
        root_fd,
        "--literal-pathspecs",
        "ls-files",
        "--stage",
        "-z",
        "--error-unmatch",
        "--",
        path,
    )
    if result.returncode == 1:
        return None
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise MutationJournalError(
            f"cannot classify workspace path {path!r}: git ls-files exited "
            f"{result.returncode}: {message}"
        )

    exact_entries: list[tuple[str, str]] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, separator, encoded_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise MutationJournalError(
                f"cannot classify workspace path {path!r}: malformed "
                "git ls-files --stage output"
            )
        if os.fsdecode(encoded_path) == path:
            if not fields[0].isdigit() or not fields[2].isdigit():
                raise MutationJournalError(
                    f"cannot classify workspace path {path!r}: malformed "
                    "git ls-files --stage metadata"
                )
            exact_entries.append(
                (
                    fields[0].decode("ascii", errors="strict"),
                    fields[2].decode("ascii", errors="strict"),
                )
            )
    if not exact_entries:
        # A literal directory pathspec can select tracked descendants.  It is
        # not itself an index entry and must therefore follow the untracked
        # empty-directory cleanup path rather than a recursive Git restore.
        return None
    if len(exact_entries) != 1 or exact_entries[0][1] != "0":
        raise MutationJournalError(
            f"cannot classify workspace path {path!r}: index entry is "
            "unmerged or ambiguous"
        )
    return exact_entries[0][0]


def _git_head_entry(root_fd: int, path: str) -> tuple[str, str, str] | None:
    """Return the exact ``HEAD`` tree entry for one literal path."""

    result = _run_bound_git(
        root_fd,
        "--literal-pathspecs",
        "ls-tree",
        "-z",
        "HEAD",
        "--",
        path,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise MutationJournalError(
            f"cannot inspect HEAD path {path!r}: git ls-tree exited "
            f"{result.returncode}: {message}"
        )
    exact_entries: list[tuple[str, str, str]] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, separator, encoded_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise MutationJournalError(
                f"cannot inspect HEAD path {path!r}: malformed git ls-tree output"
            )
        if os.fsdecode(encoded_path) != path:
            continue
        if not fields[0].isdigit() or fields[1] not in {b"blob", b"commit"}:
            raise MutationJournalError(
                f"cannot inspect HEAD path {path!r}: malformed tree metadata"
            )
        exact_entries.append(
            (
                fields[0].decode("ascii", errors="strict"),
                fields[1].decode("ascii", errors="strict"),
                fields[2].decode("ascii", errors="strict"),
            )
        )
    if not exact_entries:
        return None
    if len(exact_entries) != 1:
        raise MutationJournalError(f"HEAD path {path!r} is ambiguous")
    return exact_entries[0]


def _new_git_object_hash(expected_oid: str):
    if any(character not in "0123456789abcdef" for character in expected_oid):
        raise MutationJournalError("Git object identifier is not lowercase hex")
    if len(expected_oid) == 40:
        return hashlib.sha1(usedforsecurity=False)
    if len(expected_oid) == 64:
        return hashlib.sha256()
    raise MutationJournalError(
        f"unsupported Git object identifier length: {len(expected_oid)}"
    )


def _git_tree_entry(
    root_fd: int,
    tree_oid: str,
    path: str,
) -> tuple[str, str, str] | None:
    """Return one literal path entry from an immutable full Git object ID."""

    # Validate the complete object ID before placing it in argv. The bound Git
    # runner disables replacements and lazy fetching, so authority comes only
    # from the locally authenticated object graph selected here.
    _new_git_object_hash(tree_oid)
    result = _run_bound_git(
        root_fd,
        "--literal-pathspecs",
        "ls-tree",
        "-z",
        tree_oid,
        "--",
        path,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise MutationJournalError(
            f"cannot inspect base {tree_oid} path {path!r}: git ls-tree "
            f"exited {result.returncode}: {message}"
        )
    exact_entries: list[tuple[str, str, str]] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, separator, encoded_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise MutationJournalError(
                f"cannot inspect base {tree_oid} path {path!r}: malformed "
                "git ls-tree output"
            )
        if os.fsdecode(encoded_path) != path:
            continue
        if not fields[0].isdigit() or fields[1] not in {b"blob", b"commit"}:
            raise MutationJournalError(
                f"cannot inspect base {tree_oid} path {path!r}: malformed "
                "tree metadata"
            )
        exact_entries.append(
            (
                fields[0].decode("ascii", errors="strict"),
                fields[1].decode("ascii", errors="strict"),
                fields[2].decode("ascii", errors="strict"),
            )
        )
    if not exact_entries:
        return None
    if len(exact_entries) != 1:
        raise MutationJournalError(
            f"base {tree_oid} path {path!r} is ambiguous"
        )
    return exact_entries[0]


def _verify_blob_payload(expected_oid: str, payload: bytes, path: str) -> None:
    digest = _new_git_object_hash(expected_oid)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    if digest.hexdigest() != expected_oid:
        raise MutationJournalError(
            f"restored workspace path {path!r} remains different from HEAD"
        )


def _head_blob_payload(
    root_fd: int,
    path: str,
    expected_oid: str,
) -> bytes:
    """Read and authenticate one HEAD blob before any filesystem mutation."""

    # Validate before placing an object identifier in argv.
    _new_git_object_hash(expected_oid)
    result = _run_bound_git(root_fd, "cat-file", "blob", expected_oid)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise MutationJournalError(
            f"cannot read HEAD blob for {path!r}: git cat-file exited "
            f"{result.returncode}: {message}"
        )
    _verify_blob_payload(expected_oid, result.stdout, path)
    return result.stdout


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise MutationJournalError(
                "descriptor-relative restore made no write progress"
            )
        written += count


def _restore_head_entry_at_parent(
    root_fd: int,
    parent_fd: int,
    name: str,
    path: str,
    head_entry: tuple[str, str, str] | None,
    entry_metadata: os.stat_result | None,
    held_regular_fd: int | None,
) -> None:
    """Restore ``HEAD`` through a held parent without pathname re-resolution."""

    if head_entry is None:
        if entry_metadata is not None:
            raise MutationJournalError(
                f"cannot safely remove tracked path {path!r} absent from HEAD"
            )
        return
    expected_mode, object_type, expected_oid = head_entry
    if expected_mode == "160000" or object_type == "commit":
        raise MutationJournalError(
            f"cannot descriptor-restore gitlink/submodule path {path!r}"
        )
    if expected_mode not in {"100644", "100755", "120000"}:
        raise MutationJournalError(
            f"unsupported HEAD mode for descriptor restore: {expected_mode}"
        )
    if object_type != "blob":
        raise MutationJournalError(
            f"unsupported HEAD object type for descriptor restore: {object_type}"
        )
    if entry_metadata is not None and (
        not stat.S_ISREG(entry_metadata.st_mode)
        or expected_mode == "120000"
    ):
        raise MutationJournalError(
            f"cannot safely restore existing nonregular or type-transition "
            f"workspace path {path!r}"
        )
    payload = _head_blob_payload(root_fd, path, expected_oid)
    try:
        current_metadata = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        current_metadata = None
    except (OSError, TypeError, ValueError) as exc:
        raise MutationJournalError(
            f"cannot revalidate descriptor-relative restore path {path!r}: {exc}"
        ) from exc
    if entry_metadata is None:
        if current_metadata is not None:
            raise MutationJournalError(
                f"workspace path {path!r} appeared after classification"
            )
    elif current_metadata is None or not _same_entry_identity(
        entry_metadata,
        current_metadata,
    ):
        raise MutationJournalError(
            f"workspace path {path!r} changed identity after classification"
        )

    if entry_metadata is None:
        if expected_mode == "120000":
            if b"\0" in payload:
                raise MutationJournalError(
                    f"HEAD symlink payload for {path!r} contains an embedded NUL"
                )
            try:
                os.symlink(os.fsdecode(payload), name, dir_fd=parent_fd)
            except FileExistsError as exc:
                raise MutationJournalError(
                    f"workspace path {path!r} appeared before no-replace "
                    "symlink install"
                ) from exc
            except (OSError, TypeError, ValueError) as exc:
                raise MutationJournalError(
                    f"cannot install descriptor-relative symlink {path!r}: {exc}"
                ) from exc
            return
        try:
            created_fd = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError as exc:
            raise MutationJournalError(
                f"workspace path {path!r} appeared before no-replace file install"
            ) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise MutationJournalError(
                f"cannot create descriptor-relative restore file {path!r}: {exc}"
            ) from exc
        try:
            _write_all(created_fd, payload)
            os.fchmod(
                created_fd,
                0o755 if expected_mode == "100755" else 0o644,
            )
            os.fsync(created_fd)
        except MutationJournalError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise MutationJournalError(
                f"cannot write descriptor-relative restore file {path!r}: {exc}"
            ) from exc
        finally:
            os.close(created_fd)
        return

    if expected_mode in {"100644", "100755"} and held_regular_fd is not None:
        assert entry_metadata is not None
        try:
            held_metadata = os.fstat(held_regular_fd)
            if held_metadata.st_nlink != 1:
                raise MutationJournalError(
                    f"cannot safely restore hard-linked workspace path {path!r}"
                )
            try:
                writer_fd = os.open(
                    name,
                    os.O_WRONLY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
            except PermissionError:
                os.fchmod(held_regular_fd, held_metadata.st_mode | stat.S_IWUSR)
                writer_fd = os.open(
                    name,
                    os.O_WRONLY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
            try:
                writer_metadata = os.fstat(writer_fd)
                if not _same_entry_identity(held_metadata, writer_metadata):
                    raise MutationJournalError(
                        f"workspace path {path!r} changed before restore write"
                    )
                os.ftruncate(writer_fd, 0)
                _write_all(writer_fd, payload)
                baseline_mode = stat.S_IMODE(entry_metadata.st_mode)
                restored_mode = (baseline_mode & ~0o111) | (
                    0o111 if expected_mode == "100755" else 0
                )
                os.fchmod(
                    writer_fd,
                    restored_mode,
                )
                os.fsync(writer_fd)
            finally:
                os.close(writer_fd)
        except MutationJournalError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise MutationJournalError(
                f"cannot restore held regular workspace path {path!r}: {exc}"
            ) from exc
        return
    raise MutationJournalError(
        f"cannot safely restore unsupported workspace transition for {path!r}"
    )


def _verify_worktree_path_matches_tree(
    root_fd: int,
    path: str,
    tree_entry: tuple[str, str, str] | None,
    *,
    tree_label: str,
    held_parent_fd: int | None = None,
    held_name: str | None = None,
) -> None:
    """Require an exact descriptor-bound worktree match to one tree entry.

    This proof reads bytes and mode descriptor-relatively and compares their
    Git object digest with the selected immutable tree.  It deliberately does
    not trust the live index or ``git diff``, both of which can suppress
    differences behind assume-unchanged or skip-worktree flags.
    """

    parent_context = (
        _open_workspace_parent_if_present(root_fd, path)
        if held_parent_fd is None
        else nullcontext((held_parent_fd, held_name or path.rsplit("/", 1)[-1]))
    )
    with parent_context as parent_binding:
        if parent_binding is None:
            if tree_entry is None:
                # The initial traversal proved an ancestor absent. Rewalk from
                # the held root before accepting that proof so an ancestor
                # created during classification cannot redirect the current
                # root-relative path behind our back.
                with _open_workspace_parent_if_present(
                    root_fd,
                    path,
                ) as current_binding:
                    if current_binding is not None:
                        raise MutationJournalError(
                            f"parent of workspace path {path!r} appeared "
                            "during verification"
                        )
                return
            raise MutationJournalError(
                f"workspace path {path!r} is absent but present in {tree_label}"
            )
        parent_fd, name = parent_binding
        _assert_workspace_parent_binding(root_fd, path, parent_fd)
        try:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if tree_entry is None:
                _assert_workspace_entry_binding(
                    parent_fd,
                    name,
                    path,
                    None,
                )
                _assert_workspace_parent_binding(root_fd, path, parent_fd)
                return
            raise MutationJournalError(
                f"workspace path {path!r} is absent but present in {tree_label}"
            ) from None
        except (OSError, TypeError, ValueError) as exc:
            raise MutationJournalError(
                f"cannot inspect workspace path {path!r}: {exc}"
            ) from exc
        if tree_entry is None:
            raise MutationJournalError(
                f"workspace path {path!r} remains present but absent in "
                f"{tree_label}"
            )
        expected_mode, object_type, expected_oid = tree_entry
        if expected_mode == "160000" or object_type == "commit":
            raise MutationJournalError(
                f"cannot verify gitlink/submodule workspace path {path!r}"
            )
        if expected_mode == "120000":
            if not stat.S_ISLNK(metadata.st_mode):
                raise MutationJournalError(
                    f"workspace path {path!r} has the wrong file type"
                )
            if metadata.st_nlink != 1:
                raise MutationJournalError(
                    f"workspace path {path!r} is hard-linked; Git tree "
                    "authority cannot attest alias topology"
                )
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            try:
                target = os.readlink(name, dir_fd=parent_fd)
                observed = os.stat(
                    name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise MutationJournalError(
                    f"cannot read restored symlink {path!r}: {exc}"
                ) from exc
            if any(
                getattr(metadata, field) != getattr(observed, field)
                for field in stable_fields
            ):
                raise MutationJournalError(
                    f"workspace path {path!r} changed during verification"
                )
            _verify_blob_payload(expected_oid, os.fsencode(target), path)
            _assert_workspace_entry_binding(
                parent_fd,
                name,
                path,
                metadata,
            )
            _assert_workspace_parent_binding(root_fd, path, parent_fd)
            try:
                sealed = os.stat(
                    name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise MutationJournalError(
                    f"cannot revalidate restored symlink {path!r}: {exc}"
                ) from exc
            if any(
                getattr(metadata, field) != getattr(sealed, field)
                for field in stable_fields
            ):
                raise MutationJournalError(
                    f"workspace path {path!r} changed after verification"
                )
            return
        if expected_mode not in {"100644", "100755"} or object_type != "blob":
            raise MutationJournalError(
                f"unsupported {tree_label} entry for workspace path {path!r}: "
                f"mode={expected_mode}, type={object_type}"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise MutationJournalError(
                f"workspace path {path!r} has the wrong file type"
            )
        try:
            file_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise MutationJournalError(
                f"cannot open restored workspace path {path!r}: {exc}"
            ) from exc
        try:
            opened = os.fstat(file_fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
            ):
                raise MutationJournalError(
                    f"workspace path {path!r} changed identity"
                )
            if opened.st_nlink != 1:
                raise MutationJournalError(
                    f"workspace path {path!r} is hard-linked; Git tree "
                    "authority cannot attest alias topology"
                )
            expected_executable = expected_mode == "100755"
            if bool(opened.st_mode & stat.S_IXUSR) != expected_executable:
                raise MutationJournalError(
                    f"workspace path {path!r} has the wrong "
                    "executable mode"
                )
            digest = _new_git_object_hash(expected_oid)
            digest.update(f"blob {opened.st_size}\0".encode("ascii"))
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            finished = os.fstat(file_fd)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if any(
                getattr(opened, field) != getattr(finished, field)
                for field in stable_fields
            ):
                raise MutationJournalError(
                    f"workspace path {path!r} changed during verification"
                )
            if digest.hexdigest() != expected_oid:
                raise MutationJournalError(
                    f"workspace path {path!r} remains different from "
                    f"{tree_label}"
                )
            _assert_workspace_entry_binding(
                parent_fd,
                name,
                path,
                metadata,
            )
            _assert_workspace_parent_binding(root_fd, path, parent_fd)
            sealed = os.fstat(file_fd)
            if any(
                getattr(opened, field) != getattr(sealed, field)
                for field in stable_fields
            ):
                raise MutationJournalError(
                    f"workspace path {path!r} changed after verification"
                )
        except MutationJournalError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise MutationJournalError(
                f"cannot verify restored workspace path {path!r}: {exc}"
            ) from exc
        finally:
            os.close(file_fd)


def _verify_worktree_path_matches_head(
    root_fd: int,
    path: str,
    head_entry: tuple[str, str, str] | None,
    *,
    held_parent_fd: int | None = None,
    held_name: str | None = None,
) -> None:
    """Require the exact restored worktree path to match ``HEAD``."""

    _verify_worktree_path_matches_tree(
        root_fd,
        path,
        head_entry,
        tree_label="HEAD",
        held_parent_fd=held_parent_fd,
        held_name=held_name,
    )


def _delete_untracked_workspace_path(
    root_fd: int,
    path: str,
    *,
    held_parent_fd: int | None = None,
    held_name: str | None = None,
    expected_metadata: os.stat_result | None | object = _UNBOUND_ENTRY,
) -> bool:
    """Delete one normalized untracked path without following parent aliases.

    The caller has already normalized ``path``, but pathname validation alone
    cannot close a parent-symlink swap between validation and deletion.  Walk
    every parent with held directory descriptors and ``O_NOFOLLOW``, then
    inspect and remove the final component relative to the held parent.

    Returns ``False`` when the entry is already absent and ``True`` when it was
    removed.  All unsafe or incomplete operations raise
    :class:`MutationJournalError`; callers choose strict or best-effort policy.
    """

    parent_context = (
        _open_workspace_parent(root_fd, path)
        if held_parent_fd is None
        else nullcontext((held_parent_fd, held_name or path.rsplit("/", 1)[-1]))
    )
    with parent_context as (parent_fd, name):
        try:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        except (OSError, TypeError, ValueError) as exc:
            raise MutationJournalError(
                f"cannot inspect untracked workspace path {path!r}: {exc}"
            ) from exc
        if expected_metadata is not _UNBOUND_ENTRY:
            if expected_metadata is None:
                raise MutationJournalError(
                    f"workspace path {path!r} appeared after classification"
                )
            assert isinstance(expected_metadata, os.stat_result)
            if not _same_entry_identity(expected_metadata, metadata):
                raise MutationJournalError(
                    f"workspace path {path!r} changed identity after "
                    "classification"
                )
        _assert_workspace_parent_binding(root_fd, path, parent_fd)
        _assert_workspace_entry_binding(parent_fd, name, path, metadata)
        try:
            if stat.S_ISDIR(metadata.st_mode):
                os.rmdir(name, dir_fd=parent_fd)
            else:
                # Includes regular files, dangling symlinks, and special
                # entries.  ``unlink`` removes the directory entry itself and
                # never follows a final symlink.
                os.unlink(name, dir_fd=parent_fd)
        except (OSError, TypeError, ValueError) as exc:
            raise MutationJournalError(
                f"cannot delete untracked workspace path {path!r}: {exc}"
            ) from exc
        # A held parent descriptor prevents path traversal, but it may become
        # detached if the canonical parent is renamed during unlink/rmdir.
        # Rebind it to the root-relative path and prove the canonical entry is
        # absent before reporting deletion success.
        _assert_workspace_parent_binding(root_fd, path, parent_fd)
        _assert_workspace_entry_binding(parent_fd, name, path, None)
        return True


def revert_scope_escapes(
    impl: Implementation,
    cwd: str | Path,
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Revert/delete the files an implementer wrote OUTSIDE its declared
    scope (``Implementation.scope_escapes``), leaving its in-scope diff
    untouched.

    Tracked escapes get reverted to HEAD without changing the index;
    untracked escapes get unlinked (empty dirs ``rmdir``'d). The default is
    best-effort: a per-file failure is logged and processing continues. With
    ``strict=True`` unsafe paths and incomplete cleanup raise
    :class:`MutationJournalError`; the cross-run reaper uses that mode inside
    one durable outer transaction. Returns ``(reverted, deleted)`` path lists.

    ``rollback_slice`` calls this after reverting the full diff. It is
    ALSO called standalone by callers that must keep the in-scope writes
    on disk but cannot tolerate an out-of-scope orphan — notably the
    breach loop (``workflows.breach``), which carries the Defender's
    legitimate cumulative fixes across rounds. Without this, a stray
    out-of-scope file the Defender writes (e.g. a ``*_orig.cpp`` backup)
    gets swept into a glob-based build and poisons every subsequent
    round's build/verify with a duplicate-symbol link error, even though
    the path-scoped per-round diff never touched it.
    """
    try:
        _reject_git_redirect_env()
        cwd_path, root_fd = _open_checkout_root(cwd)
    except MutationJournalError as exc:
        if strict:
            raise
        logger.warning(
            "revert_scope_escapes %s: cannot bind checkout: %s",
            impl.slice_id,
            exc,
        )
        return [], []
    try:
        return _revert_scope_escapes_bound(
            impl,
            cwd_path,
            root_fd,
            strict=strict,
        )
    finally:
        os.close(root_fd)


def _clean_bound_scope_escape(
    impl: Implementation,
    cwd: Path,
    root_fd: int,
    path: str,
    *,
    strict: bool,
) -> tuple[bool, bool]:
    """Clean one path while holding its parent across classification."""

    try:
        parent_context = _open_workspace_parent(root_fd, path)
        parent_fd, name = parent_context.__enter__()
    except MutationJournalError as exc:
        if strict:
            raise
        logger.warning(
            "revert_scope_escapes %s: cannot bind parent for %s: %s",
            impl.slice_id,
            path,
            exc,
        )
        return False, False
    held_regular_fd: int | None = None
    try:
        try:
            entry_metadata, held_regular_fd = _snapshot_workspace_entry(
                parent_fd,
                name,
                path,
            )
            index_mode = _git_index_mode(root_fd, path)
            head_oid = _head_commit_oid(root_fd)
            head_entry = _git_tree_entry(root_fd, head_oid, path)
            _assert_workspace_parent_binding(root_fd, path, parent_fd)
        except MutationJournalError as exc:
            if strict:
                raise MutationJournalError(
                    f"cannot classify scope_escape {path!r}: {exc}"
                ) from exc
            logger.warning(
                "revert_scope_escapes %s: failed to classify scope_escape "
                "%s: %s",
                impl.slice_id,
                path,
                exc,
            )
            return False, False

        if index_mode is None:
            if head_entry is not None:
                mismatch = MutationJournalError(
                    f"cannot classify tracked scope_escape {path!r}: HEAD "
                    "contains the path but the index does not"
                )
                if strict:
                    raise mismatch
                logger.warning(
                    "revert_scope_escapes %s: %s",
                    impl.slice_id,
                    mismatch,
                )
                return False, False
            try:
                deleted = _delete_untracked_workspace_path(
                    root_fd,
                    path,
                    held_parent_fd=parent_fd,
                    held_name=name,
                    expected_metadata=entry_metadata,
                )
            except MutationJournalError as exc:
                if strict:
                    raise
                logger.warning(
                    "revert_scope_escapes %s: failed to delete scope_escape "
                    "%s: %s",
                    impl.slice_id,
                    path,
                    exc,
                )
                return False, False
            return False, deleted

        if head_entry is None or index_mode != head_entry[0]:
            mismatch = MutationJournalError(
                f"cannot restore tracked scope_escape {path!r}: index mode "
                f"{index_mode!r} does not match immutable HEAD entry "
                f"{None if head_entry is None else head_entry[0]!r}"
            )
            if strict:
                raise mismatch
            logger.warning(
                "revert_scope_escapes %s: %s",
                impl.slice_id,
                mismatch,
            )
            return False, False
        if index_mode == "160000" or (
            head_entry is not None and head_entry[0] == "160000"
        ):
            gitlink_error = MutationJournalError(
                "cannot safely restore tracked scope_escape "
                f"{path!r}: path is a gitlink/submodule"
            )
            if strict:
                raise gitlink_error
            logger.warning(
                "revert_scope_escapes %s: %s",
                impl.slice_id,
                gitlink_error,
            )
            return False, False
        try:
            transform_authority = _scope_escape_transform_authority(
                cwd,
                root_fd,
                path,
            )
            if transform_authority[0] != head_oid:
                raise MutationJournalError(
                    "HEAD changed while classifying scope_escape transform "
                    "authority"
                )
            _reject_active_scope_escape_transforms(path, transform_authority)
            _assert_workspace_parent_binding(root_fd, path, parent_fd)
        except MutationJournalError as exc:
            if strict:
                raise MutationJournalError(
                    f"cannot restore tracked scope_escape {path!r}: {exc}"
                ) from exc
            logger.warning(
                "revert_scope_escapes %s: refusing transformed scope_escape "
                "%s: %s",
                impl.slice_id,
                path,
                exc,
            )
            return False, False
        try:
            _restore_head_entry_at_parent(
                root_fd,
                parent_fd,
                name,
                path,
                head_entry,
                entry_metadata,
                held_regular_fd,
            )
        except MutationJournalError as exc:
            if strict:
                raise MutationJournalError(
                    f"cannot restore tracked scope_escape {path!r}: {exc}"
                ) from exc
            logger.warning(
                "revert_scope_escapes %s: failed to revert scope_escape "
                "%s: %s",
                impl.slice_id,
                path,
                exc,
            )
            return False, False
        try:
            after_restore_authority = _scope_escape_transform_authority(
                cwd,
                root_fd,
                path,
            )
            _reject_active_scope_escape_transforms(
                path,
                after_restore_authority,
            )
            if after_restore_authority != transform_authority:
                raise MutationJournalError(
                    "Git transform authority changed during raw restore"
                )
            _assert_workspace_parent_binding(root_fd, path, parent_fd)
            _verify_worktree_path_matches_tree(
                root_fd,
                path,
                head_entry,
                tree_label=f"HEAD {head_oid}",
                held_parent_fd=parent_fd,
                held_name=name,
            )
            final_transform_authority = _scope_escape_transform_authority(
                cwd,
                root_fd,
                path,
            )
            _reject_active_scope_escape_transforms(
                path,
                final_transform_authority,
            )
            if final_transform_authority != transform_authority:
                raise MutationJournalError(
                    "Git transform authority changed during restore verification"
                )
        except MutationJournalError as exc:
            if strict:
                raise
            logger.warning(
                "revert_scope_escapes %s: failed to verify reverted "
                "scope_escape %s: %s",
                impl.slice_id,
                path,
                exc,
            )
            return False, False
        return True, False
    finally:
        if held_regular_fd is not None:
            os.close(held_regular_fd)
        parent_context.__exit__(None, None, None)


def _revert_scope_escapes_bound(
    impl: Implementation,
    cwd_path: Path,
    root_fd: int,
    *,
    strict: bool,
) -> tuple[list[str], list[str]]:
    """Apply scope cleanup against an already-held checkout descriptor."""

    try:
        _assert_checkout_root_binding(cwd_path, root_fd)
    except MutationJournalError as exc:
        raise MutationJournalError(
            f"workspace authority root changed identity: {cwd_path}"
        ) from exc
    reverted_escapes: list[str] = []
    deleted_escapes: list[str] = []
    normalized_escapes: list[str] = []
    for rendered_path in getattr(impl, "scope_escapes", ()) or ():
        try:
            esc_path = _normalize_rollback_path(cwd_path, rendered_path)
        except MutationJournalError as exc:
            if strict:
                raise
            logger.warning(
                "revert_scope_escapes %s: rejected unsafe scope_escape %r: %s",
                impl.slice_id,
                rendered_path,
                exc,
            )
            continue
        normalized_escapes.append(esc_path)
    # Path normalization may consult the admitted public checkout spelling.
    # Complete it for the whole batch, then reprove that spelling before the
    # first mutation.  After this point all classification and cleanup is
    # relative to ``root_fd`` and remains confined if the public name is
    # concurrently replaced.
    _assert_checkout_root_binding(cwd_path, root_fd)
    for esc_path in normalized_escapes:
        reverted, deleted = _clean_bound_scope_escape(
            impl,
            cwd_path,
            root_fd,
            esc_path,
            strict=strict,
        )
        if reverted:
            reverted_escapes.append(esc_path)
        if deleted:
            deleted_escapes.append(esc_path)
        _assert_held_checkout_root(root_fd)
    _assert_held_checkout_root(root_fd)
    return reverted_escapes, deleted_escapes


def rollback_slice(
    corpus: Corpus, impl: Implementation, cwd: str | Path,
) -> bool:
    """Revert a slice's writes via ``git apply -R`` on its ``diff_ref``.

    Returns:
        True on success (including the empty-diff no-op case).
        False if the diff_ref is unresolvable or ``git apply -R``
        returns non-zero.

    The function never raises; failures are logged at WARNING/ERROR and
    surfaced via the return value. The caller decides whether to halt
    on a failed rollback (typically: log and proceed, since the slice
    has already failed and the dirty state is the lesser of two evils).
    """
    try:
        _reject_git_redirect_env()
        diff_bytes = corpus.get_artifact(impl.diff_ref)
    except Exception as e:  # noqa: BLE001 — corpus failures are non-fatal here
        logger.warning(
            "rollback_slice %s: diff_ref %s unresolvable (%s) — skipping",
            impl.slice_id, impl.diff_ref, e,
        )
        return False

    try:
        cwd_path, root_fd = _open_checkout_root(cwd)
    except MutationJournalError as exc:
        logger.error(
            "rollback_slice %s: cannot bind checkout %s: %s",
            impl.slice_id,
            cwd,
            exc,
        )
        return False
    try:
        return _rollback_slice_bound(diff_bytes, impl, cwd_path, root_fd)
    finally:
        os.close(root_fd)


def _rollback_slice_bound(
    diff_bytes: bytes,
    impl: Implementation,
    cwd_path: Path,
    root_fd: int,
) -> bool:
    """Roll back one slice against an already-held checkout descriptor."""

    cwd_str = str(cwd_path)

    if diff_bytes.strip():
        try:
            result = _run_bound_git(
                root_fd,
                "apply",
                "-R",
                input_bytes=diff_bytes,
            )
        except MutationJournalError as e:
            logger.error(
                "rollback_slice %s: git apply -R raised %s in %s",
                impl.slice_id, e, cwd_str,
            )
            return False

        if result.returncode != 0:
            logger.error(
                "rollback_slice %s: git apply -R exit=%d in %s. stderr: %s",
                impl.slice_id, result.returncode, cwd_str,
                result.stderr.decode("utf-8", errors="replace").strip(),
            )
            return False

    # ``git apply -R`` only reverts changes to *tracked* files. Files the
    # slice CREATED (declared in ``directive.creates_paths`` / captured
    # in ``files_modified`` but absent from HEAD) remain on disk as
    # untracked. Delete them so the rollback is symmetric to what the
    # slice wrote. Existing tracked files are left alone — they were
    # either reverted by ``apply -R`` (modifications) or were never
    # touched.
    deleted_untracked: list[str] = []
    for rendered_path in impl.files_modified:
        try:
            path = _normalize_rollback_path(cwd_path, rendered_path)
        except MutationJournalError as exc:
            logger.warning(
                "rollback_slice %s: rejected unsafe files_modified path %r: %s",
                impl.slice_id,
                rendered_path,
                exc,
            )
            continue
        try:
            with _open_workspace_parent(root_fd, path) as (parent_fd, name):
                entry_metadata, held_regular_fd = _snapshot_workspace_entry(
                    parent_fd,
                    name,
                    path,
                )
                try:
                    index_mode = _git_index_mode(root_fd, path)
                    _assert_workspace_parent_binding(root_fd, path, parent_fd)
                    if index_mode is not None:
                        # Tracked — apply -R already handled it (or it was
                        # unrelated).
                        continue
                    if _delete_untracked_workspace_path(
                        root_fd,
                        path,
                        held_parent_fd=parent_fd,
                        held_name=name,
                        expected_metadata=entry_metadata,
                    ):
                        deleted_untracked.append(path)
                finally:
                    if held_regular_fd is not None:
                        os.close(held_regular_fd)
        except MutationJournalError as exc:
            logger.warning(
                "rollback_slice %s: failed to classify or delete untracked "
                "%s: %s",
                impl.slice_id, path, exc,
            )

    # Clean any scope-escape paths recorded against the slice. These
    # are files the implementer wrote outside its declared scope, so
    # the path-scoped ``git apply -R`` above did NOT see them. Tracked
    # files get restored from authenticated HEAD blobs through held parent
    # descriptors; untracked files get unlinked. Best-effort (see
    # ``revert_scope_escapes``): a
    # per-file failure is logged but doesn't fail the whole rollback
    # (the slice has already failed and partial dirt is the lesser evil).
    reverted_escapes, deleted_escapes = _revert_scope_escapes_bound(
        impl,
        cwd_path,
        root_fd,
        strict=False,
    )

    logger.info(
        "rollback_slice %s: reverted %d bytes of diff, deleted %d "
        "untracked path(s), reverted %d scope_escape modification(s), "
        "deleted %d scope_escape untracked path(s) in %s",
        impl.slice_id, len(diff_bytes), len(deleted_untracked),
        len(reverted_escapes), len(deleted_escapes), cwd_str,
    )
    return True


__all__ = ["rollback_slice", "revert_scope_escapes"]
