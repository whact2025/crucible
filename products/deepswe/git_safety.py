# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Host-Git safety boundary for image-derived DeepSWE checkouts.

DeepSWE workspaces originate in benchmark images.  Their working-tree bytes
are benchmark input, but image-provided Git metadata is not trusted execution
configuration: local config, hooks, fsmonitor commands, filters, alternates,
or metadata symlinks must never make a host-side Git probe execute code or
escape the copied object store.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import stat
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from .rust_toolchain import RUST_CARGO_BIN


class GitMetadataSafetyError(RuntimeError):
    """Image-derived Git metadata is unsafe for host-side Git."""


class ContainerGitEnvironmentError(RuntimeError):
    """A managed container's inherited environment cannot be made safe."""


_SAFE_GIT_CONFIG = b"""\
[core]
\trepositoryformatversion = 0
\tfilemode = true
\tbare = false
\tlogallrefupdates = false
\thooksPath = /dev/null
\tfsmonitor = false
\tuntrackedCache = false
[protocol "file"]
\tallow = never
[protocol "ext"]
\tallow = never
"""

GIT_SAFE_OPTIONS: tuple[str, ...] = (
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.fsmonitor=false",
    "-c", "core.untrackedCache=false",
    "-c", "protocol.file.allow=never",
    "-c", "protocol.ext.allow=never",
)


CONTAINER_GIT_CONFIG: tuple[tuple[str, str], ...] = (
    ("core.hooksPath", "/dev/null"),
    ("core.fsmonitor", "false"),
    ("core.untrackedCache", "false"),
    ("core.attributesFile", "/dev/null"),
    ("core.excludesFile", "/dev/null"),
    ("safe.directory", "/app"),
    ("protocol.allow", "never"),
    ("protocol.file.allow", "never"),
    ("protocol.ext.allow", "never"),
)

_CONTAINER_GIT_COMMAND_ENV: tuple[tuple[str, str], ...] = (
    ("GIT_CONFIG_NOSYSTEM", "1"),
    ("GIT_CONFIG_SYSTEM", "/dev/null"),
    ("GIT_CONFIG_GLOBAL", "/dev/null"),
    ("GIT_ATTR_NOSYSTEM", "1"),
    ("GIT_TERMINAL_PROMPT", "0"),
    ("GIT_OPTIONAL_LOCKS", "0"),
)

_CONTAINER_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_UNSAFE_SHELL_ENV = frozenset({
    "BASH_ENV",
    "BASHOPTS",
    "CDPATH",
    "ENV",
    "GLOBIGNORE",
    "PROMPT_COMMAND",
    "PS4",
    "SHELLOPTS",
})
_DEFAULT_CONTAINER_PATH = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_WRITABLE_OR_AUTHORITY_MOUNTS = (
    "/app",
    "/logs",
    "/tests",
    "/solution",
    "/tmp",
)


def _hermetic_container_path(value: str) -> str:
    safe: list[str] = []
    for entry in value.split(":"):
        if not entry.startswith("/"):
            continue
        normalized = "/" + posixpath.normpath(entry).lstrip("/")
        # The Rust bootstrap creates this exact root-owned executable
        # directory from immutable image bytes.  Mutable Cargo state lives in
        # a separate /tmp directory and is never executable PATH authority.
        if normalized == RUST_CARGO_BIN:
            if normalized not in safe:
                safe.append(normalized)
            continue
        if any(
            normalized == root or normalized.startswith(root + "/")
            for root in _WRITABLE_OR_AUTHORITY_MOUNTS
        ):
            continue
        if normalized not in safe:
            safe.append(normalized)
    return ":".join(safe) or _DEFAULT_CONTAINER_PATH


def _container_environment_mapping(
    inherited: Mapping[str, str] | Iterable[str],
) -> dict[str, str]:
    if isinstance(inherited, Mapping):
        entries = inherited.items()
    else:
        parsed: list[tuple[str, str]] = []
        for entry in inherited:
            if not isinstance(entry, str) or "=" not in entry:
                raise ContainerGitEnvironmentError(
                    "container environment entries must be KEY=VALUE strings"
                )
            key, value = entry.split("=", 1)
            parsed.append((key, value))
        entries = parsed

    result: dict[str, str] = {}
    for key, value in entries:
        if not isinstance(key, str) or not isinstance(value, str):
            raise ContainerGitEnvironmentError(
                "container environment keys and values must be strings"
            )
        if not _CONTAINER_ENV_NAME.fullmatch(key):
            # This also rejects Bash's exported-function encoding
            # (``BASH_FUNC_name%%``) and option-shaped env assignments.
            continue
        result[key] = value
    return result


def hermetic_container_git_environment(
    inherited: Mapping[str, str] | Iterable[str],
) -> dict[str, str]:
    """Return the shared safe environment for in-container Git authority.

    Non-dangerous image environment is retained so task runtimes keep their
    dependency/toolchain configuration. Ambient Git controls, shell startup
    injection, and exported Bash functions are discarded. Repository identity
    and Git configuration are deliberately *not* exported here: the trusted
    harness receives those only through a non-exported ``git()`` wrapper, so
    nested project tests retain normal Git semantics for their own temp repos.
    """

    source = _container_environment_mapping(inherited)
    env = {
        key: value
        for key, value in source.items()
        if not key.startswith("GIT_")
        and not key.startswith("BASH_FUNC_")
        and key not in _UNSAFE_SHELL_ENV
    }
    env["PATH"] = _hermetic_container_path(
        env.get("PATH", _DEFAULT_CONTAINER_PATH)
    )
    env["HOME"] = "/tmp/home"
    return env


def hermetic_container_git_exec_prefix(
    inherited: Mapping[str, str] | Iterable[str],
) -> list[str]:
    """Build an ``env -i`` argv prefix for a hermetic container command."""

    env = hermetic_container_git_environment(inherited)
    return [
        "/usr/bin/env",
        "-i",
        "--",
        *(f"{key}={env[key]}" for key in sorted(env)),
    ]


def hermetic_container_git_shell_function() -> str:
    """Return the non-exported Git authority wrapper used by B/C and tests."""

    assignments = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in _CONTAINER_GIT_COMMAND_ENV
    )
    command = [
        "/usr/bin/git",
        "--git-dir=/app/.git",
        "--work-tree=/app",
    ]
    for key, value in CONTAINER_GIT_CONFIG:
        command.extend(("-c", f"{key}={value}"))
    rendered_command = " ".join(shlex.quote(part) for part in command)
    return (
        "git() {\n"
        '    if [ "$#" -gt 0 ] && [ "$1" = "diff" ]; then\n'
        "        shift\n"
        '        set -- diff --no-ext-diff --no-textconv "$@"\n'
        "    fi\n"
        f'    {assignments} {rendered_command} "$@"\n'
        "}"
    )


def inspect_docker_container_environment(container_id: str) -> tuple[str, ...]:
    """Read a container's configured environment without executing in it."""

    if not isinstance(container_id, str) or not container_id.strip():
        raise ContainerGitEnvironmentError("container id must be non-empty")
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .Config.Env}}",
                "--",
                container_id,
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContainerGitEnvironmentError(
            f"cannot inspect container environment: {exc}"
        ) from exc
    if result.returncode != 0:
        raise ContainerGitEnvironmentError(
            "cannot inspect container environment: "
            f"{(result.stderr or '')[-1000:]}"
        )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContainerGitEnvironmentError(
            "container environment inspect returned invalid JSON"
        ) from exc
    if payload is None:
        return ()
    if not isinstance(payload, list) or not all(
        isinstance(entry, str) and "=" in entry for entry in payload
    ):
        raise ContainerGitEnvironmentError(
            "container environment inspect returned an invalid shape"
        )
    return tuple(payload)


def sanitized_git_env(**overrides: str) -> dict[str, str]:
    """Return a per-subprocess environment with ambient Git policy removed."""

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    })
    env.update(overrides)
    return env


def git_argv(*args: str) -> list[str]:
    """Build a Git argv with non-executable local facilities disabled."""

    return ["git", *GIT_SAFE_OPTIONS, *args]


def _validate_git_tree(git_dir: Path) -> None:
    """Reject links and special files anywhere in image-provided ``.git``."""

    try:
        root_stat = git_dir.lstat()
    except FileNotFoundError as exc:
        raise GitMetadataSafetyError(
            f"copied workspace has no .git directory: {git_dir}"
        ) from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise GitMetadataSafetyError(
            "copied workspace .git must be a real standalone directory"
        )

    stack = [git_dir]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    entry_stat = entry.stat(follow_symlinks=False)
                    mode = entry_stat.st_mode
                    if stat.S_ISLNK(mode):
                        raise GitMetadataSafetyError(
                            "image-derived .git contains a symlink: "
                            f"{Path(entry.path).relative_to(git_dir)}"
                        )
                    if stat.S_ISDIR(mode):
                        stack.append(Path(entry.path))
                    elif not stat.S_ISREG(mode):
                        raise GitMetadataSafetyError(
                            "image-derived .git contains a special file: "
                            f"{Path(entry.path).relative_to(git_dir)}"
                        )
        except GitMetadataSafetyError:
            raise
        except OSError as exc:
            raise GitMetadataSafetyError(
                f"cannot validate image-derived Git metadata: {exc}"
            ) from exc


def _reject_external_git_directories(git_dir: Path) -> None:
    """Reject Git controls that redirect metadata/object resolution."""

    forbidden = (
        git_dir / "commondir",
        git_dir / "gitdir",
        git_dir / "objects" / "info" / "alternates",
        git_dir / "objects" / "info" / "http-alternates",
    )
    for path in forbidden:
        if os.path.lexists(path):
            raise GitMetadataSafetyError(
                "image-derived checkout redirects Git storage through "
                f"{path.relative_to(git_dir)}"
            )


def _replace_local_config(git_dir: Path) -> None:
    """Atomically replace, rather than parse, untrusted local Git config."""

    config = git_dir / "config"
    if os.path.lexists(config):
        mode = config.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise GitMetadataSafetyError(
                "image-derived .git/config must be a regular file"
            )

    fd, temporary_name = tempfile.mkstemp(
        prefix=".crucible-safe-config-",
        dir=git_dir,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(_SAFE_GIT_CONFIG)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, config)
        try:
            directory_fd = os.open(git_dir, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def sanitize_copied_git_metadata(workspace: Path) -> Path:
    """Validate and neutralize copied Git metadata without invoking Git.

    This function must be the first operation after ``docker cp``.  It uses
    only no-follow filesystem operations, rejects external Git stores, and
    replaces local config before any host Git executable can read it.
    """

    workspace = Path(workspace)
    try:
        workspace_stat = workspace.lstat()
    except FileNotFoundError as exc:
        raise GitMetadataSafetyError(
            f"copied workspace does not exist: {workspace}"
        ) from exc
    if not stat.S_ISDIR(workspace_stat.st_mode):
        raise GitMetadataSafetyError(
            "copied workspace must be a real directory, not a link"
        )

    git_dir = workspace / ".git"
    _validate_git_tree(git_dir)
    _reject_external_git_directories(git_dir)
    _replace_local_config(git_dir)
    return git_dir


__all__ = [
    "CONTAINER_GIT_CONFIG",
    "ContainerGitEnvironmentError",
    "GIT_SAFE_OPTIONS",
    "GitMetadataSafetyError",
    "git_argv",
    "hermetic_container_git_environment",
    "hermetic_container_git_exec_prefix",
    "hermetic_container_git_shell_function",
    "inspect_docker_container_environment",
    "sanitize_copied_git_metadata",
    "sanitized_git_env",
]
