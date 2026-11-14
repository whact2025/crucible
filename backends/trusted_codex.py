# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Deterministic identity for Codex-only trusted repository context."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


TRUSTED_CODEX_CONTEXT_SCHEMA = "crucible.trusted-codex-context.v1"
_MAX_ASSET_BYTES = 256 * 1024


def _contained_path(root: Path, path: Path) -> Path:
    root_resolved = root.resolve(strict=True)
    path_resolved = path.resolve(strict=True)
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"trusted Codex asset escapes repository root: {path}"
        ) from exc
    if path.is_symlink() or not path_resolved.is_file():
        raise ValueError(
            f"trusted Codex asset must be a non-symlink regular file: {path}"
        )
    return path_resolved


def _asset_record(root: Path, path: Path) -> dict[str, str | int]:
    resolved = _contained_path(root, path)
    payload = resolved.read_bytes()
    if len(payload) > _MAX_ASSET_BYTES:
        raise ValueError(
            f"trusted Codex asset exceeds {_MAX_ASSET_BYTES} bytes: {path}"
        )
    return {
        "path": resolved.relative_to(root.resolve(strict=True)).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def trusted_codex_asset_paths(repository_root: Path | str) -> tuple[Path, ...]:
    """Return the closed project asset inventory for interactive Codex use."""

    root = Path(repository_root).resolve(strict=True)
    required = root / "AGENTS.md"
    if not required.is_file() or required.is_symlink():
        raise ValueError(
            f"trusted Codex root instructions are missing or unsafe: {required}"
        )
    paths = [required]
    paths.extend(sorted((root / ".codex" / "agents").glob("*.toml")))
    paths.extend(
        sorted((root / ".codex" / "skills").glob("*/SKILL.md"))
    )
    paths.extend(
        sorted((root / ".codex" / "skills").glob("*/agents/openai.yaml"))
    )
    return tuple(paths)


def effective_project_instruction_paths(
    repository_root: Path | str,
    cwd: Path | str,
) -> tuple[Path, ...]:
    """Mirror Codex's root-to-cwd AGENTS override precedence.

    Trusted mode intentionally does not honor configurable fallback filenames.
    Its isolated Codex home has no global instructions, so the repository
    chain returned here is the complete native instruction authority.
    """

    root = Path(repository_root).resolve(strict=True)
    current = Path(cwd).resolve(strict=True)
    try:
        relative = current.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"trusted Codex cwd {current} is outside repository root {root}"
        ) from exc
    directories = [root]
    cursor = root
    for part in relative.parts:
        cursor /= part
        directories.append(cursor)
    selected: list[Path] = []
    for directory in directories:
        override = directory / "AGENTS.override.md"
        ordinary = directory / "AGENTS.md"
        candidate = (
            override
            if override.exists()
            else ordinary
            if ordinary.exists()
            else None
        )
        if candidate is not None:
            _contained_path(root, candidate)
            selected.append(candidate)
    return tuple(selected)


def trusted_codex_context_digest(
    repository_root: Path | str,
    cwd: Path | str,
) -> str:
    """Hash every enabled trusted Codex instruction, agent, and skill asset."""

    root = Path(repository_root).resolve(strict=True)
    assets = trusted_codex_asset_paths(root)
    instructions = effective_project_instruction_paths(root, cwd)
    payload = {
        "schema_version": TRUSTED_CODEX_CONTEXT_SCHEMA,
        "assets": [_asset_record(root, path) for path in assets],
        "effective_project_instructions": [
            _asset_record(root, path) for path in instructions
        ],
        "cwd": Path(cwd).resolve(strict=True).relative_to(root).as_posix(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "TRUSTED_CODEX_CONTEXT_SCHEMA",
    "effective_project_instruction_paths",
    "trusted_codex_asset_paths",
    "trusted_codex_context_digest",
]
