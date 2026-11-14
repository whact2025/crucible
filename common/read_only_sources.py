# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Bounded source evidence for planning, without mutation authority.

References bind whole-file bytes and inclusive UTF-8 line excerpts. Captured
JSON is evidence, never a replacement for the root specification. A later
deterministic gate captures again to detect changes before plan publication.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from assay import TierDAG

MAX_REFERENCES = 64
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_EXCERPT_BYTES = 16 * 1024
MAX_CONTEXT_BYTES = 128 * 1024

_DENIED_COMPONENTS = frozenset({
    ".git", ".hg", ".svn", ".bzr", "_darcs", ".ssh", ".aws",
    ".codex", ".claude", ".env", "agents.md", "agents.override.md",
    "claude.md", "gemini.md", "skill.md",
})


class SourceReferenceError(ValueError):
    """A declared source cannot be used as verified planning evidence."""


def _parts(path: str) -> tuple[str, ...]:
    parts = tuple(path.split("/"))
    if (
        not path or path == "."
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in path
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
        or any(char in path for char in "*?[]")
        or any(
            part.casefold() in _DENIED_COMPONENTS
            or part.casefold().startswith(".env.")
            for part in parts
        )
    ):
        raise SourceReferenceError("source path is not an allowed project file")
    return parts


def _read_file(root: Path, relative: str, submodule_path: str | None) -> bytes:
    """Open each component without following links or blocking on FIFOs."""
    parts = (*(_parts(submodule_path) if submodule_path else ()), *_parts(relative))
    with ExitStack() as stack:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, directory)
        for part in parts[:-1]:
            directory = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            stack.callback(os.close, directory)
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory,
        )
        stack.callback(os.close, descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SourceReferenceError("source must be a regular unaliased file")
        if before.st_size > MAX_FILE_BYTES:
            raise SourceReferenceError("source exceeds the whole-file byte limit")
        chunks: list[bytes] = []
        remaining = MAX_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        blob = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(blob) > MAX_FILE_BYTES:
            raise SourceReferenceError("source exceeds the whole-file byte limit")
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise SourceReferenceError("source changed during capture")
        return blob


def _canonical_write_scope(path: str, *, project_name: str) -> str:
    # Read references use exact project-relative spelling. Reject ambiguous
    # write aliases instead of guessing whether tool path normalization will
    # strip a redundant project prefix in a later workspace state.
    if (
        not path
        or "\\" in path
        or PurePosixPath(path).is_absolute()
        or PurePosixPath(path).as_posix() != path
        or ".." in PurePosixPath(path).parts
        or path == project_name
        or path.startswith(project_name + "/")
    ):
        raise SourceReferenceError("source references require canonical project-relative write scopes")
    return path


def capture_read_only_sources(
    dag: TierDAG,
    *,
    workspace_root: Path | None,
    submodule_path: str | None = None,
) -> bytes:
    """Return a canonical manifest of verified excerpts, or fail closed.

    The aggregate bound includes JSON metadata/escaping. Repeated declarations
    count toward the bound and retain slice identity. A stable reference ID
    cannot denote different evidence anywhere in the candidate.
    """
    declarations = [
        (tier.tier_id, slice_, reference)
        for tier in dag.tiers
        for slice_ in tier.slices
        for reference in slice_.directive.read_only_references
    ]
    if len(declarations) > MAX_REFERENCES:
        raise SourceReferenceError("candidate exceeds the source-reference count limit")
    entries: list[dict[str, Any]] = []
    if declarations:
        if workspace_root is None:
            raise SourceReferenceError("source references require an explicit workspace")
        try:
            root = workspace_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SourceReferenceError("source workspace cannot be resolved") from exc
        identities: dict[str, dict[str, Any]] = {}
        cache: dict[str, bytes] = {}
        for tier_id, slice_, reference in declarations:
            declaration = asdict(reference)
            previous = identities.setdefault(reference.reference_id, declaration)
            if previous != declaration:
                raise SourceReferenceError("reference ID names conflicting source evidence")
            # A read-only field must never mask authority to modify that same
            # source, including an enclosing directory in the write scope.
            for write in (*slice_.directive.files_in_scope, *slice_.directive.creates_paths):
                write = _canonical_write_scope(
                    write, project_name=Path(submodule_path).name if submodule_path else root.name,
                )
                if reference.path == write or reference.path.startswith(write.rstrip("/") + "/"):
                    raise SourceReferenceError("read-only reference overlaps the slice write scope")
            if reference.path not in cache:
                try:
                    cache[reference.path] = _read_file(root, reference.path, submodule_path)
                except OSError as exc:
                    raise SourceReferenceError("source cannot be opened as a confined regular file") from exc
            blob = cache[reference.path]
            if "sha256:" + hashlib.sha256(blob).hexdigest() != reference.sha256:
                raise SourceReferenceError("source whole-file digest does not match")
            try:
                decoded = blob.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SourceReferenceError("source is not UTF-8 text") from exc
            if "\x00" in decoded:
                raise SourceReferenceError("source contains binary NUL bytes")
            # LF is the line delimiter; preserve CRLF and exact terminal bytes.
            parts = decoded.split("\n")
            lines = [line + "\n" for line in parts[:-1]]
            if parts[-1]:
                lines.append(parts[-1])
            if reference.end_line > len(lines):
                raise SourceReferenceError("source line range exceeds the file")
            excerpt = "".join(lines[reference.start_line - 1:reference.end_line])
            if len(excerpt.encode("utf-8")) > MAX_EXCERPT_BYTES:
                raise SourceReferenceError("source excerpt exceeds the byte limit")
            entries.append({
                "tier_id": tier_id,
                "slice_id": slice_.id,
                **declaration,
                "excerpt": excerpt,
            })
    captured = json.dumps(
        {"schema": "crucible.read-only-sources.v1", "references": entries},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    if len(captured) > MAX_CONTEXT_BYTES:
        raise SourceReferenceError("source context exceeds the aggregate byte limit")
    return captured


def source_context_provider(
    *, corpus, workspace_root: Path | None, submodule_path=None, retained_refs=None,
):
    """Capture once per review round and retain exactly the supplied bytes."""
    def provide(dag: TierDAG) -> dict[str, str]:
        captured = capture_read_only_sources(
            dag, workspace_root=workspace_root, submodule_path=submodule_path,
        )
        digest = corpus.put_artifact(captured, kind="plan-source-context")
        if retained_refs is not None:
            retained_refs.append(digest)
        return {"read_only_sources": f"Captured context {digest}\n{captured.decode('utf-8')}"}

    return provide
