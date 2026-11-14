# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Validate that an implementer's diff stays inside source-level marker regions.

Deterministic enforcement layer for the
``ImplementationDirective.mutation_regions`` surface: parses each
directive's regions, locates the named start/end marker pairs in the
file at the implementer's base commit, and emits a Finding for every
hunk that escapes its declared region.

Lives in ``crucible/workflows/`` (not ``crucible/rules/``) because the
check needs to read the file at the implementer's base commit — a
git subprocess call — and the rules contract forbids I/O (per
``crucible/CLAUDE.md`` § rules invariant). Workflow helpers are the
analog of rules for the I/O-permitted layer.

The check is opt-in twice:

- A directive without ``mutation_regions`` skips entirely (no
  constraint at all).
- A file listed in mutation_regions whose pre-commit content has no
  marker pair skips silently — markers are per-file opt-in. This
  lets a slice author flag a region intent in the directive without
  committing to enforcement until the markers actually land in the
  source tree.

Both behaviors are tested.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Iterable

from assay import Evidence, Finding, Implementation
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, MutationRegion

logger = logging.getLogger(__name__)


# Hunk header: ``@@ -<old_start>(,<old_len>)? +<new_start>(,<new_len>)? @@``
_HUNK_RE = re.compile(
    rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M,
)

# Per-file headers in unified-diff format.
_FILE_HEADER_RE = re.compile(rb"^\+\+\+ (.+?)(?:\t.*)?$", re.M)


# Extension → line-comment prefix. Block-comment-only languages
# (HTML, plain markdown) are intentionally absent; for those, the
# helper falls back to "no marker prefix" and skips the file
# silently rather than emit false positives.
_COMMENT_PREFIX: dict[str, str] = {
    # C-family
    ".c": "//", ".cc": "//", ".cpp": "//", ".cxx": "//", ".h": "//",
    ".hpp": "//", ".hxx": "//", ".cu": "//", ".cuh": "//",
    ".rs": "//",
    ".js": "//", ".ts": "//", ".jsx": "//", ".tsx": "//",
    ".go": "//", ".java": "//", ".kt": "//", ".swift": "//",
    # Shell / Python / config-family
    ".py": "#", ".pyi": "#",
    ".sh": "#", ".bash": "#", ".zsh": "#",
    ".yaml": "#", ".yml": "#", ".toml": "#", ".ini": "#", ".cfg": "#",
    ".rb": "#", ".pl": "#",
    ".dockerfile": "#",
    # Lua / SQL
    ".lua": "--", ".sql": "--",
}


def _comment_prefix(file: str) -> str | None:
    for ext, prefix in _COMMENT_PREFIX.items():
        if file.endswith(ext):
            return prefix
    return None


def _marker_ranges(
    text: bytes, prefix: str, start_marker: str, end_marker: str,
) -> tuple[tuple[int, int], ...]:
    """Locate (start_line, end_line) pairs for one marker shape.

    Both lines are 1-based and INCLUSIVE — i.e., a START marker on
    line 10 and an END marker on line 20 yields ``(10, 20)``. An
    unclosed START (no matching END before EOF) is dropped silently.
    """
    start_str = f"{prefix} {start_marker}".encode("utf-8")
    end_str = f"{prefix} {end_marker}".encode("utf-8")
    pairs: list[tuple[int, int]] = []
    current_start: int | None = None
    for i, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.lstrip()
        if stripped.startswith(start_str):
            current_start = i
        elif current_start is not None and stripped.startswith(end_str):
            pairs.append((current_start, i))
            current_start = None
    return tuple(pairs)


def _file_at_base_commit(
    file: str, *, base_commit: str, cwd: str,
) -> bytes | None:
    """Return file content at the base commit, or None if unavailable.

    Uses ``git show <base_commit>:<file>``. Returns None when the file
    didn't exist at base_commit (typical for ``creates_paths``
    targets) — the caller treats that as "no markers to enforce".
    """
    # ``file`` is planner-authored (a slice's region path). A ':' would
    # corrupt git's ``<rev>:<path>`` parse (e.g. ``sha:a:b`` reparses the
    # revision); a leading '-' or a NUL is likewise not a clean
    # repo-relative pathspec. Refuse rather than feed git an ambiguous
    # arg — the caller treats None as "no markers to enforce" (CRU-08).
    if ":" in file or file.startswith("-") or "\x00" in file:
        logger.warning(
            "mutation_scope: refusing unsafe region path %r at %s "
            "(contains ':' / leading '-' / NUL — not a clean pathspec)",
            file, base_commit,
        )
        return None
    try:
        result = subprocess.run(
            ["git", "show", f"{base_commit}:{file}"],
            cwd=cwd,
            capture_output=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(
            "mutation_scope: git show failed for %s@%s: %s",
            file, base_commit, e,
        )
        return None
    if result.returncode != 0:
        # File didn't exist at base_commit, or commit isn't accessible.
        return None
    return result.stdout


def _hunks_for_file(
    diff_bytes: bytes, target_file: str,
) -> tuple[tuple[int, int], ...]:
    """Extract (old_start, old_end_inclusive) ranges for hunks affecting
    ``target_file``. Both indices are 1-based.

    For pure insertions (``old_len == 0``) the hunk header carries
    ``-N,0`` meaning "insert before line N+1 / context line N"; we
    treat the affected range as ``(N, N)`` so the in-range check
    requires line N to fall within a marker pair.
    """
    out: list[tuple[int, int]] = []
    # Walk the diff line-by-line tracking the current ``+++`` target.
    in_target = False
    target_bytes = target_file.encode("utf-8")
    # Strip leading "b/" that git uses on ``+++`` lines.
    for raw in diff_bytes.split(b"\n"):
        if raw.startswith(b"+++ "):
            tail = raw[4:]
            tab = tail.find(b"\t")
            if tab != -1:
                tail = tail[:tab].rstrip()
            if tail.startswith(b"b/"):
                tail = tail[2:]
            in_target = (tail == target_bytes)
            continue
        if not in_target:
            continue
        m = _HUNK_RE.match(raw)
        if not m:
            continue
        old_start = int(m.group(1))
        old_len = int(m.group(2) or 1)
        if old_len == 0:
            out.append((old_start, old_start))
        else:
            out.append((old_start, old_start + old_len - 1))
    return tuple(out)


def _in_any_range(
    rng: tuple[int, int], ranges: Iterable[tuple[int, int]],
) -> bool:
    s, e = rng
    for ms, me in ranges:
        if s >= ms and e <= me:
            return True
    return False


def check_mutation_scope(
    directive: ImplementationDirective,
    impl: Implementation,
    *,
    corpus: Corpus,
    cwd: str,
) -> tuple[Finding, ...]:
    """Return Findings for diff hunks that extend outside the
    directive's declared mutation regions.

    Empty tuple = passed (either no constraint or all hunks inside
    markers). Severity is High per the doc's recommendation —
    mutation-scope violations should halt a tier but not nuke the
    Spec entirely (which would be Critical).

    Never raises. Subprocess / parse errors degrade to "no findings"
    so the helper can't itself become a blocker.
    """
    if not directive.mutation_regions:
        return ()
    try:
        diff_bytes = corpus.get_artifact(impl.diff_ref)
    except (FileNotFoundError, OSError) as e:
        logger.warning(
            "mutation_scope: cannot load diff_ref %s: %s", impl.diff_ref, e,
        )
        return ()

    findings: list[Finding] = []
    for region in directive.mutation_regions:
        prefix = _comment_prefix(region.file)
        if prefix is None:
            # Block-comment-only language or unknown extension; can't
            # parse markers reliably — opt-out.
            continue
        file_content = _file_at_base_commit(
            region.file, base_commit=impl.base_commit, cwd=cwd,
        )
        if file_content is None:
            # File didn't exist at base_commit (e.g., creates_paths
            # target) → no markers to enforce against. Skip silently.
            continue
        marker_ranges = _marker_ranges(
            file_content, prefix, region.start_marker, region.end_marker,
        )
        if not marker_ranges:
            # File doesn't carry the marker pair → per-file opt-in
            # means absent markers = no constraint.
            continue
        hunks = _hunks_for_file(diff_bytes, region.file)
        for h_start, h_end in hunks:
            if _in_any_range((h_start, h_end), marker_ranges):
                continue
            findings.append(
                Finding(
                    id=f"mutation_scope.{impl.slice_id}.{region.file}.{h_start}",
                    severity="High",
                    category="mutation_scope",
                    evidence=(
                        Evidence(
                            kind="diff_hunk",
                            location=f"{region.file}:{h_start}-{h_end}",
                            excerpt=(
                                f"hunk at {region.file}:{h_start}-{h_end} "
                                f"is outside marker pairs "
                                f"{list(marker_ranges)}"
                            ),
                        ),
                    ),
                    suggested_action=(
                        f"Hunk at {region.file}:{h_start} edits lines "
                        f"outside the declared mutation_region "
                        f"({region.start_marker} / {region.end_marker}). "
                        f"Move the edit inside a marker block, or expand "
                        f"mutation_regions / files_in_scope to include "
                        f"this region intentionally."
                    ),
                    source="deterministic",
                )
            )
    return tuple(findings)


def render_violations(findings: tuple[Finding, ...]) -> str:
    """Compact text summary suitable for log_excerpt / corpus artifact."""
    if not findings:
        return ""
    lines = [f"mutation_scope: {len(findings)} violation(s):"]
    for f in findings:
        for ev in f.evidence:
            lines.append(f"  - {ev.location}")
    return "\n".join(lines)


__all__ = [
    "check_mutation_scope",
    "render_violations",
]
