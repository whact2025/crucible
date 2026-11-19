# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared utilities used across crucible subpackages."""

from __future__ import annotations

import collections
from pathlib import Path


def read_tail(path: Path, n: int = 250) -> str:
    """Return the last *n* lines of *path* as a single string.

    Returns an empty string on any I/O or decoding failure.
    Uses a bounded deque to avoid reading the entire file into memory.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            tail = collections.deque(f, maxlen=n)
        return "".join(tail)
    except Exception:
        return ""


def elide_middle(
    text: str,
    *,
    head_chars: int = 5_000,
    tail_chars: int = 5_000,
    marker_prefix: str = "[... ",
    marker_suffix: str = " characters elided ...]",
) -> str:
    """Return *text* with its middle replaced by a visible elision marker.

    When ``len(text) <= head_chars + tail_chars`` the text is returned
    unchanged. Otherwise the result is the first ``head_chars`` chars,
    a centered marker stating how many characters were elided, and the
    last ``tail_chars`` chars.

    A hard tail-only truncation policy hides relevant context (for example, a
    compile error
    buried under hundreds of deprecation warnings) and leaves the
    agent no signal to recover. With the gap size visible, the agent
    can pivot on the next turn to retrieve the elided region.

    **Counting unit — characters, not bytes.** ``len(text)`` and the
    marker count are Python codepoints; slicing is codepoint-based.
    For ASCII-clean outputs (gcc, cargo, pytest) chars and
    bytes are identical, so ``head -c N`` / ``tail -c N`` recovery
    works as expected. For unicode-bearing outputs (German docstrings,
    LaTeX, sphinx unicode IDs) byte offsets diverge from the marker's
    character count — agents recovering from non-ASCII surfaces
    should prefer line-oriented retrieval (``head -n`` / ``tail -n``
    / ``sed -n 'X,Yp'``) against the persisted full text (see
    Recovery prerequisite below). Note that the helper slices on a
    codepoint boundary, not a line boundary, so the head slice can
    end mid-line; the line count of the elided result is the line
    count of the head slice plus the line count of the tail slice,
    and a line straddling the head/elision boundary appears truncated
    in the visible output. The marker itself is on its own paragraph
    (surrounded by blank lines) so it's never mistaken for partial
    content.

    **Recovery prerequisite.** The elision marker advertises a gap
    size, which only becomes actionable if the caller has persisted
    the full pre-elision text somewhere the agent can later read
    (e.g., a sibling log file written by the upstream runner). At
    sites where the original text is *only* materialized through
    this helper, the marker is informational — the agent cannot
    retrieve the elided bytes. Surface adopters should ensure the
    full text lands in a sibling log if recovery is part of the
    intended UX.

    Cap defaults of 5000+5000 = 10 KB total visible payload match the
    mini-swe-agent template's tunables. Override per-call when a
    surface has different needs (build output benefits from more tail
    so the linker error survives; read-large-file benefits from more
    head so the file's preamble lands).

    ``head_chars`` and ``tail_chars`` must each be ``>= 0``; a value
    of zero collapses that side and is valid (e.g., ``tail_chars=0``
    produces head-only truncation with a trailing elision marker).
    """
    if head_chars < 0 or tail_chars < 0:
        raise ValueError(
            f"head_chars and tail_chars must be >= 0 "
            f"(got head={head_chars}, tail={tail_chars})"
        )
    n = len(text)
    visible = head_chars + tail_chars
    if n <= visible:
        return text
    elided = n - visible
    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars else ""
    marker = f"\n\n{marker_prefix}{elided}{marker_suffix}\n\n"
    return head + marker + tail
