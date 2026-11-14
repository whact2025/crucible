# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``creates_declaration`` — intent's "create X" must match creates_paths.

When a directive's intent text mentions creating a path-shaped token
(``foo/bar.py``, ``module.rs``, etc.) the path MUST also appear in the
directive's ``creates_paths`` field. Otherwise the implementer is free
to create files outside the declared scope and downstream
``creates_paths`` validation (cross-slice conflict detection) silently
misses the new files.

The check is heuristic: it scans the intent for tokens that look like
file paths (extension match + at least one slash OR root-level
identifier-with-extension) AND for verbs that signal creation. Both
must co-occur in a window of ~80 chars to fire.
"""

from __future__ import annotations

import re

from assay import Evidence, Finding, Spec

# Path-shaped tokens — relatively permissive but anchored on a known
# extension to avoid matching arbitrary identifiers. Multi-char
# extensions are listed before their single-char prefixes (``cpp``
# before ``c``, ``hpp`` before ``h``, ``tsx`` before ``ts``) AND a
# trailing ``(?![A-Za-z0-9])`` boundary requires the extension to be
# complete. Without both, Python's ordered alternation matches ``c``
# first inside ``Foo.cpp`` and captures ``Foo.c`` — a token that never
# equals the declared ``.cpp`` path, producing a phantom
# ``undeclared_create``.
_PATH_RE = re.compile(
    r"`?([A-Za-z0-9_./-]+\.(?:cpp|hpp|tsx|yaml|json|yml|py|rs|md|txt|sh|ts|go|c|h)"
    r"(?![A-Za-z0-9]))`?"
)

# Creation verbs, matched at WORD boundaries with optional regular
# inflections (``creates``/``created``/``adding``/…). A plain substring
# test would fire on ``write`` inside ``rewrite`` / ``rewritten`` —
# modification language — mis-flagging edits of existing files as
# undeclared creates. The ``\b`` before the stem is what excludes
# ``rewrite``; the boundary after the optional suffix excludes
# ``address`` (``add``) and similar.
_CREATION_VERB_RE = re.compile(
    r"\b(?:create|add|write|author|land|scaffold|introduce|new)(?:s|d|ed|ing)?\b"
)


def _intent_creates_paths(intent: str) -> list[tuple[str, str]]:
    """Return ``[(path_token, surrounding_excerpt)]`` for likely creations."""
    out: list[tuple[str, str]] = []
    text = intent.lower()
    for m in _PATH_RE.finditer(intent):
        path = m.group(1)
        start = max(0, m.start() - 40)
        end = min(len(intent), m.end() + 40)
        window = text[start:end]
        if _CREATION_VERB_RE.search(window):
            out.append((path, intent[start:end]))
    return out


def creates_declaration(spec: Spec) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    findings: list[Finding] = []
    for tier in spec.tier_dag.tiers:
        for s in tier.slices:
            declared = set(s.directive.creates_paths or ())
            # Intent prose routinely names a bare filename
            # (``DebugNameInternPool.h``) while ``creates_paths`` holds
            # the full repo-relative path. Match on basename too so a
            # correctly-declared create isn't flagged just because the
            # intent omitted its directory.
            declared_basenames = {p.rsplit("/", 1)[-1] for p in declared}
            for path, excerpt in _intent_creates_paths(s.directive.intent or ""):
                if path in declared or path.rsplit("/", 1)[-1] in declared_basenames:
                    continue
                findings.append(
                    Finding(
                        id=f"undeclared_create_{s.id}_{path}",
                        severity="Medium",
                        category="creates_declaration",
                        evidence=(
                            Evidence(
                                kind="intent_excerpt",
                                location=(
                                    f"tier_dag[{tier.tier_id}]"
                                    f".slices[{s.id}].directive.intent"
                                ),
                                excerpt=excerpt,
                            ),
                        ),
                        suggested_action=(
                            f"add {path!r} to directive.creates_paths "
                            f"or remove the creation language from intent"
                        ),
                        source="deterministic",
                    )
                )
    return tuple(findings)
