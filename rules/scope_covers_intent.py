# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``scope_covers_intent`` — context-aware: slice intent must not name
real source files outside its declared ``files_in_scope``/``creates_paths``.

The planner sometimes writes slice prose like "in foo.cpp do X; in
bar.h do Y" while listing only foo.cpp in ``files_in_scope``. The
implementer dutifully edits both files; the orchestrator's
scope-filtered diff capture stores only the in-scope file; rollback
leaves bar.h modified in the working tree as an orphan; the next
slice's verification sees inconsistent baseline state. This rule
catches the explicit mis-slice at plan-validate time by scanning the
slice intent for multi-component source paths and asserting they are
either in scope, or genuinely don't refer to a real file on disk.

What this rule catches:
- Intent text "modify rendering/source/foo.cpp" with
  ``files_in_scope=("bar.h",)`` and foo.cpp exists on disk.
What this rule does NOT catch:
- Implicit cross-file coupling like "declare ``void f() override``"
  which silently requires the base class header — the intent text
  doesn't name the base header, so a regex scan can't see it. Those
  cases need a critic-round file-coverage check or implementer-time
  scope enforcement; this rule is the cheap deterministic floor.

Context-aware (arity 2): the rule verifies that the referenced path
actually exists on disk before flagging, so a typo'd path in the
intent doesn't double-report against the phantom_reference rule.
``ctx.workspace_root + ctx.submodule_path`` anchors the resolution.

Free-form claims such as "context only" or "not to be edited" do not
suppress a finding. The mutation boundary cannot safely depend on
interpreting nearby prose. A real path token must be declared in mutation
scope or removed from the intent entirely.
"""

from __future__ import annotations

import re
from pathlib import Path

from assay import Evidence, Finding, RuleContext, Spec


# Multi-component source path: at least one '/' before a recognized
# source-code extension. The leading char must be word-like (not `.`),
# so we don't match relative paths like ``./foo.cpp`` as ``./foo`` —
# they're matched via the trailing ``\.<ext>`` group from the word-char.
# Excludes markdown / data / build-config (.md, .yaml, .toml, etc.) on
# purpose: those are usually informational citations in intent, not
# mutation targets.
_PATH_RE = re.compile(
    r"\b([\w][\w./-]*?/[\w./-]*?"
    r"\.(?:h|hpp|hh|hxx|cpp|cc|cxx|c|cu|cuh|"
    r"hlsl|hlsli|slang|slangh|inl|"
    r"py|rs|js|jsx|ts|tsx|go|java|kt|swift))\b"
)
# The trailing ``\b`` ends the match at any non-word boundary, so
# sentence punctuation (``.`` ``,`` ``;`` ``)``) never makes it into
# the captured group — the regex stops one char before. The extraction
# below therefore does no further normalization.


def _extract_referenced_paths(intent: str) -> set[str]:
    if not intent:
        return set()
    return {m.group(1) for m in _PATH_RE.finditer(intent)}


def _scope_omits_referenced_path_finding(
    *,
    slice_id: str,
    tier_id: str,
    referenced: str,
    scope: tuple[str, ...],
    creates: tuple[str, ...],
) -> Finding:
    return Finding(
        id=f"scope_omits_referenced_path__{slice_id}__{referenced}",
        severity="Critical",
        category="scope_omits_referenced_path",
        evidence=(
            Evidence(
                kind="intent_text_scan",
                location=f"{tier_id}.{slice_id}.intent",
                excerpt=(
                    f"intent references '{referenced}' (verified to exist "
                    f"on disk) but it is not in files_in_scope="
                    f"{list(scope)} or creates_paths={list(creates)}"
                ),
            ),
        ),
        suggested_action=(
            f"Slice '{slice_id}' intent explicitly names '{referenced}', "
            "but files_in_scope does not authorize that path. Either add "
            f"'{referenced}' to files_in_scope (and raise "
            "blast_radius.max_files if needed) when the slice may edit it, "
            f"or remove the explicit '{referenced}' path token from the "
            "intent entirely and describe any read-only context without "
            "naming that file. Merely labeling a named path 'context only' "
            "or 'not to be edited' does not satisfy this deterministic "
            "contract. Leaving the explicit out-of-scope path in the intent "
            "can cause the implementer to edit outside scope and the "
            "orchestrator's diff capture to drop the orphan, leaving the "
            "working tree inconsistent across slice boundaries."
        ),
        source="deterministic",
    )


def scope_covers_intent(spec: Spec, ctx: RuleContext) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    if ctx is None or ctx.workspace_root is None:
        return ()

    base = Path(ctx.workspace_root)
    if ctx.submodule_path:
        base = base / ctx.submodule_path

    findings: list[Finding] = []
    for tier in spec.tier_dag.tiers:
        for slice_ in tier.slices:
            tier_id = tier.tier_id
            d = slice_.directive
            scope = tuple(d.files_in_scope or ())
            creates = tuple(d.creates_paths or ())
            in_scope = set(scope) | set(creates)

            for ref in sorted(_extract_referenced_paths(d.intent or "")):
                if ref in in_scope:
                    continue
                # Only flag paths that genuinely exist on disk — if the
                # path is a typo or hallucination, phantom_reference /
                # cited_path_exists own that signal; this rule strictly
                # catches the "real file referenced, not in scope" case.
                if not (base / ref).exists():
                    continue
                findings.append(
                    _scope_omits_referenced_path_finding(
                        slice_id=d.slice_id,
                        tier_id=tier_id,
                        referenced=ref,
                        scope=scope,
                        creates=creates,
                    )
                )

    return tuple(findings)
