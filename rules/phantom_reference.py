# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``phantom_reference`` — cited symbols' files should appear in scope.

Scans each directive's intent for backtick-quoted *symbols* — callable
identifiers (``foo_bar()`` or ``MyClass``) — and flags those whose
likely-containing file is NOT in ``files_in_scope``. The heuristic is
weak by design (we can't grep without a workspace checkout), so
findings are Low severity — advisory hints, not blockers.

Severity is Low because the heuristic is not workspace-grounded and should not
gate a plan.

The emitted ``category="phantom_symbol"`` (not ``phantom_reference``,
which is the rule_id) deliberately disambiguates from the
filesystem-grounded ``cited_path_exists`` rule that emits Critical
``category="phantom_reference"`` for actual missing-file hallucinations.
Downstream consumers filtering by category alone can distinguish a
weak symbol-grep advisory from a hard missing-path failure.
"""

from __future__ import annotations

import re

from assay import Evidence, Finding, Spec

# Symbol-shape: starts with letter/underscore, optionally has a CamelCase
# or snake_case body. Excluded: tokens with dots (paths), digits-only
# (numbers), or all-uppercase short ones (most likely env-vars).
_BACKTICK_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{2,})(?:\(\))?`")

# Symbols that ALWAYS pass — built-ins, generic terms — even if the
# intent doesn't carry a containing file.
_ALLOWLIST = frozenset(
    {
        # Python built-ins / pseudo-types
        "None",
        "True",
        "False",
        "self",
        "cls",
        "dict",
        "list",
        "tuple",
        "set",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "Any",
        # Common assay/crucible types — known scope-neutral
        "Spec",
        "Verdict",
        "Finding",
        "Rubric",
        "Backend",
        "Corpus",
        # Generic protocol verbs the planner is allowed to use
        "TODO",
        "DONE",
        "init",
    }
)


def _looks_like_symbol(tok: str) -> bool:
    if tok in _ALLOWLIST:
        return False
    if tok.isupper() and len(tok) < 8:
        return False  # likely an env-var or short acronym
    return True


def _scope_has_symbol_hint(symbol: str, scope: set[str]) -> bool:
    """Heuristic: scope covers the symbol if any scoped file's stem matches.

    E.g., symbol ``RequestContext`` and scope ``{src/RequestContext.cpp}``
    — the stem matches, so we assume the symbol is in scope.
    """
    sym_lower = symbol.lower()
    for path in scope:
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if stem == sym_lower or sym_lower in stem or stem in sym_lower:
            return True
    return False


def phantom_reference(spec: Spec) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    findings: list[Finding] = []
    for tier in spec.tier_dag.tiers:
        for s in tier.slices:
            scope = set(s.directive.files_in_scope or ()) | set(
                s.directive.creates_paths or ()
            )
            seen: set[str] = set()
            for m in _BACKTICK_SYMBOL_RE.finditer(s.directive.intent or ""):
                sym = m.group(1)
                if not _looks_like_symbol(sym) or sym in seen:
                    continue
                seen.add(sym)
                if _scope_has_symbol_hint(sym, scope):
                    continue
                findings.append(
                    Finding(
                        id=f"phantom_ref_{s.id}_{sym}",
                        severity="Low",
                        category="phantom_symbol",
                        evidence=(
                            Evidence(
                                kind="intent_symbol",
                                location=(
                                    f"tier_dag[{tier.tier_id}]"
                                    f".slices[{s.id}].directive.intent"
                                ),
                                excerpt=f"`{sym}`",
                            ),
                        ),
                        suggested_action=(
                            f"add the file containing {sym!r} to "
                            f"files_in_scope, or rephrase the intent to "
                            f"avoid the bare-symbol reference"
                        ),
                        source="deterministic",
                    )
                )
    return tuple(findings)
