# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``ungrounded_citation`` — backtick-quoted paths in intent must be in scope.

A directive's intent often cites concrete files (``​``​foo/bar.py``)``. Every
such path token MUST appear in ``files_in_scope`` ∪ ``creates_paths``,
otherwise the implementer is asked to reason about a file it cannot
touch (or the cite is a phantom — the file doesn't exist at all).

This rule is intentionally intra-spec; filesystem existence belongs to the
context-aware ``cited_path_exists`` rule.

Severity is Medium because the implementer can often recover from citation
drift by reading the declared scope.
"""

from __future__ import annotations

import re

from assay import Evidence, Finding, Spec

# Match `path/with.ext` — must include an extension and be inside backticks.
_BACKTICK_PATH_RE = re.compile(
    r"`([A-Za-z0-9_./-]+\.(?:py|rs|yaml|yml|md|txt|json|sh|c|cpp|h|hpp|ts|tsx|go))`"
)


def ungrounded_citation(spec: Spec) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    findings: list[Finding] = []
    for tier in spec.tier_dag.tiers:
        for s in tier.slices:
            scope = set(s.directive.files_in_scope or ())
            creates = set(s.directive.creates_paths or ())
            covered = scope | creates
            for m in _BACKTICK_PATH_RE.finditer(s.directive.intent or ""):
                path = m.group(1)
                if path in covered:
                    continue
                # Allow citations that are a prefix-match of any covered
                # path — e.g. mentioning `foo/bar` when scope has
                # `foo/bar.py` is the same file.
                if any(c.endswith("/" + path) or c == path for c in covered):
                    continue
                findings.append(
                    Finding(
                        id=f"ungrounded_cite_{s.id}_{path}",
                        severity="Medium",
                        category="ungrounded_citation",
                        evidence=(
                            Evidence(
                                kind="intent_cite",
                                location=(
                                    f"tier_dag[{tier.tier_id}]"
                                    f".slices[{s.id}].directive.intent"
                                ),
                                excerpt=f"`{path}`",
                            ),
                        ),
                        suggested_action=(
                            f"add {path!r} to files_in_scope or "
                            f"creates_paths, or remove the citation"
                        ),
                        source="deterministic",
                    )
                )
    return tuple(findings)
