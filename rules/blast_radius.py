# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``blast_radius`` — directive scope must respect Spec-level constraints.

Two checks:
1. If a ``BlastRadiusConstraint`` is declared on the Spec, every
   slice's ``files_in_scope`` ∪ ``creates_paths`` must be a subset of
   ``no_edits_outside`` (prefix-matched).
2. If ``directive.blast_radius`` declares numeric caps
   (``max_files``, ``max_lines``), the scope must not exceed them.
   ``max_lines`` is advisory at validate time (we can't measure lines
   without an implementation diff); we only enforce ``max_files``.
"""

from __future__ import annotations

from assay import BlastRadiusConstraint, Evidence, Finding, Spec


def _strip_project_prefix(path: str, project_prefix: str) -> str:
    """Drop a leading ``<project_prefix>/`` segment so a workspace-relative
    path (``fabric-usd/crates/foo.rs``) and a project-root-relative one
    (``crates/foo.rs``) compare equal. No-op when the prefix is absent/empty.
    Mirrors the canonicalization the planner now applies, but defends the
    validate gate against hand-authored specs that mix the two conventions."""
    p = path.replace("\\", "/").lstrip("/")
    if p.startswith("./"):
        p = p[2:]
    sp = project_prefix.replace("\\", "/").strip("/")
    if sp and p.startswith(sp + "/"):
        p = p[len(sp) + 1:]
    return p


def _within_any_prefix(
    path: str, prefixes: tuple[str, ...], project_prefix: str
) -> bool:
    npath = _strip_project_prefix(path, project_prefix)
    for p in prefixes:
        if p == "**":
            return True
        norm_p = _strip_project_prefix(p, project_prefix)
        # A constraint that is empty — literally "", or exactly the project
        # prefix (e.g. "fabric-usd/" under project "fabric-usd", normalizing to
        # "") — means "anywhere under the project root": match everything.
        if norm_p == "":
            return True
        if npath == norm_p or npath.startswith(norm_p.rstrip("/") + "/"):
            return True
    return False


def blast_radius(spec: Spec) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()

    constraint: BlastRadiusConstraint | None = None
    for c in spec.constraints:
        if isinstance(c, BlastRadiusConstraint):
            constraint = c
            break

    pc = getattr(spec, "project_context", None)
    project_prefix = (
        (getattr(pc, "submodule_path", None) or getattr(pc, "project", None) or "")
        if pc is not None
        else ""
    )

    findings: list[Finding] = []
    for tier in spec.tier_dag.tiers:
        for s in tier.slices:
            scope = tuple(
                (*s.directive.files_in_scope, *s.directive.creates_paths)
            )

            if constraint is not None and constraint.no_edits_outside:
                for path in scope:
                    if _within_any_prefix(
                        path, constraint.no_edits_outside, project_prefix
                    ):
                        continue
                    findings.append(
                        Finding(
                            id=f"out_of_blast_radius_{s.id}_{path}",
                            severity="High",
                            category="blast_radius",
                            evidence=(
                                Evidence(
                                    kind="directive_path",
                                    location=(
                                        f"tier_dag[{tier.tier_id}]"
                                        f".slices[{s.id}]"
                                    ),
                                    excerpt=path,
                                ),
                                Evidence(
                                    kind="constraint",
                                    location="spec.constraints.blast_radius",
                                    excerpt=str(
                                        constraint.no_edits_outside
                                    ),
                                ),
                            ),
                            suggested_action=(
                                f"add a prefix covering {path!r} to "
                                f"BlastRadiusConstraint.no_edits_outside, "
                                f"or remove it from the slice's scope"
                            ),
                            source="deterministic",
                        )
                    )

            br = s.directive.blast_radius or {}
            max_files = br.get("max_files")
            if isinstance(max_files, int) and max_files > 0:
                actual = len(set(scope))
                if actual > max_files:
                    findings.append(
                        Finding(
                            id=f"max_files_exceeded_{s.id}",
                            severity="Medium",
                            category="blast_radius",
                            evidence=(
                                Evidence(
                                    kind="scope_size",
                                    location=(
                                        f"tier_dag[{tier.tier_id}]"
                                        f".slices[{s.id}]"
                                    ),
                                    excerpt=(
                                        f"{actual} files in scope; "
                                        f"max_files={max_files}"
                                    ),
                                ),
                            ),
                            suggested_action=(
                                "split the slice, or raise max_files "
                                "after confirming the larger scope is "
                                "intentional"
                            ),
                            source="deterministic",
                        )
                    )
    return tuple(findings)
