# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``creates_paths`` — no two slices may claim the same path as a create.

Across the whole TierDAG, ``creates_paths`` entries must be unique.
Two slices declaring identical ``creates_paths`` will collide on first
write — one will fail or silently overwrite the other.

This is stricter than ``parallel_tier_conflicts`` (which only checks
within a single tier) because a slice that creates a file in tier N
followed by a slice that ALSO creates the same file in tier N+1 is
nearly always a bug: the second slice should have ``files_in_scope``
the path (i.e., edit existing), not re-create it.
"""

from __future__ import annotations

from assay import Evidence, Finding, Spec


def creates_paths(spec: Spec) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    findings: list[Finding] = []
    owners: dict[str, list[tuple[str, str]]] = {}
    for tier in spec.tier_dag.tiers:
        for s in tier.slices:
            for p in s.directive.creates_paths or ():
                owners.setdefault(p, []).append((tier.tier_id, s.id))
    for path, locations in owners.items():
        if len(locations) < 2:
            continue
        findings.append(
            Finding(
                id=f"duplicate_create_{path}",
                severity="High",
                category="creates_paths",
                evidence=(
                    Evidence(
                        kind="duplicate_create",
                        location="tier_dag",
                        excerpt=(
                            f"{path} declared in creates_paths by "
                            f"{locations}"
                        ),
                    ),
                ),
                suggested_action=(
                    "remove the duplicate creates_paths entry — the "
                    "second slice should edit the file via files_in_scope"
                ),
                source="deterministic",
            )
        )
    return tuple(findings)
