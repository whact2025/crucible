# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``parallel_tier_conflicts`` — sibling slices must not share scope files.

Sibling slices represent independent work within a tier. Overlapping
``files_in_scope`` or ``creates_paths`` declarations contradict that plan
structure and obscure dependency order, even though the current shared-checkout
Tier executor mutates and gates slices serially.

The rule emits High-severity plan-quality evidence. It does not control runtime
parallelism or trigger a serial-execution fallback.
"""

from __future__ import annotations

from assay import Evidence, Finding, Spec


def parallel_tier_conflicts(spec: Spec) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    findings: list[Finding] = []
    for tier in spec.tier_dag.tiers:
        if len(tier.slices) < 2:
            continue
        # Build path → list[slice_id] for this tier.
        owners: dict[str, list[str]] = {}
        for s in tier.slices:
            for p in (*s.directive.files_in_scope, *s.directive.creates_paths):
                owners.setdefault(p, []).append(s.id)
        for path, slice_ids in owners.items():
            if len(slice_ids) < 2:
                continue
            findings.append(
                Finding(
                    id=f"parallel_conflict_{tier.tier_id}_{path}",
                    severity="High",
                    category="parallel_tier_conflict",
                    evidence=(
                        Evidence(
                            kind="shared_path",
                            location=f"tier_dag[{tier.tier_id}]",
                            excerpt=(
                                f"{path} claimed by slices "
                                f"{sorted(set(slice_ids))}"
                            ),
                        ),
                    ),
                    suggested_action=(
                        "move conflicting slices to different tiers via "
                        "depends_on, or narrow files_in_scope so the "
                        "siblings touch disjoint paths"
                    ),
                    source="deterministic",
                )
            )
    return tuple(findings)
