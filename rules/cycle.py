# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``cycle`` — TierDAG.depends_on must be acyclic.

Calls ``TierDAG.topo_order()`` which already enforces the invariant
internally and raises ``ValueError`` on cycle / unknown dependency. We
catch and translate into a single Critical Finding so that callers see
a structured verdict instead of an exception.
"""

from __future__ import annotations

from assay import Evidence, Finding, Spec


def cycle(spec: Spec) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    try:
        spec.tier_dag.topo_order()
    except ValueError as exc:
        return (
            Finding(
                id="cycle",
                severity="Critical",
                category="dag_structure",
                evidence=(
                    Evidence(
                        kind="dag_message",
                        location="tier_dag",
                        excerpt=str(exc),
                    ),
                ),
                suggested_action=(
                    "remove the dependency edge or restructure tiers so "
                    "the depends_on graph forms a DAG"
                ),
                source="deterministic",
            ),
        )
    return ()
