# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Verify source evidence without adding its paths to mutation authority."""

from assay import Evidence, Finding, RuleContext, Spec

from crucible.common.read_only_sources import SourceReferenceError, capture_read_only_sources


def read_only_references(spec: Spec, ctx: RuleContext) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    try:
        capture_read_only_sources(
            spec.tier_dag,
            workspace_root=ctx.workspace_root,
            submodule_path=ctx.submodule_path,
        )
    except SourceReferenceError as exc:
        return (Finding(
            id="read_only_source_invalid",
            severity="High",
            category="read_only_source_invalid",
            evidence=(Evidence(
                kind="source_reference_validation",
                location="tier_dag.read_only_references",
                excerpt=str(exc),
            ),),
            suggested_action=(
                "Supply bounded existing source references with exact hashes and lines; "
                "keep their paths separate from the slice mutation scope."
            ),
            source="deterministic",
        ),)
    return ()
