# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared deterministic validation for Planner candidates before paid review."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from assay.primitives.validate import validate as assay_validate
from assay.types.findings import Finding
from assay.types.rule_context import RuleContext
from assay import ValidationRuleset

from crucible.rules import available as available_rules


def build_pre_review_candidate_validator(
    *,
    spec: Any,
    project_root: str | Path,
) -> Callable[[Any], tuple[Finding, ...]]:
    """Return the shared Critical/High executable-plan candidate gate.

    Historical Implementation refs are intentionally excluded. A Planner
    candidate describes the next executable DAG against ``project_root``;
    retained postimage reconciliation remains the owning workflow's later
    no-transport TOCTOU boundary.
    """

    root = Path(project_root).resolve()
    ruleset = ValidationRuleset(
        name="pre_review_candidate_gate",
        rule_ids=available_rules(),
    )
    rule_ctx = RuleContext(
        workspace_root=root,
        submodule_path=None,
    )

    def validate_candidate(candidate_dag: Any) -> tuple[Finding, ...]:
        candidate_spec = replace(
            spec,
            tier_dag=candidate_dag,
            implementation_refs=(),
        )
        verdict = assay_validate(
            candidate_spec,
            ruleset=ruleset,
            ctx=rule_ctx,
        )
        return tuple(
            finding
            for finding in verdict.findings
            if finding.severity in ("Critical", "High")
        )

    return validate_candidate


__all__ = ["build_pre_review_candidate_validator"]
