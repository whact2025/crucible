# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``implementer_verify`` — directive.verify_mode must be a known value.

Allowed values match the architecture's canonical enumeration:
- ``skip``        : no verification (docs-only slices, prose Specs)
- ``build_only``  : compile/typecheck; no test execution
- ``tests``       : compile + run the test suite
- ``bench``       : compile + run the benchmark suite

A blank, ``None``, or unknown ``verify_mode`` is a High-severity finding
— the recipe layer reads this string to decide which runner subprocess
to launch, so a typo silently skips verification.
"""

from __future__ import annotations

from assay import Evidence, Finding, Spec

_ALLOWED = frozenset({"skip", "build_only", "tests", "bench"})


def implementer_verify(spec: Spec) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    findings: list[Finding] = []
    for tier in spec.tier_dag.tiers:
        for s in tier.slices:
            mode = s.directive.verify_mode
            if mode in _ALLOWED:
                continue
            findings.append(
                Finding(
                    id=f"verify_mode_{s.id}",
                    severity="High",
                    category="verify_mode",
                    evidence=(
                        Evidence(
                            kind="directive_field",
                            location=f"tier_dag[{tier.tier_id}].slices[{s.id}].directive.verify_mode",
                            excerpt=str(mode),
                        ),
                    ),
                    suggested_action=(
                        f"set verify_mode to one of "
                        f"{sorted(_ALLOWED)}; got {mode!r}"
                    ),
                    source="deterministic",
                )
            )
    return tuple(findings)
