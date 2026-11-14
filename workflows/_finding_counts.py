# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared fail-conservative finding count projections for workflow output."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


_ADVISORY_SEVERITIES = frozenset({"medium", "low"})


def is_blocking_finding(finding: Any) -> bool:
    """Classify unknown or malformed severities as blocking."""

    severity = getattr(finding, "severity", None)
    if not isinstance(severity, str):
        return True
    return severity.strip().lower() not in _ADVISORY_SEVERITIES


def blocking_findings(findings: Iterable[Any]) -> tuple[Any, ...]:
    """Return blocking findings under the shared conservative classifier."""

    return tuple(finding for finding in findings if is_blocking_finding(finding))


def finding_counts(findings: Iterable[Any]) -> dict[str, int]:
    """Project total, blocking, and advisory counts in one stable shape."""

    retained = tuple(findings)
    blocking = sum(is_blocking_finding(finding) for finding in retained)
    return {
        "total": len(retained),
        "blocking": blocking,
        "advisory": len(retained) - blocking,
    }


__all__ = ["blocking_findings", "finding_counts", "is_blocking_finding"]
