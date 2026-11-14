# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared workflow finding counts remain explicit and fail conservative."""

from __future__ import annotations

from types import SimpleNamespace

from crucible.workflows._finding_counts import (
    blocking_findings,
    finding_counts,
    is_blocking_finding,
)


def _finding(severity: object = "Low") -> SimpleNamespace:
    return SimpleNamespace(severity=severity)


def test_finding_counts_separate_blocking_and_advisory_severities():
    findings = tuple(
        _finding(severity)
        for severity in (
            "Critical",
            "High",
            "Medium",
            "Low",
            " medium ",
            "low",
        )
    )

    assert finding_counts(findings) == {
        "total": 6,
        "blocking": 2,
        "advisory": 4,
    }
    assert blocking_findings(findings) == findings[:2]


def test_unknown_missing_and_malformed_severities_fail_conservative():
    findings = (
        _finding("Notice"),
        _finding(None),
        SimpleNamespace(),
    )

    assert all(is_blocking_finding(finding) for finding in findings)
    assert finding_counts(findings) == {
        "total": 3,
        "blocking": 3,
        "advisory": 0,
    }


def test_finding_counts_accept_one_shot_iterables():
    findings = (_finding(severity) for severity in ("Low", "High"))

    assert finding_counts(findings) == {
        "total": 2,
        "blocking": 1,
        "advisory": 1,
    }
