# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows._resilience`` helpers.

The helpers are the single source of truth for detecting
SDK-transport-failure shapes in primitive outputs. Every workflow
that loops over backend invocations (breach Breaker/Defender/Judge,
tier per-slice implementer, debate workflows) reads from these
helpers to decide round-skip / slice-skip / trace-tagging behavior,
so the helpers themselves get pinned tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from crucible.workflows._resilience import (
    TRANSPORT_FAILURE_CLASSIFICATION,
    is_budget_failure_findings,
    is_budget_failure_verdict,
    is_transport_failure_findings,
    is_transport_failure_implementation,
    is_transport_failure_verdict,
)


@dataclass
class _FakeFinding:
    id: str
    category: str = "rubric_finding"
    source: str = "llm"


@dataclass
class _FakeVerdict:
    findings: tuple


@dataclass
class _FakeImpl:
    classification: str


# ----------------------------------------------------------------------
# is_transport_failure_verdict
# ----------------------------------------------------------------------


def test_verdict_with_no_findings_is_not_transport_failure():
    """Empty findings means the rubric passed (or hit some other
    path); not transport-shaped."""
    v = _FakeVerdict(findings=())
    assert is_transport_failure_verdict(v) is False


def test_verdict_with_backend_transport_finding_only_is_transport_failure():
    """The canonical case — ``assay.primitives.review`` returns
    exactly one synthetic ``backend_transport`` finding when its
    backend call surfaced a TransportError result."""
    v = _FakeVerdict(findings=(_FakeFinding(
        id="backend_transport",
        category="backend_error",
        source="deterministic",
    ),))
    assert is_transport_failure_verdict(v) is True


def test_verdict_with_backend_prefix_finding_is_transport_failure():
    """Any ``backend_<kind>`` finding counts — content_rejected,
    timeout, etc. share the synthetic shape from
    ``_backend_error_finding``."""
    v = _FakeVerdict(findings=(_FakeFinding(
        id="backend_content_rejected",
        category="backend_error",
        source="deterministic",
    ),))
    assert is_transport_failure_verdict(v) is True


@pytest.mark.parametrize(
    "finding_id",
    ["chunk_backend_transport", "reducer_backend_timeout"],
)
def test_chunk_and_reducer_backend_findings_are_transport_failures(finding_id):
    v = _FakeVerdict(findings=(_FakeFinding(
        id=finding_id,
        category="backend_error",
        source="deterministic",
    ),))

    assert is_transport_failure_verdict(v) is True


@pytest.mark.parametrize(
    "finding_id",
    [
        "backend_budget_exhausted",
        "chunk_backend_budget_exhausted",
        "reducer_backend_budget_overrun",
    ],
)
def test_budget_backend_findings_are_not_transport_failures(finding_id):
    v = _FakeVerdict(findings=(_FakeFinding(
        id=finding_id,
        category="backend_error",
        source="deterministic",
    ),))

    assert is_budget_failure_verdict(v) is True
    assert is_budget_failure_findings(v.findings) is True
    assert is_transport_failure_verdict(v) is False


def test_verdict_with_real_finding_is_not_transport_failure():
    """A real rubric-emitted finding means the rubric did its job —
    not transport-shaped."""
    v = _FakeVerdict(findings=(_FakeFinding(id="over_decomposed"),))
    assert is_transport_failure_verdict(v) is False


def test_verdict_with_mixed_backend_and_real_findings_is_not_transport_failure():
    """If even one real finding rides alongside the synthetic
    backend-error, the rubric produced some signal — don't treat as
    a pure transport failure."""
    v = _FakeVerdict(findings=(
        _FakeFinding(
            id="backend_transport",
            category="backend_error",
            source="deterministic",
        ),
        _FakeFinding(id="hallucinated_path"),
    ))
    assert is_transport_failure_verdict(v) is False


# ----------------------------------------------------------------------
# is_transport_failure_implementation
# ----------------------------------------------------------------------


def test_implementation_with_transport_classification_is_transport_failure():
    impl = _FakeImpl(classification=TRANSPORT_FAILURE_CLASSIFICATION)
    assert is_transport_failure_implementation(impl) is True


def test_implementation_with_other_hard_fail_is_not_transport_failure():
    impl = _FakeImpl(classification="hard_fail_max_turns")
    assert is_transport_failure_implementation(impl) is False


def test_implementation_with_success_classification_is_not_transport_failure():
    impl = _FakeImpl(classification="success")
    assert is_transport_failure_implementation(impl) is False


def test_implementation_with_soft_fail_is_not_transport_failure():
    impl = _FakeImpl(classification="soft_fail_no_changes")
    assert is_transport_failure_implementation(impl) is False


# ----------------------------------------------------------------------
# is_transport_failure_findings — direct (no Verdict wrapper)
# ----------------------------------------------------------------------


def test_findings_empty_tuple_is_not_transport_failure():
    """Empty tuple — no opinions either way, not transport."""
    assert is_transport_failure_findings(()) is False


def test_findings_with_single_backend_finding_is_transport_failure():
    """The canonical direct case for callers holding the raw
    findings tuple on a per-round state dataclass (debate
    workflows). Verdict-holding callers go through
    is_transport_failure_verdict."""
    findings = (_FakeFinding(
        id="backend_transport",
        category="backend_error",
        source="deterministic",
    ),)
    assert is_transport_failure_findings(findings) is True


def test_findings_with_only_real_findings_is_not_transport_failure():
    findings = (
        _FakeFinding(id="over_decomposed"),
        _FakeFinding(id="hallucinated_path"),
    )
    assert is_transport_failure_findings(findings) is False


def test_findings_mixed_real_and_backend_is_not_transport_failure():
    """The match is ALL findings must be backend-error — a single
    real finding means the rubric produced signal."""
    findings = (
        _FakeFinding(
            id="backend_transport",
            category="backend_error",
            source="deterministic",
        ),
        _FakeFinding(id="real_finding"),
    )
    assert is_transport_failure_findings(findings) is False


def test_findings_with_multiple_backend_kinds_is_transport_failure():
    """``backend_transport`` + ``backend_timeout`` + ... — every
    finding carries the synthetic prefix, so still transport-
    failure shaped from the workflow's perspective."""
    findings = (
        _FakeFinding(
            id="backend_transport",
            category="backend_error",
            source="deterministic",
        ),
        _FakeFinding(
            id="backend_timeout",
            category="backend_error",
            source="deterministic",
        ),
    )
    assert is_transport_failure_findings(findings) is True
