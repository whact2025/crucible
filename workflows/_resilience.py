# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Cross-workflow helpers for SDK transport-failure resilience.

Every assay primitive routes backend transport failures through a
structured-error path (``InvocationResult.error: TransportError``)
rather than raising. The primitives then map the error into typed
structured outputs:

- ``assay.primitives.implement`` → ``Implementation.classification
  == "hard_fail_transport"`` with an empty diff_ref.
- ``assay.primitives.review`` → ``Verdict(passed=False,
  findings=(Finding(id="backend_transport", ...),))``.
- ``assay.primitives.plan._planner_call`` → returns
  ``(None, transcript_ref)`` for that round.

This module is the single source of truth for detecting those
structured-failure shapes downstream, plus the canonical decisions
each workflow makes when it sees one:

- **breach Breaker** transport-failed Verdict → skip the round
  (don't run Defender against an empty attack set, don't ask the
  Judge to score nothing).
- **tier per-slice** transport-failed Implementation → mark the
  slice transport-skipped and continue the cascade to the next
  slice (instead of halting the whole tier).
- **Debate workflows** (goal / regression / scaling / summary /
  solutioning) transport-failed Verdict on Red / Blue / Judge →
  tag the trace; do not let the synthetic ``backend_transport``
  finding pollute the deliberation.

Every workflow's ``summary.json`` carries a
``transport_failures`` counter aggregated via
``count_transport_failures(...)`` so postmortem operators can spot
runs where SDK transport burps shaped the outcome.
"""

from __future__ import annotations

from typing import Protocol


# Legacy exact-string surfaces remain exported for callers that render them,
# but classification below uses Assay's typed, non-forgeable Finding fields.
# Chunk/reducer review phases intentionally prepend their phase to the id
# (``chunk_backend_*`` / ``reducer_backend_*``), so an id prefix is not a
# complete transport classifier.
TRANSPORT_FAILURE_FINDING_ID_PREFIX = "backend_"
TRANSPORT_FAILURE_FINDING_ID = "backend_transport"
TRANSPORT_FAILURE_CLASSIFICATION = "hard_fail_transport"


class _HasFindings(Protocol):
    findings: tuple


class _HasClassification(Protocol):
    classification: str


def is_transport_failure_findings(findings) -> bool:
    """True iff every entry in the findings tuple is a synthetic
    backend-error finding — no real assertions were made.

    Match rule: every finding is a deterministic ``backend_error`` that is
    not a typed budget terminal. Assay stamps both fields after an
    ``InvocationResult.error``; model findings are always ``source='llm'``, so
    a reviewer cannot forge this classification. Empty findings → False.
    The typed rule covers direct, chunk, and reducer ids without relying on
    their different name prefixes.

    Used by debate workflows that hold the raw findings tuple on
    a per-round state dataclass rather than the full Verdict
    object. Workflows holding the Verdict should prefer
    :func:`is_transport_failure_verdict` for clarity.
    """
    if not findings:
        return False
    return all(
        getattr(f, "source", None) == "deterministic"
        and getattr(f, "category", None) == "backend_error"
        and not _is_budget_failure_finding(f)
        for f in findings
    )


def _is_budget_failure_finding(finding) -> bool:
    """Recognize Assay's typed budget terminals without trusting model data."""

    finding_id = getattr(finding, "id", "")
    return bool(
        getattr(finding, "source", None) == "deterministic"
        and (
            getattr(finding, "category", None) == "budget_exhausted"
            or (
                isinstance(finding_id, str)
                and (
                    finding_id.endswith("budget_exhausted")
                    or finding_id.endswith("budget_overrun")
                )
            )
        )
    )


def is_budget_failure_findings(findings) -> bool:
    """True iff every finding is a deterministic budget terminal."""

    return bool(findings) and all(_is_budget_failure_finding(f) for f in findings)


def is_budget_failure_verdict(verdict: _HasFindings) -> bool:
    """True iff a Verdict contains only deterministic budget terminals."""

    return is_budget_failure_findings(
        getattr(verdict, "findings", ()) or (),
    )


def is_transport_failure_verdict(verdict: _HasFindings) -> bool:
    """True iff every finding in the Verdict is a synthetic
    backend-error finding — no real assertions were made.

    Thin wrapper around :func:`is_transport_failure_findings`
    that unwraps ``verdict.findings``.
    """
    return is_transport_failure_findings(
        getattr(verdict, "findings", ()) or (),
    )


def is_transport_failure_implementation(impl: _HasClassification) -> bool:
    """True iff the Implementation came from a transport-failed
    backend call (``assay.primitives.implement._classify`` mapped
    ``TransportError`` → ``hard_fail_transport``).
    """
    return getattr(impl, "classification", "") == TRANSPORT_FAILURE_CLASSIFICATION


__all__ = [
    "TRANSPORT_FAILURE_CLASSIFICATION",
    "TRANSPORT_FAILURE_FINDING_ID",
    "TRANSPORT_FAILURE_FINDING_ID_PREFIX",
    "is_budget_failure_findings",
    "is_budget_failure_verdict",
    "is_transport_failure_findings",
    "is_transport_failure_implementation",
    "is_transport_failure_verdict",
]
