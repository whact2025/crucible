# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Locate baseline measurements + flag regression beyond a threshold.

The policy layer for the tier-level smoke-bench gate. The corpus
stores measurements as content-addressed records under
``artifacts/measurements/``; this helper walks them, filters by
``bench_id``, optionally pins to a commit, and returns the latest
matching ``Measurement`` for regression comparison.

Why crucible owns this and not assay: "what counts as the baseline
measurement" is a domain policy decision (latest-run vs.
commit-pinned vs. release-pinned vs. last-blessed). assay's corpus
stays narrow (list + get); crucible composes the policy on top.

Regression model: ``regressed_by(m, baseline, directions, threshold)``
returns True when ANY shared metric moves in its configured regression
direction by more than ``threshold`` (relative). Direction metadata is
mandatory: guessing that every KPI is a timing silently inverts
throughput/FPS gates.

Used by ``crucible/workflows/tier.py`` Phase 2.5 between the union
diff capture and the breach pass — a single-sample bench should
fail loudly before we burn budget on breach/review of a regressed
tier.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

from assay import Measurement
from assay.corpus.store import Corpus

logger = logging.getLogger(__name__)


def latest_measurement_for(
    corpus: Corpus,
    bench_id: str,
    *,
    commit_filter: str | None = None,
) -> Measurement | None:
    """Return the latest stored ``Measurement`` matching ``bench_id``,
    or None when nothing matches.

    Args:
      corpus: Corpus to scan.
      bench_id: Restrict to measurements whose ``bench_id`` matches
        exactly. (No prefix matching — bench identity is exact.)
      commit_filter: When non-None, also restrict to measurements
        whose ``commit`` field matches exactly. Useful for "compare
        against the named-baseline commit" workflows; pass None for
        "compare against the latest run regardless of commit".

    "Latest" is measured by ``execution.timestamp`` (ISO 8601 string,
    which sorts lexicographically when the timestamps are in the
    same fixed-width form ``YYYY-MM-DDTHH:MM:SS+ZZ:ZZ`` that
    ``ExecutionContext`` snapshots produce).
    """
    candidates = list(
        corpus.iter_measurements(bench_id=bench_id, commit=commit_filter)
    )
    if not candidates:
        return None
    return max(candidates, key=_measurement_timestamp)


def _measurement_timestamp(m: Measurement) -> str:
    return getattr(m.execution, "timestamp", "") or ""


def regressed_by(
    current: Measurement,
    baseline: Measurement,
    *,
    directions: Mapping[str, str],
    threshold: float = 0.20,
) -> tuple[bool, dict[str, float]]:
    """Compare ``current`` vs ``baseline``.

    Returns ``(regressed, deltas)`` where:
      - ``regressed`` is True iff any shared metric's direction-normalized
        relative delta exceeds ``threshold``.
      - ``deltas`` maps the metric name to its relative delta
        ``(current - baseline) / baseline`` for every shared metric.
        Useful for logging which metric drove the regression.

    Every shared metric must have an explicit ``higher_is_better`` or
    ``lower_is_better`` entry. An unknown metric/direction is a
    configuration error, not an implicit pass.
    """
    deltas: dict[str, float] = {}
    regressed = False
    for key, baseline_value in baseline.values.items():
        if key not in current.values:
            continue
        direction = directions.get(key)
        if direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError(
                f"metric {key!r} has no valid direction metadata; expected "
                "'higher_is_better' or 'lower_is_better'"
            )
        if baseline_value == 0:
            current_value = current.values[key]
            if current_value == 0:
                normalized_delta = 0.0
            else:
                # Relative change from zero is undefined. Treat movement in
                # the regression direction as an infinite regression and
                # movement in the improvement direction as -infinity.
                worsened = (
                    current_value < 0
                    if direction == "higher_is_better"
                    else current_value > 0
                )
                normalized_delta = float("inf") if worsened else float("-inf")
            deltas[key] = normalized_delta
            if normalized_delta > threshold:
                regressed = True
            continue
        raw_delta = (current.values[key] - baseline_value) / abs(baseline_value)
        normalized_delta = (
            -raw_delta if direction == "higher_is_better" else raw_delta
        )
        deltas[key] = normalized_delta
        if normalized_delta > threshold:
            regressed = True
    return regressed, deltas


def format_deltas(deltas: dict[str, float]) -> str:
    """Compact log-friendly rendering of the deltas dict."""
    if not deltas:
        return "(no shared metrics)"
    parts = []
    for k, v in sorted(deltas.items()):
        if v == float("inf"):
            parts.append(f"{k}=∞")
        else:
            parts.append(f"{k}={v:+.2%}")
    return ", ".join(parts)


__all__ = [
    "latest_measurement_for",
    "regressed_by",
    "format_deltas",
]
