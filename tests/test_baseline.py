# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows._baseline``.

Covers ``latest_measurement_for`` (corpus walk + bench_id filter +
optional commit pin + timestamp-max selection) and ``regressed_by``
(shared-metric relative-delta comparison against threshold).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assay import (
    ContentHash,
    ExecutionContext,
    Measurement,
    ProjectContext,
    Spec,
)
from assay.corpus.store import Corpus

from crucible.workflows._baseline import (
    format_deltas,
    latest_measurement_for,
    regressed_by,
)


def _exec_at(ts: str) -> ExecutionContext:
    return ExecutionContext(
        timestamp=ts, machine="testhost", cwd="/tmp",
        toolchain={}, backend=None, model=None,
        recipe=None, recipe_commit=None, gpu_model=None, run_id=None,
    )


def _bare_spec(corpus: Corpus) -> Spec:
    spec = Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="t", repo_url="x", base_commit="abc",
            branch=None, submodule_path=None, runners={},
        ),
        execution=None, intent="", constraints=(), done_criteria=(),
        plan_policy=None,
    )
    corpus.put_spec(spec)
    return spec


def _put_measurement(
    corpus: Corpus, bench_id: str, commit: str, ts: str,
    values: dict[str, float],
) -> ContentHash:
    raw = corpus.put_artifact(b"raw", kind="bench_raw", name_hint="r.json")
    m = Measurement(
        bench_id=bench_id, commit=commit,
        config_hash="cfg", values=values,
        raw_output_ref=raw, execution=_exec_at(ts),
    )
    return corpus.put_record(m, kind="measurements")


def test_no_measurements_returns_none(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    assert latest_measurement_for(corpus, "any_bench") is None


def test_latest_by_timestamp(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    _put_measurement(
        corpus, "bvh", "aaa", "2026-05-01T00:00:00+00:00", {"time_us": 100.0},
    )
    _put_measurement(
        corpus, "bvh", "bbb", "2026-05-15T00:00:00+00:00", {"time_us": 110.0},
    )
    _put_measurement(
        corpus, "bvh", "ccc", "2026-05-10T00:00:00+00:00", {"time_us": 105.0},
    )
    latest = latest_measurement_for(corpus, "bvh")
    assert latest is not None
    assert latest.commit == "bbb"
    assert latest.values["time_us"] == 110.0


def test_bench_id_filter(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    _put_measurement(
        corpus, "bvh", "aaa", "2026-05-01T00:00:00+00:00", {"time_us": 100.0},
    )
    _put_measurement(
        corpus, "render", "bbb", "2026-05-15T00:00:00+00:00",
        {"time_ms": 5.0},
    )
    bvh = latest_measurement_for(corpus, "bvh")
    render = latest_measurement_for(corpus, "render")
    nothing = latest_measurement_for(corpus, "no_such_bench")
    assert bvh.commit == "aaa"
    assert render.commit == "bbb"
    assert nothing is None


def test_commit_filter_pins(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    _put_measurement(
        corpus, "bvh", "aaa", "2026-05-01T00:00:00+00:00", {"time_us": 100.0},
    )
    _put_measurement(
        corpus, "bvh", "bbb", "2026-05-15T00:00:00+00:00", {"time_us": 110.0},
    )
    pinned = latest_measurement_for(corpus, "bvh", commit_filter="aaa")
    assert pinned.commit == "aaa"
    assert pinned.values["time_us"] == 100.0


def test_regressed_by_below_threshold(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    base_ref = _put_measurement(
        corpus, "b", "aaa", "2026-05-01T00:00:00+00:00", {"time_us": 100.0},
    )
    cur_ref = _put_measurement(
        corpus, "b", "bbb", "2026-05-15T00:00:00+00:00", {"time_us": 105.0},
    )
    base = corpus.get_record(base_ref, as_type=Measurement)
    cur = corpus.get_record(cur_ref, as_type=Measurement)
    regressed, deltas = regressed_by(
        cur, base, threshold=0.20,
        directions={"time_us": "lower_is_better"},
    )
    assert not regressed
    assert abs(deltas["time_us"] - 0.05) < 1e-6  # +5%


def test_regressed_by_above_threshold(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    base_ref = _put_measurement(
        corpus, "b", "aaa", "2026-05-01T00:00:00+00:00", {"time_us": 100.0},
    )
    cur_ref = _put_measurement(
        corpus, "b", "bbb", "2026-05-15T00:00:00+00:00", {"time_us": 150.0},
    )
    base = corpus.get_record(base_ref, as_type=Measurement)
    cur = corpus.get_record(cur_ref, as_type=Measurement)
    regressed, deltas = regressed_by(
        cur, base, threshold=0.20,
        directions={"time_us": "lower_is_better"},
    )
    assert regressed
    assert deltas["time_us"] > 0.20


def test_regressed_by_shared_metrics_only(tmp_path: Path):
    """A metric in baseline that's missing from current is ignored —
    smoke is a coarse gate; strict-shape contracts belong elsewhere."""
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    base_ref = _put_measurement(
        corpus, "b", "aaa", "2026-05-01T00:00:00+00:00",
        {"time_us": 100.0, "memory_mb": 50.0},
    )
    cur_ref = _put_measurement(
        corpus, "b", "bbb", "2026-05-15T00:00:00+00:00",
        {"time_us": 90.0},
    )
    base = corpus.get_record(base_ref, as_type=Measurement)
    cur = corpus.get_record(cur_ref, as_type=Measurement)
    regressed, deltas = regressed_by(
        cur, base, threshold=0.20,
        directions={"time_us": "lower_is_better"},
    )
    assert not regressed
    assert "time_us" in deltas
    assert "memory_mb" not in deltas


def test_regressed_by_div_zero_guard(tmp_path: Path):
    """Baseline zero is an infinite regression when a lower-is-better
    metric rises above zero."""
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    base_ref = _put_measurement(
        corpus, "b", "aaa", "2026-05-01T00:00:00+00:00", {"errors": 0.0},
    )
    cur_ref = _put_measurement(
        corpus, "b", "bbb", "2026-05-15T00:00:00+00:00", {"errors": 7.0},
    )
    base = corpus.get_record(base_ref, as_type=Measurement)
    cur = corpus.get_record(cur_ref, as_type=Measurement)
    regressed, deltas = regressed_by(
        cur, base, threshold=0.20,
        directions={"errors": "lower_is_better"},
    )
    assert regressed
    assert deltas["errors"] == float("inf")


def test_regressed_by_higher_is_better(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    base_ref = _put_measurement(
        corpus, "b", "aaa", "2026-05-01T00:00:00+00:00", {"fps": 100.0},
    )
    cur_ref = _put_measurement(
        corpus, "b", "bbb", "2026-05-15T00:00:00+00:00", {"fps": 70.0},
    )
    base = corpus.get_record(base_ref, as_type=Measurement)
    cur = corpus.get_record(cur_ref, as_type=Measurement)
    regressed, deltas = regressed_by(
        cur, base, threshold=0.20,
        directions={"fps": "higher_is_better"},
    )
    assert regressed
    assert abs(deltas["fps"] - 0.3) < 1e-9


def test_regressed_by_requires_direction(tmp_path: Path):
    import pytest

    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    base_ref = _put_measurement(
        corpus, "b", "aaa", "2026-05-01T00:00:00+00:00", {"fps": 100.0},
    )
    cur_ref = _put_measurement(
        corpus, "b", "bbb", "2026-05-15T00:00:00+00:00", {"fps": 90.0},
    )
    with pytest.raises(ValueError, match="direction"):
        regressed_by(
            corpus.get_record(cur_ref, as_type=Measurement),
            corpus.get_record(base_ref, as_type=Measurement),
            threshold=0.20,
            directions={},
        )


def test_format_deltas_empty():
    assert format_deltas({}) == "(no shared metrics)"


def test_format_deltas_renders_percentages():
    out = format_deltas({"time_us": 0.25, "memory_mb": -0.05})
    assert "+25.00%" in out
    assert "-5.00%" in out


def test_format_deltas_renders_inf():
    out = format_deltas({"errors": float("inf")})
    assert "∞" in out
