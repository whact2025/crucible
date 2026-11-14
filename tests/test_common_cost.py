# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.common.cost.CostAccumulator``.

The accumulator restores per-role / per-backend / per-model cost
attribution that the pre-cutover ``crucible/common/cost.py`` provided
and the 2026-05-17 cutover deleted. Tests pin the contract so a
re-deletion or accidental schema change trips here.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from assay import Usage, UsagePricingEvidence, UsagePricingSegment
from crucible.common.cost import (
    CostAccumulator,
    CostLimitExceeded,
    PRICING_EVIDENCE_SCHEMA,
    parse_token_count,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (42, 42),
        (42.0, 42),
        (" 42 ", 42),
        (None, None),
        (True, None),
        (-1, None),
        (1.5, None),
        (float("nan"), None),
        (float("inf"), None),
        ("1.5", None),
        ([], None),
    ],
)
def test_parse_token_count_rejects_lossy_or_negative_values(value, expected):
    assert parse_token_count(value) == expected


# ----------------------------------------------------------------------
# Basic accounting
# ----------------------------------------------------------------------


def test_empty_accumulator_reports_zero():
    a = CostAccumulator()
    assert a.total_cost_usd == 0.0
    assert a.total_input_tokens == 0
    assert a.total_output_tokens == 0
    assert a.total_cache_creation_tokens == 0
    assert a.total_cache_read_tokens == 0
    assert a.pricing_evidence is None
    assert a.cost_by_role() == {}
    assert a.cost_by_backend() == {}
    assert a.cost_by_model() == {}


def test_single_record_attributes_to_role_and_backend():
    a = CostAccumulator()
    a.record(
        role="goal_red",
        usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.42),
        backend_name="claude",
        model="aws/anthropic/bedrock-claude-opus-4-8",
    )
    assert a.total_cost_usd == pytest.approx(0.42)
    assert a.total_input_tokens == 100
    assert a.total_output_tokens == 50
    assert a.cost_by_role() == {"goal_red": pytest.approx(0.42)}
    assert a.cost_by_backend() == {"claude": pytest.approx(0.42)}
    assert a.cost_by_model() == {"aws/anthropic/bedrock-claude-opus-4-8": pytest.approx(0.42)}
    assert a.calls_by_role() == {"goal_red": 1}


def test_multi_record_sums_correctly_per_dimension():
    """3 calls across 2 roles / 2 backends / 2 models; every breakdown
    sums independently."""
    a = CostAccumulator()
    a.record(
        role="goal_red",
        usage=Usage(input_tokens=100, output_tokens=50, cost_usd=1.0),
        backend_name="claude", model="aws/anthropic/bedrock-claude-opus-4-8",
    )
    a.record(
        role="goal_red",
        usage=Usage(input_tokens=200, output_tokens=80, cost_usd=2.0),
        backend_name="claude", model="aws/anthropic/bedrock-claude-opus-4-8",
    )
    a.record(
        role="goal_judge",
        usage=Usage(input_tokens=50, output_tokens=25, cost_usd=0.5),
        backend_name="codex", model="gpt-5.5",
    )

    assert a.total_cost_usd == pytest.approx(3.5)
    assert a.total_input_tokens == 350
    assert a.total_output_tokens == 155
    # by_role: red got 2 calls, judge got 1
    assert a.cost_by_role() == {
        "goal_red": pytest.approx(3.0),
        "goal_judge": pytest.approx(0.5),
    }
    assert a.calls_by_role() == {"goal_red": 2, "goal_judge": 1}
    # by_backend: claude = 1+2 = 3.0, codex = 0.5
    assert a.cost_by_backend() == {
        "claude": pytest.approx(3.0),
        "codex": pytest.approx(0.5),
    }
    # by_model: opus = 3.0, gpt = 0.5
    assert a.cost_by_model() == {
        "aws/anthropic/bedrock-claude-opus-4-8": pytest.approx(3.0),
        "gpt-5.5": pytest.approx(0.5),
    }


def test_record_handles_missing_optional_fields():
    """``model=None`` and zero token usage shouldn't blow up; missing
    ``cost_usd`` defaults to 0.0 — Local backends with no per-token
    billing report a zero-cost call."""
    a = CostAccumulator()
    a.record(
        role="rca",
        usage=Usage(),   # all zeros
        backend_name="local",
    )
    assert a.total_cost_usd == 0.0
    assert a.cost_by_backend() == {"local": 0.0}
    # model is None so by_model stays empty
    assert a.cost_by_model() == {}
    # but the role + backend are still recorded
    assert a.cost_by_role() == {"rca": 0.0}
    assert a.calls_by_role() == {"rca": 1}


def test_cache_tokens_aggregate_globally_and_per_role():
    accumulator = CostAccumulator()
    accumulator.record(
        role="planner",
        usage=Usage(
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.2,
            cache_creation_tokens=30,
            cache_read_tokens=20,
        ),
        backend_name="claude",
    )
    accumulator.record(
        role="planner",
        usage=Usage(
            input_tokens=40,
            output_tokens=5,
            cost_usd=0.1,
            cache_read_tokens=10,
        ),
        backend_name="claude",
    )
    accumulator.record(
        role="judge",
        usage=Usage(input_tokens=20, output_tokens=2, cost_usd=0.05),
        backend_name="codex",
    )

    assert accumulator.total_cache_creation_tokens == 30
    assert accumulator.total_cache_read_tokens == 30
    summary = accumulator.to_dict()
    assert summary["total_cache_creation_tokens"] == 30
    assert summary["total_cache_read_tokens"] == 30
    assert summary["by_role"]["planner"]["cache_creation_tokens"] == 30
    assert summary["by_role"]["planner"]["cache_read_tokens"] == 30
    assert "cache_creation_tokens" not in summary["by_role"]["judge"]
    assert "pricing_evidence" not in summary


def test_pricing_evidence_aggregates_in_call_and_segment_order():
    first_segment = UsagePricingSegment(
        input_tokens=100,
        output_tokens=10,
        cache_creation_tokens=20,
        cache_read_tokens=30,
        cost_usd=0.25,
    )
    second_segment = UsagePricingSegment(
        input_tokens=50,
        output_tokens=5,
        cache_creation_tokens=0,
        cache_read_tokens=10,
        cost_usd=0.75,
    )
    first_evidence = UsagePricingEvidence(
        segments=(first_segment,),
        complete=True,
    )
    second_evidence = UsagePricingEvidence(
        segments=(second_segment,),
        complete=True,
    )
    accumulator = CostAccumulator()
    accumulator.record(
        role="planner",
        usage=first_evidence.usage,
        backend_name="claude",
        model="opus",
        pricing_evidence=first_evidence,
    )
    # Workflow artifacts expose pricing_evidence but do not duplicate cache
    # counters at the top level. The accumulator derives those exact subsets
    # from the segments while preserving duck-typed legacy inputs.
    accumulator.record(
        role="judge",
        usage=SimpleNamespace(
            input_tokens=50,
            output_tokens=5,
            cost_usd=0.75,
            pricing_evidence=second_evidence,
        ),
        backend_name="codex",
        model="gpt",
    )

    aggregate = accumulator.pricing_evidence
    assert aggregate is not None
    assert aggregate.segments == (first_segment, second_segment)
    assert aggregate.complete is True
    assert aggregate.reconciles(
        Usage(
            input_tokens=150,
            output_tokens=15,
            cost_usd=1.0,
            cache_creation_tokens=20,
            cache_read_tokens=40,
        )
    )
    summary = accumulator.to_dict()
    assert summary["pricing_evidence_schema"] == PRICING_EVIDENCE_SCHEMA
    assert summary["pricing_evidence"] == {
        "segments": [
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_creation_tokens": 20,
                "cache_read_tokens": 30,
                "cost_usd": 0.25,
            },
            {
                "input_tokens": 50,
                "output_tokens": 5,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 10,
                "cost_usd": 0.75,
            },
        ],
        "complete": True,
    }
    calls = summary["pricing_evidence_by_call"]
    assert [call["role"] for call in calls] == ["planner", "judge"]
    assert [call["backend"] for call in calls] == ["claude", "codex"]
    assert [call["model"] for call in calls] == ["opus", "gpt"]
    assert calls[0]["usage"] == {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_creation_tokens": 20,
        "cache_read_tokens": 30,
        "cost_usd": 0.25,
    }
    assert calls[0]["pricing_evidence"]["complete"] is True
    assert calls[1]["pricing_evidence"]["segments"] == [
        summary["pricing_evidence"]["segments"][1]
    ]


def test_extended_pricing_summary_preserves_raw_cost_precision():
    segment = UsagePricingSegment(10, 2, 0, 2, 0.123456789)
    evidence = UsagePricingEvidence(segments=(segment,), complete=True)
    accumulator = CostAccumulator()
    accumulator.record(
        role="planner",
        usage=evidence.usage,
        backend_name="claude",
        model="opus",
        pricing_evidence=evidence,
    )

    summary = accumulator.to_dict()

    assert summary["total_usd"] == 0.123456789
    assert summary["by_role"]["planner"]["cost_usd"] == 0.123456789
    assert summary["by_backend"]["claude"] == 0.123456789
    assert summary["by_model"]["opus"] == 0.123456789
    assert summary["pricing_evidence_schema"] == PRICING_EVIDENCE_SCHEMA


def test_pricing_evidence_retains_known_segments_but_fails_closed():
    complete_segment = UsagePricingSegment(10, 2, 0, 2, 0.2)
    incomplete_segment = UsagePricingSegment(4, 1, 1, 0, 0.05)
    accumulator = CostAccumulator()
    # A legacy call before evidence is seen must keep the eventual aggregate
    # incomplete even if every later evidence-bearing call reconciles.
    accumulator.record(
        role="legacy",
        usage=Usage(input_tokens=5, output_tokens=1, cost_usd=0.1),
        backend_name="legacy",
    )
    accumulator.record(
        role="planner",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.2,
            pricing_evidence=UsagePricingEvidence(
                segments=(complete_segment,),
                complete=True,
            ),
        ),
        backend_name="claude",
    )
    accumulator.record(
        role="judge",
        usage=SimpleNamespace(
            input_tokens=4,
            output_tokens=1,
            cost_usd=0.05,
            pricing_evidence=UsagePricingEvidence(
                segments=(incomplete_segment,),
                complete=False,
            ),
        ),
        backend_name="codex",
    )
    accumulator.record(
        role="legacy_after",
        usage=Usage(input_tokens=2, output_tokens=1, cost_usd=0.01),
        backend_name="legacy",
    )

    aggregate = accumulator.pricing_evidence
    assert aggregate is not None
    assert aggregate.segments == (complete_segment, incomplete_segment)
    assert aggregate.complete is False
    summary = accumulator.to_dict()
    assert summary["pricing_evidence"]["complete"] is False
    calls = summary["pricing_evidence_by_call"]
    assert calls[0]["role"] == "legacy"
    assert calls[0]["pricing_evidence"] == {
        "segments": [],
        "complete": False,
    }
    assert calls[1]["pricing_evidence"]["complete"] is True
    assert calls[2]["pricing_evidence"]["complete"] is False
    assert calls[3]["role"] == "legacy_after"
    assert calls[3]["pricing_evidence"] == {
        "segments": [],
        "complete": False,
    }


def test_pricing_evidence_mismatch_or_rollup_debt_downgrades_completeness():
    segment = UsagePricingSegment(10, 2, 0, 2, 0.2)
    evidence = UsagePricingEvidence(segments=(segment,), complete=True)
    mismatched = CostAccumulator()
    mismatched.record(
        role="planner",
        usage=Usage(
            input_tokens=11,
            output_tokens=2,
            cost_usd=0.2,
            cache_read_tokens=2,
        ),
        backend_name="claude",
        pricing_evidence=evidence,
    )
    assert mismatched.pricing_evidence is not None
    assert mismatched.pricing_evidence.complete is False
    assert (
        mismatched.to_dict()["pricing_evidence_by_call"][0]
        ["pricing_evidence"]["complete"]
        is False
    )

    incomplete_rollup = CostAccumulator()
    incomplete_rollup.record(
        role="planner",
        usage=evidence.usage,
        backend_name="claude",
        pricing_evidence=evidence,
    )
    assert incomplete_rollup.pricing_evidence.complete is True
    incomplete_rollup.note_unloaded_ref()
    assert incomplete_rollup.pricing_evidence.complete is False


def test_legacy_none_cache_fields_defer_to_complete_pricing_evidence():
    """Legacy typed artifacts use None, while an exact new zero is integer 0."""

    segment = UsagePricingSegment(10, 2, 3, 4, 0.2)
    evidence = UsagePricingEvidence(segments=(segment,), complete=True)
    accumulator = CostAccumulator()

    accumulator.record(
        role="implementer",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.2,
            cache_creation_tokens=None,
            cache_read_tokens=None,
            pricing_evidence=evidence,
        ),
        backend_name="claude",
    )

    summary = accumulator.to_dict()
    assert summary["total_cache_creation_tokens"] == 3
    assert summary["total_cache_read_tokens"] == 4
    assert summary["pricing_evidence"]["complete"] is True


def test_explicit_zero_cache_fields_override_conflicting_complete_evidence():
    """An exact reported zero is not the same as a legacy missing field."""

    evidence = UsagePricingEvidence(
        segments=(UsagePricingSegment(10, 2, 3, 4, 0.2),),
        complete=True,
    )
    accumulator = CostAccumulator()

    accumulator.record(
        role="implementer",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.2,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            pricing_evidence=evidence,
        ),
        backend_name="claude",
    )

    summary = accumulator.to_dict()
    assert summary["total_cache_creation_tokens"] == 0
    assert summary["total_cache_read_tokens"] == 0
    assert summary["pricing_evidence"]["complete"] is False


def _final_b_incomplete_snapshot_evidence() -> UsagePricingEvidence:
    return UsagePricingEvidence(
        segments=(
            UsagePricingSegment(8_627, 2, 4_921, 3_704, 0.03266825),
            UsagePricingSegment(8_627, 2, 4_921, 3_704, 0.03266825),
            UsagePricingSegment(8_627, 2, 4_921, 3_704, 0.03266825),
            UsagePricingSegment(12_735, 2, 4_108, 8_625, 0.0300475),
            UsagePricingSegment(12_735, 2, 4_108, 8_625, 0.0300475),
            UsagePricingSegment(12_735, 2, 4_108, 8_625, 0.0300475),
            UsagePricingSegment(17_973, 3, 5_238, 12_733, 0.039189),
            UsagePricingSegment(17_973, 3, 5_238, 12_733, 0.039189),
            UsagePricingSegment(17_973, 3, 5_238, 12_733, 0.039189),
        ),
        complete=False,
    )


def test_incomplete_snapshot_ledger_does_not_supply_aggregate_cache_counters():
    """Partial SDK snapshots cannot be projected onto one terminal total."""

    snapshots = _final_b_incomplete_snapshot_evidence()
    accumulator = CostAccumulator()

    accumulator.record(
        role="implementer",
        usage=SimpleNamespace(
            input_tokens=39_335,
            output_tokens=1_730,
            cost_usd=0.14497975,
            accounting_debt=("token_usage_complete", "cost_complete"),
            pricing_evidence=snapshots,
        ),
        backend_name="claude",
        model="claude-opus-5",
    )

    summary = accumulator.to_dict()
    assert summary["total_usd"] == 0.14497975
    assert summary["total_input_tokens"] == 39_335
    assert summary["total_output_tokens"] == 1_730
    assert summary["total_cache_creation_tokens"] == 0
    assert summary["total_cache_read_tokens"] == 0
    assert summary["accounting_debt"] is True
    assert summary["accounting_debt_by_role"] == {
        "implementer": ["cost_complete", "token_usage_complete"],
    }
    assert summary["pricing_evidence"]["complete"] is False
    assert len(summary["pricing_evidence"]["segments"]) == 9
    call = summary["pricing_evidence_by_call"][0]
    assert call["usage"] == {
        "input_tokens": 39_335,
        "output_tokens": 1_730,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.14497975,
    }
    assert call["pricing_evidence"]["complete"] is False


def test_final_b_three_call_usage_retains_exact_terminal_spend_and_cache():
    """The failed final arm's third call remains a durable known total."""

    accumulator = CostAccumulator(max_cost_usd=11.04060975)
    complete_calls = (
        Usage(5_635_703, 51_629, 4.883054, 134_590, 5_500_983),
        Usage(917_756, 9_467, 1.03540175, 59_065, 858_641),
    )
    for usage in complete_calls:
        accumulator.record(
            role="implementer",
            usage=usage,
            backend_name="claude",
            model="claude-opus-5",
            pricing_evidence=UsagePricingEvidence(
                segments=(UsagePricingSegment(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_creation_tokens=usage.cache_creation_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cost_usd=usage.cost_usd,
                ),),
                complete=True,
            ),
        )

    accumulator.record(
        role="implementer",
        usage=SimpleNamespace(
            input_tokens=39_335,
            output_tokens=1_730,
            cache_creation_tokens=14_267,
            cache_read_tokens=25_062,
            cost_usd=0.14497975,
            accounting_debt=(),
            pricing_evidence=_final_b_incomplete_snapshot_evidence(),
        ),
        backend_name="claude",
        model="claude-opus-5",
    )

    summary = accumulator.to_dict()
    assert summary["total_usd"] == pytest.approx(6.0634355)
    assert summary["total_input_tokens"] == 6_592_794
    assert summary["total_output_tokens"] == 62_826
    assert summary["total_cache_creation_tokens"] == 207_922
    assert summary["total_cache_read_tokens"] == 6_384_686
    assert summary["by_role"]["implementer"]["calls"] == 3
    assert summary["accounting_debt"] is False
    assert summary["pricing_evidence"]["complete"] is False
    terminal_call = summary["pricing_evidence_by_call"][2]
    assert terminal_call["usage"] == {
        "input_tokens": 39_335,
        "output_tokens": 1_730,
        "cache_creation_tokens": 14_267,
        "cache_read_tokens": 25_062,
        "cost_usd": 0.14497975,
    }


@pytest.mark.parametrize(
    "usage",
    [
        Usage(input_tokens=10, cache_creation_tokens=-1),
        Usage(input_tokens=10, cache_creation_tokens=6, cache_read_tokens=5),
    ],
)
def test_invalid_cache_usage_is_rejected_without_mutation(usage):
    accumulator = CostAccumulator()

    with pytest.raises(ValueError, match="cache|token usage"):
        accumulator.record(
            role="untrusted_backend",
            usage=usage,
            backend_name="test",
        )

    assert accumulator.total_input_tokens == 0


def test_incomplete_evidence_cannot_mask_invalid_direct_cache_usage():
    accumulator = CostAccumulator()
    evidence = UsagePricingEvidence(
        segments=(UsagePricingSegment(10, 1, 3, 4, 0.1),),
        complete=False,
    )

    with pytest.raises(ValueError, match="cache-token subsets"):
        accumulator.record(
            role="untrusted_backend",
            usage=Usage(
                input_tokens=10,
                output_tokens=1,
                cost_usd=0.1,
                cache_creation_tokens=6,
                cache_read_tokens=5,
            ),
            backend_name="test",
            pricing_evidence=evidence,
        )

    assert accumulator.total_cost_usd == 0.0


@pytest.mark.parametrize(
    "identity",
    [
        {"role": "", "backend_name": "backend", "model": None},
        {"role": "role", "backend_name": "", "model": None},
        {"role": "role", "backend_name": "backend", "model": ""},
    ],
)
def test_invalid_cost_identity_is_rejected_without_mutation(identity):
    accumulator = CostAccumulator()

    with pytest.raises(ValueError, match="non-empty"):
        accumulator.record(**identity, usage=Usage(cost_usd=0.1))

    assert accumulator.total_cost_usd == 0.0
    assert accumulator.to_dict()["by_role"] == {}
    assert accumulator.total_cache_creation_tokens == 0
    assert accumulator.calls_by_role() == {}


def test_invalid_pricing_evidence_type_is_rejected_without_mutation():
    accumulator = CostAccumulator()

    with pytest.raises(TypeError, match="pricing_evidence"):
        accumulator.record(
            role="untrusted_backend",
            usage=Usage(input_tokens=1),
            backend_name="test",
            pricing_evidence=SimpleNamespace(),
        )

    assert accumulator.calls_by_role() == {}


@pytest.mark.parametrize("cap", [-1.0, float("nan"), float("inf")])
def test_invalid_budget_cap_is_rejected(cap):
    with pytest.raises(ValueError, match="finite, non-negative"):
        CostAccumulator(max_cost_usd=cap)


@pytest.mark.parametrize("cost", [-0.01, float("nan"), float("inf")])
def test_invalid_backend_cost_is_rejected_without_poisoning_totals(cost):
    accumulator = CostAccumulator(max_cost_usd=10.0)

    with pytest.raises(ValueError, match="cost_usd"):
        accumulator.record(
            role="untrusted_backend",
            usage=Usage(cost_usd=cost),
            backend_name="test",
        )

    assert accumulator.total_cost_usd == 0.0
    assert accumulator.calls_by_role() == {}


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [(-1, 0), (0, -1)],
)
def test_negative_token_usage_is_rejected(input_tokens, output_tokens):
    accumulator = CostAccumulator()

    with pytest.raises(ValueError, match="token usage"):
        accumulator.record(
            role="untrusted_backend",
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            backend_name="test",
        )

    assert accumulator.total_input_tokens == 0
    assert accumulator.total_output_tokens == 0


def test_record_mixed_backends_on_same_role_surfaces_both():
    """A role hitting multiple backends across calls (e.g. fallback
    from local → claude) shows BOTH backends in to_dict()['by_role']."""
    a = CostAccumulator()
    a.record(
        role="implementer", usage=Usage(cost_usd=0.0),
        backend_name="local", model="qwen2.5-coder-32b",
    )
    a.record(
        role="implementer", usage=Usage(cost_usd=0.50),
        backend_name="claude", model="aws/anthropic/bedrock-claude-opus-4-8",
    )
    blob = a.to_dict()
    impl_slot = blob["by_role"]["implementer"]
    assert sorted(impl_slot["backends"]) == ["claude", "local"]
    assert sorted(impl_slot["models"]) == [
        "aws/anthropic/bedrock-claude-opus-4-8", "qwen2.5-coder-32b",
    ]
    assert impl_slot["calls"] == 2


# ----------------------------------------------------------------------
# Budget enforcement
# ----------------------------------------------------------------------


def test_uncapped_accumulator_never_raises():
    a = CostAccumulator()   # no max_cost_usd
    for _ in range(100):
        a.record(
            role="x", usage=Usage(cost_usd=1.0), backend_name="claude",
        )
    assert a.total_cost_usd == pytest.approx(100.0)


def test_capped_accumulator_raises_on_overshoot():
    """The trigger call IS recorded before the exception fires —
    summary.json reflects accurate spend including the over-budget call."""
    a = CostAccumulator(max_cost_usd=1.0)
    a.record(role="r1", usage=Usage(cost_usd=0.4), backend_name="claude")
    a.record(role="r1", usage=Usage(cost_usd=0.4), backend_name="claude")
    # Third call pushes total to 1.2 — should raise.
    with pytest.raises(CostLimitExceeded, match=r"\$1\.2"):
        a.record(role="r1", usage=Usage(cost_usd=0.4), backend_name="claude")
    # The OVERFLOWING call was still recorded — total is past cap.
    assert a.total_cost_usd == pytest.approx(1.2)


def test_capped_does_not_raise_when_exactly_at_cap():
    """Exactly hitting the cap doesn't trip — only ``>`` does. Matches
    the pre-cutover behavior."""
    a = CostAccumulator(max_cost_usd=1.0)
    a.record(role="r", usage=Usage(cost_usd=1.0), backend_name="claude")
    # Total is now exactly 1.0. No raise on this call.
    assert a.total_cost_usd == 1.0


def test_remaining_budget_decrements():
    a = CostAccumulator(max_cost_usd=10.0)
    assert a.remaining() == 10.0
    a.record(role="r", usage=Usage(cost_usd=3.0), backend_name="claude")
    assert a.remaining() == pytest.approx(7.0)


def test_remaining_is_none_when_uncapped():
    a = CostAccumulator()
    assert a.remaining() is None


def test_remaining_clamps_to_zero_after_overshoot():
    a = CostAccumulator(max_cost_usd=1.0)
    a.record(role="r", usage=Usage(cost_usd=0.9), backend_name="claude")
    with pytest.raises(CostLimitExceeded):
        a.record(role="r", usage=Usage(cost_usd=2.0), backend_name="claude")
    # Total is now 2.9; remaining shouldn't go negative.
    assert a.remaining() == 0.0


# ----------------------------------------------------------------------
# JSON serialization
# ----------------------------------------------------------------------


def test_to_dict_shape_is_stable():
    """The dashboard + downstream summaries depend on this JSON shape.
    Pinning the top-level keys + nested shape catches accidental
    renames."""
    a = CostAccumulator(max_cost_usd=5.0)
    a.record(
        role="goal_red",
        usage=Usage(input_tokens=10, output_tokens=20, cost_usd=0.5),
        backend_name="claude", model="aws/anthropic/bedrock-claude-opus-4-8",
    )
    blob = a.to_dict()
    assert set(blob.keys()) == {
        "total_usd",
        "total_input_tokens",
        "total_output_tokens",
        "max_cost_usd",
        "remaining_usd",
        "by_role",
        "by_backend",
        "by_model",
        "unloaded_refs",
        "accounting_debt",
        "accounting_debt_by_role",
        "rollup_incomplete",
    }
    assert blob["total_usd"] == pytest.approx(0.5)
    assert blob["max_cost_usd"] == 5.0
    assert blob["remaining_usd"] == pytest.approx(4.5)
    # by_role slot shape
    slot = blob["by_role"]["goal_red"]
    assert set(slot.keys()) == {
        "cost_usd", "input_tokens", "output_tokens",
        "calls", "backends", "models",
    }
    assert slot["backends"] == ["claude"]
    assert slot["models"] == ["aws/anthropic/bedrock-claude-opus-4-8"]


def test_to_dict_uncapped_omits_remaining_via_null():
    a = CostAccumulator()
    blob = a.to_dict()
    assert blob["max_cost_usd"] is None
    assert blob["remaining_usd"] is None


# ----------------------------------------------------------------------
# Thread safety
# ----------------------------------------------------------------------


def test_record_is_thread_safe_under_contention():
    """N threads each recording M calls — final total must equal N*M
    exactly. A non-atomic increment would lose updates."""
    a = CostAccumulator()
    N_THREADS = 8
    M_CALLS = 50

    def _worker():
        for _ in range(M_CALLS):
            a.record(
                role="r", usage=Usage(cost_usd=0.01),
                backend_name="claude",
            )

    threads = [threading.Thread(target=_worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected_calls = N_THREADS * M_CALLS
    assert a.calls_by_role() == {"r": expected_calls}
    assert a.total_cost_usd == pytest.approx(expected_calls * 0.01)


def test_unloaded_refs_default_complete():
    """A fresh accumulator reports a complete rollup."""
    d = CostAccumulator().to_dict()
    assert d["unloaded_refs"] == 0
    assert d["accounting_debt"] is False
    assert d["accounting_debt_by_role"] == {}
    assert d["rollup_incomplete"] is False


def test_note_unloaded_ref_marks_rollup_incomplete():
    """note_unloaded_ref() surfaces the under-count signal in to_dict()
    (CRU-05a / ARC-2): architect + swebench cost rollups call this when
    a cost-bearing corpus record fails to load."""
    a = CostAccumulator()
    a.record(role="r", usage=Usage(cost_usd=0.01), backend_name="claude")
    a.note_unloaded_ref()
    a.note_unloaded_ref(2)
    d = a.to_dict()
    assert d["unloaded_refs"] == 3
    assert d["rollup_incomplete"] is True
    assert d["remaining_usd"] == 0.0
    assert a.remaining() == 0.0
    # The recorded total is unaffected — the flag only says it's a floor.
    assert d["total_usd"] == pytest.approx(0.01)

    capped = CostAccumulator(max_cost_usd=2.0)
    capped.record(role="r", usage=Usage(cost_usd=0.25), backend_name="claude")
    capped.note_unloaded_ref()
    assert capped.remaining() == 0.0
    assert capped.to_dict()["remaining_usd"] == 0.0


def test_note_unloaded_ref_is_thread_safe():
    a = CostAccumulator()
    threads = [threading.Thread(target=a.note_unloaded_ref) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert a.to_dict()["unloaded_refs"] == 40


def test_note_accounting_debt_preserves_numeric_floor_and_marks_rollup():
    accumulator = CostAccumulator()
    accumulator.record(
        role="impl_breaker",
        usage=Usage(
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.5,
        ),
        backend_name="claude",
    )

    accumulator.note_accounting_debt(
        role="impl_breaker",
        missing=("cost_complete", "token_usage_complete"),
    )

    summary = accumulator.to_dict()
    assert summary["total_usd"] == pytest.approx(0.5)
    assert summary["accounting_debt"] is True
    assert summary["accounting_debt_by_role"] == {
        "impl_breaker": ["cost_complete", "token_usage_complete"],
    }
    assert summary["rollup_incomplete"] is True
    assert summary["remaining_usd"] == 0.0
    assert accumulator.remaining() == 0.0

    uncapped = CostAccumulator()
    uncapped.note_accounting_debt(
        role="impl_breaker",
        missing=("cost_complete",),
    )
    assert uncapped.remaining() == 0.0
    assert uncapped.to_dict()["remaining_usd"] == 0.0


def test_record_discovers_accounting_debt_on_typed_workflow_artifact():
    accumulator = CostAccumulator(max_cost_usd=5.0)

    accumulator.record(
        role="implementer",
        usage=SimpleNamespace(
            cost_usd=0.2,
            input_tokens=10,
            output_tokens=3,
            accounting_debt=("cost_complete",),
        ),
        backend_name="claude",
    )

    summary = accumulator.to_dict()
    assert summary["total_usd"] == pytest.approx(0.2)
    assert summary["accounting_debt_by_role"] == {
        "implementer": ["cost_complete"],
    }
    assert accumulator.remaining() == 0.0


@pytest.mark.parametrize(
    ("role", "missing"),
    [
        ("", ("cost_complete",)),
        ("breaker", ()),
        ("breaker", ("terminal_complete",)),
    ],
)
def test_note_accounting_debt_rejects_invalid_contract(role, missing):
    with pytest.raises(ValueError, match="accounting"):
        CostAccumulator().note_accounting_debt(
            role=role,
            missing=missing,
        )
