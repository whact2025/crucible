# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Phase 4 review pass — ``ReviewPolicy.rubrics`` is honored.

Prior to this plumbing the tier hardcoded
``_REVIEW_RUBRICS = ("diff_scope", "spec_clarity", "tier_completeness")``
which caused false-negative review failures on SWE-bench (the spec
is dataset-authored, so ``spec_clarity`` flagged the verbatim
``problem_statement`` intent as ambiguous and the deliberately-empty
``done_criteria`` as missing — both blocked correct diffs).

These tests pin the contract:
- ``_validate_review_rubrics`` rejects unknown names at the
  ``run_tier`` entry (fail-fast before cost-bearing phases).
- Trimming the set (e.g., ``("diff_scope", "tier_completeness")``)
  is accepted and lands in the policy.
- ``SWEBENCH_REVIEW_POLICY`` matches the SWE-bench-shaped trim.
- ``ReviewPolicy`` defaults preserve the original three-rubric set
  for non-SWE-bench callers.
"""

from __future__ import annotations

import pytest

from crucible.common.policies import ReviewPolicy
from crucible.products.swebench.spec_synthesis import SWEBENCH_REVIEW_POLICY
from crucible.workflows.tier import (
    _SUPPORTED_REVIEW_RUBRICS,
    _validate_review_rubrics,
)


# ----------------------------------------------------------------------
# Default + supported set
# ----------------------------------------------------------------------


def test_review_policy_default_rubrics_match_supported_set():
    """The dataclass default must be a subset of the tier's supported
    rubric names — otherwise ``run_tier`` raises on the default."""
    policy = ReviewPolicy()
    assert set(policy.rubrics).issubset(_SUPPORTED_REVIEW_RUBRICS)


def test_review_policy_default_rubrics_unchanged():
    """The pre-policy hardcoded set was ``("diff_scope", "spec_clarity",
    "tier_completeness")``. Default kept identical so non-SWE-bench
    callers (kit / IsaacSim) see no behavior change from this lift."""
    assert ReviewPolicy().rubrics == (
        "diff_scope", "spec_clarity", "tier_completeness",
    )


# ----------------------------------------------------------------------
# _validate_review_rubrics
# ----------------------------------------------------------------------


def test_validate_review_rubrics_accepts_default():
    _validate_review_rubrics(ReviewPolicy().rubrics)  # must not raise


def test_validate_review_rubrics_accepts_trimmed_set():
    """SWE-bench's shape — drop spec_clarity — must validate."""
    _validate_review_rubrics(("diff_scope", "tier_completeness"))


def test_validate_review_rubrics_accepts_single_rubric():
    """Minimum case: one rubric. Useful for ablation runs."""
    _validate_review_rubrics(("diff_scope",))


def test_validate_review_rubrics_accepts_empty_tuple():
    """Zero rubrics = skip-review-by-policy. Functionally equivalent
    to ``skip_review=True`` but documents intent at the spec level."""
    _validate_review_rubrics(())


def test_validate_review_rubrics_rejects_unknown_name():
    """Typo'd name must surface as a clear error, not silently no-op."""
    with pytest.raises(ValueError, match=r"unsupported name"):
        _validate_review_rubrics(("diff_scope", "diff_scop"))  # typo


def test_validate_review_rubrics_rejects_mixed_known_and_unknown():
    """All unknown names listed in the error — partial-typo cases
    don't hide one bad name behind a passing one."""
    with pytest.raises(ValueError) as excinfo:
        _validate_review_rubrics(("diff_scope", "fake_rubric_a", "fake_rubric_b"))
    msg = str(excinfo.value)
    assert "fake_rubric_a" in msg
    assert "fake_rubric_b" in msg


# ----------------------------------------------------------------------
# SWEBENCH_REVIEW_POLICY — the product-specific shape
# ----------------------------------------------------------------------


def test_swebench_policy_drops_spec_clarity():
    """SWE-bench specs are dataset-authored (problem_statement
    verbatim + empty done_criteria); ``spec_clarity`` would fire
    false negatives on every instance. The SWE-bench policy must
    omit it."""
    assert "spec_clarity" not in SWEBENCH_REVIEW_POLICY.rubrics


def test_swebench_policy_keeps_implementation_rubrics():
    """``diff_scope`` (reviews the implementer's diff) and
    ``tier_completeness`` (reviews directive coverage) are
    implementation-level rubrics — load-bearing for SWE-bench
    quality signal and must remain in the trimmed set."""
    assert "diff_scope" in SWEBENCH_REVIEW_POLICY.rubrics
    assert "tier_completeness" in SWEBENCH_REVIEW_POLICY.rubrics


def test_swebench_policy_allows_one_bounded_review_repair():
    assert SWEBENCH_REVIEW_POLICY.retry_budget == 1


def test_swebench_policy_rubrics_validate():
    """End-to-end: the SWE-bench policy's rubric set must pass the
    same validator ``run_tier`` runs at entry."""
    _validate_review_rubrics(SWEBENCH_REVIEW_POLICY.rubrics)


def test_swebench_policy_carries_only_consumed_fields():
    """``ReviewPolicy.enabled`` was dropped — the master switch lives
    on the architect / tier ``include_review`` / ``skip_review``
    kwargs, not on the policy. Verify the SWE-bench instance doesn't
    accidentally regrow that field via a future ``replace`` call. The
    set is the consumed-field whitelist: ``rubrics`` (which rubrics run),
    ``advisory_rubrics`` (which run non-gating), and ``retry_budget`` (the
    review-feedback retry count) — all read in workflows/tier.py Phase 4."""
    field_names = {f.name for f in __import__("dataclasses").fields(SWEBENCH_REVIEW_POLICY)}
    assert field_names == {"rubrics", "advisory_rubrics", "retry_budget"}


# ----------------------------------------------------------------------
# Phase 4 review-loop behavior — the policy actually changes what runs.
# ----------------------------------------------------------------------
#
# These tests pin the consume-site contract by inspecting the Phase 4
# source in workflows/tier.py:
#   - The rendered ``artifacts`` dict only carries entries for rubrics
#     the policy asks for (no wasted spec_clarity render when SWE-bench
#     dropped it).
#   - The review loop iterates ``policy_rubrics`` (not the supported
#     set), so trim-by-policy actually trims the verdict count.
#
# A full end-to-end check would require a live cascade run; that costs
# real LLM budget and lives in the live_docker / smoke harness, not in
# the offline suite. These source-grep guards catch the failure mode
# the operator actually hit (an unexpected rubric running and false-
# negative'ing a correct diff).


def test_phase4_loop_iterates_policy_rubrics_not_supported_set():
    """The Phase 4 review loop must walk the policy's rubric list,
    not the supported set. If a future refactor reverts to
    ``for rubric_name in _SUPPORTED_REVIEW_RUBRICS:``, the SWE-bench
    trim no longer takes effect and ``spec_clarity`` fires again."""
    import crucible.workflows.tier as tier_mod
    from ._source_invariants import assert_source_invariant

    assert_source_invariant(
        tier_mod,
        scope="Phase 4 review loop",
        must_contain=("for rubric_name in policy_rubrics:",),
        must_not_contain=("for rubric_name in _SUPPORTED_REVIEW_RUBRICS",),
    )


def test_phase4_artifacts_dict_renders_only_policy_rubrics():
    """Each artifact render is gated by ``if <name> in policy_rubrics``,
    so dropping a rubric also drops its render cost. Regex tolerates
    either single- or double-quoted string form so a routine
    quote-style refactor doesn't trip the test."""
    import crucible.workflows.tier as tier_mod
    from ._source_invariants import assert_source_invariant

    for name in ("diff_scope", "spec_clarity", "tier_completeness"):
        assert_source_invariant(
            tier_mod,
            scope=f"Phase 4 conditional render of {name}",
            # Match either '{name}' or "{name}" in the conditional.
            must_contain_patterns=(
                rf"""['"]{name}['"] in policy_rubrics""",
            ),
        )
