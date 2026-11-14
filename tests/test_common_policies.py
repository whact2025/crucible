# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.common.policies`` + ``policy_loader``.

Coverage scope tracks the consumed-policy surface:
- ``BreachPolicy`` + ``ReviewPolicy`` are the only wired policies as
  of Phase 2. ``BuildPolicy`` / ``TierPolicy`` were dropped because
  zero of their fields had a workflow-side consumer — declaring inert
  fields invites the operator footgun of spec.yaml values silently
  defaulting (the same reason ``BuildPolicy.max_retries`` and
  ``ReviewPolicy.max_fix_rounds`` were dropped previously). When
  a future wire-through commit consumes them, re-add fields here
  + extend this test file in the same commit.

- Default values match the post-cutover-validated table.
- Dataclasses are frozen (immutable).
- YAML loader returns defaults when blocks are absent.
- YAML loader applies overrides correctly.
- Unknown fields in a yaml block surface as ValueError.
- Tuple-typed fields (e.g. ReviewPolicy.rubrics) accept yaml-list form.
"""

from __future__ import annotations

import dataclasses

import pytest

from crucible.common.policies import (
    BreachPolicy,
    CruciblePolicies,
    ReviewPolicy,
)
from crucible.common.policy_loader import load_policies_from_yaml


# --- default-value validation ---------------------------------------

def test_breach_policy_defaults_match_post_cutover():
    p = BreachPolicy()
    assert p.enabled is False           # post-cutover off by default
    assert p.max_rounds == 5            # tier workflow's stricter default
    assert p.persistence_veto_threshold == 3
    assert p.breaker_rubric is None
    assert p.judge_rubric is None
    assert p.require_attack_authority is True


def test_breach_policy_field_surface_matches_consumed_set():
    """Every BreachPolicy field must have a corresponding read in
    ``workflows/tier.py``'s Phase 3 — the dataclass is the
    consumed-surface contract. If you add a field here, also wire it
    in the consume site (and update test_workflows_tier_breach_policy
    to pin the read)."""
    names = {f.name for f in dataclasses.fields(BreachPolicy)}
    assert names == {
        "enabled",
        "max_rounds",
        "persistence_veto_threshold",
        "breaker_rubric",
        "judge_rubric",
        "require_attack_authority",
    }, (
        "BreachPolicy field set drift — every field must be consumed "
        f"in workflows/tier.py. Current: {sorted(names)}"
    )


def test_review_policy_defaults():
    p = ReviewPolicy()
    assert p.rubrics == ("diff_scope", "spec_clarity", "tier_completeness")


def test_review_policy_field_surface_matches_consumed_set():
    names = {f.name for f in dataclasses.fields(ReviewPolicy)}
    # Every field is consumed in workflows/tier.py: `rubrics` selects the
    # Phase-4 review rubrics, `advisory_rubrics` marks which record-but-don't-
    # gate, and `retry_budget` bounds the review-feedback retry loop
    # (tier.py:2121/2133). Keep this pin in sync when a field is added/wired.
    assert names == {"rubrics", "advisory_rubrics", "retry_budget"}, (
        "ReviewPolicy field set drift — every field must be consumed "
        f"in workflows/tier.py. Current: {sorted(names)}"
    )


def test_aggregate_defaults():
    cp = CruciblePolicies()
    assert isinstance(cp.breach, BreachPolicy)
    assert isinstance(cp.review, ReviewPolicy)


def test_aggregate_only_carries_wired_policies():
    """``CruciblePolicies`` should only carry policies that have a
    real consumer. ``BuildPolicy`` / ``TierPolicy`` are excluded
    (zero-consumer-site at the time of Phase 2). If you re-add
    them, do it in the same commit that wires the fields."""
    names = {f.name for f in dataclasses.fields(CruciblePolicies)}
    assert names == {"breach", "review"}


# --- immutability ---------------------------------------------------

@pytest.mark.parametrize("cls", [
    BreachPolicy, ReviewPolicy, CruciblePolicies,
])
def test_policy_is_frozen(cls):
    p = cls()
    with pytest.raises(dataclasses.FrozenInstanceError):
        first_field = dataclasses.fields(cls)[0].name
        setattr(p, first_field, None)


@pytest.mark.parametrize("cls,field_name,bad_value", [
    (BreachPolicy, "max_rounds", 99),
    (BreachPolicy, "breaker_rubric", "x"),
    (ReviewPolicy, "rubrics", ()),
])
def test_inner_policies_reject_post_construction_mutation(cls, field_name, bad_value):
    """Depth coverage so inner-policy frozen-ness is exercised on a
    representative field beyond field[0]."""
    p = cls()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(p, field_name, bad_value)


def test_crucible_policies_inner_objects_also_frozen():
    """``CruciblePolicies`` is a frozen dataclass containing frozen
    sub-policies. Verify the inner objects can't be swapped via
    attribute set on the aggregate."""
    cp = CruciblePolicies()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cp.breach = BreachPolicy(enabled=True)
    # And the inner ref is still frozen against direct mutation:
    with pytest.raises(dataclasses.FrozenInstanceError):
        cp.breach.max_rounds = 99


# --- YAML loader ----------------------------------------------------

def test_loader_empty_yaml_returns_defaults():
    cp = load_policies_from_yaml({})
    assert cp == CruciblePolicies()


def test_loader_missing_blocks_inherit_defaults():
    """Only some blocks present → others fall back."""
    cp = load_policies_from_yaml({
        "review_policy": {"rubrics": ["diff_scope"]},
    })
    assert cp.review.rubrics == ("diff_scope",)
    # Breach policy untouched → default.
    assert cp.breach == BreachPolicy()


def test_loader_applies_overrides():
    cp = load_policies_from_yaml({
        "breach_policy": {
            "enabled": True,
            "max_rounds": 10,
            "breaker_rubric": "custom_breaker",
            "require_attack_authority": True,
        },
        "review_policy": {
            "rubrics": ["diff_scope", "tier_completeness"],
        },
    })
    assert cp.breach.enabled is True
    assert cp.breach.max_rounds == 10
    assert cp.breach.breaker_rubric == "custom_breaker"
    assert cp.breach.require_attack_authority is True
    assert cp.review.rubrics == ("diff_scope", "tier_completeness")


def test_loader_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown field"):
        load_policies_from_yaml({
            "breach_policy": {"max_roundz": 10},  # typo
        })


def test_loader_rejects_dropped_field():
    """Fields that USED to exist (``exhaustion_policy``, ``blind``,
    etc.) must now be rejected — they were dropped because they
    weren't consumed. An operator copying a stale spec.yaml that
    still set them needs a loud failure, not silent defaulting."""
    with pytest.raises(ValueError, match="unknown field"):
        load_policies_from_yaml({
            "breach_policy": {"exhaustion_policy": "halt"},
        })


def test_loader_silently_ignores_unknown_top_level_block():
    """``build_policy`` and ``tier_policy`` blocks were dropped when
    the field-discipline rule deleted those policies. The loader has
    no consumer for them today. Documents the CURRENT contract:
    unknown top-level blocks pass through silently because the dict
    accessor uses ``.get()`` with default None. If we add strict-
    top-block-rejection later (the natural followup to the field-
    discipline rule), tighten this test then — the name will need
    to change to ``test_loader_rejects_unknown_top_level_block``."""
    cp = load_policies_from_yaml({"build_policy": {"anything": 1}})
    assert cp == CruciblePolicies()


def test_loader_tuple_field_from_yaml_list():
    """ReviewPolicy.rubrics is tuple[str, ...]; yaml gives a list."""
    cp = load_policies_from_yaml({
        "review_policy": {
            "rubrics": ["diff_scope", "spec_clarity"],
        },
    })
    assert cp.review.rubrics == ("diff_scope", "spec_clarity")


def test_loader_non_mapping_block_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        load_policies_from_yaml({"breach_policy": "wrong-type"})


# --- loader robustness ---------------------------------------------

def test_loader_accepts_none():
    """``yaml.safe_load("")`` returns ``None``. Synthesized specs
    (e.g. SWE-bench's starter spec) likewise carry no policy blocks.
    Both shapes must produce ``CruciblePolicies()`` defaults rather
    than forcing every caller to pre-guard ``raw or {}``."""
    assert load_policies_from_yaml(None) == CruciblePolicies()


def test_loader_yaml_text_roundtrip():
    """Real-world entry shape: spec.yaml text → yaml.safe_load → loader.
    Pins that yaml's bool / int / list parsing flows through correctly
    (e.g. ``enabled: yes`` becomes True, list-of-strings becomes the
    tuple-typed rubrics field). Direct-dict tests can miss yaml quirks."""
    import yaml as _yaml
    text = """\
breach_policy:
  enabled: yes
  max_rounds: 7
review_policy:
  rubrics:
    - diff_scope
    - tier_completeness
"""
    cp = load_policies_from_yaml(_yaml.safe_load(text))
    assert cp.breach.enabled is True
    assert cp.breach.max_rounds == 7
    assert cp.review.rubrics == ("diff_scope", "tier_completeness")


def test_loader_empty_yaml_text_returns_defaults():
    """``yaml.safe_load("")`` is the empty-file case — pairs with the
    raw=None test but exercises the actual yaml entrypoint."""
    import yaml as _yaml
    cp = load_policies_from_yaml(_yaml.safe_load(""))
    assert cp == CruciblePolicies()
