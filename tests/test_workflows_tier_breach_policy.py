# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Phase 3 breach pass — ``BreachPolicy`` plumbing.

Mirrors ``test_workflows_tier_review_policy.py`` for the breach lift.
Pins:

- The Phase 3 source reads from ``breach_policy.X`` (the policy is
  load-bearing), not from removed scalar kwargs.
- ``SWEBENCH_BREACH_POLICY`` carries the expected shape for the
  SWE-bench cascade (enabled=True, max_rounds=5, swebench rubrics).
- ``BreachPolicy`` defaults preserve no-breach behavior for kit /
  IsaacSim callers (``enabled=False``).
- Architect/tier CLI defaults defer to an authored ``breach_policy``;
  explicit CLI flags override only the supplied fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.common.policies import BreachPolicy
from crucible.products.swebench.spec_synthesis import SWEBENCH_BREACH_POLICY


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------


def test_breach_policy_default_is_disabled():
    """Kit / IsaacSim flow gets no breach by default — the policy
    default must preserve that."""
    assert BreachPolicy().enabled is False


def test_breach_policy_default_rounds():
    """Pre-policy hardcoded default in run_tier was 5; preserve it."""
    assert BreachPolicy().max_rounds == 5


def test_breach_policy_default_veto():
    """Pre-policy hardcoded default in run_tier was 3; preserve it."""
    assert BreachPolicy().persistence_veto_threshold == 3


def test_breach_policy_default_requires_attack_authority():
    """Parent workflows bind generated attacks to root clauses by default."""
    assert BreachPolicy().require_attack_authority is True


# ----------------------------------------------------------------------
# SWEBENCH_BREACH_POLICY — the product-specific shape
# ----------------------------------------------------------------------


def test_swebench_breach_policy_enabled():
    """SWE-bench's quality mechanism IS the breach loop. The policy
    must opt in by default; ablation runs flip ``enabled=False``
    via ``dataclasses.replace`` at the call site."""
    assert SWEBENCH_BREACH_POLICY.enabled is True


def test_swebench_breach_policy_uses_swebench_rubrics():
    """The Breaker rubric must be ``swebench_impl_breaker`` so the
    rubric file under ``crucible/rubrics/`` is what loads; the Judge
    inherits the bundled ``breach_judge``. These names are also the
    pool-routing role keys; changing them silently re-routes to the
    default Breaker/Judge backends."""
    assert SWEBENCH_BREACH_POLICY.breaker_rubric == "swebench_impl_breaker"
    assert SWEBENCH_BREACH_POLICY.judge_rubric == "breach_judge"
    assert SWEBENCH_BREACH_POLICY.require_attack_authority is True


def test_swebench_breach_policy_max_rounds_above_tier_default():
    """SWE-bench bumps the breach round cap to 8 (above the tier-level
    default of 5) after the v2 stratified sweep surfaced
    ``max_rounds_exhausted`` outcomes that were trending toward
    passing diffs at round 5. The budget cap is the cost-side
    governor; this is the loop-shape governor."""
    assert SWEBENCH_BREACH_POLICY.max_rounds == 8


# ----------------------------------------------------------------------
# Phase 3 consume-site source guards
# ----------------------------------------------------------------------
#
# Source-grep guards so a future refactor that re-introduces the
# scalar-kwarg shape (`args.max_breach_rounds`, etc.) is caught at
# CI rather than at the next sweep with a silent default.


def test_tier_phase3_reads_breach_policy_fields():
    """Phase 3 must read every consumed BreachPolicy field from the
    policy object — not from removed scalar kwargs on the Namespace.
    A regression that reverts to ``args.max_breach_rounds`` etc.
    leaves the wire-through inert and silently defaults the breach
    behavior."""
    import crucible.workflows.tier as tier_mod
    from ._source_invariants import assert_source_invariant

    assert_source_invariant(
        tier_mod,
        scope="Phase 3 breach consume site",
        must_contain=(
            "breach_policy.enabled",
            "breach_policy.max_rounds",
            "breach_policy.persistence_veto_threshold",
            "breach_policy.breaker_rubric",
            "breach_policy.judge_rubric",
            "breach_policy.require_attack_authority",
        ),
        must_not_contain=(
            "args.max_breach_rounds",
            "args.persistence_veto_threshold",
            "args.skip_breach",
        ),
    )


# ----------------------------------------------------------------------
# CLI translation in tier's main()
# ----------------------------------------------------------------------


def test_tier_main_uses_cli_policy_resolver():
    """The CLI boundary must preserve authored-policy precedence."""
    import inspect

    import crucible.workflows.tier as tier_mod

    main_src = inspect.getsource(tier_mod.main)
    assert "_resolve_cli_breach_policy(" in main_src
    assert "enabled=not parsed.skip_breach" not in main_src


# ----------------------------------------------------------------------
# SWE-bench breach-policy resolution helper
# ----------------------------------------------------------------------


def test_resolve_swebench_breach_policy_enabled_returns_template():
    """``enabled=True`` returns the unchanged SWE-bench template
    instance (identity, not a copy) so callers can rely on the
    template's static fields (rubric names, max_rounds, etc.)
    flowing through unchanged."""
    from crucible.products.swebench.spec_synthesis import (
        SWEBENCH_BREACH_POLICY,
        resolve_swebench_breach_policy,
    )
    assert resolve_swebench_breach_policy(enabled=True) is SWEBENCH_BREACH_POLICY


def test_resolve_swebench_breach_policy_disabled_replaces_enabled():
    """``enabled=False`` produces a policy with enabled flipped but
    all other SWE-bench fields preserved. Pre-helper this was open-
    coded as ``dataclasses.replace(...)`` at every call site
    (feedback_helpers_over_duplication)."""
    from crucible.products.swebench.spec_synthesis import (
        SWEBENCH_BREACH_POLICY,
        resolve_swebench_breach_policy,
    )
    disabled = resolve_swebench_breach_policy(enabled=False)
    assert disabled.enabled is False
    assert disabled.max_rounds == SWEBENCH_BREACH_POLICY.max_rounds
    assert disabled.breaker_rubric == SWEBENCH_BREACH_POLICY.breaker_rubric
    assert disabled.judge_rubric == SWEBENCH_BREACH_POLICY.judge_rubric
    assert (
        disabled.require_attack_authority
        is SWEBENCH_BREACH_POLICY.require_attack_authority
    )
    # New instance, not the template — confirms we didn't accidentally
    # mutate the shared default-instance.
    assert disabled is not SWEBENCH_BREACH_POLICY


# ----------------------------------------------------------------------
# spec.yaml precedence chain (kwarg > spec.yaml > dataclass default)
# ----------------------------------------------------------------------


def _capture_tier_cli_policy(
    monkeypatch, spec_dir: Path, *extra_args: str,
) -> BreachPolicy:
    import crucible.workflows.tier as tier_mod

    captured = {}

    def _fake_run_tier(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(tier_mod, "run_tier", _fake_run_tier)
    rc = tier_mod.main([
        "--spec", str(spec_dir),
        "--tier", "t0",
        "--cwd", str(spec_dir),
        *extra_args,
    ])
    assert rc == 0
    return captured["breach_policy"]


def _capture_architect_cli_policy(
    monkeypatch, spec_dir: Path, *extra_args: str,
) -> BreachPolicy:
    import crucible.workflows.architect as architect_mod

    captured = {}

    def _fake_run_architect(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(architect_mod, "run_architect", _fake_run_architect)
    rc = architect_mod.main([
        "--planner-count", "1",
        "--spec", str(spec_dir),
        "--cwd", str(spec_dir),
        *extra_args,
    ])
    assert rc == 0
    return captured["breach_policy"]


def _write_breach_policy(
    spec_dir: Path,
    *,
    enabled: bool,
    max_rounds: int = 7,
    persistence_veto_threshold: int = 4,
) -> None:
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "spec.yaml").write_text(
        "breach_policy:\n"
        f"  enabled: {str(enabled).lower()}\n"
        f"  max_rounds: {max_rounds}\n"
        f"  persistence_veto_threshold: {persistence_veto_threshold}\n"
        "  breaker_rubric: custom_breaker\n"
        "  judge_rubric: custom_judge\n"
    )


def test_cli_absent_policy_preserves_workflow_defaults(tmp_path, monkeypatch):
    """No authored block keeps architect OFF and direct tier CLI ON."""
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    (spec_dir / "spec.yaml").write_text("intent: no breach policy here\n")

    tier_policy = _capture_tier_cli_policy(monkeypatch, spec_dir)
    architect_policy = _capture_architect_cli_policy(monkeypatch, spec_dir)

    assert tier_policy == BreachPolicy(enabled=True)
    assert architect_policy == BreachPolicy(enabled=False)


@pytest.mark.parametrize("enabled", [True, False])
def test_cli_honors_authored_breach_policy(
    tmp_path, monkeypatch, enabled: bool,
):
    """Absent CLI flags leave every authored policy field intact."""
    spec_dir = tmp_path / "spec"
    _write_breach_policy(spec_dir, enabled=enabled)
    expected = BreachPolicy(
        enabled=enabled,
        max_rounds=7,
        persistence_veto_threshold=4,
        breaker_rubric="custom_breaker",
        judge_rubric="custom_judge",
    )

    assert _capture_tier_cli_policy(monkeypatch, spec_dir) == expected
    assert _capture_architect_cli_policy(monkeypatch, spec_dir) == expected


def test_explicit_cli_flags_override_only_supplied_policy_fields(
    tmp_path, monkeypatch,
):
    """Existing include/skip flags win without erasing authored fields."""
    architect_spec = tmp_path / "architect_spec"
    _write_breach_policy(architect_spec, enabled=False)
    architect_policy = _capture_architect_cli_policy(
        monkeypatch,
        architect_spec,
        "--include-breach",
        "--max-breach-rounds", "11",
    )
    assert architect_policy == BreachPolicy(
        enabled=True,
        max_rounds=11,
        persistence_veto_threshold=4,
        breaker_rubric="custom_breaker",
        judge_rubric="custom_judge",
    )

    tier_spec = tmp_path / "tier_spec"
    _write_breach_policy(tier_spec, enabled=True)
    tier_policy = _capture_tier_cli_policy(
        monkeypatch,
        tier_spec,
        "--skip-breach",
        "--persistence-veto-threshold", "9",
    )
    assert tier_policy == BreachPolicy(
        enabled=False,
        max_rounds=7,
        persistence_veto_threshold=9,
        breaker_rubric="custom_breaker",
        judge_rubric="custom_judge",
    )


def test_load_spec_yaml_policies_reads_breach_block(tmp_path):
    """``_load_spec_yaml_policies`` honors the ``breach_policy:``
    block in spec.yaml — the precedence-#2 entry the
    ``common/policies.py`` docstring claims. Without this layer
    the docstring would lie about the contract."""
    from crucible.workflows.tier import _load_spec_yaml_policies

    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    (spec_dir / "spec.yaml").write_text(
        "breach_policy:\n"
        "  enabled: true\n"
        "  max_rounds: 7\n"
        "  breaker_rubric: custom_breaker\n"
    )
    policies = _load_spec_yaml_policies(spec_dir)
    assert policies.breach.enabled is True
    assert policies.breach.max_rounds == 7
    assert policies.breach.breaker_rubric == "custom_breaker"


def test_load_spec_yaml_policies_reads_review_block(tmp_path):
    """The review_policy block is honored too — SWE-bench's
    ``synthesize_starter_spec`` writes neither block today (it
    passes ``SWEBENCH_REVIEW_POLICY`` as an explicit kwarg), but
    kit / IsaacSim authors could set this in their spec.yaml and
    expect the loaded value to take effect for programmatic
    callers."""
    from crucible.workflows.tier import _load_spec_yaml_policies

    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    (spec_dir / "spec.yaml").write_text(
        "review_policy:\n"
        "  rubrics:\n"
        "    - diff_scope\n"
    )
    policies = _load_spec_yaml_policies(spec_dir)
    assert policies.review.rubrics == ("diff_scope",)


def test_load_spec_yaml_policies_missing_file_returns_defaults(tmp_path):
    """No spec.yaml on disk → defaults, no raise. The cascade body
    surfaces a clearer error when the Spec itself can't be loaded
    via the Corpus; the policy loader doesn't double-report."""
    from crucible.common.policies import CruciblePolicies
    from crucible.workflows.tier import _load_spec_yaml_policies

    spec_dir = tmp_path / "no_such_spec"
    spec_dir.mkdir()
    assert _load_spec_yaml_policies(spec_dir) == CruciblePolicies()


def test_explicit_kwarg_overrides_spec_yaml_block(tmp_path, monkeypatch):
    """Precedence-#1: explicit kwarg on ``run_tier`` MUST outrank
    the spec.yaml block. Without this guarantee, an operator who
    passes ``breach_policy=BreachPolicy(...)`` directly (e.g., the
    SWE-bench worker's ``resolve_swebench_breach_policy(...)``) could
    have their explicit policy silently overridden by stale yaml
    content. Pin via inspecting the run_tier source for the
    ``if breach_policy is None`` guard pattern that implements the
    precedence.
    """
    import inspect

    import crucible.workflows.tier as tier_mod

    src = inspect.getsource(tier_mod.run_tier)
    # The guard pattern: only fall through to spec_policies when the
    # explicit kwarg was None. ``policy = explicit or spec_yaml`` would
    # also work for breach (BreachPolicy() is truthy) but breaks if
    # any future policy field makes the instance falsy; ``is None`` is
    # the unambiguous form. Both BreachPolicy and ReviewPolicy must
    # use this pattern.
    assert "if breach_policy is None:" in src, (
        "run_tier must check ``breach_policy is None`` before falling "
        "back to spec.yaml — explicit kwargs are precedence #1"
    )
    assert "if review_policy is None:" in src, (
        "ditto for review_policy"
    )


def test_load_spec_yaml_policies_malformed_yaml_returns_defaults(tmp_path):
    """A malformed spec.yaml triggers a yaml parse error inside
    ``_load_spec_yaml_policies``. The helper swallows it and returns
    defaults — the cascade body's subsequent ``Corpus.latest_spec()``
    call will surface a clearer error against the same file. The
    swallow exists to avoid double-reporting at the policy layer.
    """
    from crucible.common.policies import CruciblePolicies
    from crucible.workflows.tier import _load_spec_yaml_policies

    spec_dir = tmp_path / "malformed_spec"
    spec_dir.mkdir()
    # Unterminated quote — yaml.safe_load raises a yaml.YAMLError.
    (spec_dir / "spec.yaml").write_text('breach_policy:\n  enabled: "unterminated')
    assert _load_spec_yaml_policies(spec_dir) == CruciblePolicies()


def test_load_spec_yaml_policies_empty_file_returns_defaults(tmp_path):
    """An empty spec.yaml (``yaml.safe_load("") == None``) → defaults,
    no raise. The ``load_policies_from_yaml(None)`` path is
    covered by the policy-loader tests; this pins the
    file-empty-on-disk shape end-to-end."""
    from crucible.common.policies import CruciblePolicies
    from crucible.workflows.tier import _load_spec_yaml_policies

    spec_dir = tmp_path / "empty_spec"
    spec_dir.mkdir()
    (spec_dir / "spec.yaml").write_text("")
    assert _load_spec_yaml_policies(spec_dir) == CruciblePolicies()


def test_swebench_workflow_uses_helper_at_both_call_sites():
    """Source-grep guard against re-introducing the inline
    ``dataclasses.replace(SWEBENCH_BREACH_POLICY, ...)`` boilerplate
    that the helper exists to dedupe (feedback_helpers_over_duplication)."""
    import crucible.workflows.swebench as swe_mod
    from ._source_invariants import assert_source_invariant

    assert_source_invariant(
        swe_mod,
        scope="SWE-bench breach-policy resolution",
        must_contain=(
            "resolve_swebench_breach_policy(",
        ),
        must_not_contain=(
            # The boilerplate the helper replaced. If this string
            # reappears at a swebench call site, the helper got
            # bypassed.
            "dataclasses.replace(SWEBENCH_BREACH_POLICY",
        ),
    )
