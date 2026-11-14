# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for the per-model pricing table.

Pins:
- The DEFAULT_PRICING table shape (every model has explicit rates).
- ``compute_cost`` math (uncached + cache_write + cache_read +
  output portions sum correctly from absolute per-MTok rates).
- YAML override discovery + merge semantics.
- ``lookup_pricing_for_role`` routing → model → pricing lookup.
- Explicit zero-cost routes and fail-closed unknown models.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from assay import Usage
from crucible.backends.config import BackendRouting, DEFAULT_ROUTING, RoleConfig
from crucible.backends.pricing import (
    DEFAULT_PRICING,
    LATEST_VERIFIED_AT,
    ModelPricing,
    PricingConfigurationError,
    compute_cost,
    load_pricing,
    lookup_pricing_for_role,
)
from crucible.common.cost import CostAccumulator


# ----------------------------------------------------------------------
# DEFAULT_PRICING sanity
# ----------------------------------------------------------------------


def test_default_pricing_covers_known_models():
    """Every model the routing defaults reference must have a price."""
    expected_models = {
        "claude-fable-5-1",
        "anthropic.claude-fable-5-1",
        "aws/anthropic/bedrock-claude-fable-5-1",
        "azure/anthropic/claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-opus-5", "anthropic.claude-opus-5",
        "aws/anthropic/bedrock-claude-opus-5",
        "aws/anthropic/bedrock-claude-opus-4-8", "aws/anthropic/bedrock-claude-sonnet-4-6",
        # Bare first-party ids the `claude login` OAuth path requires —
        # what the workspace backends.yaml override actually routes to.
        "claude-opus-4-8", "claude-sonnet-4-6",
        "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
        "gpt-6-astra",
        "kimi-k2.7-code", "kimi-k2.7-code-highspeed", "kimi-k3",
        "nvidia/zai-org/glm-5.2",
    }
    missing = expected_models - set(DEFAULT_PRICING.keys())
    assert not missing, f"DEFAULT_PRICING missing: {missing}"
    assert DEFAULT_ROUTING.default.model in DEFAULT_PRICING


def test_default_pricing_entries_are_explicit_and_nonnegative():
    for model, p in DEFAULT_PRICING.items():
        assert p.input_per_mtok >= 0, f"{model} has negative input rate"
        assert p.output_per_mtok >= 0, f"{model} has negative output rate"
        assert (p.input_per_mtok == 0) == (p.output_per_mtok == 0), (
            f"{model} mixes zero and paid base rates"
        )
        for field in (
            "long_context_input_per_mtok",
            "long_context_output_per_mtok",
            "long_context_cache_write_per_mtok",
            "long_context_cache_read_per_mtok",
        ):
            value = getattr(p, field)
            assert value is None or value >= 0, (
                f"{model} has negative {field}"
            )


@pytest.mark.parametrize(
    (
        "model",
        "input_rate",
        "output_rate",
        "cache_read_rate",
        "cache_write_rate",
        "long_input_rate",
        "long_output_rate",
        "long_cache_read_rate",
        "long_cache_write_rate",
    ),
    [
        (
            "gpt-5.6-luna", 0.20, 1.20, 0.02, 0.25,
            0.40, 1.80, 0.04, 0.50,
        ),
        (
            "gpt-5.6-terra", 2.00, 12.00, 0.20, 2.50,
            4.00, 18.00, 0.40, 5.00,
        ),
        (
            "gpt-5.6-sol", 4.00, 20.00, 0.40, 5.00,
            8.00, 30.00, 0.80, 10.00,
        ),
        (
            "gpt-6-astra", 10.00, 50.00, 1.00, 12.50,
            20.00, 75.00, 2.00, 25.00,
        ),
    ],
)
def test_openai_session_pricing_rates(
    model,
    input_rate,
    output_rate,
    cache_read_rate,
    cache_write_rate,
    long_input_rate,
    long_output_rate,
    long_cache_read_rate,
    long_cache_write_rate,
):
    p = DEFAULT_PRICING[model]
    assert p.input_per_mtok == input_rate
    assert p.output_per_mtok == output_rate
    assert p.cache_read_per_mtok == pytest.approx(cache_read_rate)
    assert p.cache_write_per_mtok == pytest.approx(cache_write_rate)
    assert p.long_context_threshold_tokens == 272_000
    assert p.long_context_input_per_mtok == pytest.approx(long_input_rate)
    assert p.long_context_output_per_mtok == pytest.approx(long_output_rate)
    assert p.long_context_cache_read_per_mtok == pytest.approx(
        long_cache_read_rate
    )
    assert p.long_context_cache_write_per_mtok == pytest.approx(
        long_cache_write_rate
    )


@pytest.mark.parametrize(
    (
        "model",
        "base_input",
        "base_output",
        "base_cache_write",
        "base_cache_read",
        "long_input",
        "long_output",
        "long_cache_write",
        "long_cache_read",
    ),
    [
        (
            "gpt-5.6-sol",
            4.00, 20.00, 5.00, 0.40,
            8.00, 30.00, 10.00, 0.80,
        ),
        (
            "gpt-6-astra",
            10.00, 50.00, 12.50, 1.00,
            20.00, 75.00, 25.00, 2.00,
        ),
    ],
)
def test_openai_long_context_prices_entire_request(
    model,
    base_input,
    base_output,
    base_cache_write,
    base_cache_read,
    long_input,
    long_output,
    long_cache_write,
    long_cache_read,
):
    pricing = DEFAULT_PRICING[model]

    assert pricing.compute_cost(
        input_tokens=272_000,
        output_tokens=10_000,
        cache_creation_tokens=100_000,
        cache_read_tokens=100_000,
    ) == pytest.approx(
        72_000 * base_input / 1_000_000
        + 100_000 * base_cache_write / 1_000_000
        + 100_000 * base_cache_read / 1_000_000
        + 10_000 * base_output / 1_000_000
    )
    assert pricing.compute_cost(
        input_tokens=272_001,
        output_tokens=10_000,
        cache_creation_tokens=100_000,
        cache_read_tokens=100_000,
    ) == pytest.approx(
        72_001 * long_input / 1_000_000
        + 100_000 * long_cache_write / 1_000_000
        + 100_000 * long_cache_read / 1_000_000
        + 10_000 * long_output / 1_000_000
    )


def test_gpt_5_5_has_no_separate_cache_write_premium():
    p = DEFAULT_PRICING["gpt-5.5"]
    assert p.input_per_mtok == 5.00
    assert p.output_per_mtok == 30.00
    assert p.cache_read_per_mtok == pytest.approx(0.50)
    assert p.cache_write_per_mtok is None

    # No published cache-write rate means conservative ordinary-input billing.
    assert p.compute_cost(
        input_tokens=1_000_000,
        output_tokens=0,
        cache_creation_tokens=1_000_000,
    ) == pytest.approx(5.00)


def test_moonshot_kimi_k27_standard_pricing_rates():
    pricing = DEFAULT_PRICING["kimi-k2.7-code"]

    assert pricing.input_per_mtok == pytest.approx(0.95)
    assert pricing.output_per_mtok == pytest.approx(4.00)
    assert pricing.cache_read_per_mtok == pytest.approx(0.19)
    assert pricing.cache_write_per_mtok is None
    assert pricing.compute_cost(
        input_tokens=2_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    ) == pytest.approx(5.14)


def test_moonshot_kimi_k27_highspeed_pricing_rates():
    pricing = DEFAULT_PRICING["kimi-k2.7-code-highspeed"]

    assert pricing.input_per_mtok == pytest.approx(1.90)
    assert pricing.output_per_mtok == pytest.approx(8.00)
    assert pricing.cache_read_per_mtok == pytest.approx(0.38)
    assert pricing.cache_write_per_mtok is None
    assert pricing.compute_cost(
        input_tokens=2_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    ) == pytest.approx(10.28)


def test_moonshot_kimi_k3_five_minute_cache_pricing():
    pricing = DEFAULT_PRICING["kimi-k3"]
    assert pricing.input_per_mtok == 3.00
    assert pricing.output_per_mtok == 15.00
    assert pricing.cache_write_per_mtok == 3.00
    assert pricing.cache_read_per_mtok == 0.30
    # Total input includes both cached subsets. Only one million of these
    # three million prompt tokens are uncached; no double subtraction.
    assert pricing.compute_cost(
        input_tokens=3_000_000,
        output_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    ) == pytest.approx(21.30)


def test_latest_verified_at_is_a_date():
    """LATEST_VERIFIED_AT is a stamp; operators read it to gauge
    table freshness. Must be ISO-shaped so downstream tooling can
    parse it."""
    import datetime
    parsed = datetime.date.fromisoformat(LATEST_VERIFIED_AT)
    assert parsed.year >= 2026


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5",
        "anthropic.claude-opus-5",
        "aws/anthropic/bedrock-claude-opus-5",
        "azure/anthropic/claude-opus-5",
    ],
)
def test_opus5_pricing_ids(model):
    pricing = DEFAULT_PRICING[model]
    assert pricing.input_per_mtok == 5.00
    assert pricing.output_per_mtok == 25.00
    assert pricing.cache_write_per_mtok == 6.25
    assert pricing.cache_read_per_mtok == 0.50


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5-1",
        "anthropic.claude-fable-5-1",
        "aws/anthropic/bedrock-claude-fable-5-1",
        "azure/anthropic/claude-fable-5-1",
    ],
)
def test_fable_5_1_pricing(model):
    pricing = DEFAULT_PRICING[model]
    assert pricing.input_per_mtok == 10.00
    assert pricing.output_per_mtok == 50.00
    assert pricing.cache_write_per_mtok == 12.50
    assert pricing.cache_read_per_mtok == 0.25


def test_mythos_5_1_pricing():
    pricing = DEFAULT_PRICING["claude-mythos-5-1"]
    assert pricing.input_per_mtok == 10.00
    assert pricing.output_per_mtok == 50.00
    assert pricing.cache_write_per_mtok == 12.50
    assert pricing.cache_read_per_mtok == 0.25


# ----------------------------------------------------------------------
# compute_cost math
# ----------------------------------------------------------------------


def test_compute_cost_pure_uncached_input():
    """1M input tokens × $5/M + 1M output × $25/M = $30."""
    p = ModelPricing(input_per_mtok=5.00, output_per_mtok=25.00)
    cost = p.compute_cost(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(30.0)


def test_compute_cost_cache_read_discount():
    """Cache reads bill at 10% of input. 1M all-cache-read input:
    1M × 5/M × 0.10 = $0.50."""
    p = ModelPricing(
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        cache_read_per_mtok=0.50,
    )
    cost = p.compute_cost(
        input_tokens=1_000_000, output_tokens=0,
        cache_read_tokens=1_000_000,
    )
    assert cost == pytest.approx(0.50)


def test_compute_cost_cache_write_premium():
    """Cache writes bill at 125% of input. 1M all-cache-write input:
    1M × 5/M × 1.25 = $6.25."""
    p = ModelPricing(
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        cache_write_per_mtok=6.25,
    )
    cost = p.compute_cost(
        input_tokens=1_000_000, output_tokens=0,
        cache_creation_tokens=1_000_000,
    )
    assert cost == pytest.approx(6.25)


def test_compute_cost_mixed_cache_breakdown():
    """1M input split as 400K cache-write + 400K cache-read + 200K
    uncached:
      uncached: 200K × 5/M = $1.00
      cache-write: 400K × 5/M × 1.25 = $2.50
      cache-read: 400K × 5/M × 0.10 = $0.20
      output: 0
      total: $3.70
    """
    p = ModelPricing(
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        cache_write_per_mtok=6.25,
        cache_read_per_mtok=0.50,
    )
    cost = p.compute_cost(
        input_tokens=1_000_000, output_tokens=0,
        cache_creation_tokens=400_000,
        cache_read_tokens=400_000,
    )
    assert cost == pytest.approx(3.70)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"long_context_input_per_mtok": 2.0},
        {
            "long_context_threshold_tokens": 272_000,
            "long_context_input_per_mtok": 2.0,
        },
        {
            "long_context_threshold_tokens": True,
            "long_context_input_per_mtok": 2.0,
            "long_context_output_per_mtok": 4.0,
        },
    ],
)
def test_model_pricing_rejects_incomplete_long_context_tier(kwargs):
    with pytest.raises(ValueError, match="long.context"):
        ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0, **kwargs)


def test_compute_cost_unknown_model_fails_closed(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    with pytest.raises(PricingConfigurationError, match="no pricing entry"):
        compute_cost(
            "no-such-model-xyz",
            input_tokens=1_000_000, output_tokens=1_000_000,
        )


@pytest.mark.parametrize("model", ["", "unknown"])
def test_compute_cost_placeholder_model_fails_closed(model):
    with pytest.raises(PricingConfigurationError, match="concrete non-empty"):
        compute_cost(model, input_tokens=1000, output_tokens=1000)


def test_compute_cost_glm_5_2_profile_is_free():
    assert compute_cost(
        "nvidia/zai-org/glm-5.2",
        input_tokens=3_000_000,
        output_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    ) == 0.0


# ----------------------------------------------------------------------
# YAML overrides
# ----------------------------------------------------------------------


def test_load_pricing_falls_through_to_defaults(tmp_path, monkeypatch):
    """No config file → DEFAULT_PRICING returned unchanged."""
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    table = load_pricing(spec_dir=tmp_path / "empty")
    assert table["aws/anthropic/bedrock-claude-opus-4-8"] == DEFAULT_PRICING["aws/anthropic/bedrock-claude-opus-4-8"]


def test_load_pricing_yaml_override_replaces_default(tmp_path, monkeypatch):
    """Operator-supplied YAML overrides a known model; built-in
    defaults for other models survive."""
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    cfg = tmp_path / "pricing.yaml"
    cfg.write_text(yaml.safe_dump({
        "aws/anthropic/bedrock-claude-opus-4-8": {
            "input_per_mtok": 99.99,  # ridiculously high to detect
            "output_per_mtok": 199.99,
        },
    }))
    table = load_pricing(config_path=cfg)
    assert table["aws/anthropic/bedrock-claude-opus-4-8"].input_per_mtok == 99.99
    assert table["aws/anthropic/bedrock-claude-opus-4-8"].output_per_mtok == 199.99
    # Other models survive.
    assert (
        table["aws/anthropic/bedrock-claude-sonnet-4-6"]
        == DEFAULT_PRICING["aws/anthropic/bedrock-claude-sonnet-4-6"]
    )


def test_load_pricing_yaml_adds_new_model(tmp_path, monkeypatch):
    """YAML can introduce a model id that isn't in DEFAULT_PRICING."""
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    cfg = tmp_path / "pricing.yaml"
    cfg.write_text(yaml.safe_dump({
        "claude-custom-test-model": {
            "input_per_mtok": 2.00,
            "output_per_mtok": 8.00,
            "cache_write_per_mtok": 2.50,
            "cache_read_per_mtok": 0.20,
        },
    }))
    table = load_pricing(config_path=cfg)
    assert "claude-custom-test-model" in table
    assert table["claude-custom-test-model"].input_per_mtok == 2.00
    assert table["claude-custom-test-model"].cache_write_per_mtok == 2.50
    assert table["claude-custom-test-model"].cache_read_per_mtok == 0.20


def test_load_pricing_rejects_flat_override_for_tiered_builtin(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    cfg = tmp_path / "pricing.yaml"
    cfg.write_text(yaml.safe_dump({
        "gpt-5.6-sol": {
            "input_per_mtok": 4.00,
            "output_per_mtok": 24.00,
            "cache_write_per_mtok": 5.00,
            "cache_read_per_mtok": 0.40,
        },
    }))

    with pytest.raises(
        ValueError,
        match="requires the complete long-context tier",
    ):
        load_pricing(config_path=cfg)


def test_load_pricing_accepts_complete_override_for_tiered_builtin(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    cfg = tmp_path / "pricing.yaml"
    cfg.write_text(yaml.safe_dump({
        "gpt-5.6-sol": {
            "input_per_mtok": 4.00,
            "output_per_mtok": 24.00,
            "cache_write_per_mtok": 5.00,
            "cache_read_per_mtok": 0.40,
            "long_context_threshold_tokens": 300_000,
            "long_context_input_per_mtok": 8.00,
            "long_context_output_per_mtok": 36.00,
            "long_context_cache_write_per_mtok": 10.00,
            "long_context_cache_read_per_mtok": 0.80,
        },
    }))

    pricing = load_pricing(config_path=cfg)["gpt-5.6-sol"]

    assert pricing == ModelPricing(
        input_per_mtok=4.00,
        output_per_mtok=24.00,
        cache_write_per_mtok=5.00,
        cache_read_per_mtok=0.40,
        long_context_threshold_tokens=300_000,
        long_context_input_per_mtok=8.00,
        long_context_output_per_mtok=36.00,
        long_context_cache_write_per_mtok=10.00,
        long_context_cache_read_per_mtok=0.80,
    )


def test_load_pricing_yaml_accepts_long_context_tier(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    cfg = tmp_path / "pricing.yaml"
    cfg.write_text(yaml.safe_dump({
        "tiered-model": {
            "input_per_mtok": 1.0,
            "output_per_mtok": 2.0,
            "long_context_threshold_tokens": 100,
            "long_context_input_per_mtok": 3.0,
            "long_context_output_per_mtok": 4.0,
            "long_context_cache_write_per_mtok": 3.75,
            "long_context_cache_read_per_mtok": 0.30,
        },
    }))

    pricing = load_pricing(config_path=cfg)["tiered-model"]

    assert pricing.long_context_threshold_tokens == 100
    assert pricing.compute_cost(
        input_tokens=101,
        output_tokens=10,
        cache_creation_tokens=20,
        cache_read_tokens=30,
    ) == pytest.approx(
        51 * 3.0 / 1_000_000
        + 20 * 3.75 / 1_000_000
        + 30 * 0.30 / 1_000_000
        + 10 * 4.0 / 1_000_000
    )


def test_hermetic_pricing_config_loads_explicit_zero_rates(monkeypatch):
    """CI's no-network model remains free through an explicit price entry."""
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    config = Path(__file__).parents[1] / ".github" / "pricing-hermetic.yaml"

    table = load_pricing(config_path=config)

    assert table["hermetic-no-network"] == ModelPricing(
        input_per_mtok=0.0,
        output_per_mtok=0.0,
        cache_write_per_mtok=0.0,
        cache_read_per_mtok=0.0,
    )


def test_load_pricing_rejects_legacy_multiplier_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    cfg = tmp_path / "legacy-pricing.yaml"
    cfg.write_text(yaml.safe_dump({
        "legacy-model": {
            "input_per_mtok": 1.0,
            "output_per_mtok": 2.0,
            "cache_write_multiplier": 1.25,
        },
    }))

    with pytest.raises(ValueError, match="unknown fields.*cache_write_multiplier"):
        load_pricing(config_path=cfg)


def test_load_pricing_rejects_negative_rate(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    cfg = tmp_path / "negative-pricing.yaml"
    cfg.write_text(yaml.safe_dump({
        "broken-model": {
            "input_per_mtok": -1.0,
            "output_per_mtok": 2.0,
        },
    }))

    with pytest.raises(ValueError, match="finite non-negative rate"):
        load_pricing(config_path=cfg)


def test_load_pricing_uses_configured_workspace_not_package_location(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    workspace = tmp_path / "external-workspace"
    (workspace / "crucible").mkdir(parents=True)
    cfg = workspace / ".crucible/pricing.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(yaml.safe_dump({
        "workspace-model": {
            "input_per_mtok": 7.0,
            "output_per_mtok": 21.0,
        },
    }))
    monkeypatch.setenv("CRUCIBLE_WORKSPACE_ROOT", str(workspace))

    table = load_pricing(spec_dir=tmp_path / "empty-spec")

    assert table["workspace-model"].input_per_mtok == 7.0


def test_load_pricing_raises_on_invalid_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("not_a_model:\n  input_per_mtok: [unterminated")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_pricing(config_path=cfg)


def test_load_pricing_raises_on_missing_required_field(tmp_path, monkeypatch):
    """An entry with no ``input_per_mtok`` raises with a clear error."""
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    cfg = tmp_path / "incomplete.yaml"
    cfg.write_text(yaml.safe_dump({
        "claude-broken": {"output_per_mtok": 10.0},  # no input_per_mtok
    }))
    with pytest.raises(ValueError, match="missing required field"):
        load_pricing(config_path=cfg)


def test_load_pricing_env_var_takes_precedence(tmp_path, monkeypatch):
    """CRUCIBLE_PRICING_CONFIG > --pricing-config arg > spec_dir > workspace."""
    env_cfg = tmp_path / "env.yaml"
    arg_cfg = tmp_path / "arg.yaml"
    env_cfg.write_text(yaml.safe_dump({
        "aws/anthropic/bedrock-claude-opus-4-8": {"input_per_mtok": 11.0, "output_per_mtok": 11.0},
    }))
    arg_cfg.write_text(yaml.safe_dump({
        "aws/anthropic/bedrock-claude-opus-4-8": {"input_per_mtok": 22.0, "output_per_mtok": 22.0},
    }))
    monkeypatch.setenv("CRUCIBLE_PRICING_CONFIG", str(env_cfg))
    table = load_pricing(config_path=arg_cfg)
    assert table["aws/anthropic/bedrock-claude-opus-4-8"].input_per_mtok == 11.0  # env wins


# ----------------------------------------------------------------------
# Role → model → pricing lookup
# ----------------------------------------------------------------------


def test_lookup_pricing_for_role_resolves_via_routing():
    routing = BackendRouting(
        default=RoleConfig(
            backend="claude", model="aws/anthropic/bedrock-claude-opus-4-8",
        ),
        roles={
            "plan_critic": RoleConfig(
                backend="claude", model="aws/anthropic/bedrock-claude-sonnet-4-6",
            ),
        },
    )
    pricing, model = lookup_pricing_for_role("plan_critic", routing=routing)
    assert model == "aws/anthropic/bedrock-claude-sonnet-4-6"
    assert pricing == DEFAULT_PRICING["aws/anthropic/bedrock-claude-sonnet-4-6"]


def test_lookup_pricing_for_role_unknown_role_is_rejected():
    routing = BackendRouting(
        default=RoleConfig(backend="claude", model="aws/anthropic/bedrock-claude-opus-4-8"),
        roles={},
    )
    with pytest.raises(ValueError, match="unknown backend role"):
        lookup_pricing_for_role("never_named", routing=routing)


def test_lookup_pricing_for_role_unknown_model_fails_closed():
    routing = BackendRouting(
        default=RoleConfig(backend="local", model="qwen-local"),
        roles={},
    )
    with pytest.raises(PricingConfigurationError, match="qwen-local"):
        lookup_pricing_for_role("implementer", routing=routing)


# ----------------------------------------------------------------------
# CostAccumulator: backend-reported cost is authoritative
# ----------------------------------------------------------------------
#
# The CostAccumulator previously carried a vendor-specific backfill
# branch that recomputed Codex cost from tokens when cost_usd=0. That
# coupling masked a real bug: CodexBackend.invoke() was returning
# Usage() with all zeros (including tokens), so the gated backfill
# never fired. The fix moves cost computation into the backend (where
# it belongs architecturally — see
# crucible/backends/codex.py::_harvest_usage_from_events). The
# accumulator now simply sums whatever the backend reports.


def test_accumulator_trusts_backend_reported_cost():
    """The accumulator records exactly what ``Usage.cost_usd`` says.
    For Codex, the backend computes cost via
    ``_harvest_usage_from_events`` BEFORE the Usage reaches the
    accumulator — see crucible/backends/codex.py for the harvest path
    that walks the SDK's ``thread/tokenUsage/updated`` event."""
    a = CostAccumulator()
    expected = compute_cost(
        "gpt-5.5", input_tokens=100_000, output_tokens=20_000,
    )
    a.record(
        role="goal_judge",
        usage=Usage(
            input_tokens=100_000, output_tokens=20_000,
            cost_usd=expected,  # backend already computed this
        ),
        backend_name="codex",
        model="gpt-5.5",
    )
    assert a.total_cost_usd == pytest.approx(expected)
    assert a.cost_by_backend()["codex"] == pytest.approx(expected)
    assert expected > 0.0  # sanity — the test is only meaningful if non-zero


def test_accumulator_does_not_synthesize_cost_from_tokens():
    """Negative case: if a backend regresses to reporting cost_usd=0
    with positive tokens, the accumulator records $0 — no hidden
    backfill. The symptom (``total_usd: 0``) surfaces in summary.json
    immediately, pointing the operator at the backend's harvest
    path rather than silently masking it inside the accumulator."""
    a = CostAccumulator()
    a.record(
        role="goal_judge",
        usage=Usage(
            input_tokens=100_000, output_tokens=20_000, cost_usd=0.0,
        ),
        backend_name="codex",
        model="gpt-5.5",
    )
    assert a.total_cost_usd == 0.0


def test_accumulator_does_not_overwrite_nonzero_claude_cost():
    """When Claude reports a positive cost_usd (its bundled CLI's
    own number), the accumulator MUST NOT recompute — that number
    is authoritative."""
    a = CostAccumulator()
    a.record(
        role="goal_red",
        usage=Usage(
            input_tokens=100_000, output_tokens=20_000, cost_usd=2.50,
        ),
        backend_name="claude",
        model="aws/anthropic/bedrock-claude-opus-4-8",
    )
    # The accumulator should record exactly 2.50, NOT compute_cost(...)
    # which would be a different number.
    assert a.total_cost_usd == pytest.approx(2.50)


def test_accumulator_does_not_backfill_local_zero():
    """Local backends correctly report cost_usd=0 because operators
    pay machine cost out-of-band. Accumulator MUST NOT inflate
    Local cost via the pricing table (the model id may not be in
    the table, but even if it were, attributing API-style cost to
    a self-hosted server would be wrong)."""
    a = CostAccumulator()
    a.record(
        role="rca",
        usage=Usage(
            input_tokens=50_000, output_tokens=10_000, cost_usd=0.0,
        ),
        backend_name="local",
        model="qwen2.5-coder-32b",
    )
    assert a.total_cost_usd == 0.0
    assert a.cost_by_backend() == {"local": 0.0}
