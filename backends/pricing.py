# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Per-model pricing table for backend cost computation.

Backends own the ``Usage.cost_usd`` contract. The Claude CLI normally
reports ``total_cost_usd`` directly. Codex harvests
``thread/tokenUsage/updated`` and computes cost through this table, as does
the OpenAI-compatible adapter when its response contains token usage. Every
model must have an explicit entry, including deliberately zero-priced local or
hosted models; an unknown model is a configuration error, never guessed free.

The table also exists for two operator-facing reasons:

1. **Pre-run cost projection.** Operators want to estimate the cost
   of a 500-instance SWE-bench pass BEFORE launching it. They can
   query the table given a role + token-budget estimate.

2. **Cross-check / drift detection.** When a provider-reported number differs
   from the table by >X%, that's a signal: either we're stale or the
   SDK is buggy. Either way, worth surfacing.

Discovery (operator override at runtime):

1. ``CRUCIBLE_PRICING_CONFIG`` env var → absolute YAML path
2. An explicit ``config_path`` passed to :func:`load_pricing`
3. ``<spec_dir>/.crucible/pricing.yaml``
4. ``<workspace>/.crucible/pricing.yaml``
5. Built-in defaults (this module's ``DEFAULT_PRICING``)

YAML schema::

    # USD per million tokens. A null/omitted cache rate means that no
    # distinct rate is published, so those tokens bill as ordinary input.
    "aws/anthropic/bedrock-claude-opus-4-8":
      input_per_mtok: 5.00
      output_per_mtok: 25.00
      cache_write_per_mtok: 6.25
      cache_read_per_mtok: 0.50
    "gpt-6-astra":
      input_per_mtok: 10.00
      output_per_mtok: 50.00
      cache_write_per_mtok: 12.50
      cache_read_per_mtok: 1.00
      long_context_threshold_tokens: 272000
      long_context_input_per_mtok: 20.00
      long_context_output_per_mtok: 75.00
      long_context_cache_write_per_mtok: 25.00
      long_context_cache_read_per_mtok: 2.00

An override for a built-in model with long-context pricing must repeat the
complete long-context tier. This prevents a legacy flat-rate override from
silently disabling the higher-context rates.

VERIFY THE PRICES BEFORE TRUSTING THE NUMBERS BELOW. The
``LATEST_VERIFIED_AT`` constant is the date the maintainer last
audited the table against vendor pricing pages. Pricing changes
silently from vendors; operators on tight budgets should re-verify
quarterly or override via YAML.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from crucible.common.workspace import workspace_root

logger = logging.getLogger(__name__)


# Date the table below was last audited against the applicable provider table.
# Bump this when you update DEFAULT_PRICING; operator-side tooling
# may warn when the table is older than N months.
LATEST_VERIFIED_AT: str = "2026-09-13"


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token pricing for one model.

    Every populated field is an absolute USD-per-million-token rate.
    ``None`` means the vendor publishes no distinct cache rate; cost
    accounting then conservatively bills those tokens as ordinary input.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None
    long_context_threshold_tokens: int | None = None
    long_context_input_per_mtok: float | None = None
    long_context_output_per_mtok: float | None = None
    long_context_cache_write_per_mtok: float | None = None
    long_context_cache_read_per_mtok: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_per_mtok",
            "output_per_mtok",
            "cache_write_per_mtok",
            "cache_read_per_mtok",
            "long_context_input_per_mtok",
            "long_context_output_per_mtok",
            "long_context_cache_write_per_mtok",
            "long_context_cache_read_per_mtok",
        ):
            rate = getattr(self, name)
            if rate is not None and (not math.isfinite(rate) or rate < 0):
                raise ValueError(f"{name} must be a finite non-negative rate")
        threshold = self.long_context_threshold_tokens
        long_rates = (
            self.long_context_input_per_mtok,
            self.long_context_output_per_mtok,
            self.long_context_cache_write_per_mtok,
            self.long_context_cache_read_per_mtok,
        )
        if threshold is None:
            if any(rate is not None for rate in long_rates):
                raise ValueError(
                    "long-context rates require "
                    "long_context_threshold_tokens"
                )
        else:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, int)
                or threshold <= 0
            ):
                raise ValueError(
                    "long_context_threshold_tokens must be a positive integer"
                )
            if (
                self.long_context_input_per_mtok is None
                or self.long_context_output_per_mtok is None
            ):
                raise ValueError(
                    "long-context pricing requires input and output rates"
                )

    def compute_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Compute USD cost for one backend turn.

        ``input_tokens`` is the TOTAL input tokens (including any
        that hit the cache). ``cache_creation_tokens`` and
        ``cache_read_tokens`` are subsets that should be re-billed
        at their absolute cache rates; the uncached portion bills at
        the base input rate. A missing cache rate falls back to the
        input rate rather than inventing a discount or premium.
        """
        uncached_input = max(
            0,
            input_tokens - cache_creation_tokens - cache_read_tokens,
        )
        use_long_context = (
            self.long_context_threshold_tokens is not None
            and input_tokens > self.long_context_threshold_tokens
        )
        input_per_mtok = (
            self.long_context_input_per_mtok
            if use_long_context
            else self.input_per_mtok
        )
        output_per_mtok = (
            self.long_context_output_per_mtok
            if use_long_context
            else self.output_per_mtok
        )
        cache_write_per_mtok = (
            self.long_context_cache_write_per_mtok
            if use_long_context
            else self.cache_write_per_mtok
        )
        cache_read_per_mtok = (
            self.long_context_cache_read_per_mtok
            if use_long_context
            else self.cache_read_per_mtok
        )
        assert input_per_mtok is not None
        assert output_per_mtok is not None
        input_rate = input_per_mtok / 1_000_000.0
        output_rate = output_per_mtok / 1_000_000.0
        cache_write_rate = (
            input_per_mtok
            if cache_write_per_mtok is None
            else cache_write_per_mtok
        ) / 1_000_000.0
        cache_read_rate = (
            input_per_mtok
            if cache_read_per_mtok is None
            else cache_read_per_mtok
        ) / 1_000_000.0
        return (
            uncached_input * input_rate
            + cache_creation_tokens * cache_write_rate
            + cache_read_tokens * cache_read_rate
            + output_tokens * output_rate
        )


class PricingConfigurationError(ValueError):
    """A routed model has no explicit positive- or zero-cost price entry."""


# Built-in default pricing.
#
# Active comparison rates are verified against their first-party pricing
# pages. Mythos 5.1 was added and Anthropic rates were rechecked on
# 2026-09-13; retained model rates were last rechecked during the preceding
# release reviews:
# - Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
#   (canonical; the marketing page at claude.com/pricing redirects)
# - OpenAI:    https://developers.openai.com/api/docs/pricing
#              https://developers.openai.com/api/docs/guides/prompt-caching
# - Moonshot:  https://www.kimi.com/resources/kimi-k2-7-code-pricing
#
# Bump LATEST_VERIFIED_AT above when re-auditing. Providers change
# prices without API notification; treat anything older than ~3
# months as suspect.
#
# CACHE NOTES:
# - Anthropic: 5-min cache write = 1.25× input and 1-hr cache write = 2×
#   input. Built-ins use the 5-min absolute rate; operators using
#   ``cache_control`` ``ttl="1h"`` should override. Cache-read rates are
#   model-specific; Fable 5.1 publishes $0.25/M rather than a family ratio.
# - OpenAI GPT-5.6 and GPT-6 Astra: cache writes cost 1.25x ordinary input
#   and cache reads cost 0.10x. Requests above 272K input tokens use the
#   published long-context rates for the entire request.
# - OpenAI GPT-5.5: no separate cache-write price is published (``-``), so
#   cache creation remains billed as ordinary input (1.0×). Cache reads are
#   $0.50/Mtok, or 0.10× input.
#
# Operators with negotiated rates (enterprise contracts, AWS Bedrock
# vs direct, etc.) should override via .crucible/pricing.yaml rather
# than editing this file.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    # ---------------- Anthropic Claude family ----------------
    # Fable 5.1 first-party organization-session route. Anthropic publishes
    # $10/M input, $50/M output, and $0.25/M cache reads. The 5-minute cache
    # write rate is the published 1.25x input multiplier.
    "claude-fable-5-1": ModelPricing(
        input_per_mtok=10.00, output_per_mtok=50.00,
        cache_write_per_mtok=12.50, cache_read_per_mtok=0.25,
    ),
    "anthropic.claude-fable-5-1": ModelPricing(
        input_per_mtok=10.00, output_per_mtok=50.00,
        cache_write_per_mtok=12.50, cache_read_per_mtok=0.25,
    ),
    "aws/anthropic/bedrock-claude-fable-5-1": ModelPricing(
        input_per_mtok=10.00, output_per_mtok=50.00,
        cache_write_per_mtok=12.50, cache_read_per_mtok=0.25,
    ),
    "azure/anthropic/claude-fable-5-1": ModelPricing(
        input_per_mtok=10.00, output_per_mtok=50.00,
        cache_write_per_mtok=12.50, cache_read_per_mtok=0.25,
    ),
    # Mythos 5.1 is a separately served, limited-availability identity with
    # the same published token and prompt-cache rates as Fable 5.1.
    "claude-mythos-5-1": ModelPricing(
        input_per_mtok=10.00, output_per_mtok=50.00,
        cache_write_per_mtok=12.50, cache_read_per_mtok=0.25,
    ),
    # Opus 5 launched 2026-07-24 at the same base and prompt-cache
    # rates as Opus 4.8. Keep each provider/gateway identifier explicit:
    # model substitution must never hide behind a pricing alias.
    "claude-opus-5": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=25.00,
        cache_write_per_mtok=6.25, cache_read_per_mtok=0.50,
    ),
    "anthropic.claude-opus-5": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=25.00,
        cache_write_per_mtok=6.25, cache_read_per_mtok=0.50,
    ),
    "aws/anthropic/bedrock-claude-opus-5": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=25.00,
        cache_write_per_mtok=6.25, cache_read_per_mtok=0.50,
    ),
    "azure/anthropic/claude-opus-5": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=25.00,
        cache_write_per_mtok=6.25, cache_read_per_mtok=0.50,
    ),
    # Opus 4.5/4.6/4.8 — flagship; high-quality long-context reasoning.
    # All share pricing on Anthropic's published table.
    "aws/anthropic/bedrock-claude-opus-4-8": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=25.00,
        cache_write_per_mtok=6.25, cache_read_per_mtok=0.50,
    ),
    # Azure-hosted gateway id used by explicit Claude gateway profiles.
    # compute_cost()
    # uses DEFAULT_PRICING when no override table is threaded, so the
    # gateway ids MUST be priced here or --budget-usd records $0 and never
    # fires (verified 2026-06-15 against Anthropic public pricing).
    "azure/anthropic/claude-opus-4-8": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=25.00,
        cache_write_per_mtok=6.25, cache_read_per_mtok=0.50,
    ),
    # Bare first-party id — the form the `claude login` OAuth path
    # requires (the bedrock-prefixed id 404s under that auth). Explicit
    # direct-Claude profiles need this priced for budget enforcement.
    "claude-opus-4-8": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=25.00,
        cache_write_per_mtok=6.25, cache_read_per_mtok=0.50,
    ),
    "claude-opus-4-6": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=25.00,
        cache_write_per_mtok=6.25, cache_read_per_mtok=0.50,
    ),
    "claude-opus-4-5": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=25.00,
        cache_write_per_mtok=6.25, cache_read_per_mtok=0.50,
    ),
    # Opus 4.1 — prior generation, 3× more expensive than 4.5+.
    "claude-opus-4-1": ModelPricing(
        input_per_mtok=15.00, output_per_mtok=75.00,
        cache_write_per_mtok=18.75, cache_read_per_mtok=1.50,
    ),
    # Sonnet 4.5/4.6 — mid-tier models available to explicit Claude profiles.
    "aws/anthropic/bedrock-claude-sonnet-4-6": ModelPricing(
        input_per_mtok=3.00, output_per_mtok=15.00,
        cache_write_per_mtok=3.75, cache_read_per_mtok=0.30,
    ),
    # Azure-hosted gateway id (see the opus note above).
    "azure/anthropic/claude-sonnet-4-6": ModelPricing(
        input_per_mtok=3.00, output_per_mtok=15.00,
        cache_write_per_mtok=3.75, cache_read_per_mtok=0.30,
    ),
    # Bare first-party id — see the claude-opus-4-8 note above; direct
    # OAuth profiles use this form.
    "claude-sonnet-4-6": ModelPricing(
        input_per_mtok=3.00, output_per_mtok=15.00,
        cache_write_per_mtok=3.75, cache_read_per_mtok=0.30,
    ),
    "claude-sonnet-4-5": ModelPricing(
        input_per_mtok=3.00, output_per_mtok=15.00,
        cache_write_per_mtok=3.75, cache_read_per_mtok=0.30,
    ),
    "claude-sonnet-4-20250514": ModelPricing(
        input_per_mtok=3.00, output_per_mtok=15.00,
        cache_write_per_mtok=3.75, cache_read_per_mtok=0.30,
    ),
    # Haiku 4.5 — fastest, cheapest; usable for cheap reviewer paths.
    # Azure-hosted gateway id (see the opus note above).
    "azure/anthropic/claude-haiku-4-5": ModelPricing(
        input_per_mtok=1.00, output_per_mtok=5.00,
        cache_write_per_mtok=1.25, cache_read_per_mtok=0.10,
    ),
    "claude-haiku-4-5": ModelPricing(
        input_per_mtok=1.00, output_per_mtok=5.00,
        cache_write_per_mtok=1.25, cache_read_per_mtok=0.10,
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        input_per_mtok=1.00, output_per_mtok=5.00,
        cache_write_per_mtok=1.25, cache_read_per_mtok=0.10,
    ),
    # Haiku 3.5 — retired on first-party, still on Bedrock/Vertex.
    "claude-haiku-3-5": ModelPricing(
        input_per_mtok=0.80, output_per_mtok=4.00,
        cache_write_per_mtok=1.00, cache_read_per_mtok=0.08,
    ),

    # ---------------- OpenAI GPT family ----------------
    # GPT-5.6 Luna — lowest-cost tier. OpenAI reduced its standard rates on
    # 2026-07-30; long-context pricing applies above 272K input tokens.
    "gpt-5.6-luna": ModelPricing(
        input_per_mtok=0.20, output_per_mtok=1.20,
        cache_write_per_mtok=0.25, cache_read_per_mtok=0.02,
        long_context_threshold_tokens=272_000,
        long_context_input_per_mtok=0.40,
        long_context_output_per_mtok=1.80,
        long_context_cache_write_per_mtok=0.50,
        long_context_cache_read_per_mtok=0.04,
    ),
    # GPT-5.6 Terra — balanced price/performance tier. OpenAI reduced its
    # standard rates on 2026-07-30.
    "gpt-5.6-terra": ModelPricing(
        input_per_mtok=2.00, output_per_mtok=12.00,
        cache_write_per_mtok=2.50, cache_read_per_mtok=0.20,
        long_context_threshold_tokens=272_000,
        long_context_input_per_mtok=4.00,
        long_context_output_per_mtok=18.00,
        long_context_cache_write_per_mtok=5.00,
        long_context_cache_read_per_mtok=0.40,
    ),
    # GPT-5.6 Sol — frozen Codex comparison route. These promotional standard
    # rates are published through at least 2026-11-21.
    "gpt-5.6-sol": ModelPricing(
        input_per_mtok=4.00, output_per_mtok=20.00,
        cache_write_per_mtok=5.00, cache_read_per_mtok=0.40,
        long_context_threshold_tokens=272_000,
        long_context_input_per_mtok=8.00,
        long_context_output_per_mtok=30.00,
        long_context_cache_write_per_mtok=10.00,
        long_context_cache_read_per_mtok=0.80,
    ),
    # GPT-6 Astra — CodexBackend default and first-party ChatGPT-session
    # route. Requests above 272K input tokens use OpenAI's published
    # whole-request long-context rates.
    "gpt-6-astra": ModelPricing(
        input_per_mtok=10.00, output_per_mtok=50.00,
        cache_write_per_mtok=12.50, cache_read_per_mtok=1.00,
        long_context_threshold_tokens=272_000,
        long_context_input_per_mtok=20.00,
        long_context_output_per_mtok=75.00,
        long_context_cache_write_per_mtok=25.00,
        long_context_cache_read_per_mtok=2.00,
    ),
    # GPT-5.5 — prior CodexBackend default; retained for historical runs.
    "gpt-5.5": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=30.00,
        cache_write_per_mtok=None, cache_read_per_mtok=0.50,
    ),
    # GPT-5.5 Pro — premium tier for research-grade problems.
    # No cached-input rate published (large-context only).
    "gpt-5.5-pro": ModelPricing(
        input_per_mtok=30.00, output_per_mtok=180.00,
        cache_write_per_mtok=None, cache_read_per_mtok=None,
    ),
    # GPT-5.4 — prior flagship, half the price of 5.5.
    "gpt-5.4": ModelPricing(
        input_per_mtok=2.50, output_per_mtok=15.00,
        cache_write_per_mtok=None, cache_read_per_mtok=0.25,
    ),
    "gpt-5.4-mini": ModelPricing(
        input_per_mtok=0.75, output_per_mtok=4.50,
        cache_write_per_mtok=None, cache_read_per_mtok=0.075,
    ),
    "gpt-5.4-nano": ModelPricing(
        input_per_mtok=0.20, output_per_mtok=1.25,
        cache_write_per_mtok=None, cache_read_per_mtok=0.02,
    ),
    "gpt-5.4-pro": ModelPricing(
        input_per_mtok=30.00, output_per_mtok=180.00,
        cache_write_per_mtok=None, cache_read_per_mtok=None,
    ),
    # chat-latest — alias for the current flagship; same as gpt-5.5.
    "chat-latest": ModelPricing(
        input_per_mtok=5.00, output_per_mtok=30.00,
        cache_write_per_mtok=None, cache_read_per_mtok=0.50,
    ),
    # Moonshot public API standard route. Cached input is a published read
    # discount; no separate cache-write rate is published, so writes fall
    # back to the ordinary uncached-input rate.
    "kimi-k2.7-code": ModelPricing(
        input_per_mtok=0.95, output_per_mtok=4.00,
        cache_write_per_mtok=None, cache_read_per_mtok=0.19,
    ),
    # Moonshot's distinct high-speed K2.7 Code route uses the same context
    # and cache semantics at exactly twice the standard token rates.
    "kimi-k2.7-code-highspeed": ModelPricing(
        input_per_mtok=1.90, output_per_mtok=8.00,
        cache_write_per_mtok=None, cache_read_per_mtok=0.38,
    ),
    # K3 public Moonshot route, verified 2026-09-24 against
    # https://platform.kimi.ai/docs/guide/context-caching and
    # https://www.kimi.com/en/blog/kimi-k3 . The dedicated adapter explicitly
    # binds implicit caching to a five-minute TTL. One-hour writes cost $6
    # per MTok and are deliberately not admitted by that route. Input is the
    # inclusive prompt total; read/write counters are disjoint subsets.
    "kimi-k3": ModelPricing(
        input_per_mtok=3.00, output_per_mtok=15.00,
        cache_write_per_mtok=3.00, cache_read_per_mtok=0.30,
    ),
    # DeepSeek public native Responses Flash route. Automatic-cache hits use
    # the published discount; all non-hit input uses the cache-miss rate.
    # No peak-hours multiplier was published in the DSK-00a freeze.
    "deepseek-v4-flash": ModelPricing(
        input_per_mtok=0.14, output_per_mtok=0.28,
        cache_write_per_mtok=None, cache_read_per_mtok=0.0028,
    ),
    # Explicit zero-cost routes. These hosted models currently publish no
    # per-token charge to Crucible operators. Keeping them in the same table
    # makes zero a reviewed configuration value rather than an unknown-model
    # fallback. Operator-hosted local models must likewise declare a zero-rate
    # entry in pricing.yaml.
    "nvidia/nvidia/nemotron-3-super-v3": ModelPricing(
        input_per_mtok=0.0, output_per_mtok=0.0,
    ),
    "nvidia/nvidia/nemotron-3-ultra-preview": ModelPricing(
        input_per_mtok=0.0, output_per_mtok=0.0,
    ),
    "nvidia/minimaxai/minimax-m2.7": ModelPricing(
        input_per_mtok=0.0, output_per_mtok=0.0,
    ),
    "nvidia/deepseek-ai/deepseek-v4-pro": ModelPricing(
        input_per_mtok=0.0, output_per_mtok=0.0,
    ),
    "nvidia/moonshotai/kimi-k2.6": ModelPricing(
        input_per_mtok=0.0, output_per_mtok=0.0,
    ),
    "nvidia/zai-org/glm-5.2": ModelPricing(
        input_per_mtok=0.0, output_per_mtok=0.0,
    ),
}


def _candidate_paths(
    *,
    config_path: Path | str | None,
    spec_dir: Path | str | None,
) -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("CRUCIBLE_PRICING_CONFIG", "").strip()
    if env:
        out.append(Path(env))
    if config_path is not None:
        out.append(Path(config_path))
    if spec_dir is not None:
        out.append(Path(spec_dir) / ".crucible" / "pricing.yaml")
    out.append(workspace_root() / ".crucible" / "pricing.yaml")
    return out


def _parse_pricing_doc(doc: dict) -> dict[str, ModelPricing]:
    if not isinstance(doc, dict):
        raise ValueError("pricing config root must be a mapping")
    out: dict[str, ModelPricing] = {}
    allowed_fields = frozenset({
        "input_per_mtok",
        "output_per_mtok",
        "cache_write_per_mtok",
        "cache_read_per_mtok",
        "long_context_threshold_tokens",
        "long_context_input_per_mtok",
        "long_context_output_per_mtok",
        "long_context_cache_write_per_mtok",
        "long_context_cache_read_per_mtok",
    })
    long_context_fields = frozenset({
        "long_context_threshold_tokens",
        "long_context_input_per_mtok",
        "long_context_output_per_mtok",
        "long_context_cache_write_per_mtok",
        "long_context_cache_read_per_mtok",
    })
    for model_id, raw in doc.items():
        if not isinstance(raw, dict):
            raise ValueError(
                f"pricing[{model_id!r}]: expected mapping, got "
                f"{type(raw).__name__}"
            )
        unknown_fields = sorted(set(raw) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"pricing[{model_id!r}]: unknown fields {unknown_fields}; "
                f"allowed fields are {sorted(allowed_fields)}"
            )
        normalized_model_id = str(model_id)
        built_in = DEFAULT_PRICING.get(normalized_model_id)
        if (
            built_in is not None
            and built_in.long_context_threshold_tokens is not None
        ):
            incomplete_long_fields = sorted(
                field
                for field in long_context_fields
                if field not in raw or raw[field] is None
            )
            if incomplete_long_fields:
                raise ValueError(
                    f"pricing[{model_id!r}]: overriding a built-in model "
                    "with long-context pricing requires the complete "
                    "long-context tier; missing or null fields "
                    f"{incomplete_long_fields}"
                )
        try:
            cache_write = raw.get("cache_write_per_mtok")
            cache_read = raw.get("cache_read_per_mtok")
            threshold = raw.get("long_context_threshold_tokens")
            long_cache_write = raw.get(
                "long_context_cache_write_per_mtok"
            )
            long_cache_read = raw.get("long_context_cache_read_per_mtok")
            out[normalized_model_id] = ModelPricing(
                input_per_mtok=float(raw["input_per_mtok"]),
                output_per_mtok=float(raw["output_per_mtok"]),
                cache_write_per_mtok=(
                    None if cache_write is None else float(cache_write)
                ),
                cache_read_per_mtok=(
                    None if cache_read is None else float(cache_read)
                ),
                long_context_threshold_tokens=(
                    None if threshold is None else threshold
                ),
                long_context_input_per_mtok=(
                    None
                    if raw.get("long_context_input_per_mtok") is None
                    else float(raw["long_context_input_per_mtok"])
                ),
                long_context_output_per_mtok=(
                    None
                    if raw.get("long_context_output_per_mtok") is None
                    else float(raw["long_context_output_per_mtok"])
                ),
                long_context_cache_write_per_mtok=(
                    None
                    if long_cache_write is None
                    else float(long_cache_write)
                ),
                long_context_cache_read_per_mtok=(
                    None
                    if long_cache_read is None
                    else float(long_cache_read)
                ),
            )
        except KeyError as e:
            raise ValueError(
                f"pricing[{model_id!r}]: missing required field {e!s}"
            ) from e
    return out


def load_pricing(
    *,
    config_path: Path | str | None = None,
    spec_dir: Path | str | None = None,
) -> dict[str, ModelPricing]:
    """Discover + parse a pricing config, falling back to defaults.

    The returned dict is a MERGE: built-in ``DEFAULT_PRICING`` with
    operator overrides layered on top. Unknown models in the YAML
    add to the table; known models are overridden.

    Malformed YAML raises ``ValueError``. No-config returns a copy
    of ``DEFAULT_PRICING``.
    """
    import yaml  # type: ignore[import-untyped]

    merged: dict[str, ModelPricing] = dict(DEFAULT_PRICING)
    for path in _candidate_paths(
        config_path=config_path, spec_dir=spec_dir,
    ):
        if not path.is_file():
            continue
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(
                f"pricing config {path} is not valid YAML: {exc}"
            ) from exc
        overrides = _parse_pricing_doc(doc or {})
        merged.update(overrides)
        logger.info(
            "crucible: loaded pricing overrides from %s (%d entries)",
            path, len(overrides),
        )
        # First match wins. The discovery order ensures operator
        # overrides at higher-priority paths beat workspace defaults.
        return merged
    logger.debug(
        "crucible: no pricing config found; using DEFAULT_PRICING "
        "(verified %s)", LATEST_VERIFIED_AT,
    )
    return merged


def compute_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    pricing: dict[str, ModelPricing] | None = None,
) -> float:
    """Compute USD cost for a single backend call.

    ``pricing`` may be passed for an already-loaded operator table; when None,
    normal pricing discovery applies. Unknown model ids raise
    :class:`PricingConfigurationError` before a zero-cost assumption can
    disable budget enforcement. Local models must declare explicit zero rates.
    """
    entry = pricing_for_model(model, pricing=pricing)
    return entry.compute_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )


def pricing_for_model(
    model: str,
    *,
    pricing: dict[str, ModelPricing] | None = None,
) -> ModelPricing:
    """Return an explicit model price or fail closed.

    ``pricing=None`` performs normal environment/spec/workspace discovery.
    Callers that already froze a table may pass it directly.
    """
    if not isinstance(model, str) or not model.strip() or model == "unknown":
        raise PricingConfigurationError(
            "model pricing requires a concrete non-empty model id"
        )
    table = pricing if pricing is not None else load_pricing()
    try:
        return table[model]
    except KeyError as exc:
        raise PricingConfigurationError(
            f"no pricing entry for model {model!r}; add explicit positive "
            "rates or explicit zero rates in .crucible/pricing.yaml"
        ) from exc


def lookup_pricing_for_role(
    role: str,
    *,
    routing,
    pricing: dict[str, ModelPricing] | None = None,
) -> tuple[ModelPricing, str]:
    """Resolve a role through routing → model → pricing entry.

    Returns ``(pricing_entry, model_name)``. Missing models or prices are
    configuration errors; free/local routes use explicit zero-rate entries.

    Operator-side helper for cost projection. Workflows that just
    need the cost number for an observed call should call
    ``compute_cost(model, ...)`` directly with the model the
    backend reports.
    """
    cfg = routing.for_role(role)
    model = cfg.model
    if model is None:
        raise PricingConfigurationError(
            f"backend role {role!r} has no concrete model for pricing"
        )
    return pricing_for_model(model, pricing=pricing), model


__all__ = [
    "DEFAULT_PRICING",
    "LATEST_VERIFIED_AT",
    "ModelPricing",
    "PricingConfigurationError",
    "compute_cost",
    "load_pricing",
    "lookup_pricing_for_role",
    "pricing_for_model",
]
