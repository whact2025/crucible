# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Load ``CruciblePolicies`` from a spec.yaml's nested blocks.

The yaml shape mirrors the dataclass nesting:

    plan_policy:           # (assay-side — handled by assay.Spec.from_yaml)
      max_critique_rounds: 4
      ...
    breach_policy:
      enabled: true
      max_rounds: 5
      breaker_rubric: swebench_impl_breaker
      ...
    review_policy:
      rubrics: [diff_scope, tier_completeness]

Missing blocks → dataclass defaults. Unknown keys → ValueError (the
caller passed an unsupported field). Supported block names track the consumed
policy surface in :mod:`crucible.common.policies`.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any

from .policies import (
    BreachPolicy,
    CruciblePolicies,
    ReviewPolicy,
)


def _is_tuple_typed(cls, field_name: str) -> bool:
    """Return True iff ``cls.<field_name>`` is annotated as a tuple
    type (e.g. ``tuple[str, ...]``). Uses ``typing.get_type_hints`` to
    resolve the string annotation that ``from __future__ import
    annotations`` produces, then ``typing.get_origin`` to inspect the
    runtime generic origin.

    This avoids treating unrelated annotations whose rendered names merely
    contain ``"tuple"`` as tuple fields.
    """
    try:
        hints = typing.get_type_hints(cls)
    except (NameError, AttributeError):
        # If the field's annotation references an unresolved name,
        # fall back to "not tuple" — the caller will pass the value
        # through unchanged. Loud-fail is the wrong move here because
        # the field IS valid (it just has a forward-ref annotation
        # the resolver can't materialize at load time).
        return False
    field_type = hints.get(field_name)
    if field_type is None:
        return False
    return typing.get_origin(field_type) is tuple


def _kwargs_for(cls, block: dict[str, Any] | None) -> dict[str, Any]:
    """Convert a yaml block dict into kwargs for the dataclass cls.

    Validates that every key in ``block`` is a declared field on
    ``cls`` so a typo'd key surfaces as a ValueError instead of
    silently inheriting the default.
    """
    if not block:
        return {}
    if not isinstance(block, dict):
        raise ValueError(
            f"{cls.__name__} block must be a mapping, got {type(block).__name__}"
        )
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(block) - valid
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown field(s) {sorted(unknown)}. "
            f"Valid fields: {sorted(valid)}"
        )
    # Tuple-typed fields: yaml gives a list; convert via proper type
    # introspection (``typing.get_origin``), not substring match on
    # the str() of the annotation.
    out: dict[str, Any] = {}
    for k, v in block.items():
        if _is_tuple_typed(cls, k) and isinstance(v, list):
            v = tuple(v)
        out[k] = v
    return out


def load_policies_from_yaml(raw: dict[str, Any] | None) -> CruciblePolicies:
    """Parse the crucible-layer policy blocks from a yaml dict.

    Accepts the raw dict (e.g., from ``yaml.safe_load(spec_yaml_text)``)
    or ``None`` — ``yaml.safe_load`` of an empty spec.yaml returns
    ``None``, and per-product synthesized specs (e.g., SWE-bench's
    starter spec) carry no policy blocks at all. Both shapes are
    legitimate; this loader treats them as "all defaults" rather
    than forcing every caller to pre-guard ``raw or {}``.

    Returns a ``CruciblePolicies`` with each sub-policy resolved from
    its named block or the dataclass default.
    """
    raw = raw or {}
    return CruciblePolicies(
        breach=BreachPolicy(**_kwargs_for(BreachPolicy, raw.get("breach_policy"))),
        review=ReviewPolicy(**_kwargs_for(ReviewPolicy, raw.get("review_policy"))),
    )


__all__ = ["load_policies_from_yaml"]
