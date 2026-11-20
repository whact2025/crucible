# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared workflow argparse helpers + cross-workflow constants.

Many ``crucible.workflows.<name>`` modules expose the same
four argparse flags (``--spec``, ``--backend-config``, ``--max-turns``,
``--budget-usd``) and the same native-tool allowlist.

This module is the single source of truth for those shared shapes.
Workflows call the ``add_*_arg`` helpers from inside their ``_parse_args``;
help text and types live here so shared changes have one owner.

Constants:

- ``IMPLEMENTER_BUILTIN_TOOLS`` — the explicit SDK-native tool allowlist for
  implementer calls. It is intentionally empty: greenfield workflows expose
  source/write/build operations only through path-confined MCP profiles, so an
  unrestricted native shell cannot bypass scope enforcement.

Helpers:

- ``add_spec_arg(p)`` — required ``--spec PATH``.
- ``add_backend_config_arg(p, *roles)`` — optional path to a YAML
  override; help text names the role(s) consulted so operators see
  which role keys to edit in their backends.yaml.
- ``add_max_turns_arg(p, default, scope)`` — int with default and a
  brief scope clause; the scope explains *what* a turn is in this
  workflow (per backend call vs per debate round vs per slice).
- ``add_budget_arg(p, kind)`` — float with ``kind="per_call"`` or
  ``"cumulative"`` — help text spells out the semantics so
  operators don't have to guess.
- ``add_backend_limit_args(p)`` — opt-in strict/diagnostic route admission
  plus one per-invocation monotonic deadline.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


IMPLEMENTER_BUILTIN_TOOLS: tuple[str, ...] = ()


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return parsed


def nonnegative_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def validate_nonnegative_int(value: int, *, name: str) -> int:
    """Validate a programmatic integer argument that permits zero."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_positive_int(value: int, *, name: str) -> int:
    """Validate a programmatic integer argument that requires progress."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_budget(value: str, *, allow_zero: bool) -> float:
    parsed = float(value)
    valid_floor = parsed >= 0.0 if allow_zero else parsed > 0.0
    if not math.isfinite(parsed) or not valid_floor:
        qualifier = "non-negative" if allow_zero else "greater than zero"
        raise argparse.ArgumentTypeError(
            f"must be a finite number {qualifier}"
        )
    return parsed


def nonnegative_budget_arg(value: str) -> float:
    """Parse a finite cumulative budget; zero means no paid calls."""

    return _finite_budget(value, allow_zero=True)


def positive_budget_arg(value: str) -> float:
    """Parse a finite per-invocation budget."""

    return _finite_budget(value, allow_zero=False)


def add_spec_arg(p: argparse.ArgumentParser) -> None:
    """Add the canonical ``--spec`` flag to ``p``.

    Every workflow takes a required Corpus-shaped spec directory; we
    standardize on the same wording so ``--help`` looks consistent
    across workflows.
    """
    p.add_argument(
        "--spec", required=True, type=Path,
        help="Path to a Corpus-shaped spec directory (the spec's "
             "manifests live under <spec>/versions/).",
    )


def add_backend_config_arg(
    p: argparse.ArgumentParser, *roles: str,
) -> None:
    """Add ``--backend-config`` for per-role routing overrides.

    ``roles`` names the role keys this workflow's ``resolve_for_role``
    calls will consult — the help text surfaces them so operators
    know which keys to set in ``.crucible/backends.yaml``.
    """
    if not roles:
        roles_phrase = "see workflow source for role names"
    elif len(roles) == 1:
        roles_phrase = f"role consulted: {roles[0]}"
    else:
        roles_phrase = f"roles consulted: {', '.join(roles)}"
    p.add_argument(
        "--backend-config", type=Path, default=None,
        help=f"Path to a backend routing YAML overriding discovery "
             f"(see crucible.backends.config for the schema and the "
             f"env / spec_dir / workspace fall-through order); "
             f"{roles_phrase}.",
    )


def add_max_turns_arg(
    p: argparse.ArgumentParser, *, default: int, scope: str,
) -> None:
    """Add ``--max-turns`` with explicit default + scope-clarifying help.

    ``scope`` should be a short phrase describing what a single turn
    corresponds to in this workflow (e.g. "per backend call",
    "per debate round", "per slice's implementer phase"). Without it
    operators can't tell whether 50 means "the whole run" or "each
    of three sub-roles."
    """
    p.add_argument(
        "--max-turns", type=positive_int_arg, default=default,
        help=f"Max conversational turns {scope} "
             f"(default: {default}).",
    )


def add_budget_arg(
    p: argparse.ArgumentParser, *, kind: str = "per_call",
) -> None:
    """Add ``--budget-usd`` with explicit per-call vs cumulative semantics.

    ``kind="per_call"`` — ceiling on each individual backend invocation;
    the backend raises ``BudgetExhausted`` past it.

    ``kind="cumulative"`` — ceiling on the sum of all backend
    invocations made by the workflow; checked after each call, halts
    further calls when exceeded.

    Workflows that have BOTH per-instance and total budgets
    (currently only swebench) carry their own --budget-usd-per-instance
    flag on top of this one — that's a swebench-specific extension and
    lives in its own ``_parse_args``.
    """
    if kind == "cumulative":
        help_text = (
            "Cumulative cost ceiling (USD) across every backend call "
            "made by this workflow. Checked after each call; the "
            "workflow halts further calls once exceeded and exits 1."
        )
    elif kind == "per_call":
        help_text = (
            "Cost ceiling (USD) per backend call. The backend raises "
            "BudgetExhausted on overspend; downstream handling is "
            "workflow-specific."
        )
    else:
        raise ValueError(
            f"add_budget_arg: kind must be 'per_call' or "
            f"'cumulative'; got {kind!r}"
        )
    budget_type = (
        nonnegative_budget_arg if kind == "cumulative" else positive_budget_arg
    )
    p.add_argument("--budget-usd", type=budget_type, default=None, help=help_text)


def add_backend_limit_args(p: argparse.ArgumentParser) -> None:
    """Add the explicit UIR-03 invocation-limit policy controls.

    The pair is opt-in so pre-UIR-03 invocations retain their prior execution
    shape. ``strict`` admits only routes with hard turn and (when the workflow
    has a USD budget) hard USD enforcement. ``diagnostic`` records truthful
    downgrades, forces one transport attempt, and binds the adapter call to the
    supplied monotonic deadline.
    """

    p.add_argument(
        "--backend-limit-mode",
        choices=("strict", "diagnostic"),
        default=None,
        help=(
            "UIR-03 provider-limit admission mode. Must be paired with "
            "--backend-limit-deadline-seconds. Existing behavior is retained "
            "when omitted."
        ),
    )
    p.add_argument(
        "--backend-limit-deadline-seconds",
        type=positive_budget_arg,
        default=None,
        help=(
            "Finite positive monotonic wall deadline for each provider "
            "invocation. Must be paired with --backend-limit-mode."
        ),
    )
    p.add_argument(
        "--diagnostic-cost-policy",
        choices=("kimi-k3.1-pro-cache-read-upper-bound-v1",),
        default=None,
        help=(
            "Explicit diagnostic-only budget accounting for validated Kimi "
            "K3.1 Pro cache-read upper bounds. Requires diagnostic backend limits "
            "and a finite USD budget; exact cost remains incomplete and "
            "results are not promotable or rankable."
        ),
    )


def validate_backend_limit_args(
    p: argparse.ArgumentParser,
    parsed: argparse.Namespace,
) -> None:
    """Fail argparse closed when exactly one UIR-03 control is supplied."""

    if (parsed.backend_limit_mode is None) != (
        parsed.backend_limit_deadline_seconds is None
    ):
        p.error(
            "--backend-limit-mode and --backend-limit-deadline-seconds must "
            "be supplied together"
        )
    from ._diagnostic_cost import validate_diagnostic_cost_policy

    try:
        validate_diagnostic_cost_policy(
            getattr(parsed, "diagnostic_cost_policy", None),
            limit_mode=parsed.backend_limit_mode,
            budget_usd=getattr(parsed, "budget_usd", None),
        )
    except ValueError as exc:
        p.error(str(exc))


__all__ = [
    "IMPLEMENTER_BUILTIN_TOOLS",
    "add_backend_limit_args",
    "add_spec_arg",
    "add_backend_config_arg",
    "add_max_turns_arg",
    "add_budget_arg",
    "nonnegative_int_arg",
    "nonnegative_budget_arg",
    "positive_budget_arg",
    "positive_int_arg",
    "validate_backend_limit_args",
    "validate_nonnegative_int",
]
