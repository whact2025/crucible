# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Deterministic plan-validation rules for ``assay.validate``.

Each ``<rule_id>.py`` module here exposes a top-level callable named
``<rule_id>`` that takes a ``Spec`` and returns ``tuple[Finding, ...]``.
``assay.validate``'s default resolver imports them lazily via
``crucible.rules.<rule_id>``.

The bundled rules validate structural and workspace-grounded properties of
the ``assay.Spec`` dataclass:

  cycle                     — TierDAG depends_on graph is acyclic
  runner_path_prefix        — project_context.runners[*] are sane paths
  implementer_verify        — directive.verify_mode is a known value
  creates_declaration       — intent's "create X" matches creates_paths
  blast_radius              — files_in_scope ⊆ no_edits_outside
  parallel_tier_conflicts   — sibling slices don't share scope files
  creates_paths             — no two slices claim the same creation
  deletion_violates_goal    — intent's "delete X" respects no_edits_outside
  ungrounded_citation       — cited paths appear in scope ∪ creates
  phantom_reference         — cited symbols' files appear in scope
  cited_path_exists         — cited paths exist in the supplied RuleContext
  scope_covers_intent       — requested workspace paths are covered by scope
  read_only_references      — source identity, excerpts and read/write separation

``cited_path_exists``, ``scope_covers_intent`` and ``read_only_references``
require the caller to supply an Assay ``RuleContext``.
"""

from .blast_radius import blast_radius
from .cited_path_exists import cited_path_exists
from .creates_declaration import creates_declaration
from .creates_paths import creates_paths
from .cycle import cycle
from .deletion_violates_goal import deletion_violates_goal
from .implementer_verify import implementer_verify
from .parallel_tier_conflicts import parallel_tier_conflicts
from .phantom_reference import phantom_reference
from .runner_path_prefix import runner_path_prefix
from .read_only_references import read_only_references
from .scope_covers_intent import scope_covers_intent
from .ungrounded_citation import ungrounded_citation

# Rules split into two arities for ``assay.validate``:
# - Pure (rule(spec)) — run unconditionally, deterministic.
# - Context-aware (rule(spec, ctx)) — run only when a RuleContext is
#   threaded through; verdict reflects the local working tree.
_PURE_RULE_NAMES = (
    "blast_radius",
    "creates_declaration",
    "creates_paths",
    "cycle",
    "deletion_violates_goal",
    "implementer_verify",
    "parallel_tier_conflicts",
    "phantom_reference",
    "runner_path_prefix",
    "ungrounded_citation",
)

_CONTEXT_AWARE_RULE_NAMES = (
    "cited_path_exists",
    "scope_covers_intent",
    "read_only_references",
)

_RULE_NAMES = _PURE_RULE_NAMES + _CONTEXT_AWARE_RULE_NAMES


def available() -> tuple[str, ...]:
    return _RULE_NAMES


def context_aware() -> tuple[str, ...]:
    """Names of rules that take ``(spec, ctx)`` and need a RuleContext."""
    return _CONTEXT_AWARE_RULE_NAMES


__all__ = [
    "available",
    "context_aware",
    "blast_radius",
    "cited_path_exists",
    "creates_declaration",
    "creates_paths",
    "cycle",
    "deletion_violates_goal",
    "implementer_verify",
    "parallel_tier_conflicts",
    "phantom_reference",
    "runner_path_prefix",
    "read_only_references",
    "scope_covers_intent",
    "ungrounded_citation",
]
