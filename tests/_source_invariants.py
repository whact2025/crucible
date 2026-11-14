# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Source-grep invariant helper.

A small number of tests assert that specific identifiers appear (or
DON'T appear) in a module's source. The pattern catches a specific
class of regressions — a refactor that reverts a wire-through and
re-introduces a known-bad code shape, without changing behavior in
a way the offline mocked tests would notice. (Concrete example:
the ``spec_clarity`` false-negative on SWE-bench specs was driven
by a hardcoded ``_REVIEW_RUBRICS`` constant; the wire-through
replaced it with policy-driven iteration. A future refactor that
restores the constant would re-create the bug, and source-grep is
the cheapest signal.)

Tradeoffs (per reviewer feedback):
- Brittle to legitimate renames — ``breach_policy`` → ``bp`` would
  trip the test even with preserved behavior. The fix is to update
  the invariant when intentional, not to weaken it.
- Catches source-structure regressions, not behavior — a bug that
  preserves the strings but breaks logic slips through. These tests
  are paired with behavior-shape tests (defaults, validation, etc.)
  to cover that gap.
- Substring matches can collide with unrelated comments. The
  ``re.search`` form supports tighter patterns where collision is a
  concern.

Use sparingly; for high-coverage behavior tests prefer integration
or unit tests against the real consume sites.
"""

from __future__ import annotations

import inspect
import re
from types import ModuleType


def assert_source_invariant(
    module: ModuleType,
    *,
    must_contain: tuple[str, ...] = (),
    must_not_contain: tuple[str, ...] = (),
    must_contain_patterns: tuple[str, ...] = (),
    must_not_contain_patterns: tuple[str, ...] = (),
    scope: str | None = None,
) -> None:
    """Assert source-grep invariants on ``module``.

    Args:
      module: imported Python module whose source we'll inspect.
      must_contain: literal substrings that MUST appear in the source.
      must_not_contain: literal substrings that MUST NOT appear.
      must_contain_patterns: regex patterns that MUST match (use when
        a literal substring would be too brittle — e.g., either
        single- or double-quoted form).
      must_not_contain_patterns: regex patterns that MUST NOT match.
      scope: human-readable label for error messages (e.g.
        ``"Phase 4 review loop"``). Helps the failure point at the
        contract rather than just the offending file.

    Raises ``AssertionError`` with a contract-pointing message on
    any violation.
    """
    src = inspect.getsource(module)
    label = f"{module.__name__}" + (f" — {scope}" if scope else "")

    for needle in must_contain:
        assert needle in src, (
            f"{label}: source must contain {needle!r} — "
            f"either the wire-through was reverted, or the identifier "
            f"was renamed. Update the invariant if the rename was "
            f"intentional."
        )
    for needle in must_not_contain:
        assert needle not in src, (
            f"{label}: source must NOT contain {needle!r} — its "
            f"presence indicates a regression of a removed code shape."
        )
    for pattern in must_contain_patterns:
        assert re.search(pattern, src), (
            f"{label}: source must match regex {pattern!r}"
        )
    for pattern in must_not_contain_patterns:
        assert not re.search(pattern, src), (
            f"{label}: source must NOT match regex {pattern!r}"
        )
