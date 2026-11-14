# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.common.utils``.

Covers ``elide_middle`` — the head + visible-gap + tail truncation
helper that replaces silent tail-only truncation with a soft elision
the agent can pivot from (per mini-swe-agent v2.0.0).
"""

from __future__ import annotations

import pytest

from crucible.common.utils import elide_middle


def test_short_text_returned_unchanged():
    text = "hello world"
    assert elide_middle(text, head_chars=100, tail_chars=100) == text


def test_text_at_threshold_returned_unchanged():
    # Exactly head + tail = no elision needed.
    text = "x" * 200
    assert elide_middle(text, head_chars=100, tail_chars=100) == text


def test_text_one_char_over_threshold_elides_one_char():
    text = "h" * 100 + "M" + "t" * 100
    result = elide_middle(text, head_chars=100, tail_chars=100)
    assert result.startswith("h" * 100)
    assert result.endswith("t" * 100)
    assert "1 characters elided" in result
    assert "M" not in result  # the middle char was the one elided


def test_long_text_has_head_tail_and_gap_count():
    text = "A" * 5000 + "M" * 90_000 + "Z" * 5000
    result = elide_middle(text, head_chars=5000, tail_chars=5000)
    assert result.startswith("A" * 5000)
    assert result.endswith("Z" * 5000)
    assert "90000 characters elided" in result
    # Middle should be elided — none of the M's land in the result.
    assert "M" not in result
    # Result is well under the original length.
    assert len(result) < len(text) // 4


def test_head_only_truncation_with_zero_tail():
    text = "X" * 1000
    result = elide_middle(text, head_chars=100, tail_chars=0)
    assert result.startswith("X" * 100)
    assert "900 characters elided" in result
    # No trailing content beyond the marker.
    assert result.endswith("]\n\n")


def test_tail_only_truncation_with_zero_head():
    text = "X" * 1000
    result = elide_middle(text, head_chars=0, tail_chars=100)
    # Result starts with the marker (after a leading newline).
    assert result.startswith("\n\n[... ")
    assert "900 characters elided" in result
    assert result.endswith("X" * 100)


def test_negative_chars_rejected():
    with pytest.raises(ValueError, match="head_chars and tail_chars"):
        elide_middle("hello", head_chars=-1, tail_chars=10)
    with pytest.raises(ValueError, match="head_chars and tail_chars"):
        elide_middle("hello", head_chars=10, tail_chars=-1)


def test_marker_text_visible_at_default_position():
    text = "A" * 10_000 + "B" * 10_000
    result = elide_middle(text, head_chars=5_000, tail_chars=5_000)
    # Default marker shape — agent can grep for "characters elided"
    # to find the gap and compute byte offsets for a follow-up
    # head -c / tail -c retrieval.
    assert "[... 10000 characters elided ...]" in result


def test_custom_marker_prefix_suffix():
    text = "x" * 10_000
    result = elide_middle(
        text,
        head_chars=100,
        tail_chars=100,
        marker_prefix="<<< ",
        marker_suffix=" bytes hidden >>>",
    )
    assert "<<< 9800 bytes hidden >>>" in result


def test_unicode_safe():
    # len() counts codepoints, not bytes; we slice on codepoints
    # consistently throughout. No surrogate-pair drama.
    text = "あ" * 1000  # 1000 codepoints, each 3 bytes UTF-8
    result = elide_middle(text, head_chars=100, tail_chars=100)
    assert result.startswith("あ" * 100)
    assert result.endswith("あ" * 100)
    assert "800 characters elided" in result
