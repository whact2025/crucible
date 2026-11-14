# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.backends._classify`` shared constants.

Both ClaudeBackend and CodexBackend route through these constants,
and their per-backend test suites already parametrize over the
contents — a regression that, say, accidentally renamed
``TERMINAL_EXCEPTION_NAMES`` to ``ABORT_NAMES`` would fire in two
places. This file pins the API surface so an accidental rename /
delete / structural change here fires immediately, regardless of
backend coverage.
"""

from __future__ import annotations

from assay import ContentRejected, ExternalCancellation, TransportError
from crucible.backends._classify import (
    RETRIABLE_HTTP_STATUSES,
    TERMINAL_EXCEPTION_NAMES,
    ClassifiedError,
)


def test_terminal_exception_names_is_frozenset_of_str():
    assert isinstance(TERMINAL_EXCEPTION_NAMES, frozenset)
    assert all(isinstance(n, str) for n in TERMINAL_EXCEPTION_NAMES)


def test_terminal_exception_names_exact_contents():
    """Pin the abort-list so a rename / addition / deletion fires a
    visible test failure, not a silent classification change.

    ``ClaudeSessionIdentityError`` is terminal because retrying cannot repair
    drift between the configured organization identity and the authenticated
    Claude session principal.
    """
    assert TERMINAL_EXCEPTION_NAMES == frozenset({
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "ContextWindowExceededError",
        "UnsupportedParamsError",
        "ClaudeSessionIdentityError",
    })


def test_retriable_http_statuses_is_frozenset_of_int():
    assert isinstance(RETRIABLE_HTTP_STATUSES, frozenset)
    assert all(isinstance(s, int) for s in RETRIABLE_HTTP_STATUSES)


def test_retriable_http_statuses_exact_contents():
    """Pin the retriable-HTTP-status set so a policy change is a
    deliberate test-edit rather than a silent regression. The eight
    codes are: 408 timeout, 409 idempotency conflict, 429 rate limit,
    the 5xx upstream transients (500/502/503/504), and 529 (Anthropic
    ``overloaded_error``)."""
    assert RETRIABLE_HTTP_STATUSES == frozenset({
        408, 409, 429, 500, 502, 503, 504, 529,
    })


def test_classified_error_is_union_of_three_backend_error_types():
    """The ``ClassifiedError`` alias documents the non-None return
    contract of ``_classify_exception``. A future refactor that
    drops one of the three subtypes (or adds a fourth) must update
    this assertion alongside the alias."""
    # typing.get_args works on Union types in Python 3.11+.
    import typing
    args = typing.get_args(ClassifiedError)
    assert set(args) == {TransportError, ExternalCancellation, ContentRejected}
