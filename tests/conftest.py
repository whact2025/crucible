# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for Crucible tests.

This module contains only fixtures shared across the current test suite:

- ``_contain_process_trace_state`` (autouse): restores an exact incoming
  interpreter tracer and fails a test that leaks trace-state drift;

"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _contain_process_trace_state():
    """Restore and reject a test-owned ``sys.settrace`` leak."""

    incoming_trace = sys.gettrace()
    try:
        yield
    finally:
        outgoing_trace = sys.gettrace()
        if outgoing_trace is not incoming_trace:
            if incoming_trace is None:
                sys.settrace(None)
            else:
                sys.settrace(incoming_trace)
            pytest.fail(
                "test leaked sys.settrace state; restore the incoming tracer",
                pytrace=False,
            )
