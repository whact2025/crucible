# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for Crucible tests.

This module contains only fixtures shared across the current test suite:

- ``_contain_process_trace_state`` (autouse): restores an exact incoming
  interpreter tracer and fails a test that leaks trace-state drift;
- ``_contain_workspace_authority`` (autouse): fails a test during which a
  workspace lock chain or a registered workspace authority appeared and was
  still held when the test ended, and releases a dropped lock owner before
  the next test runs;

"""

from __future__ import annotations

import gc
import os
import sys

import pytest


_JOURNAL_MODULE = "crucible.workflows._mutation_journal"
_release_dropped_owners_first = False


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


def _held_workspace_authority() -> tuple[
    frozenset[int], frozenset[int], frozenset[str]
]:
    """Both held descriptor sets and this thread's registered authorities.

    The journal module is read through ``sys.modules``: a test that never
    loads it holds nothing, and a test that rebinds the module's state is
    read through the objects that are current.
    """

    journals = sys.modules.get(_JOURNAL_MODULE)
    if journals is None:
        return frozenset(), frozenset(), frozenset()
    return (
        frozenset(journals._HELD_AUTHORITY_DESCRIPTORS),
        frozenset(journals._HELD_LOCK_DESCRIPTORS),
        frozenset(
            os.fspath(key)
            for key in tuple(getattr(journals._LOCK_STATE, "authorities", {}))
        ),
    )


def _release_dropped_owners() -> None:
    """Finalize the lock owners an earlier test dropped, at a test boundary.

    A dropped owner is released when a garbage collection finalizes its lock
    generators.  During the test that dropped it, and during that test's
    fixture teardown, a collection does not reach it: the log capture of the
    test still references the logged error that owns it.  pytest also keeps
    the exception of a failed test for post-mortem use until the next test
    is called.  Here the capture has ended and the post-mortem names are
    removed, so one collection releases the chain before the next test can
    be interrupted by it.
    """

    for name in ("last_exc", "last_value", "last_traceback", "last_type"):
        if hasattr(sys, name):
            delattr(sys, name)
    gc.collect()


@pytest.fixture(autouse=True)
def _contain_workspace_authority(request):
    """Fail a test that ends with workspace authority it did not start with.

    The test is charged with every held authority descriptor, held lock
    descriptor and registered authority that is present when it ends and was
    absent when its setup began: an owner it caught and dropped, an owner it
    still references, or a lock that one of its threads still holds.  What a
    wider fixture or an earlier test already held is not charged again.  A
    test settles the owner it caught (``retry_cleanup``), exits the lock it
    entered, and joins every thread that takes a lock before it returns.
    """

    global _release_dropped_owners_first
    if _release_dropped_owners_first:
        _release_dropped_owners_first = False
        _release_dropped_owners()
    incoming = _held_workspace_authority()
    yield
    left = tuple(
        sorted(after - before)
        for after, before in zip(_held_workspace_authority(), incoming)
    )
    if any(left):
        _release_dropped_owners_first = True
        pytest.fail(
            f"{request.node.nodeid} ended with workspace authority that "
            f"appeared during it: {len(left[0])} authority descriptor(s) "
            f"{left[0]}, {len(left[1])} lock descriptor(s) {left[1]}, "
            f"authority registered for {left[2]}; settle the lock owner the "
            "test caught (retry_cleanup), exit the lock it entered, and "
            "join its threads before the test returns",
            pytrace=False,
        )
