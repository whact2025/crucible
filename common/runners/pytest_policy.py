# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared pytest policy for iterative project-suite verification.

Development verification needs more than the first failure so one repair
turn can address a coherent batch, but it must not let a pathological suite
produce an unbounded failure report.  This cap is deliberately separate from
the exact one-node probes used for breach observations.
"""

from __future__ import annotations


PYTEST_DEV_VERIFY_MAX_FAILURES = 16
PYTEST_DEV_VERIFY_MAXFAIL_ARG = f"--maxfail={PYTEST_DEV_VERIFY_MAX_FAILURES}"


__all__ = [
    "PYTEST_DEV_VERIFY_MAX_FAILURES",
    "PYTEST_DEV_VERIFY_MAXFAIL_ARG",
]
