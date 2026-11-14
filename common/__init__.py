# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared workflow, runner, policy, cost, and execution support."""

from datetime import datetime, timezone

_TS_FMT = "%Y%m%d_%H%M%S"


def generate_timestamp() -> str:
    """Return a UTC timestamp string for use in output directory and log file names."""
    return datetime.now(timezone.utc).strftime(_TS_FMT)


__all__ = [
    "generate_timestamp",
]
