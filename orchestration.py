# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Public programmatic orchestration contracts.

These surfaces are deliberately off by default.  Read-only fan-out is used by
the opt-in Architect pilot.  Isolated writer fan-out returns a validated patch
but never lands it.  Each lane retains bounded costs and exact child-settlement evidence.
"""

from __future__ import annotations

from .workflows._readonly_scheduler import (
    ReadOnlyChildOutcome,
    ReadOnlyChildRequest,
    ReadOnlyCleanupError,
    ReadOnlyFanoutResult,
    ReadOnlyIsolationError,
    ReadOnlySchedulerError,
    run_read_only_fanout,
)
from .workflows._run_policy import (
    ORCHESTRATION_POLICY_VERSION,
    WRITER_LANE_POLICY_VERSION,
    OrchestrationPolicy,
    WriterLanePolicy,
)
from .workflows._writer_scheduler import (
    IntegrationGateResult,
    WriterCleanupError,
    WriterFanoutResult,
    WriterIntegrationConflict,
    WriterIsolationError,
    WriterLaneOutcome,
    WriterLaneRequest,
    WriterSchedulerError,
    run_isolated_writer_fanout,
)


__all__ = [
    "IntegrationGateResult",
    "ORCHESTRATION_POLICY_VERSION",
    "OrchestrationPolicy",
    "ReadOnlyChildOutcome",
    "ReadOnlyChildRequest",
    "ReadOnlyCleanupError",
    "ReadOnlyFanoutResult",
    "ReadOnlyIsolationError",
    "ReadOnlySchedulerError",
    "WRITER_LANE_POLICY_VERSION",
    "WriterFanoutResult",
    "WriterCleanupError",
    "WriterIntegrationConflict",
    "WriterIsolationError",
    "WriterLaneOutcome",
    "WriterLanePolicy",
    "WriterLaneRequest",
    "WriterSchedulerError",
    "run_isolated_writer_fanout",
    "run_read_only_fanout",
]
