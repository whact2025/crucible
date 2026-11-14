# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""DeepSWE-facing exports for the shared contained-execution engine.

Benchmark-neutral mechanics live under :mod:`crucible.common.runners`; this
module exposes the DeepSWE profile and its established product import path.
"""

from crucible.common.runners.container_exec import (
    CONTAINMENT_UNUSABLE_RC,
    ContainerExecProfile,
    DEEPSWE_EXEC_PROFILE,
    ExecContainmentError,
    run_contained_docker_exec,
)


__all__ = [
    "CONTAINMENT_UNUSABLE_RC",
    "ContainerExecProfile",
    "DEEPSWE_EXEC_PROFILE",
    "ExecContainmentError",
    "run_contained_docker_exec",
]
