# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""DeepSWE dataset, workspace, policy, and verifier integration.

The :mod:`crucible.workflows.deepswe` workflow loads local DeepSWE task
records, synthesizes an Assay Spec for each task, prepares an isolated Docker
workspace, runs the implementation workflow, and records verifier rewards.

Subpackage layout:

- ``dataset.py`` — load + validate ``DeepSWEInstance`` records from
  a local deep-swe clone. The dataset is fetched out-of-band
  (``git clone https://github.com/datacurve-ai/deep-swe.git``) and
  supplied via ``--dataset <path-to-tasks/manifest.json>``; the
  adapter intentionally has no network-fetch path so the workflow
  stays testable + reproducible. Mirrors the
  ``products/swebench/dataset.py`` boundary.
- ``spec_synthesis.py`` — ``synthesize_starter_spec(inst) → Spec`` translating
  instruction prose and typed
  ``DeepSWEInstance`` resource fields into a crucible Spec.
- ``instance_container.py`` — per-task container lifecycle:
  pull → run with ``--network=none`` → bind-mount worktree at
  ``/app`` → teardown + image cleanup.
- ``runners/`` — verifier and development-verification adapters. The main
  verifier wraps
  ``bash /tests/test.sh`` inside the container; parses
  one v1.0/v1.1 reward artifact under ``/logs/verifier`` to expose
  ``reward: 0|1`` as the build/verify outcome.
"""

from crucible.products.deepswe.dataset import (
    DeepSWEInstance,
    DeepSWEInstanceError,
    filter_instances,
    load_dataset,
    load_dataset_with_tests_identity,
)
from crucible.products.deepswe.instance_container import (
    InstanceContainerSetupError,
    check_disk_headroom,
    docker_available,
    populate_workspace_from_image,
    pull_image,
    setup_instance_container,
    teardown_instance_container,
)
from crucible.products.deepswe.spec_synthesis import (
    DEEPSWE_BREACH_POLICY,
    DEEPSWE_REVIEW_POLICY,
    resolve_deepswe_breach_policy,
    synthesize_starter_spec,
)

__all__ = [
    "DEEPSWE_BREACH_POLICY",
    "DEEPSWE_REVIEW_POLICY",
    "DeepSWEInstance",
    "DeepSWEInstanceError",
    "InstanceContainerSetupError",
    "check_disk_headroom",
    "docker_available",
    "filter_instances",
    "load_dataset",
    "load_dataset_with_tests_identity",
    "populate_workspace_from_image",
    "pull_image",
    "resolve_deepswe_breach_policy",
    "setup_instance_container",
    "synthesize_starter_spec",
    "teardown_instance_container",
]
