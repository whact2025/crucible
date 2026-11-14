# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""SWE-bench dataset, harness, prediction, and workflow integration.

Workflow entry point: ``crucible/workflows/swebench.py``.

Subpackage layout:

- ``dataset.py``    — load + validate instance records from a local
  JSON / JSONL file. The dataset is fetched out-of-band (e.g. via
  ``datasets-cli`` or ``curl`` from HuggingFace) and supplied via
  ``--dataset <path>``; the adapter intentionally has no
  network-fetch path so the workflow stays testable + reproducible.
- ``predictions.py`` — emit predictions JSONL in the SWE-bench
  leaderboard contract:
  ``{"instance_id": "...", "model_patch": "...", "model_name_or_path": "..."}``.
- ``instance_container.py`` — isolated per-instance Docker lifecycle.
- ``evaluator_compat.py`` — guarded integration with the pinned upstream
  SWE-bench evaluator.
- ``runners/swebench_pytest.py`` — test runner and result classification for
  generated evaluator scripts.

Each instance has its own Corpus under ``<spec>/instances/<instance_id>/corpus``.
Architect and Tier write their normal lineage there; the workflow adds a
reserved terminal marker for the surfaced prediction and a per-instance
Measurement for the evaluator result.
"""
