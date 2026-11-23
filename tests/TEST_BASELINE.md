<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Crucible test guide

The ordinary suite verifies the reusable solution-building workflow and the
SWE-bench/DeepSWE adapters without paid model calls or a Docker daemon.

```bash
uv run python -m pytest -q
uv run ruff check .
git diff --check
```

The configured marker filter excludes `llm_live` and `live_docker`. Run a
focused set first, then the complete non-live suite for cross-cutting changes.
Linux-only process containment, filesystem, and settlement tests require their
actual host facilities; report platform skips explicitly.

## Hosted CI

CI assigns every discovered test module to one of eight groups and runs each
file intact. The groups share one inventory in `scripts/test_shard.py`; new
test files are included automatically. The final quality gate requires every
group to succeed and supply its coverage, test report, and inventory. It
combines coverage before enforcing the 55% floor.

The ephemeral Ubuntu test runners enable unprivileged user namespaces and
verify root mapping before the quota helper's real descriptor-execution
tests. A missing runner prerequisite fails setup instead of bypassing those
tests.

## Contract checks

Each regression preserves an observable behavior or failure boundary. Keep
scalar fields, wire types, authority identities, and rollback or publication
phases independently covered. A shared successful cascade can own multiple
assertions. Gate doubles settle their exact writer epoch so a dispatch
regression fails its behavior assertion with cleanup still intact.

A shared guard fails any test that ends with a workspace lock chain or a
registered workspace authority that appeared during it, and names what the
test left. Settle the lock owner a test caught with `retry_cleanup`, exit
every lock it entered, and join every thread that takes a lock before the
test returns.

| Area | Representative tests |
|---|---|
| Current product and docs | `test_axioms.py`, `test_documentation.py`, `test_workflows_dispatcher.py` |
| Plans and accepted authority | `test_workflows_plan_validation.py`, `test_plan_provenance.py`, `test_plan_source_integrity.py` |
| Build and adversarial gates | `test_workflows_build_verify.py`, `test_workflows_breach.py`, `test_workflows_tier_breach_policy.py` |
| Suite behavior | `test_workflows_swebench.py`, `test_workflows_deepswe.py`, suite dataset/container/verifier tests |
| Parallel work | `test_workflows_readonly_scheduler.py`, `test_workflows_writer_scheduler.py`, `test_workflow_outcomes.py` |
| Mutation and recovery | Mutation-journal, writer-epoch, settlement, rollback, reconcile, and journal-recovery tests |
| Routes and reviews | Backend conformance, config/features/limits, pricing, rubric, and calibration tests |
| Distribution | `test_root_wheel.py`, `test_editable_install_identity.py` |

The root pytest bootstrap verifies that `import crucible` resolves to the
checkout being tested. Use worktrees shaped `<unique-parent>/crucible` with
Assay as its sibling, and install that checkout's development environment.
Testing another editable checkout is invalid evidence.

## Live tests

Docker probes and paid model calibration are separate, explicitly selected
lanes. Run only the authorized tests and exact route. Containers and model
sessions can consume substantial resources; their results need complete
cleanup and identity evidence. Hidden benchmark evaluation is a product run,
not an incidental unit-test action.

Report tests executed, failures, skipped platform checks, and any unavailable
live gates. Never claim that offline tests establish solved benchmark tasks or
promote a diagnostic backend route.
