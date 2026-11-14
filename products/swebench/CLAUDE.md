<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# SWE-bench product guide

SWE-bench is a Crucible adapter for SWE-bench Verified. It uses the shared
Architect/Tier solution workflow, exact candidate capture, independent pristine
visible verification, and optional official hidden evaluation. See the
[workflow reference](../../docs/workflows/swebench.md) for flags and outcomes.

## Ownership

| Module | Responsibility |
|---|---|
| `dataset.py` | Strict JSON/JSONL loading and explicit task filters. |
| `repo_workspace.py` | Standalone authoring/verifier checkouts and acquisition caching. |
| `instance_container.py` | Immutable image identity, confined authoring, ownership, and cleanup. |
| `spec_synthesis.py`, `prompts.py` | Task Spec, authoring constraints, plan/gate policy, and prompts. |
| `paths.py`, `plan_normalization.py` | Source/test classification and audited plan normalization. |
| `runners/swebench_pytest.py` | Public development tests. |
| `pristine_verifier.py` | Independent replay and visible checks for exact candidate bytes. |
| `predictions.py` | Durable deduplicating leaderboard JSONL publication. |
| `eval.py`, `evaluator_compat.py` | Official hidden harness and reviewed-version cleanup guard. |
| `harness_reclassify.py` | Conservative interpretation of retained harness output. |
| `backend_convergence*.py`, `backend_convergence_matrix*.yaml` | Byte-bound cohort planning and comparison analysis. |
| `backend_profile_catalog.yaml`, `backends*.yaml` | Exact selectable route identities and diagnostic admission. |

## Task validity

Each task owns an isolated Corpus, checkout, container, budget identity, and
child operation. Models see public inputs only. Hidden tests and verifier
material stay out of prompts, source-tool roots, authoring images, and traces.
Submission uses captured source-only bytes, not an arbitrary final worktree.
The pristine verifier and official harness consume those exact bytes.

Each task defaults to three independent concurrent complete plans, sharing at
most 40% of its unchanged whole-task budget for all planning roles. Task-worker
limits do not serialize those nested draft requests. Explicit count-one
diagnostics keep their single-track identity and existing cumulative task cap.

An Architect return code and implementation classification describe the
cascade. Harness outcome determines the benchmark result. An unresolved task,
skipped evaluation, incomplete accounting, or failed cleanup stays explicit.

## Routes and comparisons

Use the exact suite profile from `backend_profile_catalog.yaml`. Profile
metadata grants neither launch authority nor promotion. Diagnostic writer
opt-ins, served-model pins, runtime fingerprints, limits, credential source,
and completeness policy remain part of the treatment identity.

DeepSeek templates require provider-free `backend_config_bind_deepseek`
materialization. Independently locked Codex, DeepSeek, and Kimi workers retain
their own runtime identity. Do not change a matrix's bytes, route, cohort, or
analysis semantics and reinterpret existing results under the changed identity.
Use separate plan/output roots for different experiments and compare only
properly paired evidence.

## Resume and maintenance

Reuse checkpoints only when all bound dataset, route-source, recipe, policy,
image, candidate, pricing, and budget identities match. Explicit rerun filters
select which tasks to attempt again. Never weaken scope, pristine replay,
hidden-test isolation, or settlement merely to reuse an old checkpoint.
Planning count, share, execution fingerprint, and source recipe also bind the
accepted DAG. A completed checkpoint mismatch requires explicit rerun selection;
an authorized new attempt freshly plans a retained DAG with a mismatched stamp.
A matching stamp permits normal model-free plan reuse. Typed benchmark policy
rejects exact digest reacceptance. Historical convergence evidence retains its
original identity.

Changes to the upstream harness version require auditing the exact generated
script shape guarded by `evaluator_compat.py` and its fixtures. Run focused
suite/verifier tests before the complete non-live suite. Live containers and
hidden evaluation require explicit task authorization.
