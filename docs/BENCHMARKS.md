<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Running the benchmark suites

SWE-bench Verified and DeepSWE use the same solution-building cascade, with
separate dataset, submission, and verifier contracts. Start with one explicitly
selected task and inspect its evidence before increasing the cohort.

## Prepare the execution host

| Requirement | What must be ready |
|---|---|
| Linux development environment | A non-root account with nonzero uid/gid, Python 3.12+, uv, Git, the current Crucible checkout, and its sibling Assay dependency. Run `uv sync` from Crucible. |
| Container runtime | A reachable Linux Docker daemon, registry access to the suite images, and permission to create, inspect, execute, and remove managed containers. |
| Runtime tools | The selected backend's authenticated CLI/session or environment credentials; independently locked workers when the selected route requires them. Contained test execution also needs `cc` and static-linking support to build its supervisor. |
| Dataset | A local SWE-bench Verified JSON/JSONL file, or the complete DeepSWE tree with `tasks/manifest.json` and its task files. The workflow does not fetch the dataset. |
| Capacity | CPU/memory appropriate to the selected images, space for images and checkouts, and a quota-capable filesystem for writable task data. The 10 GiB free-disk guard is a minimum admission check, not a capacity estimate. |
| Kernel storage quota | An operator-installed helper selected by `CRUCIBLE_QUOTA_HELPER`, with durable reservation/release/retirement support for every monitored path. Both suites require it for implementation; a polling-only quota is insufficient. |

The bundled [quota helper](../scripts/crucible_quota_helper.py) uses a dedicated
ext4 mount with the `project` feature and `prjquota` option. Its root-owned
configuration defines `allowed_root`, a ledger outside that root, and the
project/inode limits. Lease paths must be strict descendants on the same mount.
Installing or configuring that privileged service is an operator setup step;
setting the environment variable alone does not establish quota support.

Place the DeepSWE Corpus under the configured quota root: its task workspaces,
logs, and private verification state live below run directories. For SWE-bench,
put `--repo-cache-dir` there as well; isolated authoring checkouts are created
beneath its `_worktrees/`. Without a cache directory, SWE-bench uses the process
temporary directory, which must satisfy the same quota contract. Do not put
unrelated jobs inside a leased task directory.

Images are pulled and resolved to immutable repository digests before managed
containers start. An unavailable or unverifiable digest is a setup failure.
Authoring containers have no network, a read-only image root, explicit writable
mounts, non-root task execution, and resource limits. Image dependencies must
already support public development checks without network installation. Hidden
test material stays outside authoring containers and model tools.

Use the exact suite [SWE-bench profile catalog](../products/swebench/backend_profile_catalog.yaml)
or [DeepSWE profile catalog](../products/deepswe/backend_profile_catalog.yaml).
Worker installation and diagnostic status are part of the route, not automatic
fallback choices. A diagnostic profile remains non-promotable even if it returns
a patch. DeepSeek construction templates require
[provider-free binding](workflows/backend_config_bind_deepseek.md).

## Select and inspect one task

Run from the Crucible checkout. Substitute the local dataset paths and quota
root in these examples. `--list` invokes neither a model nor a hidden verifier.

```bash
uv run crucible swebench --spec /data/crucible-benchmarks/swebench-one \
  --dataset /datasets/verified.json --limit 1 --list

uv run crucible deepswe --spec /data/crucible-benchmarks/deepswe-one \
  --dataset /datasets/deep-swe/tasks/manifest.json --limit 1 --list
```

Copy the selected identity into the relevant variable, then dry-run that exact
selection:

```bash
SWEBENCH_INSTANCE_ID='<listed-instance-id>'
DEEPSWE_TASK_ID='<listed-task-id>'

uv run crucible swebench --spec /data/crucible-benchmarks/swebench-one \
  --dataset /datasets/verified.json --instance-id "$SWEBENCH_INSTANCE_ID" \
  --dry-run

uv run crucible deepswe --spec /data/crucible-benchmarks/deepswe-one \
  --dataset /datasets/deep-swe/tasks/manifest.json --task-id "$DEEPSWE_TASK_ID" \
  --dry-run
```

SWE-bench dry-run writes empty-patch predictions. DeepSWE dry-run prints its
task plan. Neither proves Docker, image, quota, runtime, model, or verifier
readiness, and neither is benchmark success.

## Run the authorized task with an explicit budget

These implementation commands start model-backed work and the suite evaluator.
Run only the suite and route authorized for the task. The example USD ceilings
are explicit limits, not predicted prices or promises of completion.

```bash
uv run crucible swebench --spec /data/crucible-benchmarks/swebench-one \
  --dataset /datasets/verified.json --instance-id "$SWEBENCH_INSTANCE_ID" \
  --repo-cache-dir /data/crucible-benchmarks/swebench-cache \
  --backend-config products/swebench/backends_codex_astra.yaml \
  --budget-usd 20 --budget-usd-per-instance 20 --implement

uv run crucible deepswe --spec /data/crucible-benchmarks/deepswe-one \
  --dataset /datasets/deep-swe/tasks/manifest.json --task-id "$DEEPSWE_TASK_ID" \
  --backend-config products/deepswe/backends_codex_astra.yaml \
  --budget-usd 20 --budget-usd-per-instance 20 --implement
```

Aggregate and per-task budgets are separate. Enforcement strength depends on
the exact adapter; an indivisible provider response can overshoot a
post-response spending guard. Do not disable pristine verification, scope,
settlement, or quota requirements to get past a setup failure. DeepSWE's disk
override bypasses only free-disk admission, never its quota requirement.

Both suites default to three independent complete plans per task. Drafts run
concurrently, then each proposal receives deterministic checks and adversarial
review. Comparison selects a plan or proposes a synthesis that must pass fresh
review and validation. All branches, including failed proposals, consume the
same task allowance.

Competition receives at most 40% of the task budget for drafts, reviews,
comparison, and synthesis gates combined. For a $20 task, that is at most $8
across planning roles, further bounded by remaining funds and the development
ledger. The total task allowance stays $20. DeepSWE also retains its 50%
implementation/verifier reserve and 10% contingency. Adapter-specific spending
and settlement limits still apply; the split is not a guarantee that an
indivisible provider response cannot overshoot.

Use `--planner-count 1` only for an explicitly selected single-track diagnostic.
DeepSWE diagnostic cost pilots require it; DeepSWE still retains its task
budget reserves. SWE-bench's single-track override uses its existing cumulative
task cap without the competition's 40% partition. Listing and dry-run stay
provider-free. Competing plans require a positive finite effective task cap;
the suites do not reduce the planner count silently when funds are insufficient.

Allow for nested requests when sizing provider concurrency. One active task
can issue three draft requests, so two SWE-bench instance workers can issue six.
DeepSWE's serial task loop still permits three drafts inside its active task.
Convergence limits on active cells do not limit those nested requests. Planner
source-call and result-byte ceilings are shared across the task's planners.

## Inspect evidence and resume deliberately

Read the top-level `summary.json`, `result.json`, and `outcome.json`, then the
per-task summaries and saga-bound child runs. Check captured candidate identity,
pristine replay, actual cost/completeness, verifier result, and cleanup status.

| Suite | Result to inspect |
|---|---|
| SWE-bench | Published `predictions.jsonl`, candidate eligibility, pristine visible verification, and each instance's official `eval_status`. A zero aggregate exit can include unresolved predictions. |
| DeepSWE | Each task's status, exact candidate/verifier binding, binary reward, verification completeness, and cleanup. Aggregate success requires every selected task to resolve. `--lifecycle-out` is an optional durable task report, not a predictions file. |

Repeat the same command against the same Corpus to resume. Preserve dataset,
route-source bytes, policies, recipe, image identity, and budget inputs;
checkpoint validation rejects changed authority. Planner count, planning budget
share, and planning source recipe are part of that identity. A completed
checkpoint conflict requires an explicit rerun selection or a separate
experiment Corpus; it does not automatically launch new hidden evaluation.
Once execution is authorized, a retained DAG with a mismatched benchmark
planning stamp is freshly planned; a matching stamp permits normal plan reuse
without planning calls. Benchmark policy rejects exact digest reacceptance.
DeepSWE's `--replan` explicitly requests fresh planning. Do not edit completion
markers or delete retained journals to force reuse.

Use `--rerun` only when deliberately re-executing selected tasks.
`--rerun-status` accepts suite-specific terminal classifications. Infrastructure
failures and hidden-oracle results have different reuse rules; consult the
suite reference before retrying. A retry can incur additional model and
evaluation cost.

After the one-task result is understood, broaden the explicit filters or limit.
SWE-bench exposes `--max-concurrent-instances`; DeepSWE executes tasks serially.
Keep separate output/Corpus roots for different route or convergence treatments.
Current convergence plans bind three planners and the same per-task budget;
historical single-track receipts and qualification remain evidence only for
their original treatment.
See [SWE-bench](workflows/swebench.md) and [DeepSWE](workflows/deepswe.md) for all
parser options, artifacts, and exit mappings.
