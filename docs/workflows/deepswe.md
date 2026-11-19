<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `deepswe` — Run DeepSWE

DeepSWE selects tasks from a manifest and runs an isolated solution cascade.
Public repository source and development checks support authoring. A separate,
one-shot verifier alone receives private tests; its binary reward is the
benchmark result. Architect progress remains diagnostic.

```bash
uv run crucible deepswe --spec /path/to/deepswe-corpus \
  --dataset /path/to/deep-swe/tasks/manifest.json --limit 1 --list

uv run crucible deepswe --spec /path/to/deepswe-corpus \
  --dataset /path/to/deep-swe/tasks/manifest.json --limit 1 --implement \
  --budget-usd 20 --budget-usd-per-instance 20
```

Listing opens no run. Dry-run shows selected work without a model cascade.
Implementation runs selected tasks serially with individual Corpora, isolated
workspaces, container identities, storage budgets, and child operations.
Repository, language, category, and task filters define the explicit cohort.

Each task defaults to three independent concurrent complete plans. Each proposal
receives deterministic checks and adversarial review. Comparison may select a
plan or synthesize one that receives fresh acceptance gates. One unchanged task
budget covers every branch and subsequent implementation. Planning shares at
most 40%, with the implementation/verifier reserve of at least 50% and the 10%
contingency retained. A stricter remaining ledger allowance further limits
planning. Competing plans require a positive finite effective task cap before
container or provider setup.

The serial task loop does not serialize the active task's draft requests: up
to three can run concurrently. `--planner-count 1` explicitly selects the
single-track path and is required for `--diagnostic-cost-policy`; the same task
reserves still apply. Backend limit diagnostic mode alone does not select a
single planner.

Planner source dispatch and UTF-8 payload budgets span all planning children.
The defaults are 32 calls and 262,144 result bytes for the entire task, shared
by all three planners rather than granted to each. Competition uses one shared
call tranche. Explicit single-track planning retains its 24/4/4 discovery and
correction tranches and bounded no-tool refinement. A positive implementer read
ceiling prevents indefinite source browsing without successful writes.

Candidate capture validates source scope, patch size, and mutation settlement.
Pristine visible verification and hidden verification replay exact candidate
bytes independently. Private tests, hidden commands, evaluator mounts, and
truth artifacts stay outside authoring tools, prompts, and image context.

Before any model call the development gate runs once on the untouched
candidate and baseline checkouts, and both are restored. The paths that run
created are recorded where a later slice gate would charge them to its slice:
an untracked file Git does not ignore, and a new directory that Git does not
ignore or that is ignored below a directory the checkout already had. A new
top-level directory Git ignores as a whole, such as a cache that ignores
itself, and an ignored file are not recorded; a slice gate accepts them. Two
effects of that run are restored without being recorded, and fail a slice
whose gate repeats them: a path added below a directory that was already
untracked in the checkout, even an empty one, and a changed or deleted file
the checkout already had, tracked or untracked. The
task summary lists the candidate's record in
`dev_verify_preflight.gate_created_paths` (at most 256 of
`gate_created_path_count` paths). When `gate_created_paths_bound` is true,
the development verifier removes exactly those paths after each gate run in
which they were new, so they are not charged to a slice. A record that cannot
be bound exactly (more than 256 paths, more than 16,384 bytes of names, or a
name that is not a canonical UTF-8 path) is listed unbound and the task runs
as without it. Any other path a gate run leaves outside the slice's scope
still fails the slice.

A verifier result is a grade only when its reward artifact matches the private
tests snapshot. When that snapshot's `tests/config.json` declares a whitelist,
the grade is the DeepSWE 1.1 `reward.json`: exactly the contract fields, a
reward and ratios that follow from its own fail-to-pass and pass-to-pass
counts, and totals equal to the whitelist sizes. When the snapshot declares no
whitelist, the grade is the `reward.txt` token `0` or `1`, which carries no
counts. A missing artifact, a malformed one (`reward_invalid`), or one that
contradicts itself or the snapshot (`reward_mismatch`) leaves the task
`verification_failed` with no reward; it is never reported as reward 0.

The [profile catalog](../../products/deepswe/backend_profile_catalog.yaml)
records exact selectable adapters, models, transports, and diagnostic status.
Use the matching suite routing file. Diagnostic writer opt-ins and incomplete
cost intervals remain non-rankable; a nominal provider response does not
qualify a route. DeepSeek templates require
[route binding](backend_config_bind_deepseek.md) before execution.

Before binding a DeepSeek profile, install its locked worker using independent
file copies so the worker's single-link runtime identity checks can succeed:

```bash
uv sync --project backends/deepseek_worker --frozen --reinstall --link-mode copy
```

Checkpoints bind dataset, task, candidate, recipe, route-source bytes, policy,
image, and budget identities. The dataset pins a few tasks by an abbreviated
base commit: the first workspace population resolves it once, from the image
checkout, to the full object id that every Git comparison then binds, while
the dataset pin remains the task identity of checkpoints and measurements.
Resume accepts only matching evidence. `--rerun`
explicitly replaces completed task attempts; repeatable `--rerun-status` filters
that selection. Count, planning share, execution fingerprint, and planning
source recipe also bind the accepted plan. A completed candidate or hidden
verification checkpoint mismatch requires explicit rerun selection; it never
automatically consumes hidden verification again. Once execution is authorized,
a retained DAG without the matching benchmark planning stamp is freshly
planned. A matching stamp permits normal plan reuse without planning calls;
`--replan` explicitly requests fresh planning. Benchmark policy does not admit
exact digest reacceptance.
Historical convergence receipts retain their original planning identity.
Skipping disk preflight does not disable per-task quotas.

Run artifacts include aggregate and per-task summaries, candidate and verifier
evidence, lifecycle records, cost completeness, and typed outcomes. Report
resolved/unresolved tasks, binary rewards, incomplete verification, failed
setup, cleanup, and budget terminals separately. A task is resolved only when
its verifier and cleanup contracts complete.

`summary.json` keeps unsolved, ungraded, and unreached tasks apart. Of the
`tasks_selected` tasks, `tasks_total` have a task summary, `tasks_unrecorded`
were handed to execution but ended without one (a lifecycle conflict or a
checkpoint or lease error, possibly after the cascade ran), and
`tasks_not_started` were never reached. The task status decides what is a
grade: `tasks_graded` are the `resolved` and `unresolved` summaries
(`tasks_resolved` plus `tasks_unresolved`), `tasks_lease_busy` belong to
another process, and `tasks_ungraded` are the rest, executed or resumed with
any other status. `tasks_rerun_by_default` counts results a plain re-run does
not reuse; it is the lifecycle status policy, not a claim that the failure was
transient. `tasks_by_status` lists every task status.

A DeepSWE 1.1 task summary reports the `reward_counts` of its valid reward
artifact: fail-to-pass and pass-to-pass passed and total, and whether the
grader could not apply the patch. A valid artifact alone is not a grade. A
verifier that wrote one and then exited non-zero, timed out, or failed the
candidate patch proof leaves the task `verification_failed`; its summary and
lifecycle record still show the `reward` and `reward_counts` the artifact
held, and the task stays ungraded. The run-level `reward_counts` sums the
per-test counts of graded tasks only: `tasks_counted` of them carry counts and
`tasks_apply_failed` of those report a patch that did not apply. These counts
do not change the exit status.

## Options

`uv run crucible deepswe --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--backend-config` | Explicit closed routing YAML override; roles and discovery are defined by the workflow. |
| `--backend-limit-deadline-seconds` | Finite positive monotonic deadline for each provider invocation; pair with limit mode. |
| `--backend-limit-mode` | Explicit strict or diagnostic provider-limit admission; pair with the invocation deadline. |
| `--budget-usd` | USD ceiling; consult this workflow’s per-call or cumulative budget semantics. |
| `--budget-usd-per-instance` | Whole-task USD cap applied independently to each cascade. Planning retains its 40% allowance, implementation/verifier at least 50%, and contingency 10%, across all planners. |
| `--category` | Whitelist by category (feature_request / bugfix / enhancement); repeatable. |
| `--dataset` | Path to the deep-swe clone's tasks/manifest.json. Fetch out-of-band: `git clone https://github.com/datacurve-ai/deep-swe.git`. |
| `--diagnostic-cost-policy` | Explicit diagnostic-only conservative cost interval policy; requires `--planner-count 1` and results remain non-promotable. |
| `--dry-run` | Print the selected task plan without running the cascade. |
| `--implement` | Run the cascade per task (plan → tier → breach → review → verify) in a serial task loop. |
| `--language` | Whitelist by language (typescript / python / go / rust / javascript); case-insensitive; repeatable. |
| `--lifecycle-out` | If set, checkpoint one durable lifecycle record per task (`{task_id, ext_id, status, rc, wall_seconds, ...}`) in JSONL format. Repeated runs repair a torn tail, skip already-recorded task identities, and never add duplicate rows. This is a LIFECYCLE report, not a leaderboard predictions JSONL — model.patch capture from /logs/artifacts/model.patch is a separate concern not exposed by this flag. |
| `--limit` | Cap the post-filter task count (default: no cap). |
| `--list` | Print the filtered task list (task_id\trepo\tlanguage\tcategory) and exit. No cascade work. |
| `--max-patch-bytes` | Absolute candidate/model.patch byte cap (default: 64 MiB; bounded above by 512 MiB). |
| `--max-turns` | Maximum conversational turns in the scope reported by this command’s --help. |
| `--no-include-breach` | Disable the breach loop for this run. Default ON via DEEPSWE_BREACH_POLICY.enabled=True. |
| `--planner-count` | Independent complete proposals per task, 1–8; default 3 sharing one task's planning allowance. Explicit 1 selects the single-track path. |
| `--planner-tool-call-limit` | Planner tool call limit. See parser help for accepted values and defaults. |
| `--planner-tool-result-bytes` | Planner tool result bytes. See parser help for accepted values and defaults. |
| `--replan` | Explicitly regenerate each task's DAG with the selected planner count and DeepSWE rubric. A changed benchmark planning identity already forces fresh planning during an authorized attempt. |
| `--repo` | Whitelist by repo (owner/name short form, e.g. ``abs-lang/abs``); repeatable. |
| `--rerun` | Re-execute every selected task even when its canonical checkpoint is reusable. |
| `--rerun-status` | Re-execute checkpoints with this status; repeatable. Pre-oracle infrastructure statuses retry by default; hidden oracle statuses require this flag or --rerun. |
| `--skip-disk-check` | Skip the free-disk floor check. Equivalent to setting CRUCIBLE_DEEPSWE_SKIP_DISK_CHECK=1. |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |
| `--task-id` | Whitelist by task_id; repeatable. Empty (default) = no task-id filter. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/deepswe.py](../../workflows/deepswe.py).

## Exit status

| Code | Meaning |
|---|---|
| `0` | Listing or dry-run succeeded, or every implemented task resolved and cleanup completed. |
| `1` | Dataset content/filter failure, empty implement selection, any non-resolved task, incomplete cost, cleanup failure, lifecycle/lease conflict, or budget halt. |
| `2` | Missing dataset, invalid routing or CLI input, missing mode, unavailable Docker, or container/disk preflight failure. |

`result.json`/`outcome.json` distinguish ordinary permanent `tasks_failed`
from `retryable_tasks_failed` and `inconclusive_tasks_failed`; all three use
exit code `1`. `retryable_tasks_failed` means every failed task stopped before
the hidden oracle with a retryable Architect terminal.
`inconclusive_tasks_failed` means no failed task was judged: each one either
stopped before Architect was started (`setup_failed` or
`toolchain_unavailable`, including a development-verifier preflight that was
red or could not run), or stopped before the hidden oracle with an Architect
terminal that is not a ruling on its work (inconclusive, retryable, or
cancelled). Both need complete cost accounting and cleanup for every failed
task, and one judged task keeps `tasks_failed`. The status is a label for the
run: it changes no task status, checkpoint, or rerun rule, and it is not a
claim that another attempt would succeed.
