<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `tier` — Execute and gate one tier

Tier executes one accepted tier using directive-scoped mutation transactions,
then smoke/build/verify and its selected adversarial gates.

```bash
uv run crucible tier --spec /path/to/corpus --tier tier-id \
  --cwd /path/to/repository --budget-usd 10
```

Startup restores eligible settled journals and partitions resume state from
attached records. Each slice receives a confined implementer, source scope,
and read-progress ceiling. Deterministic code executes language smoke checks
and configured build/verification, with bounded implementer retries for failures.
Only verified, in-scope changes can enter the tier union diff.

Breach and rubric review follow typed policy and explicit options. An optional
generic smoke benchmark needs an authored runner and `metric_directions` for
every metric it compares. There is no product registry or inferred metric
direction. Missing direction is a setup failure, not permission to ignore a
regression. Source and metric gates retain separate evidence.

Before final acceptance, Tier recaptures exact shipping bytes and checks their
plan binding, adversarial state, scope, and final gate evidence. Budget, model
transport, incomplete accounting, failed review, or unresolved writer settlement
cannot authorize a partial tier. A failed run restores only after exact
settlement and retains its typed terminal reason.

Programmatic `run_tier(..., cwd=...)` requires an explicit source root. Its
optional `writer_frontier` binds one exact directive per lane, independent
checkouts/journals/backends, deterministic integration order, and one owning
landing transaction. It rejects incompatible breach, review, smoke variants,
restart, wrapper worktree, and skipped verification before writing. A concurrency
count never authorizes simultaneous writes to a shared checkout. See the
[orchestration facade](../../orchestration.py).

`--worktree` isolates an invocation. Repeating without `--restart` resumes only
records accepted by the current plan and lineage contract. Nested source
repositories must be initialized before creating the worktree.

## Options

`uv run crucible tier --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--backend-config` | Explicit closed routing YAML override; roles and discovery are defined by the workflow. |
| `--backend-limit-deadline-seconds` | Finite positive monotonic deadline for each provider invocation; pair with limit mode. |
| `--backend-limit-mode` | Explicit strict or diagnostic provider-limit admission; pair with the invocation deadline. |
| `--breach-tests` | Override the breach test file path (forwarded to workflows.breach). |
| `--budget-usd` | USD ceiling; consult this workflow’s per-call or cumulative budget semantics. |
| `--build-retries` | On a build_failed / verify_failed slice, re-invoke the implementer with the compiler/test error + its prior diff as feedback, up to this many times, before halting. Lets non-trivial edits self-correct strict (-Werror) build slips. 0 disables. Default: 2. |
| `--build-timeout-s` | Per-slice build subprocess timeout in seconds (default: 3600). Set 0 or a negative value to disable the cap. |
| `--cwd` | Repo root for build/verify and breach subprocesses (default: process cwd). |
| `--diagnostic-cost-policy` | Explicit diagnostic-only conservative cost interval policy; results remain non-promotable. |
| `--implementer-read-call-limit` | Maximum consecutive source-tool reads per implementer call without a successful write (default: 96). The workflow warns near the boundary and denies only further reads; a successful write resets the counter. |
| `--max-breach-rounds` | Override max Breaker/Defender/Judge cycles in Phase 3. |
| `--max-concurrent-slices` | Requested model concurrency. Mutations sharing one checkout are always staged serially under exclusive transactions; parallelism is only safe across distinct checkout roots. |
| `--max-turns` | Maximum conversational turns in the scope reported by this command’s --help. |
| `--no-reap` | Skill-internal postmortem opt-out: skip exact mutation-journal recovery and lifecycle-ordered retained-state reconciliation at Tier startup. Default behavior restores dead journals first, then cleans only current-plan attached failure/scope evidence that cannot overlap newer or accepted bytes. Retained-state cleanup is atomic and publishes a CAS-bound settlement. Forward --no-reap only when an operator must preserve the crashed intermediate state. |
| `--persistence-veto-threshold` | Override the breach persistence veto threshold. |
| `--response-retry-budget` | Response retry budget. See parser help for accepted values and defaults. |
| `--restart` | Skill-internal opt-out: ignore prior Implementation records for this tier and re-run every slice. Default behavior auto-resumes — slices whose latest persisted Implementation classification is `success` or `soft_fail_no_changes` are skipped; everything else (hard_fail_*, transport, build_failed, verify_failed) retries. The skill layer forwards --restart when an operator explicitly wants the full tier redone. |
| `--skip-breach` | Skip Phase 3 breach pass. |
| `--skip-build-verify` | Skip per-slice build/verify. Implementer classification stays the verdict. |
| `--skip-mutation-scope` | Skip the per-slice mutation_scope check that enforces source-level marker regions declared on directive.mutation_regions. Slices without mutation_regions are unaffected; this flag exists for debugging the enforcement layer itself on a tier whose markers may have drifted. |
| `--skip-review` | Skip Phase 4 rubric reviews. |
| `--skip-smoke-bench` | Skip the Phase 2.5 tier-level smoke bench. The smoke bench runs only when the tier carries a smoke_bench_config; this flag lets operators bypass even that when investigating a tier regression manually. |
| `--skip-smoke-check` | Skip the Phase 1.5 language-aware smoke check (cargo check / compileall + ruff F) that runs before per-slice build/verify. The smoke check is cheap (~seconds) and catches syntax-only diffs before they burn the full build (~minutes). Use only if the smoke check is itself problematic for a specific tier. |
| `--smoke-baseline-commit` | Pin the Phase 2.5 baseline lookup to a specific commit's measurement. Default None = use the latest stored measurement for the tier's smoke_bench_config.bench_id regardless of commit. |
| `--smoke-regression-threshold` | Relative-regression threshold for the Phase 2.5 smoke bench. A shared metric whose delta (current-baseline)/baseline exceeds this halts the tier (default: 0.20 = 20%%). Single-sample variance is high; keep this generous unless the bench has noise-level guarantees that justify tightening. |
| `--smoke-timeout` | Per-slice smoke-check subprocess timeout in seconds (default: 120). Hard cap so a runaway cargo check on a misconfigured tree can't stall the tier. |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |
| `--tier` | Tier id (e.g. 't0', 't1') from spec.tier_dag. |
| `--verify-timeout-s` | Per-slice verification subprocess timeout in seconds (default: 3600). Set 0 or a negative value to disable the cap. |
| `--worktree` | Run the tier in a fresh git worktree off the cwd's HEAD instead of the live checkout. Provides full isolation between concurrent crucible runs against the same submodule and free per-tier rollback (the entire worktree is removed on exit). Default: off (uses the selected checkout — implementer writes land in the operator's checkout). Opt in for hostile-concurrent scenarios or when the operator wants a clean substrate they can throw away. Constraints: (1) targets spec-root work — specs that need a nested submodule's files (nested source repositories) require the submodules to be initialized in the source checkout BEFORE this runs; the worktree inherits the source's submodule pointers but does NOT re-init them. (2) Implies --restart on resume — the prior worktree's writes can't be recovered, so resume of a crashed worktree run starts every pending slice fresh against a new worktree. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/tier.py](../../workflows/tier.py).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Tier complete — all slices, breach, and review passed. |
| `1` | A slice's implementer / smoke / mutation_scope / build / verify failed; or Phase 2.5 surfaced a regression beyond threshold; or Phase 3 breach exited non-zero for a non-reject reason. |
| `2` | Setup/preflight failed, the tier id was absent, or Breach Judge issued `reject`. |
| `3` | A non-advisory review remained failing after its configured retry budget. |

The `summary.json` `status` field disambiguates exit-1 sub-cases:
`slice_failed` (covers implementer / smoke / mutation_scope / build /
verify failures), `all_slices_transport_failed`,
`incomplete_transport_failed`, `hard_fail_tool_mutation_indeterminate`,
`budget_exhausted`, `empty_union_diff`,
`smoke_regression`, `smoke_bench_error`, `smoke_direction_config_error`,
`breach_exception`, and other typed Breach failures.
`all_slices_transport_failed` and `incomplete_transport_failed` remain
non-successful and cannot authorize a partial tier, but `outcome.json`
classifies them as `retryable_failure`. This preserves the Slice transport
type for a fresh parent attempt after mutation and accounting settlement;
it does not continue Phase 2, breach, review, or promotion in the failed run.
An accepted breach whose final recapture, adversarial-state gate,
shipping-state gate, or plan binding cannot be proven against the original
plan terminates as `final_binding_failed`.
