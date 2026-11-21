<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `architect` — Build a complete solution

Architect establishes or reuses an accepted plan, validates its authority,
and executes tiers in topological order through the shared Tier workflow.
Every composite operation retains child saga bindings and a cumulative budget.

```bash
uv run crucible architect --spec /path/to/corpus --cwd /path/to/repository \
  --planner-count 3 --include-breach --include-review --budget-usd 20
```

The default path builds and verifies every slice. Breach and review follow the
selected product/policy and explicit options. Directly reused plans must retain
current source and plan-contract provenance. `--replan` authorizes replacement.
For ordinary solutions, `--reaccept-plan-digest` instead binds an exact retained
DAG and reruns current deterministic checks. These options are mutually exclusive.

Fresh general solution planning defaults to three independent concurrent
proposals, adversarial evaluation, and a hash-bound comparison. The comparison
may select an accepted whole plan or propose a full synthesis; a synthesis
requires fresh review and deterministic validation. See [Plan](plan.md) for
snapshot isolation, evidence, limits, and reserved budget shares. All discarded
attempts and comparison calls consume the same Architect budget as subsequent
implementation. A positive finite budget is required when multiple planners
actually run; retained-plan reuse and ordinary solution exact reacceptance do
not launch planners.

`--planner-count 1` uses the bounded single-track loop. SWE-bench and DeepSWE
also default to three planners, sharing at most 40% of each task's unchanged
total budget for the complete planning competition. Their explicit count-one
override is available for diagnostics. Benchmark planning stamps bind the
accepted terminal, DAG, count, budget share, execution identity, and source
recipe. A matching stamp permits normal plan reuse without planning calls;
a mismatch requires fresh planning. Typed benchmark policy rejects exact digest
reacceptance, and `replan=True` requests a fresh benchmark plan.
Programmatic `run_architect(..., cwd=...)` requires an explicit repository root.
Its separate opt-in read-only analysis fan-out joins reports into one planner
and therefore requires `planner_count=1`; it does not produce competing complete plans.
Diagnostic cost pilots likewise require explicit single-track planning.
Discovery budgets cover all participating children. Shared-checkout writes
remain serialized; separate
`worktree=True` invocations isolate concurrent solutions.

`--stop-after-tier` produces a partial checkpoint, not a complete solution.
Resume uses attached implementation and terminal evidence. A failed child
stops the current topological execution; a retryable terminal permits a fresh
attempt after settlement rather than silent continuation.

## Options

`uv run crucible architect --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--backend-config` | Explicit closed routing YAML override; roles and discovery are defined by the workflow. |
| `--backend-limit-deadline-seconds` | Finite positive monotonic deadline for each provider invocation; pair with limit mode. |
| `--backend-limit-mode` | Explicit strict or diagnostic provider-limit admission; pair with the invocation deadline. |
| `--breach-tests` | Override the breach test file path (forwarded to run_tier when --include-breach). |
| `--budget-usd` | USD ceiling; consult this workflow’s per-call or cumulative budget semantics. |
| `--build-retries` | On a failed build or verification, let each tier re-invoke its implementer with failure feedback up to this many times. 0 disables retries. Forwarded unchanged to every run_tier dispatch. Default: 2. |
| `--build-timeout-s` | Per-slice build-phase subprocess timeout in seconds (forwarded to run_tier → _build_verify). Default matches DEFAULT_BUILD_TIMEOUT_S (60 min). Raise for large builds — e.g. --build-timeout-s 7200. |
| `--critic-rubric` | Critic rubric. See parser help for accepted values and defaults. |
| `--cwd` | Working directory for build/verify subprocesses (default: process cwd). Must be the target repo root. |
| `--diagnostic-cost-policy` | Explicit diagnostic-only conservative cost interval policy; results remain non-promotable. |
| `--enable-source-tools` | Attach path-confined build_source_tools(ctx) to the planner backend while leaving SDK-native built-ins disabled. Default: off (the planner sees only the spec + critic feedback). Enable when planning must ground files_in_scope, paths, and symbols in the checkout. Cost: additional planner-side read passes. |
| `--implementer-read-call-limit` | Maximum consecutive source-tool reads per implementer call without a successful write (default: 96). Forwarded to every tier; writes reset the counter. |
| `--include-breach` | Run the multi-round Breaker / Defender / Judge loop against each tier's union diff after build+verify. Overrides an authored spec policy; when no policy is authored, breach remains off. |
| `--include-review` | Run the per-tier rubric pass (diff_scope + spec_clarity + tier_completeness) after breach. Off by default; programmatic callers opt in when their cascade contract requires review. |
| `--judge-rubric` | Judge rubric. See parser help for accepted values and defaults. |
| `--max-breach-rounds` | Override max Breaker/Defender/Judge cycles per tier. |
| `--max-concurrent-slices` | Requested model concurrency. Mutations sharing one checkout are always staged serially under exclusive transactions; parallelism is only safe across distinct checkout roots. |
| `--max-turns` | Maximum conversational turns in the scope reported by this command’s --help. |
| `--no-reap` | Skill-internal postmortem opt-out: skip exact mutation-journal recovery and lifecycle-ordered retained-state reconciliation at each tier's startup. Default behavior restores dead journals first, then atomically reconciles eligible retained failed state. Forward only when an operator must preserve the crashed intermediate state. |
| `--persistence-veto-threshold` | Override how many surviving rounds halt the tier. |
| `--planner-count` | Independent concurrent complete proposals, 1–8; default 3. One finite planning/build budget covers drafting, adversarial review, comparison, synthesis gates, and execution. Reuse and exact reacceptance make no draft calls. |
| `--planner-tool-call-limit` | Hard workflow-owned ceiling on planner source-tool dispatches. Must be paired with --planner-tool-result-bytes. |
| `--planner-tool-result-bytes` | Hard cumulative UTF-8 ceiling on successful planner tool payload bytes (workflow control/error annotations excluded). Must be paired with --planner-tool-call-limit. |
| `--reaccept-plan-digest` | For ordinary solutions, explicitly reaccept the retained DAG only when its canonical digest exactly matches this value. No planning models are called; current deterministic validation and v3 provenance binding remain mandatory. Rejected with typed benchmark planning policy. |
| `--replan` | Force re-planning even when the spec already carries an accepted tier_dag. Reuse is allowed only when its root/planner/reviewer provenance fingerprint still matches; stale plans fail closed until --replan explicitly authorizes replacement. |
| `--replan-on-validate-fail` | If the existing tier_dag fails the validation gate, re-run plan() to produce a fresh tier_dag and validate that. Without this flag, a failing validation aborts. |
| `--restart` | Skill-internal opt-out: ignore prior Implementation records and re-run every slice in every tier. Default behavior auto-resumes skip-worthy slices (success / soft_fail_no_changes) from the Corpus. Forwarded to run_tier. |
| `--skip-build-verify` | Skip the build+verify subprocess wrapping per slice. By default, every assay.implement result is fed to _build_verify.build_and_verify(...) and the Implementation.classification is overwritten with the richer outcome (success / build_failed / verify_failed / partial / runner_missing). |
| `--skip-validate` | Skip generic validation rules, never read-only source identity checks for fresh or reused plans. By default, architect runs assay.validate against the tier_dag before per-slice implement and halts on any Critical/High finding. Pass --skip-validate to override (NOT recommended; the gate catches plan-quality issues the implementer would otherwise have to clean up). |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |
| `--stop-after-tier` | Stop after the given tier index (0-based, topo order). |
| `--verify-timeout-s` | Per-slice verify-phase subprocess timeout in seconds (forwarded to run_tier → _build_verify). Default matches DEFAULT_VERIFY_TIMEOUT_S (60 min). Raise for large test suites — e.g. --verify-timeout-s 14400. |
| `--worktree` | Open one fresh git worktree off cwd's HEAD for the Architect invocation. Every dispatched tier uses that worktree, which is cleaned up when Architect exits. Provides checkout isolation between concurrent Crucible runs. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/architect.py](../../workflows/architect.py).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All tiers completed, or a requested `--stop-after-tier` checkpoint produced explicit `partial_stopped` status. |
| `1` | A tier failed, the single cumulative budget was exhausted during planning, validation repair, or tier execution, or corpus-backed cost accounting became incomplete. |
| `2` | Planning returned a non-budget rejected/inconclusive outcome, or the pre-implementation validation gate failed for a non-budget reason. |

An exit-1 summary may report `retryable_tier_failure` when every failed child
Tier supplied an immutable `retryable_failure` terminal. This is retry
classification only: Architect has already stopped and has not promoted a
partial topological result.

In the same way an exit-1 summary reports `cancelled` when every failed child
Tier was cancelled, and `inconclusive_tier_failure` when every failed child
Tier ended without a ruling on its work (an inconclusive, retryable, or
cancelled terminal, for example `slice_inconclusive`, `breach_inconclusive`,
or `review_inconclusive`). One judged, contract-error, or untyped Tier keeps
`partial_failed`; incomplete accounting and budget exhaustion are reported
first. Neither status continues the walk, retries a Tier, or promotes a
partial result.

A validation-repair re-plan that nobody ruled on keeps its own typed status
under exit `2` (for example `planner_infrastructure_failure` when the Planner
backend was lost) instead of `replan_validate_failed`. Planning reports the
same status when no Planner backend returned a reply: by the end of the round
budget with one planner, when no independent draft came back with several. It
reports `critic_`/`judge_infrastructure_failure` or
`critic_`/`judge_protocol_failure` when a draft reviewer or the comparison
reviewer gave no verdict while a candidate had no ruling against it;
`outcome.json` files these as `inconclusive`. When such a call stopped on its
budget instead, the status is `budget_exhausted` under exit `1`.
`plan_competition_inconclusive` under exit `2` is kept when a ruling ended the
run (the comparison selected nothing, or every candidate was refused) and when
the records do not prove that a lost call ended it, for example when every
planner stopped on its turn limit or a content rejection, or a draft's
deterministic gate could not run. A planner that stopped on its budget does
not change this while every draft that came back was refused: that run stays
under exit `2`. A lost call that reported incomplete usage stays an accounting
terminal at every planner count; the [plan reference](plan.md) lists the cases
that still differ from one planner.
