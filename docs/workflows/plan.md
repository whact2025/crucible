<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `plan` — Derive and adjudicate an executable plan

Plan derives independent proposals from one promoted goal, evaluates their
complete DAGs through deterministic checks and separate Critic/Judge sessions,
and compares their evidence before publishing one executable DAG.

```bash
uv run crucible plan --spec /path/to/corpus --cwd /path/to/context-root \
  --planner-count 3 --enable-source-tools --budget-usd 10
```

When `project_context.submodule_path` is set, `--cwd` is the outer context root;
source tools and validators compose the declared submodule once. Scope paths in
every model-authored plan (a single-track Planner reply, each independent
draft, a proposed synthesis) are canonicalized against that resolved project
root, never the process working directory; the scope paths of a
`--review-candidate` file are not rewritten. Source access
is explicit and read-only. Planner source-call and payload-byte ceilings are
workflow-owned and apply across the shared operation.

The default is three concurrent planners. Each receives the same goal and a
separate backend, tools, conversation, and debug directory. Source tools read
one frozen copy of the project's tracked and nonignored untracked files;
Corpus state, sibling proposals, instructions, secrets, and control files are
excluded by the existing source policy. Snapshot capture preserves current
source bytes and modes, including local edits, with limits of two MiB per
file, 128 MiB total, and 20,000 files. Read-only citations must match admitted
snapshot bytes and are independently checked against the original project.

Each draft receives one fixed-candidate adjudication. A malformed draft,
including one with a scope path that is empty after canonicalization, or a
deterministically invalid draft stays diagnostic; valid siblings can continue.
The separate comparison reviewer can choose an accepted candidate by exact
id/digest, abstain, or propose a full synthesis. A synthesis must pass fresh
Critic, Judge, and deterministic gates. The source Spec is unchanged; only the
publication owner advances it after checking the original source identity
under the Corpus lock. A source or concurrent Spec change blocks publication.

Parallel planning requires a positive finite total budget. It reserves 30%
for all concurrent drafts, 30% for all candidate reviews, 10% for comparison,
and 30% for fresh synthesis review. Unused shares are not additional spending
authority. Failed and discarded attempts count toward the same total. Provider
limits can be post-hoc: these are disjoint workflow allowances, not a provider
hard-stop guarantee. An overrun, missing pricing, or incomplete execution
evidence stops later calls and approval. Cancellation closes and joins leases
before a terminal is published. The configured route/model/effort is preserved.

`--planner-count 1` selects the bounded single-track refinement loop and
permits zero or one pre-review deterministic correction. Planner conversations,
cooperative planner control, and diagnostic validation overrides require this
mode. The low-level injected-backend `run_plan` API remains single-candidate;
workflow orchestration owns concurrency. Explicit
`--review-candidate` mode parses one fixed candidate and runs Critic/Judge
without acquiring a Planner backend. A changed fixed candidate needs a new
explicit input. Required source references and exact citation identities remain
mandatory even when generic validation is deliberately skipped.

An accepted result commits the DAG, review/validation refs, and plan-contract
provenance together. Rejection, inconclusive coverage, validation veto, stale
source identity, or failed final TOCTOU checks cannot leave an executable
candidate. Review response repair is separate from semantic plan rounds.

## Options

`uv run crucible plan --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--backend-config` | Explicit closed routing YAML override; roles and discovery are defined by the workflow. |
| `--budget-usd` | USD ceiling; consult this workflow’s per-call or cumulative budget semantics. |
| `--critic-rubric` | Name of a bundled rubric (under crucible/rubrics/). |
| `--cwd` | Workspace root containing the Spec's optional project_context.submodule_path. Planner backends and source tools run from the resolved project root. Default: process cwd. |
| `--enable-planner-control` | Enable the run-scoped heartbeat and cooperative pause/steer/cancel checkpoint before each planner provider call. Off by default. |
| `--enable-source-tools` | Attach build_source_tools(ctx) — read_source / grep_symbol / find_callers — as an in-memory MCP server so the planner can ground its tier_dag against verified repo paths through a confined read-only MCP surface. Off by default to preserve the minimal no-tools planning shape. |
| `--judge-rubric` | Required independent judge rubric name (default: plan_judge). |
| `--max-turns` | Maximum conversational turns in the scope reported by this command’s --help. |
| `--max-validate-replan-attempts` | Allow 0 or 1 Planner-only deterministic correction in single-track mode (default: 1). Parallel drafts and fixed candidates use 0. Correction does not repeat the full Critic/Judge cycle. |
| `--no-validate-gate` | Skip generic deterministic candidate validation and TOCTOU rules, never read-only source identity checks. Default: validate each candidate before paid Critic/Judge review, then recheck once before publication. |
| `--planner-count` | Independent complete proposals, 1–8; default 3. Parallel generation requires a finite shared total budget. Fixed-candidate review defaults to 1 and rejects larger values. |
| `--planner-recovery-actor` | Auditable operator identity attached to a not-started recovery resolution. |
| `--planner-recovery-evidence` | SHA-256 digest of external evidence proving the exact pending operation was not accepted by the provider. |
| `--planner-recovery-operation` | Exact pending sha256 operation digest to resolve as not started. Requires the evidence and actor flags. |
| `--planner-session-id` | Enable an opt-in durable workflow-bound planner conversation under this explicit session name. The selected route must declare workflow_bound continuity. --max-turns and --budget-usd become cumulative ceilings for this planner session as well as their ordinary workflow limits. |
| `--planner-session-state-dir` | Durable planner-session store. Requires --planner-session-id. Default: <spec>/.crucible/planner-sessions. |
| `--replace-plan-digest` | With --review-candidate only, replace this exact existing sha256 TierDAG after independent adjudication. Rejection retains the old DAG; publication requires the exact reviewed base Spec. |
| `--review-candidate` | Review this explicit TierDAG JSON without acquiring or invoking plan_planner. The fixed candidate receives one deterministic gate and one independent Critic/Judge adjudication; a non-pass requires a new invocation with corrected bytes. |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. A competition also
retains `plan_competition.json`, content-addressed physical call receipts and
prompts, every candidate and review, the comparison packet/decision, synthesis
gates, source identity, policy, and implementation digests. A draft's `raw_ref`
and the comparison response keep the model's text as written; candidate
artifacts, digests, reviews, the comparison packet, and the published DAG hold
the plan that was judged, with canonical scope paths. Losing drafts never
become canonical Spec versions. Source:
[workflows/plan.py](../../workflows/plan.py).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | `PlanOutcome.decision="accepted"`; the enabled validation gate passed and an executable DAG was persisted. |
| `1` | The accepted candidate failed the final deterministic TOCTOU recheck. |
| `2` | `PlanOutcome.decision="rejected"` or a handled setup failure. |
| `3` | `PlanOutcome.decision="inconclusive"`, including budget/round exhaustion, missing DAG, incomplete review coverage, or a Planner backend that returned no reply by the end of the round budget (`planner_infrastructure_failure`). |
