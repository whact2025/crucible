<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `swebench` — Run SWE-bench Verified

SWE-bench runs the reusable Architect/Tier cascade against selected
[SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
instances. It captures source-only candidate bytes, applies submission policy,
replays the candidate through independent pristine visible verification, and
optionally invokes the official hidden harness inline.

```bash
uv run crucible swebench --spec /path/to/swebench-corpus \
  --dataset /path/to/verified.json --limit 1 --list

uv run crucible swebench --spec /path/to/swebench-corpus \
  --dataset /path/to/verified.json --limit 1 --implement \
  --budget-usd 20 --budget-usd-per-instance 20
```

`--list` opens no run. `--dry-run` writes empty-patch leaderboard records without
model or harness calls. Implement mode enables breach, review, per-slice
visible build/verify, pristine visible verification, and official inline
evaluation by default. Disabling mutable build/verify does not disable the
independent pristine gate or Breach's deterministic checks.

Every instance has a separate Corpus under
`<spec>/instances/<instance_id>/corpus/`, standalone source checkout, container,
budget identity, and Architect child operation. A root Corpus can be synthesized
when empty. Concurrent instance workers never share task Spec authority.

Each instance defaults to three independent concurrent planners. Deterministic
checks and adversarial review evaluate each complete proposal; comparison may
select one or synthesize a plan that must pass fresh gates. All planning roles
share at most 40% of the instance's unchanged total allowance. The effective
task cap comes from `--budget-usd-per-instance`, the aggregate budget's worker
allocation, or the smaller of both. Competing plans require that cap to be
finite and positive. An explicit `--planner-count 1` selects the single-track
diagnostic path without the competition's 40% partition.

Task-worker concurrency is separate from planner concurrency. Each active
instance can issue three simultaneous draft requests; two workers can issue
six. Source-discovery limits remain shared across the instance's planning
operation. The planner count does not expand mutation or verifier authority.

Patch capture admits production source and rejects test-only, out-of-scope,
metadata, or ambiguous candidates according to suite policy. Exact captured
bytes drive pristine replay and prediction publication. Hidden tests are
available only to deterministic evaluator code. Official evaluation outcome
remains distinct from the authoring cascade classification. A cascade that
ends on its budget ceiling, like one that raises, is partial work: its
candidate stays in the Corpus, the instance is reported
`cascade_aborted_partial`, and official inline evaluation is skipped even when
`--submit-any-nonempty-best-effort` publishes the patch.

The [profile catalog](../../products/swebench/backend_profile_catalog.yaml)
names exact selectable adapter/model/transport identities. Bind an explicit
profile with `--backend-config` to change routes. Diagnostic Kimi, DeepSeek,
local, and OpenAI-compatible profiles retain their declared admission and
completeness limits. The diagnostic Kimi SDK profiles need the cost
upper-bound policy, which `crucible swebench` has no option to select: a
budgeted suite run on one of them pays for the planner replies and then stops
on incomplete accounting, so use them with `architect`, `tier`, `breach`, or
`review` and the policy. DeepSeek construction templates require
[route binding](backend_config_bind_deepseek.md). Catalog metadata does not
establish launch or promotion authority.

Before binding a DeepSeek profile, install its locked worker using independent
file copies so the worker's single-link runtime identity checks can succeed:

```bash
uv sync --project backends/deepseek_worker --frozen --reinstall --link-mode copy
```

Convergence tooling expands deterministic cohorts and byte-bound experiment
plans. Preserve separate matrix and analysis identities; do not pool different
model/transport treatments or reinterpret one matrix's results under another.
The product guide describes [its modules](../../products/swebench/CLAUDE.md).

Resume validates dataset, recipe, image, route-source, semantic routing,
pricing, policy, candidate, and budget identities before checkpoint reuse.
Planner count, planning budget share, and planning source recipe participate
in that identity. A completed prediction checkpoint mismatch remains a conflict
until an explicit rerun selection; it never silently reruns hidden evaluation.
Once an attempt is authorized, a retained DAG without the current benchmark
planning stamp is freshly planned. A matching stamp permits normal plan reuse
without planning calls. Benchmark policy rejects exact digest reacceptance;
historical convergence evidence keeps its original planning identity.
Use `--rerun` or explicit `--rerun-status` only for the intended cohort.
Predictions publish durably with deduplication. `--predictions-out` selects the
output; otherwise use the run's `predictions.jsonl`.

Report selected/completed instances, submitted candidates, harness results,
failed setup, cost completeness, skipped gates, and cleanup. A zero scheduling
exit can coexist with unresolved individual predictions; inspect their outcomes.

## Options

`uv run crucible swebench --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--backend-config` | Explicit closed routing YAML override; roles and discovery are defined by the workflow. |
| `--budget-usd` | USD ceiling; consult this workflow’s per-call or cumulative budget semantics. |
| `--budget-usd-per-instance` | Whole-instance cost ceiling across planning, implementation, and gates. Competition shares at most 40% for planning. Aggregate worker allocation may impose a smaller effective cap. An aborted cascade retains captured diagnostic candidates under suite policy. |
| `--dataset` | Path to a SWE-bench Verified instances file (JSON list or JSONL). Fetch the dataset out-of-band, e.g. via ``huggingface-cli download princeton-nlp/SWE-bench_Verified``. |
| `--dry-run` | Walk the filtered subset and emit a zero-patch predictions file (every Prediction.model_patch = empty string). Used to validate dataset + filter + predictions shape before spending LLM budget. |
| `--eval-timeout` | Per-instance official harness subprocess timeout in seconds; default 7200. |
| `--implement` | Execute the isolated Architect/Tier cascade, capture the source candidate, run pristine visible verification, and publish predictions plus optional official evaluation. |
| `--instance-id` | Whitelist by instance_id; repeatable. Empty (default) = no instance-id filter. |
| `--limit` | Cap the post-filter instance count (default: no cap). |
| `--list` | List instance ids + repo + base_commit for the filtered subset; no backend invocation, no predictions emitted. |
| `--max-concurrent-instances` | Bound independent per-instance workers; default 1. Each active worker can also issue three concurrent draft requests. Use separate task workspaces and an acquisition cache with concurrent runs. |
| `--max-turns` | Maximum conversational turns in the scope reported by this command’s --help. |
| `--no-build-verify` | Opt out of the per-slice visible-test gate. The isolated authoring container remains mandatory because benchmark code and agent shell commands are never executed on the host. The final patch-only pristine visible gate remains mandatory. Hidden evaluation tests are never used by either gate. |
| `--no-evaluate-inline` | Disable official hidden evaluation during this run. Candidate and visible-verification evidence remain distinct from a solved-task result. |
| `--no-include-breach` | Skip the per-instance breach (adversarial test) phase. Default: on. Off is for smoke and ablation runs only. |
| `--no-include-review` | Skip the per-instance review phase. Default: on (run_architect.include_review=True). Off is for cheap-smoke / ablation runs only. |
| `--planner-count` | Independent complete plan proposals per instance, 1–8; default 3 within one task budget. Explicit 1 selects single-track diagnostics. |
| `--predictions-out` | Override the predictions JSONL path. Default: <run_dir>/predictions.jsonl. |
| `--repo` | Whitelist by repo (e.g. ``django/django``); repeatable. Empty (default) = no repo filter. |
| `--repo-cache-dir` | Directory for reusable acquisition caches plus isolated per-instance checkouts. The cache accelerates downloads but is never shared with or mounted into authoring sandboxes. Default: direct ephemeral fetches. |
| `--rerun` | Re-execute the selected instances instead of reusing matching completed checkpoints. This can incur additional model and evaluation cost. |
| `--rerun-status` | Re-execute reusable terminal classifications of this kind, including empty source predictions whose test edits were filtered; repeatable. Infrastructure failures retry by default. |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |
| `--submit-any-nonempty-best-effort` | Submit the latest non-empty source patch even when its Implementation is classified as build_failed, verify_failed, dev_verify_failed, another non-success outcome, or the final pristine verifier fails. Default: off; failed candidates and verifier reports remain in the Corpus for diagnostics but are surfaced as empty predictions. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/swebench.py](../../workflows/swebench.py).

## Exit status

| Code | Meaning |
|---|---|
| `0` | Listing/dry-run completed, or implement scheduling and accounting completed without an aggregate-budget or incomplete-cost terminal. Individual predictions may still be unresolved or failed. |
| `1` | Invalid dataset content, explicit filters matched nothing, implement dataset was empty, no mode was selected, storage quota failed, aggregate budget halted the run, or cost accounting was incomplete. |
| `2` | Dataset path missing, Docker unavailable, or mandatory container/disk preflight failed. |

Argument parsing errors use `argparse`'s exit status. Routing errors outside a
handled terminal propagate rather than being remapped by this workflow.
