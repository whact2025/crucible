<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `implement` — Record one implementation attempt

Implement invokes the selected implementer against one accepted slice using
its directive-scoped source/write tools. It records an attempt and its captured
Implementation; it does not perform the complete Tier gate sequence.

```bash
cd /path/to/target-repository
uv run --project /path/to/crucible crucible implement \
  --spec /path/to/corpus --slice slice-id --budget-usd 5
```

This CLI uses the process current directory and has no `--cwd`. Validate that
source root before dispatch. Native executors and ambient tools cannot expand
directive scope. Successful capture, provider settlement, journal state, and
cost/trace completeness determine whether the workspace can be retained.
Failure restores the transaction after exact quiescence.

Use [Tier](tier.md) or [Architect](architect.md) when the requested outcome
includes smoke, build/verify, breach, review, and accepted completion. A
successful model call alone does not establish those gates.

## Options

`uv run crucible implement --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--backend-config` | Explicit closed routing YAML override; roles and discovery are defined by the workflow. |
| `--budget-usd` | USD ceiling; consult this workflow’s per-call or cumulative budget semantics. |
| `--max-turns` | Maximum conversational turns in the scope reported by this command’s --help. |
| `--slice` | Slice ID to implement (must exist in the spec's tier_dag). |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/implement.py](../../workflows/implement.py).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | `classification.startswith("success")`. |
| `1` | Any non-success classification (`soft_fail_*` / `hard_fail_*`, including `hard_fail_empty_diff`). |
| `2` | Missing Spec, missing/unknown slice, or mutation-journal recovery setup failure. |
