<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `goal` — Refine a solution goal

Goal runs bounded Red, Blue, deterministic validation, and Judge rounds over
the promoted Spec intent, constraints, and done criteria.

```bash
uv run crucible goal --spec /path/to/corpus --max-rounds 3 --budget-usd 5
```

Revisions remain staged until a typed approval publishes the goal and evidence
in one Corpus transition. `final_goal.yaml` is retained as an output artifact;
review and validation records attach to the accepted lineage. Generated
revisions do not become source evidence. Goal refinement is optional before
planning; planning requires a valid promoted Spec, not a previous Goal run.
The budget covers every role across rounds. A Red review that ends
`inconclusive` stops the run before Blue, because an unchallenged goal is not
adjudicated.

## Options

`uv run crucible goal --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--backend-config` | Explicit closed routing YAML override; roles and discovery are defined by the workflow. |
| `--budget-usd` | USD ceiling; consult this workflow’s per-call or cumulative budget semantics. |
| `--max-rounds` | Max rounds. See parser help for accepted values and defaults. |
| `--max-turns` | Maximum conversational turns in the scope reported by this command’s --help. |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/goal.py](../../workflows/goal.py).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Judge issued `pass`. |
| `1` | Judge issued `revise`, including when the round limit was exhausted before `pass`. |
| `2` | Judge issued `reject`, or the Spec was missing before the debate. |
| `3` | Red or Judge issued `inconclusive`, the budget was exhausted, or no complete round ran. |
