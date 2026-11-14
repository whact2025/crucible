<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `reconcile` — Reconcile durable solution state

Reconcile recovers eligible dead runs and settled mutation journals, checks
complete implementation/gate evidence against the current accepted plan,
validates deterministic rules, and can atomically land the evidenced result.
It makes no model calls.

```bash
uv run crucible reconcile --spec /path/to/corpus --cwd /path/to/repository \
  --dry-run
```

A directory or previous diff alone cannot establish completion. Reconciliation
binds exact postimages, provenance, typed records, rule results, and current
Spec identity. Missing evidence, unsafe journals, source-root drift, or a
concurrent Spec transition blocks landing. A dry-run reports eligibility
without claiming a landed solution.

Automatic recovery cannot manufacture writer quiescence. Transactions requiring
specific external authority remain retained for [Journal Recovery](journal_recover.md).

## Options

`uv run crucible reconcile --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--cwd` | Git workspace associated with mutation journals (default: cwd) |
| `--dry-run` | report the reconciliation decision without publishing records |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/reconcile.py](../../workflows/reconcile.py).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | `landed`, `no_work`, or an eligible `dry_run_complete`. |
| `1` | Missing plan/completion evidence, invalid postimages, rule findings, ineligible dry run, or a concurrent Spec change. |
| `2` | Missing Spec, escaping/changing source root, unsafe runner or journal state, or automatic mutation-recovery failure. |
