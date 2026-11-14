<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `journal_recover` — Recover one exact retained mutation

Journal Recovery checks or applies a narrowly authorized retained transaction.
It uses the exact run, transaction, diff, stash, source-root, and accepted
mutation identities supplied by the operator. It never infers broader authority
from a crashed process or a similar-looking checkout.

Start with a check-only request assembled from the retained journal and its
reported expected state. Apply the same request only when the user's task
explicitly authorizes recovery, adding `--apply`. The command requires the
full request; consult `--help` and the option table below rather than copying
placeholder identities.

Missing/malformed writer settlement, changed source or Git topology, unsafe
aliases, mismatched postimages, unauthorized additions, and ineligible journal
state fail closed. Recovery preserves an audit record and reports exact removed
paths, peer transactions, accepted tips/objects, and expected diff identity.
It updates the selected target journal and run audit rather than opening a new
workflow envelope. An interrupted valid recovery remains checkable/resumable;
a successful recovery is not solution completion.

## Options

`uv run crucible journal_recover --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--accept-ancestor-change-ref` | transaction-bound Corpus manifest for one exact preserved ancestor change |
| `--accept-appended-stash-tip` | Accept appended stash tip. See parser help for accepted values and defaults. |
| `--accept-removed-linked-worktree` | exact removed guard-only linked worktree path (repeatable) |
| `--apply` | perform recovery; without this flag the workflow is check-only |
| `--cwd` | outer workspace root; the latest Spec submodule_path is resolved exactly once beneath this directory |
| `--expected-diff-ref` | Expected diff ref. See parser help for accepted values and defaults. |
| `--reason` | nonempty operator reason retained in the durable audit |
| `--run-id` | Run id. See parser help for accepted values and defaults. |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |
| `--transaction-id` | Transaction id. See parser help for accepted values and defaults. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. This command follows its explicit check/apply or materialization contract
rather than creating the normal Corpus run envelope. Source:
[workflows/journal_recover.py](../../workflows/journal_recover.py).

## Output and exit codes

The command writes one JSON report to stdout. Fields include the request
identity, source root, removed paths and peer transactions, baseline and
accepted stash tips, accepted object additions, expected diff ref, optional
ancestor-submodule identity, and the durable audit path.

| Code | Meaning |
|---|---|
| `0` | The exact request is ready/checkable, an interrupted stage is still valid, or apply completed recovery. |
| `2` | Missing Spec, escaping source root, malformed authority, unknown or ineligible journal, state mismatch, unsafe alias/topology, or recovery verification failure. No broader recovery is attempted. |
