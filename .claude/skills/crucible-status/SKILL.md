---
name: crucible-status
description: Read the saved state of a Crucible solution or benchmark Corpus, list known specs, and recommend the next action from typed evidence.
allowed-tools: Read, Bash, Glob
---
<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->


# /crucible-status

Read status without changing the Corpus or launching a workflow. Use the
[shared discovery contract](../_shared/spec_discovery.md) to resolve an exact
name or canonical path. Surface ambiguous matches for the operator to choose.
For a list request, call `list_all_specs` and render its target-grouped output.
First resolve the physical skill folder and `CRUCIBLE_DIR` using
[preflight.md](../_shared/preflight.md). Locate the target workspace separately;
do not create a repository or Spec during a status request. An empty folder
has no saved solution yet; recommend conversational `crucible-design`.

## Inspect the selected Corpus

Run the read-only helper from the Crucible project:

```bash
uv run --project "$CRUCIBLE_DIR" python \
  "$CRUCIBLE_DIR/.claude/skills/_shared/bin/spec_state.py" \
  "$SPEC_DIR"
```

The helper scans filenames and selected YAML scalars; it never constructs a
Corpus or creates directories. Report the resolved Spec and source paths,
latest version/manifest, run counts for design/build/benchmark/recovery, and
most recent run. Counts are navigation hints, not proof of acceptance.

Read the latest manifest and the relevant `outcome.json`, `result.json`, and
`summary.json` before recommending a next action:

- An unpromoted goal needs authoring and promotion before execution.
- A promoted goal without an accepted plan can proceed to planning after the
  user's explicit approval of that goal; otherwise request goal approval.
- An accepted plan with pending work can proceed to Architect, Tier, or
  handoff for externally completed work.
- A benchmark run needs suite-specific per-instance outcomes and verifier
  evidence; source changes alone do not establish a solved task. Report its
  planner count and planning identity. The default is three planners within
  the same task budget; an explicit single-track diagnostic keeps that identity.
- A completed benchmark checkpoint conflict needs the suite's explicit rerun
  selection. A mismatched retained plan is freshly planned only when execution
  is authorized. A matching stamp permits normal plan reuse. Do not recommend
  digest reacceptance for benchmark attempts.
- A failed or interrupted run needs the owning workflow's resume or recovery
  path, with its recorded terminal reason.

Do not launch the recommended command. If the operator requests recent
detail, show the saved terminal status and a bounded event tail. Missing
evidence stays missing; timestamps never establish child completion.
