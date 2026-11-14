---
name: crucible-handoff
description: Capture work completed manually or by an interactive external agent as scoped Implementation records, with optional verification, breach, review, and landed lifecycle evidence.
allowed-tools: Read, Bash, Glob, Grep, Edit, Write
---
<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->


# /crucible-handoff

Use handoff when the target checkout already contains completed work that was
not produced by the autonomous tier pipeline. Handoff captures only the
selected directive scopes, optionally verifies and reviews the result, and
records explicit external/operator provenance.

## Preflight

Locate the physical skill folder and `CRUCIBLE_DIR` through
[the shared preflight](../_shared/preflight.md), then use
`REQUESTED_STAGE=build`.

Before editing or recording:

1. confirm the latest Spec has an executable TierDAG;
2. resolve `$TARGET_DIR` and show its current `git status`;
3. show the selected slice/tier directives, including intent,
   `files_in_scope`, `creates_paths`, mutation regions, requirement refs, and
   verify mode; and
4. distinguish preexisting unrelated changes from work intended for this
   handoff.

Targets are mutually exclusive: use one `--slice`, one `--tier`, or `--all`.
With no selector the workflow defaults to all pending slices, but the skill
must make that scope explicit to the operator before invocation.

## Produce the work

Offer the operator one mode:

- manual editing, followed by a fresh status/diff inspection;
- assisted editing in this interactive session; or
- autonomous `implement`, run from `$TARGET_DIR` because its CLI has no
  `--cwd`:

```bash
(
  cd "$TARGET_DIR"
  uv run --project "$CRUCIBLE_DIR" crucible implement \
    --spec "$SPEC_DIR" --slice <slice-id>
)
```

The last option already creates an Implementation attempt. Inspect its result
before deciding whether a separate handoff record is appropriate.

## Preview first

```bash
uv run --project "$CRUCIBLE_DIR" crucible handoff \
  --spec "$SPEC_DIR" \
  --slice <slice-id> \
  --cwd "$TARGET_DIR" \
  --dry-run
```

Dry-run performs recovery and scoped capture and creates a normal diagnostic
run envelope, but it does not publish Implementation/artifact/Spec records.
Show the returned JSON fields for files, diff bytes, verification status, and
`out_of_scope_changes`.

Out-of-scope changes are computed once against the union of all selected
directives. They are reported and excluded, not automatically fatal. Stop to
decide whether they belong to another slice or the accepted plan must change.

An empty scoped diff fails unless the operator deliberately supplies
`--allow-empty`, which records `soft_fail_no_changes`.

## Record

```bash
uv run --project "$CRUCIBLE_DIR" crucible handoff \
  --spec "$SPEC_DIR" \
  --slice <slice-id> \
  --cwd "$TARGET_DIR" \
  --operator "<operator-or-agent>" \
  --note "<why this work was completed externally>"
```

Build/verify runs by default when the diff is nonempty. `--skip-verify` is an
explicit evidence waiver. A `runner_missing` outcome logs a warning but still
records the handoff without verification; report that limitation rather than
calling it verified. Already skip-worthy, identity-matching slices are skipped
unless `--redo` is supplied.

Attribution uses `execution.recipe="handoff"`, `backend=None`, and zero model
cost. The operator/note text is retained in the transcript artifact.

## Chain gates and land

To run both child gates before landing:

```bash
uv run --project "$CRUCIBLE_DIR" crucible handoff \
  --spec "$SPEC_DIR" \
  --slice <slice-id> \
  --cwd "$TARGET_DIR" \
  --operator "<operator-or-agent>" \
  --note "<reason>" \
  --with-breach \
  --with-review \
  --review-rubric diff_scope \
  --budget-usd <per-child-gate-usd> \
  --land
```

Handoff first publishes a provisional `pending_gates` Implementation, then
invokes breach/review as child workflows. It publishes a second terminal
Implementation as `success`/`soft_fail_no_changes` or `hard_fail_gate`.
`--land` is suppressed when any selected target or gate fails.

`--budget-usd` is forwarded separately to each chained child invocation; it
is not one aggregate cap across all slices and both gates. Multiply the
potential spend within the user's authorized budget. `--backend-config` is likewise
forwarded to each child.

Gates may instead be run manually against the recorded `diff_ref` using
`crucible breach --diff-hash ... --cwd "$TARGET_DIR"` and
`crucible review --artifact-hash ... --slice-id ...`. A later handoff
invocation with `--land` can stamp the lifecycle event once the operator has
reviewed the complete evidence.

## Result and failure handling

- Exit 0: dry-run or all selected records/gates completed.
- Exit 1: at least one selected target or chained gate failed.
- Exit 2: usage/setup failure.

The command prints its summary JSON and also retains it in the run envelope.
Report provisional and terminal Implementation refs separately, the final
classification, excluded changes, verification state, child-gate results,
landed status, Spec version, and run/child run paths.

If another writer advances the Spec, transactional publication may fail with
`ConcurrentSpecError`. Reload current authority and repeat the preview; do not
assume the previous capture can be attached unchanged.
