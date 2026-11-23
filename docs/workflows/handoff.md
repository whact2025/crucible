<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `handoff` — Record interactive implementation

Handoff captures work completed outside the autonomous pipeline against an
accepted slice, tier, or Spec. Preview exactly what will be captured:

```bash
uv run crucible handoff --spec /path/to/corpus --slice slice-id \
  --cwd /path/to/repository --dry-run
```

Then execute the authorized capture with build/verify and any requested gates:

```bash
uv run crucible handoff --spec /path/to/corpus --slice slice-id \
  --cwd /path/to/repository --with-breach --with-review
```

The captured diff is directive-scoped. Verification is enabled by default.
`--operator` and `--note` attribute the work; `--land` records landing only when
requested targets and gates succeed. Empty captures fail unless `--allow-empty`
explicitly permits a no-change record. A skipped verification weakens evidence
and remains visible. Chained gate budgets apply independently to each child,
so account for their total before execution.

The summary reports each Implementation and its `diff_ref`; subsequent Breach
or Review can consume that exact artifact. Failed targets remain pending.
Scope capture, journal recovery, and cleanup must complete before publication.
A working-tree delta Git cannot capture ends the run as `setup_error` before
anything is recorded for that target. The out-of-scope report captures the
whole tree, so an uncapturable entry outside the slice scope also stops the
run. Targets recorded earlier in the same run keep their records; follow-on
gates and landing do not run. The `--land` re-check differs: a recorded
candidate Git can no longer capture blocks landing (exit 1), and its target
row names the capture error. A scoped capture that lists a directory (an
untracked nested repository) fails the target, because the diff cannot carry
its content.

## Options

`uv run crucible handoff --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--all` | hand-complete every pending slice in the spec |
| `--allow-empty` | record an empty-diff target as soft_fail_no_changes instead of failing |
| `--backend-config` | Explicit closed routing YAML override; roles and discovery are defined by the workflow. |
| `--budget-usd` | USD ceiling; consult this workflow’s per-call or cumulative budget semantics. |
| `--conditional-absence-decision-slice` | accepted dependency slice that owns the selector artifact |
| `--conditional-absence-equals` | expected selector value encoded as a JSON scalar |
| `--conditional-absence-json-pointer` | strict RFC 6901 pointer selecting the deciding scalar |
| `--conditional-absence-selected-slice` | accepted sibling slice selected by the decision |
| `--conditional-absence-selector` | repository-relative JSON decision artifact proving this slice is unselected |
| `--cwd` | repo root the work was done in (default: process cwd) |
| `--dry-run` | preview the capture per slice; write nothing to the corpus |
| `--land` | stamp LifecycleEvent(kind='landed') when all targets complete |
| `--note` | free-text completion note |
| `--operator` | who/what produced the work (recorded in the transcript note) |
| `--redo` | re-record slices that already have a completed Implementation |
| `--review-rubric` | rubric for --with-review (default: diff_scope) |
| `--skip-verify` | skip the build/verify gate (record capture only) |
| `--slice` | hand-complete one slice id |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |
| `--tier` | hand-complete every slice in one tier id |
| `--with-breach` | after recording, run the breach gate (Breaker→Defender→Judge) on each recorded diff — LLM-backed; uses --budget-usd/--backend-config |
| `--with-review` | after recording, run the review gate on each recorded diff — LLM-backed; rubric via --review-rubric |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/handoff.py](../../workflows/handoff.py).

## Exit codes

| rc | Meaning |
|----|---------|
| 0  | all targets recorded (or previewed under `--dry-run`) |
| 1  | a target failed — empty diff without `--allow-empty`, a directory entry in the scoped capture, verification failure, or optional-gate/accounting failure |
| 2  | missing Spec, unknown/missing target, Spec without a `tier_dag`, mutation-recovery setup failure, or a working-tree delta Git could not capture before recording |
