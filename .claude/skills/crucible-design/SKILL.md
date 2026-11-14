---
name: crucible-design
description: Develop a goal interactively from plain language, create its spec from existing code or an empty folder, and produce an accepted plan after goal approval. Also review or validate Crucible artifacts.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---
<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->


# /crucible-design

Own the specification files for the user. A plain-language request is sufficient;
never require the user to write YAML, run an editor, or supply starter code.

| Intent | Action |
|---|---|
| Develop a new solution or clarify its goal | Converse, inspect relevant code, author/promote the Spec, then request explicit goal approval |
| Produce an accepted plan from an approved goal | `plan` |
| Explicitly request adversarial goal refinement | `goal`, then request approval of the resulting goal |
| Review one explicit artifact with one rubric | `review` |
| Run deterministic Spec rules | `validate` |

## Develop the goal

Start with the user's desired outcome, affected users, constraints, and
observable completion criteria. For existing projects, inspect relevant source,
tests, and build configuration in the working folder. For an empty folder,
clarify the product and any consequential stack choice; leave scaffolding to
the implementation plan. Read [authoring.md](../_shared/authoring.md) for the
first-Spec procedure.

Ask focused questions only about choices that change the intended behavior or
scope. Reuse information and authorization already supplied. Distinguish hard
constraints from preferences and unresolved design choices. Do not choose an
architecture, invent detailed APIs, or enumerate implementation slices merely
to make the goal look complete. Keep behavioral examples concrete without
claiming they are executed tests or a complete acceptance suite.

Present a concise goal with scope, constraints, non-goals, and observable
completion criteria. **Wait for the user's explicit approval before starting
any planner**, including Architect's implicit planning. Silence is not approval.
If the user requests revisions, update the draft and ask approval of the revised
goal. Once approved, carry the authorized planning work forward without asking
again about routine implementation choices. Reopen approval only when the goal
materially changes. An explicit request to plan an unchanged, already approved
goal satisfies this prerequisite.

Execute the workflow sequence authorized by the user's task. Check and report
its durable results before advancing to the next dependent stage.

## Preflight

Locate the physical skill folder and `CRUCIBLE_DIR` as described in
[the shared preflight](../_shared/preflight.md), then use
`REQUESTED_STAGE=design`. Resolve fuzzy names only after user confirmation.
Compute both `TARGET_DIR` and `PLAN_CONTEXT_ROOT`.

All four workflow commands require a promoted Spec. For a new solution, follow
[authoring.md](../_shared/authoring.md) before ordinary discovery. Choose an
unambiguous local name when the user has not specified one, write the strict
input yourself, and promote it:

```bash
uv run --project "$CRUCIBLE_DIR" python \
  "$CRUCIBLE_DIR/.claude/skills/_shared/bin/promote_spec_yaml.py" \
  "$SPEC_DIR"
```

The YAML must satisfy Assay's strict schema. A directory or unpromoted YAML
draft is not enough. In particular, current `Spec.from_yaml` accepts
`plan_policy` but rejects Crucible's raw `breach_policy` and `review_policy`
blocks; do not put those blocks in a file that must be promoted.

## Goal refinement

`goal` runs bounded Red → Blue → deterministic validation → Judge rounds. It
stages revisions in memory and publishes an approved goal plus its evidence in
one Corpus transition. The defaults are three rounds and ten turns per review
invocation.

```bash
uv run --project "$CRUCIBLE_DIR" crucible goal \
  --spec "$SPEC_DIR" \
  --max-rounds 3 \
  --budget-usd <cumulative-usd>
```

The run writes `final_goal.yaml`, stores it as a content-addressed output, and
attaches the review/validation verdicts. Goal refinement is useful, not a
mandatory prerequisite imposed by `plan`. This skill still requires explicit
user approval of the goal before planning; a model Judge's approval does not
replace it. Keep goal debate optional so routine clarification does not overfit
the solution before planners explore alternatives.

## Planning

After explicit goal approval, use the default three independent planners with
one shared planning budget. Independent proposal generation uses separate
agent contexts and read-only source tools. Each proposal receives adversarial
Critic/Judge evaluation and deterministic checks. Comparison may select an
exact eligible proposal or synthesize a complete plan; a synthesis requires
fresh review and validation. A comparison recommendation is not acceptance.
Report the alternatives, comparison rationale, selection/synthesis, and final
acceptance evidence, including rejected or incomplete branches and their cost.

The default planner already uses `plan_critic`, the required `plan_judge`, the
Spec's `PlanPolicy` (or its defaults), and a post-plan deterministic validation
gate. Source inspection is the opt-in part.

```bash
uv run --project "$CRUCIBLE_DIR" crucible plan \
  --spec "$SPEC_DIR" \
  --cwd "$PLAN_CONTEXT_ROOT" \
  --planner-count 3 \
  --enable-source-tools \
  --critic-rubric plan_critic \
  --judge-rubric plan_judge \
  --budget-usd <cumulative-usd>
```

Implemented defaults:

- `--max-turns 5` per planner/Critic/Judge invocation;
- three independent proposals sharing one explicit positive total budget;
- validation enabled with all bundled rules, including context-aware rules;
- fixed-candidate review with no repair during competition; and
- source ToolSet disabled unless `--enable-source-tools` is present.

Use `--planner-count 1` only when the user requests a single planning track.
That track allows zero or one Planner-only deterministic correction through
`--max-validate-replan-attempts`; fixed-candidate review makes no Planner calls
and does not fan out. Parallel planning requires deterministic validation and
does not admit `--no-validate-gate` or candidate repair. A single-track
diagnostic override must report its skipped validation gate.

When `project_context.submodule_path` is nonempty, `PLAN_CONTEXT_ROOT` must be
the outer workspace because the workflow composes the submodule itself. An
empty path binds the workspace root; the Corpus target label is not a path.

Honor one total budget for the requested design sequence. `goal` and `plan`
are separate top-level commands with separate caps: deduct the actual prior
cost before launching a later command. Do not give each command the full
design budget or start planning without the required goal approval.

The benchmark runners also default to three planners, but partition at most
40% of a task's whole execution allowance for planning. That partition does not
apply to a standalone design budget. For a suite request, use the
[benchmark guide](../../../docs/BENCHMARKS.md); do not manually give each
proposal the task budget or replace the suite's checkpoint identity.

An accepted run commits the TierDAG, review refs, validation refs, and plan
contract provenance. Rejected or validation-blocked candidates do not remain
as executable `Spec.tier_dag` authority.

## Rubric review

List active rubrics from the manifest-backed API:

```bash
uv run --project "$CRUCIBLE_DIR" python -c \
  'from crucible.rubrics import available; print(*available(include_dormant=False), sep="\n")'
```

Review a file:

```bash
uv run --project "$CRUCIBLE_DIR" crucible review \
  --spec "$SPEC_DIR" \
  --rubric <rubric-name> \
  --artifact "$ARTIFACT" \
  [--slice-id <slice-id>] \
  [--budget-usd <per-call-usd>]
```

Or review an existing Corpus artifact:

```bash
uv run --project "$CRUCIBLE_DIR" crucible review \
  --spec "$SPEC_DIR" \
  --rubric <rubric-name> \
  --artifact-hash sha256:<hex> \
  [--slice-id <slice-id>]
```

Exactly one artifact selector is required. A path-based `diff_scope` review
requires `--slice-id`; a hash may infer its slice only when an attached
Implementation provides an unambiguous binding. Only `decision=pass` approves.

## Deterministic validation

List the registered rules:

```bash
uv run --project "$CRUCIBLE_DIR" python -c \
  'from crucible.rules import available; print(*available(), sep="\n")'
```

Run all rules against the same context root used for planning:

```bash
uv run --project "$CRUCIBLE_DIR" crucible validate \
  --spec "$SPEC_DIR" \
  --workspace-root "$PLAN_CONTEXT_ROOT"
```

Optional filtering uses a comma-separated `--rules` value; `--ruleset-name`
changes the name stored in the Verdict. There is no `--submodule-path` flag.
The workflow reads `project_context.submodule_path` from the latest Spec and
composes it with `--workspace-root`. If the context is wrong or missing,
context-aware rules are not silently skipped: selected context-aware rules
require a `RuleContext`.

## Report

For the run created by this invocation, report:

- run directory and `outcome.json` terminal class/detail;
- latest Spec version;
- decision and attached verdict/output refs;
- whether source tools and deterministic validation ran;
- any skipped gate or incomplete evidence; and
- actual cost for LLM workflows.

Recommended next actions are conditional: refine a non-pass goal, repair and
re-plan a blocked DAG, address review findings, or send an accepted/validated
plan to `/crucible-build`.
