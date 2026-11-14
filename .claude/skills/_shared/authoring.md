<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Author a solution from a conversation

Use this path when `crucible-design` creates a solution or revises its authored
goal. The agent owns file authoring, schema validation, and promotion. The user
supplies intent in natural language; starter code and hand-written YAML are
not prerequisites.

## Clarify the outcome

Inspect relevant existing source, visible tests, and project configuration when
available. Establish the desired behavior, scope, hard constraints, non-goals,
and observable completion criteria. Keep architecture and implementation
choices open unless the user has constrained them. Ask about consequential
ambiguities, then present the goal and wait for explicit user approval before
any planning. Model-based `goal` debate is optional and does not replace that
approval. Do not invoke paid workflows unless the task authorizes execution
and a route/budget boundary.

## Bind the folder and establish a baseline

Set `TARGET_DIR` to the intended working folder, independently of
`CRUCIBLE_DIR`. For an existing repository, preserve its working tree and index;
inspect its actual Git root and any superproject. Use that workspace for the
Corpus. Do not infer the source root from a Corpus target label.

For a new standalone project, create only the requested target folder if it
does not exist, initialize local Git there, and establish an empty baseline
commit if there is no `HEAD`. Use the configured Git identity; do not invent
one. `git commit --allow-empty --only` can establish an empty baseline without
committing pre-staged operator files. Never use blanket staging or commit
unrelated files. If the folder is already inside another repository, bind it
as an explicit relative source path in that workspace rather than initializing
or moving the parent repository. Report any existing changes that prevent a
clean implementation run.

Set `WORKSPACE_ROOT` to the outer workspace and choose an unambiguous local
`<target>/<solution>` label. For a standalone repository the workspace is
`TARGET_DIR` and `project_context.submodule_path` is null. For a target below
the workspace, use its actual confined relative path with `/` separators.
Show the resulting target and Corpus paths. Ask only when naming or path
choices would collide with another solution or conflict with user intent.

Create `SPEC_DIR` at
`$WORKSPACE_ROOT/.crucible/specs/<target>/<solution>/` before calling the
ordinary discovery helper. A missing canonical root is normal for first-time
authoring, not a reason to stop. Keep operator state out of the implementation
diff: reuse an existing ignore rule or add a local `/.crucible/` exclusion to
the workspace repository's Git `info/exclude` file without replacing its other
contents. Do not stage Corpus records as application changes.

## Write a strict executable input

Read the installed Assay `Spec.load_v0` schema and the applicable Crucible
language runner contract instead of guessing keys. Write `spec.yaml` yourself:

- `schema_version: 1` and factual `project_context` (actual project, repository
  URL or local `file://` URI when no remote exists, base commit, source path);
- approved intent, typed constraints, and observable `done_criteria`;
- real `runners.build` and `runners.verify` adapter entrypoints, with
  `runner_parameters.per_language` commands appropriate to the target; and
- only supported authored policy fields such as `plan_policy` when needed.

The strict loader rejects unknown/generated fields. Do not author a
`tier_dag`, version/history/evidence refs, raw `breach_policy`/`review_policy`,
or `runner_parameters.working_dir`. Workflow code supplies the actual target
directory. Language adapters live in
`common/runners/lang/{python,rust,cpp}.py`; the polyglot dispatcher is
`common/runners/polyglot.py`. An empty `runners: {}` is not a verified-build
configuration.

For existing code, derive build/test commands from the project and its visible
checks. For a new project, select real commands for the agreed stack and make
creation of the package/build/test configuration and behavior tests part of
the first executable slice. Include those paths in the planner's required
`creates_paths`. Do not run nonexistent project tests during authoring or
claim that planned checks have passed. Unsupported stacks need a concrete
runner choice before verified execution, not a no-op success command.

## Promote, inspect, and hand off

Promote the authored file using the canonical helper:

```bash
uv run --project "$CRUCIBLE_DIR" python \
  "$CRUCIBLE_DIR/.claude/skills/_shared/bin/promote_spec_yaml.py" "$SPEC_DIR"
```

Inspect the returned version/digest and latest manifest. An existing Spec edit
preserves append-only evidence and clears its old plan by default. Do not
retain a stale DAG to bypass planning after the goal changes. Follow
[spec_discovery.md](spec_discovery.md) for exact-digest retention when an
ordinary solution's unchanged plan is deliberately reaccepted.

Return to [preflight.md](preflight.md) with the exact canonical path and source
binding. Once the user has approved the goal, run the authorized planning
stage and report its actual acceptance, deterministic validation, evidence,
and cost. Leave application implementation to `crucible-build` unless the
user also authorized it.
