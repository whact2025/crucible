<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Crucible skill preflight

Every stateful stage skill resolves one canonical Spec, reads the helper's
non-mutating state report, derives the correct source/context roots, and checks
the typed prerequisite described in [lifecycle.md](lifecycle.md) before
dispatch.

## Inputs

- `SPEC_NAME_OR_PATH`: exact short name or canonical path supplied by the user.
- `REQUESTED_STAGE`: one of the values accepted by `spec_state.py`:
  `design`, `build`, `benchmark`, or `recovery`.

## 0. Locate Crucible and choose the authoring path

Resolve the physical directory of the loaded `SKILL.md` before following its
shared references. The installer symlinks individual skill folders; a sibling
`_shared` directory at the installation destination is not required.

```bash
SKILL_DIR="$(cd -P "<directory containing the loaded SKILL.md>" && pwd -P)"
CRUCIBLE_DIR="$(cd "$SKILL_DIR/../../.." && pwd -P)"
```

`CRUCIBLE_DIR` is the installed Crucible checkout. It is independent of the
user's working folder, `WORKSPACE_ROOT`, and `TARGET_DIR`. Use it for every
`uv run --project` and shared helper path. Do not assume a `crucible/` child
exists in the target workspace.

For a new solution, follow [authoring.md](authoring.md) **before** Spec
discovery. That path handles an empty/non-Git folder and a missing
`.crucible/specs` tree, writes and promotes the input, and supplies the exact
`SPEC_DIR`, `TARGET_DIR`, and `WORKSPACE_ROOT`. A creation request already
authorizes those local authoring steps. Other stages must not invent a Spec
to conceal a missing prerequisite.

## 1. Resolve the workspace and Spec

Use the working repository as the initial `TARGET_DIR` unless the user named
another target. An explicit canonical Spec path identifies its workspace;
resolve that workspace before discovery if it differs from the working one.
`git rev-parse --show-superproject-working-tree` succeeds with empty output
outside a superproject, so test the value before falling back:

```bash
WORKSPACE_ROOT="$(git -C "$TARGET_DIR" rev-parse --show-superproject-working-tree 2>/dev/null || true)"
if [ -z "$WORKSPACE_ROOT" ]; then
  WORKSPACE_ROOT="$(git -C "$TARGET_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
fi
[ -n "$WORKSPACE_ROOT" ] || { echo "ERROR: no Git workspace root" >&2; exit 3; }

source "$CRUCIBLE_DIR/.claude/skills/_shared/bin/spec_discovery.sh"

SPEC_DIR="$(resolve_spec "$SPEC_NAME_OR_PATH" "$WORKSPACE_ROOT")"
resolve_status=$?
case "$resolve_status" in
  0) ;;
  1) echo "Stop: no canonical Spec matched '$SPEC_NAME_OR_PATH'." >&2; exit 1 ;;
  2) echo "Stop: confirm one of the candidates printed by the resolver." >&2; exit 2 ;;
  3) echo "Stop: Spec discovery setup failed." >&2; exit 3 ;;
  *) echo "Stop: unexpected resolver status $resolve_status." >&2; exit 3 ;;
esac

is_canonical_spec_path "$SPEC_DIR" "$WORKSPACE_ROOT" || {
  echo "ERROR: resolver returned a non-canonical path: $SPEC_DIR" >&2
  exit 3
}
```

The resolver writes candidates/errors to stderr and only writes a successful
path to stdout. On status 2, surface the candidates verbatim and require the
user to choose—even when there is only one fuzzy substring match.

## 2. Read the non-mutating state report

Parse only the helper's known lowercase keys; never `eval` user-controlled
output:

```bash
STATE_REPORT="$(
  uv run --project "$CRUCIBLE_DIR" python \
    "$CRUCIBLE_DIR/.claude/skills/_shared/bin/spec_state.py" \
    "$SPEC_DIR" --stage "$REQUESTED_STAGE"
)" || { echo "ERROR: Spec state inspection failed" >&2; exit 3; }

while IFS='=' read -r key value; do
  case "$key" in
    spec_dir|spec_version|manifest_path|workspace_root|target|project|submodule_path|\
    runs_total|runs_design|runs_build|runs_benchmark|runs_recovery|\
    last_run|warn_out_of_order)
      printf -v "$key" '%s' "$value"
      ;;
  esac
done <<< "$STATE_REPORT"
```

The helper scans filenames and selected YAML scalars directly; it does not
construct `Corpus(spec_dir)` and therefore does not create Corpus directories.
It emits:

| Variable | Meaning |
|---|---|
| `spec_dir` | Absolute resolved Spec path. |
| `spec_version` | Highest manifest version, or `none`. |
| `manifest_path` | Relative highest-version manifest path, or `none`. |
| `workspace_root` | Git-derived workspace, or `none`. |
| `target` | Path segment under `.crucible/specs/`. |
| `project`, `submodule_path` | Best-effort scalar reads from `spec.yaml`. |
| `runs_total`, `last_run` | Directory-count/navigation telemetry. |
| `runs_<stage>` | Counts based on the helper's internal stage map. |
| `warn_out_of_order` | Coarse run-count hint for the requested stage. |

Run counts do not prove acceptance or complete verifier evidence. Inspect the
latest Spec and typed terminal outcomes before advancing the solution.

## 3. Derive source and validation roots

Different CLIs interpret their path arguments differently:

- `architect`, `tier`, `handoff`, `breach` take the target repository
  root directly.
- `implement` has no path flag and uses the process current directory.
- `plan --cwd` and `validate --workspace-root` construct rule context with the
  Spec's `project_context.submodule_path`. When that field is set, pass the
  outer workspace to avoid applying the submodule twice.

Derive both roots and fail if a declared path escapes or is missing:

```bash
if [ -n "$submodule_path" ]; then
  TARGET_CANDIDATE="$WORKSPACE_ROOT/$submodule_path"
  PLAN_CONTEXT_ROOT="$WORKSPACE_ROOT"
else
  TARGET_CANDIDATE="$WORKSPACE_ROOT"
  PLAN_CONTEXT_ROOT="$WORKSPACE_ROOT"
fi

TARGET_DIR="$(realpath "$TARGET_CANDIDATE" 2>/dev/null)" || {
  echo "ERROR: target repository does not exist: $TARGET_CANDIDATE" >&2
  exit 3
}
case "$TARGET_DIR" in
  "$WORKSPACE_ROOT"|"$WORKSPACE_ROOT"/*) ;;
  *) echo "ERROR: target escapes workspace: $TARGET_DIR" >&2; exit 3 ;;
esac
```

Read `submodule_path` from the latest promoted Spec, not only the helper's
best-effort YAML scan. A null/empty path binds the workspace root. The Corpus
`<target>` label never selects a same-named child directory. Compare the
resolved root with the user's intended working folder before mutation; resolve
an actual disagreement instead of guessing. Keep Corpus state outside the
candidate diff as described in [authoring.md](authoring.md).

## 4. Check typed prerequisites

Read [lifecycle.md](lifecycle.md), the latest manifest, and relevant terminal
`outcome.json`/attached records. Do not use `runs_build > 0` as proof of an
accepted build or solved benchmark task.

If required evidence is missing, diagnose it before dispatch. Only
`handoff` exposes general attribution fields (`--operator`, `--note`); do not
promise that another workflow persists a generic override reason when its CLI
has no such option.

## 5. Dispatch without changing directory by default

Use the selected Crucible project while preserving explicit workflow paths:

```bash
uv run --project "$CRUCIBLE_DIR" crucible <workflow> \
  --spec "$SPEC_DIR" \
  [workflow-specific arguments]
```

Pass `$TARGET_DIR` to `architect --cwd`, `tier --cwd`, `handoff --cwd`,
`breach --cwd`. Pass `$PLAN_CONTEXT_ROOT` to
`plan --cwd` and `validate --workspace-root`.

`implement` is the exception because its CLI has no `--cwd`:

```bash
(
  cd "$TARGET_DIR"
  uv run --project "$CRUCIBLE_DIR" crucible implement \
    --spec "$SPEC_DIR" --slice <slice-id>
)
```

## 6. Identify the run created by this invocation

Concurrent and composite workflows make “newest run directory” unsafe: a
directory-set difference can include another process's run or this command's
children. Set `EXPECTED_WORKFLOW` to the top-level command, snapshot manifests
before dispatch, then retain only new manifests whose workflow matches and
whose `parent_run_id` is null:

```bash
RUN_SNAPSHOT="$(mktemp)"
find "$SPEC_DIR/runs" -mindepth 2 -maxdepth 2 -name run_manifest.json \
  -type f -print 2>/dev/null | sort > "$RUN_SNAPSHOT"

# Run the workflow here.

mapfile -t MATCHING_RUNS < <(
  uv run --project "$CRUCIBLE_DIR" python - \
    "$RUN_SNAPSHOT" "$SPEC_DIR/runs" "$EXPECTED_WORKFLOW" <<'PY'
import json
import sys
from pathlib import Path

before = set(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines())
for manifest_path in sorted(Path(sys.argv[2]).glob("*/run_manifest.json")):
    if str(manifest_path) in before:
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("workflow") == sys.argv[3] and manifest.get("parent_run_id") is None:
        print(manifest_path.parent)
PY
)
rm -f "$RUN_SNAPSHOT"

if [ "${#MATCHING_RUNS[@]}" -ne 1 ]; then
  printf 'ERROR: expected one new top-level %s run, found %d\n' \
    "$EXPECTED_WORKFLOW" "${#MATCHING_RUNS[@]}" >&2
  printf 'candidate: %s\n' "${MATCHING_RUNS[@]}" >&2
  exit 3
fi
RUN_DIR="${MATCHING_RUNS[0]}"
```

This rejects child runs and fails on multiple matching top-level runs. It still
cannot prove ownership if another process launches the same workflow against
the same Spec during this exact window; when such concurrency is known, report
the ambiguity instead of attributing either run. Inspect `$RUN_DIR`'s
`outcome.json`, `result.json`, and `summary.json`. Events, SDK dumps, build logs,
and benchmark logs are conditional. The agent's reply is a summary of durable
evidence, not the authority itself.
