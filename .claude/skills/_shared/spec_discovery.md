<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Spec discovery

The operator skills accept only Specs at this exact depth:

```text
<workspace>/.crucible/specs/<target>/<spec>/
```

The direct Crucible CLI can operate on another Corpus-shaped directory, but
the skills and state helper enforce the
workspace layout.

## Resolve a name or path

Resolve the physical skill checkout as `CRUCIBLE_DIR` using
[preflight.md](preflight.md). Source the implementation once after setting
`WORKSPACE_ROOT` for the user's target workspace:

```bash
source "$CRUCIBLE_DIR/.claude/skills/_shared/bin/spec_discovery.sh"
```

It provides:

- `resolve_spec <value> [workspace]`
- `list_all_specs [workspace]`
- `is_canonical_spec_path <path> [workspace]`

`resolve_spec` writes only a successful absolute path to stdout. Candidate
lists and errors go to stderr. Its statuses are:

| Status | Meaning | Skill action |
|---:|---|---|
| 0 | One exact basename match or an existing canonical explicit path | Assert canonical depth and continue. |
| 1 | No match, nonexistent path, or explicit path outside the canonical tree | Surface stderr and stop; authoring may offer to create a new Spec. |
| 2 | Multiple exact/fuzzy matches, or exactly one fuzzy substring match | Surface every candidate and ask the user to choose. |
| 3 | Workspace/spec-root setup failure | Surface the setup error and stop. |

The two passes are exact case-sensitive basename matching followed by
case-insensitive basename substring matching. A single substring result still
returns 2 because fuzzy intent requires confirmation. Never choose by recency,
alphabetical order, target, or apparent similarity.

Explicit absolute paths and paths containing `/` are resolved with `realpath`
and accepted only when they are existing directories exactly two levels below
the canonical root. The canonical predicate rejects the root itself, target
directories, deeper children such as `runs/`, and symlinks that resolve
outside the tree.

## Canonical calling pattern

```bash
SPEC_DIR="$(resolve_spec "$SPEC_NAME_OR_PATH" "$WORKSPACE_ROOT")"
status=$?
case "$status" in
  0) is_canonical_spec_path "$SPEC_DIR" "$WORKSPACE_ROOT" || exit 3 ;;
  1) exit 1 ;;
  2) exit 2 ;;
  3) exit 3 ;;
esac
```

Do not discard resolver stderr: that is where the user-visible candidate list
is printed.

## Enumerate Specs

`list_all_specs "$WORKSPACE_ROOT"` sorts paths and groups them by target:

```text
== assay ==
  <workspace>/.crucible/specs/assay/goal_01_assay_validate_vertical

== swebench ==
  <workspace>/.crucible/specs/swebench/verified
```

The target inventory is derived from live directories; there is no target
allowlist.

## Source-root derivation

The parent directory name of a canonical Spec is reported as `target` by
`spec_state.py`. It is a routing convention, not proof that
`<workspace>/<target>` exists. The shared preflight uses the latest promoted
`project_context.submodule_path`; a null/empty path binds the workspace root.
It never selects a child directory merely because its name matches the target
label. Mutating skills must compare that binding with the intended source root.

## Author a new Spec

The resolver does not create directories. Follow
[authoring.md](authoring.md) before discovery for a new solution, including an
empty folder with no Git repository or canonical root. The agent, rather than
the user, performs these authoring steps:

1. create `<workspace>/.crucible/specs/<target>/<spec>/`;
2. write a strict Assay schema-version-1 `spec.yaml`;
3. promote it with
   the helper under `$CRUCIBLE_DIR/.claude/skills/_shared/bin/`; and
4. resolve the new exact name again before dispatch.

Creating the directory alone does not create a workflow-readable Spec;
workflows read the promoted manifest under `versions/`.

## Promote an edit to an existing Spec

Editing `spec.yaml` does not alter canonical execution state. Promote the edit
before dispatch. Promotion rebases the authored fields onto the latest locked
Corpus lineage and preserves append-only evidence refs. It clears the
generated TierDAG by default:

```bash
uv run --project "$CRUCIBLE_DIR" python \
  "$CRUCIBLE_DIR/.claude/skills/_shared/bin/promote_spec_yaml.py" "$SPEC_DIR"
```

For an ordinary solution, if the edit is intentionally compatible with the
exact already reviewed DAG, retain it only by digest:

```bash
uv run --project "$CRUCIBLE_DIR" python \
  "$CRUCIBLE_DIR/.claude/skills/_shared/bin/promote_spec_yaml.py" "$SPEC_DIR" \
  --retain-plan-digest sha256:<64-hex-digits>
```

This promotion is not plan acceptance. Run Architect separately with the same
`--reaccept-plan-digest`; it reruns deterministic validation without another
paid planning debate. Reacceptance never reads authored changes directly and
fails before backend construction when `spec.yaml` still differs from the
canonical manifest.

Benchmark attempts use normal plan reuse when their current planning stamp
matches; typed benchmark policy rejects exact digest reacceptance. A changed
planner count, planning allowance, execution fingerprint, or source recipe
requires fresh planning during an authorized suite attempt. DeepSWE's
`--replan` or programmatic Architect's `replan=True` requests a fresh plan even
when the stamp matches.
