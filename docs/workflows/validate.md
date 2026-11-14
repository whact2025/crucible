<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `validate` — Validate an accepted or candidate Spec

Validate runs the selected deterministic rules against the latest Spec. Rules
check scope, created paths, DAG structure, citations, declared verification, and
intent coverage.

```bash
uv run crucible validate --spec /path/to/corpus \
  --workspace-root /path/to/context-root
```

Pure rules accept one Spec. Context-aware rules also require a RuleContext;
selecting them without usable context fails setup before checks execute. When
the Spec declares `project_context.submodule_path`, pass its outer workspace
so the path is composed once. Exceptions in rule execution become blocking
rule-error findings; malformed rule wiring remains an error.

The summary and attached validation verdict retain findings and severity.
Critical or High findings block the gate. Validation makes no model calls and
cannot establish semantic plan approval by itself.

## Options

`uv run crucible validate --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--rules` | Comma-separated subset of rule_ids. Default: all bundled rules. |
| `--ruleset-name` | Name embedded in the Verdict.rubric field. |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |
| `--workspace-root` | Workspace root for context-aware rules (e.g. cited_path_exists). When set, context-aware rules fire against the local working tree. Default: the spec's grandparent dir (<workspace>/.crucible/specs/<target>/<spec>/ → <workspace>). |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/validate.py](../../workflows/validate.py).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All findings below Critical/High severity floor. |
| `1` | At least one Critical/High finding. |
| `2` | Setup failure, including unavailable context for a selected context-aware rule. |
