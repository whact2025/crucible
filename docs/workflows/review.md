<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `review` — Review an explicit artifact

Review evaluates one file or Corpus artifact through one executable rubric
contract. The artifact, Spec intent, and explicitly allowed source evidence
remain separate from generated findings and proposals.

```bash
uv run crucible review --spec /path/to/corpus --rubric diff_scope \
  --artifact /path/to/candidate.patch --budget-usd 3
```

`--artifact-hash` selects an exact retained artifact instead of a file. The
rubric manifest governs context, evidence sources, completeness, criteria,
finding severity, decisions, and target closure. Invalid contracts or truncated
required evidence cannot produce approval. Bounded response repair fixes only
the closed response protocol; it does not invent an extra semantic review round.

The result is a persisted typed Verdict and terminal run evidence. Its
`decision` controls flow; the derived `passed` field is only a convenience.
See [rubric contracts](../../rubrics/README.md) for prompt authoring and linting.

## Options

`uv run crucible review --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--artifact` | Path to a file whose contents will be reviewed. |
| `--artifact-hash` | `sha256:<hex>` corpus key for an existing artifact. |
| `--backend-config` | Explicit closed routing YAML override; roles and discovery are defined by the workflow. |
| `--backend-limit-deadline-seconds` | Finite positive monotonic deadline for each provider invocation; pair with limit mode. |
| `--backend-limit-mode` | Explicit strict or diagnostic provider-limit admission; pair with the invocation deadline. |
| `--budget-usd` | USD ceiling; consult this workflow’s per-call or cumulative budget semantics. |
| `--diagnostic-cost-policy` | Explicit diagnostic-only conservative cost interval policy; results remain non-promotable. |
| `--response-retry-budget` | Response retry budget. See parser help for accepted values and defaults. |
| `--rubric` | Name of a bundled rubric (under crucible/rubrics/). |
| `--slice-id` | Slice whose directive scopes a diff_scope review. Required for path artifacts; hash artifacts may resolve it from an attached Implementation. |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/review.py](../../workflows/review.py).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | `Verdict.decision == "pass"`. |
| `1` | `Verdict.decision == "revise"`; pre-review rubric/artifact/config errors that raise `SystemExit` or another uncaught setup exception also terminate nonzero without a Verdict. |
| `2` | `Verdict.decision == "reject"`, or the Spec is missing. |
| `3` | `Verdict.decision == "inconclusive"`, including incomplete coverage or budget exhaustion. |

`passed` remains serialized as a derived convenience field, but it is not the
workflow control contract; the typed `decision` determines status and exit.
