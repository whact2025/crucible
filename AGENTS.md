<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Crucible development guidance

Crucible is a solution-building incubator with SWE-bench Verified and DeepSWE
benchmark adapters. Assay owns typed records, canonical serialization, Corpus
storage, and generic primitives. Crucible owns workflow composition, routing,
backend adapters, scoped tools, rules, rubrics, and suite policy.

This guidance is for trusted interactive repository work. Confined backend
adapters and hidden evaluators receive authority through typed workflow inputs
and suppress ambient project/user instructions.

- Read [DESIGN_AXIOMS.md](DESIGN_AXIOMS.md) and the relevant
  [workflow reference](docs/workflows/README.md) before changing behavior.
- Preserve unrelated work. Give independent writers isolated checkouts, then
  integrate serially. Commit Assay and Crucible changes independently.
- Use `rg`, focused tests, then required non-live checks. Finish with Ruff,
  documentation validation, and `git diff --check`.
- Keep route, model, endpoint, SDK/runtime, feature, prompt, policy, ToolSet,
  schema, and session identities explicit. Never add silent fallback.
- Keep hidden tests, secrets, native tools, and ambient configuration outside
  authoring contexts. Never print sourced credentials or tokens.
- Require complete terminal, cost, trace, and settlement evidence for a
  promotable result. Diagnostic results remain diagnostic.
- Run paid live calls, hidden evaluators, artifact publication, and external
  mutations only when the task authorizes their exact boundary.
- Maintain one current implementation and its tests. Delete replaced paths
  and documentation in the same change; do not ship compatibility shims or
  migration helpers.

## Review priorities

Flag authority expansion, path/symlink escape, post-timeout mutation, unjoined
child work, stale identity reuse, incomplete evidence accepted as success,
secret-bearing provenance, and model/provider fallback. Safety and lifecycle
fixes need a deterministic failure-path regression. Review configured parallel
execution for isolated writer roots, joined children, deterministic integration,
and complete cancellation settlement.
