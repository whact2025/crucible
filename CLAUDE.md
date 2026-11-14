<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Crucible operator guide

Crucible builds and evaluates solutions through a reusable goal, plan,
implementation, breach, and review workflow. SWE-bench Verified and DeepSWE
provide its two benchmark adapters.

Read [README.md](README.md) for installation, [DESIGN_AXIOMS.md](DESIGN_AXIOMS.md)
for invariants, and the selected [workflow reference](docs/workflows/README.md)
for command arguments and terminal semantics.

## Execute within scope

- Resolve the intended Corpus and target repository before mutation.
- Preserve unrelated working-tree changes. Use isolated roots for concurrent
  writers and keep Crucible and Assay changes in separate commits.
- Promote `spec.yaml` before execution; the latest version manifest carries
  authority. An authored goal or rejected plan does not authorize a build.
- Use Architect for a complete solution DAG, Tier for one tier, Implement for
  one attempt, and Handoff for work completed outside the pipeline.
- Run configured build/verify and adversarial gates before claiming accepted
  completion. Only a typed review `decision == "pass"` approves an artifact.
- Keep paid calls, Docker evaluation, hidden tests, and external publication
  within the user's explicit task authorization.

## Protect benchmark validity

Each task receives an isolated Corpus, checkout, budget, and container. Models
see public source and development checks. Deterministic workflow code owns
hidden verification. Capture candidate bytes once, replay them independently,
and report the suite verifier's result separately from cascade progress.

Freeze exact backend routing for the cascade. Do not fall back between models,
providers, SDKs, endpoints, or feature profiles. Secrets stay in the environment;
never print, hash, or persist credentials as provenance.

## Record and report evidence

Read `outcome.json`, `result.json`, `summary.json`, attached record refs, and
parent/child saga bindings. A run directory, log tail, or generated narrative
is not completion. Report failure, interruption, incomplete accounting, skipped
gates, and diagnostic status explicitly.

Record interactive work with `crucible handoff --dry-run`, then the authorized
capture and gates. Recover interrupted mutations with Reconcile or exact
Journal Recovery, using the retained state required by their contracts.

## Contributor checks

Use `rg` for discovery, focused tests before the complete non-live suite,
Ruff, documentation checks, and `git diff --check`. Repository skills live in
[.claude/skills](.claude/skills/) and [.codex/skills](.codex/skills/).
