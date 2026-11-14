<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Crucible design axioms

These invariants define the solution-building incubator. The
[architecture](docs/ARCHITECTURE.md) explains their composition; the
[workflow references](docs/workflows/README.md) describe operator behavior.

## Layering and interface

**L1 — Assay owns generic contracts; Crucible owns composition.** Typed
records, canonical serialization, Corpus, and generic primitives belong to
Assay. Suite policy, role routing, transports, tools, and workflows belong to
Crucible. The validation rule resolver is an explicit integration seam.

**L2 — Workflows are bounded commands.** Each public workflow module exposes
`main(argv) -> int`. The filesystem dispatcher discovers the public modules;
private helpers start with `_`. `--help` is the option authority, and every
current command has a matching reference page.

**L3 — The product scope is explicit.** SWE-bench Verified and DeepSWE are the
benchmark suites. Reusable goal, plan, implement, breach/review, orchestration,
verification, and recovery code supports those suites and ordinary solutions.
Optional smoke metrics require explicit runner and direction configuration.

## Durable authority

**A1 — Manifests carry execution authority.** `spec.yaml` is authoring input.
Promotion publishes a canonical version. A build consumes the latest accepted
DAG with current source, review, validation, and plan-contract provenance.
Rejected or stale candidates never acquire execution authority from prose.

**A2 — Records and transitions publish transactionally.** A new Spec manifest,
its attached records, and lifecycle evidence become visible together. A file in
a run directory is not automatically attached evidence.

**A3 — Completion is typed and bound.** Runs own manifests, progressive
summaries, immutable terminal results, and typed outcomes. Parent operations
bind child outcomes through sagas. Timestamps, last log lines, and directory
existence never establish completion. Route materialization and exact journal
recovery follow their documented nonstandard run-envelope contracts.

**A4 — Interrupted owners and settled writers are distinct.** A dead process
is diagnostic. Every armed writer generation must have a durable, exact
quiescence receipt before capture, restore, commit, or cleanup. Missing,
malformed, or partial settlement stays blocking.

## Routes and tool authority

**B1 — Identity is explicit.** Adapter, model, endpoint, SDK/runtime, feature,
prompt, schema, policy, ToolSet, session, and pricing identities remain
attributable. No automatic provider, model, endpoint, or SDK substitution.

**B2 — Routing is closed and frozen.** Unknown schema fields, roles, backend
names, and unsupported feature requests fail before transport. A composite
benchmark run shares one resolved route snapshot.

**B3 — Roles have capability ceilings.** Text, read-only, and workspace-write
are distinct. Tool attachment cannot escalate the selected role. Confined
backend processes suppress ambient tools, hooks, plugins, instructions,
settings, and native executors. Backend-private MCP is scoped to one attached
ToolSet and invocation identity.

**B4 — Limits and accounting are truthful.** Declared enforcement strength
matches the concrete adapter. Every routed model has explicit pricing.
Unknown usage is not zero cost. A diagnostic interval covers only its admitted
uncertainty; it does not establish exact cost or promotable evidence.

## Review and benchmark validity

**V1 — Typed decisions control approval.** Bundled rubrics produce
`assay.review.v2` and satisfy the manifest's evidence, criterion, completeness,
and target-closure contract. Only `decision == "pass"` approves. Parse failure,
missing evidence, or a non-pass decision cannot become success.

**V2 — Generated proposals are not evidence.** Retain original source artifacts
and executed observations independently from model reasoning, prior findings,
replacement suggestions, and declarative attacks. Evidence quotes resolve to
allowed, hash-bound source material.

**V3 — Deterministic rules require their declared context.** Pure rules accept
one Spec; context-aware rules require a RuleContext. Setup cannot silently omit
checks because context is unavailable.

**V4 — Hidden truth stays outside authoring.** Models see public source and
development checks. Deterministic suite code owns hidden verification. Capture
exact candidate bytes and replay them independently. Cascade progress,
diagnostic success, and a plausible diff never substitute for verifier truth.

## Mutation and concurrency

**M1 — Directives define mutation scope.** Canonical paths and mutation regions
bound writes. Git metadata, hooks, symlink escape, and out-of-scope files remain
outside authoring authority. Model text cannot expand it.

**M2 — Transactions end committed or restored.** Journals retain original
bytes, modes, links, index state, and lineage. Recovery must prove exact state
and settled writer authority before applying changes.

**M3 — Parallel writers use isolated roots.** One shared checkout serializes
mutation. Read-only fan-out gives each child an independent backend and source
context and joins all work. Isolated-writer lanes own separate checkouts,
journals, budgets, and cancellation, then integrate in deterministic order.
The owning Tier performs the final landing transaction.

**M4 — Workflow code executes gates.** Models author source and proposals;
deterministic code owns smoke/build/visible tests, attack execution, pristine
replay, and hidden verification. Runner START, cancellation, and terminal
attestations bind one launch and writer epoch.

**M5 — Untrusted growth is bounded.** Storage, process, timeout, cancellation,
and cleanup contracts fail closed when their required enforcement is missing.
A terminal result cannot hide continuing child mutation.

## Current-product discipline

**G1 — One implementation owns each capability.** Update callers and tests
with the implementation. Delete replaced paths, one-time migration helpers,
and documentation for removed features together.

**G2 — Documentation describes the current product.** Scope, examples,
workflow inventory, links, route defaults, and rubric counts agree with source.
Historical proposals and delivery narratives are not part of the product docs.

**G3 — Overrides remain visible.** Skipped gates and diagnostic admissions
retain their limitations in arguments, run identity, and terminal reporting.
They do not create alternate promotion or solved-task claims.
