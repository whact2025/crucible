<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Crucible public API

Crucible exposes bounded workflows and explicit composition helpers for
solution building and its SWE-bench/DeepSWE adapters. Assay owns the shared
types and primitives. Private workflow helpers are implementation details.

## Commands and reusable composition

`crucible <workflow> [args]` and `python -m crucible.workflows <workflow> [args]`
share the filesystem dispatcher. `crucible --list` lists commands. Each public
workflow exports `main(argv: list[str] | None) -> int`.

`run_plan`, `run_tier`, and `run_architect` are the keyword-based composition
entrypoints in their respective workflow modules. Programmatic callers supply
an explicit `cwd`; there is no process-current-directory fallback for fan-out
callers. Pass the same frozen routing, budgets, policy, parent operation, and
source identities throughout a composite run. `worktree=True` gives the
invocation an isolated Git checkout.

`plan` CLI generation and `run_architect(planner_count=3, ...)` use
three concurrent independent complete proposals by default. Supply an explicit
positive finite total `budget_usd` when planning is needed. The budget covers
all draft calls, adversarial evaluations, comparison, and any synthesis review.
`planner_count=1` selects the single-track refinement loop. SWE-bench and
DeepSWE also default to three planners, with an explicit count of one available
for diagnostics. Reusing a current accepted DAG does not launch a competition.
Ordinary solution workflows also support exact digest reacceptance without
planning calls; typed benchmark policy does not admit that override.

Suite adapters pass a frozen `planning_policy` to `run_architect` while keeping
`budget_usd` equal to the whole task allowance. Competition receives at most
40% of that allowance, bounded further by remaining funds and any development
ledger. The whole task allowance and configured reserves remain unchanged.
Count, share, execution fingerprint, and planning recipe bind the accepted terminal
and DAG. A matching stamp permits normal model-free plan reuse. A mismatched
retained plan triggers fresh planning, and `replan=True` explicitly requests
a fresh plan. Completed checkpoint conflicts retain the suite's explicit
rerun rules.

Each active benchmark task can make three concurrent draft requests even when
task/cell scheduling concurrency is one. Planner source-call and result-byte
limits span all children. A single-track override does not establish evidence
for the three-planner recipe.

The injected-backend `run_plan` entrypoint remains a single-candidate primitive.
It can generate with `planner_backend` or adjudicate `candidate_dag` without a
Planner call. Competition orchestration stages alternatives and owns final
selection; only the publication owner advances the canonical Spec. A synthesized
candidate receives fresh Critic/Judge review and deterministic validation.

Use the [workflow index](workflows/README.md) for command references and
source-linked signatures. The runtime's typed return and terminal evidence
carry outcome authority; a generated summary alone does not.

## Read-only and isolated-writer fan-out

[orchestration.py](../orchestration.py) exports the supported fan-out facade:

- `OrchestrationPolicy`, `ReadOnlyChildRequest`, `ReadOnlyChildOutcome`,
  `ReadOnlyFanoutResult`, and `run_read_only_fanout` support bounded independent
  source-reading children. Every child owns a backend, context, ToolSet, source
  authority, and cancellation state. Results are joined before consumption.
- `WriterLanePolicy`, `WriterLaneRequest`, `WriterLaneOutcome`,
  `WriterFanoutResult`, `IntegrationGateResult`, and
  `run_isolated_writer_fanout` support independent mutation lanes and validated
  patch integration. The scheduler does not itself land the owning solution.
- Cleanup, isolation, scheduling, and integration-conflict exceptions preserve
  failure detail. A failed cleanup cannot be reported as successful isolation.

Architect's opt-in read-only orchestration and Tier's programmatic
`writer_frontier` use these contracts. Writer requests bind exact directive
scope and a clean base. Each lane uses an independent checkout and journal;
integration follows declared order. Incompatible breach/review/smoke/worktree
or skipped-verification combinations reject before mutation. Consult the
[Tier reference](workflows/tier.md) and current dataclass signatures before
constructing a writer request.

## Backend routing

[backends/config.py](../backends/config.py) provides `RoleConfig`,
`BackendRouting`, `DEFAULT_ROUTING`, `load_routing`, `resolve_for_role`, and
routing fingerprints. The closed routing schema is
`crucible.backend-routing.v1`. Discovery uses explicit path, environment path,
Spec-local file, workspace file, then defaults. Unknown roles and fields fail.
`bind_routing_snapshot` carries one resolved route through nested workflows.

[backends/roles.py](../backends/roles.py) defines role identities and capability
ceilings. [backends/capabilities.py](../backends/capabilities.py) distinguishes
text, read-only, and workspace-write authority.
[backends/features.py](../backends/features.py) validates a concrete route's
feature requirements and enforcement strength before transport. An explicit
diagnostic admission retains its limitations and cannot support promotion.

Adapters expose `attach_toolset` and typed invocation results. Codex supports
its configured app-server or isolated official-SDK transport. Claude uses its
Agent SDK transport. Dedicated DeepSeek and Kimi adapters use independently
locked workers. OpenAI-compatible routes cover explicit hosted or local
endpoints. Switching transports or SDKs changes provenance; it is never an
automatic fallback.

Suite [profile catalogs](../products/swebench/backend_profile_catalog.yaml)
record exact route identities. Use the corresponding
[DeepSWE catalog](../products/deepswe/backend_profile_catalog.yaml) for that
suite. The diagnostic DeepSeek construction template is bound by
[backend_config_bind_deepseek](workflows/backend_config_bind_deepseek.md).
Keep all credential values out of route receipts and run identities.

## Rubrics and rules

`crucible.rubrics.load(name)` loads prompts with their executable contract.
`available(include_dormant=False)`, `contract`, `lint_contracts`, and
`validate_verdict_contract` expose the inventory and validation boundary.
Direct YAML loading omits that shared boundary. Reviews emit
`assay.review.v2`; only a typed pass approves.

`crucible.rules.available()` lists deterministic rules. Pure rules accept a
Spec; context-aware rules also require an explicit `RuleContext`. Match rule
arity to registration and fail setup when required context is unavailable.

## Tools, runners, and policy

ToolSets carry explicit capability classification, dispatch accounting, and
scope. Compose source, write, and build tools through their factories
or [profiles](../tools/profiles.py). Backend-private MCP and function-calling
transports expose only attached tools. Directives own writable paths.

`common.runners` defines runner contexts and normalized build/test/verify
results. Configured runner modules use the Assay JSON subprocess contract.
A runner outside the `crucible.` namespace starts in Python safe-path mode,
and the empty and relative `PYTHONPATH` entries of its environment are
removed, because Python resolves them against the working directory. Neither
the launch nor such an entry adds the gated workspace (the working directory)
to the import path, so the module must be importable from the environment
(installed, or under an absolute `PYTHONPATH` entry). The rest of the
environment reaches the runner unchanged.
Deterministic workflow code owns gate execution, timeout, and settlement.
An optional Tier smoke benchmark requires an explicit runner and explicit
`metric_directions`; unknown metric direction is a configuration error.

[common/policies.py](../common/policies.py) provides typed breach/review policy.
Product policy instances live next to each suite's `spec_synthesis.py`.
The two product adapters own dataset semantics, isolated task execution,
candidate rules, verifiers, predictions/reward, and exact resume identities.

See [architecture](ARCHITECTURE.md) for durable authority and mutation safety.
