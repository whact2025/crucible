---
name: qualify-crucible-backend
description: Qualify an exact Codex, Claude, DeepSeek, Kimi, or OpenAI-compatible route for Crucible without fallback. Use when adding or upgrading a backend SDK, model, endpoint, tool loop, session mode, limit claim, routing manifest, or A/B/C evaluation arm.
---
<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->


# Qualify Crucible Backend

Treat backend family, exact model, endpoint, runtime revision, feature profile,
tool authority, and evidence policy as one immutable route.

1. Read `DESIGN_AXIOMS.md`, `docs/ARCHITECTURE.md`, and the selected
   suite reference and exact checked-in backend profile.
2. Resolve credentials from the approved environment at runtime. Never print,
   persist, hash, place in process arguments, or copy them into YAML.
3. Reject unsupported features before constructing a paid transport. Never
   substitute a model, endpoint, SDK/CLI, provider, or backend family.
4. Keep confined runs free of ambient project/user instructions, tools,
   plugins, MCP servers, skills, agents, hooks, and provider overrides.
5. Bind the route/profile/schema/prompt/tool/policy/runtime/session identities
   into provenance. Require exact call IDs, balanced tool call/results, one
   terminal event, and truthful token/cost/trace completeness.
6. Prove turn, response-token, UTF-8 context, USD, retry, deadline,
   cancellation, and cleanup behavior at the enforcement level declared by
   the concrete route. Do not label post-hoc or boundary-only controls hard.
7. Run synthetic conformance and hostile-source probes before any live call.
   Keep mutation tools process-isolated and crash-atomic; prove timeout,
   cancellation, and forced death cannot mutate after the terminal result.
8. For an authorized live canary, freeze the exact endpoint/model/revision,
   use the smallest bounded request, capture served identity and quota/usage,
   and clean up every process/listener/credential artifact.
   Benchmark qualification also binds planner count and the whole task budget.
   The default three draft requests can run concurrently inside one task/cell;
   account for nested concurrency even when the cell limit is one. Historical
   single-track receipts remain evidence for that original recipe.
9. Mark evidence diagnostic unless all dependency, license, security,
   entitlement, identity, completeness, and route-specific promotion gates
   pass. Never enable an A/B/C arm solely because a nominal call succeeded.
