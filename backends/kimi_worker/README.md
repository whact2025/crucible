<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Isolated Kimi SDK worker

The dedicated Kimi adapter uses this independently locked Python project.
Its dependency graph is separate from Crucible's root environment.

```bash
uv sync --project backends/kimi_worker --locked
```

The worker protocol binds exact model, reasoning effort, response/context
bounds, and diagnostic cost policy. The configured K2.7 high-speed route and K3
route use different request fields and retain separate identities. Unsupported
model/effort pairs, runtime drift, and older protocols fail before dispatch.
Checked-in project-locked profiles reject arbitrary interpreter overrides and
revalidate the installed runtime before probes and real spawns.

Tool events bind the invocation, attempt, and call ID. Only the classified
attached ToolSet is exposed through an invocation-private bridge. Native tools,
ambient configuration, hidden retries, and automatic compaction remain disabled.
Cancellation must close the worker and listener and establish complete settlement.

Accounting accepts only complete, matching terminal usage and exact stream
completion. Duplicate or malformed counters, served-model drift, contradictory
terminals, or missing final evidence fail closed. Cached reads/writes remain
explicit and are not subtracted twice.

The opt-in K3 diagnostic cost policy permits only its specifically validated
missing cache-read uncertainty. It retains lower/upper bounds, never invents
exact reads or actual charges, and stops further work when the conservative
ceiling is reached. Post-response spending guards can overshoot on the final
indivisible response; diagnostic continuation never grants ranking, exact
accounting, or promotion.

Select the exact suite profile and inspect its admission and runtime binding.
A new worker source or lock requires new runtime evidence. Existing retained
run evidence is immutable. See [routing and capabilities](../../docs/PUBLIC_API.md).
