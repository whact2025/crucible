<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Crucible

Crucible is a solution-building incubator. It turns a stated problem into an
accepted plan, implements scoped changes, challenges them with adversarial
review, and retains the evidence needed to evaluate the result. Its two
benchmark suites are **SWE-bench Verified** and **DeepSWE**.

Crucible composes the typed records, Corpus, and primitives in
[Assay](https://github.com/whact2025/assay). Each benchmark task uses the same
reusable solution-building workflow as an ordinary repository change.

```mermaid
flowchart LR
    Problem[Goal or benchmark task] --> Plan[Plan and deterministic validation]
    Plan --> Build[Scoped implementation and visible verification]
    Build --> Challenge[Adversarial breach and review]
    Challenge --> Result[Recorded candidate and terminal outcome]
    Result --> Hidden[Independent benchmark verification]
```

## Install

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). Keep Crucible
and Assay as siblings because the development dependency is an editable path:

```bash
git clone https://github.com/whact2025/assay.git assay
git clone https://github.com/whact2025/crucible.git crucible
cd crucible
uv sync
bash .claude/skills/_shared/bin/install.sh
```

The built-in route uses the `codex` backend, model `gpt-6-astra`, and `max`
reasoning. Install the system Codex CLI and authenticate its session before
running a model-backed workflow. Claude routes use a system `claude` CLI.
The installer makes the same four operator skills available in Codex and
Claude Code across your projects. It links to this checkout, so keep the
checkout available. Start a new agent session if the skills do not appear.
On Windows, run the setup in WSL; the examples below use Bash.
Exact SDK versions and runtime dependencies live in
[pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

## Describe a solution

Open the folder where you want to work and describe the outcome to your agent.
You can start with existing code or an empty folder. You do not need to write
code, YAML, or a spec template to create a specification and plan.

For an existing project:

> Use crucible-design to inspect this folder and create a spec and accepted
> plan for adding CSV export. Preserve existing behavior and cover empty
> results and Unicode. Handle the spec files for me. Use the configured route
> with a $10 budget for the design work.

For a new project:

> Use crucible-design to start a new Python command-line project in this
> empty folder. It should track local tasks, support adding and completing
> tasks, and retain them between runs. Create the spec and accepted plan for
> me. Use the configured route with a $10 budget for the design work.

The [design skill](.claude/skills/crucible-design/SKILL.md) turns your
description into goals, constraints, observable completion criteria, and
build/test configuration. It inspects existing code when present, creates and
promotes the spec, and runs the requested goal and planning stages. In an empty
folder it establishes the local project baseline and plans the files to create.
It asks about consequential missing choices; you can leave routine decisions
to the agent. It presents the goal and waits for your explicit approval before
planning. Planning does not implement the application.

After approval, three planners independently develop complete plans from the
same goal and source context, sharing the total planning budget. Each proposal
receives adversarial review and deterministic checks. An evaluator compares
the proposals and may select one or synthesize a stronger plan. A synthesis
must pass fresh review and validation before it becomes executable.

When you want the plan executed:

> Use crucible-build to implement the accepted plan, run its checks, and
> complete adversarial review. Use the configured route with a $20 cumulative
> budget.

The [build skill](.claude/skills/crucible-build/SKILL.md) executes the plan
through implementation, build/test checks, breach, and review. It reports the
saved results, remaining work, and actual cost. You can also ask
`crucible-status` for the current state or use `crucible-handoff` to record work
you completed interactively.

The skills manage a solution's saved records under
`.crucible/specs/<target>/<solution>/`: the authored spec, accepted plan,
implementation attempts, verification, and review evidence. The
[shared lifecycle](.claude/skills/_shared/lifecycle.md) explains how those
records establish completion. Model-backed design and build work uses the
configured route and consumes its authorized budget.

## Run a benchmark

Benchmark execution requires Linux, Docker, the selected dataset, sufficient
local disk, and the suite's runtime prerequisites. Listing tasks and inspecting
saved evidence do not require paid model calls. Hidden test inputs stay outside
model-visible authoring contexts.

The [benchmark operator guide](docs/BENCHMARKS.md) covers immutable images,
kernel quotas, one-task dry-runs, bounded execution, evidence, and resume.

Select tasks before starting an implementation run:

```bash
uv run crucible swebench --spec /absolute/path/to/swebench-corpus \
  --dataset /absolute/path/to/verified.json --limit 1 --list

uv run crucible deepswe --spec /absolute/path/to/deepswe-corpus \
  --dataset /absolute/path/to/manifest.json --limit 1 --list
```

To execute the selected task, ask your agent:

> Run this SWE-bench instance through Crucible using the selected dataset and
> configured route. Keep the total task budget at $20, use the default three
> independent planners, and report the candidate, verification, and actual cost.

Both suites use three independent planners by default within the same total
task budget. Planning, adversarial evaluation, comparison, and synthesis share
an allowance of at most 40% of that budget; each planner does not receive its
own task budget. Each active task can issue three concurrent draft requests.
An explicit `--planner-count 1` selects a single planning track for diagnostics.
Changing the planning recipe requires fresh planning and matching completion
evidence; old single-track results retain their original identity.

Implementation requires a positive finite task allowance for competing plans.
Set aggregate and per-instance limits deliberately, and bind an explicit route
with `--backend-config` when needed. Execution can start paid model calls,
build containers, and run the suite's evaluator; consult the suite reference:

- [SWE-bench](docs/workflows/swebench.md): instance isolation, source-only
  predictions, pristine visible verification, official evaluation, and resume.
- [DeepSWE](docs/workflows/deepswe.md): task selection, public development
  checks, isolated hidden verification, and binary reward reporting.

Backend profiles and convergence inventories live under
[products/swebench](products/swebench/) and
[products/deepswe](products/deepswe/). A checked-in profile describes an exact
route; diagnostic profiles do not confer promotion or launch authority.

The [workflow reference](docs/workflows/README.md) covers all 14 commands.
The [architecture](docs/ARCHITECTURE.md) explains authority and isolation;
the [public API](docs/PUBLIC_API.md) covers programmatic composition and
read-only or isolated-writer fan-out.

## Configure a route

Routing uses the closed `crucible.backend-routing.v1` schema. Discovery checks
explicit `--backend-config`, `CRUCIBLE_BACKEND_CONFIG`, Spec-local routing,
workspace routing, then the built-in default, in that order. A missing explicit
file is an error. The resolved route is frozen across a benchmark cascade.

```yaml
schema_version: crucible.backend-routing.v1
default:
  backend: codex
  model: gpt-6-astra
  effort: max
  feature_admission: enforce
roles: {}
```

Keep secrets in the runtime environment. Routing files can reference
`${VARIABLE}` values; credentials do not belong in committed configuration or
run provenance. Adapter, model, endpoint, runtime, prompt, policy, ToolSet, and
session identities remain explicit. Unsupported routes fail before transport;
Crucible never substitutes another provider or model.

Backend-private MCP transports expose scoped tools to individual model
invocations. Tool authority comes from the current directive and role; ambient
user/project tools and settings cannot expand it.

## Validate changes

```bash
uv run python -m pytest -q
uv run ruff check .
git diff --check
```

Live Docker and model tests require explicit execution. Start with focused
tests for the changed behavior. See [the test guide](tests/TEST_BASELINE.md)
and [documentation guide](docs/README.md) for the current contracts.
