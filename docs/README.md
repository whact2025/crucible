<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Crucible documentation

These documents describe the current solution-building incubator and its two
benchmark suites.

| Need | Reference |
|---|---|
| Install and run | [Repository guide](../README.md) |
| Understand authority, isolation, and orchestration | [Architecture](ARCHITECTURE.md) |
| Compose supported Python APIs | [Public API](PUBLIC_API.md) |
| Select a command | [Workflow index](workflows/README.md) |
| Run SWE-bench Verified | [SWE-bench reference](workflows/swebench.md) |
| Run DeepSWE | [DeepSWE reference](workflows/deepswe.md) |
| Prepare and operate a one-task benchmark run | [Benchmark operator guide](BENCHMARKS.md) |
| Understand executable invariants | [Design axioms](../DESIGN_AXIOMS.md) |
| Change review prompts | [Rubric contracts](../rubrics/README.md) |
| Validate rubric behavior | [Rubric calibration](RUBRIC_CALIBRATION.md) |
| Test a change | [Test guide](../tests/TEST_BASELINE.md) |

The filesystem dispatcher owns command inventory. Each current workflow has
one reference page, with its purpose, inputs, behavior, options, outputs, and
terminal semantics. `crucible <workflow> --help` is the parser authority.

Documentation describes supported behavior in present tense. Keep examples
bounded, routes explicit, secrets in the environment, and implementation links
local to their owners. Delete documentation for removed capabilities.

```bash
uv run python -m pytest tests/test_documentation.py tests/test_axioms.py
uv run python -m pytest tests/test_workflows_dispatcher.py tests/test_rubrics.py
uv run ruff check .
git diff --check
```

The documentation checks cover local links, balanced fences, Python examples,
Mermaid declarations, exact workflow/page inventory, CLI option coverage,
backend defaults, and rubric inventory.
