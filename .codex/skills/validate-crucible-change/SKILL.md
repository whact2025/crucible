---
name: validate-crucible-change
description: Validate a Crucible or sibling Assay code change with scope-aware tests, static checks, documentation checks, and a final security/provenance review. Use when Codex changes backend adapters, workflow orchestration, Assay contracts, tools, product runners, documentation, dependencies, or release-critical configuration in this repository.
---
<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->


# Validate Crucible Change

Validate from the repository root and report evidence, not confidence.

1. Inspect `git status --short` in Crucible and `../assay`. Preserve unrelated
   user changes and identify which repository owns every touched file.
2. Read the applicable invariants in `DESIGN_AXIOMS.md`, the relevant workflow
   document under `docs/workflows/`, and any product-local instructions.
3. Run the smallest focused test modules that exercise the changed contract.
   Include regression cases for failure, cancellation, cleanup, and stale
   provenance when those paths are in scope.
4. Run `uv run ruff check` on changed Python files. Run
   `uv run python -m pytest tests/test_documentation.py` after documentation or public
   inventory changes.
5. Run the full non-live Crucible suite for cross-cutting changes:
   `uv run python -m pytest -q`. In `../assay`, run `uv run python -m pytest -q`.
6. Run `git diff --check` in both repositories. Review the final diff for
   credential exposure, hidden-test access, route/model fallback, unbounded
   concurrency, unsafe mutation, incomplete evidence acceptance, and cleanup
   gaps.
7. Do not run `llm_live`, paid cohorts, hidden evaluators, or production
   mutations merely as validation. Run them only when the task authorizes the
   exact route/cost/security boundary and record requested plus served
   identity.
8. State every skipped, blocked, or externally governed gate explicitly.
   Never describe diagnostic evidence as promotion evidence.
