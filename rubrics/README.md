<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Rubric contracts

Crucible ships 18 active prompt bodies and one executable inventory:
[_manifest.yaml](_manifest.yaml). Load prompts through
`crucible.rubrics.load()` so the shared trust boundary and generated
`assay.review.v2` contract apply.

| Purpose | Rubrics |
|---|---|
| Goal refinement | `goal_red`, `goal_blue`, `goal_judge` |
| Planning | `plan_critic`, `plan_judge`, `plan_selection`, `swebench_plan_critic`, `deepswe_plan_critic` |
| Adversarial gates | `impl_breaker`, `swebench_impl_breaker`, `breach_attack_authority`, `breach_attack_entailment`, `breach_judge` |
| Implementation review | `api_contract_fidelity`, `diff_scope`, `spec_clarity`, `tier_completeness`, `review_conflict_authority` |

The manifest owns consumer, role, context, artifact, capability, completeness,
evidence-source, decision, severity-floor, criterion, and target-closure rules.
Only a typed `decision == "pass"` approves. Active findings at or above a
rubric's floor block approval. Missing or malformed responses never pass.

Generated proposals and declarative attacks are distinct from evidence.
Evidence resolves to retained allowed source bytes and quotes them exactly.
Review handoffs retain original artifacts and prior Verdicts without turning
model reasoning into source authority. Templates delimit injected artifacts
and context with `BEGIN_UNTRUSTED_*` / `END_UNTRUSTED_*` markers.

```bash
uv run python -m pytest tests/test_rubrics.py tests/test_rubric_calibration.py
uv run python -c "from crucible.rubrics import lint_contracts; assert not lint_contracts()"
```

`available(include_dormant=False)` gives the current selectable inventory.
`render_inventory_markdown()` renders its contract metadata. Add a prompt and
manifest entry together, wire its consuming workflow, pass every required
context/source, and validate decisions and evidence before branching. Do not
ship unused prompts or duplicate the shared response schema inside a prompt.
