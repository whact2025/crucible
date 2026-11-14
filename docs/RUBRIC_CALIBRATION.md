<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Rubric calibration

Calibration measures decision accuracy, false passes/rejections, criterion
coverage, evidence recall, injection resistance, and stability against retained
cases. The offline fixtures cover every active rubric.

```bash
uv run python -m pytest tests/test_rubric_calibration.py tests/test_rubrics.py
```

`crucible.rubrics.calibration` provides case/prediction loaders, inventory
coverage validation, scoring, and explicitly selected backend generation.
Cases bind original artifact and prior-input bytes by digest. A changed prompt,
manifest, source fixture, or route creates a different calibration identity.

Keep generated proposals separate from source evidence. Include adversarial
cases for missing sources, prompt injection, contradictory prior findings,
truncation, unsupported decisions, and false approval. Update the fixture
inventory when an active rubric changes. Golden offline scores test contract
behavior; they do not establish a model's real-world quality.

Live calibration is marked `llm_live` and requires explicit route, credential,
budget, and execution authorization. Record exact route identity, token/cost
completeness, and terminal outcomes. See [rubric contracts](../rubrics/README.md).
