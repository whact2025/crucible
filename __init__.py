# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""An adversarial solution-building incubator built on Assay.

Crucible composes goal, plan, validation, implementation, and independent
adversarial review into bounded workflows with durable execution evidence.
SWE-bench Verified and DeepSWE adapters evaluate solutions against their
separate dataset, authoring, submission, and verifier contracts.

The fourteen public commands are goal, plan, validate, implement, breach,
review, handoff, tier, architect, reconcile, journal_recover, swebench,
deepswe, and backend_config_bind_deepseek. Both ``crucible`` and
``python -m crucible`` discover these commands under ``crucible.workflows``.

Layer map:

- ``crucible.backends`` owns backend adapters, role routing, and transport policy.
- ``crucible.rubrics`` owns bundled review rubrics and their evidence contracts.
- ``crucible.rules`` owns deterministic Spec validation.
- ``crucible.constraints`` registers domain constraint subtypes with Assay.
- ``crucible.common.runners`` owns language build/test adapters and declared
  verification.
- ``crucible.tools`` owns scoped source, build, and mutation tools and their
  backend-private transport.
- ``crucible.products`` owns the SWE-bench and DeepSWE integrations.

Assay owns typed records, canonical serialization, Corpus storage, and generic
primitives. Crucible owns domain policy and workflow composition.
"""

__version__ = "0.4.0"
