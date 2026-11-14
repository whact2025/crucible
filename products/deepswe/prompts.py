# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""DeepSWE implementer system prompt.

Specializes ``SWEBENCH_SYSTEM_PROMPT`` by replacing its source-only/test-path
rule with DeepSWE's canonical full-patch contract, then appends an API-contract
fidelity clause. The replacement is exact and cardinality-checked so prompt
drift cannot silently leave contradictory instructions in place.
DeepSWE ``instruction.md`` bodies routinely state the EXACT public API
(property names, object shapes, return values, function/method signatures)
in code blocks, and the held-out verifier asserts that surface. The
cascade's decomposition into slice directives can PARAPHRASE the API
(e.g. "per-registration metrics") and lose the literal shape (a name-keyed
record vs an array), so the implementer must be told the issue's stated API
is a hard contract that outranks any paraphrase.

Kept separate from ``SWEBENCH_SYSTEM_PROMPT`` (composed, not edited) so the
SWE-bench prompt ↔ swebench-rubric alignment contract is undisturbed.
"""

from __future__ import annotations

from crucible.products.swebench.prompts import SWEBENCH_SYSTEM_PROMPT

_SWEBENCH_TEST_PATH_RULE = """\
- Do NOT modify test files. You may read the repository's visible tests, but
  Crucible excludes standard test paths when it captures the source-only
  prediction. The evaluator later supplies its own tests, so a test edit is
  wasted effort and may also confuse the diagnosis.
"""

_DEEPSWE_FULL_PATCH_RULE = """\
- DEEPSWE PATCH CONTRACT: the benchmark grades the COMPLETE repository patch
  left in the visible ``/app`` checkout. A visible repository path may be
  legitimate production code even when its name looks like a test (for
  example ``base_test.py``, ``test_set.py``, or
  ``browser_test_runner.js``). Do not classify a path by its name. Edit any
  visible repository file needed to implement the issue, including a project
  test when the issue genuinely requires that change.
- ``/tests`` and ``/solution`` are hidden benchmark mounts, not paths in the
  visible repository, and are unavailable during authoring. Never attempt to
  access, reproduce, or modify their contents.
- Crucible may temporarily create a pipeline-owned breach artifact while
  checking a slice. Its provenance-bound mutation journal restores the exact
  pre-breach path before candidate capture. Do not preserve or recreate an
  artifact identified as pipeline-owned. An ordinary visible repository file
  is not pipeline-owned merely because its path contains ``test`` or ``spec``.
"""

if SWEBENCH_SYSTEM_PROMPT.count(_SWEBENCH_TEST_PATH_RULE) != 1:
    raise RuntimeError(
        "SWE-bench prompt test-path rule drifted; DeepSWE specialization "
        "requires exactly one replaceable policy block"
    )

_DEEPSWE_BASE_PROMPT = SWEBENCH_SYSTEM_PROMPT.replace(
    _SWEBENCH_TEST_PATH_RULE,
    _DEEPSWE_FULL_PATCH_RULE,
    1,
)

_API_CONTRACT_CLAUSE = """\

API CONTRACT — match the issue's stated public API EXACTLY:
- When the issue shows a public API (code blocks with property accesses,
  function/method signatures, return shapes, option names), that surface is
  a CONTRACT the hidden acceptance tests assert against. Implement those
  names, object shapes, and signatures VERBATIM.
- Do NOT rename, restructure, or "improve" the stated shape. If the issue
  shows ``result.metrics.database.duration``, expose a name-keyed ``metrics``
  record whose entries have a ``duration`` field — NOT an array of
  per-item objects, NOT ``durationMs``. If it shows ``cancel()``, give
  ``cancel`` exactly that signature — do not add parameters.
- A slice directive may PARAPHRASE the API ("per-registration metrics").
  The issue's literal code block always wins over a paraphrase; when they
  differ, implement the literal code block.
"""


DEEPSWE_SYSTEM_PROMPT = (
    _DEEPSWE_BASE_PROMPT + _API_CONTRACT_CLAUSE
)


__all__ = ["DEEPSWE_SYSTEM_PROMPT"]
