# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""DeepSWE-specific runner modules.

Included adapters:

- ``deepswe_verifier.py`` — wraps ``bash /tests/test.sh`` inside the per-task
  container; parses the v1.0 text or v1.1 JSON reward artifact and emits the
  binary reward as a build_verify KPI.
- ``deepswe_dev_verify.py`` — runs task-local development checks with the
  benchmark-neutral containment engine.
- ``dev_verify_outcome.py`` — classifies development-verification evidence.

Build-role runners are per-language and live under
``crucible.common.runners.lang.*``. DeepSWE's uniform hidden verifier and its
language-dispatching development verifier live here. The runner contract is
the same as ``assay.bench``:
JSON BenchConfig in on stdin, JSON KPI payload out on stdout.
"""
