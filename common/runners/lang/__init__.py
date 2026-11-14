# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Per-language runners satisfying the assay.bench subprocess contract.

Each module in this package is invokable via
``python -m crucible.common.runners.lang.<lang>``:

- Reads one JSON BenchConfig-shaped object from stdin
  (``{"bench_id", "parameters", "env"}``).
- Runs the language's canonical verify or build command.
- Emits one JSON object on stdout with shape
  ``{"values": {...}, "raw_output": "..."}``.
- Exits 0 on success, non-zero on test/build failure.

The runners are deliberately content-free at the spec layer:
``crucible/common/runners/polyglot.py`` dispatches to them in the
order declared by its workflow-injected ``parameters.languages`` field,
aggregates KPI values, and short-circuits on the first non-zero exit. Canonical
Spec authors declare ``constraints[kind=languages]``; the workflow owns that
top-level runner parameter.

Runners encapsulate the default command for each language. Spec authors
declare which languages a slice touches, and the workflow build/verify layer
threads those declarations into the subprocess configuration.

Bundled languages:

- ``crucible.common.runners.lang.python`` — ``uv run pytest`` (default;
  configurable via ``parameters.test_command`` / ``test_args``).
- ``crucible.common.runners.lang.rust`` — ``cargo test --workspace``
  (or ``cargo test -p <crate>`` via ``parameters.crate``).
- ``crucible.common.runners.lang.cpp`` — ``cmake --build build &&
  ctest --test-dir build`` (configurable build_dir + ctest_args).

Each module also exposes per-language *convention* helpers consumed
outside the bench-runner subprocess contract:

- ``breach_test_path(repo_root) -> Path`` — where the breach
  Defender writes adversarial test files for this language.
- ``build_command(repo_root) -> str`` — canonical default project-build
  command for convention consumers. Typed crate/feature selection remains in
  the Spec parameter envelope.
- ``verify_command(repo_root) -> str`` — canonical default project-test
  command, with the same parameter boundary.

The bundled breach-test filenames currently use the ``swebench_breach``
prefix even when the helpers are called by another workflow; callers with
different project conventions should supply an explicit breach-test path.

The ``conventions_for(language)`` helper below returns the matching
module so dispatch lives in one place. Adding a new language means
adding a new ``lang/<name>.py`` exposing the three helpers + adding
one entry here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class LanguageConventions(Protocol):
    """Per-language convention surface every ``lang/<X>.py`` exposes."""

    @staticmethod
    def breach_test_path(repo_root: Path) -> Path: ...

    @staticmethod
    def build_command(repo_root: Path) -> str: ...

    @staticmethod
    def verify_command(repo_root: Path) -> str: ...


def conventions_for(language: str) -> LanguageConventions:
    """Return the ``crucible.common.runners.lang.<language>`` module.

    Lookup is case-insensitive; ``python`` / ``py`` / ``rust`` / ``rs``
    / ``cpp`` / ``c++`` all resolve. Raises ``ValueError`` for unknown
    languages so callers see a clear error rather than a None default.
    """
    canonical = language.strip().lower()
    if canonical in ("python", "py"):
        from . import python as _mod
        return _mod  # type: ignore[return-value]
    if canonical in ("rust", "rs"):
        from . import rust as _mod
        return _mod  # type: ignore[return-value]
    if canonical in ("cpp", "c++", "cxx"):
        from . import cpp as _mod
        return _mod  # type: ignore[return-value]
    raise ValueError(
        f"no language conventions registered for {language!r}; "
        f"add crucible/common/runners/lang/<lang>.py exposing "
        f"breach_test_path / build_command / verify_command + a "
        f"branch here"
    )


__all__ = ["LanguageConventions", "conventions_for"]
