# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Repository-wide pytest bootstrap invariants."""

from __future__ import annotations

from pathlib import Path

import crucible as _imported_crucible
import pytest


def _require_tested_checkout_is_imported(
    *,
    collected_root: Path,
    imported_init: str | None,
) -> None:
    """Fail before collection if pytest imported a different checkout.

    Crucible's package lives at the repository root, so Hatch's editable
    install adds the repository's parent directory to ``sys.path``. A Git
    worktree whose basename is not ``crucible`` can therefore import a sibling
    checkout named ``crucible`` while pytest still collects tests from the
    current directory. That produces plausible but invalid test evidence.
    """
    expected_root = collected_root.resolve()
    if imported_init is None:
        imported_root = None
    else:
        imported_root = Path(imported_init).resolve().parent

    if imported_root == expected_root:
        return

    imported_display = "<namespace package>" if imported_root is None else str(imported_root)
    raise pytest.UsageError(
        "Crucible editable-install mismatch: pytest is collecting "
        f"{expected_root}, but `import crucible` resolves to {imported_display}. "
        "Create development worktrees as <unique-parent>/crucible (with Assay "
        "at <unique-parent>/assay), run `uv sync` in that checkout, and retry."
    )


_require_tested_checkout_is_imported(
    collected_root=Path(__file__).resolve().parent,
    imported_init=_imported_crucible.__file__,
)
