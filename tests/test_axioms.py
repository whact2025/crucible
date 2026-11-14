# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Executable contracts for the supported Crucible architecture."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

_CRUCIBLE_ROOT = Path(__file__).resolve().parent.parent  # crucible/
_WORKFLOWS_DIR = _CRUCIBLE_ROOT / "workflows"

_SUPPORTED_TOP_LEVEL_PYTHON_DIRS = frozenset({
    "backends", "common", "products", "rubrics", "rules", "scripts",
    "tests", "tools", "workflows",
})


def test_current_python_surfaces_match_the_architecture() -> None:
    """A new top-level capability requires an explicit architecture decision."""
    actual = {path.parent.name for path in _CRUCIBLE_ROOT.glob("*/*.py")}
    unexpected = actual - _SUPPORTED_TOP_LEVEL_PYTHON_DIRS
    assert not unexpected, f"undocumented top-level Python surfaces: {sorted(unexpected)}"


def _workflow_module_names() -> list[str]:
    return sorted(
        p.stem
        for p in _WORKFLOWS_DIR.glob("*.py")
        if not p.stem.startswith("_") and p.stem != "__init__"
    )


@pytest.mark.parametrize("mod_name", _workflow_module_names())
def test_l2_every_workflow_exposes_main_argv_int(mod_name: str):
    """L2 — workflows support direct execution and explicit argv calls."""
    mod = importlib.import_module(f"crucible.workflows.{mod_name}")
    main = getattr(mod, "main", None)
    assert callable(main), f"workflow {mod_name!r} has no callable main()"
    signature = inspect.signature(main)
    params = list(signature.parameters.values())
    assert params, f"{mod_name}.main must accept argv"
    argv = params[0]
    assert argv.name == "argv", f"{mod_name}.main's first parameter must be argv"
    assert argv.default is None, f"{mod_name}.main must default argv to None"
    assert argv.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
        f"{mod_name}.main must accept positional and keyword argv"
    )
    signature.bind()
    signature.bind(argv=[])


@pytest.mark.skipif(
    not (_CRUCIBLE_ROOT.parent / "assay" / "src").is_dir(),
    reason="sibling assay/src not present in this checkout",
)
def test_l1_assay_does_not_import_crucible():
    """L1 — crucible depends on assay, never the reverse. A non-comment
    `import crucible` inside assay/src breaks the one-way layering."""
    assay_src = _CRUCIBLE_ROOT.parent / "assay" / "src"
    offenders: list[str] = []
    for py in assay_src.rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if s.startswith("import crucible") or s.startswith("from crucible"):
                offenders.append(f"{py}:{i}: {s}")
    assert not offenders, "L1 violation — assay imports crucible:\n" + "\n".join(offenders)
