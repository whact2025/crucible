# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Static workflow contracts for elapsed time and source provenance."""

from __future__ import annotations

import ast
from pathlib import Path


_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"


def _workflow_files() -> tuple[Path, ...]:
    return tuple(sorted(_WORKFLOWS_DIR.glob("*.py")))


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_run_results_cannot_silently_retain_zero_default_duration() -> None:
    offenders: list[str] = []
    for path in _workflow_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != "RunResult":
                continue
            if any(keyword.arg == "duration_s" for keyword in node.keywords):
                continue
            parent = parents.get(node)
            if (
                not isinstance(parent, ast.Call)
                or _call_name(parent) != "finalize_elapsed"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "RunResult constructors must provide duration_s or be finalized through "
        "RunHandle.finalize_elapsed: " + ", ".join(offenders)
    )


def test_direct_handle_finalize_requires_explicit_duration() -> None:
    offenders: list[str] = []
    for path in _workflow_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != "finalize":
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "handle"
            ):
                continue
            result = node.args[0] if node.args else None
            if not (
                isinstance(result, ast.Call)
                and _call_name(result) == "RunResult"
                and any(
                    keyword.arg == "duration_s"
                    for keyword in result.keywords
                )
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "direct RunHandle.finalize calls require an explicit duration; use "
        "finalize_elapsed otherwise: " + ", ".join(offenders)
    )


def test_every_workflow_run_open_declares_provenance_root() -> None:
    offenders: list[str] = []
    for path in _workflow_files():
        # _run.py implements the forwarding API itself. Its omitted-keyword
        # legacy path is covered by behavioral tests; workflow callers must be
        # explicit so they cannot accidentally promote ambient Git state.
        if path.name == "_run.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _call_name(node) == "open_run_with_capture"
                and not any(
                    keyword.arg == "provenance_root"
                    for keyword in node.keywords
                )
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "workflow run openers must bind one target root or pass explicit None: "
        + ", ".join(offenders)
    )


def test_every_handled_terminal_declares_provenance_root() -> None:
    offenders: list[str] = []
    for path in _workflow_files():
        if path.name == "_run.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _call_name(node) == "record_handled_terminal"
                and not any(
                    keyword.arg == "provenance_root"
                    for keyword in node.keywords
                )
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "handled workflow terminals must bind one target root or pass explicit "
        "None: " + ", ".join(offenders)
    )
