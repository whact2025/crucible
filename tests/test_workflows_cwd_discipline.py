# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Cwd-discipline audit: every ``subprocess.run(...)`` reachable from
``workflows/*.py`` must pass ``cwd=`` explicitly.

The audit is structural, not runtime. Process-cwd reliance inside any
workflow body is incompatible with the parallel-safety contract the
SWE-bench adapter needs (per-instance fan-out runs many workflow
invocations against different workspaces concurrently — any
subprocess.run that defaults to process cwd would race).

The check walks every ``subprocess.run`` and ``subprocess.Popen`` call
AST node under ``crucible/workflows/`` and asserts a ``cwd=`` keyword
argument is present. Indirect helpers (e.g., ``_build_verify``'s
``_run_runner``) are caught because the audit walks every workflow-
adjacent ``.py`` file.

The audit excludes:
- ``contextlib.chdir`` usage (different mechanism; flagged separately
  in the body if surviving).
- subprocess wrappers that accept ``cwd=`` as their own kwarg and
  forward it (the audit treats these as transparent — the kwarg is
  threaded explicitly, just not passed to subprocess.run at the
  literal call site if the wrapper handles it).

A failure here means a workflow grew a process-cwd dependency. Fix by
threading ``cwd=`` to the call site or, if the surrounding helper
takes ``cwd=`` and forwards it, ensuring the literal ``subprocess.run``
call carries the kwarg.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"


def _collect_subprocess_calls(tree: ast.AST) -> list[ast.Call]:
    """Find every ``subprocess.run`` / ``subprocess.Popen`` Call node."""
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match ``subprocess.run(...)`` and ``subprocess.Popen(...)``.
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
            and func.attr in ("run", "Popen", "check_output", "check_call")
        ):
            out.append(node)
    return out


def _has_cwd_kwarg(call: ast.Call) -> bool:
    return any(kw.arg == "cwd" for kw in call.keywords)


def _workflow_py_files() -> list[Path]:
    return sorted(p for p in _WORKFLOWS_DIR.glob("*.py") if p.name != "__init__.py")


def test_every_subprocess_call_in_workflows_passes_cwd() -> None:
    """Every literal ``subprocess.run``/``Popen`` inside ``workflows/``
    must pass ``cwd=``. Failure indicates a workflow grew a
    process-cwd dependency — fix by threading ``cwd=`` to the call.
    """
    offenders: list[str] = []
    for py in _workflow_py_files():
        src = py.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for call in _collect_subprocess_calls(tree):
            if not _has_cwd_kwarg(call):
                rel = py.relative_to(_WORKFLOWS_DIR.parent)
                offenders.append(f"{rel}:{call.lineno}")
    assert not offenders, (
        "subprocess calls without explicit cwd= (process-cwd reliance "
        "breaks parallel-safe workflow execution): "
        + ", ".join(offenders)
    )


def test_no_contextlib_chdir_in_workflow_bodies() -> None:
    """``contextlib.chdir(...)`` in a workflow body process-cwd-locks
    every nested call. The cwd= kwarg pattern supersedes this. Audit
    catches regressions.

    The check looks for ``contextlib.chdir`` ATTRIBUTE references AND
    aliased forms (``_contextlib.chdir`` etc) by scanning text — AST
    aliasing introspection isn't worth the complexity here.
    """
    offenders: list[str] = []
    for py in _workflow_py_files():
        src = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(src.splitlines(), start=1):
            # Skip docstrings / comments that reference the API.
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if "contextlib.chdir(" in line and not stripped.startswith(("\"", "'")):
                rel = py.relative_to(_WORKFLOWS_DIR.parent)
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, (
        "contextlib.chdir(...) in workflow body breaks parallel-safe "
        "execution; pass cwd= explicitly instead: "
        + ", ".join(offenders)
    )


# Workflows that get a CLI ``main()`` are allowed exactly ONE
# ``os.getcwd()`` call — the process-cwd fallback for CLI ergonomics
# when ``--cwd`` is unset. Anywhere else, ``os.getcwd()`` re-reads
# process state mid-flow and re-introduces the parallel-safety hole
# the cwd= kwarg pattern was designed to close. The audit caps the
# count per file at 1.
#
# Workflows that don't expose a CLI ``main()`` taking ``--cwd``
# (e.g., bench / sweep / validate / etc) are still allowed 0 or 1
# os.getcwd() reads — the test asserts the upper bound, not a
# positive presence.
_OS_GETCWD_MAX_PER_FILE = 1


def _count_os_getcwd_calls(tree: ast.AST) -> list[int]:
    """Find every ``os.getcwd()`` / ``_os.getcwd()`` Call lineno."""
    out: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in ("os", "_os")
            and func.attr == "getcwd"
        ):
            out.append(node.lineno)
    return out


def test_os_getcwd_capped_per_workflow_file() -> None:
    """Each workflow file gets at most one ``os.getcwd()`` call,
    reserved for ``main()``'s CLI fallback when ``--cwd`` is unset.

    Programmatic entries (``run_tier``, ``run_architect``, etc) MUST
    take ``cwd`` as an explicit kwarg and propagate it; they MUST
    NOT call ``os.getcwd()`` themselves. The cap catches a regression
    where someone re-introduces an ``os.getcwd()`` deep in a workflow
    body, which would silently re-introduce process-cwd reliance.
    """
    offenders: list[tuple[Path, list[int]]] = []
    for py in _workflow_py_files():
        src = py.read_text(encoding="utf-8")
        tree = ast.parse(src)
        linenos = _count_os_getcwd_calls(tree)
        if len(linenos) > _OS_GETCWD_MAX_PER_FILE:
            offenders.append((py, linenos))
    if offenders:
        msgs = [
            f"{py.relative_to(_WORKFLOWS_DIR.parent)} has "
            f"{len(linenos)} os.getcwd() calls (lines {linenos}); "
            f"max is {_OS_GETCWD_MAX_PER_FILE} (reserved for main()'s "
            f"CLI fallback)"
            for py, linenos in offenders
        ]
        raise AssertionError(
            "os.getcwd() over-use re-introduces parallel-safety "
            "hole; pass cwd= explicitly inside run_* bodies: "
            + "; ".join(msgs)
        )
