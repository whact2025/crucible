# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Filesystem-convention dispatcher for crucible workflows.

Usage:

    python -m crucible.workflows <workflow-name> [args...]
    python -m crucible.workflows --list

Workflows are modules under ``crucible/workflows/<name>.py`` that
expose a top-level ``main(argv: list[str] | None = None) -> int``
callable. This dispatcher auto-discovers them by importing the module
and calling ``main(remaining_argv)``.

The discovery rules:
- ``<name>.py`` must define a ``main`` callable.
- Names starting with ``_`` are skipped (treated as internal).
- ``__init__`` and ``__main__`` are skipped (package machinery).

The dispatcher converts the workflow's return value to ``int``. ``SystemExit``
and other unhandled exceptions propagate to the caller.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

_WORKFLOWS_DIR = Path(__file__).parent


def _discover_workflows() -> dict[str, str]:
    """Return ``{workflow_name: dotted_module_path}`` for all discoverable workflows."""
    found: dict[str, str] = {}
    for p in sorted(_WORKFLOWS_DIR.glob("*.py")):
        stem = p.stem
        if stem.startswith("_") or stem in ("__init__", "__main__"):
            continue
        found[stem] = f"crucible.workflows.{stem}"
    return found


def _resolve_main(workflow: str) -> Callable[[list[str] | None], int]:
    workflows = _discover_workflows()
    dotted = workflows.get(workflow)
    if dotted is None:
        raise SystemExit(
            f"unknown workflow {workflow!r}; available: "
            f"{', '.join(sorted(workflows)) or '(none)'}"
        )
    mod = importlib.import_module(dotted)
    main = getattr(mod, "main", None)
    if not callable(main):
        raise SystemExit(
            f"workflow {workflow!r} ({dotted}) does not define a callable main()"
        )
    return main


def _print_listing() -> int:
    # Docstring blurbs may carry non-cp1252 glyphs ('→', '∪'); on a
    # Windows console the default charmap codec hard-crashes the listing
    # — the primary CLI discovery surface. Degrade to replacement chars
    # instead of dying.
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001 — non-reconfigurable stream (pipe/test)
        pass
    workflows = _discover_workflows()
    if not workflows:
        sys.stdout.write("(no workflows found)\n")
        return 0
    sys.stdout.write("Available workflows:\n")
    for name in sorted(workflows):
        # Pull the first line of the module docstring as a description hint.
        try:
            mod = importlib.import_module(workflows[name])
            doc = (mod.__doc__ or "").strip().splitlines()
            blurb = doc[0] if doc else ""
        except Exception:  # noqa: BLE001
            blurb = ""
        sys.stdout.write(f"  {name:20s} {blurb}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:]) if argv is None else list(argv)
    if not args or args[0] in ("-h", "--help"):
        sys.stdout.write((__doc__.strip() if __doc__ else "") + "\n\n")
        return _print_listing()
    if args[0] == "--list":
        return _print_listing()
    workflow = args[0]
    rest = args[1:]
    runner = _resolve_main(workflow)
    return int(runner(rest) or 0)


if __name__ == "__main__":
    sys.exit(main())
