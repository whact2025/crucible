# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows.__main__`` dispatcher."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

import pytest

from crucible.workflows.__main__ import (
    _discover_workflows,
    _print_listing,
    _resolve_main,
    main,
)


def test_discover_workflows_includes_known():
    workflows = _discover_workflows()
    # Every supported solution-building and benchmark command is discoverable.
    for name in (
        "architect",
        "breach",
        "deepswe",
        "goal",
        "handoff",
        "implement",
        "plan",
        "review",
        "swebench",
        "tier",
        "validate",
    ):
        assert name in workflows
        assert workflows[name].startswith("crucible.workflows.")


def test_discover_workflows_skips_internals():
    workflows = _discover_workflows()
    assert "__main__" not in workflows
    assert "__init__" not in workflows


def test_resolve_main_returns_callable():
    fn = _resolve_main("implement")
    assert callable(fn)


def test_resolve_main_unknown_raises_systemexit():
    with pytest.raises(SystemExit, match="unknown workflow"):
        _resolve_main("does_not_exist")


def test_print_listing_emits_known_recipes():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _print_listing()
    assert rc == 0
    output = buf.getvalue()
    assert "implement" in output
    assert "validate" in output
    assert "architect" in output


def test_main_with_list_flag():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--list"])
    assert rc == 0
    assert "Available workflows" in buf.getvalue()


def test_main_help_flag():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--help"])
    assert rc == 0
    assert "Available workflows" in buf.getvalue()


def test_main_unknown_recipe_systemexit():
    with pytest.raises(SystemExit, match="unknown workflow"):
        main(["nope_not_a_recipe"])
