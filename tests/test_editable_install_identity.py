# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the editable-checkout identity preflight."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from crucible.conftest import _require_tested_checkout_is_imported


def test_editable_identity_accepts_collected_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "crucible"
    checkout.mkdir()
    package_init = checkout / "__init__.py"
    package_init.touch()

    _require_tested_checkout_is_imported(
        collected_root=checkout,
        imported_init=str(package_init),
    )


def test_editable_identity_rejects_sibling_checkout(tmp_path: Path) -> None:
    collected = tmp_path / "candidate" / "crucible"
    imported = tmp_path / "other" / "crucible"
    collected.mkdir(parents=True)
    imported.mkdir(parents=True)
    imported_init = imported / "__init__.py"
    imported_init.touch()

    with pytest.raises(pytest.UsageError, match="editable-install mismatch") as caught:
        _require_tested_checkout_is_imported(
            collected_root=collected,
            imported_init=str(imported_init),
        )

    message = str(caught.value)
    assert str(collected) in message
    assert str(imported) in message
    assert "<unique-parent>/crucible" in message


def test_editable_identity_rejects_namespace_package(tmp_path: Path) -> None:
    checkout = tmp_path / "crucible"
    checkout.mkdir()

    with pytest.raises(pytest.UsageError, match="namespace package"):
        _require_tested_checkout_is_imported(
            collected_root=checkout,
            imported_init=None,
        )


def test_dashboard_only_collection_rejects_sibling_checkout(tmp_path: Path) -> None:
    """The repository guard must cover focused dashboard test invocations."""

    candidate = tmp_path / "candidate"
    dashboard_tests = candidate / "dashboard" / "tests"
    dashboard_tests.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve().parents[1] / "conftest.py", candidate)
    (candidate / "__init__.py").touch()
    (candidate / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["dashboard/tests"]\n',
        encoding="utf-8",
    )
    (dashboard_tests / "test_probe.py").write_text(
        "def test_probe():\n    assert True\n",
        encoding="utf-8",
    )

    imported_parent = tmp_path / "imported"
    imported_checkout = imported_parent / "crucible"
    imported_checkout.mkdir(parents=True)
    (imported_checkout / "__init__.py").touch()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(imported_parent), environment.get("PYTHONPATH")))
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "dashboard/tests",
        ],
        cwd=candidate,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "editable-install mismatch" in output
    assert str(candidate.resolve()) in output
    assert str(imported_checkout.resolve()) in output
