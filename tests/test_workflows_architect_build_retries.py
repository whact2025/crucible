# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Architect build-retry boundary and forwarding tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.workflows import architect, tier


def test_architect_cli_forwards_build_retries(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}

    def _run_architect(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(architect, "run_architect", _run_architect)

    assert architect.main([
        "--planner-count", "1",
        "--spec", str(tmp_path / "spec"),
        "--cwd", str(tmp_path),
        "--build-retries", "0",
    ]) == 0
    assert captured["build_retries"] == 0


@pytest.mark.parametrize(
    ("parse_args", "required"),
    [
        (architect._parse_args, []),
        (tier._parse_args, ["--tier", "t0"]),
    ],
)
def test_workflow_clis_reject_negative_build_retries(
    tmp_path: Path,
    parse_args,
    required: list[str],
):
    with pytest.raises(SystemExit):
        parse_args([
            "--spec", str(tmp_path / "spec"),
            *required,
            "--build-retries", "-1",
        ])


@pytest.mark.parametrize("entrypoint", [architect.run_architect, tier.run_tier])
def test_programmatic_workflows_reject_negative_build_retries(
    tmp_path: Path,
    entrypoint,
):
    kwargs = {
        "spec": tmp_path / "missing-spec",
        "cwd": str(tmp_path),
        "build_retries": -1,
    }
    if entrypoint is tier.run_tier:
        kwargs["tier_id"] = "t0"

    with pytest.raises(
        ValueError,
        match="build_retries must be a non-negative integer",
    ):
        entrypoint(**kwargs)
