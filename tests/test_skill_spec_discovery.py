# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Greenfield contract tests for the operator skill spec resolver."""

from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / ".claude"
    / "skills"
    / "_shared"
    / "bin"
    / "spec_discovery.sh"
)


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    spec = workspace / ".crucible" / "specs" / "kit" / "render_contract"
    spec.mkdir(parents=True)
    return workspace, spec


def _resolve(workspace: Path, value: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; resolve_spec "$2" "$3"',
            "bash",
            str(SCRIPT),
            value,
            str(workspace),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _is_canonical(workspace: Path, value: Path) -> int:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; is_canonical_spec_path "$2" "$3"',
            "bash",
            str(SCRIPT),
            str(value),
            str(workspace),
        ],
        check=False,
    )
    return completed.returncode


def test_resolve_accepts_short_name_and_explicit_canonical_path(tmp_path: Path) -> None:
    workspace, spec = _workspace(tmp_path)

    short = _resolve(workspace, "render_contract")
    explicit = _resolve(workspace, str(spec))
    relative = _resolve(workspace, ".crucible/specs/kit/render_contract")

    assert short.returncode == 0
    assert short.stdout.strip() == str(spec)
    assert explicit.returncode == 0
    assert explicit.stdout.strip() == str(spec)
    assert relative.returncode == 0
    assert relative.stdout.strip() == str(spec)


def test_resolve_rejects_explicit_path_outside_canonical_tree(tmp_path: Path) -> None:
    workspace, _ = _workspace(tmp_path)
    outside = workspace / "kit" / ".crucible" / "plans" / "render_contract"
    outside.mkdir(parents=True)

    completed = _resolve(workspace, str(outside))

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "outside the canonical tree" in completed.stderr


def test_canonical_predicate_requires_exact_target_and_spec_depth(tmp_path: Path) -> None:
    workspace, spec = _workspace(tmp_path)
    nested = spec / "runs"
    nested.mkdir()

    assert _is_canonical(workspace, spec) == 0
    assert _is_canonical(workspace, spec.parent) == 1
    assert _is_canonical(workspace, nested) == 1
