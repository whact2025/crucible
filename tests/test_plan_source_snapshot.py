# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Independent planners cannot discover sibling drafts through source tools."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess

import pytest

from crucible.tools import ToolContext
from crucible.tools.source import build_source_tools
from crucible.workflows import _plan_source_snapshot as snapshot


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *arguments],
        check=True, capture_output=True,
    )


def _repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    return path


def _write(root: Path, relative: str, blob: bytes = b"original\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def test_snapshot_preserves_current_source_and_narrows_to_explicit_project(tmp_path):
    repo = _repo(tmp_path / "repo")
    project = repo / "project"
    project.mkdir()
    tracked = _write(project, "src/app.py")
    tracked.chmod(0o755)
    _write(project, "deleted.py")
    _write(project, "README.md", b"User-facing source documentation\n")
    _write(repo, "sibling/unrelated.py")
    _write(project, ".gitignore", b"ignored/\n")
    for relative in (
        ".crucible/specs/app/goal/spec.yaml", ".claude/settings.json",
        ".codex/settings.json", ".agents/skills/local/SKILL.md",
        ".aws/credentials", ".env.local", "AGENTS.md", "controls/call.json",
    ):
        _write(project, relative, b"Excluded even when tracked\n")
    _git(repo, "add", ".")
    tracked.write_bytes(b"dirty tracked bytes\r\n")
    (project / "deleted.py").unlink()
    _write(project, "new.py", b"nonignored untracked bytes\n")
    _write(project, "ignored/cache.py")
    output = snapshot.create_plan_source_snapshot(
        project, tmp_path / "snapshot", excluded_roots=(project / "controls",),
    )
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == {
        ".gitignore", "README.md", "src/app.py", "new.py",
    }
    assert (output / "src/app.py").read_bytes() == b"dirty tracked bytes\r\n"
    assert stat.S_IMODE((output / "src/app.py").stat().st_mode) == 0o555
    assert not stat.S_IMODE(output.stat().st_mode) & 0o222


def test_snapshot_source_tools_never_see_live_sibling_drafts(tmp_path):
    repo = _repo(tmp_path / "repo")
    source = b"def visible_source():\n    return 1\n\nvisible_source()\n"
    _write(repo, "app.py", source)
    _git(repo, "add", "app.py")
    index_before = (repo / ".git/index").read_bytes()
    (repo / ".git/info/exclude").write_text("/.crucible/\n")
    corpus = repo / ".crucible/specs/app/goal"
    _write(corpus, "runs/planner_1/debug.txt", b"OPPONENT_DRAFT\n")
    output = snapshot.create_plan_source_snapshot(
        repo, corpus / "source", excluded_roots=(corpus,),
    )
    assert (repo / ".git/index").read_bytes() == index_before
    assert (repo / "app.py").read_bytes() == source
    _write(corpus, "runs/planner_2/debug.txt", b"LATER_OPPONENT_DRAFT\n")
    (repo / "app.py").write_bytes(b"live source changed after capture\n")
    tools = build_source_tools(ToolContext(workspace_root=output, repo_root=output))
    assert "visible_source" in tools.dispatch("read_source", {"file_path": "app.py"})
    assert "app.py:" in tools.dispatch("grep_symbol", {"pattern": "visible_source"})
    assert "app.py:" in tools.dispatch("find_callers", {
        "function_name": "visible_source", "languages": ["python"],
    })
    assert tools.dispatch("grep_symbol", {"pattern": "OPPONENT_DRAFT"}).startswith("No matches")
    for relative in (
        ".crucible/specs/app/goal/runs/planner_1/debug.txt",
        "../runs/planner_2/debug.txt", str(corpus / "runs/planner_1/debug.txt"),
    ):
        assert "OPPONENT_DRAFT" not in tools.dispatch("read_source", {"file_path": relative})
    assert "outside the repository" in tools.dispatch("grep_symbol", {
        "pattern": "OPPONENT_DRAFT", "directory": str(corpus),
    })
    assert (output / "app.py").read_bytes() == source


@pytest.mark.parametrize("alias", ["file_symlink", "directory_symlink", "hardlink"])
def test_snapshot_rejects_git_visible_aliases_without_partial_copy(tmp_path, alias):
    repo = _repo(tmp_path / "repo")
    outside = _write(tmp_path, "outside/private.py")
    if alias == "directory_symlink":
        (repo / "alias").symlink_to(outside.parent, target_is_directory=True)
    elif alias == "file_symlink":
        (repo / "alias.py").symlink_to(outside)
    else:
        os.link(outside, repo / "alias.py")
    output = tmp_path / "snapshot"
    with pytest.raises(snapshot.PlanSourceSnapshotError, match="unaliased"):
        snapshot.create_plan_source_snapshot(repo, output, excluded_roots=())
    assert not output.exists()
    assert outside.read_bytes() == b"original\n"


def test_snapshot_detects_source_changes_after_a_read_and_cleans_up(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    source = _write(repo, "app.py")
    real_read = snapshot._read_file

    def racing_read(*arguments):
        blob = real_read(*arguments)
        source.write_bytes(b"raced\n")
        return blob

    monkeypatch.setattr(snapshot, "_read_file", racing_read)
    output = tmp_path / "snapshot"
    with pytest.raises(snapshot.PlanSourceSnapshotError, match="changed"):
        snapshot.create_plan_source_snapshot(repo, output, excluded_roots=())
    assert not output.exists()


def test_snapshot_limits_fail_closed_and_existing_destination_is_preserved(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    _write(repo, "app.py", b"source exceeds the injected aggregate bound\n")
    output = tmp_path / "snapshot"
    monkeypatch.setattr(snapshot, "_MAX_SNAPSHOT_BYTES", 8)
    with pytest.raises(snapshot.PlanSourceSnapshotError, match="byte limit"):
        snapshot.create_plan_source_snapshot(repo, output, excluded_roots=())
    assert not output.exists()
    output.mkdir()
    sentinel = _write(output, "operator.txt")
    with pytest.raises(snapshot.PlanSourceSnapshotError, match="already exists"):
        snapshot.create_plan_source_snapshot(repo, output, excluded_roots=())
    assert sentinel.read_bytes() == b"original\n"
