# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Exercise operator skill installation without touching user configuration."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


SKILLS = ("crucible-status", "crucible-design", "crucible-build", "crucible-handoff")
SOURCE = Path(__file__).parents[1] / ".claude" / "skills"
SCRIPT = SOURCE / "_shared" / "bin" / "install.sh"


def _install(
    home: Path, *args: str, script: Path = SCRIPT
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        env={**os.environ, "HOME": str(home)},
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_links(root: Path, source: Path = SOURCE) -> None:
    for client in (".claude", ".agents"):
        installed = root / client / "skills"
        assert sorted(path.name for path in installed.iterdir()) == sorted(SKILLS)
        for skill in SKILLS:
            assert (installed / skill).is_symlink()
            assert (installed / skill).resolve() == (source / skill).resolve()
            assert (installed / skill / "SKILL.md").is_file()


def test_default_install_shares_canonical_skills_for_both_clients(tmp_path: Path) -> None:
    home = tmp_path / "home with spaces"

    first = _install(home)
    second = _install(home)

    assert first.returncode == second.returncode == 0
    _assert_links(home)
    assert "already linked" in second.stdout


def test_project_install_works_in_empty_folder_without_touching_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "new project"
    project.mkdir()

    result = _install(home, "--project", str(project))

    assert result.returncode == 0
    _assert_links(project)
    assert not home.exists()


def test_conflicting_user_skill_is_preserved_and_reported(tmp_path: Path) -> None:
    home = tmp_path / "home"
    conflict = home / ".agents" / "skills" / "crucible-design"
    conflict.mkdir(parents=True)
    marker = conflict / "user-owned.txt"
    marker.write_text("keep me")

    result = _install(home)

    assert result.returncode == 1
    assert marker.read_text() == "keep me"
    assert not conflict.is_symlink()
    assert "CONFLICT" in result.stderr
    assert (home / ".claude" / "skills" / "crucible-design").is_symlink()
    assert (home / ".agents" / "skills" / "crucible-build").is_symlink()


def test_stale_link_is_replaced_and_dry_run_does_not_write(tmp_path: Path) -> None:
    home = tmp_path / "home"
    untouched = _install(home, "--dry-run")
    assert untouched.returncode == 0
    assert not home.exists()

    stale = home / ".agents" / "skills" / "crucible-build"
    stale.parent.mkdir(parents=True)
    stale.symlink_to(tmp_path / "removed-checkout", target_is_directory=True)
    preview = _install(home, "--dry-run")
    assert preview.returncode == 0
    assert stale.readlink() == tmp_path / "removed-checkout"
    assert not (home / ".claude").exists()

    installed = _install(home)
    assert installed.returncode == 0
    _assert_links(home)


def test_project_source_is_not_replaced_with_self_links(tmp_path: Path) -> None:
    project = tmp_path / "crucible"
    source = project / ".claude" / "skills"
    shutil.copytree(SOURCE, source)
    script = source / "_shared" / "bin" / "install.sh"

    result = _install(tmp_path / "home", "--project", str(project), script=script)

    assert result.returncode == 0
    for skill in SKILLS:
        assert (source / skill).is_dir()
        assert not (source / skill).is_symlink()
        assert (project / ".agents" / "skills" / skill).resolve() == source / skill


def test_invalid_project_is_rejected_before_writing(tmp_path: Path) -> None:
    home = tmp_path / "home"

    for args in (("--project",), ("--project", str(tmp_path / "missing"))):
        result = _install(home, *args)
        assert result.returncode == 2
    assert not home.exists()


def test_unwritable_destination_cannot_report_success(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    blocked = home / ".agents"
    blocked.write_text("user-owned configuration")

    result = _install(home)

    assert result.returncode == 1
    assert blocked.read_text() == "user-owned configuration"
    assert "LINK:" not in "\n".join(
        line for line in result.stdout.splitlines() if "/.agents/" in line
    )
