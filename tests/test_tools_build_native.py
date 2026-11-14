# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the native ``crucible.tools.build`` handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from assay import ToolSet
from crucible.tools import ToolContext
from crucible.tools.build import build_build_tools


def _ctx(tmp_path: Path, **kwargs) -> ToolContext:
    kwargs.setdefault("write_scope", ("crates", "Cargo.toml"))
    return ToolContext(workspace_root=tmp_path, repo_root=tmp_path, **kwargs)


# ----------------------------------------------------------------------
# Surface / protocol
# ----------------------------------------------------------------------


def test_build_toolset_protocol(tmp_path):
    ts = build_build_tools(_ctx(tmp_path))
    assert isinstance(ts, ToolSet)


def test_build_toolset_names(tmp_path):
    ts = build_build_tools(_ctx(tmp_path))
    assert ts.tool_names == {"build_project", "create_rust_crate"}


def test_build_toolset_all_write(tmp_path):
    ts = build_build_tools(_ctx(tmp_path))
    assert ts.write_names == ts.tool_names


def test_build_anthropic_shape(tmp_path):
    ts = build_build_tools(_ctx(tmp_path))
    out = {e["name"]: e for e in ts.to_anthropic_tools()}
    assert "project" in out["build_project"]["input_schema"]["properties"]
    assert "name" in out["create_rust_crate"]["input_schema"]["properties"]
    assert "prefix" in out["create_rust_crate"]["input_schema"]["properties"]


# ----------------------------------------------------------------------
# build_project — no runners → skipped path
# ----------------------------------------------------------------------


def test_build_project_skipped_when_no_runners(tmp_path):
    ts = build_build_tools(_ctx(tmp_path))  # runners=None by default
    result = ts.dispatch("build_project", {"project": "."})
    assert "BUILD_STATUS: skipped" in result
    assert "BUILD_ROOT:" in result
    assert "No runners configured" in result


def test_build_project_legacy_status_prefix_preserved(tmp_path):
    """The 'BUILD_STATUS: <status>' grep anchor must survive the port."""
    ts = build_build_tools(_ctx(tmp_path))
    result = ts.dispatch("build_project", {})
    assert result.startswith("BUILD_STATUS:")


# ----------------------------------------------------------------------
# create_rust_crate — end-to-end against a tmp workspace
# ----------------------------------------------------------------------


def test_create_rust_crate_scaffolds_directory(tmp_path):
    # Seed a workspace Cargo.toml so the membership update has something
    # to write to.
    (tmp_path / "Cargo.toml").write_text(
        '[workspace]\nmembers = [\n    "crates/existing",\n]\n',
    )
    ts = build_build_tools(_ctx(tmp_path))
    result = ts.dispatch("create_rust_crate", {"name": "demo"})
    assert "Created crate demo" in result
    crate_root = tmp_path / "crates" / "demo"
    assert (crate_root / "Cargo.toml").is_file()
    assert (crate_root / "src" / "lib.rs").is_file()
    cargo_text = (crate_root / "Cargo.toml").read_text()
    assert 'name = "demo"' in cargo_text


def test_create_rust_crate_default_has_no_prefix(tmp_path):
    """Without an explicit prefix, the package name == the directory name."""
    ts = build_build_tools(_ctx(tmp_path))
    ts.dispatch("create_rust_crate", {"name": "vanilla"})
    cargo_text = (tmp_path / "crates" / "vanilla" / "Cargo.toml").read_text()
    assert 'name = "vanilla"' in cargo_text
    assert "fabric-" not in cargo_text


def test_create_rust_crate_with_prefix_applies_house_style(tmp_path):
    """prefix='fabric-' yields a 'fabric-<name>' package, dir stays '<name>'."""
    ts = build_build_tools(_ctx(tmp_path))
    result = ts.dispatch(
        "create_rust_crate", {"name": "core", "prefix": "fabric-"},
    )
    assert "Created crate fabric-core" in result
    crate_root = tmp_path / "crates" / "core"  # directory NOT prefixed
    cargo_text = (crate_root / "Cargo.toml").read_text()
    assert 'name = "fabric-core"' in cargo_text
    lib_text = (crate_root / "src" / "lib.rs").read_text()
    assert "fabric-core" in lib_text


def test_create_rust_crate_bin_uses_main_rs(tmp_path):
    ts = build_build_tools(_ctx(tmp_path))
    ts.dispatch("create_rust_crate", {"name": "tool", "crate_type": "bin"})
    crate_root = tmp_path / "crates" / "tool"
    assert (crate_root / "src" / "main.rs").is_file()
    assert not (crate_root / "src" / "lib.rs").is_file()


def test_create_rust_crate_registers_in_workspace(tmp_path):
    (tmp_path / "Cargo.toml").write_text(
        '[workspace]\nmembers = [\n    "crates/existing",\n]\n',
    )
    ts = build_build_tools(_ctx(tmp_path))
    ts.dispatch("create_rust_crate", {"name": "new"})
    workspace = (tmp_path / "Cargo.toml").read_text()
    # Member entry was added (either by toml-write or regex fallback).
    assert "crates/new" in workspace
