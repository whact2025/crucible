# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Scoped native write tools, artifact materialization, and mutation authority."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from assay import ToolSet
from crucible.tools import ToolContext
from crucible.tools.write import build_write_tools
from crucible.tools.write import handlers as write_handlers
from crucible.workflows._worktree import WorktreeResolver


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("new.py", "x.py", "deep", "f.py", "nope.py"),
    )


def _container_ctx(
    tmp_path: Path,
    *,
    container_workdir: str = "/app",
    write_scope: tuple[str, ...] = ("new.py",),
    write_exact_scope: tuple[str, ...] = (),
    write_deny_scope: tuple[str, ...] = (),
    path_resolver=None,
) -> ToolContext:
    return ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=write_scope,
        write_exact_scope=write_exact_scope,
        write_deny_scope=write_deny_scope,
        path_resolver=path_resolver,
        container_id="container-id",
        container_workdir=container_workdir,
    )


def _artifact_ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("artifacts",),
    )


def _staged_artifact(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "target" / "staging" / "artifact.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


class _ActiveWorktreeResolver:
    def __init__(self, repository: Path, worktree: Path) -> None:
        self.repository = repository.resolve()
        self.worktree = worktree.resolve()

    def resolve_read(self, relative: str) -> Path:
        worktree_path = (self.worktree / relative).resolve()
        if worktree_path.exists():
            return worktree_path
        return (self.repository / relative).resolve()

    def resolve_write(self, relative: str) -> Path:
        return (self.worktree / relative).resolve()

    def resolve_worktree_root(self, project: str = ".") -> Path:
        return self.worktree


# ----------------------------------------------------------------------
# Protocol + schema shape
# ----------------------------------------------------------------------


def test_write_toolset_protocol(tmp_path):
    ts = build_write_tools(_ctx(tmp_path))
    assert isinstance(ts, ToolSet)


def test_write_toolset_names(tmp_path):
    ts = build_write_tools(_ctx(tmp_path))
    assert ts.tool_names == {
        "write_source", "apply_patch", "delete_source",
        "materialize_artifact",
    }


def test_write_toolset_all_marked_write(tmp_path):
    ts = build_write_tools(_ctx(tmp_path))
    assert ts.write_names == ts.tool_names
    assert ts.read_only_names == set()


def test_write_toolset_anthropic_shape(tmp_path):
    ts = build_write_tools(_ctx(tmp_path))
    out = {e["name"]: e for e in ts.to_anthropic_tools()}
    assert set(out) == {
        "write_source", "apply_patch", "delete_source", "materialize_artifact",
    }
    assert "file_path" in out["write_source"]["input_schema"]["properties"]
    assert "content" in out["write_source"]["input_schema"]["properties"]
    assert "before" in out["apply_patch"]["input_schema"]["properties"]
    assert "after" in out["apply_patch"]["input_schema"]["properties"]
    delete_schema = out["delete_source"]["input_schema"]
    assert set(delete_schema["properties"]) == {"file_path", "recursive"}
    assert delete_schema["required"] == ["file_path", "recursive"]
    assert delete_schema["additionalProperties"] is False
    artifact_properties = out["materialize_artifact"]["input_schema"][
        "properties"
    ]
    assert set(artifact_properties) == {
        "source_path", "destination_path", "replace",
    }
    assert artifact_properties["replace"]["default"] is False
    for entry in out.values():
        assert "description" in entry
        assert "input_schema" in entry


# ----------------------------------------------------------------------
# write_source — end-to-end
# ----------------------------------------------------------------------


def test_write_source_creates_new_file(tmp_path):
    ts = build_write_tools(_ctx(tmp_path))
    result = ts.dispatch("write_source", {
        "file_path": "new.py",
        "content": "def hello():\n    return 'hi'\n",
    })
    assert "OK" in result
    assert "2 lines" in result
    assert (tmp_path / "new.py").read_text() == "def hello():\n    return 'hi'\n"


def test_write_source_overwrites_existing_file(tmp_path):
    original = "OLD\n"
    (tmp_path / "x.py").write_text(original)
    ts = build_write_tools(_ctx(tmp_path))
    ts.dispatch("write_source", {
        "file_path": "x.py",
        "content": "NEW\n",
        "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
    })
    assert (tmp_path / "x.py").read_text() == "NEW\n"


def test_write_source_rejects_hash_bound_identical_replacement(tmp_path):
    path = tmp_path / "x.py"
    original = "unchanged\n"
    path.write_text(original)

    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": "x.py",
        "content": original,
        "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
    })

    assert result.startswith("Error: replacement")
    assert "requires an actual edit" in result
    assert path.read_text() == original


def test_write_source_rejects_unverified_existing_file_expansion(tmp_path):
    path = tmp_path / "x.py"
    path.write_text("old\n")
    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": "x.py",
        "content": "a much larger replacement\n",
    })
    assert "refusing to replace" in result
    assert path.read_text() == "old\n"


def test_write_source_rejects_unverified_existing_file_shrink(tmp_path):
    path = tmp_path / "x.py"
    path.write_text("one\ntwo\nthree\n")
    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": "x.py",
        "content": "one\ntwo\n",
    })
    assert "refusing to shrink" in result
    assert path.read_text() == "one\ntwo\nthree\n"


def test_write_source_rejects_equal_line_count_byte_shrink(tmp_path):
    path = tmp_path / "x.py"
    path.write_text("x" * 100 + "\n" + "y" * 100 + "\n")
    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": "x.py",
        "content": "x\ny\n",
    })
    assert "refusing to shrink" in result
    assert path.read_text() == "x" * 100 + "\n" + "y" * 100 + "\n"


def test_write_source_allows_hash_bound_existing_file_shrink(tmp_path):
    path = tmp_path / "x.py"
    original = "one\ntwo\nthree\n"
    path.write_text(original)
    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": "x.py",
        "content": "one\ntwo\n",
        "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
    })
    assert result.startswith("OK:")
    assert path.read_text() == "one\ntwo\n"


def test_write_source_rejects_stale_hash(tmp_path):
    path = tmp_path / "x.py"
    path.write_text("one\ntwo\n")
    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": "x.py",
        "content": "replacement\n",
        "expected_sha256": "0" * 64,
    })
    assert "stale expected_sha256" in result
    assert path.read_text() == "one\ntwo\n"


def test_write_deny_scope_overrides_authorized_parent_directory(tmp_path):
    test_file = tmp_path / "tests" / "test_breach.py"
    test_file.parent.mkdir()
    test_file.write_text("original\n")
    ctx = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("tests",),
        write_deny_scope=("tests/test_breach.py",),
    )
    result = build_write_tools(ctx).dispatch("apply_patch", {
        "file_path": "tests/test_breach.py",
        "before": "original",
        "after": "mutated",
    })
    assert "cannot resolve write path" in result
    assert test_file.read_text() == "original\n"


def test_write_exact_scope_rejects_descendant_but_accepts_exact_file(tmp_path):
    ctx = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("required.py",),
        write_exact_scope=("required.py",),
    )
    tools = build_write_tools(ctx)

    child_result = tools.dispatch("write_source", {
        "file_path": "required.py/child.py",
        "content": "CHILD = True\n",
    })
    exact_result = tools.dispatch("write_source", {
        "file_path": "required.py",
        "content": "REQUIRED = True\n",
    })

    assert "cannot resolve" in child_result
    assert not tmp_path.joinpath("required.py", "child.py").exists()
    assert exact_result == "OK: wrote 1 lines to required.py"
    assert tmp_path.joinpath("required.py").read_text() == (
        "REQUIRED = True\n"
    )


def test_write_source_normalizes_redundant_workspace_prefix(tmp_path):
    original = "OLD\n"
    (tmp_path / "x.py").write_text(original)
    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": f"{tmp_path.name}/x.py",
        "content": "NEW\n",
        "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
    })
    assert result.startswith("OK:")
    assert (tmp_path / "x.py").read_text() == "NEW\n"
    assert not (tmp_path / tmp_path.name).exists()


@pytest.mark.parametrize("container_workdir", ["/app", "/testbed"])
def test_write_source_normalizes_container_workdir_absolute_alias(
    tmp_path,
    container_workdir,
):
    tools = build_write_tools(_container_ctx(
        tmp_path,
        container_workdir=container_workdir,
    ))

    result = tools.dispatch("write_source", {
        "file_path": f"{container_workdir}/new.py",
        "content": "VALUE = 1\n",
    })

    assert result == f"OK: wrote 1 lines to {container_workdir}/new.py"
    assert (tmp_path / "new.py").read_text() == "VALUE = 1\n"


def test_container_absolute_alias_writes_only_active_worktree(tmp_path):
    repository = tmp_path / "repository"
    worktree = tmp_path / "worktree"
    repository.mkdir()
    worktree.mkdir()
    resolver = _ActiveWorktreeResolver(repository, worktree)
    tools = build_write_tools(_container_ctx(
        repository,
        path_resolver=resolver,
    ))

    result = tools.dispatch("write_source", {
        "file_path": "/app/new.py",
        "content": "WORKTREE_ONLY = True\n",
    })

    assert result.startswith("OK:")
    assert (worktree / "new.py").read_text() == "WORKTREE_ONLY = True\n"
    assert not (repository / "new.py").exists()


def test_container_context_preserves_relative_backslash_normalization(tmp_path):
    tools = build_write_tools(_container_ctx(
        tmp_path,
        write_scope=("deep",),
    ))

    result = tools.dispatch("write_source", {
        "file_path": r"deep\nested\new.py",
        "content": "RELATIVE = True\n",
    })

    assert result.startswith("OK:")
    assert (tmp_path / "deep" / "nested" / "new.py").read_text() == (
        "RELATIVE = True\n"
    )


def test_container_absolute_alias_preserves_exact_and_deny_scopes(tmp_path):
    exact_tools = build_write_tools(_container_ctx(
        tmp_path,
        write_scope=("deep",),
        write_exact_scope=("deep/exact.py",),
    ))
    denied_tools = build_write_tools(_container_ctx(
        tmp_path,
        write_scope=("deep",),
        write_deny_scope=("deep/denied.py",),
    ))

    exact_result = exact_tools.dispatch("write_source", {
        "file_path": "/app/deep/exact.py",
        "content": "EXACT = True\n",
    })
    descendant_result = exact_tools.dispatch("write_source", {
        "file_path": "/app/deep/exact.py/child.py",
        "content": "ESCAPED = True\n",
    })
    denied_result = denied_tools.dispatch("write_source", {
        "file_path": "/app/deep/denied.py",
        "content": "DENIED = False\n",
    })

    assert exact_result.startswith("OK:")
    assert (tmp_path / "deep" / "exact.py").read_text() == "EXACT = True\n"
    assert "cannot resolve" in descendant_result
    assert "cannot resolve" in denied_result
    assert not (tmp_path / "deep" / "exact.py" / "child.py").exists()
    assert not (tmp_path / "deep" / "denied.py").exists()


@pytest.mark.parametrize(
    "file_path",
    [
        "/app",
        "/app/undeclared.py",
        "/app2/new.py",
        "/testbed/new.py",
        "/app/../new.py",
        "/solution/new.py",
        r"\app\new.py",
        r"/app\new.py",
    ],
)
def test_container_absolute_alias_rejects_non_scoped_spellings(
    tmp_path,
    file_path,
):
    result = build_write_tools(_container_ctx(tmp_path)).dispatch(
        "write_source",
        {"file_path": file_path, "content": "ESCAPED = True\n"},
    )

    assert "cannot resolve" in result
    assert not (tmp_path / "new.py").exists()
    assert not (tmp_path / "undeclared.py").exists()
    assert not (tmp_path / ".git" / "config").exists()


def test_container_absolute_alias_preserves_vcs_metadata_denial(tmp_path):
    tools = build_write_tools(_container_ctx(
        tmp_path,
        write_scope=(".git/config",),
    ))

    result = tools.dispatch("write_source", {
        "file_path": "/app/.git/config",
        "content": "ESCAPED = True\n",
    })

    assert "cannot resolve" in result
    assert not (tmp_path / ".git" / "config").exists()


@pytest.mark.parametrize(
    "container_workdir",
    ["app", "/app/../workspace", r"\app"],
)
def test_container_absolute_alias_requires_valid_absolute_posix_workdir(
    tmp_path,
    container_workdir,
):
    result = build_write_tools(_container_ctx(
        tmp_path,
        container_workdir=container_workdir,
    )).dispatch("write_source", {
        "file_path": "/app/new.py",
        "content": "ESCAPED = True\n",
    })

    assert "cannot resolve" in result
    assert not (tmp_path / "new.py").exists()


def test_container_absolute_alias_is_rejected_without_container(tmp_path):
    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": "/app/new.py",
        "content": "ESCAPED = True\n",
    })

    assert "cannot resolve" in result
    assert not (tmp_path / "new.py").exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_container_absolute_alias_preserves_link_containment(
    tmp_path,
    link_kind,
):
    actual = tmp_path / "actual.py"
    alias = tmp_path / "new.py"
    actual.write_text("SAFE = True\n")
    if link_kind == "symlink":
        alias.symlink_to(actual.name)
    else:
        alias.hardlink_to(actual)

    result = build_write_tools(_container_ctx(tmp_path)).dispatch(
        "write_source",
        {"file_path": "/app/new.py", "content": "SAFE = False\n"},
    )

    assert "cannot resolve" in result
    assert actual.read_text() == "SAFE = True\n"


def test_write_source_creates_parent_dirs(tmp_path):
    ts = build_write_tools(_ctx(tmp_path))
    ts.dispatch("write_source", {
        "file_path": "deep/nested/x.py",
        "content": "x = 1\n",
    })
    assert (tmp_path / "deep" / "nested" / "x.py").exists()


def test_write_source_rejects_outside_repo(tmp_path):
    ts = build_write_tools(_ctx(tmp_path))
    result = ts.dispatch("write_source", {
        "file_path": "../outside.py", "content": "x",
    })
    assert "cannot resolve" in result.lower()


def test_write_source_rejects_undeclared_in_repo_path(tmp_path):
    ts = build_write_tools(_ctx(tmp_path))
    result = ts.dispatch("write_source", {
        "file_path": "undeclared.py", "content": "x",
    })
    assert "cannot resolve" in result.lower()
    assert not (tmp_path / "undeclared.py").exists()


@pytest.mark.parametrize("metadata", [".git/config", ".hg/hgrc", "src/.svn/wc.db"])
def test_write_source_categorically_rejects_vcs_metadata(tmp_path, metadata):
    ctx = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=(metadata,),
    )
    result = build_write_tools(ctx).dispatch("write_source", {
        "file_path": metadata, "content": "owned",
    })
    assert "cannot resolve" in result.lower()
    assert not (tmp_path / metadata).exists()


def test_write_source_rejects_symlink_alias_to_other_repo_path(tmp_path):
    (tmp_path / "actual.py").write_text("safe\n")
    (tmp_path / "x.py").symlink_to("actual.py")
    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": "x.py", "content": "escaped\n",
    })
    assert "cannot resolve" in result.lower()
    assert (tmp_path / "actual.py").read_text() == "safe\n"


def test_write_source_rejects_symlink_alias_to_git_metadata(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("safe\n")
    (tmp_path / "x.py").symlink_to(".git/config")
    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": "x.py", "content": "escaped\n",
    })
    assert "cannot resolve" in result.lower()
    assert (tmp_path / ".git" / "config").read_text() == "safe\n"


def test_write_source_rejects_hardlink_alias(tmp_path):
    (tmp_path / "actual.py").write_text("safe\n")
    (tmp_path / "x.py").hardlink_to(tmp_path / "actual.py")
    result = build_write_tools(_ctx(tmp_path)).dispatch("write_source", {
        "file_path": "x.py", "content": "escaped\n",
    })
    assert "cannot resolve" in result.lower()
    assert (tmp_path / "actual.py").read_text() == "safe\n"


# ----------------------------------------------------------------------
# delete_source — exact directive-root removal
# ----------------------------------------------------------------------


def _delete_tools(
    tmp_path: Path,
    *,
    scope: tuple[str, ...] = ("donor",),
    deny: tuple[str, ...] = (),
):
    return build_write_tools(ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=scope,
        write_deny_scope=deny,
    ))


def test_delete_source_removes_exact_declared_file(tmp_path):
    target = tmp_path / "obsolete.py"
    target.write_text("obsolete\n")

    result = _delete_tools(
        tmp_path,
        scope=("obsolete.py",),
    ).dispatch("delete_source", {
        "file_path": "obsolete.py",
        "recursive": False,
    })

    assert result == "OK: deleted file obsolete.py"
    assert not target.exists()


def test_delete_source_recursively_removes_exact_declared_tree_only(tmp_path):
    donor = tmp_path / "donor"
    (donor / "nested").mkdir(parents=True)
    (donor / "root.txt").write_text("root\n")
    (donor / "nested" / "child.txt").write_text("child\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n")
    (donor / "outside-link").symlink_to(outside)

    result = _delete_tools(tmp_path).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })

    assert result == (
        "OK: recursively deleted 3 files and 2 directories from donor"
    )
    assert not donor.exists()
    assert outside.read_text() == "outside\n"


@pytest.mark.parametrize("phase", ("validate", "revalidate", "delete"))
def test_delete_tree_enumeration_stops_at_limit_plus_one(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    yielded: list[str] = []

    class _Entry:
        def __init__(self, name: str) -> None:
            self.name = name

    class _NonExhaustibleScan:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def __iter__(self):
            return self

        def __next__(self):
            if len(yielded) >= 2:
                raise AssertionError("directory iterator was over-consumed")
            name = ("a", "b")[len(yielded)]
            yielded.append(name)
            return _Entry(name)

    monkeypatch.setattr(
        write_handlers.os,
        "scandir",
        lambda _descriptor: _NonExhaustibleScan(),
    )
    monkeypatch.setattr(write_handlers, "_MAX_DELETE_ENTRIES", 1)
    manifest = (type("Expected", (), {"name": "a"})(),)

    with pytest.raises(ValueError, match="exceeds|changed"):
        if phase == "validate":
            write_handlers._validate_delete_tree(
                123,
                root_device=1,
                root_mount_id=1,
                depth=0,
                count=[0],
                regular_bytes=[0],
            )
        elif phase == "revalidate":
            write_handlers._revalidate_delete_tree(
                123,
                manifest=manifest,
                root_device=1,
                root_mount_id=1,
            )
        else:
            write_handlers._delete_tree_contents(
                123,
                manifest=manifest,
                root_device=1,
                root_mount_id=1,
            )

    assert yielded == ["a", "b"]


def test_nested_delete_inventory_charges_one_global_ceiling_and_closes_scans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    child = donor / "a"
    child.mkdir(parents=True)
    (child / "nested.txt").write_text("nested\n")
    (donor / "z.txt").write_text("root\n")
    real_scandir = write_handlers.os.scandir
    yielded: list[str] = []
    closed = 0

    class _TrackingScan:
        def __init__(self, descriptor: int) -> None:
            self._delegate = real_scandir(descriptor)
            self._iterator = None

        def __enter__(self):
            self._iterator = self._delegate.__enter__()
            return self

        def __exit__(self, *args):
            nonlocal closed
            closed += 1
            return self._delegate.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            assert self._iterator is not None
            entry = next(self._iterator)
            yielded.append(entry.name)
            return entry

    monkeypatch.setattr(
        write_handlers.os,
        "scandir",
        lambda descriptor: _TrackingScan(descriptor),
    )
    monkeypatch.setattr(write_handlers, "_MAX_DELETE_ENTRIES", 2)
    descriptor = os.open(donor, write_handlers._directory_open_flags())
    try:
        with pytest.raises(ValueError, match="exceeds 2 entries"):
            write_handlers._validate_delete_tree(
                descriptor,
                root_device=os.fstat(descriptor).st_dev,
                root_mount_id=write_handlers._mount_id_for_descriptor(
                    descriptor
                ),
                depth=0,
                count=[0],
                regular_bytes=[0],
            )
    finally:
        os.close(descriptor)

    assert len(yielded) == 3
    assert closed == 2


def test_delete_source_entry_ceiling_fails_before_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "a.txt").write_text("a\n")
    (donor / "b.txt").write_text("b\n")
    monkeypatch.setattr(write_handlers, "_MAX_DELETE_ENTRIES", 1)

    result = _delete_tools(tmp_path).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })

    assert "exceeds 1 entries" in result
    assert (donor / "a.txt").read_text() == "a\n"
    assert (donor / "b.txt").read_text() == "b\n"
    assert not tuple(tmp_path.glob(".donor.crucible-delete-*.quarantine"))


@pytest.mark.parametrize(
    ("failure_call", "terminal"),
    ((2, False), (3, True)),
    ids=("revalidation", "deletion"),
)
def test_delete_source_bounded_inventory_failure_restores_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_call: int,
    terminal: bool,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    kept = donor / "kept.txt"
    kept.write_text("kept\n")
    real_inventory = write_handlers._bounded_delete_names
    inventory_calls = 0

    def fail_selected_inventory(*args, **kwargs):
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls == failure_call:
            raise ValueError("injected bounded inventory failure")
        return real_inventory(*args, **kwargs)

    monkeypatch.setattr(
        write_handlers,
        "_bounded_delete_names",
        fail_selected_inventory,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("donor",),
    )

    if terminal:
        with pytest.raises(
            write_handlers.DeleteMutationIndeterminate,
            match="journal rollback",
        ):
            write_handlers._delete_source_impl(
                context,
                file_path="donor",
                recursive=True,
            )
    else:
        result = write_handlers._delete_source_impl(
            context,
            file_path="donor",
            recursive=True,
        )
        assert "injected bounded inventory failure" in result

    assert inventory_calls == failure_call
    assert kept.read_text() == "kept\n"
    assert not tuple(tmp_path.glob(".donor.crucible-delete-*.quarantine"))


def test_delete_source_requires_recursive_ack_for_directory(tmp_path):
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "kept.txt").write_text("kept\n")

    result = _delete_tools(tmp_path).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": False,
    })

    assert "retry with recursive=true" in result
    assert (donor / "kept.txt").read_text() == "kept\n"


@pytest.mark.parametrize(
    ("scope", "requested"),
    [
        (("donor",), "donor/nested"),
        (("donor/nested",), "donor"),
        (("donor",), "."),
        (("donor",), "/"),
        (("donor",), "undeclared"),
    ],
)
def test_delete_source_rejects_inferred_broad_and_unscoped_targets(
    tmp_path,
    scope,
    requested,
):
    nested = tmp_path / "donor" / "nested"
    nested.mkdir(parents=True)
    (nested / "kept.txt").write_text("kept\n")

    result = _delete_tools(
        tmp_path,
        scope=scope,
    ).dispatch("delete_source", {
        "file_path": requested,
        "recursive": True,
    })

    assert result.startswith("Error:")
    assert (nested / "kept.txt").read_text() == "kept\n"


def test_delete_source_rejects_declared_parent_with_denied_descendant(tmp_path):
    denied = tmp_path / "donor" / "immutable.txt"
    denied.parent.mkdir()
    denied.write_text("immutable\n")

    result = _delete_tools(
        tmp_path,
        deny=("donor/immutable.txt",),
    ).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })

    assert "exactly match an explicit, permitted" in result
    assert denied.read_text() == "immutable\n"


def test_delete_source_rejects_final_symlink_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kept.txt").write_text("kept\n")
    (tmp_path / "donor").symlink_to(outside, target_is_directory=True)

    result = _delete_tools(tmp_path).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })

    assert result.startswith("Error:")
    assert (tmp_path / "donor").is_symlink()
    assert (outside / "kept.txt").read_text() == "kept\n"


def test_delete_source_rejects_nested_vcs_metadata(tmp_path):
    metadata = tmp_path / "donor" / ".git"
    metadata.mkdir(parents=True)
    (metadata / "config").write_text("protected\n")

    result = _delete_tools(tmp_path).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })

    assert "protected VCS metadata" in result
    assert (metadata / "config").read_text() == "protected\n"


def test_delete_source_rejects_descendant_hardlink(tmp_path):
    donor = tmp_path / "donor"
    donor.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("shared\n")
    (donor / "alias.txt").hardlink_to(outside)

    result = _delete_tools(tmp_path).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })

    assert "multiply-linked entry" in result
    assert (donor / "alias.txt").exists()
    assert outside.read_text() == "shared\n"


def test_delete_source_rejects_multiply_linked_root_file(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("shared\n")
    target = tmp_path / "obsolete.txt"
    target.hardlink_to(outside)

    result = _delete_tools(
        tmp_path,
        scope=("obsolete.txt",),
    ).dispatch("delete_source", {
        "file_path": "obsolete.txt",
        "recursive": False,
    })

    assert "cannot resolve delete path" in result
    assert target.read_text() == "shared\n"
    assert outside.read_text() == "shared\n"


def test_delete_source_rejects_parent_omitted_from_exact_scope(tmp_path):
    donor = tmp_path / "donor"
    donor.mkdir()
    kept = donor / "kept.txt"
    kept.write_text("kept\n")
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("donor",),
        write_exact_scope=("donor/kept.txt",),
    )

    result = build_write_tools(context).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })

    assert "exactly match an explicit, permitted" in result
    assert kept.read_text() == "kept\n"


def test_delete_source_fails_closed_on_malformed_exact_scope(tmp_path):
    donor = tmp_path / "donor"
    donor.mkdir()
    kept = donor / "kept.txt"
    kept.write_text("kept\n")
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("donor",),
        write_exact_scope=("../outside",),
    )

    result = build_write_tools(context).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })

    assert "exactly match an explicit, permitted" in result
    assert kept.read_text() == "kept\n"


@pytest.mark.parametrize("special_at_root", [False, True])
def test_delete_source_rejects_fifo_entries(tmp_path, special_at_root):
    if special_at_root:
        fifo = tmp_path / "donor"
    else:
        donor = tmp_path / "donor"
        donor.mkdir()
        fifo = donor / "pipe"
    os.mkfifo(fifo)

    result = _delete_tools(tmp_path).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })

    assert result.startswith("Error")
    assert stat.S_ISFIFO(os.lstat(fifo).st_mode)


@pytest.mark.parametrize("replacement_is_directory", [False, True])
def test_delete_source_restores_pre_quarantine_root_replacement(
    tmp_path,
    replacement_is_directory,
):
    target = tmp_path / "donor"
    replacement = tmp_path / "replacement"
    parked = tmp_path / "parked"
    if replacement_is_directory:
        target.mkdir()
        (target / "original.txt").write_text("original\n")
        replacement.mkdir()
        (replacement / "replacement.txt").write_text("replacement\n")
    else:
        target.write_text("original\n")
        replacement.write_text("replacement\n")
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("donor",),
    )

    def replace_public_root() -> None:
        target.rename(parked)
        replacement.rename(target)

    result = write_handlers._delete_source_impl(
        context,
        file_path="donor",
        recursive=replacement_is_directory,
        publication_barrier=replace_public_root,
    )

    assert "changed before atomic quarantine" in result
    if replacement_is_directory:
        assert (target / "replacement.txt").read_text() == "replacement\n"
        assert (parked / "original.txt").read_text() == "original\n"
    else:
        assert target.read_text() == "replacement\n"
        assert parked.read_text() == "original\n"
    assert not tuple(tmp_path.glob(".*.crucible-delete-*.quarantine"))


def test_delete_source_preserves_public_collision_and_quarantined_tree(
    tmp_path,
    monkeypatch,
):
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "original.txt").write_text("original\n")
    moved = threading.Event()
    collision_created = threading.Event()
    real_rename = write_handlers._rename_noreplace_at

    def pause_after_quarantine(source_fd, source, destination_fd, destination):
        real_rename(source_fd, source, destination_fd, destination)
        if destination == write_handlers._DELETE_QUARANTINE_PAYLOAD:
            moved.set()
            assert collision_created.wait(2.0)

    def create_collision() -> None:
        assert moved.wait(2.0)
        donor.mkdir()
        (donor / "collision.txt").write_text("collision\n")
        collision_created.set()

    monkeypatch.setattr(
        write_handlers,
        "_rename_noreplace_at",
        pause_after_quarantine,
    )
    writer = threading.Thread(target=create_collision)
    writer.start()
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("donor",),
    )

    with pytest.raises(
        write_handlers.DeleteMutationIndeterminate,
        match="preserved",
    ):
        write_handlers._delete_source_impl(
            context,
            file_path="donor",
            recursive=True,
        )
    writer.join(timeout=2.0)

    assert not writer.is_alive()
    assert (donor / "collision.txt").read_text() == "collision\n"
    quarantines = tuple(
        tmp_path.glob(".donor.crucible-delete-*.quarantine")
    )
    assert len(quarantines) == 1
    assert (
        quarantines[0]
        / write_handlers._DELETE_QUARANTINE_PAYLOAD
        / "original.txt"
    ).read_text() == "original\n"


def test_delete_source_never_reports_success_after_late_public_recreation(
    tmp_path,
    monkeypatch,
):
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "original.txt").write_text("original\n")
    private_contents_removed = threading.Event()
    collision_created = threading.Event()
    real_delete = write_handlers._delete_tree_contents

    def pause_after_private_delete(*args, **kwargs):
        counts = real_delete(*args, **kwargs)
        private_contents_removed.set()
        assert collision_created.wait(2.0)
        return counts

    def recreate_public_root() -> None:
        assert private_contents_removed.wait(2.0)
        donor.mkdir()
        (donor / "collision.txt").write_text("collision\n")
        collision_created.set()

    monkeypatch.setattr(
        write_handlers,
        "_delete_tree_contents",
        pause_after_private_delete,
    )
    writer = threading.Thread(target=recreate_public_root)
    writer.start()
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("donor",),
    )

    with pytest.raises(
        write_handlers.DeleteMutationIndeterminate,
        match="payload removal",
    ):
        write_handlers._delete_source_impl(
            context,
            file_path="donor",
            recursive=True,
        )
    writer.join(timeout=2.0)

    assert not writer.is_alive()
    assert (donor / "collision.txt").read_text() == "collision\n"
    assert not tuple(
        tmp_path.glob(".donor.crucible-delete-*.quarantine")
    )


def test_delete_source_never_unlinks_post_barrier_insertion(tmp_path):
    donor = tmp_path / "donor"
    donor.mkdir()
    original = donor / "original.txt"
    original.write_text("original\n")
    late = donor / "late.txt"
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("donor",),
    )

    result = write_handlers._delete_source_impl(
        context,
        file_path="donor",
        recursive=True,
        publication_barrier=lambda: late.write_text("late\n"),
    )

    assert "changed" in result
    assert original.read_text() == "original\n"
    assert late.read_text() == "late\n"


def test_delete_source_never_unlinks_post_barrier_hardlink(tmp_path):
    donor = tmp_path / "donor"
    donor.mkdir()
    original = donor / "original.txt"
    original.write_text("original\n")
    outside_alias = tmp_path / "outside-alias.txt"
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("donor",),
    )

    result = write_handlers._delete_source_impl(
        context,
        file_path="donor",
        recursive=True,
        publication_barrier=lambda: outside_alias.hardlink_to(original),
    )

    assert "changed" in result
    assert original.read_text() == "original\n"
    assert outside_alias.read_text() == "original\n"


@pytest.mark.parametrize("mutation", ["content", "mode"])
def test_delete_source_rejects_in_place_descendant_drift(tmp_path, mutation):
    donor = tmp_path / "donor"
    donor.mkdir()
    target = donor / "target.txt"
    target.write_text("original\n")
    target.chmod(0o644)
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("donor",),
    )

    def mutate() -> None:
        if mutation == "content":
            target.write_text("replaced\n")
        else:
            target.chmod(0o600)

    result = write_handlers._delete_source_impl(
        context,
        file_path="donor",
        recursive=True,
        publication_barrier=mutate,
    )

    assert "changed" in result
    if mutation == "content":
        assert target.read_text() == "replaced\n"
    else:
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not tuple(tmp_path.glob(".*.crucible-delete-*.quarantine"))


def test_delete_source_rejects_oversized_sparse_regular_root(tmp_path):
    target = tmp_path / "obsolete.bin"
    with target.open("wb") as stream:
        stream.truncate(write_handlers._MAX_DELETE_REGULAR_BYTES + 1)

    result = _delete_tools(
        tmp_path,
        scope=("obsolete.bin",),
    ).dispatch("delete_source", {
        "file_path": "obsolete.bin",
        "recursive": False,
    })

    assert "regular-file bytes" in result
    assert target.stat().st_size == write_handlers._MAX_DELETE_REGULAR_BYTES + 1


def test_delete_source_fails_closed_without_kernel_mount_identity(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "obsolete.txt"
    target.write_text("kept\n")

    def unavailable(_descriptor):
        raise ValueError("injected unavailable mount identity")

    monkeypatch.setattr(
        write_handlers,
        "_mount_id_for_descriptor",
        unavailable,
    )
    result = _delete_tools(
        tmp_path,
        scope=("obsolete.txt",),
    ).dispatch("delete_source", {
        "file_path": "obsolete.txt",
        "recursive": False,
    })

    assert "unavailable mount identity" in result
    assert target.read_text() == "kept\n"


_BIND_MOUNT_DELETE_PROBE = r"""
import json
import os
import subprocess
import sys
from pathlib import Path

from crucible.tools import ToolContext
from crucible.tools.write import build_write_tools

repo = Path(sys.argv[1])
source = Path(sys.argv[2])
target = Path(sys.argv[3])
subprocess.run(
    ["mount", "--bind", str(source), str(target)],
    check=True,
    capture_output=True,
    text=True,
)
result = build_write_tools(ToolContext(
    workspace_root=repo,
    repo_root=repo,
    write_scope=("donor",),
)).dispatch("delete_source", {
    "file_path": "donor",
    "recursive": True,
})
print(json.dumps({
    "result": result,
    "same_device": os.stat(source).st_dev == os.stat(target).st_dev,
}))
"""


def _bind_mount_delete_probe(
    repo: Path,
    source: Path,
    target: Path,
) -> dict:
    completed = subprocess.run(
        [
            "unshare",
            "-Urnm",
            sys.executable,
            "-c",
            _BIND_MOUNT_DELETE_PROBE,
            str(repo),
            str(source),
            str(target),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(
            "kernel does not permit an unprivileged bind-mount namespace: "
            + completed.stderr[-500:]
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_delete_source_rejects_same_device_bind_mount_at_root(tmp_path):
    repo = tmp_path / "repo"
    donor = repo / "donor"
    external = tmp_path / "external"
    donor.mkdir(parents=True)
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("external\n")

    payload = _bind_mount_delete_probe(repo, external, donor)

    assert payload["same_device"] is True
    assert "mount boundary" in payload["result"]
    assert sentinel.read_text() == "external\n"


def test_delete_source_rejects_same_device_nested_bind_mount(tmp_path):
    repo = tmp_path / "repo"
    donor = repo / "donor"
    nested = donor / "nested"
    external = tmp_path / "external"
    nested.mkdir(parents=True)
    external.mkdir()
    local = donor / "local.txt"
    local.write_text("local\n")
    sentinel = external / "sentinel.txt"
    sentinel.write_text("external\n")

    payload = _bind_mount_delete_probe(repo, external, nested)

    assert payload["same_device"] is True
    assert "mount boundary" in payload["result"]
    assert local.read_text() == "local\n"
    assert sentinel.read_text() == "external\n"


# ----------------------------------------------------------------------
# materialize_artifact — scoped binary publication
# ----------------------------------------------------------------------


def test_materialize_artifact_copies_ignored_staging_bytes_exactly(tmp_path):
    content = bytes(range(256)) + b"\x00\xffnot-utf8\x80"
    source = _staged_artifact(tmp_path, content)
    source.chmod(0o640)

    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )

    destination = tmp_path / "artifacts" / "published.bin"
    assert result == (
        f"OK: materialized {len(content)} bytes to artifacts/published.bin"
    )
    assert destination.read_bytes() == content
    assert destination.stat().st_mode & 0o777 == 0o640
    assert not list(destination.parent.glob(".crucible-artifact-*"))


def test_materialize_artifact_reads_ignored_staging_from_active_worktree(
    tmp_path,
):
    repository = tmp_path / "repository"
    worktree = tmp_path / "worktree"
    repository.mkdir()
    worktree.mkdir()
    _staged_artifact(repository, b"stale repository artifact")
    _staged_artifact(worktree, b"current worktree artifact")
    resolver = _ActiveWorktreeResolver(repository, worktree)
    ctx = ToolContext(
        workspace_root=repository,
        repo_root=repository,
        write_scope=("artifacts",),
        path_resolver=resolver,
    )

    result = build_write_tools(ctx).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )

    assert result.startswith("OK: materialized")
    assert (worktree / "artifacts" / "published.bin").read_bytes() == (
        b"current worktree artifact"
    )
    assert not (repository / "artifacts" / "published.bin").exists()


def test_materialize_artifact_with_production_worktree_resolver(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "crucible@example.invalid")
    _git(repository, "config", "user.name", "Crucible Test")
    (repository / ".gitignore").write_text("target/\n", encoding="utf-8")
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "README.md")
    _git(repository, "commit", "-q", "-m", "fixture")

    resolver = WorktreeResolver(repository)
    worktree = resolver.ensure_worktree()
    try:
        content = b"ignored worktree artifact\x00\xff"
        _staged_artifact(worktree, content)
        ignored = _git(
            worktree,
            "check-ignore",
            "target/staging/artifact.bin",
        )
        assert ignored.stdout.strip() == "target/staging/artifact.bin"
        ctx = ToolContext(
            workspace_root=repository,
            repo_root=repository,
            write_scope=("artifacts",),
            path_resolver=resolver,
        )

        result = build_write_tools(ctx).dispatch(
            "materialize_artifact",
            {
                "source_path": "target/staging/artifact.bin",
                "destination_path": "artifacts/published.bin",
            },
        )

        assert result.startswith("OK: materialized")
        assert (worktree / "artifacts" / "published.bin").read_bytes() == content
        assert not (repository / "artifacts" / "published.bin").exists()
    finally:
        resolver.close()


def test_materialize_artifact_rejects_stale_repository_fallback(tmp_path):
    repository = tmp_path / "repository"
    worktree = tmp_path / "worktree"
    repository.mkdir()
    worktree.mkdir()
    _staged_artifact(repository, b"stale repository artifact")
    resolver = _ActiveWorktreeResolver(repository, worktree)
    ctx = ToolContext(
        workspace_root=repository,
        repo_root=repository,
        write_scope=("artifacts",),
        path_resolver=resolver,
    )

    result = build_write_tools(ctx).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )

    assert result.startswith("Error materializing")
    assert not (worktree / "artifacts" / "published.bin").exists()


def test_materialize_artifact_identical_destination_is_idempotent(tmp_path):
    content = b"already-there\x00"
    _staged_artifact(tmp_path, content)
    destination = tmp_path / "artifacts" / "published.bin"
    destination.parent.mkdir()
    destination.write_bytes(content)
    before = destination.stat()

    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )

    after = destination.stat()
    assert "already materialized" in result
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert not list(destination.parent.glob(".crucible-artifact-*"))


def test_materialize_artifact_mode_difference_is_not_idempotent(tmp_path):
    content = b"same bytes, different mode"
    source = _staged_artifact(tmp_path, content)
    source.chmod(0o640)
    destination = tmp_path / "artifacts" / "published.bin"
    destination.parent.mkdir()
    destination.write_bytes(content)
    destination.chmod(0o600)

    toolset = build_write_tools(_artifact_ctx(tmp_path))
    rejected = toolset.dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )
    assert "differs in bytes or mode" in rejected
    assert destination.stat().st_mode & 0o777 == 0o600

    replaced = toolset.dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
            "replace": True,
        },
    )
    assert replaced.startswith("OK: materialized")
    assert destination.read_bytes() == content
    assert destination.stat().st_mode & 0o777 == 0o640


def test_materialize_artifact_requires_explicit_replace_for_difference(tmp_path):
    _staged_artifact(tmp_path, b"new bytes")
    destination = tmp_path / "artifacts" / "published.bin"
    destination.parent.mkdir()
    destination.write_bytes(b"old bytes")

    toolset = build_write_tools(_artifact_ctx(tmp_path))
    rejected = toolset.dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )
    assert "destination differs" in rejected
    assert destination.read_bytes() == b"old bytes"

    replaced = toolset.dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
            "replace": True,
        },
    )
    assert replaced.startswith("OK: materialized")
    assert destination.read_bytes() == b"new bytes"


def test_materialize_artifact_replace_failure_keeps_old_destination(
    tmp_path, monkeypatch,
):
    _staged_artifact(tmp_path, b"new bytes")
    destination = tmp_path / "artifacts" / "published.bin"
    destination.parent.mkdir()
    destination.write_bytes(b"old bytes")

    def fail_replace(_source, _destination, **_kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(write_handlers.os, "replace", fail_replace)
    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
            "replace": True,
        },
    )

    assert "injected publication failure" in result
    assert destination.read_bytes() == b"old bytes"
    assert not list(destination.parent.glob(".crucible-artifact-*"))


def test_materialize_artifact_rolls_back_if_post_link_cleanup_fails(
    tmp_path, monkeypatch,
):
    _staged_artifact(tmp_path, b"payload")
    real_unlink = write_handlers.os.unlink
    failed_once = False

    def fail_first_temporary_unlink(path, *args, **kwargs):
        nonlocal failed_once
        if str(path).startswith(".crucible-artifact-") and not failed_once:
            failed_once = True
            raise OSError("injected temporary unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(write_handlers.os, "unlink", fail_first_temporary_unlink)
    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )

    destination = tmp_path / "artifacts" / "published.bin"
    assert "publication was rolled back" in result
    assert not destination.exists()
    assert not list(destination.parent.glob(".crucible-artifact-*"))


@pytest.mark.parametrize(
    ("source_path", "destination_path"),
    [
        ("../outside.bin", "artifacts/published.bin"),
        ("target/staging/artifact.bin", "../outside.bin"),
        ("target/staging/artifact.bin", "undeclared/published.bin"),
        (".git/config", "artifacts/published.bin"),
    ],
)
def test_materialize_artifact_rejects_escapes_and_unsafe_paths(
    tmp_path, source_path, destination_path,
):
    _staged_artifact(tmp_path, b"payload")
    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": source_path,
            "destination_path": destination_path,
        },
    )
    assert result.lower().startswith("error")
    assert not (tmp_path / "artifacts" / "published.bin").exists()


def test_materialize_artifact_rejects_source_symlink(tmp_path):
    actual = _staged_artifact(tmp_path, b"payload")
    alias = actual.with_name("alias.bin")
    alias.symlink_to(actual.name)
    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/alias.bin",
            "destination_path": "artifacts/published.bin",
        },
    )
    assert "unsafe non-private regular file" in result
    assert not (tmp_path / "artifacts" / "published.bin").exists()


def test_materialize_artifact_rejects_source_hardlink(tmp_path):
    source = _staged_artifact(tmp_path, b"payload")
    source.with_name("alias.bin").hardlink_to(source)
    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )
    assert "unsafe non-private regular file" in result
    assert not (tmp_path / "artifacts" / "published.bin").exists()


def test_materialize_artifact_rejects_destination_hardlink(tmp_path):
    _staged_artifact(tmp_path, b"new payload")
    destination = tmp_path / "artifacts" / "published.bin"
    destination.parent.mkdir()
    other = tmp_path / "other.bin"
    other.write_bytes(b"existing payload")
    destination.hardlink_to(other)
    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
            "replace": True,
        },
    )
    assert "cannot resolve write path" in result
    assert other.read_bytes() == b"existing payload"


def test_materialize_artifact_rejects_parent_symlink_swap(
    tmp_path, monkeypatch,
):
    _staged_artifact(tmp_path, b"payload")
    destination_parent = tmp_path / "artifacts"
    destination_parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    original_resolve = write_handlers.resolve_for_write

    def resolve_then_swap(ctx, relative):
        resolved = original_resolve(ctx, relative)
        destination_parent.rename(tmp_path / "former-artifacts")
        destination_parent.symlink_to(outside, target_is_directory=True)
        return resolved

    monkeypatch.setattr(
        write_handlers,
        "resolve_for_write",
        resolve_then_swap,
    )
    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )
    assert result.startswith("Error materializing")
    assert not (outside / "published.bin").exists()


def test_materialize_artifact_rejects_non_regular_source(tmp_path):
    directory = tmp_path / "target" / "staging" / "artifact.bin"
    directory.mkdir(parents=True)
    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )
    assert "unsafe non-private regular file" in result


def test_materialize_artifact_rejects_same_path(tmp_path):
    source = _staged_artifact(tmp_path, b"payload")
    ctx = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("target/staging/artifact.bin",),
    )
    result = build_write_tools(ctx).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "target/staging/artifact.bin",
            "replace": True,
        },
    )
    assert "must be different paths" in result
    assert source.read_bytes() == b"payload"


def test_materialize_artifact_rejects_oversized_source(tmp_path):
    source = _staged_artifact(tmp_path, b"")
    with source.open("r+b") as artifact:
        artifact.truncate(write_handlers._MAX_ARTIFACT_BYTES + 1)
    result = build_write_tools(_artifact_ctx(tmp_path)).dispatch(
        "materialize_artifact",
        {
            "source_path": "target/staging/artifact.bin",
            "destination_path": "artifacts/published.bin",
        },
    )
    assert "exceeds" in result
    assert not (tmp_path / "artifacts" / "published.bin").exists()


# ----------------------------------------------------------------------
# apply_patch — end-to-end
# ----------------------------------------------------------------------


def test_apply_patch_unique_match_replaces(tmp_path):
    (tmp_path / "f.py").write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
    ts = build_write_tools(_ctx(tmp_path))
    result = ts.dispatch("apply_patch", {
        "file_path": "f.py",
        "before": "def foo():\n    return 1",
        "after": "def foo():\n    return 42",
    })
    assert "OK" in result
    assert "return 42" in (tmp_path / "f.py").read_text()
    assert "return 1" not in (tmp_path / "f.py").read_text()


def test_apply_patch_missing_match_error(tmp_path):
    (tmp_path / "f.py").write_text("def foo(): return 1\n")
    ts = build_write_tools(_ctx(tmp_path))
    result = ts.dispatch("apply_patch", {
        "file_path": "f.py",
        "before": "nonexistent text",
        "after": "x",
    })
    assert "not found" in result
    assert "Re-read" in result


def test_apply_patch_multiple_matches_error(tmp_path):
    (tmp_path / "f.py").write_text("foo\nfoo\nfoo\n")
    ts = build_write_tools(_ctx(tmp_path))
    result = ts.dispatch("apply_patch", {
        "file_path": "f.py", "before": "foo", "after": "bar",
    })
    assert "matches 3 locations" in result


def test_apply_patch_rejects_noop_replacement(tmp_path):
    original = "value = 1\n"
    (tmp_path / "f.py").write_text(original)
    ts = build_write_tools(_ctx(tmp_path))

    result = ts.dispatch("apply_patch", {
        "file_path": "f.py",
        "before": original,
        "after": original,
    })

    assert "requires an actual edit" in result
    assert (tmp_path / "f.py").read_text() == original


def test_apply_patch_missing_file_error(tmp_path):
    ts = build_write_tools(_ctx(tmp_path))
    result = ts.dispatch("apply_patch", {
        "file_path": "nope.py", "before": "x", "after": "y",
    })
    assert "file not found" in result


def test_apply_patch_reports_line_counts(tmp_path):
    (tmp_path / "f.py").write_text("a\nb\nc\n")
    ts = build_write_tools(_ctx(tmp_path))
    result = ts.dispatch("apply_patch", {
        "file_path": "f.py", "before": "a\nb", "after": "X\nY\nZ",
    })
    assert "replaced 2 lines with 3 lines" in result
