# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows._mutation_scope.check_mutation_scope``.

Covers the four behaviours the docstring promises:
- Empty mutation_regions on the directive → no findings (no constraint).
- File without marker pair → no findings (per-file opt-in).
- Hunks fully inside markers → no findings.
- Hunks outside markers → High-severity Finding per hunk.

Uses a tmp_path-backed git repo + Corpus so the helper exercises real
``git show`` + ``corpus.get_artifact`` round-trips.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from assay import (
    ContentHash,
    ExecutionContext,
    Implementation,
    ImplementationDirective,
    MutationRegion,
    ProjectContext,
    Spec,
)
from assay.corpus.store import Corpus

from crucible.workflows._mutation_scope import (
    check_mutation_scope,
    render_violations,
)


def _exec_ctx() -> ExecutionContext:
    return ExecutionContext(
        timestamp="2026-05-20T00:00:00+00:00",
        machine="testhost", cwd="/tmp", toolchain={},
        backend=None, model=None, recipe=None, recipe_commit=None,
        gpu_model=None, run_id=None,
    )


def _spec_with(corpus: Corpus) -> Spec:
    spec = Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="t", repo_url="x", base_commit="abc",
            branch=None, submodule_path=None, runners={},
        ),
        execution=None, intent="", constraints=(), done_criteria=(),
        plan_policy=None,
    )
    corpus.put_spec(spec)
    return spec


def _init_repo(path: Path) -> str:
    """Init a git repo at path. Returns the resulting HEAD sha."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"], cwd=path, check=True,
    )
    return ""


def _commit(path: Path, msg: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", msg], cwd=path, check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return sha


def _put_impl(
    corpus: Corpus, slice_id: str, diff_bytes: bytes, base_commit: str,
) -> Implementation:
    return Implementation(
        slice_id=slice_id,
        base_commit=base_commit,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            diff_bytes, kind="diffs", name_hint=f"{slice_id}.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint=f"{slice_id}.json",
        ),
        files_modified=("foo.py",),
        scope_escapes=(),
        classification="success",
        execution=_exec_ctx(),
    )


def _directive(
    files_in_scope: tuple[str, ...] = ("foo.py",),
    mutation_regions: tuple[MutationRegion, ...] = (),
) -> ImplementationDirective:
    return ImplementationDirective(
        slice_id="s0", intent="t",
        files_in_scope=files_in_scope, creates_paths=(),
        verify_mode="build_only", blast_radius={},
        mutation_regions=mutation_regions,
    )


def test_empty_mutation_regions_yields_no_findings(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _spec_with(corpus)
    impl = _put_impl(corpus, "s0", b"", "abc")
    findings = check_mutation_scope(
        _directive(mutation_regions=()),
        impl, corpus=corpus, cwd=str(tmp_path),
    )
    assert findings == ()


def test_file_without_markers_skips_silently(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "foo.py").write_text("def hi(): return 1\n")
    base = _commit(repo, "init")

    corpus = Corpus(tmp_path / "c")
    _spec_with(corpus)
    # Diff edits foo.py line 1, but foo.py has NO marker pair → no
    # constraint to enforce.
    diff = (
        b"diff --git a/foo.py b/foo.py\n"
        b"--- a/foo.py\n"
        b"+++ b/foo.py\n"
        b"@@ -1,1 +1,1 @@\n"
        b"-def hi(): return 1\n"
        b"+def hi(): return 2\n"
    )
    impl = _put_impl(corpus, "s0", diff, base)
    findings = check_mutation_scope(
        _directive(mutation_regions=(MutationRegion(file="foo.py"),)),
        impl, corpus=corpus, cwd=str(repo),
    )
    assert findings == ()


def test_hunk_inside_markers_passes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    src = (
        "def setup():\n"          # line 1
        "    pass\n"               # line 2
        "# CRUCIBLE-MUTATE-START\n"  # line 3
        "def hot_path(x):\n"      # line 4
        "    return x * 2\n"      # line 5
        "# CRUCIBLE-MUTATE-END\n"   # line 6
        "def teardown():\n"        # line 7
        "    pass\n"               # line 8
    )
    (repo / "foo.py").write_text(src)
    base = _commit(repo, "init")

    corpus = Corpus(tmp_path / "c")
    _spec_with(corpus)
    # Edit line 5 (inside markers at lines 3-6).
    diff = (
        b"diff --git a/foo.py b/foo.py\n"
        b"--- a/foo.py\n"
        b"+++ b/foo.py\n"
        b"@@ -5,1 +5,1 @@\n"
        b"-    return x * 2\n"
        b"+    return x * 4\n"
    )
    impl = _put_impl(corpus, "s0", diff, base)
    findings = check_mutation_scope(
        _directive(mutation_regions=(MutationRegion(file="foo.py"),)),
        impl, corpus=corpus, cwd=str(repo),
    )
    assert findings == ()


def test_hunk_outside_markers_emits_finding(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    src = (
        "def setup():\n"          # line 1
        "    pass\n"               # line 2
        "# CRUCIBLE-MUTATE-START\n"  # line 3
        "def hot_path(x):\n"      # line 4
        "    return x * 2\n"      # line 5
        "# CRUCIBLE-MUTATE-END\n"   # line 6
        "def teardown():\n"        # line 7
        "    pass\n"               # line 8
    )
    (repo / "foo.py").write_text(src)
    base = _commit(repo, "init")

    corpus = Corpus(tmp_path / "c")
    _spec_with(corpus)
    # Edit line 8 (outside markers).
    diff = (
        b"diff --git a/foo.py b/foo.py\n"
        b"--- a/foo.py\n"
        b"+++ b/foo.py\n"
        b"@@ -8,1 +8,1 @@\n"
        b"-    pass\n"
        b"+    return 42\n"
    )
    impl = _put_impl(corpus, "s0", diff, base)
    findings = check_mutation_scope(
        _directive(mutation_regions=(MutationRegion(file="foo.py"),)),
        impl, corpus=corpus, cwd=str(repo),
    )
    assert len(findings) == 1
    assert findings[0].severity == "High"
    assert findings[0].category == "mutation_scope"
    assert findings[0].source == "deterministic"
    assert "foo.py:8" in findings[0].evidence[0].location


def test_rust_comment_prefix_recognised(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    src = (
        "fn setup() {}\n"               # line 1
        "// CRUCIBLE-MUTATE-START\n"     # line 2
        "fn hot_path(x: i32) -> i32 {\n"  # line 3
        "    x * 2\n"                     # line 4
        "}\n"                              # line 5
        "// CRUCIBLE-MUTATE-END\n"        # line 6
        "fn teardown() {}\n"              # line 7
    )
    (repo / "lib.rs").write_text(src)
    base = _commit(repo, "init")

    corpus = Corpus(tmp_path / "c")
    _spec_with(corpus)
    # Edit line 7 (outside markers).
    diff = (
        b"diff --git a/lib.rs b/lib.rs\n"
        b"--- a/lib.rs\n"
        b"+++ b/lib.rs\n"
        b"@@ -7,1 +7,1 @@\n"
        b"-fn teardown() {}\n"
        b"+fn teardown() { unsafe {} }\n"
    )
    impl = _put_impl(corpus, "s0", diff, base)
    findings = check_mutation_scope(
        _directive(
            files_in_scope=("lib.rs",),
            mutation_regions=(MutationRegion(file="lib.rs"),),
        ),
        impl, corpus=corpus, cwd=str(repo),
    )
    assert len(findings) == 1
    assert "lib.rs:7" in findings[0].evidence[0].location


def test_custom_markers_honored(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    src = (
        "# OPTIM-START\n"   # line 1
        "x = 1\n"            # line 2
        "# OPTIM-END\n"     # line 3
        "y = 2\n"            # line 4
    )
    (repo / "foo.py").write_text(src)
    base = _commit(repo, "init")

    corpus = Corpus(tmp_path / "c")
    _spec_with(corpus)
    # Edit line 4 (outside the OPTIM markers).
    diff = (
        b"diff --git a/foo.py b/foo.py\n"
        b"--- a/foo.py\n"
        b"+++ b/foo.py\n"
        b"@@ -4,1 +4,1 @@\n"
        b"-y = 2\n"
        b"+y = 3\n"
    )
    impl = _put_impl(corpus, "s0", diff, base)
    findings = check_mutation_scope(
        _directive(
            mutation_regions=(
                MutationRegion(
                    file="foo.py",
                    start_marker="OPTIM-START",
                    end_marker="OPTIM-END",
                ),
            ),
        ),
        impl, corpus=corpus, cwd=str(repo),
    )
    assert len(findings) == 1


def test_creates_paths_target_skips_silently(tmp_path: Path):
    """A directive declaring mutation_regions for a file that doesn't
    exist at base_commit (e.g., a creates_paths target) must skip
    silently — there are no markers to enforce against."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "existing.py").write_text("# placeholder\n")
    base = _commit(repo, "init")

    corpus = Corpus(tmp_path / "c")
    _spec_with(corpus)
    diff = (
        b"diff --git a/new.py b/new.py\n"
        b"--- /dev/null\n"
        b"+++ b/new.py\n"
        b"@@ -0,0 +1,1 @@\n"
        b"+def f(): pass\n"
    )
    impl = _put_impl(corpus, "s0", diff, base)
    findings = check_mutation_scope(
        _directive(mutation_regions=(MutationRegion(file="new.py"),)),
        impl, corpus=corpus, cwd=str(repo),
    )
    assert findings == ()


def test_render_violations_handles_empty():
    assert render_violations(()) == ""


def test_render_violations_summarizes():
    from assay import Evidence, Finding
    findings = (
        Finding(
            id="x", severity="High", category="mutation_scope",
            evidence=(Evidence(kind="diff_hunk", location="foo.py:8-8", excerpt="x"),),
            suggested_action="a", source="deterministic",
        ),
    )
    out = render_violations(findings)
    assert "1 violation" in out
    assert "foo.py:8-8" in out


def test_file_at_base_commit_refuses_unsafe_pathspec(tmp_path, caplog):
    """CRU-08: a planner-authored region path containing ':' (or a
    leading '-' / NUL) must not be fed to ``git show <rev>:<path>`` —
    it would corrupt the rev-parse. The helper returns None (treated as
    'no markers to enforce') and logs, rather than invoking git."""
    import logging
    from crucible.workflows._mutation_scope import _file_at_base_commit

    with caplog.at_level(logging.WARNING):
        for bad in ("foo:bar.py", "-rf", "a\x00b.py"):
            assert _file_at_base_commit(
                bad, base_commit="HEAD", cwd=str(tmp_path)
            ) is None
    assert "refusing unsafe region path" in caplog.text
