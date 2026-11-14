# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.tools.source`` ToolSet."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from assay import ToolSet
from crucible.tools import ToolContext
from crucible.tools.source import build_source_tools


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _ctx(repo: Path) -> ToolContext:
    return ToolContext(workspace_root=repo, repo_root=repo)


@pytest.fixture
def repo_with_files(tmp_path: Path) -> Path:
    """A tmp 'repo' containing a small Python file + a subdirectory."""
    (tmp_path / "demo.py").write_text(
        "def hello():\n    return 'hi'\n\ndef other():\n    return hello()\n"
    )
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "foo.py").write_text("from .. import demo\n\nprint(demo.hello())\n")
    (sub / "bar.cpp").write_text("// uses hello()\nvoid bar() { hello(); }\n")
    return tmp_path


# ----------------------------------------------------------------------
# Protocol conformance + schema shape
# ----------------------------------------------------------------------


def test_source_toolset_satisfies_protocol(tmp_path):
    ts = build_source_tools(_ctx(tmp_path))
    assert isinstance(ts, ToolSet)


def test_source_emits_expected_tools(tmp_path):
    ts = build_source_tools(_ctx(tmp_path))
    assert ts.tool_names == {
        "read_source",
        "grep_symbol",
        "find_callers",
        "find_definition",
    }


def test_source_all_tools_read_only(tmp_path):
    ts = build_source_tools(_ctx(tmp_path))
    assert ts.read_only_names == ts.tool_names
    assert ts.write_names == set()


def test_source_anthropic_shape_well_formed(tmp_path):
    ts = build_source_tools(_ctx(tmp_path))
    out = ts.to_anthropic_tools()
    by_name = {e["name"]: e for e in out}
    assert "file_path" in by_name["read_source"]["input_schema"]["properties"]
    assert "pattern" in by_name["grep_symbol"]["input_schema"]["properties"]
    assert "function_name" in by_name["find_callers"]["input_schema"]["properties"]


def test_source_openai_shape_well_formed(tmp_path):
    ts = build_source_tools(_ctx(tmp_path))
    out = ts.to_openai_tools()
    assert all(e["type"] == "function" for e in out)
    by_name = {e["function"]["name"]: e for e in out}
    assert "read_source" in by_name


# ----------------------------------------------------------------------
# read_source
# ----------------------------------------------------------------------


def test_read_source_basic(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("read_source", {"file_path": "demo.py"})
    assert "def hello" in out
    assert "1 |" in out  # line number prefix
    assert "# file_sha256:" in out


def test_read_source_line_range(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("read_source", {
        "file_path": "demo.py", "start_line": 3, "end_line": 4,
    })
    assert "def other" in out
    assert "def hello" not in out
    assert "# file_sha256:" not in out
    assert "# partial_file:" in out


def test_read_source_capped_output_withholds_whole_file_hash(
    tmp_path,
    monkeypatch,
):
    from crucible.tools.source import handlers

    cap = 160
    monkeypatch.setattr(handlers, "_MAX_OUTPUT_CHARS", cap)
    (tmp_path / "long.py").write_text("x" * 300 + "\n", encoding="utf-8")
    ts = build_source_tools(_ctx(tmp_path))

    out = ts.dispatch("read_source", {"file_path": "long.py"})

    assert out.startswith("# partial_file:")
    assert "whole-file hash withheld" in out
    assert "# file_sha256:" not in out
    assert len(out) <= cap


def test_read_source_repo_name_prefix_stripped(repo_with_files):
    """Agent passing 'repo/demo.py' when repo_root is already at 'repo' should resolve."""
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("read_source", {
        "file_path": f"{repo_with_files.name}/demo.py",
    })
    assert "def hello" in out


@pytest.mark.parametrize("container_workdir", ["/app", "/testbed"])
def test_read_source_normalizes_container_workdir_absolute_alias(
    repo_with_files,
    container_workdir,
):
    ctx = ToolContext(
        workspace_root=repo_with_files,
        repo_root=repo_with_files,
        container_id="container-id",
        container_workdir=container_workdir,
    )
    out = build_source_tools(ctx).dispatch("read_source", {
        "file_path": f"{container_workdir}/demo.py",
    })

    assert "def hello" in out
    assert "outside the repository" not in out


def test_read_source_outside_repo_rejected(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("read_source", {"file_path": "../etc/passwd"})
    assert "outside the repository" in out


def test_read_source_vcs_metadata_rejected(repo_with_files):
    (repo_with_files / ".git").mkdir()
    (repo_with_files / ".git" / "config").write_text("secret")
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("read_source", {"file_path": ".git/config"})
    assert "outside the repository" in out


def test_read_source_missing_file(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("read_source", {"file_path": "does_not_exist.py"})
    assert "file not found" in out


# ----------------------------------------------------------------------
# grep_symbol
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_grep_symbol_finds_matches(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("grep_symbol", {"pattern": "hello"})
    assert "hello" in out
    # Match lines include line numbers from rg/grep.
    assert ":" in out
    assert str(repo_with_files) not in out


@pytest.mark.parametrize("directory", [".", "./"])
@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_grep_symbol_accepts_repository_root_alias(
    repo_with_files,
    directory,
):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("grep_symbol", {
        "pattern": "hello",
        "directory": directory,
    })
    assert "def hello" in out
    assert "outside the repository" not in out


def test_grep_symbol_rejects_short_pattern(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("grep_symbol", {"pattern": "x"})
    assert "at least 2 characters" in out


def test_grep_symbol_no_matches_message(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("grep_symbol", {"pattern": "ZZZ_no_such_token_ZZZ"})
    assert "No matches" in out


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_grep_symbol_accepts_leading_dash_pattern(repo_with_files):
    (repo_with_files / "src" / "pointer.cpp").write_text(
        "value->field = 1;\n",
        encoding="utf-8",
    )
    out = build_source_tools(_ctx(repo_with_files)).dispatch(
        "grep_symbol",
        {"pattern": "->field"},
    )
    assert "src/pointer.cpp:" in out
    assert "->field" in out


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_grep_symbol_reports_invalid_regex(repo_with_files):
    out = build_source_tools(_ctx(repo_with_files)).dispatch(
        "grep_symbol",
        {"pattern": "[invalid"},
    )
    assert out.startswith("Error: search failed:")


def test_grep_symbol_replaces_invalid_subprocess_bytes_and_keeps_output_cap(
    repo_with_files,
    monkeypatch,
):
    from crucible.tools.source import handlers

    payload = (
        str(repo_with_files).encode()
        + b"/src/foo.py:1:ok\xff"
        + b"x" * 100
    )
    script = (
        "import os; "
        f"os.write(1, {payload!r}); "
        "os.write(2, b'ignored\\xffstderr')"
    )
    monkeypatch.setattr(
        handlers,
        "_grep_cmd",
        lambda *_args, **_kwargs: [sys.executable, "-c", script],
    )
    monkeypatch.setattr(handlers, "_MAX_OUTPUT_CHARS", 40)

    out = build_source_tools(_ctx(repo_with_files)).dispatch(
        "grep_symbol",
        {"pattern": "ok"},
    )

    assert "�" in out
    assert str(repo_with_files) not in out
    assert out.startswith("src/foo.py:1:ok")
    assert len(out) == 40


def test_grep_symbol_replaces_invalid_error_bytes(repo_with_files, monkeypatch):
    from crucible.tools.source import handlers

    script = "import os; os.write(2, b'bad\\xffregex'); raise SystemExit(2)"
    monkeypatch.setattr(
        handlers,
        "_grep_cmd",
        lambda *_args, **_kwargs: [sys.executable, "-c", script],
    )

    out = build_source_tools(_ctx(repo_with_files)).dispatch(
        "grep_symbol",
        {"pattern": "ok"},
    )

    assert out == "Error: search failed: bad�regex"


def test_grep_symbol_rejects_directory_outside_repo(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("grep_symbol", {
        "pattern": "hello", "directory": "../",
    })
    assert "outside the repository" in out


def test_grep_symbol_rejects_vcs_metadata_directory(repo_with_files):
    (repo_with_files / ".git").mkdir()
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("grep_symbol", {
        "pattern": "secret", "directory": ".git",
    })
    assert "outside the repository" in out


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_grep_symbol_directory_scope(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("grep_symbol", {"pattern": "hello", "directory": "src"})
    # demo.py is at repo root, not in src/ — should not match it.
    assert "src/foo.py" in out or "src/bar.cpp" in out
    assert "demo.py" not in out


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_grep_symbol_accepts_file_scope_and_returns_repo_relative_path(
    repo_with_files,
):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("grep_symbol", {
        "pattern": "hello",
        "directory": "src/foo.py",
    })
    assert "hello" in out
    assert str(repo_with_files) not in out
    assert "src/foo.py:" in out


# ----------------------------------------------------------------------
# find_callers
# ----------------------------------------------------------------------


def test_find_callers_rejects_short_name(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("find_callers", {"function_name": "ab"})
    assert "at least 3 characters" in out


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_find_callers_finds_call_sites(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("find_callers", {"function_name": "hello"})
    assert "bar.cpp" in out
    assert "demo.py" in out


# ----------------------------------------------------------------------
# find_callers language-awareness
# ----------------------------------------------------------------------


def test_find_callers_schema_advertises_known_languages(tmp_path):
    """The languages enum should include rust, python, cpp, ts, etc."""
    ts = build_source_tools(_ctx(tmp_path))
    spec = next(e for e in ts.to_anthropic_tools() if e["name"] == "find_callers")
    enum = spec["input_schema"]["properties"]["languages"]["items"]["enum"]
    for lang in ("python", "rust", "cpp", "typescript", "go", "java"):
        assert lang in enum, f"missing {lang!r} from languages enum"


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_find_callers_rust_only_skips_cpp(repo_with_files):
    """languages=['rust'] should miss bar.cpp and only see .rs files."""
    (repo_with_files / "src" / "lib.rs").write_text("pub fn caller() { hello(); }\n")
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("find_callers", {
        "function_name": "hello",
        "languages": ["rust"],
    })
    assert "bar.cpp" not in out
    assert "lib.rs" in out


def test_languages_to_glob_helper():
    """Combine multiple languages into one brace glob."""
    from crucible.tools.source.handlers import _languages_to_glob
    glob = _languages_to_glob(["rust", "python"])
    assert "rs" in glob
    assert "py" in glob
    assert glob.startswith("*.{")
    assert glob.endswith("}")


def test_languages_to_glob_unknown_language_skipped():
    """Asking for a language we don't have in the registry yields no extensions."""
    from crucible.tools.source.handlers import _languages_to_glob
    assert _languages_to_glob(["nope_not_a_language"]) == ""


# ----------------------------------------------------------------------
# find_definition
# ----------------------------------------------------------------------


def test_find_definition_rejects_short_symbol(tmp_path):
    ts = build_source_tools(_ctx(tmp_path))
    out = ts.dispatch("find_definition", {"symbol": "ab", "language": "cpp"})
    assert "at least 3 characters" in out


def test_find_definition_rejects_unknown_language(tmp_path):
    ts = build_source_tools(_ctx(tmp_path))
    out = ts.dispatch("find_definition", {"symbol": "Foo", "language": "cobol"})
    assert "unsupported language" in out


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_find_definition_cpp_class(tmp_path):
    """class Foo declaration is found in a C++ header."""
    (tmp_path / "ns.h").write_text(
        "// some comment\n"
        "namespace omni::usd::hydra {\n"
        "class IFoo : public Base {\n"
        "public:\n"
        "    virtual void doit() = 0;\n"
        "};\n"
        "}\n"
    )
    ts = build_source_tools(_ctx(tmp_path))
    out = ts.dispatch("find_definition", {"symbol": "IFoo", "language": "cpp"})
    assert "ns.h" in out
    assert "class IFoo" in out


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_find_definition_cpp_function_excludes_call_sites(tmp_path):
    """The function pattern must NOT match call sites — only declarations."""
    (tmp_path / "a.cpp").write_text(
        "void other() {\n"
        "    publishFrameRenderState();        // CALL — must not match\n"
        "    publishFrameRenderState(42);      // CALL — must not match\n"
        "}\n"
        "\n"
        "void publishFrameRenderState(int x) { /* defn */ }\n"
    )
    (tmp_path / "a.h").write_text(
        "class Engine {\n"
        "public:\n"
        "    void publishFrameRenderState();   // decl\n"
        "};\n"
    )
    ts = build_source_tools(_ctx(tmp_path))
    out = ts.dispatch("find_definition", {
        "symbol": "publishFrameRenderState", "language": "cpp",
    })
    # Expect both decl (a.h) and defn (a.cpp def line) — NOT the two call lines.
    assert "a.h" in out and "publishFrameRenderState()" in out
    assert "void publishFrameRenderState(int x)" in out
    # Call lines start with whitespace then the bare symbol; they must not
    # appear by themselves.
    for line in out.splitlines():
        # A captured line that is exactly "    publishFrameRenderState();"
        # (after the file:line: prefix) would be a call-site false positive.
        if line.endswith("    publishFrameRenderState();") or line.endswith(
            "    publishFrameRenderState(42);"
        ):
            pytest.fail(f"call-site leaked into find_definition output: {line!r}")


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_find_definition_strips_namespace_qualifier(tmp_path):
    """omni::usd::hydra::IFoo resolves the same as bare IFoo."""
    (tmp_path / "ns.h").write_text(
        "namespace omni::usd::hydra {\nclass IFoo {};\n}\n"
    )
    ts = build_source_tools(_ctx(tmp_path))
    bare = ts.dispatch("find_definition", {"symbol": "IFoo", "language": "cpp"})
    qualified = ts.dispatch("find_definition", {
        "symbol": "omni::usd::hydra::IFoo", "language": "cpp",
    })
    assert bare == qualified


@pytest.mark.skipif(
    shutil.which("rg") is None and shutil.which("grep") is None,
    reason="neither ripgrep nor grep available",
)
def test_find_definition_python_def(tmp_path):
    """def foo: is found in Python; an indented assignment is NOT (module-level only)."""
    (tmp_path / "m.py").write_text(
        "FOO = 1\n"
        "\n"
        "def calculate(x):\n"
        "    FOO = 2  # local — must not match the assignment pattern\n"
        "    return x\n"
        "\n"
        "class Widget:\n"
        "    pass\n"
    )
    ts = build_source_tools(_ctx(tmp_path))
    out_fn = ts.dispatch("find_definition", {"symbol": "calculate", "language": "python"})
    assert "m.py" in out_fn
    assert "def calculate" in out_fn

    out_cls = ts.dispatch("find_definition", {"symbol": "Widget", "language": "python"})
    assert "class Widget" in out_cls

    # Module-level `FOO = 1` matches; the indented `FOO = 2` must not.
    out_const = ts.dispatch("find_definition", {"symbol": "FOO", "language": "python"})
    assert "FOO = 1" in out_const
    for line in out_const.splitlines():
        if line.endswith("    FOO = 2  # local — must not match the assignment pattern"):
            pytest.fail(f"indented assignment leaked: {line!r}")


def test_find_definition_schema_lists_languages(tmp_path):
    """find_definition's language enum exposes the supported languages."""
    ts = build_source_tools(_ctx(tmp_path))
    spec = next(e for e in ts.to_anthropic_tools() if e["name"] == "find_definition")
    enum = spec["input_schema"]["properties"]["language"]["enum"]
    for lang in ("cpp", "python", "rust", "java", "go", "c"):
        assert lang in enum, f"missing {lang!r} from language enum"


# ----------------------------------------------------------------------
# Unknown tool routing
# ----------------------------------------------------------------------


def test_unknown_tool_returns_error_string(repo_with_files):
    ts = build_source_tools(_ctx(repo_with_files))
    out = ts.dispatch("not_a_real_tool", {})
    assert out.startswith("ERROR: unknown tool")
