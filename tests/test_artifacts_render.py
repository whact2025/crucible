# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows._artifacts.render_prior_artifacts``.

Covers the four behaviours the docstring promises:
- Returns empty string when no records carry matching-kind artifacts.
- Renders ``## Previous Execution Feedback`` header + per-artifact
  fenced blocks when records have populated ``artifacts`` dicts.
- Honors ``max_bytes`` line-granular tail truncation.
- Routes blobs through ``redact_secrets`` when ``security_filter=True``.

Uses a tmp_path-backed Corpus so the round-trip ``put_artifact`` →
``get_artifact`` exercises the real persistence layer.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from assay import (
    ContentHash,
    ExecutionContext,
    Implementation,
    ProjectContext,
    Spec,
)
from assay.corpus.store import Corpus

from crucible.workflows._artifacts import (
    DEFAULT_ARTIFACT_KINDS,
    render_prior_artifacts,
)


def _exec_ctx() -> ExecutionContext:
    return ExecutionContext(
        timestamp="2026-05-20T00:00:00+00:00",
        machine="testhost",
        cwd="/tmp",
        toolchain={},
        backend=None,
        model=None,
        recipe=None,
        recipe_commit=None,
        gpu_model=None,
        run_id=None,
    )


def _bare_spec(corpus: Corpus) -> Spec:
    spec = Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="t", repo_url="x", base_commit="abc",
            branch=None, submodule_path=None, runners={},
        ),
        execution=None,
        intent="", constraints=(), done_criteria=(),
        plan_policy=None,
    )
    corpus.put_spec(spec)
    return spec


def _make_impl(corpus: Corpus, slice_id: str,
               artifact_kinds_and_blobs: dict[str, bytes]) -> Implementation:
    """Build an Implementation whose artifacts dict refs real corpus blobs."""
    refs: dict[str, ContentHash] = {}
    for kind, blob in artifact_kinds_and_blobs.items():
        refs[kind] = corpus.put_artifact(
            blob, kind="logs", name_hint=f"{slice_id}_{kind}.log",
        )
    return Implementation(
        slice_id=slice_id,
        base_commit="abc",
        result_commit=None,
        diff_ref=corpus.put_artifact(b"", kind="diffs", name_hint=f"{slice_id}.diff"),
        transcript_ref=corpus.put_artifact(b"[]", kind="transcripts",
                                            name_hint=f"{slice_id}.json"),
        files_modified=(),
        scope_escapes=(),
        classification="success",
        execution=_exec_ctx(),
        artifacts=refs,
    )


def test_empty_inputs_return_empty_string(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    assert render_prior_artifacts(corpus) == ""
    assert render_prior_artifacts(
        corpus, implementations=(), verdicts=(),
    ) == ""


def test_impl_without_matching_kind_returns_empty(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    impl = _make_impl(corpus, "s0", {"unrelated_kind": b"hello"})
    assert render_prior_artifacts(
        corpus, implementations=(impl,),
        kinds=("verify_stderr",),
    ) == ""


def test_single_impl_renders_header_and_fenced_block(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    impl = _make_impl(corpus, "s0", {
        "verify_stderr": b"FAILED tests/t.py::test_x -- AssertionError",
    })
    out = render_prior_artifacts(
        corpus, implementations=(impl,),
        kinds=("verify_stderr",),
    )
    assert "## Previous Execution Feedback" in out
    assert "verify_stderr" in out
    assert "FAILED tests/t.py::test_x" in out
    assert "s0" in out


def test_multiple_impls_render_in_most_recent_first(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    older = _make_impl(corpus, "s_older", {
        "verify_stderr": b"OLDER ERROR",
    })
    newer = _make_impl(corpus, "s_newer", {
        "verify_stderr": b"NEWER ERROR",
    })
    out = render_prior_artifacts(
        corpus, implementations=(older, newer),
        kinds=("verify_stderr",),
    )
    # Most recent is rendered first so prompt-tail-truncated readers
    # still see the freshest context.
    assert out.index("NEWER ERROR") < out.index("OLDER ERROR")


def test_max_lookback_caps_records(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    impls = tuple(
        _make_impl(corpus, f"s{i}", {"verify_stderr": f"ERR_{i}".encode()})
        for i in range(5)
    )
    out = render_prior_artifacts(
        corpus, implementations=impls,
        kinds=("verify_stderr",),
        max_lookback=2,
    )
    # Only the last 2 (s3, s4) get rendered.
    assert "ERR_4" in out
    assert "ERR_3" in out
    assert "ERR_2" not in out
    assert "ERR_1" not in out
    assert "ERR_0" not in out


def test_max_bytes_truncates_line_granular(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    long_blob = ("line\n" * 1000).encode("utf-8")  # 5 KB
    impl = _make_impl(corpus, "s0", {"verify_stderr": long_blob})
    out = render_prior_artifacts(
        corpus, implementations=(impl,),
        kinds=("verify_stderr",),
        max_bytes=512,
    )
    assert len(out.encode("utf-8")) < 1500   # 512 budget + headers/fence
    # Truncation marker present.
    assert "earlier lines truncated" in out


def test_security_filter_redacts_token_in_blob(tmp_path: Path):
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    blob = b"build failed; token ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA leaked"
    impl = _make_impl(corpus, "s0", {"verify_stderr": blob})
    out = render_prior_artifacts(
        corpus, implementations=(impl,),
        kinds=("verify_stderr",),
        security_filter=True,
    )
    assert "ghp_AAAAAA" not in out
    assert "<REDACTED>" in out


def test_security_filter_off_passes_blob_through(tmp_path: Path):
    """For internal-only renders the workflow can opt out — confirm it
    actually skips the filter rather than silently always-on."""
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    blob = b"build failed; token ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA leaked"
    impl = _make_impl(corpus, "s0", {"verify_stderr": blob})
    out = render_prior_artifacts(
        corpus, implementations=(impl,),
        kinds=("verify_stderr",),
        security_filter=False,
    )
    assert "ghp_AAAA" in out


def test_default_kinds_set_matches_module_constant():
    """Sanity: the public DEFAULT_ARTIFACT_KINDS reflects the docstring."""
    assert "verify_stderr" in DEFAULT_ARTIFACT_KINDS
    assert "stderr" in DEFAULT_ARTIFACT_KINDS
    assert "tool_output" in DEFAULT_ARTIFACT_KINDS


def test_missing_corpus_ref_is_skipped_silently(tmp_path: Path):
    """A stale or mis-routed ContentHash should not crash render —
    just skip and continue."""
    corpus = Corpus(tmp_path / "c")
    _bare_spec(corpus)
    impl = Implementation(
        slice_id="s0", base_commit="a", result_commit=None,
        diff_ref=ContentHash("sha256:00"),
        transcript_ref=ContentHash("sha256:00"),
        files_modified=(), classification="success",
        scope_escapes=(),
        execution=_exec_ctx(),
        artifacts={"verify_stderr": ContentHash("sha256:" + "f" * 64)},
    )
    out = render_prior_artifacts(
        corpus, implementations=(impl,),
        kinds=("verify_stderr",),
    )
    assert out == ""  # nothing renderable → empty section
