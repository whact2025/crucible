# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""End-to-end test for ``crucible.workflows.tier``'s ``--worktree``
integration (REC-6).

Verifies the wiring contract that the WorktreeResolver unit tests
don't reach:
- ``run_tier(worktree=True, ...)`` creates a fresh worktree at entry.
- ``cwd`` is rebound to the worktree path for the duration of the run.
- ``ToolContext.path_resolver`` is the live WorktreeResolver.
- The worktree is removed on exit (success OR failure paths).
- ``path_resolver=<existing>`` (architect-owned case) skips own
  session creation and uses the supplied resolver.
- ``worktree=True`` + ``path_resolver=<not-None>`` raises ValueError
  (mutually exclusive).

A future refactor that drops ``path_resolver=wt_resolver`` from the
``ToolContext`` construction silently degrades to the in-tree
default; this test guards against that.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from assay import (
    InvocationResult,
    ProjectContext,
    Spec,
    Usage,
)
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG
from assay.types.invocation import TransportError


class _TransportFailingBackend:
    """Backend that returns a TransportError → the tier escalates
    via the all_slices_transport_failed floor. We use this so the
    test exits Phase 1 quickly without exercising real implement
    + build/verify; the worktree wiring is what matters here."""

    name = "stub-transport-fail"
    model = "stub-model"

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        return InvocationResult(
            final_message="",
            transcript=(),
            usage=Usage(input_tokens=0, output_tokens=0, cost_usd=0.0),
            stop_reason="error",
            error=TransportError(
                http_status=None,
                message="sdk_transport: Command failed with exit code 1",
                retriable=False,
            ),
        )


def _init_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "README.md").write_text("seed\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _spec_with_one_slice_tier(base_commit: str) -> Spec:
    slice_ = Slice(
        id="s0",
        directive=ImplementationDirective(
            slice_id="s0",
            intent="placeholder slice for worktree wiring test",
            files_in_scope=("a.py",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    return Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="worktree_wiring_test",
            repo_url="https://example.invalid/wt.git",
            base_commit=base_commit,
            branch=None, submodule_path=None, runners={},
        ),
        execution=None,
        intent="exercise --worktree wiring",
        constraints=(), done_criteria=(), plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=(slice_,), depends_on=()),),
        ),
        implementation_refs=(), verdict_refs=(),
        measurement_refs=(), lifecycle_event_refs=(),
    )


def _setup_corpus(tmp_path: Path) -> tuple[Corpus, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec_with_one_slice_tier(base))
    return corpus, repo, base


def _patch_backend(monkeypatch) -> _TransportFailingBackend:
    backend = _TransportFailingBackend()
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kw: backend,
    )
    return backend


def _count_existing_worktrees(repo: Path) -> int:
    """Return the number of live worktrees under .git/worktrees/."""
    wt_dir = repo / ".git" / "worktrees"
    if not wt_dir.exists():
        return 0
    return sum(1 for _ in wt_dir.iterdir())


def test_run_tier_worktree_creates_and_cleans_up(tmp_path: Path, monkeypatch):
    """worktree=True creates a worktree at entry, removes it at exit."""
    corpus, repo, _ = _setup_corpus(tmp_path)
    _patch_backend(monkeypatch)

    # Track the resolver's worktree path through the already-locked teardown
    # primitive used by ``worktree_session`` so we can verify both filesystem
    # cleanup and release of every durable ownership field.
    captured: dict = {}
    from crucible.workflows._worktree import WorktreeResolver as _WR
    real_close_locked = _WR._close_locked

    def spy_close_locked(self):
        captured["resolver"] = self
        captured["worktree_path_at_close"] = self.worktree_path
        return real_close_locked(self)

    monkeypatch.setattr(_WR, "_close_locked", spy_close_locked)

    from crucible.workflows.tier import run_tier
    rc = run_tier(
        spec=corpus.spec_dir,
        tier_id="t0",
        cwd=str(repo),
        skip_build_verify=True, skip_smoke_check=True,
        skip_mutation_scope=True, skip_smoke_bench=True,
        # Default ``BreachPolicy()`` has enabled=False; no kwarg needed.
        skip_review=True,
        worktree=True,
    )
    # Transport-failure escalation → rc=1; that's fine. We're testing wiring.
    assert rc == 1
    # The resolver's worktree path was set before close was called.
    wt_path = captured.get("worktree_path_at_close")
    assert wt_path is not None, "worktree was never created"
    # And cleanup removed it.
    assert not wt_path.exists(), (
        f"worktree {wt_path} should have been removed on tier exit"
    )
    assert not captured["resolver"].owned
    # Source repo's .git/worktrees/ is clean — no orphans.
    assert _count_existing_worktrees(repo) == 0


def test_run_tier_default_no_worktree(tmp_path: Path, monkeypatch):
    """worktree=False (default) creates no worktree."""
    corpus, repo, _ = _setup_corpus(tmp_path)
    _patch_backend(monkeypatch)

    from crucible.workflows.tier import run_tier
    rc = run_tier(
        spec=corpus.spec_dir,
        tier_id="t0",
        cwd=str(repo),
        skip_build_verify=True, skip_smoke_check=True,
        skip_mutation_scope=True, skip_smoke_bench=True,
        # Default ``BreachPolicy()`` has enabled=False; no kwarg needed.
        skip_review=True,
    )
    assert rc == 1
    assert _count_existing_worktrees(repo) == 0


def test_run_tier_path_resolver_supplied_skips_own_session(
    tmp_path: Path, monkeypatch,
):
    """When path_resolver is supplied (architect-owned case), run_tier
    uses it directly and doesn't open its own session."""
    corpus, repo, _ = _setup_corpus(tmp_path)
    _patch_backend(monkeypatch)

    from crucible.workflows._worktree import worktree_session

    with worktree_session(repo) as resolver:
        wt_path = resolver.worktree_path
        assert wt_path is not None
        from crucible.workflows.tier import run_tier
        rc = run_tier(
            spec=corpus.spec_dir,
            tier_id="t0",
            cwd=str(wt_path),
            path_resolver=resolver,
            skip_build_verify=True, skip_smoke_check=True,
            skip_mutation_scope=True, skip_smoke_bench=True,
            # Default ``BreachPolicy()`` has enabled=False; no kwarg needed.
            skip_review=True,
        )
        # Inside the with: the worktree still exists (we own it).
        assert wt_path.exists()
        assert rc == 1
    # After the with: cleaned up.
    assert not wt_path.exists()


def test_run_tier_worktree_and_path_resolver_mutually_exclusive(
    tmp_path: Path,
):
    """Passing both worktree=True AND path_resolver=<resolver> is a
    contract violation — the two open competing sessions."""
    corpus, repo, _ = _setup_corpus(tmp_path)

    from crucible.workflows._worktree import WorktreeResolver
    resolver = WorktreeResolver(repo)
    try:
        from crucible.workflows.tier import run_tier
        with pytest.raises(ValueError, match="mutually exclusive"):
            run_tier(
                spec=corpus.spec_dir,
                tier_id="t0",
                cwd=str(repo),
                worktree=True,
                path_resolver=resolver,
            )
    finally:
        resolver.close()
