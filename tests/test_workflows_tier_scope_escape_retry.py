# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Scope-escape recovery in ``crucible.workflows.tier``.

When the implementer writes files OUTSIDE its declared ``files_in_scope``,
``assay.implement`` tags the result ``hard_fail_scope_escape``, which
otherwise rolls back + halts the tier. The recovery loop reverts ONLY the
out-of-scope writes (keeping the in-scope diff), tells the implementer
which files it must not touch, and re-implements in place — up to
``build_retries`` times.

Behavioral e2e: a stub backend that writes an out-of-scope file on the
first implement (→ scope escape) and an in-scope file on the retry. With
``build_retries=1`` the tier must recover to ``rc=0``; with
``build_retries=0`` the same backend must halt at ``rc=1`` (no recovery).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    InvocationResult,
    ProjectContext,
    Spec,
    Usage,
)
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG

from crucible.common.policies import BreachPolicy


_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_TRACE = BackendExecutionTrace(
    backend="stub-scope-escape",
    requested_model="stub-model",
    events=(
        BackendExecutionEvent(
            sequence=0,
            kind="terminal",
            name="end_turn",
        ),
    ),
)


class _ScopeEscapeThenInScopeBackend:
    """First implement writes an OUT-of-scope file (→ hard_fail_scope_escape);
    the retry writes only the IN-scope file (→ success)."""

    name = "stub-scope-escape"
    model = "stub-model"

    def __init__(self, repo: Path, events: list[str]):
        self.repo = repo
        self.implement_calls = 0
        self.events = events
        self.closed = False

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        self.closed = False

    def close(self) -> None:
        self.events.append("backend_close")
        self.closed = True

    def invoke(self, invocation) -> InvocationResult:
        self.implement_calls += 1
        self.events.append(f"invoke_{self.implement_calls}")
        if self.implement_calls == 1:
            # Out-of-scope write → assay flags scope_escapes=[file_other.py].
            (self.repo / "file_other.py").write_text("# out of scope\n")
        else:
            # In-scope write only (the escape was reverted before this).
            (self.repo / "file_0.py").write_text(
                f"# in scope, attempt {self.implement_calls}\n"
            )
        return InvocationResult(
            final_message="done", transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


def _init_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "README.md").write_text("seed\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _spec_single_slice(base_commit: str) -> Spec:
    sl = Slice(
        id="s0",
        directive=ImplementationDirective(
            slice_id="s0",
            intent="slice zero work",
            files_in_scope=("file_0.py",),
            creates_paths=("file_0.py",),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    return Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="tier_scope_escape_test",
            repo_url="https://example.invalid/se.git",
            base_commit=base_commit,
            branch=None, submodule_path=None, runners={},
        ),
        execution=None,
        intent="exercise scope-escape recovery",
        constraints=(), done_criteria=(), plan_policy=None, depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=(sl,), depends_on=()),),
        ),
        implementation_refs=(), verdict_refs=(),
        measurement_refs=(), lifecycle_event_refs=(),
    )


def _run(tmp_path: Path, monkeypatch, build_retries: int):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    Corpus(corpus_dir).put_spec(_spec_single_slice(base))

    events: list[str] = []
    backend = _ScopeEscapeThenInScopeBackend(repo, events)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *a, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kw: backend,
    )
    from crucible.workflows._mutation_journal import MutationTransaction

    real_prepare_capture = MutationTransaction.prepare_git_capture
    real_restore_paths = MutationTransaction.restore_paths_from_snapshot
    real_commit = MutationTransaction.commit

    def observed_prepare_capture(self, *args, **kwargs):
        assert backend.closed
        if events and events[-1] == "backend_close":
            events.append("prepare_capture")
        return real_prepare_capture(self, *args, **kwargs)

    def observed_restore_paths(self, *args, **kwargs):
        assert backend.closed
        events.append("restore_paths")
        return real_restore_paths(self, *args, **kwargs)

    def observed_commit(self, *args, **kwargs):
        assert backend.closed
        events.append(f"commit:{self.slice_id}")
        return real_commit(self, *args, **kwargs)

    monkeypatch.setattr(
        MutationTransaction,
        "restore_paths_from_snapshot",
        observed_restore_paths,
    )
    monkeypatch.setattr(
        MutationTransaction,
        "prepare_git_capture",
        observed_prepare_capture,
    )
    monkeypatch.setattr(MutationTransaction, "commit", observed_commit)
    from crucible.workflows.tier import run_tier

    rc = run_tier(
        spec=corpus_dir, tier_id="t0", cwd=str(repo),
        skip_build_verify=True, skip_smoke_check=True,
        skip_mutation_scope=True, skip_smoke_bench=True, skip_review=True,
        breach_policy=BreachPolicy(enabled=False),
        build_retries=build_retries,
    )
    runs = [r for r in (corpus_dir / "runs").iterdir() if "tier" in r.name]
    summary = json.loads((runs[-1] / "summary.json").read_text())
    return rc, summary, backend, repo, events


def test_scope_escape_recovers_to_rc0(tmp_path, monkeypatch):
    """build_retries=1: first implement escapes scope, the escape is
    reverted and the slice re-implemented in scope → tier lands (rc=0),
    and the out-of-scope file is gone."""
    rc, summary, backend, repo, events = _run(
        tmp_path,
        monkeypatch,
        build_retries=1,
    )
    assert rc == 0, f"expected recovery to rc=0; status={summary.get('status')}"
    assert summary["status"] == "complete"
    assert backend.implement_calls == 2  # initial escape + one recovery
    # The out-of-scope write was reverted; the in-scope file survives.
    assert not (repo / "file_other.py").exists()
    assert (repo / "file_0.py").exists()
    assert events == [
        "invoke_1",
        "backend_close",
        "prepare_capture",
        "restore_paths",
        "invoke_2",
        "backend_close",
        "prepare_capture",
        "commit:s0__scope_retry_1",
        "commit:s0",
        "commit:s0",
    ]


def test_no_retry_budget_halts_on_scope_escape(tmp_path, monkeypatch):
    """build_retries=0: no recovery — the scope escape halts the tier
    (rc=1) after a single implement."""
    rc, summary, backend, repo, events = _run(
        tmp_path,
        monkeypatch,
        build_retries=0,
    )
    assert rc == 1
    assert backend.implement_calls == 1  # no re-implement
    assert events[0:3] == [
        "invoke_1",
        "backend_close",
        "prepare_capture",
    ]
