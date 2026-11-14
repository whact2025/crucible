# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``run_tier(impl_system_prompt=...)`` reaches the implementer.

Re-pins the contract that Area F (commit a7218a9, 2026-05-21) dropped
when it moved the SWE-bench worker onto the
``run_architect → run_tier → implement()`` cascade and stopped
forwarding ``system_prompt=``. Without the forward, the issue-fixing
implementer (swebench / deepswe) silently ran under assay's default
``_IMPLEMENTER_SYSTEM_PROMPT`` instead of the benchmark-container contract
that says to inspect visible tests, edit early, and let the workflow execute
tests deterministically.

These tests drive ``run_tier`` with a recording-stub backend that
captures every ``Invocation`` and asserts on ``invocation.system``:

- ``impl_system_prompt="SENTINEL"`` ⇒ the implementer's Invocation
  carries exactly ``"SENTINEL"``.
- default (no kwarg) ⇒ the implementer's Invocation carries Assay prompt ABI
  v2 with the concrete Crucible ToolSet metadata.

The stub returns a benign empty-diff result; we assert on the captured
system prompt, not on the cascade outcome.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from assay import (
    InvocationResult,
    ProjectContext,
    Spec,
    Usage,
)
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG


class _RecordingBackend:
    """Backend that records every Invocation and returns a benign
    empty-text result so the implement phase completes normally."""

    name = "stub-recording"
    model = "stub-model"

    def __init__(self):
        self.invocations: list = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=Usage(input_tokens=0, output_tokens=0, cost_usd=0.0),
            stop_reason="end_turn",
            error=None,
        )


def _init_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=path, check=True,
    )
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
            intent="slice 0 intent",
            files_in_scope=("file_0.py",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    return Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="tier_impl_prompt_test",
            repo_url="https://example.invalid/tt.git",
            base_commit=base_commit,
            branch=None, submodule_path=None, runners={},
        ),
        execution=None,
        intent="exercise impl_system_prompt forwarding",
        constraints=(), done_criteria=(), plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=(slice_,), depends_on=()),),
        ),
        implementation_refs=(), verdict_refs=(),
        measurement_refs=(), lifecycle_event_refs=(),
    )


def _run(tmp_path: Path, monkeypatch, **run_tier_kwargs):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)

    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    Corpus(corpus_dir).put_spec(_spec_with_one_slice_tier(base))

    backend = _RecordingBackend()
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kw: backend,
    )

    from crucible.workflows.tier import run_tier

    run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
        **run_tier_kwargs,
    )
    return backend


def test_run_tier_forwards_impl_system_prompt(tmp_path, monkeypatch):
    """A non-None ``impl_system_prompt`` reaches the implementer's
    Invocation verbatim — this is the wiring Area F dropped."""
    backend = _run(
        tmp_path, monkeypatch, impl_system_prompt="SENTINEL_PROMPT_123",
    )
    assert backend.invocations, "implementer backend was never invoked"
    systems = {inv.system for inv in backend.invocations}
    assert "SENTINEL_PROMPT_123" in systems, (
        f"impl_system_prompt did not reach the Invocation; "
        f"captured systems={systems!r}"
    )


def test_run_tier_default_uses_assay_implementer_prompt(tmp_path, monkeypatch):
    """Default renders Assay prompt ABI v2 with the attached ToolSet."""
    backend = _run(tmp_path, monkeypatch)
    assert backend.invocations, "implementer backend was never invoked"
    systems = {inv.system for inv in backend.invocations}
    assert any("assay.prompt.v2" in system for system in systems)
    assert any("`read_source`" in system for system in systems)
    assert any("`write_source`" in system for system in systems)
    assert all("25K-token" not in system for system in systems)


def test_swebench_prompt_matches_container_execution_boundary():
    from crucible.products.swebench.prompts import SWEBENCH_SYSTEM_PROMPT

    assert "not a shell or test-execution tool" in SWEBENCH_SYSTEM_PROMPT
    assert "Crucible runs the visible" in SWEBENCH_SYSTEM_PROMPT
    assert "Do not claim to have run tests" in SWEBENCH_SYSTEM_PROMPT
    assert "agent SHOULD run tests" not in SWEBENCH_SYSTEM_PROMPT
