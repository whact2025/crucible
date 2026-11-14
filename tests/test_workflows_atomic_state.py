# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""State-transition tests for standalone Crucible workflows."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from assay import ImplementationDirective, Slice, Spec, Tier, TierDAG
from assay.canonical import hash_record
from assay.corpus.store import Corpus
from assay.types.artifacts import Implementation, Measurement
from assay.types.execution import ExecutionContext
from assay.types.identity import ContentHash

from crucible.workflows._atomic_persistence import (
    commit_staged_spec,
    persist_and_attach,
)


MINIMAL_YAML = """
schema_version: 1
project_context:
  project: atomic-state-test
  repo_url: https://example.invalid/x.git
  base_commit: abc123
  branch: main
  submodule_path: null
  runners: {}
intent: Exercise workflow state transitions.
constraints: []
done_criteria: []
"""


def _execution(recipe: str) -> ExecutionContext:
    return ExecutionContext(
        timestamp="2026-06-28T12:00:00+00:00",
        machine="test",
        cwd="/tmp",
        toolchain={},
        backend=None,
        model=None,
        recipe=recipe,
        recipe_commit=None,
        gpu_model=None,
        run_id=None,
    )


def _measurement(bench_id: str) -> Measurement:
    return Measurement(
        bench_id=bench_id,
        commit="abc123",
        config_hash=f"cfg-{bench_id}",
        values={"latency_ms": 1.0},
        raw_output_ref=ContentHash("sha256:" + "1" * 64),
        execution=_execution("bench"),
    )


def _slice(slice_id: str, files: tuple[str, ...], creates: tuple[str, ...] = ()):
    return Slice(
        id=slice_id,
        directive=ImplementationDirective(
            slice_id=slice_id,
            intent=f"implement {slice_id}",
            files_in_scope=files,
            creates_paths=creates,
            verify_mode="build_only",
            blast_radius={},
        ),
    )


def _spec_with_dag() -> Spec:
    base = Spec.from_yaml(MINIMAL_YAML)
    return base.with_tier_dag(TierDAG(tiers=(
        Tier(
            tier_id="dependency",
            depends_on=(),
            slices=(_slice("dep", ("dep.cc",), ("generated/api.h",)),),
        ),
        Tier(
            tier_id="target",
            depends_on=("dependency",),
            slices=(
                _slice("s0", ("src/a.py",), ("src/new.py",)),
                _slice("s1", ("src/b.py",)),
            ),
        ),
    )))


def _corpus(tmp_path: Path, *, dag: bool = False) -> Corpus:
    corpus = Corpus(tmp_path / "corpus")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    if dag:
        corpus.put_spec(base.with_tier_dag(_spec_with_dag().tier_dag))
    return corpus


def _latest_summary(corpus: Corpus) -> dict:
    summaries = list((Path(corpus.spec_dir) / "runs").glob("*/summary.json"))
    assert summaries
    return json.loads(max(summaries, key=lambda path: path.stat().st_mtime_ns).read_text())


def test_persist_and_attach_rebases_after_an_earlier_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    corpus = _corpus(tmp_path)
    current = _measurement("current")
    concurrent = _measurement("concurrent")
    real_commit = corpus.commit
    injected = False
    concurrent_ref = None

    def racing_commit(**kwargs):
        nonlocal concurrent_ref, injected
        if not injected:
            injected = True
            result = real_commit(measurements=(concurrent,))
            concurrent_ref = result.refs.measurement_refs[0]
        return real_commit(**kwargs)

    monkeypatch.setattr(corpus, "commit", racing_commit)
    committed, current_ref = persist_and_attach(
        corpus,
        current,
        kind="measurements",
        attachment="measurement",
    )

    assert concurrent_ref is not None
    assert committed == corpus.latest_spec()
    assert set(committed.measurement_refs) == {concurrent_ref, current_ref}

    # Content-addressed retries do not create duplicate refs or Spec versions.
    same_spec, same_ref = persist_and_attach(
        corpus,
        current,
        kind="measurements",
        attachment="measurement",
    )
    assert same_ref == current_ref
    assert same_spec.version == committed.version
    assert same_spec.measurement_refs.count(current_ref) == 1


def test_commit_staged_spec_rebases_refs_and_structural_state(tmp_path: Path):
    corpus = _corpus(tmp_path)
    base = corpus.latest_spec()
    assert base is not None

    staged_measurement = _measurement("staged")
    staged_ref = corpus.put_record(staged_measurement, kind="measurements")
    staged = base.with_appended_refs(
        measurement_refs=(staged_ref,),
        tier_dag=_spec_with_dag().tier_dag,
    )

    # A different writer advances the manifest after the draft was built.
    concurrent = corpus.commit(measurements=(_measurement("concurrent"),))
    committed = commit_staged_spec(
        corpus,
        base=base,
        staged=staged,
        structural_fields=("tier_dag",),
    )

    assert committed == corpus.latest_spec()
    assert committed.tier_dag == staged.tier_dag
    assert set(committed.measurement_refs) == {
        staged_ref,
        concurrent.refs.measurement_refs[0],
    }


def test_commit_staged_spec_rejects_rewritten_ref_prefix(tmp_path: Path):
    corpus = _corpus(tmp_path)
    first = corpus.commit(measurements=(_measurement("first"),)).spec
    replacement_ref = corpus.put_record(
        _measurement("replacement"), kind="measurements",
    )
    rewritten = first.with_appended_refs()
    rewritten = dataclasses.replace(
        rewritten,
        measurement_refs=(replacement_ref,),
    )

    with pytest.raises(ValueError, match="rewrote the existing"):
        commit_staged_spec(corpus, base=first, staged=rewritten)


def test_review_scope_is_slice_local_and_resolves_implementation_ref(tmp_path: Path):
    from crucible.workflows.review import _scope_context

    corpus = _corpus(tmp_path, dag=True)
    diff_ref = corpus.put_artifact(b"diff", kind="diffs")
    transcript_ref = corpus.put_artifact(b"transcript", kind="transcripts")
    impl = Implementation(
        slice_id="s0",
        base_commit="abc123",
        result_commit=None,
        diff_ref=diff_ref,
        transcript_ref=transcript_ref,
        files_modified=("src/a.py",),
        scope_escapes=(),
        classification="success",
        execution=_execution("implement"),
    )
    persist_and_attach(
        corpus,
        impl,
        kind="implementations",
        attachment="implementation",
    )

    slice_id, context = _scope_context(
        corpus,
        slice_id=None,
        artifact_ref=diff_ref,
    )
    assert slice_id == "s0"
    assert context["files_in_scope"] == "src/a.py"
    assert context["creates_paths"] == "src/new.py"
    assert "src/b.py" not in context["files_in_scope"]
    assert context["dependency_provided_files"] == "generated/api.h"
    assert context["directive_intent"] == "implement s0"

    with pytest.raises(SystemExit, match="needs --slice-id"):
        _scope_context(corpus, slice_id=None, artifact_ref=None)


def test_review_scope_uses_explicit_slice_for_shared_empty_diff_ref(tmp_path: Path):
    from crucible.workflows.review import _scope_context

    corpus = _corpus(tmp_path, dag=True)
    empty_ref = corpus.put_artifact(b"", kind="diffs")
    transcript_ref = corpus.put_artifact(b"transcript", kind="transcripts")
    for slice_id in ("s0", "s1"):
        persist_and_attach(
            corpus,
            Implementation(
                slice_id=slice_id,
                base_commit="abc123",
                result_commit=None,
                diff_ref=empty_ref,
                transcript_ref=transcript_ref,
                files_modified=(),
                scope_escapes=(),
                classification="soft_fail_no_changes",
                execution=_execution("handoff"),
            ),
            kind="implementations",
            attachment="implementation",
        )

    slice_id, context = _scope_context(
        corpus,
        slice_id="s0",
        artifact_ref=empty_ref,
    )
    assert slice_id == "s0"
    assert context["files_in_scope"] == "src/a.py"
    assert "src/b.py" not in context["files_in_scope"]
    assert context["directive_intent"] == "implement s0"

    with pytest.raises(SystemExit, match="attached to multiple slices"):
        _scope_context(corpus, slice_id=None, artifact_ref=empty_ref)


def test_review_rejects_untrusted_generated_path_scope_override(
    tmp_path: Path,
):
    from crucible.workflows.review import _parse_args

    with pytest.raises(SystemExit) as exc_info:
        _parse_args([
            "--spec",
            str(tmp_path),
            "--rubric",
            "diff_scope",
            "--artifact",
            str(tmp_path / "candidate.diff"),
            "--workflow-generated-path",
            "sensitive.py",
        ])
    assert exc_info.value.code == 2


def test_implement_persists_attempt_even_when_not_successful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.workflows import implement as module

    corpus = _corpus(tmp_path, dag=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=workspace, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=workspace, check=True,
    )
    (workspace / "README").write_text("base\n")
    subprocess.run(["git", "add", "README"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=workspace, check=True)
    monkeypatch.chdir(workspace)
    diff_ref = corpus.put_artifact(b"", kind="diffs")
    transcript_ref = corpus.put_artifact(b"agent", kind="transcripts")
    attempt = Implementation(
        slice_id="s0",
        base_commit="abc123",
        result_commit=None,
        diff_ref=diff_ref,
        transcript_ref=transcript_ref,
        files_modified=(),
        scope_escapes=(),
        classification="hard_fail_transport",
        execution=_execution("implement"),
    )

    class Backend:
        name = "fake"
        model = "fake-model"

    pool_calls = []

    class Pool:
        def get(self, *args, **kwargs):
            pool_calls.append((args, kwargs))
            return Backend()

    @contextmanager
    def fake_pool(*args, **kwargs):
        yield Pool()

    monkeypatch.setattr(module, "backend_pool", fake_pool)
    monkeypatch.setattr(module, "implement", lambda *args, **kwargs: attempt)

    assert module.main(["--spec", str(corpus.spec_dir), "--slice", "s0"]) == 1
    assert pool_calls[0][1]["native_shell"] is False
    latest = corpus.latest_spec()
    assert len(latest.implementation_refs) == 1
    restored = corpus.get_record(latest.implementation_refs[0], as_type=Implementation)
    assert restored.classification == "hard_fail_transport"
    assert _latest_summary(corpus)["implementation_ref"] == latest.implementation_refs[0]


def test_review_persists_verdict_and_attaches_latest_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from assay.types.findings import ReviewCoverage, Verdict
    from crucible.workflows import review as module

    corpus = _corpus(tmp_path, dag=True)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("review me")
    verdict = Verdict(
        rubric="tier_completeness",
        decision="pass",
        findings=(),
        artifact_ref=ContentHash("sha256:" + "2" * 64),
        artifact_commit="abc123",
        reasoning="complete",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
            artifact_chars_total=9,
            artifact_chars_reviewed=9,
        ),
        backend_name="fake",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
        kind="review",
    )

    class Backend:
        name = "fake"
        model = "fake-model"

    class Pool:
        def get(self, *args, **kwargs):
            return Backend()

    @contextmanager
    def fake_pool(*args, **kwargs):
        yield Pool()

    monkeypatch.setattr(module, "backend_pool", fake_pool)
    monkeypatch.setattr(module, "load_rubric", lambda name: object())
    monkeypatch.setattr(module, "review", lambda *args, **kwargs: verdict)

    assert module.main([
        "--spec", str(corpus.spec_dir),
        "--rubric", "tier_completeness",
        "--artifact", str(artifact),
    ]) == 0
    latest = corpus.latest_spec()
    assert len(latest.verdict_refs) == 1
    restored = corpus.get_record(latest.verdict_refs[0], as_type=Verdict)
    assert restored.decision == "pass"
    summary = _latest_summary(corpus)
    assert summary["verdict_ref"] == latest.verdict_refs[0]
    assert summary["spec_version"] == latest.version


def test_diff_scope_review_rejects_concurrent_tier_dag_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from assay.types.findings import ReviewCoverage, Verdict
    from crucible.workflows import review as module

    corpus = _corpus(tmp_path, dag=True)
    artifact = tmp_path / "candidate.diff"
    artifact.write_text("diff --git a/src/a.py b/src/a.py\n")
    verdict = Verdict(
        rubric="diff_scope",
        decision="pass",
        findings=(),
        artifact_ref=ContentHash("sha256:" + "3" * 64),
        artifact_commit="abc123",
        reasoning="within the original slice scope",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
            artifact_chars_total=len(artifact.read_text()),
            artifact_chars_reviewed=len(artifact.read_text()),
        ),
        backend_name="fake",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
        kind="review",
    )
    replacement_dag = TierDAG(tiers=(
        Tier(
            tier_id="replacement",
            depends_on=(),
            slices=(_slice("replacement", ("replacement.py",)),),
        ),
    ))

    class Backend:
        name = "fake"
        model = "fake-model"

    class Pool:
        def get(self, *args, **kwargs):
            return Backend()

    @contextmanager
    def fake_pool(*args, **kwargs):
        yield Pool()

    def review_after_structure_change(*args, **kwargs):
        corpus.commit(
            transform=lambda candidate: dataclasses.replace(
                candidate,
                tier_dag=replacement_dag,
            )
        )
        return verdict

    monkeypatch.setattr(module, "backend_pool", fake_pool)
    monkeypatch.setattr(module, "load_rubric", lambda name: object())
    monkeypatch.setattr(module, "review", review_after_structure_change)

    assert module.main([
        "--spec", str(corpus.spec_dir),
        "--rubric", "diff_scope",
        "--artifact", str(artifact),
        "--slice-id", "s0",
    ]) == 3

    latest = corpus.latest_spec()
    assert latest.tier_dag == replacement_dag
    assert latest.verdict_refs == ()
    summary = _latest_summary(corpus)
    assert summary["status"] == "concurrent_spec_change"
    assert summary["error"] == (
        "Spec execution authority changed during review"
    )
    assert summary["original_decision"] == "pass"


def test_validate_persists_complete_pure_rule_verdict(tmp_path: Path):
    from assay.types.findings import Verdict
    from crucible.workflows import validate as module

    corpus = _corpus(tmp_path, dag=True)
    assert module.main([
        "--spec", str(corpus.spec_dir),
        "--rules", "cycle",
    ]) == 0

    latest = corpus.latest_spec()
    assert len(latest.verdict_refs) == 1
    verdict = corpus.get_record(latest.verdict_refs[0], as_type=Verdict)
    assert verdict.rubric == "default"
    summary = _latest_summary(corpus)
    assert summary["run_rule_ids"] == ["cycle"]
    assert summary["finding_count"] == 0
    assert summary["finding_counts"] == {
        "total": 0,
        "blocking": 0,
        "advisory": 0,
    }
    assert summary["verdict_ref"] == ContentHash(hash_record(verdict))


def test_validate_rejects_missing_context_without_publishing_verdict(
    tmp_path: Path,
):
    from crucible.workflows import validate as module

    corpus = _corpus(tmp_path, dag=True)
    assert module.main([
        "--spec", str(corpus.spec_dir),
        "--rules", "cycle,cited_path_exists",
    ]) == 2

    assert corpus.latest_spec().verdict_refs == ()
    summary = _latest_summary(corpus)
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "missing_rule_context"
    assert summary["context_rule_ids"] == ["cited_path_exists"]
