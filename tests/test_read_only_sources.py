# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Source evidence stays bounded, grounded and outside mutation authority."""

from dataclasses import asdict, replace
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from assay import ImplementationDirective, ReadOnlySourceReference, RuleContext, Slice, Tier, TierDAG
from assay.corpus.store import Corpus

from crucible.common import read_only_sources as sources
from crucible.rules.read_only_references import read_only_references
from crucible.rules.scope_covers_intent import scope_covers_intent
from crucible.workflows import plan as plan_workflow
from tests.test_workflows_plan_candidate_replacement import replacement  # noqa: F401


def _reference(path="src/evidence.py", blob=b"first\r\nsecond\nlast"):
    return ReadOnlySourceReference(
        reference_id="evidence", path=path,
        sha256="sha256:" + hashlib.sha256(blob).hexdigest(),
        start_line=2, end_line=3,
    )


def _dag(*references, writes=(), intent="Use evidence for the existing behavior"):
    return TierDAG(tiers=(Tier(tier_id="t0", depends_on=(), slices=(Slice(
        id="s0", directive=ImplementationDirective(
            slice_id="s0", intent=intent, files_in_scope=writes,
            creates_paths=(), verify_mode="tests", blast_radius={},
            read_only_references=references,
        ),
    ),)),))


def _write(tmp_path, blob=b"first\r\nsecond\nlast"):
    source = tmp_path / "src/evidence.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(blob)
    return source


def test_capture_binds_whole_file_preserves_lines_and_retains_exact_context(tmp_path):
    _write(tmp_path)
    dag = _dag(_reference())
    captured = sources.capture_read_only_sources(dag, workspace_root=tmp_path)
    entry = json.loads(captured)["references"][0]
    assert entry == {**asdict(_reference()), "tier_id": "t0", "slice_id": "s0", "excerpt": "second\nlast"}
    corpus = Corpus(tmp_path / "corpus")
    context = sources.source_context_provider(corpus=corpus, workspace_root=tmp_path)(dag)
    digest = hashlib.sha256(captured).hexdigest()
    assert context["read_only_sources"] == f"Captured context sha256:{digest}\n{captured.decode()}"
    assert (corpus.spec_dir / "artifacts/plan-source-context" / f"{digest}.bin").read_bytes() == captured
    assert dag.tiers[0].slices[0].directive.files_in_scope == ()


def test_source_change_after_capture_fails_final_rule(tmp_path):
    source = _write(tmp_path)
    spec = SimpleNamespace(tier_dag=_dag(_reference()))
    ctx = RuleContext(workspace_root=tmp_path)
    assert read_only_references(spec, ctx) == ()
    source.write_text("changed\nsecond\nlast")
    findings = read_only_references(spec, ctx)
    assert len(findings) == 1 and findings[0].severity == "High"
    assert "digest" in findings[0].evidence[0].excerpt


@pytest.mark.parametrize("alias", ["src", "src/evidence.py", "src/", "./src", "src//evidence.py", "src\\evidence.py", "../src"])
def test_read_evidence_never_masks_same_slice_write_authority(tmp_path, alias):
    _write(tmp_path)
    with pytest.raises(sources.SourceReferenceError):
        sources.capture_read_only_sources(_dag(_reference(), writes=(alias,)), workspace_root=tmp_path)


def test_redundant_project_prefix_scope_is_rejected(tmp_path):
    _write(tmp_path)
    with pytest.raises(sources.SourceReferenceError, match="canonical"):
        sources.capture_read_only_sources(
            _dag(_reference(), writes=(tmp_path.name + "/src/evidence.py",)), workspace_root=tmp_path,
        )


def test_readonly_field_does_not_waive_mutation_intent_scan(tmp_path):
    _write(tmp_path)
    spec = SimpleNamespace(tier_dag=_dag(_reference(), intent="Modify src/evidence.py"))
    assert scope_covers_intent(spec, RuleContext(workspace_root=tmp_path))[0].severity == "Critical"


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "submodule_symlink", "hardlink", "fifo"])
def test_aliased_and_nonregular_sources_are_rejected_without_reads(tmp_path, kind):
    source = _write(tmp_path / "real")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    target = project / "src/evidence.py"
    submodule = None
    if kind == "file_symlink":
        target.symlink_to(source)
    elif kind == "directory_symlink":
        (project / "src").rmdir()
        (project / "src").symlink_to(source.parent, target_is_directory=True)
    elif kind == "submodule_symlink":
        (project / "nested").symlink_to(tmp_path / "real", target_is_directory=True)
        submodule = "nested"
    elif kind == "hardlink":
        os.link(source, target)
    else:
        os.mkfifo(target)
    with pytest.raises(sources.SourceReferenceError):
        sources.capture_read_only_sources(_dag(_reference()), workspace_root=project, submodule_path=submodule)


def test_nested_real_submodule_is_composed_once(tmp_path):
    _write(tmp_path / "nested/project")
    captured = sources.capture_read_only_sources(_dag(_reference()), workspace_root=tmp_path, submodule_path="nested/project")
    assert json.loads(captured)["references"][0]["path"] == "src/evidence.py"


@pytest.mark.parametrize("path", [".git/config", "AGENTS.md", "nested/CLAUDE.md", ".env.local", ".ssh/id_rsa"])
def test_governed_metadata_and_instruction_files_are_unavailable(tmp_path, path):
    ref = replace(_reference(), path=path)
    with pytest.raises(sources.SourceReferenceError):
        sources.capture_read_only_sources(_dag(ref), workspace_root=tmp_path)


@pytest.mark.parametrize("blob", [b"\xff\nsecond\nlast", b"\x00\nsecond\nlast"])
def test_binary_source_is_not_reviewer_context(tmp_path, blob):
    _write(tmp_path, blob)
    with pytest.raises(sources.SourceReferenceError):
        sources.capture_read_only_sources(_dag(_reference(blob=blob)), workspace_root=tmp_path)


@pytest.mark.parametrize("limit", ["MAX_FILE_BYTES", "MAX_EXCERPT_BYTES", "MAX_CONTEXT_BYTES", "MAX_REFERENCES"])
def test_byte_and_count_limits_fail_closed(tmp_path, monkeypatch, limit):
    _write(tmp_path)
    monkeypatch.setattr(sources, limit, 0)
    with pytest.raises(sources.SourceReferenceError, match="limit"):
        sources.capture_read_only_sources(_dag(_reference()), workspace_root=tmp_path)


def test_out_of_range_excerpt_is_rejected(tmp_path):
    _write(tmp_path)
    with pytest.raises(sources.SourceReferenceError, match="line range"):
        sources.capture_read_only_sources(_dag(replace(_reference(), end_line=4)), workspace_root=tmp_path)


def test_missing_workspace_fails_only_when_source_evidence_is_declared():
    assert json.loads(sources.capture_read_only_sources(_dag(), workspace_root=None))["references"] == []
    with pytest.raises(sources.SourceReferenceError, match="explicit workspace"):
        sources.capture_read_only_sources(_dag(_reference()), workspace_root=None)


def test_fixed_candidate_closed_import_preserves_reference_and_legacy_hash():
    from assay.canonical import hash_record
    from assay.primitives.plan import _serialize_tier_dag

    dag = _dag(_reference())
    payload = json.loads(_serialize_tier_dag(dag))
    payload["tiers"][0]["slices"][0]["directive"]["blast_radius"] = {"max_files": 0, "languages": []}
    parsed = plan_workflow._strict_review_candidate_dag(payload)
    assert parsed.tiers[0].slices[0].directive.read_only_references == (_reference(),)
    directive = payload["tiers"][0]["slices"][0]["directive"]
    directive["read_only_references"][0]["extra"] = True
    with pytest.raises(ValueError, match="unknown"):
        plan_workflow._strict_review_candidate_dag(payload)
    del directive["read_only_references"]
    legacy = plan_workflow._strict_review_candidate_dag(payload)
    directive["read_only_references"] = []
    assert hash_record(legacy) == hash_record(plan_workflow._strict_review_candidate_dag(payload))


@pytest.mark.parametrize("stage,expected_rc", [("stable", 0), ("before_review", 3), ("during_review", 1)])
def test_source_snapshot_is_retained_and_revalidated_before_publication(
    replacement, monkeypatch, stage, expected_rc,
):
    from assay import ExecutionCompleteness, ReviewCoverage, Verdict

    corpus, base, candidate, args = replacement
    workspace = Path(args[3])
    source = _write(workspace)
    payload = json.loads(candidate.read_text())
    payload["tiers"][0]["slices"][0]["directive"]["read_only_references"] = [asdict(_reference())]
    candidate.write_text(json.dumps(payload))
    contexts = []

    class Backend:
        name = "fixture"
        model = "fixture"

    class Pool:
        def get(self, role, **_kwargs):
            assert role in {"plan_critic", "plan_judge"}
            return Backend()

    @contextmanager
    def pool(*_args, **_kwargs):
        yield Pool()

    def review(artifact, rubric, **kwargs):
        contexts.append(kwargs["context"]["read_only_sources"])
        assert "{read_only_sources}" in rubric.user_template
        assert "never root requirements" in rubric.system_prompt
        if stage == "during_review" and kwargs["kind"] == "critic":
            source.write_text("changed\nsecond\nlast")
        return Verdict(
            rubric=rubric.name, decision="pass", findings=(),
            artifact_ref=corpus.put_artifact(artifact, kind="reviewed_artifacts"),
            artifact_commit=None, reasoning="fixture review",
            coverage=ReviewCoverage(
                complete=True, sources_reviewed=("artifact", "context:read_only_sources"),
                omissions=(), artifact_chars_total=len(artifact), artifact_chars_reviewed=len(artifact),
            ),
            backend_name="fixture", transcript_ref=None, cost_usd=0,
            execution=None, kind=kwargs["kind"], cache_creation_tokens=0, cache_read_tokens=0,
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=True, cost_complete=True,
                terminal_complete=True, trace_complete=True,
            ),
        )

    monkeypatch.setattr(plan_workflow, "backend_pool", pool)
    monkeypatch.setattr(plan_workflow, "strict_review", review)
    monkeypatch.setattr(plan_workflow, "plan_contract_fingerprint", lambda *_a, **_k: "sha256:" + "a" * 64)
    monkeypatch.setattr(plan_workflow, "bind_plan_contract", lambda _corpus, staged, **_kwargs: staged)
    if stage == "before_review":
        original = plan_workflow.build_pre_review_candidate_validator

        def changing_validator(**kwargs):
            validate = original(**kwargs)

            def validate_then_change(dag):
                findings = validate(dag)
                assert not findings
                source.write_text("changed\nsecond\nlast")
                return findings

            return validate_then_change

        monkeypatch.setattr(plan_workflow, "build_pre_review_candidate_validator", changing_validator)

    assert plan_workflow.main(["--planner-count", "1", *args]) == expected_rc
    latest = corpus.latest_spec()
    assert latest.implementation_refs == base.implementation_refs
    assert corpus.get_spec(base.version) == base
    if stage == "stable":
        assert latest.tier_dag != base.tier_dag
    else:
        assert latest.tier_dag == base.tier_dag
    if stage == "before_review":
        assert contexts == []
        (run,) = (corpus.spec_dir / "runs").iterdir()
        assert json.loads((run / "summary.json").read_text())["status"] == "candidate_context_failed"
    else:
        assert len(contexts) == 2 and contexts[0] == contexts[1]
        retained = [corpus.get_record(ref, as_type=Verdict) for ref in latest.verdict_refs]
        reviewed = [verdict for verdict in retained if "plan_source_context" in verdict.artifacts]
        assert len(reviewed) == 2
        assert reviewed[0].artifacts["plan_source_context"] == reviewed[1].artifacts["plan_source_context"]
        blob = corpus.get_artifact(reviewed[0].artifacts["plan_source_context"])
        assert json.loads(blob)["references"][0]["sha256"] == _reference().sha256


def test_real_review_adapter_renders_the_same_captured_evidence_for_both_roles(tmp_path):
    from assay import Verdict
    from crucible.rubrics import load
    from tests.test_plan_review_response_retry import _Backend, _pass_response
    from tests.test_workflows_plan_candidate_import import _candidate_payload, _spec

    _write(tmp_path)
    payload = _candidate_payload()
    payload["tiers"][0]["slices"][0]["directive"]["read_only_references"] = [asdict(_reference())]
    dag = plan_workflow._strict_review_candidate_dag(payload)
    critic, judge = _Backend(_pass_response()), _Backend(_pass_response())
    corpus = Corpus(tmp_path / "corpus")
    spec = _spec()
    corpus.put_spec(spec)
    outcome = plan_workflow.run_plan(
        spec, candidate_dag=dag, critic_backend=critic, judge_backend=judge,
        corpus=corpus, critic_rubric=load("spec_clarity"), judge_rubric=load("spec_clarity"),
        cwd=str(tmp_path),
    )
    assert outcome.accepted
    for backend in (critic, judge):
        assert len(backend.invocations) == 1
        prompt = backend.invocations[0].user
        assert "context:read_only_sources" in prompt
        assert _reference().sha256 in prompt
        assert "src/evidence.py" in prompt
        assert "second" in prompt and "last" in prompt
    verdicts = [corpus.get_record(ref, as_type=Verdict) for ref in outcome.spec.verdict_refs]
    assert len(verdicts) == 2
    assert verdicts[0].artifacts["plan_source_context"] == verdicts[1].artifacts["plan_source_context"]
