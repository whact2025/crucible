# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the operator-facing Spec promotion helper."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from assay import (
    Implementation,
    ImplementationDirective,
    Measurement,
    ReviewCoverage,
    Slice,
    Spec,
    Tier,
    TierDAG,
    Verdict,
)
from assay.canonical import hash_record
from assay.corpus.store import Corpus


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".claude/skills/_shared/bin/promote_spec_yaml.py"
)
_YAML = """
schema_version: 1
project_context:
  project: promotion-test
  repo_url: https://example.invalid/promotion.git
  base_commit: abc123
  branch: main
  submodule_path: null
  runners: {}
intent: Initial authored intent.
constraints: []
done_criteria: []
"""


def _load_module():
    module_spec = importlib.util.spec_from_file_location(
        "promote_spec_yaml_test_module",
        _SCRIPT,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _write_yaml(spec_dir: Path, *, intent: str = "Initial authored intent.") -> None:
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "spec.yaml").write_text(
        _YAML.replace("Initial authored intent.", intent),
        encoding="utf-8",
    )


def _dag() -> TierDAG:
    directive = ImplementationDirective(
        slice_id="s0",
        intent="Implement the retained slice.",
        files_in_scope=("src/example.py",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={},
    )
    return TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(Slice(id="s0", directive=directive),),
                depends_on=(),
            ),
        )
    )


def test_first_promotion_publishes_version_zero(tmp_path, capsys):
    spec_dir = tmp_path / "spec"
    _write_yaml(spec_dir)

    assert _load_module().main([str(spec_dir)]) == 0

    latest = Corpus(spec_dir).latest_spec()
    assert latest is not None
    assert latest.version == 0
    assert latest.parent is None
    assert latest.root is None
    assert latest.tier_dag is None
    assert "tier_dag: cleared" in capsys.readouterr().out


def test_promotion_rebases_lineage_and_preserves_append_only_refs(
    tmp_path,
    capsys,
):
    spec_dir = tmp_path / "spec"
    _write_yaml(spec_dir)
    corpus = Corpus(spec_dir)
    base = Spec.from_yaml(_YAML)
    corpus.put_spec(base)
    planned = base.with_tier_dag(_dag())
    corpus.put_spec(planned)

    # The exact record payload is irrelevant to promotion. Reuse a valid
    # lifecycle record already attached through the public Spec transition to
    # prove that an append-only reference survives the authored rebase.
    from assay.types.artifacts import LifecycleEvent
    from assay.types.execution import ExecutionContext

    execution = ExecutionContext(
        timestamp="2026-08-05T00:00:00+00:00",
        machine="test",
        cwd=str(tmp_path),
        toolchain={},
        backend=None,
        model=None,
        recipe="test",
        recipe_commit=None,
        gpu_model=None,
        run_id=None,
    )
    event = LifecycleEvent(
        kind="planned",
        reason="test evidence",
        affected_versions=(3,),
        execution=execution,
    )
    event_ref = corpus.put_record(event, kind="lifecycle_events")
    diff_ref = corpus.put_artifact(b"diff", kind="diffs")
    transcript_ref = corpus.put_artifact(b"[]", kind="transcripts")
    implementation_ref = corpus.put_record(
        Implementation(
            slice_id="s0",
            base_commit="abc123",
            result_commit=None,
            diff_ref=diff_ref,
            transcript_ref=transcript_ref,
            files_modified=("src/example.py",),
            classification="success",
            execution=execution,
            scope_escapes=(),
        ),
        kind="implementations",
    )
    reviewed_artifact_ref = corpus.put_artifact(
        b"review input", kind="reviewed_artifacts"
    )
    verdict_ref = corpus.put_record(
        Verdict(
            rubric="promotion-test",
            decision="pass",
            findings=(),
            artifact_ref=reviewed_artifact_ref,
            artifact_commit=None,
            reasoning="Typed preservation fixture.",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
                artifact_chars_total=12,
                artifact_chars_reviewed=12,
            ),
            backend_name=None,
            transcript_ref=transcript_ref,
            cost_usd=0.0,
            execution=execution,
        ),
        kind="verdicts",
    )
    raw_output_ref = corpus.put_artifact(b"{}", kind="raw_output")
    measurement_ref = corpus.put_record(
        Measurement(
            bench_id="promotion-test",
            commit="abc123",
            config_hash="fixture",
            values={"score": 1.0},
            raw_output_ref=raw_output_ref,
            execution=execution,
        ),
        kind="measurements",
    )
    evidenced = planned.with_appended_refs(
        implementation_refs=(implementation_ref,),
        verdict_refs=(verdict_ref,),
        measurement_refs=(measurement_ref,),
        lifecycle_event_refs=(event_ref,),
    )
    corpus.put_spec(evidenced)
    _write_yaml(spec_dir, intent="Edited authored intent.")

    assert _load_module().main([str(spec_dir)]) == 0

    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.version == evidenced.version + 1
    assert latest.parent == hash_record(evidenced)
    assert latest.root == evidenced.root
    assert latest.intent == "Edited authored intent."
    assert latest.lifecycle_event_refs == evidenced.lifecycle_event_refs
    assert latest.implementation_refs == evidenced.implementation_refs
    assert latest.verdict_refs == evidenced.verdict_refs
    assert latest.measurement_refs == evidenced.measurement_refs
    assert latest.tier_dag is None
    assert "tier_dag: cleared" in capsys.readouterr().out


def test_exact_digest_can_retain_plan_for_explicit_reacceptance(
    tmp_path,
    capsys,
):
    spec_dir = tmp_path / "spec"
    _write_yaml(spec_dir)
    corpus = Corpus(spec_dir)
    base = Spec.from_yaml(_YAML)
    corpus.put_spec(base)
    planned = base.with_tier_dag(_dag())
    corpus.put_spec(planned)
    digest = hash_record(planned.tier_dag)
    _write_yaml(spec_dir, intent="Runner parameters were deliberately edited.")

    assert _load_module().main([str(spec_dir), "--retain-plan-digest", digest]) == 0

    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.tier_dag == planned.tier_dag
    assert hash_record(latest.tier_dag) == digest
    assert latest.intent == "Runner parameters were deliberately edited."
    assert f"tier_dag: retained {digest}" in capsys.readouterr().out


def test_wrong_retain_digest_fails_without_publishing(tmp_path, capsys):
    spec_dir = tmp_path / "spec"
    _write_yaml(spec_dir)
    corpus = Corpus(spec_dir)
    base = Spec.from_yaml(_YAML)
    corpus.put_spec(base)
    planned = base.with_tier_dag(_dag())
    corpus.put_spec(planned)

    assert (
        _load_module().main(
            [str(spec_dir), "--retain-plan-digest", "sha256:" + "0" * 64]
        )
        == 1
    )

    assert corpus.latest_spec() == planned
    assert "does not match" in capsys.readouterr().err


def test_concurrent_canonical_advance_aborts_authored_promotion(
    tmp_path,
    capsys,
    monkeypatch,
):
    spec_dir = tmp_path / "spec"
    _write_yaml(spec_dir)
    corpus = Corpus(spec_dir)
    base = Spec.from_yaml(_YAML)
    corpus.put_spec(base)
    planned = base.with_tier_dag(_dag())
    corpus.put_spec(planned)
    _write_yaml(spec_dir, intent="Stale authored edit.")

    original_commit = Corpus.commit
    injected = False

    def commit_after_concurrent_advance(self, *args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            current = self.latest_spec()
            assert current is not None
            self.put_spec(current.with_appended_refs())
        return original_commit(self, *args, **kwargs)

    monkeypatch.setattr(Corpus, "commit", commit_after_concurrent_advance)

    assert _load_module().main([str(spec_dir)]) == 1

    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.version == planned.version + 1
    assert latest.intent == planned.intent
    assert "changed during promotion" in capsys.readouterr().err


def test_corrupt_latest_manifest_aborts_without_publishing(tmp_path, capsys):
    spec_dir = tmp_path / "spec"
    _write_yaml(spec_dir)
    corpus = Corpus(spec_dir)
    base = Spec.from_yaml(_YAML)
    corpus.put_spec(base)
    planned = base.with_tier_dag(_dag())
    corpus.put_spec(planned)
    _write_yaml(spec_dir, intent="Authored edit must not bypass corruption.")

    corrupt = (
        spec_dir
        / "versions"
        / f"v{planned.version + 1}_{'0' * 64}.json"
    )
    corrupt.write_text("{not valid json", encoding="utf-8")
    manifests_before = {
        path.name for path in (spec_dir / "versions").iterdir()
    }

    assert _load_module().main([str(spec_dir)]) == 1

    assert {
        path.name for path in (spec_dir / "versions").iterdir()
    } == manifests_before
    assert "spec promotion failed" in capsys.readouterr().err


def test_valid_json_manifest_tamper_aborts_without_publishing(tmp_path, capsys):
    spec_dir = tmp_path / "spec"
    _write_yaml(spec_dir)
    corpus = Corpus(spec_dir)
    base = Spec.from_yaml(_YAML)
    corpus.put_spec(base)
    planned = base.with_tier_dag(_dag())
    corpus.put_spec(planned)
    _write_yaml(spec_dir, intent="Authored edit must not trust tampered state.")

    manifest = next(
        (spec_dir / "versions").glob(f"v{planned.version}_*.json")
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["intent"] = "Valid JSON, invalid content-addressed identity."
    manifest.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    manifests_before = {
        path.name for path in (spec_dir / "versions").iterdir()
    }

    assert _load_module().main([str(spec_dir)]) == 1

    assert {
        path.name for path in (spec_dir / "versions").iterdir()
    } == manifests_before
    error = capsys.readouterr().err
    assert "manifest" in error and "filename" in error


def test_tampered_manifest_with_short_matching_prefix_is_rejected(
    tmp_path,
    capsys,
):
    spec_dir = tmp_path / "spec"
    _write_yaml(spec_dir)
    corpus = Corpus(spec_dir)
    base = Spec.from_yaml(_YAML)
    corpus.put_spec(base)
    planned = base.with_tier_dag(_dag())
    corpus.put_spec(planned)

    manifest = next(
        (spec_dir / "versions").glob(f"v{planned.version}_*.json")
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["intent"] = "Tampered and renamed with an undersized prefix."
    tampered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    short_prefix = hashlib.sha256(tampered).hexdigest()[:1]
    renamed = manifest.with_name(f"v{planned.version}_{short_prefix}.json")
    manifest.unlink()
    renamed.write_bytes(tampered)
    manifests_before = {
        path.name for path in (spec_dir / "versions").iterdir()
    }

    assert _load_module().main([str(spec_dir)]) == 1

    assert {
        path.name for path in (spec_dir / "versions").iterdir()
    } == manifests_before
    error = capsys.readouterr().err
    assert "manifest" in error and "filename" in error
