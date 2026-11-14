# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from assay import ProjectContext, Spec, TierDAG
from assay.canonical import hash_record
from assay.corpus.store import Corpus
from assay.types.artifacts import LifecycleEvent
from assay.types.constraints import BlastRadiusConstraint

from crucible.constraints import LanguagesConstraint
from crucible.workflows import architect as arch
from crucible.workflows._plan_provenance import (
    PlanContractFingerprint,
    accepted_plan_contract_state,
)
from crucible.workflows._run import exec_ctx_now


def _fixed_fingerprint() -> PlanContractFingerprint:
    components = {"authority.intent": "sha256:" + "1" * 64}
    digest = hash_record({
        "schema_version": "crucible.plan-contract.v3",
        "component_digests": components,
    })
    return PlanContractFingerprint(digest, components)


def _corpus_with_later_inconclusive(tmp_path: Path) -> tuple[Corpus, str]:
    corpus = Corpus(tmp_path / "corpus")
    spec = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="reaccept-test",
            repo_url="https://example.invalid/reaccept.git",
            base_commit="a" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="Retain the reviewed plan.",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
    )
    corpus.put_spec(spec)
    dag = TierDAG(tiers=())
    planned = LifecycleEvent(
        kind="planned",
        reason="original accepted plan",
        affected_versions=(1,),
        execution=exec_ctx_now("plan"),
    )
    planned_ref = corpus.put_record(planned, kind="lifecycle_events")
    accepted = spec.with_appended_refs(
        tier_dag=dag,
        lifecycle_event_refs=(planned_ref,),
    )
    corpus.put_spec(accepted)
    inconclusive = LifecycleEvent(
        kind="plan_inconclusive",
        reason="later replan exhausted its review rounds",
        affected_versions=(2,),
        execution=exec_ctx_now("plan"),
    )
    inconclusive_ref = corpus.put_record(
        inconclusive, kind="lifecycle_events",
    )
    corpus.put_spec(accepted.with_appended_refs(
        lifecycle_event_refs=(inconclusive_ref,),
    ))
    return corpus, hash_record(dag)


def _install_reaccept_stubs(monkeypatch, *, validation_status: str = "passed"):
    fingerprint = _fixed_fingerprint()
    calls = {
        "backend_pool": 0,
        "run_plan": 0,
        "validation": [],
    }

    class _Backend:
        name = "unused"
        model = "unused"

        def invoke(self, _invocation):  # pragma: no cover - must stay unused
            raise AssertionError("planning backend must not be invoked")

    class _Pool:
        def get(self, _role, **_kwargs):
            return _Backend()

    @contextmanager
    def _pool(*_args, **_kwargs):
        calls["backend_pool"] += 1
        yield _Pool()

    def _run_plan(*_args, **_kwargs):
        calls["run_plan"] += 1
        raise AssertionError("explicit reacceptance must not call run_plan")

    def _validation_gate(**kwargs):
        calls["validation"].append(kwargs)
        if validation_status == "passed":
            return kwargs["planned"], "passed", 0.0
        return replace(kwargs["planned"], tier_dag=None), validation_status, 0.0

    monkeypatch.setattr(arch, "backend_pool", _pool)
    monkeypatch.setattr(arch, "load_rubric", lambda name: name)
    monkeypatch.setattr(
        arch, "plan_contract_fingerprint", lambda *args, **kwargs: fingerprint,
    )
    monkeypatch.setattr(arch, "run_plan", _run_plan)
    monkeypatch.setattr(arch, "_validation_gate", _validation_gate)
    return fingerprint, calls


def _only_summary(corpus: Corpus) -> dict:
    paths = list(Path(corpus.spec_dir).glob("runs/*_architect_*/summary.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_exact_digest_reaccepts_after_inconclusive_and_is_fresh_on_reuse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus, digest = _corpus_with_later_inconclusive(tmp_path)
    fingerprint, calls = _install_reaccept_stubs(monkeypatch)

    assert arch.run_architect(
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        reaccept_plan_digest=digest,
    ) == 0

    assert calls["backend_pool"] == 0
    assert calls["run_plan"] == 0
    assert len(calls["validation"]) == 1
    assert calls["validation"][0]["allow_repair"] is False
    assert calls["validation"][0]["planner_backend"] is None
    current = corpus.latest_spec()
    assert current is not None
    events = [
        corpus.get_record(ref, as_type=LifecycleEvent)
        for ref in current.lifecycle_event_refs[-2:]
    ]
    assert [event.kind for event in events] == [
        "planned", "plan_contract_bound",
    ]
    assert digest in events[0].reason
    assert '"schema_version":"crucible.plan-contract.v3"' in events[1].reason
    assert accepted_plan_contract_state(
        corpus,
        current,
        expected_fingerprint=fingerprint,
    ).status == "fresh"

    # A normal subsequent Architect invocation reuses the newly bound plan;
    # it may construct adapters for later tiers but still makes no plan call.
    assert arch.run_architect(
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
    ) == 0
    assert calls["backend_pool"] == 1
    assert calls["run_plan"] == 0
    assert len(calls["validation"]) == 2


@pytest.mark.parametrize(
    "supplied",
    ["not-a-digest", "sha256:" + "f" * 64],
)
def test_reaccept_rejects_malformed_or_mismatched_digest_before_publication(
    tmp_path: Path,
    monkeypatch,
    supplied: str,
) -> None:
    corpus, _digest = _corpus_with_later_inconclusive(tmp_path)
    before = corpus.latest_spec()
    monkeypatch.setattr(
        arch,
        "backend_pool",
        lambda *args, **kwargs: pytest.fail("backend_pool must not open"),
    )

    assert arch.run_architect(
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        reaccept_plan_digest=supplied,
    ) == 2
    assert corpus.latest_spec() == before
    assert _only_summary(corpus)["error_kind"] == "invalid_plan_reacceptance"


def test_reaccept_rejects_unpromoted_spec_yaml_drift_before_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus, digest = _corpus_with_later_inconclusive(tmp_path)
    before = corpus.latest_spec()
    Path(corpus.spec_dir, "spec.yaml").write_text(
        """
schema_version: 1
project_context:
  project: reaccept-test
  repo_url: https://example.invalid/reaccept.git
  base_commit: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  branch: null
  submodule_path: null
  runners: {}
intent: This edit has not been promoted.
constraints: []
done_criteria: []
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        arch,
        "backend_pool",
        lambda *args, **kwargs: pytest.fail("backend_pool must not open"),
    )

    assert (
        arch.run_architect(
            spec=Path(corpus.spec_dir),
            cwd=str(tmp_path),
            reaccept_plan_digest=digest,
        )
        == 2
    )

    assert corpus.latest_spec() == before
    summary = _only_summary(corpus)
    assert summary["error_kind"] == "invalid_plan_reacceptance"
    assert "unpromoted authoring changes" in summary["message"]
    assert "intent" in summary["message"]
    assert "--retain-plan-digest" in summary["message"]


def test_reaccept_drift_compare_normalizes_authored_constraint_lists(
    tmp_path: Path,
) -> None:
    corpus, _digest = _corpus_with_later_inconclusive(tmp_path)
    corpus.commit(transform=lambda latest: replace(
        latest,
        constraints=(
            BlastRadiusConstraint(no_edits_outside=("native", "tests")),
            LanguagesConstraint(languages=("python", "cpp")),
        ),
    ))
    Path(corpus.spec_dir, "spec.yaml").write_text(
        """
schema_version: 1
project_context:
  project: reaccept-test
  repo_url: https://example.invalid/reaccept.git
  base_commit: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  branch: null
  submodule_path: null
  runners: {}
intent: Retain the reviewed plan.
constraints:
  - kind: blast_radius
    no_edits_outside: [native, tests]
  - kind: languages
    languages: [python, cpp]
done_criteria: []
""",
        encoding="utf-8",
    )

    assert arch._unpromoted_authoring_drift(
        Path(corpus.spec_dir), corpus.latest_spec(),
    ) == ()


def test_reaccept_validation_failure_does_not_publish_terminal_or_bind(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus, digest = _corpus_with_later_inconclusive(tmp_path)
    before = corpus.latest_spec()
    _fingerprint, calls = _install_reaccept_stubs(
        monkeypatch, validation_status="validate_failed",
    )

    assert arch.run_architect(
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        reaccept_plan_digest=digest,
    ) == 2
    assert corpus.latest_spec() == before
    assert calls["backend_pool"] == 0
    assert calls["run_plan"] == 0
    assert calls["validation"][0]["allow_repair"] is False
    summary = _only_summary(corpus)
    assert summary["status"] == "validate_failed"
    assert summary["version"] == before.version
    assert summary["reaccept_plan_digest"] == digest


@pytest.mark.parametrize(
    "extra",
    [{"replan": True}, {"skip_validate": True}],
)
def test_programmatic_reaccept_rejects_conflicting_authority(
    tmp_path: Path,
    monkeypatch,
    extra: dict[str, bool],
) -> None:
    corpus, digest = _corpus_with_later_inconclusive(tmp_path)
    before = corpus.latest_spec()
    monkeypatch.setattr(
        arch,
        "backend_pool",
        lambda *args, **kwargs: pytest.fail("backend_pool must not open"),
    )

    assert arch.run_architect(
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        reaccept_plan_digest=digest,
        **extra,
    ) == 2
    assert corpus.latest_spec() == before


def test_architect_cli_reaccept_argument_conflicts() -> None:
    digest = "sha256:" + "1" * 64
    with pytest.raises(SystemExit) as replan_error:
        arch._parse_args([
            "--spec", "unused", "--replan",
            "--reaccept-plan-digest", digest,
        ])
    assert replan_error.value.code == 2
    with pytest.raises(SystemExit) as validate_error:
        arch._parse_args([
            "--spec", "unused", "--skip-validate",
            "--reaccept-plan-digest", digest,
        ])
    assert validate_error.value.code == 2
