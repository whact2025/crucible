# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""End-to-end test for ``crucible.workflows.tier``'s
all-slices-transport-failed escalation floor (NF2 / R3).

Constructs a 3-slice tier where every slice's implementer backend
returns a ``TransportError`` result. ``assay.primitives.implement
._classify`` maps each to ``Implementation.classification ==
"hard_fail_transport"``; the tier loop's transport-failure handler
skips each slice + increments ``summary["transport_failure_slices"]``;
the post-loop escalation conditional fires because
``transport_failure_slices == len(tier.slices)`` → ``status =
"all_slices_transport_failed"`` + ``return 1``.

Without the R3 floor, the tier would finalize with
``status="success"`` and ``union_diff_bytes=0`` — silently shipping
an empty patch downstream.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    Implementation,
    InvocationResult,
    ProjectContext,
    Spec,
    Usage,
)
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG
from assay.types.invocation import TransportError
from crucible.common.policies import BreachPolicy
from crucible.constraints import LanguagesConstraint


_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_TRACE = BackendExecutionTrace(
    backend="stub-transport",
    requested_model="stub-model",
    events=(
        BackendExecutionEvent(
            sequence=0,
            kind="terminal",
            name="terminal_result",
        ),
    ),
)


class _TransportFailingBackend:
    """Backend that always returns a TransportError-tagged
    InvocationResult — assay.primitives.implement maps these to
    Implementation(classification='hard_fail_transport')."""

    name = "stub-transport-fail"
    model = "stub-model"

    def __init__(self):
        self.calls: list = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.calls.append(invocation)
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
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _FirstTransportSecondSuccessBackend:
    name = "stub-mixed-transport"
    model = "stub-model"

    def __init__(self, repo: Path):
        self.repo = repo
        self.calls: list[str] = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        slice_id = "s0" if not self.calls else "s1"
        self.calls.append(slice_id)
        if slice_id == "s0":
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
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        (self.repo / "b.py").write_text("second slice landed\n")
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            error=None,
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


def _init_repo(path: Path) -> str:
    """Init a git repo at ``path`` and return its HEAD sha."""
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


def _spec_with_three_slice_tier(base_commit: str) -> Spec:
    """Build a Spec carrying a tier_dag with a single tier and 3
    slices, files_in_scope disjoint."""
    slices = tuple(
        Slice(
            id=f"s{i}",
            directive=ImplementationDirective(
                slice_id=f"s{i}",
                intent=f"slice {i} intent",
                files_in_scope=(f"file_{i}.py",),
                creates_paths=(),
                verify_mode="skip",
                blast_radius={"max_files": 1},
            ),
        )
        for i in range(3)
    )
    return Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="tier_transport_test",
            repo_url="https://example.invalid/tt.git",
            base_commit=base_commit,
            branch=None, submodule_path=None, runners={},
        ),
        execution=None,
        intent="exercise the all-slices-transport-failed floor",
        constraints=(), done_criteria=(), plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=slices, depends_on=()),),
        ),
        implementation_refs=(), verdict_refs=(),
        measurement_refs=(), lifecycle_event_refs=(),
    )


def _spec_with_two_slice_tier(base_commit: str) -> Spec:
    slices = tuple(
        Slice(
            id=f"s{i}",
            directive=ImplementationDirective(
                slice_id=f"s{i}",
                intent=f"slice {i} intent",
                files_in_scope=(("a.py",) if i == 0 else ("b.py",)),
                creates_paths=(),
                verify_mode="skip",
                blast_radius={"max_files": 1},
            ),
        )
        for i in range(2)
    )
    base = _spec_with_three_slice_tier(base_commit)
    return replace(
        base,
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=slices, depends_on=()),),
        ),
        intent="transport rollback does not invalidate later slices",
    )


def test_tier_all_slices_transport_failed_escalates_status_and_returns_1(
    tmp_path: Path, monkeypatch,
):
    """Every slice's implementer transport-fails → tier escalates
    to ``all_slices_transport_failed`` + ``return 1`` instead of
    silently finalizing with an empty union diff."""
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)

    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec_with_three_slice_tier(base))

    backend = _TransportFailingBackend()
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kw: backend,
    )

    from crucible.workflows.tier import run_tier

    rc = run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        # Skip everything that requires the build_verify subsystem
        # — the transport-failure escalation fires before we get
        # to those gates anyway.
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        # Default ``BreachPolicy()`` has enabled=False; no explicit
        # kwarg needed to skip breach.
        skip_review=True,
    )
    assert rc == 1, "all-slices-transport-failed must return 1"

    # Find the tier's run dir and check the summary.
    runs = list((corpus_dir / "runs").iterdir())
    tier_runs = [r for r in runs if "tier" in r.name]
    assert tier_runs, "no tier run dir was created"
    summary = json.loads(
        (tier_runs[-1] / "summary.json").read_text(encoding="utf-8"),
    )
    assert summary["status"] == "all_slices_transport_failed", (
        f"expected R3 escalation; got status={summary['status']!r}"
    )
    assert summary["transport_failure_slices"] == 3, (
        f"expected 3 transport-failed slices; "
        f"got {summary['transport_failure_slices']}"
    )
    # union_diff was never written — we escalated before Phase 2.
    assert summary["phase_2_union_diff_bytes"] == 0
    outcome = json.loads(
        (tier_runs[-1] / "outcome.json").read_text(encoding="utf-8"),
    )
    assert outcome["detail_status"] == "all_slices_transport_failed"
    assert outcome["terminal_class"] == "retryable_failure"


def test_transport_failure_rolls_back_only_current_slice_and_invokes_later_once(
    tmp_path: Path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    (repo / "a.py").write_text("a baseline\n")
    (repo / "b.py").write_text("b baseline\n")
    subprocess.run(["git", "add", "a.py", "b.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "slice baselines"], cwd=repo, check=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec_with_two_slice_tier(base))
    backend = _FirstTransportSecondSuccessBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kw: backend,
    )

    from crucible.workflows.tier import run_tier

    rc = run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    )

    assert rc == 1
    assert backend.calls == ["s0", "s1"]
    assert (repo / "a.py").read_text() == "a baseline\n"
    assert (repo / "b.py").read_text() == "second slice landed\n"
    run_dir = max(
        (path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name),
        key=lambda path: path.stat().st_mtime_ns,
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "incomplete_transport_failed"
    assert summary["transport_failure_slices"] == 1
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["detail_status"] == "incomplete_transport_failed"
    assert outcome["terminal_class"] == "retryable_failure"
    records = [
        corpus.get_record(ref, as_type=Implementation)
        for ref in corpus.latest_spec().implementation_refs
    ]
    s1_records = [record for record in records if record.slice_id == "s1"]
    assert len(s1_records) == 1
    assert s1_records[0].classification.startswith("success")


def test_transport_without_local_write_cannot_inherit_prior_slice_diff(
    tmp_path: Path,
    monkeypatch,
):
    """Ambient in-scope dirt is evidence, not work by this invocation."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    target = repo / "a.py"
    target.write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "add target"], cwd=repo, check=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    target.write_text("inherited prior-tier edit\n", encoding="utf-8")

    spec = _spec_with_two_slice_tier(base)
    first_slice = spec.tier_dag.tiers[0].slices[0]
    spec = replace(
        spec,
        project_context=replace(
            spec.project_context,
            runners={"verify": "crucible.common.runners.lang.python"},
        ),
        tier_dag=TierDAG(
            tiers=(
                Tier(
                    tier_id="t0",
                    slices=(first_slice,),
                    depends_on=(),
                ),
            ),
        ),
    )
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    corpus = Corpus(corpus_dir)
    corpus.put_spec(spec)

    backend = _TransportFailingBackend()
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kw: backend,
    )

    def forbidden(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("no-write transport failure reached a gate")

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        forbidden,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        forbidden,
    )

    from crucible.workflows.tier import run_tier

    rc = run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        breach_policy=BreachPolicy(enabled=True),
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    )

    assert rc == 1
    assert len(backend.calls) == 1
    assert target.read_text(encoding="utf-8") == "inherited prior-tier edit\n"
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == "hard_fail_transport"
    # The cumulative artifact remains available for composition/postmortem,
    # but cannot authorize verification or Breach for this failed call.
    assert implementation.files_modified == ("a.py",)
    run_dir = max(
        (path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name),
        key=lambda path: path.stat().st_mtime_ns,
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "all_slices_transport_failed"
    assert summary["transport_failure_slices"] == 1


@pytest.mark.parametrize(
    ("language", "runner"),
    (("python", "untrusted.project.runner"), ("javascript", "custom.npm")),
)
def test_tier_rejects_unsupported_observation_runner_before_backend(
    tmp_path: Path,
    monkeypatch,
    language: str,
    runner: str,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    spec = _spec_with_two_slice_tier(base)
    spec = replace(
        spec,
        constraints=(LanguagesConstraint(languages=(language,)),),
        project_context=replace(
            spec.project_context,
            runners={"verify": runner},
        ),
    )
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    Corpus(corpus_dir).put_spec(spec)

    def forbidden(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("backend acquired before observation preflight")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        forbidden,
    )

    from crucible.workflows.tier import run_tier

    assert run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        breach_policy=BreachPolicy(enabled=True),
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    ) == 2


def test_tier_rejects_python_deepswe_override_before_backend(
    tmp_path: Path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    spec = _spec_with_two_slice_tier(base)
    spec = replace(
        spec,
        constraints=(LanguagesConstraint(languages=("python",)),),
        project_context=replace(
            spec.project_context,
            runners={
                "dev_verify": (
                    "crucible.products.deepswe.runners.deepswe_dev_verify"
                ),
            },
        ),
    )
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    Corpus(corpus_dir).put_spec(spec)

    def forbidden(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("backend acquired before DeepSWE preflight")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        forbidden,
    )
    from crucible.workflows.tier import run_tier

    assert run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        breach_policy=BreachPolicy(enabled=True),
        extra_parameters={
            "deepswe_metadata": {
                "language": "python",
                "container_id": "authoring-container",
                "dev_verify_command": "pytest -q forged_target",
            },
        },
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    ) == 2


def test_tier_rejects_spec_runner_controls_before_backend(
    tmp_path: Path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    spec = _spec_with_two_slice_tier(base)
    spec = replace(
        spec,
        project_context=replace(
            spec.project_context,
            runner_parameters={
                "per_language": {"rust": {"role": "build"}},
            },
        ),
    )
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    Corpus(corpus_dir).put_spec(spec)

    def forbidden(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("backend acquired before runner preflight")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        forbidden,
    )
    from crucible.workflows.tier import run_tier

    assert run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        breach_policy=BreachPolicy(enabled=False),
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    ) == 2


def test_tier_rejects_runtime_runner_controls_before_backend(
    tmp_path: Path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus" / "spec1"
    corpus_dir.mkdir(parents=True)
    Corpus(corpus_dir).put_spec(_spec_with_two_slice_tier(base))

    def forbidden(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("backend acquired before runtime preflight")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        forbidden,
    )
    from crucible.workflows.tier import run_tier

    assert run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        extra_parameters={"working_dir": "/unlocked/checkout"},
        breach_policy=BreachPolicy(enabled=False),
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    ) == 2
