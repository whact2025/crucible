# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from assay import (
    ImplementationDirective,
    ProjectContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
)

from crucible.workflows import _outcome as outcome_module
from crucible.workflows._outcome import (
    TerminalClass,
    WorkflowOutcome,
    begin_run,
    checkpoint_settled_tool_mutation_terminal,
    checkpoint_unsettled_tool_mutation_terminal,
    child_operation,
    child_outcome,
    child_run_dir,
    classify_terminal_status,
    finalize_run,
    record_provenance_inputs,
    reconcile_stale_runs,
)


class _Corpus:
    def __init__(self, root: Path) -> None:
        self.spec_dir = root


class _CheckpointHandle:
    def __init__(self) -> None:
        self.finalized = False
        self.summaries: list[dict[str, object]] = []
        self.results: list[object] = []

    def write_summary(self, summary: object) -> None:
        assert isinstance(summary, dict)
        self.summaries.append(summary)

    def finalize_elapsed(self, result: object) -> None:
        self.results.append(result)
        self.finalized = True


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _spec_with_directive() -> Spec:
    directive = ImplementationDirective(
        slice_id="s0",
        intent="make the change",
        files_in_scope=("src/lib.py",),
        creates_paths=(),
        verify_mode="build_and_test",
        blast_radius={"api": "none"},
    )
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="test",
            repo_url="https://example.invalid/repo.git",
            base_commit="a" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="test intent",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG((Tier("t0", (Slice("s0", directive),), ()),)),
    )


def test_terminal_status_taxonomy_fails_unknown_values_to_contract_error() -> None:
    assert classify_terminal_status("accepted") is TerminalClass.SUCCEEDED
    assert classify_terminal_status("completed") is TerminalClass.SUCCEEDED
    assert classify_terminal_status("no_attacks") is TerminalClass.SUCCEEDED
    assert classify_terminal_status("no_work") is TerminalClass.NO_WORK
    assert (
        classify_terminal_status("budget_exhausted") is TerminalClass.BUDGET_EXHAUSTED
    )
    assert (
        classify_terminal_status("incomplete_accounting") is TerminalClass.INCONCLUSIVE
    )
    assert (
        classify_terminal_status("task_lease_busy") is TerminalClass.RETRYABLE_FAILURE
    )
    assert (
        classify_terminal_status("planner_contract_failure")
        is TerminalClass.PERMANENT_FAILURE
    )
    assert (
        classify_terminal_status("hard_fail_tool_mutation_indeterminate")
        is TerminalClass.PERMANENT_FAILURE
    )
    assert (
        classify_terminal_status("hard_fail_transport")
        is TerminalClass.RETRYABLE_FAILURE
    )
    for status in (
        "all_slices_transport_failed",
        "incomplete_transport_failed",
        "retryable_tier_failure",
        "retryable_tasks_failed",
    ):
        assert classify_terminal_status(status) is TerminalClass.RETRYABLE_FAILURE
    for status in (
        "critic_failures_exhausted",
        "critic_infrastructure_failure",
        "critic_protocol_failure",
        "inconclusive_authority",
        "inconclusive_post_strip_verify",
        "judge_failures_exhausted",
        "judge_infrastructure_failure",
        "judge_protocol_failure",
    ):
        assert classify_terminal_status(status) is TerminalClass.INCONCLUSIVE
    assert (
        classify_terminal_status("brand_new_unclassified")
        is TerminalClass.CONTRACT_ERROR
    )


def test_tool_mutation_checkpoints_own_provider_settlement_truth() -> None:
    terminal = RuntimeError("mutation settlement unknown")
    cleanup = RuntimeError("provider descendants still live")

    unsettled = _CheckpointHandle()
    checkpoint_unsettled_tool_mutation_terminal(
        handle=unsettled,
        summary={
            "provider_cleanup_pending": False,
            "cleanup_pending": False,
        },
        cost_usd=1.0,
        terminal_error=terminal,
        cleanup_error=cleanup,
        accounting_evidence="complete",
        mutation_cleanup_pending=False,
    )
    assert unsettled.summaries[-1]["provider_cleanup_pending"] is True
    assert unsettled.summaries[-1]["cleanup_pending"] is True

    settled = _CheckpointHandle()
    checkpoint_settled_tool_mutation_terminal(
        handle=settled,
        summary={
            "provider_cleanup_pending": True,
            "cleanup_pending": True,
        },
        cost_usd=1.0,
        terminal_error=terminal,
        accounting_evidence="complete",
    )
    assert settled.summaries[-1]["provider_cleanup_pending"] is False
    assert settled.summaries[-1]["cleanup_pending"] is False


def test_tool_mutation_checkpoints_reject_provider_truth_overrides() -> None:
    terminal = RuntimeError("mutation settlement unknown")
    cleanup = RuntimeError("provider descendants still live")

    with pytest.raises(TypeError, match="provider_cleanup_pending"):
        checkpoint_unsettled_tool_mutation_terminal(
            handle=_CheckpointHandle(),
            summary={},
            cost_usd=0.0,
            terminal_error=terminal,
            cleanup_error=cleanup,
            provider_cleanup_pending=False,
        )
    with pytest.raises(TypeError, match="provider_cleanup_pending"):
        checkpoint_settled_tool_mutation_terminal(
            handle=_CheckpointHandle(),
            summary={},
            cost_usd=0.0,
            terminal_error=terminal,
            provider_cleanup_pending=True,
        )


@pytest.mark.parametrize("accounting_evidence", ["", "unknown", "Complete"])
def test_tool_mutation_checkpoints_reject_ambiguous_accounting_evidence(
    accounting_evidence: str,
) -> None:
    handle = _CheckpointHandle()

    with pytest.raises(ValueError, match="accounting_evidence"):
        checkpoint_settled_tool_mutation_terminal(
            handle=handle,
            summary={},
            cost_usd=0.0,
            terminal_error=RuntimeError("mutation settlement unknown"),
            accounting_evidence=accounting_evidence,
        )
    assert handle.summaries == []
    assert handle.results == []


def test_incomplete_accounting_projects_to_inconclusive_outcome(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs" / "accounting-debt"
    run.mkdir(parents=True)
    begin_run(run, workflow="tier")
    _write_json(
        run / "result.json",
        {"status": "incomplete_accounting", "cost_usd": 0.25},
    )
    _write_json(
        run / "summary.json",
        {
            "status": "incomplete_accounting",
            "costs": {
                "total_usd": 0.25,
                "accounting_debt": True,
                "remaining_usd": 0.0,
            },
        },
    )

    outcome = finalize_run(run, spec_version=3)

    assert outcome.detail_status == "incomplete_accounting"
    assert outcome.terminal_class is TerminalClass.INCONCLUSIVE
    persisted = json.loads((run / "outcome.json").read_text())
    assert persisted["detail_status"] == "incomplete_accounting"
    assert persisted["terminal_class"] == "inconclusive"


def test_unjudged_work_statuses_are_typed_and_not_permanent() -> None:
    """Work that nothing judged is never filed under a permanent class."""

    for status in (
        # Tier: a slice, the breach loop, or a gating review without a verdict.
        "slice_inconclusive",
        "breach_inconclusive",
        "review_inconclusive",
        # Breach: a gate that could not run.
        "inconclusive_gate_unavailable",
        # Parents that carry an unjudged child upward.
        "inconclusive_tier_failure",
        "inconclusive_tasks_failed",
        # Planner backend lost; delta of a slice not capturable.
        "planner_infrastructure_failure",
        "hard_fail_capture",
        # Breach's own inconclusive terminals. Four are loops that reached no
        # ruling; ``post_strip_verify`` and ``workspace_mismatch`` record a
        # deterministic refusal, which the tier files as ``breach_failed``.
        "inconclusive_authority",
        "inconclusive_breaker",
        "inconclusive_judge",
        "inconclusive_post_strip_verify",
        "inconclusive_transport_failed",
        "inconclusive_workspace_mismatch",
    ):
        assert classify_terminal_status(status) is TerminalClass.INCONCLUSIVE, status
    for status in ("cancelled", "hard_fail_cancellation"):
        assert classify_terminal_status(status) is TerminalClass.CANCELLED, status


def test_plan_competition_statuses_separate_a_ruling_from_a_run_nothing_ruled_on() -> None:
    """Registering a ruling would move a judged refusal for every caller."""

    # The status the competition keeps for a ruling, and for a run whose
    # records do not prove a lost call, is in no table; neither are the
    # rulings on a fixed candidate.
    for status in (
        "plan_competition_inconclusive",
        "candidate_rejected",
        "candidate_revision_required",
        "candidate_validation_rejected",
    ):
        assert classify_terminal_status(status) is TerminalClass.CONTRACT_ERROR, status
    # Independent drafts that shared provider state, and a fixed candidate
    # whose reviewer answered ``inconclusive`` itself: nothing ruled on a plan.
    for status in (
        "plan_competition_isolation_failed",
        "candidate_review_inconclusive",
    ):
        assert classify_terminal_status(status) is TerminalClass.INCONCLUSIVE, status


def test_tier_breach_failure_statuses_are_permanent() -> None:
    """The tier's two judged breach terminals are registered, not unknown."""

    for status in ("breach_failed", "breach_reject"):
        assert (
            classify_terminal_status(status) is TerminalClass.PERMANENT_FAILURE
        ), status


def test_judged_statuses_stay_permanent() -> None:
    """Typing the unjudged cases must not move a judged failure."""

    for status in (
        "slice_failed",
        "review_failed",
        "incomplete",
        "partial_failed",
        "tasks_failed",
        "hard_fail_smoke",
        "hard_fail_max_turns",
        "hard_fail_mutation_scope",
        "hard_fail_gate_mutation",
        "rounds_exhausted",
        "no_candidate",
    ):
        assert (
            classify_terminal_status(status) is TerminalClass.PERMANENT_FAILURE
        ), status
    for status in ("hard_fail_timeout", "all_slices_transport_failed"):
        assert (
            classify_terminal_status(status) is TerminalClass.RETRYABLE_FAILURE
        ), status


def test_unjudged_terminal_classes_are_the_three_non_verdict_classes() -> None:
    assert outcome_module.UNJUDGED_TERMINAL_CLASSES == frozenset({
        TerminalClass.INCONCLUSIVE,
        TerminalClass.RETRYABLE_FAILURE,
        TerminalClass.CANCELLED,
    })
    assert "UNJUDGED_TERMINAL_CLASSES" in outcome_module.__all__
    # Budget, accounting-free success, and every ruling stay outside the set.
    for judged in (
        TerminalClass.SUCCEEDED,
        TerminalClass.NO_WORK,
        TerminalClass.PERMANENT_FAILURE,
        TerminalClass.BUDGET_EXHAUSTED,
        TerminalClass.INTERRUPTED,
        TerminalClass.CONTRACT_ERROR,
    ):
        assert judged not in outcome_module.UNJUDGED_TERMINAL_CLASSES


def test_tier_unjudged_status_constants_match_the_taxonomy() -> None:
    from crucible.workflows import tier

    expected = {
        tier.STATUS_SLICE_INCONCLUSIVE: TerminalClass.INCONCLUSIVE,
        tier.STATUS_BREACH_INCONCLUSIVE: TerminalClass.INCONCLUSIVE,
        tier.STATUS_REVIEW_INCONCLUSIVE: TerminalClass.INCONCLUSIVE,
        tier.STATUS_CANCELLED: TerminalClass.CANCELLED,
        tier.STATUS_BREACH_FAILED: TerminalClass.PERMANENT_FAILURE,
        tier.STATUS_BREACH_REJECT: TerminalClass.PERMANENT_FAILURE,
    }
    assert {
        tier.STATUS_SLICE_INCONCLUSIVE,
        tier.STATUS_BREACH_INCONCLUSIVE,
        tier.STATUS_REVIEW_INCONCLUSIVE,
        tier.STATUS_CANCELLED,
        tier.STATUS_BREACH_FAILED,
        tier.STATUS_BREACH_REJECT,
    } == {
        "slice_inconclusive",
        "breach_inconclusive",
        "review_inconclusive",
        "cancelled",
        "breach_failed",
        "breach_reject",
    }
    for status, terminal_class in expected.items():
        assert classify_terminal_status(status) is terminal_class, status


def test_unjudged_tier_result_projects_to_an_inconclusive_outcome(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs" / "unjudged-slice"
    run.mkdir(parents=True)
    begin_run(run, workflow="tier")
    _write_json(
        run / "result.json",
        {"status": "slice_inconclusive", "cost_usd": 0.25},
    )
    _write_json(
        run / "summary.json",
        {"status": "slice_inconclusive", "rollback_applied": True},
    )

    outcome = finalize_run(run, spec_version=3)

    assert outcome.detail_status == "slice_inconclusive"
    assert outcome.terminal_class is TerminalClass.INCONCLUSIVE
    persisted = json.loads((run / "outcome.json").read_text())
    assert persisted["detail_status"] == "slice_inconclusive"
    assert persisted["terminal_class"] == "inconclusive"


def test_run_manifest_projects_result_into_typed_outcome(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    begin_run(run, workflow="tier")
    record_provenance_inputs(
        run,
        backend_routing="sha256:" + "b" * 64,
        backend_features="sha256:" + "c" * 64,
        backend_routing_source_sha256="sha256:" + "d" * 64,
    )
    _write_json(run / "result.json", {"status": "complete", "cost_usd": 1.25})
    _write_json(
        run / "summary.json",
        {"status": "complete", "verdict_ref": "sha256:" + "a" * 64},
    )

    outcome = finalize_run(run, spec_version=7)

    assert outcome.terminal_class is TerminalClass.SUCCEEDED
    assert outcome.spec_version == 7
    assert outcome.refs == ("sha256:" + "a" * 64,)
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["state"] == "terminal"
    assert manifest["provenance"]["inputs"]["rubric_manifest"].startswith("sha256:")
    assert manifest["provenance"]["inputs"]["backend_routing"] == (
        "sha256:" + "b" * 64
    )
    assert manifest["provenance"]["inputs"]["backend_features"] == (
        "sha256:" + "c" * 64
    )
    assert manifest["provenance"]["inputs"][
        "backend_routing_source_sha256"
    ] == ("sha256:" + "d" * 64)


def test_provenance_rejects_non_identity_routing_data(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    begin_run(run, workflow="tier")

    with pytest.raises(ValueError, match="SHA-256 identity"):
        record_provenance_inputs(
            run,
            backend_routing="api_key=secret",
            backend_routing_source_sha256=None,
        )

    with pytest.raises(ValueError, match="backend_features"):
        record_provenance_inputs(
            run,
            backend_routing="sha256:" + "b" * 64,
            backend_routing_source_sha256=None,
            backend_features="not-a-digest",
        )

    with pytest.raises(ValueError, match="backend_routing_source_sha256"):
        record_provenance_inputs(
            run,
            backend_routing="sha256:" + "b" * 64,
            backend_routing_source_sha256="not-a-digest",
        )


def test_provenance_route_identity_is_immutable_and_idempotent(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    begin_run(run, workflow="tier")
    routing = "sha256:" + "b" * 64
    features = "sha256:" + "c" * 64
    source = "sha256:" + "d" * 64

    record_provenance_inputs(
        run,
        backend_routing=routing,
        backend_features=features,
        backend_routing_source_sha256=source,
    )
    record_provenance_inputs(
        run,
        backend_routing=routing,
        backend_features=features,
        backend_routing_source_sha256=source,
    )

    with pytest.raises(ValueError, match="backend_routing"):
        record_provenance_inputs(
            run,
            backend_routing="sha256:" + "d" * 64,
            backend_features=features,
            backend_routing_source_sha256=source,
        )
    with pytest.raises(ValueError, match="backend_features"):
        record_provenance_inputs(
            run,
            backend_routing=routing,
            backend_features="sha256:" + "e" * 64,
            backend_routing_source_sha256=source,
        )
    with pytest.raises(ValueError, match="backend_routing_source_sha256"):
        record_provenance_inputs(
            run,
            backend_routing=routing,
            backend_features=features,
            backend_routing_source_sha256="sha256:" + "e" * 64,
        )

    manifest = json.loads((run / "run_manifest.json").read_text())
    inputs = manifest["provenance"]["inputs"]
    assert inputs["backend_routing"] == routing
    assert inputs["backend_features"] == features
    assert inputs["backend_routing_source_sha256"] == source


def test_provenance_retains_builtin_route_source_absence(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    begin_run(run, workflow="tier")
    routing = "sha256:" + "b" * 64

    record_provenance_inputs(
        run,
        backend_routing=routing,
        backend_routing_source_sha256=None,
    )

    with pytest.raises(ValueError, match="backend_routing_source_sha256"):
        record_provenance_inputs(
            run,
            backend_routing=routing,
            backend_routing_source_sha256="sha256:" + "c" * 64,
        )
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["provenance"]["inputs"][
        "backend_routing_source_sha256"
    ] is None


def test_manifest_binds_spec_directives_and_dirty_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=workspace, check=True,
    )
    tracked = workspace / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "base"], cwd=workspace, check=True,
    )
    monkeypatch.chdir(workspace)
    spec = _spec_with_directive()

    first_run = tmp_path / "runs" / "first"
    first_run.mkdir(parents=True)
    first = begin_run(first_run, workflow="tier", spec=spec)
    inputs = first["provenance"]["inputs"]
    first_workspace = first["provenance"]["source"]["workspace"]
    assert inputs["spec"].startswith("sha256:")
    assert inputs["spec_version"] == 0
    assert list(inputs["directives"]) == ["t0/s0"]
    assert inputs["directives"]["t0/s0"].startswith("sha256:")
    assert first_workspace["workspace_state_complete"] is True

    tracked.write_text("changed\n", encoding="utf-8")
    (workspace / "untracked.txt").write_text("new\n", encoding="utf-8")
    second_run = tmp_path / "runs" / "second"
    second_run.mkdir(parents=True)
    second = begin_run(second_run, workflow="tier", spec=spec)
    second_workspace = second["provenance"]["source"]["workspace"]
    assert second_workspace["workspace_state_complete"] is True
    assert (
        second_workspace["workspace_state_sha256"]
        != first_workspace["workspace_state_sha256"]
    )


def test_git_metadata_disables_optional_index_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("retained\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "base"],
        cwd=repository,
        check=True,
    )
    tracked_stat = tracked.stat()
    os.utime(
        tracked,
        ns=(tracked_stat.st_atime_ns, tracked_stat.st_mtime_ns + 2_000_000_000),
    )
    index = repository / ".git" / "index"
    index_bytes = index.read_bytes()
    index_stat = index.stat()
    index_identity = (
        index_stat.st_dev,
        index_stat.st_ino,
        index_stat.st_mode,
        index_stat.st_size,
        index_stat.st_mtime_ns,
        index_stat.st_ctime_ns,
    )
    observed_environments: list[dict[str, str]] = []
    real_run = subprocess.run

    def recording_run(
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        command = args[0] if args else kwargs.get("args")
        if isinstance(command, (list, tuple)) and command and command[0] == "git":
            environment = kwargs.get("env")
            assert isinstance(environment, dict)
            observed_environments.append(environment)
        return real_run(*args, **kwargs)

    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    monkeypatch.setattr(outcome_module.subprocess, "run", recording_run)

    metadata = outcome_module._git_metadata(repository)

    assert metadata["dirty"] is False
    assert observed_environments
    assert all(
        environment.get("GIT_OPTIONAL_LOCKS") == "0"
        for environment in observed_environments
    )
    assert not (repository / ".git" / "index.lock").exists()
    assert index.read_bytes() == index_bytes
    final_index_stat = index.stat()
    assert (
        final_index_stat.st_dev,
        final_index_stat.st_ino,
        final_index_stat.st_mode,
        final_index_stat.st_size,
        final_index_stat.st_mtime_ns,
        final_index_stat.st_ctime_ns,
    ) == index_identity


def test_manifest_explicit_target_is_independent_of_ambient_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def initialize(repository: Path, name: str) -> str:
        repository.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.invalid"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repository,
            check=True,
        )
        (repository / name).write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "add", name], cwd=repository, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "base"], cwd=repository, check=True,
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    ambient = tmp_path / "ambient"
    target = tmp_path / "target"
    initialize(ambient, "ambient.txt")
    target_head = initialize(target, "target.txt")
    monkeypatch.chdir(ambient)

    first_run = tmp_path / "runs" / "first-explicit"
    first_run.mkdir(parents=True)
    first = begin_run(
        first_run,
        workflow="plan",
        provenance_root=target,
    )
    first_workspace = first["provenance"]["source"]["workspace"]
    assert first_workspace["binding"] == "explicit"
    assert first_workspace["root"] == str(target.resolve())
    assert first_workspace["git_root"] == str(target.resolve())
    assert first_workspace["commit"] == target_head
    assert first_workspace["workspace_state_complete"] is True
    assert first_workspace["identity_complete"] is True
    assert first["provenance"]["source"]["ambient"] == {
        "cwd": str(ambient.resolve()),
    }

    (ambient / "ambient.txt").write_text("ambient changed\n", encoding="utf-8")
    (ambient / "untracked.txt").write_text("ambient only\n", encoding="utf-8")
    second_run = tmp_path / "runs" / "second-explicit"
    second_run.mkdir(parents=True)
    second = begin_run(
        second_run,
        workflow="plan",
        provenance_root=target,
    )
    second_workspace = second["provenance"]["source"]["workspace"]
    assert (
        second_workspace["workspace_state_sha256"]
        == first_workspace["workspace_state_sha256"]
    )

    (target / "target.txt").write_text("target changed\n", encoding="utf-8")
    (target / "untracked.txt").write_text("target only\n", encoding="utf-8")
    third_run = tmp_path / "runs" / "third-explicit"
    third_run.mkdir(parents=True)
    third = begin_run(
        third_run,
        workflow="plan",
        provenance_root=target,
    )
    third_workspace = third["provenance"]["source"]["workspace"]
    assert third_workspace["workspace_state_complete"] is True
    assert (
        third_workspace["workspace_state_sha256"]
        != first_workspace["workspace_state_sha256"]
    )


def test_manifest_explicit_none_never_promotes_ambient_git_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=ambient, check=True)
    monkeypatch.chdir(ambient)
    run = tmp_path / "runs" / "no-target"
    run.mkdir(parents=True)

    manifest = begin_run(run, workflow="summary", provenance_root=None)

    workspace = manifest["provenance"]["source"]["workspace"]
    assert workspace == {
        "binding": "none",
        "root": None,
        "git_root": None,
        "commit": None,
        "dirty": None,
        "workspace_state_sha256": None,
        "workspace_state_complete": False,
        "identity_complete": False,
    }
    assert manifest["provenance"]["source"]["ambient"] == {
        "cwd": str(ambient.resolve()),
    }


def test_manifest_rejects_unknown_target_binding_type(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "invalid-target"
    run.mkdir(parents=True)

    with pytest.raises(TypeError, match="provenance_root"):
        begin_run(run, workflow="summary", provenance_root=object())


def test_parent_child_saga_uses_explicit_operation_identity(tmp_path: Path) -> None:
    parent = tmp_path / "runs" / "parent"
    child = tmp_path / "runs" / "child"
    parent.mkdir(parents=True)
    child.mkdir(parents=True)
    begin_run(parent, workflow="architect")

    with child_operation(parent, operation_id="tier-t0", workflow="tier"):
        begin_run(child, workflow="tier")
        _write_json(child / "result.json", {"status": "complete", "cost_usd": 2.0})
        _write_json(child / "summary.json", {"status": "complete"})
        finalize_run(child, spec_version=3)

    loaded = child_outcome(parent, "tier-t0")
    assert loaded.run_id == "child"
    assert loaded.parent_run_id == "parent"
    assert loaded.operation_id == "tier-t0"
    assert child_run_dir(parent, "tier-t0") == child.resolve()


def test_child_outcome_rejects_schema_valid_saga_copy_undercount(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "runs" / "parent"
    child = tmp_path / "runs" / "child"
    parent.mkdir(parents=True)
    child.mkdir(parents=True)
    begin_run(parent, workflow="handoff")
    with child_operation(parent, operation_id="breach-s0", workflow="breach"):
        begin_run(child, workflow="breach")
        _write_json(
            child / "result.json",
            {"status": "proceed", "cost_usd": 7.25},
        )
        _write_json(child / "summary.json", {"status": "proceed"})
        finalize_run(child)

    saga_path = parent / "saga.json"
    saga = json.loads(saga_path.read_text())
    saga["children"]["breach-s0"]["outcome"]["cost_usd"] = 0.0
    _write_json(saga_path, saga)

    with pytest.raises(ValueError, match="differs from its committed outcome"):
        child_outcome(parent, "breach-s0")
    assert json.loads((child / "outcome.json").read_text())["cost_usd"] == 7.25


def test_breach_child_outcome_preserves_exact_implementation_refs(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "runs" / "parent"
    child = tmp_path / "runs" / "breach-child"
    parent.mkdir(parents=True)
    child.mkdir(parents=True)
    owned_ref = "sha256:" + "1" * 64
    unrelated_ref = "sha256:" + "2" * 64
    begin_run(parent, workflow="tier")

    with child_operation(parent, operation_id="breach-t0", workflow="breach"):
        begin_run(child, workflow="breach")
        _write_json(child / "result.json", {"status": "proceed"})
        _write_json(
            child / "summary.json",
            {
                "status": "proceed",
                "implementation_refs": [owned_ref],
                "concurrent_breach_plan_slice_ref": unrelated_ref,
            },
        )
        finalized = finalize_run(child, spec_version=4)

    typed = child_outcome(parent, "breach-t0")
    assert finalized.implementation_refs == (owned_ref,)
    assert typed.implementation_refs == (owned_ref,)
    assert unrelated_ref in typed.refs
    assert unrelated_ref not in typed.implementation_refs
    persisted = json.loads((child / "outcome.json").read_text())
    assert persisted["implementation_refs"] == [owned_ref]


def test_malformed_implementation_refs_fail_terminal_contract(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs" / "breach-invalid-refs"
    run.mkdir(parents=True)
    duplicate = "sha256:" + "3" * 64
    begin_run(run, workflow="breach")
    _write_json(run / "result.json", {"status": "proceed"})
    _write_json(
        run / "summary.json",
        {
            "status": "proceed",
            "implementation_refs": [duplicate, duplicate],
        },
    )

    outcome = finalize_run(run)

    assert outcome.terminal_class is TerminalClass.CONTRACT_ERROR
    assert outcome.detail_status == "contract_error"
    assert outcome.error_kind == "invalid_implementation_refs"
    assert outcome.implementation_refs == ()


def test_parent_child_saga_supports_cross_corpus_runs(tmp_path: Path) -> None:
    parent = tmp_path / "primary" / "runs" / "parent"
    child = tmp_path / "remediation" / "runs" / "child"
    parent.mkdir(parents=True)
    child.mkdir(parents=True)
    begin_run(parent, workflow="regression")

    with child_operation(parent, operation_id="remediation", workflow="plan"):
        begin_run(child, workflow="plan")
        _write_json(child / "result.json", {"status": "complete"})
        _write_json(
            child / "summary.json",
            {
                "status": "complete",
                "evidence_refs": [
                    "sha256:" + "1" * 64,
                    "not-a-ref",
                ],
            },
        )
        outcome = finalize_run(child)

    assert child_run_dir(parent, "remediation") == child.resolve()
    assert outcome.refs == ("sha256:" + "1" * 64,)


def test_startup_reconciliation_marks_dead_exact_owner(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "abandoned"
    run.mkdir(parents=True)
    begin_run(run, workflow="breach")
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["owner"] = {"pid": 999_999_999, "pid_start_token": "dead"}
    _write_json(manifest_path, manifest)

    recovered = reconcile_stale_runs(_Corpus(tmp_path))

    assert recovered == ("abandoned",)
    outcome = json.loads((run / "outcome.json").read_text())
    assert outcome["terminal_class"] == "interrupted"
    assert outcome["detail_status"] == "process_lost"


def test_startup_reconciliation_preserves_valid_debt_result(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs" / "debt"
    run.mkdir(parents=True)
    begin_run(run, workflow="handoff", spec=_spec_with_directive())
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["owner"] = {"pid": 999_999_999, "pid_start_token": "dead"}
    _write_json(manifest_path, manifest)
    _write_json(
        run / "result.json",
        {
            "status": "incomplete_accounting",
            "duration_s": 4.5,
            "cost_usd": 0.625,
            "error_kind": "RuntimeError",
            "error_message": "terminal publication interrupted",
        },
    )
    _write_json(
        run / "summary.json",
        {
            # A pre-terminal checkpoint may disagree with the create-once
            # result.  Recovery must quarantine it, including its refs.
            "status": "pass",
            "cost_usd": 0.0,
            "implementation_refs": ["sha256:" + "9" * 64],
        },
    )

    assert reconcile_stale_runs(_Corpus(tmp_path)) == ("debt",)
    outcome = json.loads((run / "outcome.json").read_text())
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["cost_usd"] == pytest.approx(0.625)
    assert outcome["error_kind"] == "process_lost_after_result"
    assert "RuntimeError" in outcome["error_message"]
    assert outcome["spec_version"] == 0
    assert outcome["implementation_refs"] == []


@pytest.mark.parametrize(
    "invalid_case",
    ("partial_result", "nonterminal_status", "symlink_result"),
)
def test_startup_reconciliation_rejects_untrusted_result(
    tmp_path: Path,
    invalid_case: str,
) -> None:
    run = tmp_path / "runs" / invalid_case
    run.mkdir(parents=True)
    begin_run(run, workflow="handoff")
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["owner"] = {"pid": 999_999_999, "pid_start_token": "dead"}
    _write_json(manifest_path, manifest)
    result = {
        "status": "incomplete_accounting",
        "duration_s": 4.5,
        "cost_usd": 0.625,
        "error_kind": "RuntimeError",
        "error_message": "terminal publication interrupted",
    }
    summary = {
        "status": "incomplete_accounting",
        "cost_usd": 0.625,
    }
    if invalid_case == "partial_result":
        del result["error_message"]
    elif invalid_case == "nonterminal_status":
        result["status"] = "running"
        summary["status"] = "running"
    if invalid_case == "symlink_result":
        _write_json(run / "forged-result.json", result)
        (run / "result.json").symlink_to("forged-result.json")
    else:
        _write_json(run / "result.json", result)
    _write_json(run / "summary.json", summary)

    with pytest.raises(ValueError):
        reconcile_stale_runs(_Corpus(tmp_path))
    assert not (run / "outcome.json").exists()
    assert json.loads(manifest_path.read_text())["state"] == "running"


def test_startup_reconciliation_preserves_exact_live_owner(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "live"
    run.mkdir(parents=True)
    manifest = begin_run(run, workflow="tier")

    assert manifest["owner"]["pid"] == os.getpid()
    assert manifest["owner"]["pid_start_token"]
    assert reconcile_stale_runs(_Corpus(tmp_path)) == ()
    assert not (run / "outcome.json").exists()


def test_invalid_terminal_cost_cannot_publish_success(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "bad-cost"
    run.mkdir(parents=True)
    begin_run(run, workflow="tier")
    _write_json(run / "result.json", {"status": "complete", "cost_usd": -1})
    _write_json(run / "summary.json", {"status": "complete"})

    outcome = finalize_run(run)

    assert outcome.terminal_class is TerminalClass.INCONCLUSIVE
    assert outcome.detail_status == "cost_incomplete"
    assert outcome.error_kind == "invalid_cost"


def test_terminal_outcome_is_idempotent_but_immutable(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "immutable"
    run.mkdir(parents=True)
    begin_run(run, workflow="tier")
    _write_json(run / "result.json", {"status": "complete", "cost_usd": 1})
    _write_json(run / "summary.json", {"status": "complete"})

    first = finalize_run(run, spec_version=2)
    assert finalize_run(run, spec_version=2) == first

    _write_json(run / "result.json", {"status": "failed", "cost_usd": 1})
    with pytest.raises(ValueError, match="immutable"):
        finalize_run(run, spec_version=2)


def test_legacy_outcome_without_implementation_refs_remains_immutable(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs" / "legacy-outcome"
    run.mkdir(parents=True)
    begin_run(run, workflow="tier")
    _write_json(run / "result.json", {"status": "complete", "cost_usd": 1})
    _write_json(run / "summary.json", {"status": "complete"})
    current = finalize_run(run, spec_version=2)
    legacy_payload = current.to_dict()
    legacy_payload.pop("implementation_refs")
    _write_json(run / "outcome.json", legacy_payload)

    parsed = WorkflowOutcome.from_mapping(legacy_payload)
    repeated = finalize_run(run, spec_version=2)

    assert parsed.implementation_refs == ()
    assert repeated == parsed
    assert "implementation_refs" not in json.loads(
        (run / "outcome.json").read_text()
    )


def test_reconciliation_preserves_legacy_parent_saga_without_implementation_refs(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent" / "runs" / "p"
    child_root = tmp_path / "child"
    child = child_root / "runs" / "c"
    parent.mkdir(parents=True)
    child.mkdir(parents=True)
    begin_run(parent, workflow="architect")
    with child_operation(parent, operation_id="tier-t0", workflow="tier"):
        begin_run(child, workflow="tier")
        _write_json(child / "result.json", {"status": "complete"})
        _write_json(child / "summary.json", {"status": "complete"})
        finalize_run(child)

    saga_path = parent / "saga.json"
    saga = json.loads(saga_path.read_text())
    legacy = saga["children"]["tier-t0"]["outcome"]
    legacy.pop("implementation_refs")
    _write_json(saga_path, saga)

    assert reconcile_stale_runs(_Corpus(child_root)) == ()
    persisted = json.loads(saga_path.read_text())
    assert "implementation_refs" not in (
        persisted["children"]["tier-t0"]["outcome"]
    )


def test_reconciliation_completes_parent_saga_for_lost_child(tmp_path: Path) -> None:
    parent = tmp_path / "parent" / "runs" / "p"
    child_root = tmp_path / "child"
    child = child_root / "runs" / "c"
    parent.mkdir(parents=True)
    child.mkdir(parents=True)
    begin_run(parent, workflow="architect")
    with child_operation(parent, operation_id="tier-t0", workflow="tier"):
        begin_run(child, workflow="tier")
    manifest_path = child / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["owner"] = {"pid": 999_999_999, "pid_start_token": "dead"}
    _write_json(manifest_path, manifest)

    assert reconcile_stale_runs(_Corpus(child_root)) == ("c",)
    recovered = child_outcome(parent, "tier-t0")
    assert recovered.terminal_class is TerminalClass.INTERRUPTED


def test_reconciliation_rolls_committed_outcome_forward_without_downgrade(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent" / "runs" / "p"
    child_root = tmp_path / "child"
    child = child_root / "runs" / "c"
    parent.mkdir(parents=True)
    child.mkdir(parents=True)
    begin_run(parent, workflow="architect")
    with child_operation(parent, operation_id="tier-t0", workflow="tier"):
        begin_run(child, workflow="tier")
        _write_json(child / "result.json", {"status": "complete", "cost_usd": 2})
        _write_json(child / "summary.json", {"status": "complete"})
        committed = finalize_run(child, spec_version=4)

    # Model a crash after outcome publication but before both derived writes.
    manifest_path = child / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"state": "running", "completed_at": None})
    manifest.pop("terminal_class")
    manifest.pop("detail_status")
    manifest["owner"] = {"pid": 999_999_999, "pid_start_token": "dead"}
    _write_json(manifest_path, manifest)
    saga_path = parent / "saga.json"
    saga = json.loads(saga_path.read_text())
    saga["children"]["tier-t0"].update({"state": "running", "outcome": None})
    _write_json(saga_path, saga)

    assert reconcile_stale_runs(_Corpus(child_root)) == ("c",)
    assert child_outcome(parent, "tier-t0") == committed


def test_relative_corpus_rerun_binds_fresh_child_without_rewriting_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun gets a new child while prior terminal history stays immutable."""

    from assay.corpus.store import Corpus

    from crucible.workflows import _outcome
    from crucible.workflows._run import open_run_with_capture

    monkeypatch.chdir(tmp_path)
    corpus = Corpus(Path("instance-corpus"))
    operation_id = "architect-instance-1"

    first_parent = Path("parent-runs/first")
    first_parent.mkdir(parents=True)
    begin_run(first_parent, workflow="swebench")
    with child_operation(
        first_parent,
        operation_id=operation_id,
        workflow="architect",
    ):
        with open_run_with_capture(corpus, "architect") as first_handle:
            first_handle.write_summary({"status": "complete"})
    first_child = first_handle.run_dir.resolve()
    assert child_run_dir(first_parent, operation_id) == first_child

    second_parent = Path("parent-runs/second")
    second_parent.mkdir(parents=True)
    begin_run(second_parent, workflow="swebench")

    writes: list[Path] = []
    original_atomic_json = _outcome._atomic_json

    def recording_atomic_json(path: Path, value: object) -> None:
        writes.append(Path(path).resolve())
        original_atomic_json(path, value)

    monkeypatch.setattr(_outcome, "_atomic_json", recording_atomic_json)
    with child_operation(
        second_parent,
        operation_id=operation_id,
        workflow="architect",
    ):
        with open_run_with_capture(corpus, "architect") as second_handle:
            second_handle.write_summary({"status": "complete"})
    second_child = second_handle.run_dir.resolve()

    assert second_child != first_child
    assert child_run_dir(first_parent, operation_id) == first_child
    assert child_run_dir(second_parent, operation_id) == second_child
    assert (first_child / "run_manifest.json").resolve() not in writes
    assert (first_parent / "saga.json").resolve() not in writes


def test_child_operation_identity_cannot_be_reused(tmp_path: Path) -> None:
    parent = tmp_path / "runs" / "parent"
    parent.mkdir(parents=True)
    begin_run(parent, workflow="architect")

    with child_operation(parent, operation_id="tier-t0", workflow="tier"):
        pass

    with pytest.raises(ValueError, match="transition"):
        with child_operation(parent, operation_id="tier-t0", workflow="tier"):
            pass
