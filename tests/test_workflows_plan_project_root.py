# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Plan CLI keeps workspace and project roots at distinct boundaries."""

from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from assay import (
    ImplementationDirective,
    PlanOutcome,
    ProjectContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
)
from assay.corpus.store import Corpus
from assay.corpus.runs import RunHandle

from crucible.backends.config import BackendRouting, RoleConfig
from crucible.backends.features import feature_profile_for_route
from crucible.workflows import plan as plan_workflow
from crucible.workflows._outcome import workspace_state_digest
from crucible.workflows._planner_session import PlannerBoundaryBackend


_GIT_IDENTITY_REDIRECT_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_INDEX_FILE",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG_COUNT",
)


def _spec(
    *,
    submodule_path: str | None,
    base_commit: str = "0" * 40,
) -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="demo",
            repo_url="https://example.invalid/demo.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=submodule_path,
            runners={},
        ),
        execution=None,
        intent="edit the existing crate source",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=None,
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _dag() -> TierDAG:
    directive = ImplementationDirective(
        slice_id="s0",
        intent="edit the existing crate source",
        files_in_scope=("crates/demo/src/lib.rs",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    return TierDAG(tiers=(
        Tier(
            tier_id="t0",
            depends_on=(),
            slices=(Slice(id="s0", directive=directive),),
        ),
    ))


def _accepted(spec: Spec) -> PlanOutcome:
    dag = _dag()
    return PlanOutcome(
        spec=replace(spec, tier_dag=dag),
        candidate_dag=dag,
        candidate_ref=None,
        decision="accepted",
        status="accepted",
        rounds_used=1,
        final_critic_decision="pass",
        final_judge_decision="pass",
        reason="stub",
        cost_usd=0.0,
    )


def _initialize_repository(repository: Path) -> str:
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "planner@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Planner Test"],
        cwd=repository,
        check=True,
    )
    (repository / "anchor.txt").write_text("base\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "base"],
        cwd=repository,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


@pytest.mark.parametrize("backend_name", ("codex", "claude", "kimi"))
def test_plan_review_forces_shared_anchor_and_repair_policy(
    monkeypatch,
    backend_name: str,
) -> None:
    seen: dict[str, object] = {}
    verdict = SimpleNamespace(findings=())

    def _strict_review(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return verdict

    monkeypatch.setattr(plan_workflow, "strict_review", _strict_review)
    monkeypatch.setattr(
        plan_workflow,
        "is_review_infrastructure_failure",
        lambda _verdict: False,
    )
    completed: list[tuple[str, object]] = []

    result = plan_workflow._plan_review(
        completed,
        b"candidate",
        object(),
        kind="critic",
        backend=SimpleNamespace(name=backend_name),
        review_anchor_policy="off",
    )

    assert result is verdict
    assert completed == [("critic", verdict)]
    assert seen["kwargs"]["review_anchor_policy"] == "anchors_only"
    assert seen["kwargs"]["response_retry_budget"] == 2


def test_workspace_state_digest_binds_tracked_and_untracked_changes(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "planner@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Planner Test"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("base\n")
    subprocess.run(
        ["git", "add", "tracked.txt"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "base"],
        cwd=repository,
        check=True,
    )

    clean = workspace_state_digest(repository)
    tracked.write_text("changed\n")
    tracked_change = workspace_state_digest(repository)
    (repository / "untracked.txt").write_text("new\n")
    untracked_change = workspace_state_digest(repository)

    assert clean != tracked_change
    assert tracked_change != untracked_change


@pytest.mark.parametrize("variable", _GIT_IDENTITY_REDIRECT_ENV)
def test_workspace_digest_rejects_ambient_git_identity_redirects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    monkeypatch.setenv(variable, "attacker-controlled")

    with pytest.raises(
        ValueError,
        match="ambient Git redirect environment is unsupported",
    ):
        workspace_state_digest(repository)


def test_workspace_digest_excludes_only_governed_in_repo_plan_state(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "planner@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Planner Test"],
        cwd=repository,
        check=True,
    )
    (repository / "tracked.txt").write_text("base\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "base"],
        cwd=repository,
        check=True,
    )
    spec_dir = repository / "spec"
    run_root = spec_dir / "runs"
    session_root = spec_dir / ".crucible" / "planner-sessions"
    exclusions = (run_root, session_root)
    before = workspace_state_digest(
        repository,
        excluded_roots=exclusions,
    )

    run_root.mkdir(parents=True)
    (run_root / "run.json").write_text("{}\n")
    session_root.mkdir(parents=True)
    (session_root / "handle.json").write_text("{}\n")
    after_governed_state = workspace_state_digest(
        repository,
        excluded_roots=exclusions,
    )
    (spec_dir / "operator-note.txt").write_text("changed\n")
    after_adjacent_change = workspace_state_digest(
        repository,
        excluded_roots=exclusions,
    )

    assert after_governed_state == before
    assert after_adjacent_change != before


@pytest.mark.parametrize("deleted", [False, True])
def test_workspace_digest_rejects_tracked_content_under_exclusion(
    tmp_path: Path,
    *,
    deleted: bool,
) -> None:
    repository = tmp_path / "repository"
    tracked = repository / "spec" / "runs" / "tracked.json"
    tracked.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "planner@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Planner Test"],
        cwd=repository,
        check=True,
    )
    tracked.write_text("{}\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "tracked state"],
        cwd=repository,
        check=True,
    )
    if deleted:
        tracked.unlink()

    with pytest.raises(
        ValueError,
        match="exclusion overlaps tracked/index content",
    ):
        workspace_state_digest(
            repository,
            excluded_roots=(repository / "spec" / "runs",),
        )


def test_workspace_digest_rejects_gitlink_at_exclusion_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    (repository / "spec").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "planner@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Planner Test"],
        cwd=repository,
        check=True,
    )
    (repository / "anchor.txt").write_text("base\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "base"],
        cwd=repository,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{head},spec/runs",
        ],
        cwd=repository,
        check=True,
    )

    with pytest.raises(
        ValueError,
        match="exclusion overlaps tracked/index content",
    ):
        workspace_state_digest(
            repository,
            excluded_roots=(repository / "spec" / "runs",),
        )


@pytest.mark.parametrize("alias_origin", ["inside", "outside"])
def test_workspace_digest_rejects_symlinked_exclusion_alias(
    tmp_path: Path,
    alias_origin: str,
) -> None:
    repository = tmp_path / "repository"
    target = repository / "real-state"
    target.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "planner@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Planner Test"],
        cwd=repository,
        check=True,
    )
    (repository / "anchor.txt").write_text("base\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "base"],
        cwd=repository,
        check=True,
    )
    alias = (
        repository / "state-alias"
        if alias_origin == "inside"
        else tmp_path / "state-alias"
    )
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        workspace_state_digest(
            repository,
            excluded_roots=(alias / "runs",),
        )


@pytest.mark.parametrize(
    ("literal_root", "glob_match"),
    [
        ("state[ab]", "statea"),
        ("state?", "statex"),
        ("state*", "statex"),
    ],
)
def test_workspace_digest_uses_literal_exclusion_pathspecs(
    tmp_path: Path,
    literal_root: str,
    glob_match: str,
) -> None:
    repository = tmp_path / "repository"
    adjacent = repository / glob_match / "runs" / "source.txt"
    adjacent.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "planner@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Planner Test"],
        cwd=repository,
        check=True,
    )
    adjacent.write_text("base\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "base"],
        cwd=repository,
        check=True,
    )
    exclusion = repository / literal_root / "runs"
    before = workspace_state_digest(
        repository,
        excluded_roots=(exclusion,),
    )

    exclusion.mkdir(parents=True)
    (exclusion / "state.json").write_text("{}\n")
    after_self_state = workspace_state_digest(
        repository,
        excluded_roots=(exclusion,),
    )
    adjacent.write_text("changed\n")
    after_adjacent_change = workspace_state_digest(
        repository,
        excluded_roots=(exclusion,),
    )

    assert after_self_state == before
    assert after_adjacent_change != before


def test_plan_cli_resolves_project_for_backend_and_source_tools_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "fabric-usd"
    source = project / "crates" / "demo" / "src" / "lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("pub fn existing() {}\n")

    spec_dir = tmp_path / "corpus"
    Corpus(spec_dir).put_spec(_spec(submodule_path="fabric-usd"))
    seen: dict[str, object] = {}
    source_toolset = object()
    role_backends = {
        "plan_planner": object(),
        "plan_critic": object(),
        "plan_judge": object(),
    }

    class _Pool:
        def get(self, role: str, **kwargs):
            if role == "plan_planner":
                seen["planner_toolset"] = kwargs.get("toolset")
            return role_backends[role]

    @contextmanager
    def _pool(_args, _run_dir, *, cwd=None):
        seen["backend_cwd"] = cwd
        yield _Pool()

    def _source_tools(ctx):
        seen["tool_context"] = ctx
        return source_toolset

    def _unexpected_opt_in_helper(*_args, **_kwargs):
        raise AssertionError(
            "default Plan path called an opt-in session/control helper"
        )

    def _run_plan(spec, **kwargs):
        seen["run_plan_cwd"] = kwargs["cwd"]
        return _accepted(spec)

    def _plan_fingerprint(spec, **kwargs):
        seen["fingerprint_cwd"] = kwargs["cwd"]
        return "sha256:" + "a" * 64

    real_validate = plan_workflow.validate

    def _validate(spec, *, ruleset, ctx):
        seen["rule_context"] = ctx
        return real_validate(spec, ruleset=ruleset, ctx=ctx)

    monkeypatch.setattr(plan_workflow, "backend_pool", _pool)
    monkeypatch.setattr(plan_workflow, "build_source_tools", _source_tools)
    for helper_name in (
        "planner_tool_authority_digest",
        "routing_fingerprint",
        "routing_feature_fingerprint",
        "conversation_value_digest",
        "bind_workflow_control",
        "PlannerBoundaryBackend",
        "workspace_state_digest",
    ):
        monkeypatch.setattr(
            plan_workflow,
            helper_name,
            _unexpected_opt_in_helper,
        )
    monkeypatch.setattr(plan_workflow, "run_plan", _run_plan)
    monkeypatch.setattr(
        plan_workflow,
        "plan_contract_fingerprint",
        _plan_fingerprint,
    )
    monkeypatch.setattr(
        plan_workflow,
        "bind_plan_contract",
        lambda _corpus, staged_spec, *, fingerprint: staged_spec,
    )
    monkeypatch.setattr(plan_workflow, "validate", _validate)
    monkeypatch.setattr(
        RunHandle,
        "elapsed_s",
        property(lambda _handle: 4416.25),
    )
    monkeypatch.setattr(
        plan_workflow,
        "available_rules",
        lambda: ("cited_path_exists",),
    )

    rc = plan_workflow.main([
        "--planner-count", "1",
        "--spec", str(spec_dir),
        "--cwd", str(workspace),
        "--enable-source-tools",
        "--max-validate-replan-attempts", "0",
    ])

    resolved_workspace = workspace.resolve()
    resolved_project = project.resolve()
    assert rc == 0
    assert seen["backend_cwd"] == str(resolved_project)
    assert seen["run_plan_cwd"] == str(resolved_project)
    assert seen["fingerprint_cwd"] == str(resolved_project)
    assert seen["planner_toolset"] is source_toolset
    tool_context = seen["tool_context"]
    assert tool_context.workspace_root == resolved_project
    assert tool_context.repo_root == resolved_project
    rule_context = seen["rule_context"]
    assert rule_context.workspace_root == resolved_workspace
    assert rule_context.submodule_path == "fabric-usd"

    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert summary["status"] == "accepted"
    assert summary["duration_s"] == result["duration_s"] == 4416.25
    assert summary["validate_attempts"] == [{
        "attempt": 0,
        "decision": "pass",
        "passed": True,
        "finding_count": 0,
        "finding_counts": {
            "total": 0,
            "blocking": 0,
            "advisory": 0,
        },
        "findings": [],
    }]
    assert summary["final_finding_counts"] == {
        "total": 0,
        "blocking": 0,
        "advisory": 0,
    }
    workspace_provenance = manifest["provenance"]["source"]["workspace"]
    assert workspace_provenance["binding"] == "explicit"
    assert workspace_provenance["root"] == str(resolved_project)
    assert manifest["provenance"]["source"]["ambient"]["cwd"] != str(
        resolved_project
    )


def test_plan_cli_opt_in_admits_and_wraps_the_post_routing_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec_dir = tmp_path / "corpus"
    Corpus(spec_dir).put_spec(_spec(submodule_path=None))
    seen: dict[str, object] = {}

    class _Planner:
        name = "codex"
        model = "gpt-5.6-sol"
        transport = "legacy_app_server"
        capability = "read_only"
        features = feature_profile_for_route(name, model=model)

    planner = _Planner()
    critic = object()
    judge = object()

    class _Control:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    control = _Control()

    class _Pool:
        routing = BackendRouting(
            default=RoleConfig(
                backend="codex",
                model="gpt-5.6-sol",
                effort="max",
            )
        )

        def get(self, role: str, **kwargs):
            if role == "plan_planner":
                seen["planner_get_kwargs"] = kwargs
                return planner
            return {
                "plan_critic": critic,
                "plan_judge": judge,
            }[role]

    @contextmanager
    def _pool(_args, _run_dir, *, cwd=None):
        seen["backend_cwd"] = cwd
        yield _Pool()

    def _run_plan(spec, **kwargs):
        seen["planner_backend"] = kwargs["planner_backend"]
        return _accepted(spec)

    monkeypatch.setattr(plan_workflow, "backend_pool", _pool)
    monkeypatch.setattr(plan_workflow, "run_plan", _run_plan)
    monkeypatch.setattr(
        plan_workflow,
        "bind_workflow_control",
        lambda *_args, **_kwargs: control,
    )
    monkeypatch.setattr(
        plan_workflow,
        "plan_contract_fingerprint",
        lambda *_args, **_kwargs: "sha256:" + "a" * 64,
    )
    monkeypatch.setattr(
        plan_workflow,
        "workspace_state_digest",
        lambda _root, **_kwargs: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        plan_workflow,
        "bind_plan_contract",
        lambda _corpus, staged_spec, *, fingerprint: staged_spec,
    )

    rc = plan_workflow.main([
        "--planner-count", "1",
        "--spec",
        str(spec_dir),
        "--cwd",
        str(workspace),
        "--no-validate-gate",
        "--planner-session-id",
        "operator-run-1",
        "--enable-planner-control",
    ])

    assert rc == 0
    assert control.close_calls == 1
    assert seen["planner_get_kwargs"] == {
        "capability": "read_only",
        "conversation_mode": "workflow_bound",
    }
    boundary = seen["planner_backend"]
    assert isinstance(boundary, PlannerBoundaryBackend)
    conversation = boundary._conversation  # noqa: SLF001
    assert conversation is not None
    assert conversation.trusted_context_identity[
        "plan_contract_fingerprint"
    ] == "sha256:" + "a" * 64
    assert conversation.trusted_context_identity[
        "workspace_state_digest"
    ] == "sha256:" + "b" * 64
    assert conversation.trusted_context_identity[
        "governed_workspace_exclusions"
    ] == (
        str((spec_dir / "runs").resolve()),
        str(
            (
                spec_dir / ".crucible" / "planner-sessions"
            ).resolve()
        ),
    )


def test_plan_cli_releases_control_before_backend_pool_on_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec_dir = tmp_path / "corpus"
    Corpus(spec_dir).put_spec(_spec(submodule_path=None))
    events: list[str] = []

    class _Planner:
        name = "codex"
        model = "gpt-5.6-sol"
        transport = "legacy_app_server"
        capability = "read_only"
        features = feature_profile_for_route(name, model=model)

    class _Pool:
        routing = BackendRouting(
            default=RoleConfig(
                backend="codex",
                model="gpt-5.6-sol",
                effort="max",
            )
        )

        def get(self, role: str, **_kwargs):
            if role == "plan_planner":
                return _Planner()
            return object()

    class _Control:
        def close(self) -> None:
            events.append("control-close")

    @contextmanager
    def _pool(_args, _run_dir, *, cwd=None):
        del cwd
        try:
            yield _Pool()
        finally:
            events.append("pool-close")

    def _raise_during_plan(*_args, **_kwargs):
        raise RuntimeError("injected planner failure")

    monkeypatch.setattr(plan_workflow, "backend_pool", _pool)
    monkeypatch.setattr(plan_workflow, "run_plan", _raise_during_plan)
    monkeypatch.setattr(
        plan_workflow,
        "bind_workflow_control",
        lambda *_args, **_kwargs: _Control(),
    )
    monkeypatch.setattr(
        plan_workflow,
        "plan_contract_fingerprint",
        lambda *_args, **_kwargs: "sha256:" + "a" * 64,
    )
    monkeypatch.setattr(
        plan_workflow,
        "workspace_state_digest",
        lambda _root, **_kwargs: "sha256:" + "b" * 64,
    )

    with pytest.raises(RuntimeError, match="injected planner failure"):
        plan_workflow.main([
            "--planner-count", "1",
            "--spec",
            str(spec_dir),
            "--cwd",
            str(workspace),
            "--no-validate-gate",
            "--enable-planner-control",
        ])

    assert events == ["control-close", "pool-close"]


def test_plan_cli_rejects_unfingerprintable_opt_in_workspace_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec_dir = tmp_path / "corpus"
    Corpus(spec_dir).put_spec(_spec(submodule_path=None))

    @contextmanager
    def _unexpected_pool(*_args, **_kwargs):
        raise AssertionError(
            "backend acquisition must follow workspace fingerprinting"
        )
        yield  # pragma: no cover

    def _unavailable_workspace(_root, **_kwargs):
        raise ValueError("incomplete workspace state")

    monkeypatch.setattr(plan_workflow, "backend_pool", _unexpected_pool)
    monkeypatch.setattr(
        plan_workflow,
        "plan_contract_fingerprint",
        lambda *_args, **_kwargs: "sha256:" + "a" * 64,
    )
    monkeypatch.setattr(
        plan_workflow,
        "workspace_state_digest",
        _unavailable_workspace,
    )

    rc = plan_workflow.main([
        "--planner-count", "1",
        "--spec",
        str(spec_dir),
        "--cwd",
        str(workspace),
        "--planner-session-id",
        "operator-run-1",
    ])

    assert rc == 2
    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == (
        "invalid_planner_workspace_identity"
    )


@pytest.mark.parametrize(
    "variable",
    ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CONFIG_COUNT"),
)
def test_plan_cli_rejects_git_identity_redirect_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    workspace = tmp_path / "workspace"
    base_commit = _initialize_repository(workspace)
    spec_dir = tmp_path / "corpus"
    Corpus(spec_dir).put_spec(
        _spec(submodule_path=None, base_commit=base_commit)
    )

    @contextmanager
    def _unexpected_pool(*_args, **_kwargs):
        raise AssertionError(
            "backend acquisition must follow Git identity admission"
        )
        yield  # pragma: no cover

    monkeypatch.setattr(plan_workflow, "backend_pool", _unexpected_pool)
    monkeypatch.setenv(variable, "attacker-controlled")

    rc = plan_workflow.main([
        "--planner-count", "1",
        "--spec",
        str(spec_dir),
        "--cwd",
        str(workspace),
        "--planner-session-id",
        "operator-run-1",
    ])

    assert rc == 2
    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == (
        "invalid_planner_workspace_identity"
    )
    assert variable in summary["message"]


@pytest.mark.parametrize("submodule_path", ["../outside", "missing", "file"])
def test_plan_cli_rejects_invalid_project_root_before_backend_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submodule_path: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file").write_text("not a directory\n")
    (tmp_path / "outside").mkdir()
    spec_dir = tmp_path / "corpus"
    Corpus(spec_dir).put_spec(_spec(submodule_path=submodule_path))

    @contextmanager
    def _unexpected_pool(*_args, **_kwargs):
        raise AssertionError("backend acquisition must follow root validation")
        yield  # pragma: no cover

    monkeypatch.setattr(plan_workflow, "backend_pool", _unexpected_pool)

    rc = plan_workflow.main([
        "--planner-count", "1",
        "--spec", str(spec_dir),
        "--cwd", str(workspace),
    ])

    assert rc == 2
    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "invalid_project_root"


def test_plan_cli_rejects_submodule_symlink_that_escapes_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "linked-project").symlink_to(outside, target_is_directory=True)
    spec_dir = tmp_path / "corpus"
    Corpus(spec_dir).put_spec(_spec(submodule_path="linked-project"))

    @contextmanager
    def _unexpected_pool(*_args, **_kwargs):
        raise AssertionError("escaping project must not reach a backend")
        yield  # pragma: no cover

    monkeypatch.setattr(plan_workflow, "backend_pool", _unexpected_pool)

    rc = plan_workflow.main([
        "--planner-count", "1",
        "--spec", str(spec_dir),
        "--cwd", str(workspace),
    ])

    assert rc == 2
