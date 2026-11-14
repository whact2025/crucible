# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Standalone implement uses the same scoped durable mutation boundary."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    Implementation,
    InvocationResult,
    Message,
    ProjectContext,
    Spec,
    Usage,
)
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _setup(tmp_path: Path) -> tuple[Path, Corpus]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "implement@test.invalid")
    _git(repo, "config", "user.name", "implement test")
    (repo / "allowed.py").write_text("before\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    directive = ImplementationDirective(
        slice_id="s0",
        intent="edit the declared file",
        files_in_scope=("allowed.py",),
        creates_paths=(),
        verify_mode="skip",
        blast_radius={"max_files": 1},
    )
    spec = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="implement_transaction",
            repo_url="https://example.invalid/repo.git",
            base_commit=_git(repo, "rev-parse", "HEAD"),
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="transaction test",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG(tiers=(Tier(
            tier_id="t0",
            slices=(Slice(id="s0", directive=directive),),
            depends_on=(),
        ),)),
    )
    corpus = Corpus(tmp_path / "corpus")
    corpus.put_spec(spec)
    return repo, corpus


class _Backend:
    name = "implement-transaction-stub"
    model = "stub"

    def __init__(
        self,
        repo: Path,
        *,
        raise_after_write: bool,
        accounting_complete: bool = True,
        cost_usd: float = 0.0,
    ) -> None:
        self.repo = repo
        self.raise_after_write = raise_after_write
        self.accounting_complete = accounting_complete
        self.cost_usd = cost_usd
        self.toolset = None

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        self.toolset = toolset

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        assert self.toolset is not None
        denied = self.toolset.dispatch("write_source", {
            "file_path": "undeclared.py", "content": "denied\n",
        })
        assert "cannot resolve" in denied.lower()
        allowed = self.toolset.dispatch("write_source", {
            "file_path": "allowed.py",
            "content": "after\n",
            "expected_sha256": hashlib.sha256(
                (self.repo / "allowed.py").read_bytes()
            ).hexdigest(),
        })
        assert allowed.startswith("OK:")
        if self.raise_after_write:
            # Simulate a backend/runtime bypass outside the MCP surface. The
            # invoking journal must still discover and remove it on unwind.
            (self.repo / "rogue.py").write_text("rogue\n")
            raise RuntimeError("backend died")
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=Usage(
                input_tokens=11,
                output_tokens=7,
                cost_usd=self.cost_usd,
            ),
            stop_reason="end_turn",
            error=None,
            completeness=ExecutionCompleteness(
                token_usage_complete=self.accounting_complete,
                cost_complete=self.accounting_complete,
                terminal_complete=True,
                trace_complete=True,
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                served_model=self.model,
                events=(BackendExecutionEvent(
                    sequence=0,
                    kind="terminal",
                    name="end_turn",
                ),),
            ),
        )


class _BlockedBackend(_Backend):
    def invoke(self, invocation):
        result = super().invoke(invocation)
        message = "BLOCKED: concrete dependency is unavailable"
        return replace(
            result,
            final_message=message,
            transcript=(Message(role="assistant", content=message),),
        )


class _MutationIndeterminateBackend(_Backend):
    def __init__(self, repo: Path) -> None:
        super().__init__(repo, raise_after_write=False)
        self.invocations = 0

    def invoke(self, invocation):
        from assay.types.invocation import TransportError

        self.invocations += 1
        result = super().invoke(invocation)
        return replace(
            result,
            stop_reason="tool_mutation_indeterminate",
            error=TransportError(
                message="delete publication settlement is unknown",
                retriable=False,
            ),
        )


class _RaisedMutationIndeterminateBackend(_Backend):
    def __init__(self, repo: Path, *, close_settles: bool) -> None:
        super().__init__(repo, raise_after_write=False)
        self.close_settles = close_settles
        self.invocations = 0
        self.close_calls = 0

    def invoke(self, invocation):
        from crucible.tools import ToolMutationIndeterminate

        self.invocations += 1
        super().invoke(invocation)
        raise ToolMutationIndeterminate(
            "delete publication settlement is unknown"
        )

    def close(self) -> None:
        self.close_calls += 1
        if not self.close_settles:
            raise RuntimeError("provider tool worker remains live")


class _ReturnedMutationIndeterminateCloseBackend(
    _MutationIndeterminateBackend
):
    def __init__(self, repo: Path) -> None:
        super().__init__(repo)
        self.close_calls = 0
        self.close_settles = False

    def close(self) -> None:
        self.close_calls += 1
        if not self.close_settles:
            raise RuntimeError("returned-TMI provider remains live")


class _PaidResultCloseBackend(_Backend):
    """Return complete paid usage, then fail provider settlement."""

    def __init__(self, repo: Path) -> None:
        super().__init__(
            repo,
            raise_after_write=False,
            cost_usd=4.25,
        )
        self.close_calls = 0
        self.close_settles = False

    def close(self) -> None:
        self.close_calls += 1
        if not self.close_settles:
            raise RuntimeError("paid provider descendants remain live")


def _patch_backend(monkeypatch, backend: _Backend) -> None:
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )


def test_implement_cleanup_owner_replays_settled_interrupt_after_lock_release(
    monkeypatch,
) -> None:
    """A provider's settled SIGINT cannot skip the retained workspace lock."""

    from crucible.workflows.implement import (
        _ImplementSettlementStack,
        _RetainedCleanupStack,
    )
    from crucible.workflows._provider_settlement import (
        ProviderSettlementExitStack,
    )

    events: list[str] = []
    interrupt = KeyboardInterrupt("provider cleanup interrupted after settle")

    def settle_provider_then_interrupt(self) -> None:
        events.append("provider-settled")
        self._complete = True
        raise interrupt

    monkeypatch.setattr(
        ProviderSettlementExitStack,
        "retry_cleanup",
        settle_provider_then_interrupt,
    )

    workspace = _RetainedCleanupStack()
    workspace.callback(events.append, "workspace-released")
    owner = _ImplementSettlementStack(workspace)

    with pytest.raises(KeyboardInterrupt) as captured:
        owner.retry_cleanup()

    assert captured.value is interrupt
    assert events == ["provider-settled", "workspace-released"]
    assert owner.cleanup_complete


@pytest.mark.parametrize("owner_kind", ("runtime_error", "keyboard_interrupt"))
def test_implement_cleanup_owner_retains_workspace_release_retry(
    owner_kind: str,
) -> None:
    """A failed lock release stays behind the same explicit retry owner."""

    from crucible.workflows.implement import (
        _ImplementSettlementStack,
        _RetainedCleanupStack,
    )

    release_error: BaseException
    if owner_kind == "runtime_error":
        release_error = RuntimeError("workspace release failed")
    else:
        release_error = KeyboardInterrupt("workspace release interrupted")
    fail_release = True
    release_attempts = 0

    def release_workspace() -> None:
        nonlocal release_attempts
        release_attempts += 1
        if fail_release:
            raise release_error

    workspace = _RetainedCleanupStack()
    workspace.callback(release_workspace)
    owner = _ImplementSettlementStack(workspace)
    owner._complete = True

    with pytest.raises(BaseException) as captured:
        owner.retry_cleanup()

    assert captured.value is release_error
    assert not owner.cleanup_complete
    assert release_attempts == 1

    fail_release = False
    owner.retry_cleanup()
    assert release_attempts == 2
    assert owner.cleanup_complete


def test_implement_cleanup_recognition_rejects_foreign_provider_owner() -> None:
    """Only the precomposed owner is accepted as invocation authority."""

    from crucible.workflows.implement import (
        _ImplementSettlementStack,
        _RetainedCleanupStack,
        _owned_provider_cleanup_error,
    )
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    workspace = _RetainedCleanupStack()
    workspace.callback(lambda: None)
    expected_owner = _ImplementSettlementStack(workspace)
    foreign_owner = ProviderSettlementExitStack()
    primary = RuntimeError("foreign provider cleanup failed")
    cleanup_error = ProviderSettlementCleanupError(
        foreign_owner,
        primary_error=primary,
        cleanup_error=primary,
    )

    assert _owned_provider_cleanup_error(cleanup_error, expected_owner) is None
    assert cleanup_error.owner is foreign_owner
    assert not workspace.cleanup_complete


@pytest.mark.parametrize("phase", ("pre", "mid", "post"))
def test_implement_cleanup_recognition_interrupt_retains_precomposed_owner(
    monkeypatch,
    phase: str,
) -> None:
    """SIGINT around recognition keeps the complete owner as direct cause."""

    from crucible.common.cost import CostAccumulator
    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    state = implement_workflow._ImplementFailureState(
        costs=CostAccumulator(max_cost_usd=5.0),
    )
    boundary = implement_workflow._ImplementRunBoundary(
        corpus=object(),  # type: ignore[arg-type]
        cwd=Path("/unused"),
        failure_state=state,
    )
    released: list[bool] = []
    boundary.workspace_release.callback(released.append, True)
    owner = boundary.settlement_stack
    primary = RuntimeError("provider cleanup failed")
    cleanup_error = ProviderSettlementCleanupError(
        owner,
        primary_error=primary,
        cleanup_error=primary,
    )
    interrupt = KeyboardInterrupt(f"recognition interrupted at {phase}")

    exit_calls: list[tuple[object, object, object]] = []

    class RecordingRunContext:
        def __exit__(self, *args):
            exit_calls.append(args)
            return False

    class UnusedDeferred:
        pass

    boundary._run_context = RecordingRunContext()
    boundary._run_entry_started = True
    boundary._deferred = UnusedDeferred()  # type: ignore[assignment]

    original_lookup = implement_workflow._owned_provider_cleanup_error

    if phase in {"pre", "mid"}:
        def interrupted_lookup(error, expected_owner):
            if phase == "mid":
                assert original_lookup(error, expected_owner) is cleanup_error
                assert cleanup_error.owner is owner
            raise interrupt

        monkeypatch.setattr(
            implement_workflow,
            "_owned_provider_cleanup_error",
            interrupted_lookup,
        )
    else:
        def interrupted_mark_cleanup(**_kwargs):
            raise interrupt

        monkeypatch.setattr(state, "mark_cleanup", interrupted_mark_cleanup)

    with pytest.raises(KeyboardInterrupt) as captured:
        boundary.__exit__(
            ProviderSettlementCleanupError,
            cleanup_error,
            cleanup_error.__traceback__,
        )

    assert captured.value is interrupt
    assert captured.value.__cause__ is cleanup_error
    assert cleanup_error.owner is owner
    assert not owner.cleanup_complete
    assert released == []
    assert len(exit_calls) == 1
    assert exit_calls[0][0] is KeyboardInterrupt
    assert exit_calls[0][1] is interrupt


def test_implement_real_recognition_interrupt_closes_run_and_retains_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SIGINT in owner recognition closes capture before explicit retry."""

    repo, corpus = _setup(tmp_path)
    backend = _PaidResultCloseBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    events: list[str] = []
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    interrupt = KeyboardInterrupt("provider-owner recognition interrupted")

    @contextmanager
    def observed_workspace_lock(_corpus, _cwd):
        events.append("workspace_enter")
        try:
            yield object()
        finally:
            events.append("workspace_exit")

    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        observed_workspace_lock,
    )
    original_lookup = implement_workflow._owned_provider_cleanup_error
    lookup_calls = 0

    def interrupt_after_recognition(error, expected_owner):
        nonlocal lookup_calls
        lookup_calls += 1
        retained = original_lookup(error, expected_owner)
        assert isinstance(retained, ProviderSettlementCleanupError)
        raise interrupt

    monkeypatch.setattr(
        implement_workflow,
        "_owned_provider_cleanup_error",
        interrupt_after_recognition,
    )

    with pytest.raises(KeyboardInterrupt) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is interrupt
    assert lookup_calls == 1
    cleanup_owner = captured.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert isinstance(
        cleanup_owner.owner,
        implement_workflow._ImplementSettlementStack,
    )
    assert not cleanup_owner.owner.cleanup_complete
    assert events == ["workspace_enter"]
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == "incomplete_accounting"
    assert result["cost_usd"] == pytest.approx(4.25)
    assert result["error_kind"] == "KeyboardInterrupt"
    assert summary["provider_cleanup_pending"] is True
    assert summary["mutation_cleanup_pending"] is True
    assert summary["workspace_cleanup_pending"] is True
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["error_kind"] == "KeyboardInterrupt"

    backend.close_settles = True
    cleanup_owner.retry_cleanup()
    assert cleanup_owner.owner.cleanup_complete
    assert events == ["workspace_enter", "workspace_exit"]
    assert (repo / "allowed.py").read_text() == "before\n"


def test_implement_real_run_handle_transfer_interrupt_closes_all_owners(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A real post-yield SIGINT cannot leak capture or the workspace lock."""

    repo, corpus = _setup(tmp_path)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow

    events: list[str] = []
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    interrupt = KeyboardInterrupt(
        "run handle transfer interrupted after capture yielded"
    )

    @contextmanager
    def observed_workspace_lock(_corpus, _cwd):
        events.append("workspace_enter")
        try:
            yield object()
        finally:
            events.append("workspace_exit")

    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        observed_workspace_lock,
    )
    source, first_line = inspect.getsourcelines(
        implement_workflow._ImplementRunBoundary.open
    )
    transfer_line = next(
        first_line + offset
        for offset, text in enumerate(source)
        if text.strip()
        == "self._deferred = _DeferredImplementRunHandle(raw_handle)"
    )
    observed_boundary = None

    def interrupt_transfer(frame, event, _arg):
        nonlocal observed_boundary
        if (
            event == "line"
            and frame.f_code
            is implement_workflow._ImplementRunBoundary.open.__code__
            and frame.f_lineno == transfer_line
        ):
            observed_boundary = frame.f_locals["self"]
            assert "raw_handle" in frame.f_locals
            assert observed_boundary._deferred is None
            assert sys.stdout is not original_stdout
            assert sys.stderr is not original_stderr
            sys.settrace(None)
            raise interrupt
        return interrupt_transfer

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_transfer)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            implement_workflow.main([
                "--spec", str(corpus.spec_dir), "--slice", "s0",
            ])
    finally:
        sys.settrace(previous_trace)

    assert captured.value is interrupt
    assert events == ["workspace_enter", "workspace_exit"]
    assert observed_boundary is not None
    assert observed_boundary.workspace_release.cleanup_complete
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == "failed"
    assert result["cost_usd"] == 0.0
    assert result["error_kind"] == "KeyboardInterrupt"
    assert result["error_message"] == str(interrupt)
    assert outcome["detail_status"] == "failed"
    assert outcome["error_kind"] == "KeyboardInterrupt"


def test_standalone_implement_commits_only_declared_write(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus = _setup(tmp_path)
    _patch_backend(monkeypatch, _Backend(repo, raise_after_write=False))
    monkeypatch.chdir(repo)

    from crucible.workflows.implement import main

    assert main(["--spec", str(corpus.spec_dir), "--slice", "s0"]) == 0
    assert (repo / "allowed.py").read_text() == "after\n"
    assert not (repo / "undeclared.py").exists()
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))


def test_standalone_implement_exception_restores_all_process_writes(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus = _setup(tmp_path)
    _patch_backend(monkeypatch, _Backend(repo, raise_after_write=True))
    monkeypatch.chdir(repo)

    from crucible.workflows.implement import main

    with pytest.raises(RuntimeError, match="backend died"):
        main(["--spec", str(corpus.spec_dir), "--slice", "s0"])
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not (repo / "rogue.py").exists()
    assert not (repo / "undeclared.py").exists()
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))


@pytest.mark.parametrize("owner_kind", ("runtime_error", "keyboard_interrupt"))
def test_standalone_cleanup_and_lock_precede_typed_terminal(
    tmp_path: Path,
    monkeypatch,
    owner_kind: str,
) -> None:
    """Exact control flow escapes after provider, journal, and lock cleanup."""

    repo, corpus = _setup(tmp_path)
    events: list[str] = []
    owner: BaseException
    if owner_kind == "runtime_error":
        owner = RuntimeError("implement body failed")
    else:
        owner = KeyboardInterrupt("implement body interrupted")

    class ExactFailureBackend(_Backend):
        def __init__(self) -> None:
            super().__init__(repo, raise_after_write=False)
            self.closed = False

        def invoke(self, invocation):
            super().invoke(invocation)
            (repo / "rogue.py").write_text("rogue\n")
            raise owner

        def close(self) -> None:
            self.closed = True
            events.append("provider-closed")

    backend = ExactFailureBackend()
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows import _outcome
    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._mutation_journal import MutationTransaction

    original_lock = implement_workflow.recovery_workspace_mutation_lock
    original_settle = MutationTransaction.settle_or_restore
    original_finalize = _outcome.finalize_run

    @contextmanager
    def tracking_lock(*args, **kwargs):
        assert not tuple((corpus.spec_dir / "runs").glob("*_implement_*"))
        with original_lock(*args, **kwargs) as authority:
            events.append("workspace-acquired")
            yield authority
        events.append("workspace-released")

    def tracking_settle(self, bound_corpus) -> bool:
        assert backend.closed
        events.append("journal-settled")
        return original_settle(self, bound_corpus)

    def tracking_finalize(*args, **kwargs):
        events.append("typed-terminal-published")
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        tracking_lock,
    )
    monkeypatch.setattr(
        MutationTransaction,
        "settle_or_restore",
        tracking_settle,
    )
    monkeypatch.setattr(_outcome, "finalize_run", tracking_finalize)

    with pytest.raises(BaseException) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is owner
    assert events == [
        "workspace-acquired",
        "provider-closed",
        "journal-settled",
        "workspace-released",
        "typed-terminal-published",
    ]
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not (repo / "rogue.py").exists()
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["error_kind"] == type(owner).__name__


@pytest.mark.parametrize("owner_kind", ("runtime_error", "keyboard_interrupt"))
def test_standalone_workspace_release_failure_is_inside_run_boundary(
    tmp_path: Path,
    monkeypatch,
    owner_kind: str,
) -> None:
    """A lock-release failure keeps identity and cannot leave a success outcome."""

    repo, corpus = _setup(tmp_path)
    _patch_backend(monkeypatch, _Backend(repo, raise_after_write=False))
    monkeypatch.chdir(repo)

    from crucible.workflows import _outcome
    from crucible.workflows import implement as implement_workflow

    owner: BaseException
    if owner_kind == "runtime_error":
        owner = RuntimeError("workspace authority release failed")
    else:
        owner = KeyboardInterrupt("workspace authority release interrupted")
    original_lock = implement_workflow.recovery_workspace_mutation_lock
    original_finalize = _outcome.finalize_run
    events: list[str] = []

    @contextmanager
    def failing_lock(*args, **kwargs):
        with original_lock(*args, **kwargs) as authority:
            events.append("workspace-acquired")
            yield authority
        events.append("workspace-release-failed")
        raise owner

    def tracking_finalize(*args, **kwargs):
        events.append("typed-terminal-published")
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        failing_lock,
    )
    monkeypatch.setattr(
        _outcome,
        "finalize_run",
        tracking_finalize,
    )

    with pytest.raises(BaseException) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is owner
    assert events == [
        "workspace-acquired",
        "workspace-release-failed",
        "typed-terminal-published",
    ]
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["error_kind"] == type(owner).__name__
    assert outcome["implementation_refs"] == []


def test_standalone_interrupt_remains_primary_over_workspace_release_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A settled body SIGINT owns the terminal while lock cleanup stays retryable."""

    repo, corpus = _setup(tmp_path)
    interrupt = KeyboardInterrupt("operator interrupted Implement")
    release_error = RuntimeError("workspace authority release failed")
    release_attempts = 0

    class InterruptingBackend(_Backend):
        def invoke(self, invocation):
            super().invoke(invocation)
            raise interrupt

    _patch_backend(monkeypatch, InterruptingBackend(
        repo,
        raise_after_write=False,
    ))
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow

    original_lock = implement_workflow.recovery_workspace_mutation_lock

    @contextmanager
    def failing_lock(*args, **kwargs):
        nonlocal release_attempts
        with original_lock(*args, **kwargs) as authority:
            try:
                yield authority
            finally:
                release_attempts += 1
                raise release_error

    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        failing_lock,
    )

    with pytest.raises(KeyboardInterrupt) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is interrupt
    assert captured.value.__cause__ is release_error
    owner = getattr(
        captured.value,
        implement_workflow._WORKSPACE_CLEANUP_OWNER_ATTR,
    )
    assert not owner.cleanup_complete
    assert release_attempts == 1
    with pytest.raises(RuntimeError) as repeated:
        owner.retry_cleanup()
    assert repeated.value is release_error
    assert release_attempts == 1
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    assert result["status"] == "incomplete_accounting"
    assert result["error_kind"] == "KeyboardInterrupt"


def test_standalone_settled_tmi_remains_primary_over_workspace_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A staged mutation terminal is not replaced by later lock cleanup."""

    repo, corpus = _setup(tmp_path)
    backend = _MutationIndeterminateBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow

    release_error = RuntimeError("workspace authority release failed")
    release_attempts = 0
    original_lock = implement_workflow.recovery_workspace_mutation_lock

    @contextmanager
    def failing_lock(*args, **kwargs):
        nonlocal release_attempts
        with original_lock(*args, **kwargs) as authority:
            try:
                yield authority
            finally:
                release_attempts += 1
                raise release_error

    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        failing_lock,
    )

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    terminal = captured.value
    assert terminal.__cause__ is release_error
    owner = getattr(
        terminal,
        implement_workflow._WORKSPACE_CLEANUP_OWNER_ATTR,
    )
    assert not owner.cleanup_complete
    assert release_attempts == 1
    with pytest.raises(RuntimeError) as repeated:
        owner.retry_cleanup()
    assert repeated.value is release_error
    assert release_attempts == 1
    assert backend.invocations == 1
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == "incomplete_accounting"
    assert result["error_kind"] == "ToolMutationIndeterminate"
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["implementation_refs"] == []


def test_standalone_workspace_releases_if_run_entry_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Pre-provenance lock acquisition retains an entry-failure guard."""

    repo, corpus = _setup(tmp_path)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow

    owner = KeyboardInterrupt("run capture entry interrupted")
    original_lock = implement_workflow.recovery_workspace_mutation_lock
    events: list[str] = []

    @contextmanager
    def tracking_lock(*args, **kwargs):
        with original_lock(*args, **kwargs) as authority:
            events.append("workspace-acquired")
            yield authority
        events.append("workspace-released")

    @contextmanager
    def failing_open_run(*_args, **_kwargs):
        events.append("run-entry")
        raise owner
        yield  # pragma: no cover - contextmanager shape only

    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        tracking_lock,
    )
    monkeypatch.setattr(
        implement_workflow,
        "open_run_with_capture",
        failing_open_run,
    )

    with pytest.raises(KeyboardInterrupt) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is owner
    assert events == [
        "workspace-acquired",
        "run-entry",
        "workspace-released",
    ]


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_standalone_real_capture_entry_failure_releases_before_publication(
    tmp_path: Path,
    monkeypatch,
    cleanup_fails: bool,
) -> None:
    """The real pre-yield capture failure selector closes workspace first."""

    repo, corpus = _setup(tmp_path)
    monkeypatch.chdir(repo)

    from crucible.workflows import _run as run_workflow
    from crucible.workflows import implement as implement_workflow

    capture_error = RuntimeError("real capture context failed before yield")
    cleanup_error = KeyboardInterrupt("real workspace release interrupted")
    original_lock = implement_workflow.recovery_workspace_mutation_lock
    events: list[str] = []
    release_attempts = 0

    @contextmanager
    def tracking_lock(*args, **kwargs):
        nonlocal release_attempts
        events.append("workspace-enter")
        with original_lock(*args, **kwargs) as authority:
            try:
                yield authority
            finally:
                release_attempts += 1
                events.append("workspace-exit")
                assert not any(
                    (run_dir / "result.json").exists()
                    for run_dir in (corpus.spec_dir / "runs").glob(
                        "*_implement_*"
                    )
                )
                if cleanup_fails:
                    raise cleanup_error

    @contextmanager
    def failing_capture(_run_dir):
        events.append("capture-enter")
        raise capture_error
        yield  # pragma: no cover - contextmanager shape only

    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        tracking_lock,
    )
    monkeypatch.setattr(run_workflow, "capture_to_run_dir", failing_capture)

    with pytest.raises(BaseException) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    expected = cleanup_error if cleanup_fails else capture_error
    assert captured.value is expected
    assert events == ["workspace-enter", "capture-enter", "workspace-exit"]
    assert release_attempts == 1
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == (
        "incomplete_accounting" if cleanup_fails else "failed"
    )
    assert outcome["error_kind"] == type(capture_error).__name__
    if cleanup_fails:
        assert captured.value.__cause__ is capture_error
        owner = getattr(
            captured.value,
            implement_workflow._WORKSPACE_CLEANUP_OWNER_ATTR,
        )
        assert not owner.cleanup_complete
        with pytest.raises(KeyboardInterrupt) as repeated:
            owner.retry_cleanup()
        assert repeated.value is cleanup_error
        assert release_attempts == 1


def test_standalone_generic_entry_failure_never_calls_unentered_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A generic failed entry retains standard context-manager semantics."""

    repo, corpus = _setup(tmp_path)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow

    interrupt = KeyboardInterrupt("interrupted after workspace acquisition")
    events: list[str] = []

    class InterruptedLock:
        def __enter__(self):
            events.append("workspace-acquired")
            raise interrupt

        def __exit__(self, *_exc):
            events.append("workspace-released")
            return False

    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        lambda *_args, **_kwargs: InterruptedLock(),
    )
    monkeypatch.setattr(
        implement_workflow,
        "open_run_with_capture",
        lambda *_args, **_kwargs: pytest.fail(
            "run provenance must not open after interrupted lock admission"
        ),
    )

    with pytest.raises(KeyboardInterrupt) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is interrupt
    assert events == ["workspace-acquired"]
    assert not tuple((corpus.spec_dir / "runs").glob("*_implement_*"))


def test_standalone_retained_lock_return_interrupt_releases_without_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The source-published recovery lease closes the return-value gap."""

    repo, corpus = _setup(tmp_path)
    monkeypatch.chdir(repo)

    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows import implement as implement_workflow

    retained = journals.recovery_workspace_mutation_lock(corpus, repo)
    cleanup = implement_workflow._RetainedCleanupStack()
    enter_code = type(retained).__enter__.__code__
    interrupt = KeyboardInterrupt("retained lock __enter__ return boundary")
    injected = False

    def interrupt_return(frame, event, _arg):
        nonlocal injected
        if not injected and frame.f_code is enter_code and event == "return":
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            cleanup.enter_context(retained)
    finally:
        sys.settrace(previous_trace)

    assert captured.value is interrupt
    assert injected
    cleanup.close()
    assert retained.cleanup_complete
    assert cleanup.cleanup_complete
    assert journals._active_workspace_mutation_authority(repo) is None
    with journals.recovery_workspace_mutation_lock(corpus, repo):
        assert journals._active_workspace_mutation_authority(repo) is not None
    assert not tuple((corpus.spec_dir / "runs").glob("*_implement_*"))


def test_standalone_implement_blocked_report_restores_partial_write(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus = _setup(tmp_path)
    _patch_backend(
        monkeypatch,
        _BlockedBackend(repo, raise_after_write=False),
    )
    monkeypatch.chdir(repo)

    from crucible.workflows.implement import main

    assert main(["--spec", str(corpus.spec_dir), "--slice", "s0"]) == 1
    assert (repo / "allowed.py").read_text() == "before\n"

    committed_spec = corpus.latest_spec()
    assert committed_spec is not None
    implementation = corpus.get_record(
        committed_spec.implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == (
        "hard_fail_implementer_blocked"
    )
    disposition_ref = implementation.artifacts[
        "implementer_terminal_disposition"
    ]
    disposition = json.loads(corpus.get_artifact(disposition_ref))
    assert disposition["disposition"] == "blocked"

    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["implementer_terminal_disposition"] == "blocked"


def test_standalone_indeterminate_mutation_is_permanent_and_restored(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    backend = _MutationIndeterminateBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows.implement import main

    assert main(["--spec", str(corpus.spec_dir), "--slice", "s0"]) == 1
    assert backend.invocations == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))

    committed_spec = corpus.latest_spec()
    assert committed_spec is not None
    implementation = corpus.get_record(
        committed_spec.implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert implementation.stop_reason == "tool_mutation_indeterminate"

    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["classification"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["detail_status"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert outcome["terminal_class"] == "permanent_failure"


def test_standalone_raised_indeterminate_synthesizes_after_quiescence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    backend = _RaisedMutationIndeterminateBackend(
        repo,
        close_settles=True,
    )
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows.implement import main

    assert main(["--spec", str(corpus.spec_dir), "--slice", "s0"]) == 1
    assert backend.invocations == 1
    assert backend.close_calls == 1
    assert (repo / "allowed.py").read_text() == "before\n"

    committed_spec = corpus.latest_spec()
    assert committed_spec is not None
    implementation = corpus.get_record(
        committed_spec.implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert implementation.stop_reason == "tool_mutation_indeterminate"
    assert implementation.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["classification"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["accounting_debt"] is True


def test_standalone_raised_indeterminate_retains_unsettled_cleanup_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    backend = _RaisedMutationIndeterminateBackend(
        repo,
        close_settles=False,
    )
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._mutation_journal import MutationTransaction
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    capture_calls = 0
    workspace_releases = 0
    real_prepare = MutationTransaction.prepare_git_capture
    original_lock = implement_workflow.recovery_workspace_mutation_lock

    @contextmanager
    def tracking_lock(*args, **kwargs):
        nonlocal workspace_releases
        with original_lock(*args, **kwargs) as authority:
            yield authority
        workspace_releases += 1

    def observed_prepare(self) -> None:
        nonlocal capture_calls
        capture_calls += 1
        real_prepare(self)

    monkeypatch.setattr(
        MutationTransaction,
        "prepare_git_capture",
        observed_prepare,
    )
    monkeypatch.setattr(
        implement_workflow,
        "synthesize_tool_mutation_indeterminate_implementation",
        lambda **_kwargs: pytest.fail("unsettled provider must not synthesize"),
    )
    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        tracking_lock,
    )

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    cleanup_owner = captured.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.primary_error is captured.value
    assert backend.invocations == 1
    assert backend.close_calls == 1
    assert capture_calls == 0
    assert workspace_releases == 0
    assert (repo / "allowed.py").read_text() == "after\n"
    assert tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert summary["status"] == "incomplete_accounting"
    assert summary["partial_summary"]["status"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert summary["provider_cleanup_pending"] is True
    assert summary["mutation_cleanup_pending"] is True
    assert summary["journal_cleanup_pending"] is True
    assert summary["cleanup_pending"] is True
    assert summary["accounting_evidence"] == "incomplete"
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["error_kind"] == "ToolMutationIndeterminate"
    assert outcome["implementation_refs"] == []

    backend.close_settles = True
    cleanup_owner.retry_cleanup()
    assert backend.close_calls == 2
    assert workspace_releases == 1
    assert cleanup_owner.owner.cleanup_complete
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))


def test_standalone_returned_indeterminate_close_failure_keeps_known_usage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Returned TMI is latched before Assay's pre-capture provider close."""

    repo, corpus = _setup(tmp_path)
    backend = _ReturnedMutationIndeterminateCloseBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._mutation_journal import MutationTransaction
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    capture_calls = 0
    unsafe_capture_calls = 0
    real_prepare = MutationTransaction.prepare_git_capture

    def observed_prepare(self) -> None:
        nonlocal capture_calls, unsafe_capture_calls
        capture_calls += 1
        if backend.close_calls < 2:
            unsafe_capture_calls += 1
        real_prepare(self)

    monkeypatch.setattr(
        MutationTransaction,
        "prepare_git_capture",
        observed_prepare,
    )

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    cleanup_owner = captured.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.primary_error is captured.value
    assert backend.invocations == 1
    assert backend.close_calls == 1
    assert capture_calls == 0
    assert (repo / "allowed.py").read_text() == "after\n"

    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["provider_cleanup_pending"] is True
    assert summary["mutation_cleanup_pending"] is True
    assert summary["accounting_evidence"] == "incomplete"
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["total_input_tokens"] == 11
    assert summary["costs"]["total_output_tokens"] == 7
    assert summary["costs"]["accounting_debt"] is False

    backend.close_settles = True
    cleanup_owner.retry_cleanup()
    assert backend.close_calls == 2
    assert backend.invocations == 1
    assert unsafe_capture_calls == 0
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))


def test_standalone_paid_result_close_failure_retains_cost_and_debt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Usage returned before provider close remains a durable lower bound."""

    repo, corpus = _setup(tmp_path)
    backend = _PaidResultCloseBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    cleanup_owner = captured.value
    assert backend.close_calls == 1
    assert (repo / "allowed.py").read_text() == "after\n"
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == "incomplete_accounting"
    assert result["cost_usd"] == pytest.approx(4.25)
    assert summary["cost_usd"] == pytest.approx(4.25)
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt_by_gate"] == {
        "implement_provider_cleanup": [
            "cost_complete",
            "token_usage_complete",
        ],
        "implement_workspace_cleanup": [
            "cost_complete",
            "token_usage_complete",
        ],
    }
    assert summary["costs"]["total_usd"] == pytest.approx(4.25)
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["cost_usd"] == pytest.approx(4.25)

    backend.close_settles = True
    cleanup_owner.retry_cleanup()
    assert backend.close_calls == 2
    assert cleanup_owner.owner.cleanup_complete
    assert (repo / "allowed.py").read_text() == "before\n"


def test_standalone_paid_result_persistence_failure_retains_exact_cost(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A post-call persistence error cannot flatten a completed call to zero."""

    repo, corpus = _setup(tmp_path)
    backend = _Backend(
        repo,
        raise_after_write=False,
        cost_usd=4.25,
    )
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow

    persistence_error = RuntimeError("Implementation persistence failed")
    monkeypatch.setattr(
        implement_workflow,
        "persist_and_attach",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(persistence_error),
    )

    with pytest.raises(RuntimeError) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is persistence_error
    assert (repo / "allowed.py").read_text() == "before\n"
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == "failed"
    assert result["cost_usd"] == pytest.approx(4.25)
    assert summary["status"] == "failed"
    assert summary["cost_usd"] == pytest.approx(4.25)
    assert summary.get("accounting_debt") is not True
    assert outcome["cost_usd"] == pytest.approx(4.25)


def test_standalone_usage_projection_failure_is_exact_and_publishes_debt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A projection-only failure is one-shot, exact, and never free-looking."""

    repo, corpus = _setup(tmp_path)
    backend = _Backend(
        repo,
        raise_after_write=False,
        cost_usd=4.25,
    )
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow

    projection_error = RuntimeError("usage projection failed")
    projection_calls = 0

    def fail_projection(_result):
        nonlocal projection_calls
        projection_calls += 1
        raise projection_error

    monkeypatch.setattr(
        implement_workflow,
        "_observed_invocation_usage",
        fail_projection,
    )

    with pytest.raises(RuntimeError) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is projection_error
    assert projection_calls == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == "incomplete_accounting"
    assert result["cost_usd"] == 0.0
    assert result["error_kind"] == "RuntimeError"
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt"] is True
    assert summary["accounting_error_kind"] == "RuntimeError"
    assert summary["accounting_debt_by_gate"] == {
        "implement_accounting": [
            "cost_complete",
            "token_usage_complete",
        ],
    }
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["cost_usd"] == 0.0


def test_standalone_projection_failure_preserves_prior_body_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Secondary telemetry failure cannot replace exact body cancellation."""

    repo, corpus = _setup(tmp_path)
    backend = _Backend(
        repo,
        raise_after_write=False,
        cost_usd=4.25,
    )
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow

    body_error = KeyboardInterrupt("Assay body interrupted after paid return")
    projection_error = RuntimeError("usage projection failed while unwinding")
    projection_calls = 0

    def fail_after_return(*_args, backend, **_kwargs):
        backend.invoke(object())
        raise body_error

    def fail_projection(_result):
        nonlocal projection_calls
        projection_calls += 1
        raise projection_error

    monkeypatch.setattr(implement_workflow, "implement", fail_after_return)
    monkeypatch.setattr(
        implement_workflow,
        "_observed_invocation_usage",
        fail_projection,
    )

    with pytest.raises(KeyboardInterrupt) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is body_error
    assert projection_calls == 1
    assert any(
        "could not record Implement invocation usage" in note
        for note in getattr(body_error, "__notes__", ())
    )
    assert (repo / "allowed.py").read_text() == "before\n"
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["status"] == "incomplete_accounting"
    assert result["error_kind"] == "KeyboardInterrupt"
    assert result["cost_usd"] == 0.0
    assert summary["accounting_error_kind"] == "RuntimeError"
    assert summary["accounting_debt"] is True


@pytest.mark.parametrize("malformed_field", ("usage", "completeness"))
def test_standalone_malformed_telemetry_accessor_preserves_first_exception(
    tmp_path: Path,
    monkeypatch,
    malformed_field: str,
) -> None:
    """A second accessor failure is debt, not a replacement body owner."""

    repo, corpus = _setup(tmp_path)
    first = KeyboardInterrupt(
        f"first {malformed_field} accessor failure in Assay"
    )
    second = RuntimeError(
        f"second {malformed_field} accessor failure in projection"
    )

    class MalformedResult:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped
            self.reads = 0

        def _read(self, name: str):
            if name != malformed_field:
                return getattr(self._wrapped, name)
            self.reads += 1
            raise first if self.reads == 1 else second

        @property
        def usage(self):
            return self._read("usage")

        @property
        def completeness(self):
            return self._read("completeness")

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    class MalformedBackend(_Backend):
        def invoke(self, invocation):
            self.returned = MalformedResult(super().invoke(invocation))
            return self.returned

    backend = MalformedBackend(
        repo,
        raise_after_write=False,
        cost_usd=4.25,
    )
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow

    with pytest.raises(KeyboardInterrupt) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is first
    assert backend.returned.reads == 2
    assert any(
        type(second).__name__ in note and str(second) in note
        for note in getattr(first, "__notes__", ())
    )
    assert (repo / "allowed.py").read_text() == "before\n"
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["status"] == "incomplete_accounting"
    assert result["error_kind"] == "KeyboardInterrupt"
    assert result["cost_usd"] == pytest.approx(
        4.25 if malformed_field == "completeness" else 0.0
    )
    assert summary["accounting_error_kind"] == "RuntimeError"
    assert summary["accounting_debt"] is True
    assert summary["costs"]["total_usd"] == pytest.approx(
        4.25 if malformed_field == "completeness" else 0.0
    )


@pytest.mark.parametrize("terminal_kind", ("body", "tmi"))
def test_standalone_malformed_tokens_retain_known_cost_and_primary_terminal(
    tmp_path: Path,
    monkeypatch,
    terminal_kind: str,
) -> None:
    """Known cost survives token failure without replacing body/TMI identity."""

    repo, corpus = _setup(tmp_path)
    first = KeyboardInterrupt("first input-token accessor failure in Assay")
    second = RuntimeError("second input-token accessor failure in projection")

    class MalformedUsage:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped
            self.input_reads = 0

        @property
        def cost_usd(self):
            return self._wrapped.cost_usd

        @property
        def input_tokens(self):
            self.input_reads += 1
            raise first if self.input_reads == 1 else second

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    class MalformedResult:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped
            self.usage = MalformedUsage(wrapped.usage)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    class MalformedBackend(_Backend):
        def __init__(self) -> None:
            super().__init__(
                repo,
                raise_after_write=False,
                cost_usd=7.25,
            )
            self.invocations = 0

        def invoke(self, invocation):
            from assay.types.invocation import TransportError

            self.invocations += 1
            result = super().invoke(invocation)
            if terminal_kind == "tmi":
                result = replace(
                    result,
                    stop_reason="tool_mutation_indeterminate",
                    error=TransportError(
                        message="mutation acknowledgement is unknown",
                        retriable=False,
                    ),
                )
            self.returned = MalformedResult(result)
            return self.returned

    backend = MalformedBackend()
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow

    with pytest.raises(BaseException) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    if terminal_kind == "body":
        assert captured.value is first
    else:
        assert isinstance(captured.value, ToolMutationIndeterminate)
        assert "mutation" in str(captured.value).lower()
    assert backend.invocations == 1
    assert backend.returned.usage.input_reads == 2
    assert (repo / "allowed.py").read_text() == "before\n"
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["cost_usd"] == pytest.approx(7.25)
    assert summary["costs"]["total_usd"] == pytest.approx(7.25)
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["accounting_debt_by_role"] == {
        "implementer": ["token_usage_complete"],
    }
    assert summary["accounting_error_kind"] == "RuntimeError"


def test_standalone_malformed_pricing_retains_known_usage_and_primary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Untrusted pricing provenance cannot erase valid numeric usage."""

    repo, corpus = _setup(tmp_path)
    primary = KeyboardInterrupt("Assay cancelled after the paid return")

    class MalformedResult:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped

        @property
        def pricing_evidence(self):
            return {"schema": "untrusted-pricing-object"}

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    class MalformedBackend(_Backend):
        def __init__(self) -> None:
            super().__init__(
                repo,
                raise_after_write=False,
                cost_usd=7.25,
            )
            self.invocations = 0

        def invoke(self, invocation):
            self.invocations += 1
            return MalformedResult(super().invoke(invocation))

    backend = MalformedBackend()
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow

    def fail_after_return(*_args, backend, **_kwargs):
        backend.invoke(object())
        raise primary

    monkeypatch.setattr(implement_workflow, "implement", fail_after_return)

    with pytest.raises(KeyboardInterrupt) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is primary
    assert backend.invocations == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["status"] == "incomplete_accounting"
    assert result["cost_usd"] == pytest.approx(7.25)
    assert result["error_kind"] == "KeyboardInterrupt"
    assert summary["costs"]["total_usd"] == pytest.approx(7.25)
    assert summary["costs"]["total_input_tokens"] == 11
    assert summary["costs"]["total_output_tokens"] == 7
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["accounting_debt_by_role"] == {
        "implementer": ["cost_complete"],
    }
    assert summary["accounting_error_kind"] == "TypeError"
    assert summary["accounting_debt_by_gate"]["implement_accounting"] == [
        "cost_complete",
    ]


@pytest.mark.parametrize(
    ("primary_kind", "invalid_flag"),
    (("assay", "true"), ("ki", 1), ("tmi", "false")),
)
def test_standalone_nonboolean_completeness_retains_usage_and_primary(
    tmp_path: Path,
    monkeypatch,
    primary_kind: str,
    invalid_flag: object,
) -> None:
    """Truthy strings/integers are debt, never completeness proof."""

    repo, corpus = _setup(tmp_path)
    assay_error: BaseException = (
        KeyboardInterrupt("Assay cancellation remains primary")
        if primary_kind == "ki"
        else RuntimeError("Assay post-return processing failed")
    )

    class MalformedCompleteness:
        token_usage_complete = invalid_flag
        cost_complete = True
        terminal_complete = True
        trace_complete = True

    class MalformedResult:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped
            self.completeness = MalformedCompleteness()

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    class MalformedBackend(_Backend):
        def __init__(self) -> None:
            super().__init__(
                repo,
                raise_after_write=False,
                cost_usd=7.25,
            )
            self.invocations = 0

        def invoke(self, invocation):
            from assay.types.invocation import TransportError

            self.invocations += 1
            result = super().invoke(invocation)
            if primary_kind == "tmi":
                result = replace(
                    result,
                    stop_reason="tool_mutation_indeterminate",
                    error=TransportError(
                        message="mutation acknowledgement is unknown",
                        retriable=False,
                    ),
                )
            return MalformedResult(result)

    backend = MalformedBackend()
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow

    def fail_after_return(*_args, backend, **_kwargs):
        backend.invoke(object())
        raise assay_error

    monkeypatch.setattr(implement_workflow, "implement", fail_after_return)

    with pytest.raises(BaseException) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    if primary_kind == "tmi":
        assert isinstance(captured.value, ToolMutationIndeterminate)
    else:
        assert captured.value is assay_error
    assert backend.invocations == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["cost_usd"] == pytest.approx(7.25)
    assert summary["costs"]["total_usd"] == pytest.approx(7.25)
    assert summary["costs"]["total_input_tokens"] == 11
    assert summary["costs"]["total_output_tokens"] == 7
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["accounting_debt_by_role"] == {
        "implementer": ["token_usage_complete"],
    }
    assert summary["accounting_error_kind"] == "TypeError"


def test_standalone_projection_and_provider_close_keep_precomposed_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Projection debt cannot detach live provider/workspace retry authority."""

    repo, corpus = _setup(tmp_path)
    backend = _PaidResultCloseBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    projection_error = RuntimeError("paid result projection failed")
    monkeypatch.setattr(
        implement_workflow,
        "_observed_invocation_usage",
        lambda _result: (_ for _ in ()).throw(projection_error),
    )

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    cleanup_error = captured.value
    assert cleanup_error.primary_error is projection_error
    assert isinstance(
        cleanup_error.owner,
        implement_workflow._ImplementSettlementStack,
    )
    assert not cleanup_error.owner.cleanup_complete
    assert backend.close_calls == 1
    assert (repo / "allowed.py").read_text() == "after\n"
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["status"] == "incomplete_accounting"
    assert result["cost_usd"] == 0.0
    assert summary["accounting_error_kind"] == "RuntimeError"
    assert summary["provider_cleanup_pending"] is True
    assert summary["workspace_cleanup_pending"] is True

    backend.close_settles = True
    cleanup_error.retry_cleanup()
    assert backend.close_calls == 2
    assert cleanup_error.owner.cleanup_complete
    assert (repo / "allowed.py").read_text() == "before\n"


def test_standalone_returned_indeterminate_outranks_assay_postinvoke_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Known returned usage survives failure before Implementation creation."""

    repo, corpus = _setup(tmp_path)
    backend = _MutationIndeterminateBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow

    def fail_after_return(*_args, backend, **_kwargs):
        result = backend.invoke(object())
        assert result.stop_reason == "tool_mutation_indeterminate"
        raise OSError("Assay artifact store unavailable after provider return")

    monkeypatch.setattr(implement_workflow, "implement", fail_after_return)

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert isinstance(captured.value.__cause__, OSError)
    assert backend.invocations == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["secondary_error_kind"] == "OSError"
    assert summary["accounting_evidence"] == "complete"
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["total_input_tokens"] == 11
    assert summary["costs"]["total_output_tokens"] == 7


@pytest.mark.parametrize(
    ("phase", "message"),
    (
        ("validation", "validation storage unavailable"),
        ("prepare", "prepare storage unavailable"),
        ("persistence", "persistence storage unavailable"),
    ),
)
def test_standalone_returned_indeterminate_preserves_terminal_over_secondary(
    tmp_path: Path,
    monkeypatch,
    phase: str,
    message: str,
) -> None:
    """Post-provider evidence failures stay secondary to one exact TMI."""

    repo, corpus = _setup(tmp_path)
    backend = _MutationIndeterminateBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._mutation_journal import MutationTransaction

    if phase == "validation":
        monkeypatch.setattr(
            MutationTransaction,
            "validate_implementation",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError(message)
            ),
        )
    elif phase == "prepare":
        monkeypatch.setattr(
            MutationTransaction,
            "prepare",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError(message)
            ),
        )
    else:
        monkeypatch.setattr(
            implement_workflow,
            "persist_and_attach",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError(message)
            ),
        )

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert message in str(captured.value.__cause__)
    assert backend.invocations == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))

    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert summary["status"] == "hard_fail_tool_mutation_indeterminate"
    assert summary["secondary_error_kind"] == "RuntimeError"
    assert summary["accounting_evidence"] == "complete"
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["accounting_debt"] is False
    assert outcome["detail_status"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert outcome["terminal_class"] == "permanent_failure"
    assert outcome["error_kind"] == "ToolMutationIndeterminate"
    assert outcome["implementation_refs"] == []


def test_standalone_raw_indeterminate_storage_failure_is_secondary_after_close(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Unsafe Git/artifact work starts only after the provider is quiescent."""

    repo, corpus = _setup(tmp_path)
    backend = _RaisedMutationIndeterminateBackend(
        repo,
        close_settles=True,
    )
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._mutation_journal import MutationTransaction

    prepare_calls = 0
    real_prepare = MutationTransaction.prepare_git_capture

    def failed_prepare(transaction) -> None:
        nonlocal prepare_calls
        assert backend.close_calls == 1
        if prepare_calls == 0:
            prepare_calls += 1
            raise OSError("post-quiescence Git storage unavailable")
        real_prepare(transaction)

    monkeypatch.setattr(
        MutationTransaction,
        "prepare_git_capture",
        failed_prepare,
    )
    monkeypatch.setattr(
        implement_workflow,
        "synthesize_tool_mutation_indeterminate_implementation",
        lambda **_kwargs: pytest.fail(
            "failed capture preparation must not synthesize artifacts"
        ),
    )

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert isinstance(captured.value.__cause__, OSError)
    assert backend.invocations == 1
    assert backend.close_calls == 1
    assert prepare_calls == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "hard_fail_tool_mutation_indeterminate"
    assert summary["secondary_error_kind"] == "OSError"
    assert summary["accounting_evidence"] == "incomplete"
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1


def test_standalone_raw_indeterminate_synthesis_failure_stays_secondary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    backend = _RaisedMutationIndeterminateBackend(
        repo,
        close_settles=True,
    )
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow

    synthesis_calls = 0

    def failed_synthesis(**_kwargs):
        nonlocal synthesis_calls
        synthesis_calls += 1
        assert backend.close_calls == 1
        raise OSError("terminal artifact storage unavailable")

    monkeypatch.setattr(
        implement_workflow,
        "synthesize_tool_mutation_indeterminate_implementation",
        failed_synthesis,
    )

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert isinstance(captured.value.__cause__, OSError)
    assert synthesis_calls == 1
    assert backend.invocations == 1
    assert backend.close_calls == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["secondary_error_kind"] == "OSError"
    assert summary["accounting_evidence"] == "incomplete"
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1


def test_standalone_indeterminate_rollback_failure_retains_one_retry_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    backend = _MutationIndeterminateBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._mutation_journal import MutationTransaction
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    restore_attempts = 0
    real_restore = MutationTransaction.restore

    def fail_once_then_restore(self, *args, **kwargs) -> bool:
        nonlocal restore_attempts
        restore_attempts += 1
        if restore_attempts == 1:
            return False
        # ``MutationTransaction.restore`` may deliberately recurse once after
        # acquiring its descriptor authority. Count the retained workflow
        # action, not that internal implementation detail.
        monkeypatch.setattr(MutationTransaction, "restore", real_restore)
        return real_restore(self, *args, **kwargs)

    monkeypatch.setattr(
        MutationTransaction,
        "restore",
        fail_once_then_restore,
    )

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    cleanup_owner = captured.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.primary_error is captured.value
    assert restore_attempts == 1
    assert backend.invocations == 1
    assert (repo / "allowed.py").read_text() == "after\n"
    assert tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))

    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["provider_cleanup_pending"] is False
    assert summary["mutation_cleanup_pending"] is True
    assert summary["journal_cleanup_pending"] is True
    assert summary["cleanup_pending"] is True
    assert summary["accounting_evidence"] == "incomplete"
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["accounting_debt"] is False

    cleanup_owner.retry_cleanup()
    assert restore_attempts == 2
    assert backend.invocations == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))


def test_standalone_raw_indeterminate_accounting_failure_never_masks_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Unknown-usage bookkeeping is attempted once before unsafe Git."""

    repo, corpus = _setup(tmp_path)
    backend = _RaisedMutationIndeterminateBackend(
        repo,
        close_settles=False,
    )
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.common.cost import CostAccumulator
    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow
    from crucible.workflows._mutation_journal import MutationTransaction
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    accounting_attempts = 0
    capture_calls = 0

    def failed_accounting(self, **_kwargs) -> None:
        nonlocal accounting_attempts
        accounting_attempts += 1
        raise OSError("usage ledger storage unavailable")

    def observed_capture(_transaction) -> None:
        nonlocal capture_calls
        capture_calls += 1

    monkeypatch.setattr(CostAccumulator, "record", failed_accounting)
    monkeypatch.setattr(
        MutationTransaction,
        "prepare_git_capture",
        observed_capture,
    )

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    cleanup_owner = captured.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.primary_error is captured.value
    assert accounting_attempts == 1
    assert backend.invocations == 1
    assert backend.close_calls == 1
    assert capture_calls == 0
    assert (repo / "allowed.py").read_text() == "after\n"

    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["accounting_error_kind"] == "OSError"
    assert summary["accounting_evidence"] == "incomplete"
    assert summary["provider_cleanup_pending"] is True
    assert summary["mutation_cleanup_pending"] is True
    assert summary["journal_cleanup_pending"] is True
    assert summary["cleanup_pending"] is True

    backend.close_settles = True
    cleanup_owner.retry_cleanup()
    assert accounting_attempts == 1
    assert backend.invocations == 1
    assert (repo / "allowed.py").read_text() == "before\n"


def test_standalone_indeterminate_finalization_failure_keeps_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    backend = _MutationIndeterminateBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from assay.corpus.runs import RunHandle
    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import implement as implement_workflow

    real_finalize = RunHandle.finalize_elapsed
    finalize_calls = 0

    def finalize_then_fail(self, result):
        nonlocal finalize_calls
        finalize_calls += 1
        recorded = real_finalize(self, result)
        raise RuntimeError("terminal finalization acknowledgement failed")

    monkeypatch.setattr(RunHandle, "finalize_elapsed", finalize_then_fail)

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert finalize_calls == 1
    assert backend.invocations == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == "hard_fail_tool_mutation_indeterminate"
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["error_kind"] == "ToolMutationIndeterminate"
    assert outcome["implementation_refs"] == []


def test_standalone_success_finalization_effect_then_error_is_not_retried(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An acknowledged result cannot be republished after an effect error."""

    repo, corpus = _setup(tmp_path)
    backend = _Backend(repo, raise_after_write=False, cost_usd=0.75)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from assay.corpus.runs import RunHandle
    from crucible.workflows import implement as implement_workflow

    publication_error = RuntimeError(
        "success finalization acknowledgement failed"
    )
    real_finalize = RunHandle.finalize_elapsed
    finalize_calls = 0

    def finalize_then_fail(self, result):
        nonlocal finalize_calls
        finalize_calls += 1
        real_finalize(self, result)
        raise publication_error

    monkeypatch.setattr(RunHandle, "finalize_elapsed", finalize_then_fail)

    with pytest.raises(RuntimeError) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value is publication_error
    assert finalize_calls == 1
    assert backend.toolset is not None
    assert (repo / "allowed.py").read_text() == "after\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == "success"
    assert result["cost_usd"] == pytest.approx(0.75)
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["implementation_refs"] == []


def test_standalone_indeterminate_outcome_failure_keeps_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A failed outcome projection cannot replace the settled TMI terminal."""

    repo, corpus = _setup(tmp_path)
    backend = _MutationIndeterminateBackend(repo)
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import _outcome
    from crucible.workflows import implement as implement_workflow

    outcome_error = RuntimeError("typed outcome publication failed")
    monkeypatch.setattr(
        _outcome,
        "finalize_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(outcome_error),
    )

    with pytest.raises(ToolMutationIndeterminate) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    assert captured.value.__cause__ is outcome_error
    assert backend.invocations == 1
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))
    run_dir = next((corpus.spec_dir / "runs").glob("*_implement_*"))
    result = json.loads((run_dir / "result.json").read_text())
    assert result["status"] == "hard_fail_tool_mutation_indeterminate"
    assert result["error_kind"] == "ToolMutationIndeterminate"


def test_standalone_implement_close_failure_retains_retryable_journal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Standalone Implement closes the provider before transaction restore."""

    repo, corpus = _setup(tmp_path)
    events: list[str] = []

    class CloseFailingBackend(_Backend):
        def __init__(self) -> None:
            super().__init__(repo, raise_after_write=True)
            self.allow_close = False
            self.closed = False

        def close(self) -> None:
            events.append("backend_close")
            if not self.allow_close:
                raise RuntimeError("provider tool worker remains live")
            self.closed = True

    backend = CloseFailingBackend()
    _patch_backend(monkeypatch, backend)
    monkeypatch.chdir(repo)

    from crucible.workflows._mutation_journal import MutationTransaction
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )
    from crucible.workflows import implement as implement_workflow

    real_settle = MutationTransaction.settle_or_restore
    original_lock = implement_workflow.recovery_workspace_mutation_lock

    @contextmanager
    def tracking_lock(*args, **kwargs):
        with original_lock(*args, **kwargs) as authority:
            yield authority
        events.append("workspace_release")

    def observed_settle(self, bound_corpus) -> bool:
        events.append("journal_settle")
        assert backend.closed
        return real_settle(self, bound_corpus)

    monkeypatch.setattr(
        MutationTransaction,
        "settle_or_restore",
        observed_settle,
    )
    monkeypatch.setattr(
        implement_workflow,
        "recovery_workspace_mutation_lock",
        tracking_lock,
    )

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        implement_workflow.main([
            "--spec", str(corpus.spec_dir), "--slice", "s0",
        ])

    cleanup_owner = captured.value
    assert isinstance(cleanup_owner.primary_error, RuntimeError)
    assert "backend died" in str(cleanup_owner.primary_error)
    assert events == ["backend_close"]
    assert (repo / "allowed.py").read_text() == "after\n"
    assert (repo / "rogue.py").read_text() == "rogue\n"
    assert tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))
    assert tuple((repo / ".git" / "crucible-mutation-journals").glob("*.json"))

    with pytest.raises(ProviderSettlementCleanupError):
        cleanup_owner.retry_cleanup()
    assert events == ["backend_close", "backend_close"]
    assert (repo / "allowed.py").read_text() == "after\n"

    backend.allow_close = True
    cleanup_owner.retry_cleanup()

    assert events == [
        "backend_close",
        "backend_close",
        "backend_close",
        "journal_settle",
        "workspace_release",
    ]
    assert cleanup_owner.owner.cleanup_complete
    assert (repo / "allowed.py").read_text() == "before\n"
    assert not (repo / "rogue.py").exists()
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))
    registry = repo / ".git" / "crucible-mutation-journals"
    assert not registry.exists() or not tuple(registry.glob("*.json"))


def test_standalone_implement_accounting_debt_restores_inherited_dirt(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus = _setup(tmp_path)
    (repo / "allowed.py").write_text("inherited\n")
    _git(repo, "add", "allowed.py")
    (repo / "ambient.py").write_text("ambient\n")
    pre_status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    pre_worktree_diff = _git(repo, "diff", "--binary")
    pre_index_diff = _git(repo, "diff", "--cached", "--binary")

    _patch_backend(monkeypatch, _Backend(
        repo,
        raise_after_write=False,
        accounting_complete=False,
        cost_usd=0.375,
    ))
    monkeypatch.chdir(repo)

    from crucible.workflows.implement import main

    assert main(["--spec", str(corpus.spec_dir), "--slice", "s0"]) == 1
    assert (repo / "allowed.py").read_text() == "inherited\n"
    assert (repo / "ambient.py").read_text() == "ambient\n"
    assert _git(
        repo, "status", "--porcelain=v1", "--untracked-files=all",
    ) == pre_status
    assert _git(repo, "diff", "--binary") == pre_worktree_diff
    assert _git(repo, "diff", "--cached", "--binary") == pre_index_diff
    assert not tuple(corpus.spec_dir.glob("runs/*/mutations/*/manifest.json"))

    committed_spec = corpus.latest_spec()
    assert committed_spec is not None
    implementation = corpus.get_record(
        committed_spec.implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == "hard_fail_incomplete_evidence"
    assert implementation.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert implementation.cost_usd == pytest.approx(0.375)
    assert implementation.input_tokens == 11
    assert implementation.output_tokens == 7

    run_dirs = sorted(
        path
        for path in (corpus.spec_dir / "runs").iterdir()
        if path.is_dir() and "_implement_" in path.name
    )
    assert len(run_dirs) == 1
    summary = json.loads(
        (run_dirs[0] / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["classification"] == "hard_fail_incomplete_evidence"
    assert summary["costs"]["total_usd"] == pytest.approx(0.375)
    assert summary["costs"]["total_input_tokens"] == 11
    assert summary["costs"]["total_output_tokens"] == 7
    assert summary["costs"]["remaining_usd"] == 0.0
    assert summary["costs"]["accounting_debt"] is True
    assert summary["costs"]["accounting_debt_by_role"] == {
        "implementer": ["cost_complete", "token_usage_complete"],
    }
