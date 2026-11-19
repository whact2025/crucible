# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows.handoff`` — hand-completion path.

The operator (or an interactively-driven agent) finished a slice's work
in the tree OUTSIDE crucible; ``handoff`` captures the scoped diff and
records a proper Implementation (+ optional landed stamp) so resume /
status see the slice as complete. No backend, no LLM — fully
deterministic against a tmp git repo + Corpus.
"""

from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from assay import (
    ExecutionContext,
    ImplementationDirective,
    Slice,
    Spec,
    Tier,
    TierDAG,
)
from assay.corpus.store import Corpus
from assay.primitives.implement import GitCaptureError, capture_workspace_delta
from assay.types.artifacts import Implementation

from crucible.constraints import AuthoringFinalizationConstraint
from crucible.workflows.handoff import main as handoff_main


_CHILD_COMPOSITION_TESTS_WITH_FAKE_BREACH = frozenset(
    {
        "test_cleanup_exception_after_paid_children_retains_aggregate_cost",
        "test_with_breach_and_review_chain_after_persist",
        "test_typed_breach_outcome_controls_handoff_gate",
        "test_missing_typed_breach_outcome_fails_closed",
        "test_breach_exit_code_and_typed_outcome_mismatch_fails_closed",
        "test_review_pass_is_preserved_when_breach_is_inconclusive",
        "test_incomplete_accounting_breach_revokes_all_later_paid_gates",
        "test_incomplete_child_cleanup_exception_preserves_accounting_debt",
        "test_escaped_child_terminal_preserves_accounting_debt",
        "test_child_provider_owner_retains_parent_cleanup_in_order",
        "test_child_operation_exit_failure_preserves_accounting_debt",
        "test_failed_breach_child_can_rebind_exact_parent_candidate_inode",
        "test_corrupt_child_run_binding_is_unknown_accounting",
        "test_unknown_child_persistence_exception_preserves_accounting_debt",
        "test_gate_failure_marks_failed_and_suppresses_land",
        "test_review_failure_restores_accepted_defender_bytes",
        "test_mutating_post_breach_verifier_restores_operator_bytes",
    }
)


@pytest.fixture(autouse=True)
def _isolate_fake_child_composition_from_binding_preflight(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep legacy fake-child tests focused on Handoff composition.

    The real preflight is covered by the dedicated full-entrypoint rejection
    test below. These tests intentionally use runnerless Specs and replace the
    Breach child, so they have no executable observation channel to validate.
    """

    test_name = getattr(request.node, "originalname", request.node.name)
    if test_name not in _CHILD_COMPOSITION_TESTS_WITH_FAKE_BREACH:
        return
    from crucible.workflows import handoff as handoff_mod

    monkeypatch.setattr(
        handoff_mod,
        "preflight_breach_observation_binding",
        lambda *_args, **_kwargs: None,
    )


def test_child_provider_settlement_owner_is_never_flattened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import handoff
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    owner = ProviderSettlementExitStack()
    cleanup_error = RuntimeError("provider descendants remain live")
    live_owner = ProviderSettlementCleanupError(
        owner,
        primary_error=cleanup_error,
        cleanup_error=cleanup_error,
    )
    monkeypatch.setattr(
        handoff._outcome,
        "child_operation",
        lambda *_args, **_kwargs: nullcontext(),
    )

    def _raise_live_owner() -> int:
        raise live_owner

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        handoff._invoke_typed_child_gate(
            parent_run_dir=tmp_path,
            operation_id="provider-settlement",
            workflow="breach",
            invoke=_raise_live_owner,
        )

    assert captured.value is live_owner


def test_secondary_child_read_failure_preserves_cleanup_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import handoff
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    cleanup_error = RuntimeError("provider descendants remain live")
    live_owner = ProviderSettlementCleanupError(
        ProviderSettlementExitStack(),
        primary_error=cleanup_error,
        cleanup_error=cleanup_error,
    )
    secondary = KeyboardInterrupt("child outcome storage interrupted")
    retained = []
    monkeypatch.setattr(
        handoff._outcome,
        "child_operation",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        handoff._outcome,
        "child_outcome",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(secondary),
    )

    def raise_live_owner() -> int:
        raise live_owner

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        handoff._invoke_typed_child_gate(
            parent_run_dir=tmp_path,
            operation_id="provider-settlement",
            workflow="breach",
            invoke=raise_live_owner,
            on_escaped_terminal=retained.append,
        )

    assert captured.value is live_owner
    assert retained[0].escaped_terminal is True
    assert retained[0].accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )


@pytest.mark.parametrize("owner_kind", ("cleanup_owner", "keyboard_interrupt"))
def test_child_read_owner_is_harvested_before_reraise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_kind: str,
) -> None:
    from crucible.workflows import handoff
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    cleanup_error = RuntimeError("provider descendants remain live")
    if owner_kind == "cleanup_owner":
        owner: BaseException = ProviderSettlementCleanupError(
            ProviderSettlementExitStack(),
            primary_error=cleanup_error,
            cleanup_error=cleanup_error,
        )
    else:
        owner = KeyboardInterrupt("child outcome read interrupted")
    retained = []
    monkeypatch.setattr(
        handoff._outcome,
        "child_operation",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        handoff._outcome,
        "child_outcome",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(owner),
    )

    with pytest.raises(BaseException) as captured:
        handoff._invoke_typed_child_gate(
            parent_run_dir=tmp_path,
            operation_id="provider-settlement",
            workflow="breach",
            invoke=lambda: 0,
            on_escaped_terminal=retained.append,
        )

    assert captured.value is owner
    assert retained[0].escaped_terminal is True
    assert retained[0].accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )


def test_success_child_with_error_fields_cannot_pass_gate() -> None:
    from types import SimpleNamespace

    from crucible.workflows import handoff

    gate = handoff._TypedChildGate(
        workflow="breach",
        exit_code=0,
        outcome=SimpleNamespace(
            workflow="breach",
            terminal_class=handoff._outcome.TerminalClass.SUCCEEDED,
            detail_status="proceed",
            error_kind="ProviderWarning",
            error_message="transport ownership ambiguous",
        ),
        run_dir=None,
        error_kind=None,
        error_message=None,
    )

    assert gate.succeeded is False


def test_composed_provider_owner_preserves_parent_rollback_lifo() -> None:
    from crucible.workflows import handoff

    events: list[str] = []
    fail_gate_restore = True

    class ChildOwner:
        cleanup_complete = False

        def retry_cleanup(self) -> None:
            events.extend(("child-provider", "child-callback"))
            self.cleanup_complete = True

    def restore_gate() -> None:
        nonlocal fail_gate_restore
        events.append("gate-rollback")
        if fail_gate_restore:
            raise RuntimeError("gate rollback remains pending")

    rollback = handoff._RetainedCleanupStack()
    rollback.callback(restore_gate)
    rollback.callback(events.append, "breach-test-rollback")
    workspace = handoff._RetainedCleanupStack()
    workspace.callback(events.append, "workspace-release")
    owner = handoff._HandoffProviderCleanupOwner(
        ChildOwner(),
        rollback.pop_all(),
        workspace.pop_all(),
    )

    with pytest.raises(RuntimeError, match="gate rollback remains pending"):
        owner.retry_cleanup()
    assert events == [
        "child-provider",
        "child-callback",
        "breach-test-rollback",
        "gate-rollback",
    ]

    fail_gate_restore = False
    owner.retry_cleanup()
    assert events == [
        "child-provider",
        "child-callback",
        "breach-test-rollback",
        "gate-rollback",
        "gate-rollback",
        "workspace-release",
    ]
    assert owner.cleanup_complete


_FINALIZATION_OUTPUT_PATHS = (
    "internal/qualifications/build-manifest.json",
    "internal/qualifications/build-manifest.json.sha256",
)


def _finalization_env(
    tmp_path: Path,
    *,
    gate: str = "verify",
    runner: str | None = None,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    qualifications = repo / "internal" / "qualifications"
    qualifications.mkdir(parents=True)
    (repo / _FINALIZATION_OUTPUT_PATHS[0]).write_bytes(b'{"phase":5}\n')
    (repo / _FINALIZATION_OUTPUT_PATHS[1]).write_bytes(b"phase5  manifest\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "manifest predecessor"],
        cwd=repo,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    selected_runner = runner or "tests.fake_finalization_runner"
    runners = (
        {"dev_verify": selected_runner}
        if gate == "dev_verify"
        else {"build": selected_runner, "verify": selected_runner}
    )
    base = Spec.from_yaml(MINIMAL_YAML)
    base = replace(
        base,
        project_context=replace(
            base.project_context,
            base_commit=head,
            runners=runners,
        ),
        constraints=(
            AuthoringFinalizationConstraint(
                slice_id="s0",
                materializes_paths=_FINALIZATION_OUTPUT_PATHS,
                gate=gate,
            ),
        ),
    )
    directive = ImplementationDirective(
        slice_id="s0",
        intent="capture operator source and finalize its manifest pair",
        files_in_scope=("seed.txt", *_FINALIZATION_OUTPUT_PATHS),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 3},
    )
    planned = base.with_tier_dag(TierDAG(tiers=(
        Tier(
            tier_id="t0",
            depends_on=(),
            slices=(Slice(id="s0", directive=directive),),
        ),
    )))
    corpus = Corpus(tmp_path / "corpus")
    corpus.put_spec(base)
    corpus.put_spec(planned)
    return repo, corpus, corpus.spec_dir


def _finalization_outcome():
    from crucible.workflows._build_verify import BuildVerifyOutcome, RunnerResult

    def result(role: str) -> RunnerResult:
        return RunnerResult(
            role=role,
            runner="tests.fake_finalization_runner",
            returncode=0,
            values={"returncode": 0.0},
            raw_output=b'{"values":{"returncode":0.0}}',
            stderr=b"",
            duration_s=0.1,
        )

    return BuildVerifyOutcome(
        "success", build=result("build"), verify=result("verify"),
    )


def test_handoff_source_lease_closes_interrupted_acquisition_return() -> None:
    """A source-published lease closes authority lost at the return boundary."""

    from crucible.workflows import handoff
    from crucible.workflows._mutation_journal import (
        RetainedRecoveryWorkspaceMutationLock,
        WorkspaceMutationLockAcquisitionLease,
    )

    events: list[str] = []
    interrupt = KeyboardInterrupt("interrupted while returning acquired lock")
    cleanup = handoff._RetainedCleanupStack()
    lease = WorkspaceMutationLockAcquisitionLease()

    class InterruptingContext:
        def __enter__(self):
            events.append("acquired")
            lease._publish()
            assert cleanup.cleanup_complete is False
            raise interrupt

        def __exit__(self, *_exc):
            lease._begin_settlement()
            events.append("released")
            return False

    context = RetainedRecoveryWorkspaceMutationLock(
        InterruptingContext(),
        lease,
    )

    with pytest.raises(KeyboardInterrupt) as captured:
        with cleanup:
            cleanup.enter_context(context)

    assert captured.value is interrupt
    assert events == ["acquired", "released"]
    assert cleanup.cleanup_complete


def test_handoff_pending_source_lease_never_enters_context_during_unwind() -> None:
    """Interrupted callback publication cannot start an unentered generator."""

    from crucible.workflows import handoff
    from crucible.workflows._mutation_journal import (
        RetainedRecoveryWorkspaceMutationLock,
        WorkspaceMutationLockAcquisitionLease,
    )

    events: list[str] = []
    interrupt = KeyboardInterrupt("before workspace __enter__")
    lease = WorkspaceMutationLockAcquisitionLease()

    @contextmanager
    def authority():
        events.append("acquired-by-enter")
        lease._publish()
        try:
            yield object()
        finally:
            lease._begin_settlement()
            events.append("released")

    context = RetainedRecoveryWorkspaceMutationLock(authority(), lease)

    class InterruptAfterPublish(handoff._RetainedCleanupStack):
        def callback(self, callback, /, *args, **kwargs):
            result = super().callback(callback, *args, **kwargs)
            raise interrupt

    cleanup = InterruptAfterPublish()
    with pytest.raises(KeyboardInterrupt) as captured:
        with cleanup:
            cleanup.enter_context(context)

    assert captured.value is interrupt
    assert events == []
    assert cleanup.cleanup_complete
    with pytest.raises(RuntimeError, match="not reusable"):
        context.__enter__()
    assert events == []


def test_handoff_retained_stack_closes_lock_at_context_return_boundary(
    env,
) -> None:
    """The caller-retained owner covers the wrapper's ``RETURN_VALUE`` gap."""

    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows import handoff

    repo, corpus, _spec_dir = env
    retained = journals.recovery_workspace_mutation_lock(corpus, repo)
    cleanup = handoff._RetainedCleanupStack()
    enter_code = type(retained).__enter__.__code__
    interrupt = KeyboardInterrupt("retained __enter__ return boundary")
    injected = False

    def interrupt_return(frame, event, _arg):
        nonlocal injected
        if (
            not injected
            and frame.f_code is enter_code
            and event == "return"
        ):
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            with cleanup:
                cleanup.enter_context(retained)
                pytest.fail("interrupted acquisition unexpectedly returned")
    finally:
        sys.settrace(previous_trace)

    assert captured.value is interrupt
    assert retained.cleanup_complete
    assert cleanup.cleanup_complete
    assert journals._active_workspace_mutation_authority(repo) is None
    with journals.recovery_workspace_mutation_lock(corpus, repo):
        assert journals._active_workspace_mutation_authority(repo) is not None


def test_handoff_provider_owner_transfer_defers_interrupt_and_keeps_order() -> None:
    """Owner publication survives SIGINT without duplicate provider cleanup."""

    from crucible.workflows import handoff
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    events: list[str] = []
    fail_rollback = True
    transfer_interrupt = KeyboardInterrupt("interrupted during owner publication")

    class ChildOwner:
        cleanup_complete = False
        cleanup_calls = 0

        def retry_cleanup(self) -> None:
            self.cleanup_calls += 1
            events.append("child-provider-quiesced")
            self.cleanup_complete = True

    class InterruptingCleanupError(ProviderSettlementCleanupError):
        def __init__(self, owner: ChildOwner) -> None:
            self._bound_owner = None
            self.interrupt_publication = False
            self.publication_interrupts = 0
            primary = RuntimeError("child provider remains live")
            super().__init__(
                owner,  # type: ignore[arg-type]
                primary_error=primary,
                cleanup_error=primary,
            )

        @property
        def owner(self):
            return self._bound_owner

        @owner.setter
        def owner(self, value):
            self._bound_owner = value
            if (
                self.interrupt_publication
                and isinstance(value, handoff._HandoffProviderCleanupOwner)
                and self.publication_interrupts == 0
            ):
                self.publication_interrupts += 1
                raise transfer_interrupt

    child = ChildOwner()
    rollback = handoff._RetainedCleanupStack()

    def restore_gate() -> None:
        events.append("parent-gate-rollback")
        if fail_rollback:
            raise RuntimeError("gate rollback remains pending")

    rollback.callback(restore_gate)
    workspace = handoff._RetainedCleanupStack(retain_failures_on_exit=True)
    workspace.callback(events.append, "workspace-released")
    delegation = handoff._DeferHandoffCleanupToProviderOwner(
        rollback,
        workspace,
    )
    cleanup_error = InterruptingCleanupError(child)
    cleanup_error.interrupt_publication = True

    assert delegation.__exit__(
        type(cleanup_error),
        cleanup_error,
        cleanup_error.__traceback__,
    ) is False
    assert cleanup_error.publication_interrupts == 1
    assert isinstance(
        cleanup_error.owner,
        handoff._HandoffProviderCleanupOwner,
    )

    # Lexical unwind observes the published shared delegation and therefore
    # cannot roll back or release the workspace ahead of provider settlement.
    assert rollback.__exit__(type(cleanup_error), cleanup_error, None) is False
    assert workspace.__exit__(type(cleanup_error), cleanup_error, None) is False
    assert events == []

    with pytest.raises(ProviderSettlementCleanupError) as retry_failed:
        cleanup_error.retry_cleanup()
    assert retry_failed.value is cleanup_error
    assert events == [
        "child-provider-quiesced",
        "parent-gate-rollback",
    ]
    assert child.cleanup_calls == 1

    fail_rollback = False
    with pytest.raises(KeyboardInterrupt) as replayed:
        cleanup_error.retry_cleanup()
    assert replayed.value is transfer_interrupt
    assert events == [
        "child-provider-quiesced",
        "parent-gate-rollback",
        "parent-gate-rollback",
        "workspace-released",
    ]
    assert child.cleanup_calls == 1
    assert cleanup_error.owner.cleanup_complete


def test_handoff_provider_owner_getter_interrupt_cannot_release_parent_first(
) -> None:
    """An owner-read interrupt is retained behind the precomposed owner."""

    from crucible.workflows import handoff
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    events: list[str] = []
    transfer_interrupt = KeyboardInterrupt("while reading child owner")

    class ChildOwner:
        cleanup_complete = False
        cleanup_calls = 0

        def retry_cleanup(self) -> None:
            self.cleanup_calls += 1
            events.append("child-provider-quiesced")
            self.cleanup_complete = True

    class GetterInterrupt(ProviderSettlementCleanupError):
        def __init__(self, owner: ChildOwner) -> None:
            self._bound_owner = owner
            self.reads = 0
            primary = RuntimeError("child provider remains live")
            super().__init__(
                owner,  # type: ignore[arg-type]
                primary_error=primary,
                cleanup_error=primary,
            )
            self.reads = 0

        @property
        def owner(self):
            self.reads += 1
            if self.reads == 1:
                raise transfer_interrupt
            return self._bound_owner

        @owner.setter
        def owner(self, value):
            self._bound_owner = value

    child = ChildOwner()
    rollback = handoff._RetainedCleanupStack()
    rollback.callback(events.append, "parent-rollback")
    workspace = handoff._RetainedCleanupStack(retain_failures_on_exit=True)
    workspace.callback(events.append, "workspace-released")
    delegation = handoff._DeferHandoffCleanupToProviderOwner(
        rollback,
        workspace,
    )
    cleanup_error = GetterInterrupt(child)

    assert delegation.__exit__(type(cleanup_error), cleanup_error, None) is False
    assert cleanup_error.reads == 2
    assert cleanup_error.owner is delegation.cleanup_owner

    assert rollback.__exit__(type(cleanup_error), cleanup_error, None) is False
    assert workspace.__exit__(type(cleanup_error), cleanup_error, None) is False
    assert events == []

    with pytest.raises(KeyboardInterrupt) as replayed:
        cleanup_error.retry_cleanup()
    assert replayed.value is transfer_interrupt
    assert events == [
        "child-provider-quiesced",
        "parent-rollback",
        "workspace-released",
    ]
    assert child.cleanup_calls == 1
    assert cleanup_error.owner.cleanup_complete


def test_handoff_unexpected_owner_transfer_error_exposes_retry_owner() -> None:
    """Unexpected transfer failure cannot strand fail-closed parent cleanup."""

    from crucible.workflows import handoff
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    events: list[str] = []
    fail_getter = True

    class ChildOwner:
        cleanup_complete = False

        def retry_cleanup(self) -> None:
            events.append("child-provider-quiesced")
            self.cleanup_complete = True

    class GetterFailure(ProviderSettlementCleanupError):
        def __init__(self, owner: ChildOwner) -> None:
            self._bound_owner = owner
            primary = RuntimeError("child provider remains live")
            super().__init__(
                owner,  # type: ignore[arg-type]
                primary_error=primary,
                cleanup_error=primary,
            )

        @property
        def owner(self):
            if fail_getter:
                raise RuntimeError("owner lookup unavailable")
            return self._bound_owner

        @owner.setter
        def owner(self, value):
            self._bound_owner = value

    child = ChildOwner()
    rollback = handoff._RetainedCleanupStack()
    rollback.callback(events.append, "parent-rollback")
    workspace = handoff._RetainedCleanupStack(retain_failures_on_exit=True)
    workspace.callback(events.append, "workspace-released")
    delegation = handoff._DeferHandoffCleanupToProviderOwner(
        rollback,
        workspace,
    )
    cleanup_error = GetterFailure(child)

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        delegation.__exit__(type(cleanup_error), cleanup_error, None)
    transfer_error = captured.value
    assert transfer_error is not cleanup_error
    assert transfer_error.owner is delegation.cleanup_owner

    assert rollback.__exit__(type(transfer_error), transfer_error, None) is False
    assert workspace.__exit__(type(transfer_error), transfer_error, None) is False
    assert events == []

    fail_getter = False
    transfer_error.retry_cleanup()
    assert events == [
        "child-provider-quiesced",
        "parent-rollback",
        "workspace-released",
    ]
    assert transfer_error.owner.cleanup_complete


def test_handoff_loop_boundary_interrupt_publishes_precomposed_retry_owner(
) -> None:
    """A KI outside the transfer try still escapes with reachable authority."""

    import inspect

    from crucible.workflows import handoff
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    events: list[str] = []
    transfer_interrupt = KeyboardInterrupt("at provider transfer loop boundary")

    class ChildOwner:
        cleanup_complete = False

        def retry_cleanup(self) -> None:
            events.append("child-provider-quiesced")
            self.cleanup_complete = True

    child = ChildOwner()
    primary = RuntimeError("child provider remains live")
    provider_error = ProviderSettlementCleanupError(
        child,  # type: ignore[arg-type]
        primary_error=primary,
        cleanup_error=primary,
    )
    rollback = handoff._RetainedCleanupStack()
    rollback.callback(events.append, "parent-rollback")
    workspace = handoff._RetainedCleanupStack(retain_failures_on_exit=True)
    workspace.callback(events.append, "workspace-released")
    delegation = handoff._DeferHandoffCleanupToProviderOwner(
        rollback,
        workspace,
    )

    exit_method = handoff._DeferHandoffCleanupToProviderOwner.__exit__
    source_lines, source_start = inspect.getsourcelines(exit_method)
    loop_line = next(
        source_start + offset
        for offset, source_line in enumerate(source_lines)
        if source_line.strip() == "while True:"
    )
    injected = False

    def interrupt_loop_boundary(frame, event, _arg):
        nonlocal injected
        if (
            not injected
            and event == "line"
            and frame.f_code is exit_method.__code__
            and frame.f_lineno == loop_line
        ):
            injected = True
            raise transfer_interrupt
        return interrupt_loop_boundary

    previous_trace = sys.gettrace()
    with pytest.raises(ProviderSettlementCleanupError) as captured:
        sys.settrace(interrupt_loop_boundary)
        try:
            with workspace, rollback, delegation:
                raise provider_error
        finally:
            sys.settrace(previous_trace)

    retained = captured.value
    assert retained is not provider_error
    assert retained.owner is delegation.cleanup_owner
    assert provider_error.owner is child
    assert events == []

    with pytest.raises(KeyboardInterrupt) as replayed:
        retained.retry_cleanup()
    assert replayed.value is transfer_interrupt
    assert provider_error.owner is retained.owner
    assert events == [
        "child-provider-quiesced",
        "parent-rollback",
        "workspace-released",
    ]
    assert retained.owner.cleanup_complete


def test_handoff_loop_boundary_interrupt_without_provider_settles_parents(
) -> None:
    """A pre-classification KI can prove the provider side was absent."""

    import inspect

    from crucible.workflows import handoff
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    events: list[str] = []
    transfer_interrupt = KeyboardInterrupt("at empty transfer loop boundary")
    rollback = handoff._RetainedCleanupStack()
    rollback.callback(events.append, "parent-rollback")
    workspace = handoff._RetainedCleanupStack(retain_failures_on_exit=True)
    workspace.callback(events.append, "workspace-released")
    delegation = handoff._DeferHandoffCleanupToProviderOwner(
        rollback,
        workspace,
    )

    exit_method = handoff._DeferHandoffCleanupToProviderOwner.__exit__
    source_lines, source_start = inspect.getsourcelines(exit_method)
    loop_line = next(
        source_start + offset
        for offset, source_line in enumerate(source_lines)
        if source_line.strip() == "while True:"
    )
    injected = False

    def interrupt_loop_boundary(frame, event, _arg):
        nonlocal injected
        if (
            not injected
            and event == "line"
            and frame.f_code is exit_method.__code__
            and frame.f_lineno == loop_line
        ):
            injected = True
            raise transfer_interrupt
        return interrupt_loop_boundary

    previous_trace = sys.gettrace()
    with pytest.raises(ProviderSettlementCleanupError) as captured:
        sys.settrace(interrupt_loop_boundary)
        try:
            with workspace, rollback, delegation:
                pass
        finally:
            sys.settrace(previous_trace)

    retained = captured.value
    assert retained.owner is delegation.cleanup_owner
    assert events == []

    with pytest.raises(KeyboardInterrupt) as replayed:
        retained.retry_cleanup()
    assert replayed.value is transfer_interrupt
    assert events == ["parent-rollback", "workspace-released"]
    assert retained.owner.cleanup_complete


@pytest.mark.parametrize("owner_kind", ("runtime_error", "keyboard_interrupt"))
def test_workspace_lock_exit_failure_is_inside_run_terminal_boundary(
    env,
    monkeypatch: pytest.MonkeyPatch,
    owner_kind: str,
) -> None:
    import crucible.workflows.handoff as handoff_mod

    repo, _corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate before lock exit failure\n")
    if owner_kind == "runtime_error":
        owner: BaseException = RuntimeError("workspace authority release failed")
    else:
        owner = KeyboardInterrupt("workspace authority release interrupted")
    original_lock = handoff_mod.recovery_workspace_mutation_lock

    @contextmanager
    def failing_lock(*args, **kwargs):
        with original_lock(*args, **kwargs) as authority:
            yield authority
        raise owner

    monkeypatch.setattr(
        handoff_mod,
        "recovery_workspace_mutation_lock",
        failing_lock,
    )

    with pytest.raises(BaseException) as captured:
        handoff_main([
            "--spec",
            str(spec_dir),
            "--slice",
            "s0",
            "--cwd",
            str(repo),
            "--skip-verify",
        ])

    assert captured.value is owner
    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    outcome = json.loads((handoff_run / "outcome.json").read_text())
    result = json.loads((handoff_run / "result.json").read_text())
    assert result["status"] == "complete"
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["error_kind"] == type(owner).__name__
    assert outcome["implementation_refs"] == []


def test_workspace_lock_releases_if_run_capture_entry_fails(
    env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.workflows.handoff as handoff_mod

    repo, _corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate before run entry failure\n")
    owner = KeyboardInterrupt("run capture entry interrupted")
    original_lock = handoff_mod.recovery_workspace_mutation_lock
    release_count = 0

    @contextmanager
    def tracking_lock(*args, **kwargs):
        nonlocal release_count
        with original_lock(*args, **kwargs) as authority:
            try:
                yield authority
            finally:
                release_count += 1

    @contextmanager
    def failing_open_run(*_args, **_kwargs):
        raise owner
        yield  # pragma: no cover - contextmanager shape only

    monkeypatch.setattr(
        handoff_mod,
        "recovery_workspace_mutation_lock",
        tracking_lock,
    )
    monkeypatch.setattr(handoff_mod, "open_run_with_capture", failing_open_run)

    with pytest.raises(KeyboardInterrupt) as captured:
        handoff_main([
            "--spec",
            str(spec_dir),
            "--slice",
            "s0",
            "--cwd",
            str(repo),
            "--skip-verify",
        ])

    assert captured.value is owner
    assert release_count == 1


@pytest.mark.parametrize(
    ("status", "terminal_class"),
    (
        (" INCOMPLETE_ACCOUNTING ", "INCONCLUSIVE"),
        ("cost_incomplete", "INCONCLUSIVE"),
        ("budget_accounting_incomplete", "INCONCLUSIVE"),
        ("process_lost", "INTERRUPTED"),
        ("brand_new_status", "CONTRACT_ERROR"),
    ),
)
def test_incomplete_accounting_status_is_normalized_for_paid_authority(
    status: str,
    terminal_class: str,
) -> None:
    from types import SimpleNamespace

    from crucible.workflows import handoff

    gate = handoff._TypedChildGate(
        workflow="breach",
        exit_code=3,
        outcome=SimpleNamespace(
            detail_status=status,
            terminal_class=getattr(handoff._outcome.TerminalClass, terminal_class),
        ),
        run_dir=None,
        error_kind=None,
        error_message=None,
    )

    assert gate.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert gate.revokes_paid_authority is True


MINIMAL_YAML = """
schema_version: 1
project_context:
  project: handoff-test
  repo_url: https://example.invalid/x.git
  base_commit: "0000000"
  branch: main
  submodule_path: null
  runners: {}
intent: |
  Hand-completion test spec.
constraints: []
done_criteria: []
"""


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def _slice(sid: str, files: tuple[str, ...], creates: tuple[str, ...] = ()):
    return Slice(
        id=sid,
        directive=ImplementationDirective(
            slice_id=sid,
            intent=f"do {sid}",
            files_in_scope=files,
            creates_paths=creates,
            verify_mode="build_only",
            blast_radius={},
        ),
    )


def _spec_with_dag() -> Spec:
    spec = Spec.from_yaml(MINIMAL_YAML)
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                depends_on=(),
                slices=(
                    _slice("s0", ("seed.txt",), creates=("gen.py",)),
                    _slice("s1", ("other.txt",)),
                ),
            ),
        )
    )
    return spec.with_tier_dag(dag)


@pytest.fixture()
def env(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    corpus = Corpus(tmp_path / "corpus")
    planned = _spec_with_dag()
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    corpus.put_spec(base.with_tier_dag(planned.tier_dag))
    return repo, corpus, tmp_path / "corpus"


def _latest_impls(corpus: Corpus) -> dict[str, Implementation]:
    out: dict[str, Implementation] = {}
    for ref in corpus.list_artifacts(kind="implementations"):
        impl = corpus.get_record(ref, as_type=Implementation)
        out[impl.slice_id] = impl
    return out


def _publish_child_outcome(
    corpus: Corpus,
    *,
    workflow: str,
    status: str,
    cost_usd: float = 0.0,
) -> None:
    """Publish the saga-bound outcome required by Handoff child composition."""

    from assay.corpus.runs import RunResult

    from crucible.workflows._run import open_run_with_capture

    with open_run_with_capture(
        corpus,
        workflow,
        provenance_root=None,
    ) as handle:
        handle.write_summary({"status": status, "cost_usd": cost_usd})
        handle.finalize_elapsed(RunResult(
            status=status,
            cost_usd=cost_usd,
        ))


def _gate_outcome(classification: str):
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )

    build = None
    if classification.startswith("success"):
        build = RunnerResult(
            role="build",
            runner="tests.fake_build",
            returncode=0,
            values={},
            raw_output=b'{"values":{}}',
            stderr=b"",
            duration_s=0.1,
        )
    return BuildVerifyOutcome(classification, build=build, verify=None)


def _contract_aware_gate_stub(callback):
    """Model a synchronous gate that owns and settles its writer boundary."""

    def settled(*args, **kwargs):
        outcome = callback(*args, **kwargs)
        from crucible.workflows import _build_verify

        _build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_handoff_gate_stub",
            succeeded=True,
        )
        return outcome

    return settled


def _settle_fake_breach_writer_epoch(argv, *, phase: str) -> None:
    """Give a synchronous Breach child fake its exact parent-family proof."""

    from crucible.workflows import _build_verify, _mutation_journal

    parent_id = argv[
        argv.index("--parent-mutation-transaction-id") + 1
    ]
    parent = _mutation_journal._require_live_parent_transaction(  # noqa: SLF001
        parent_id
    )
    _build_verify._settle_no_process_writer_epoch(
        _mutation_journal.live_writer_epoch_family(parent),
        phase=phase,
        succeeded=True,
    )


def test_candidate_change_before_gate_settles_nested_writer_family(env) -> None:
    """A rejected pre-dispatch candidate must not strand a mixed epoch."""

    from crucible.workflows import _build_verify
    from crucible.workflows._mutation_journal import (
        MutationTransaction,
        live_writer_epoch_family,
        workspace_mutation_lock,
    )
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows.handoff import _run_readonly_gate

    repo, corpus, _spec_dir = env
    spec = corpus.latest_spec()
    scope = ("seed.txt",)
    expected = capture_workspace_delta(cwd=str(repo), path_scope=scope)

    with workspace_mutation_lock(repo):
        parent = MutationTransaction.begin(
            cwd=repo,
            journal_root=corpus.spec_dir / "test_handoff_parent_mutations",
            slice_id="handoff-parent",
            scope=scope,
            sequence=1,
            spec=spec,
        )
        try:
            _build_verify._settle_no_process_writer_epoch(
                live_writer_epoch_family(parent),
                phase="test_handoff_parent_generation",
                succeeded=True,
            )
            (repo / "seed.txt").write_text(
                "changed after candidate capture\n",
                encoding="utf-8",
            )

            outcome, integrity_error = _run_readonly_gate(
                lambda **_kwargs: pytest.fail(
                    "a stale candidate must be rejected before gate dispatch"
                ),
                expected_candidate=expected,
                cwd=str(repo),
                journal_root=(
                    corpus.spec_dir / "test_handoff_child_mutations"
                ),
                scope=scope,
                sequence=2,
                spec=spec,
                parent_transaction_id=parent.transaction_id,
            )

            assert outcome is None
            assert integrity_error is not None
            assert "changed after candidate capture" in integrity_error
            assert parent.active
        finally:
            if parent.active:
                assert parent.restore()

    assert (repo / "seed.txt").read_text(encoding="utf-8") == "seed\n"


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_interrupted_supervised_handoff_gate_cleans_journal(
    env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authenticated cancellation permits exact Handoff checkpoint cleanup."""

    from crucible.workflows import _build_verify
    from crucible.workflows.handoff import _run_readonly_gate

    repo, corpus, _spec_dir = env
    spec = corpus.latest_spec()
    scope = ("seed.txt",)
    candidate_bytes = "seed\noperator candidate\n"
    (repo / "seed.txt").write_text(candidate_bytes, encoding="utf-8")
    expected = capture_workspace_delta(cwd=str(repo), path_scope=scope)
    target_marker = repo.parent / "handoff-interrupt-target.pid"
    journal_root = corpus.spec_dir / "test_interrupted_handoff_mutations"

    def interrupt_after_target_start(*_args, **_kwargs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if target_marker.exists():
                raise KeyboardInterrupt
            time.sleep(0.005)
        pytest.fail("Handoff gate target did not start before cancellation")

    monkeypatch.setattr(
        _build_verify,
        "_bounded_runner_communicate",
        interrupt_after_target_start,
    )

    def interrupted_gate(*, writer_epoch_members):
        return _build_verify._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                (
                    "import os,pathlib,sys,time; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(30)"
                ),
                str(target_marker),
            ],
            config_json=b"",
            cwd=str(repo),
            timeout=35,
            env=os.environ.copy(),
            writer_epoch_members=writer_epoch_members,
        )

    with pytest.raises(KeyboardInterrupt):
        _run_readonly_gate(
            interrupted_gate,
            expected_candidate=expected,
            cwd=str(repo),
            journal_root=journal_root,
            scope=scope,
            sequence=1,
            spec=spec,
        )

    assert not psutil.pid_exists(
        int(target_marker.read_text(encoding="ascii"))
    )
    assert (repo / "seed.txt").read_text(encoding="utf-8") == candidate_bytes
    assert capture_workspace_delta(cwd=str(repo), path_scope=scope) == expected
    assert not tuple(journal_root.rglob("manifest.json"))
    marker_root = repo / ".git" / "crucible-mutation-journals"
    assert not marker_root.exists() or not tuple(marker_root.glob("*.json"))


def _verification_only_env(tmp_path: Path) -> tuple[Path, Corpus, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    corpus = Corpus(tmp_path / "corpus")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    corpus.put_spec(
        base.with_tier_dag(
            TierDAG(
                tiers=(
                    Tier(
                        tier_id="t0",
                        depends_on=(),
                        slices=(_slice("verify-only", ()),),
                    ),
                )
            )
        )
    )
    return repo, corpus, tmp_path / "corpus"


def test_dry_run_previews_without_writing(env, capsys):
    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("seed\nedited by hand\n")
    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0",
        "--cwd", str(repo), "--dry-run", "--skip-verify",
    ])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "dry-run"
    (row,) = [r for r in summary["targets"] if r["slice"] == "s0"]
    assert row["action"] == "would-record"
    assert "seed.txt" in row["files_modified"]
    # Nothing persisted.
    assert _latest_impls(corpus) == {}
    assert corpus.latest_spec().implementation_refs == ()


def test_spec_bound_handoff_finalizes_before_cumulative_capture(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _build_verify, _gate_evidence, handoff
    from crucible.workflows._mutation_journal import discover_mutation_journals

    repo, corpus, spec_dir = _finalization_env(tmp_path)
    (repo / "seed.txt").write_text("operator postimage\n", encoding="utf-8")
    gate_families = []

    def finalizing_gate(*_args, **kwargs):
        gate_families.append(kwargs["writer_epoch_members"])
        assert kwargs["gate"] == "verify"
        (repo / _FINALIZATION_OUTPUT_PATHS[0]).write_bytes(b'{"phase":6}\n')
        (repo / _FINALIZATION_OUTPUT_PATHS[1]).write_bytes(b"phase6  manifest\n")
        _build_verify._settle_no_process_writer_epoch(
            kwargs["writer_epoch_members"],
            phase="test_handoff_authoring_finalization",
            succeeded=True,
        )
        return _finalization_outcome()

    monkeypatch.setattr(handoff, "build_and_verify", finalizing_gate)
    prior_version = corpus.latest_spec().version

    rc = handoff_main([
        "--spec", str(spec_dir),
        "--slice", "s0",
        "--cwd", str(repo),
        "--operator", "test-operator",
    ])

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    (row,) = summary["targets"]
    assert row["authoring_finalization"] == {
        "gate": "verify",
        "materializes_paths": list(_FINALIZATION_OUTPUT_PATHS),
        "status": "succeeded",
    }
    assert len(gate_families) == 1
    parent, child = gate_families[0]
    assert parent.transaction_id != child.transaction_id
    assert child.parent_transaction_id == parent.transaction_id
    latest = corpus.latest_spec()
    assert latest.version == prior_version + 1
    implementation = corpus.get_record(
        latest.implementation_refs[-1],
        as_type=Implementation,
    )
    observed_diff, observed_paths, observed_base = capture_workspace_delta(
        cwd=repo,
        path_scope=("seed.txt", *_FINALIZATION_OUTPUT_PATHS),
    )
    assert corpus.get_artifact(implementation.diff_ref) == observed_diff
    assert implementation.files_modified == observed_paths
    assert implementation.base_commit == observed_base
    assert set(implementation.files_modified) == {
        "seed.txt",
        *_FINALIZATION_OUTPUT_PATHS,
    }
    evidence = _gate_evidence.load_verification_evidence(
        corpus,
        str(implementation.artifacts[_gate_evidence.HANDOFF_VERIFICATION_ARTIFACT]),
    )
    assert evidence.candidate_diff_sha256 == str(implementation.diff_ref)
    assert evidence.satisfies_required_gate
    finalization_ref = implementation.artifacts[
        _gate_evidence.HANDOFF_AUTHORIZING_FINALIZATION_ARTIFACT
    ]
    finalization_evidence = (
        _gate_evidence.load_handoff_authoring_finalization_evidence(
            corpus,
            str(finalization_ref),
        )
    )
    assert finalization_evidence.output_sha256 == tuple(
        (path, "sha256:" + hashlib.sha256((repo / path).read_bytes()).hexdigest())
        for path in _FINALIZATION_OUTPUT_PATHS
    )
    from crucible.workflows import _partition

    partition = _partition.partition_for_required_verification(
        corpus,
        latest.tier_dag.tiers[0],
        spec=latest,
    )
    assert tuple(partition.verified_retained) == ("s0",)
    assert not partition.verification_pending
    assert partition.implementation_pending == ()

    wrong_output_ref = (
        _gate_evidence.persist_handoff_authoring_finalization_evidence(
            corpus,
            tier_id="t0",
            slice_id="s0",
            spec_execution_authority_hash=(
                _partition.spec_execution_structure_hash(latest)
            ),
            candidate_diff_ref=implementation.diff_ref,
            base_commit=implementation.base_commit,
            files_modified=implementation.files_modified,
            gate="verify",
            output_sha256=finalization_evidence.output_sha256[:1],
            transcript_ref=implementation.transcript_ref,
            verification_evidence_ref=(
                implementation.artifacts[
                    _gate_evidence.HANDOFF_VERIFICATION_ARTIFACT
                ]
            ),
        )
    )
    wrong_output = replace(
        implementation,
        artifacts={
            **(implementation.artifacts or {}),
            _gate_evidence.HANDOFF_AUTHORIZING_FINALIZATION_ARTIFACT: (
                wrong_output_ref
            ),
        },
    )
    wrong_output = _partition.attach_resume_identity(
        corpus,
        latest,
        latest.tier_dag.tiers[0],
        latest.tier_dag.tiers[0].slices[0],
        wrong_output,
    )
    corpus.commit(implementations=(wrong_output,))
    wrong_output_spec = corpus.latest_spec()
    wrong_output_partition = _partition.partition_for_required_verification(
        corpus,
        wrong_output_spec.tier_dag.tiers[0],
        spec=wrong_output_spec,
    )
    assert not wrong_output_partition.verified_retained
    assert tuple(wrong_output_partition.verification_pending) == ("s0",)

    stripped = replace(
        implementation,
        artifacts={
            key: value for key, value in (implementation.artifacts or {}).items()
            if key != _gate_evidence.HANDOFF_AUTHORIZING_FINALIZATION_ARTIFACT
        },
    )
    stripped = _partition.attach_resume_identity(
        corpus,
        wrong_output_spec,
        wrong_output_spec.tier_dag.tiers[0],
        wrong_output_spec.tier_dag.tiers[0].slices[0],
        stripped,
    )
    corpus.commit(implementations=(stripped,))
    stripped_spec = corpus.latest_spec()
    stripped_partition = _partition.partition_for_required_verification(
        corpus,
        stripped_spec.tier_dag.tiers[0],
        spec=stripped_spec,
    )
    assert not stripped_partition.verified_retained
    assert tuple(stripped_partition.verification_pending) == ("s0",)
    assert stripped_partition.implementation_pending == ()
    assert not discover_mutation_journals(corpus, workspace=repo)


def test_spec_bound_handoff_finalizes_with_configured_dev_verify_gate(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _build_verify, _gate_evidence, handoff

    runner = "tests.fake_dev_verify"
    repo, corpus, spec_dir = _finalization_env(
        tmp_path,
        gate="dev_verify",
        runner=runner,
    )
    (repo / "seed.txt").write_text("operator postimage\n", encoding="utf-8")

    def finalizing_gate(*_args, **kwargs):
        assert kwargs["gate"] == "dev_verify"
        (repo / _FINALIZATION_OUTPUT_PATHS[0]).write_bytes(b'{"phase":6}\n')
        (repo / _FINALIZATION_OUTPUT_PATHS[1]).write_bytes(b"phase6  manifest\n")
        _build_verify._settle_no_process_writer_epoch(
            kwargs["writer_epoch_members"],
            phase="test_handoff_dev_verify_finalization",
            succeeded=True,
        )
        result = _build_verify.RunnerResult(
            role="dev_verify",
            runner=runner,
            returncode=0,
            values={"returncode": 0.0},
            raw_output=b'{"values":{"returncode":0.0}}',
            stderr=b"",
            duration_s=0.1,
        )
        return _build_verify.BuildVerifyOutcome(
            "success",
            build=None,
            verify=None,
            dev_verify=result,
        )

    monkeypatch.setattr(handoff, "build_and_verify", finalizing_gate)

    assert handoff_main([
        "--spec", str(spec_dir),
        "--slice", "s0",
        "--cwd", str(repo),
        "--land",
    ]) == 0
    json.loads(capsys.readouterr().out)
    latest = corpus.latest_spec()
    implementation = corpus.get_record(
        latest.implementation_refs[-1],
        as_type=Implementation,
    )
    evidence = _gate_evidence.load_verification_evidence(
        corpus,
        str(implementation.artifacts[_gate_evidence.HANDOFF_VERIFICATION_ARTIFACT]),
    )
    assert evidence.gate == "dev_verify"
    assert evidence.build is None
    assert evidence.verify is None
    assert evidence.dev_verify is not None
    assert evidence.satisfies_required_gate
    from crucible.workflows import _partition

    partition = _partition.partition_for_required_verification(
        corpus,
        latest.tier_dag.tiers[0],
        spec=latest,
    )
    assert tuple(partition.verified_retained) == ("s0",)
    assert not partition.verification_pending
    assert partition.implementation_pending == ()


@pytest.mark.parametrize("violation", ("out_of_scope", "missing_output"))
def test_spec_bound_handoff_finalization_failure_restores_exact_preimage(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
) -> None:
    from crucible.workflows import _build_verify, handoff
    from crucible.workflows._mutation_journal import discover_mutation_journals

    repo, corpus, spec_dir = _finalization_env(tmp_path)
    candidate = b"operator postimage\n"
    (repo / "seed.txt").write_bytes(candidate)
    before_outputs = tuple(
        (repo / path).read_bytes() for path in _FINALIZATION_OUTPUT_PATHS
    )


    def escaping_gate(*_args, **kwargs):
        (repo / _FINALIZATION_OUTPUT_PATHS[0]).write_bytes(b'{"phase":6}\n')
        (repo / _FINALIZATION_OUTPUT_PATHS[1]).write_bytes(b"phase6  manifest\n")
        if violation == "out_of_scope":
            (repo / "seed.txt").write_bytes(b"unauthorized gate mutation\n")
        else:
            (repo / _FINALIZATION_OUTPUT_PATHS[1]).unlink()
        _build_verify._settle_no_process_writer_epoch(
            kwargs["writer_epoch_members"],
            phase="test_handoff_authoring_escape",
            succeeded=True,
        )
        return _finalization_outcome()

    monkeypatch.setattr(handoff, "build_and_verify", escaping_gate)

    assert handoff_main([
        "--spec", str(spec_dir),
        "--slice", "s0",
        "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert "exact authorized output set" in summary["targets"][0]["reason"]
    assert (repo / "seed.txt").read_bytes() == candidate
    assert tuple(
        (repo / path).read_bytes() for path in _FINALIZATION_OUTPUT_PATHS
    ) == before_outputs
    assert corpus.latest_spec().implementation_refs == ()
    assert not discover_mutation_journals(corpus, workspace=repo)


def test_spec_bound_handoff_rejects_already_materialized_output_pair(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _build_verify, handoff
    from crucible.workflows._mutation_journal import discover_mutation_journals

    repo, corpus, spec_dir = _finalization_env(tmp_path)
    candidate = b"operator postimage\n"
    materialized = (b'{"phase":6}\n', b"phase6  manifest\n")
    (repo / "seed.txt").write_bytes(candidate)
    for path, content in zip(_FINALIZATION_OUTPUT_PATHS, materialized, strict=True):
        (repo / path).write_bytes(content)

    def non_materializing_gate(*_args, **kwargs):
        _build_verify._settle_no_process_writer_epoch(
            kwargs["writer_epoch_members"],
            phase="test_handoff_already_materialized",
            succeeded=True,
        )
        return _finalization_outcome()

    monkeypatch.setattr(handoff, "build_and_verify", non_materializing_gate)

    assert handoff_main([
        "--spec", str(spec_dir),
        "--slice", "s0",
        "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert "exact authorized output set" in summary["targets"][0]["reason"]
    assert (repo / "seed.txt").read_bytes() == candidate
    assert tuple(
        (repo / path).read_bytes() for path in _FINALIZATION_OUTPUT_PATHS
    ) == materialized
    assert corpus.latest_spec().implementation_refs == ()
    assert not discover_mutation_journals(corpus, workspace=repo)


def test_spec_bound_handoff_cas_failure_restores_finalized_outputs(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _build_verify, handoff
    from crucible.workflows._mutation_journal import discover_mutation_journals

    repo, corpus, spec_dir = _finalization_env(tmp_path)
    candidate = b"operator postimage\n"
    (repo / "seed.txt").write_bytes(candidate)
    before_outputs = tuple(
        (repo / path).read_bytes() for path in _FINALIZATION_OUTPUT_PATHS
    )

    def finalizing_gate(*_args, **kwargs):
        (repo / _FINALIZATION_OUTPUT_PATHS[0]).write_bytes(b'{"phase":6}\n')
        (repo / _FINALIZATION_OUTPUT_PATHS[1]).write_bytes(b"phase6  manifest\n")
        _build_verify._settle_no_process_writer_epoch(
            kwargs["writer_epoch_members"],
            phase="test_handoff_authoring_cas_failure",
            succeeded=True,
        )
        return _finalization_outcome()

    monkeypatch.setattr(handoff, "build_and_verify", finalizing_gate)
    monkeypatch.setattr(
        handoff,
        "_persist_implementation_guarded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            handoff._ConcurrentSpecChange("forced finalization CAS failure")
        ),
    )

    assert handoff_main([
        "--spec", str(spec_dir),
        "--slice", "s0",
        "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["targets"][0]["classification"] == (
        "hard_fail_concurrent_spec_change"
    )
    assert (repo / "seed.txt").read_bytes() == candidate
    assert tuple(
        (repo / path).read_bytes() for path in _FINALIZATION_OUTPUT_PATHS
    ) == before_outputs
    assert corpus.latest_spec().implementation_refs == ()
    assert not discover_mutation_journals(corpus, workspace=repo)


def test_spec_bound_handoff_dry_run_never_dispatches_finalization(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import handoff

    repo, corpus, spec_dir = _finalization_env(tmp_path)
    (repo / "seed.txt").write_text("operator postimage\n", encoding="utf-8")
    before_outputs = tuple(
        (repo / path).read_bytes() for path in _FINALIZATION_OUTPUT_PATHS
    )
    monkeypatch.setattr(
        handoff,
        "build_and_verify",
        lambda *_args, **_kwargs: pytest.fail("dry-run dispatched a gate"),
    )

    assert handoff_main([
        "--spec", str(spec_dir),
        "--slice", "s0",
        "--cwd", str(repo),
        "--dry-run",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["targets"][0]["authoring_finalization"]["status"] == (
        "would-run"
    )
    assert tuple(
        (repo / path).read_bytes() for path in _FINALIZATION_OUTPUT_PATHS
    ) == before_outputs
    assert corpus.latest_spec().implementation_refs == ()


def test_spec_bound_handoff_rejects_skip_verify(
    tmp_path: Path,
) -> None:
    repo, corpus, spec_dir = _finalization_env(tmp_path)
    (repo / "seed.txt").write_text("operator postimage\n", encoding="utf-8")

    assert handoff_main([
        "--spec", str(spec_dir),
        "--slice", "s0",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 2
    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    summary = json.loads((handoff_run / "summary.json").read_text())
    assert summary["error_kind"] == "invalid_authoring_finalization"
    assert "--skip-verify is incompatible" in summary["error"]
    assert corpus.latest_spec().implementation_refs == ()


@pytest.mark.parametrize("follow_on_gate", ("--with-breach", "--with-review"))
def test_spec_bound_handoff_rejects_follow_on_gates(
    tmp_path: Path,
    follow_on_gate: str,
) -> None:
    repo, corpus, spec_dir = _finalization_env(tmp_path)
    (repo / "seed.txt").write_text("operator postimage\n", encoding="utf-8")

    assert handoff_main([
        "--spec", str(spec_dir),
        "--slice", "s0",
        "--cwd", str(repo),
        follow_on_gate,
    ]) == 2
    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    summary = json.loads((handoff_run / "summary.json").read_text())
    assert summary["error_kind"] == "invalid_authoring_finalization"
    assert "follow-on gates are incompatible" in summary["error"]
    assert follow_on_gate in summary["error"]
    assert corpus.latest_spec().implementation_refs == ()


def _retained_handoff_journal(repo: Path, corpus: Corpus):
    from crucible.workflows import _mutation_journal

    legacy = repo / "internal/tools/run_previous_gate.sh"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("#!/bin/sh\necho previous\n", encoding="utf-8")
    with _mutation_journal.workspace_mutation_lock(repo):
        journal = _mutation_journal.MutationTransaction.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "retained-handoff" / "mutations"
            ),
            slice_id="retained-handoff-gates",
            scope=("seed.txt",),
            sequence=1,
            spec=corpus.latest_spec(),
        )
    _mutation_journal._unregister_live(journal.transaction_id)  # noqa: SLF001
    return journal


def test_dry_run_reports_retained_journal_without_recovering(env, capsys):
    repo, corpus, spec_dir = env
    journal = _retained_handoff_journal(repo, corpus)
    legacy = repo / "internal/tools/run_previous_gate.sh"
    replacement = repo / "internal/tools/run_current_gate.sh"
    legacy.unlink()
    replacement.write_text("#!/bin/sh\necho current\n", encoding="utf-8")
    (repo / "seed.txt").write_text("new operator work\n", encoding="utf-8")

    status_before = subprocess.run(
        ["git", "status", "--porcelain=v2", "-z", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    manifest_before = (journal.journal_dir / "manifest.json").read_bytes()

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0",
        "--cwd", str(repo), "--dry-run", "--skip-verify",
    ])

    assert rc == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "mutation_recovery_required"
    assert "no recovery was attempted" in summary["error"]
    assert summary["mutation_journals"] == [{
        "run_id": "retained-handoff",
        "journal_id": journal.journal_dir.name,
        "journal_manifest": str(journal.journal_dir / "manifest.json"),
        "transaction_id": journal.transaction_id,
        "slice_id": "retained-handoff-gates",
        "state": "invoking",
    }]
    assert (journal.journal_dir / "manifest.json").read_bytes() == manifest_before
    assert journal.journal_dir.is_dir()
    assert not legacy.exists()
    assert replacement.read_text(encoding="utf-8") == "#!/bin/sh\necho current\n"
    assert (repo / "seed.txt").read_text(encoding="utf-8") == "new operator work\n"
    status_after = subprocess.run(
        ["git", "status", "--porcelain=v2", "-z", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    assert status_after == status_before
    assert _latest_impls(corpus) == {}


def test_non_dry_run_keeps_startup_journal_recovery(env, capsys):
    repo, corpus, spec_dir = env
    journal = _retained_handoff_journal(repo, corpus)
    legacy = repo / "internal/tools/run_previous_gate.sh"
    replacement = repo / "internal/tools/run_current_gate.sh"
    legacy.unlink()
    replacement.write_text("#!/bin/sh\necho current\n", encoding="utf-8")
    (repo / "seed.txt").write_text("new operator work\n", encoding="utf-8")

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0",
        "--cwd", str(repo), "--skip-verify",
    ])

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "failed"
    assert not journal.journal_dir.exists()
    assert legacy.read_text(encoding="utf-8") == "#!/bin/sh\necho previous\n"
    assert not replacement.exists()
    assert (repo / "seed.txt").read_text(encoding="utf-8") == "seed\n"


def test_handoff_records_implementation_with_operator_attribution(env, capsys):
    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("seed\nedited by hand\n")
    (repo / "gen.py").write_text("print('new')\n")  # creates_paths target
    (repo / "unrelated.txt").write_text("ambient\n")  # out-of-scope

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0",
        "--cwd", str(repo), "--skip-verify",
        "--operator", "rfuoco", "--note", "hand-finished after abandon",
    ])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    (row,) = [r for r in summary["targets"] if r["slice"] == "s0"]
    assert row["action"] == "recorded"
    assert row["classification"] == "success"
    assert set(row["files_modified"]) == {"seed.txt", "gen.py"}
    # Out-of-scope edit reported, not captured.
    assert "unrelated.txt" in row["out_of_scope_changes"]

    impls = _latest_impls(corpus)
    impl = impls["s0"]
    assert impl.classification == "success"
    # ``execution`` rehydrates as a dict from the corpus deserializer
    # (same tolerance _partition.py applies).
    ex = impl.execution
    ex = ex if isinstance(ex, dict) else ex.__dict__
    assert ex["recipe"] == "handoff"
    assert ex["backend"] is None  # human/external attribution
    # Diff artifact retrievable + scoped (no unrelated.txt).
    diff = corpus.get_artifact(impl.diff_ref)
    assert b"edited by hand" in diff and b"unrelated.txt" not in diff
    # Transcript note carries the operator.
    note = json.loads(corpus.get_artifact(impl.transcript_ref))
    assert note["operator"] == "rfuoco"
    # Spec lineage advanced with the implementation ref.
    assert len(corpus.latest_spec().implementation_refs) == 1
    # The summary row carries the diff ref — the chaining handle for
    # `breach --diff-hash` / `review --artifact-hash`.
    assert row["diff_ref"] == str(impl.diff_ref)


def test_completed_slice_is_skipped_unless_redo(env, capsys):
    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("v1\n")
    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
                "--skip-verify",
            ]
        )
        == 0
    )
    capsys.readouterr()
    # Second handoff without --redo skips.
    rc = handoff_main(
        [
            "--spec",
            str(spec_dir),
            "--slice",
            "s0",
            "--cwd",
            str(repo),
            "--skip-verify",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    (row,) = summary["targets"]
    assert row["action"] == "skipped"
    assert "--redo" in row["reason"]


def test_required_verify_replays_waived_handoff_without_implementer(
    env,
    capsys,
    monkeypatch,
):
    from crucible.workflows import _partition, handoff as handoff_mod
    from crucible.workflows._gate_evidence import (
        HANDOFF_VERIFICATION_ARTIFACT,
        load_verification_evidence,
    )

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate awaiting verification\n")
    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
                "--skip-verify",
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert first["targets"][0]["verify"] == "skipped"
    waived_spec = corpus.latest_spec()
    waived = corpus.get_record(
        waived_spec.implementation_refs[-1],
        as_type=Implementation,
    )
    artifact_identified = replace(
        waived,
        execution=replace(waived.execution, recipe=None),
    )
    artifact_identified = _partition.attach_resume_identity(
        corpus,
        waived_spec,
        waived_spec.tier_dag.tiers[0],
        waived_spec.tier_dag.tiers[0].slices[0],
        artifact_identified,
    )
    assert _partition._is_handoff_candidate(artifact_identified)  # noqa: SLF001
    corpus.commit(implementations=(artifact_identified,))

    gate_calls = []

    def passing_gate(*args, **kwargs):
        gate_calls.append((args, kwargs))
        return _gate_outcome("success")

    monkeypatch.setattr(handoff_mod, "build_and_verify", passing_gate)
    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
            ]
        )
        == 0
    )
    replay = json.loads(capsys.readouterr().out)
    assert replay["targets"][0]["action"] == "recorded"
    assert len(gate_calls) == 1

    latest = corpus.latest_spec()
    implementation = corpus.get_record(
        latest.implementation_refs[-1],
        as_type=Implementation,
    )
    evidence = load_verification_evidence(
        corpus,
        str(implementation.artifacts[HANDOFF_VERIFICATION_ARTIFACT]),
    )
    assert evidence.satisfies_required_gate
    assert evidence.candidate_diff_sha256 == str(implementation.diff_ref)

    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
            ]
        )
        == 0
    )
    retained = json.loads(capsys.readouterr().out)
    assert retained["targets"][0]["action"] == "skipped"
    assert len(gate_calls) == 1


def test_newly_requested_review_is_not_bypassed_by_prior_handoff(
    env,
    capsys,
    monkeypatch,
):
    import crucible.workflows.handoff as handoff_mod

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate awaiting review\n")
    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
                "--skip-verify",
            ]
        )
        == 0
    )
    capsys.readouterr()

    review_calls = []

    def passing_review(argv):
        review_calls.append(argv)
        _publish_child_outcome(
            corpus,
            workflow="review",
            status="pass",
        )
        return 0

    monkeypatch.setattr(handoff_mod, "review_main", passing_review)
    requested = [
        "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
        "--skip-verify", "--with-review",
    ]
    assert handoff_main(requested) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["targets"][0]["action"] == "recorded"
    assert len(review_calls) == 1

    assert handoff_main(requested) == 0
    retained = json.loads(capsys.readouterr().out)
    assert retained["targets"][0]["action"] == "skipped"
    assert len(review_calls) == 1


def test_empty_diff_fails_unless_allow_empty(env, capsys):
    repo, corpus, spec_dir = env
    # No edits to s1's scope (other.txt doesn't even exist).
    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s1",
        "--cwd", str(repo), "--skip-verify",
    ])
    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    (row,) = summary["targets"]
    assert row["action"] == "failed" and "empty diff" in row["reason"]
    # With --allow-empty it records soft_fail_no_changes (skip-worthy).
    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s1",
        "--cwd", str(repo), "--skip-verify", "--allow-empty",
    ])
    assert rc == 0
    impls = _latest_impls(corpus)
    assert impls["s1"].classification == "soft_fail_no_changes"


def _fail_capture_when(monkeypatch, should_fail) -> None:
    """Make Handoff's working-tree capture fail for the selected scopes."""
    from crucible.workflows import handoff as handoff_mod

    def capture(*, cwd, path_scope):
        if should_fail(path_scope):
            raise GitCaptureError(
                "capture Git status failed: fatal: index file corrupt"
            )
        return capture_workspace_delta(cwd=cwd, path_scope=path_scope)

    monkeypatch.setattr(handoff_mod, "capture_workspace_delta", capture)


def _assert_durable_capture_setup_error(spec_dir: Path, printed: dict) -> None:
    """The retained run, not stdout, is the authority for the terminal."""
    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    summary = json.loads((handoff_run / "summary.json").read_text())
    result = json.loads((handoff_run / "result.json").read_text())
    outcome = json.loads((handoff_run / "outcome.json").read_text())
    assert summary == printed
    assert result["status"] == "setup_error"
    assert outcome["detail_status"] == "setup_error"
    assert outcome["error_kind"] == "workspace_capture_failed"


def test_unscoped_capture_failure_is_a_setup_error(env, capsys, monkeypatch):
    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("seed\nedited by hand\n")
    _fail_capture_when(monkeypatch, lambda path_scope: path_scope is None)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0",
        "--cwd", str(repo), "--skip-verify",
    ])

    assert rc == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "workspace_capture_failed"
    assert "index file corrupt" in summary["error"]
    assert summary["targets"] == []
    assert summary["landed"] is False
    assert _latest_impls(corpus) == {}
    assert corpus.latest_spec().implementation_refs == ()


def test_slice_capture_failure_stops_the_handoff_as_setup_error(
    env, capsys, monkeypatch,
):
    import crucible.workflows.handoff as handoff_mod

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("seed\nedited by hand\n")
    (repo / "other.txt").write_text("second slice\n")
    _fail_capture_when(
        monkeypatch,
        lambda path_scope: path_scope == ("other.txt",),
    )
    review_calls = []

    def forbidden_review(argv):
        review_calls.append(argv)
        pytest.fail("a capture setup error must stop the follow-on gates")

    monkeypatch.setattr(handoff_mod, "review_main", forbidden_review)

    rc = handoff_main([
        "--spec", str(spec_dir), "--tier", "t0",
        "--cwd", str(repo), "--skip-verify", "--with-review", "--land",
    ])

    assert rc == 2
    assert review_calls == []
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "workspace_capture_failed"
    assert "'s1'" in summary["error"]
    assert summary["landed"] is False
    _assert_durable_capture_setup_error(spec_dir, summary)
    # The earlier target keeps its own evidence; the typed terminal still
    # names it so the setup error is not read as "nothing happened".
    assert {row["slice"]: row["action"] for row in summary["targets"]} == {
        "s0": "recorded",
        "s1": "failed",
    }
    # Its requested gate never ran, so the record stays provisional.
    (recorded_row,) = [r for r in summary["targets"] if r["slice"] == "s0"]
    assert recorded_row["classification"] == "pending_gates"
    (failed_row,) = [r for r in summary["targets"] if r["slice"] == "s1"]
    assert "index file corrupt" in failed_row["reason"]
    assert failed_row["out_of_scope_changes"] == []
    recorded = _latest_impls(corpus)
    assert set(recorded) == {"s0"}
    assert recorded["s0"].classification == "pending_gates"
    final = corpus.latest_spec()
    assert len(final.implementation_refs) == 1
    assert final.lifecycle_event_refs == ()


def test_prior_candidate_capture_failure_is_a_setup_error(
    env, capsys, monkeypatch,
):
    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("v1\n")
    argv = [
        "--spec", str(spec_dir), "--slice", "s0",
        "--cwd", str(repo), "--skip-verify",
    ]
    assert handoff_main(argv) == 0
    capsys.readouterr()
    _fail_capture_when(
        monkeypatch,
        lambda path_scope: path_scope == ("seed.txt", "gen.py"),
    )

    rc = handoff_main(argv)

    assert rc == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "workspace_capture_failed"
    # An uncapturable live candidate cannot prove it still matches the
    # prior record, so the target is never reported as already complete.
    (row,) = summary["targets"]
    assert row["action"] == "failed"
    assert len(corpus.latest_spec().implementation_refs) == 1


def test_repository_filter_config_is_reported_as_capture_setup_error(
    env, capsys,
):
    repo, corpus, spec_dir = env
    # What a repository-local Git LFS install writes; the capture refuses
    # repository filter controls because they can rewrite captured content.
    subprocess.run(
        ["git", "config", "filter.lfs.clean", "git-lfs clean -- %f"],
        cwd=repo,
        check=True,
    )
    (repo / "seed.txt").write_text("seed\nedited by hand\n")

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0",
        "--cwd", str(repo), "--skip-verify",
    ])

    assert rc == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "workspace_capture_failed"
    assert "filter.lfs.clean" in summary["error"]
    _assert_durable_capture_setup_error(spec_dir, summary)
    assert _latest_impls(corpus) == {}
    assert corpus.latest_spec().implementation_refs == ()


def test_uncapturable_out_of_scope_entry_is_a_capture_setup_error(
    env, capsys,
):
    repo, corpus, spec_dir = env
    # An untracked nested repository without a commit lies outside the slice
    # scope, but the out-of-scope report captures the whole tree and Git
    # cannot stage that entry.
    nested = repo / "vendor"
    nested.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=nested, check=True)
    (repo / "seed.txt").write_text("seed\nedited by hand\n")

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0",
        "--cwd", str(repo), "--skip-verify",
    ])

    assert rc == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "workspace_capture_failed"
    assert "vendor" in summary["error"]
    assert summary["targets"] == []
    assert _latest_impls(corpus) == {}
    assert corpus.latest_spec().implementation_refs == ()


def test_land_is_blocked_when_the_live_candidate_cannot_be_recaptured(
    env, capsys, monkeypatch,
):
    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("seed\nedited by hand\n")
    scoped_captures = []

    def fails_after_the_record(path_scope) -> bool:
        if path_scope != ("seed.txt", "gen.py"):
            return False
        scoped_captures.append(path_scope)
        # First scoped capture records the slice; the next one is the
        # landing re-check of the live tree against that record.
        return len(scoped_captures) > 1

    _fail_capture_when(monkeypatch, fails_after_the_record)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0",
        "--cwd", str(repo), "--skip-verify", "--land",
    ])

    assert rc == 1
    assert len(scoped_captures) == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "failed"
    assert summary["landed"] is False
    assert summary["candidate_landing_blockers"] == ["s0"]
    (row,) = summary["targets"]
    assert row["action"] == "candidate-mismatch"
    # Nothing was compared, so the evidence names the capture failure and
    # never claims that the tree differs from the record.
    capture_error = "capture Git status failed: fatal: index file corrupt"
    assert row["reason"] == (
        "live candidate could not be re-captured before landing: "
        + capture_error
    )
    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    (land_event,) = [
        event
        for event in map(
            json.loads,
            (handoff_run / "events.jsonl").read_text().splitlines(),
        )
        if event.get("phase") == "land"
    ]
    assert land_event["slices"] == ["s0"]
    assert land_event["capture_errors"] == {"s0": capture_error}
    assert json.loads((handoff_run / "summary.json").read_text()) == summary
    assert corpus.latest_spec().lifecycle_event_refs == ()


def _nested_repository_env(tmp_path: Path) -> tuple[Path, Corpus, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    vendor = repo / "vendor"
    vendor.mkdir()
    _init_repo(vendor)
    corpus = Corpus(tmp_path / "corpus")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    corpus.put_spec(
        base.with_tier_dag(
            TierDAG(
                tiers=(
                    Tier(
                        tier_id="t0",
                        depends_on=(),
                        slices=(_slice("vendored", (), creates=("vendor",)),),
                    ),
                )
            )
        )
    )
    return repo, corpus, tmp_path / "corpus"


@pytest.mark.parametrize(
    "mode_args",
    [("--skip-verify",), (), ("--dry-run", "--skip-verify")],
    ids=["skip_verify", "default_gate", "dry_run"],
)
def test_nested_repository_in_scope_is_not_recorded(
    tmp_path, capsys, mode_args,
):
    """Git lists an untracked nested repository as one ``dir/`` entry.

    Its diff is a commit pointer, not content, so the target fails before
    any gate, preview row, or record treats it as captured work.
    """
    repo, corpus, spec_dir = _nested_repository_env(tmp_path)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "vendored",
        "--cwd", str(repo), *mode_args,
    ])

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "failed"
    (row,) = summary["targets"]
    assert row["action"] == "failed"
    assert "vendor/" in row["reason"]
    assert _latest_impls(corpus) == {}
    assert corpus.latest_spec().implementation_refs == ()


def _conditional_absence_env(tmp_path, *, selector_symlink: bool = False):
    from crucible.workflows import _partition

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    selector_payload = '{"selection":{"language":"rust"}}\n'
    if selector_symlink:
        external_selector = tmp_path / "external-decision.json"
        external_selector.write_text(selector_payload)
        (repo / "decision.json").symlink_to(external_selector)
    else:
        (repo / "decision.json").write_text(selector_payload)
    (repo / "native/runtime-rust").mkdir(parents=True)
    (repo / "native/runtime-rust/lib.rs").write_text("// selected\n")
    corpus = Corpus(tmp_path / "corpus")
    base = Spec.from_yaml(MINIMAL_YAML)
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="decision-tier",
                depends_on=(),
                slices=(_slice("decision", (), creates=("decision.json",)),),
            ),
            Tier(
                tier_id="runtime-tier",
                depends_on=("decision-tier",),
                slices=(
                    _slice("rust", (), creates=("native/runtime-rust",)),
                    _slice(
                        "cpp",
                        (),
                        creates=("native/runtime-cpp", "generated/runtime-cpp"),
                    ),
                ),
            ),
            Tier(
                tier_id="consumer-tier",
                depends_on=("runtime-tier",),
                slices=(_slice("consumer", ("consumer.txt",)),),
            ),
        )
    )
    corpus.put_spec(base)
    spec = base.with_tier_dag(dag)
    corpus.put_spec(spec)
    execution = ExecutionContext(
        timestamp="2026-08-07T00:00:00+00:00",
        machine="test",
        cwd=str(repo),
        toolchain={},
        backend=None,
        model=None,
        recipe="tier",
        recipe_commit=None,
        gpu_model=None,
        run_id="test",
    )
    transcript = corpus.put_artifact(b"[]", kind="transcripts")

    def accepted(slice_id: str):
        location = next(
            (tier, slice_)
            for tier in dag.tiers
            for slice_ in tier.slices
            if slice_.id == slice_id
        )
        scope = (
            *location[1].directive.files_in_scope,
            *location[1].directive.creates_paths,
        )
        content, files, base_commit = capture_workspace_delta(
            cwd=str(repo), path_scope=scope
        )
        implementation = Implementation(
            slice_id=slice_id, base_commit=base_commit, result_commit=None,
            diff_ref=corpus.put_artifact(content, kind="diffs"),
            transcript_ref=transcript, files_modified=files, scope_escapes=(),
            classification="success", execution=execution, artifacts={},
        )
        return _partition.attach_resume_identity(
            corpus, spec, location[0], location[1], implementation
        )

    corpus.commit(implementations=(
        accepted("decision"),
        accepted("rust"),
    ))
    return repo, corpus, tmp_path / "corpus"


def _conditional_absence_args(repo: Path, spec_dir: Path) -> list[str]:
    return [
        "--spec", str(spec_dir), "--slice", "cpp",
        "--cwd", str(repo), "--allow-empty", "--land",
        "--conditional-absence-selector", "decision.json",
        "--conditional-absence-json-pointer", "/selection/language",
        "--conditional-absence-equals", '"rust"',
        "--conditional-absence-decision-slice", "decision",
        "--conditional-absence-selected-slice", "rust",
    ]


def test_conditional_absence_is_required_pass_and_can_land(tmp_path, capsys):
    from crucible.workflows import _partition
    from crucible.workflows._gate_evidence import CONDITIONAL_ABSENCE_ARTIFACT

    repo, corpus, spec_dir = _conditional_absence_env(tmp_path)
    rc = handoff_main(_conditional_absence_args(repo, spec_dir))

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["landed"] is True
    implementation = _latest_impls(corpus)["cpp"]
    assert CONDITIONAL_ABSENCE_ARTIFACT in implementation.artifacts
    assert _partition._is_handoff_candidate(  # noqa: SLF001
        replace(implementation, execution={})
    )
    latest = corpus.latest_spec()
    assert latest is not None
    partition = _partition.partition_for_required_verification(
        corpus, latest.tier_dag.tiers[1], spec=latest
    )
    assert set(partition.verified_retained) == {"rust", "cpp"}


def test_conditional_absence_recaptures_authority_after_lock(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import handoff as handoff_workflow

    repo, corpus, spec_dir = _conditional_absence_env(tmp_path)
    original_lock = handoff_workflow.recovery_workspace_mutation_lock

    @contextmanager
    def mutate_selector_after_lock(corpus_arg, cwd):
        with original_lock(corpus_arg, cwd) as authority:
            (repo / "decision.json").write_text(
                '{"selection":{"language":"cpp"}}\n'
            )
            yield authority

    monkeypatch.setattr(
        handoff_workflow,
        "recovery_workspace_mutation_lock",
        mutate_selector_after_lock,
    )

    with pytest.raises(SystemExit) as exc:
        handoff_main(_conditional_absence_args(repo, spec_dir))

    assert exc.value.code == 2
    assert "cpp" not in _latest_impls(corpus)


def test_conditional_absence_refreshes_latest_corpus_authority_after_lock(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import _partition
    from crucible.workflows import handoff as handoff_workflow

    repo, corpus, spec_dir = _conditional_absence_env(tmp_path)
    original_lock = handoff_workflow.recovery_workspace_mutation_lock

    @contextmanager
    def append_same_bytes_authority_after_lock(corpus_arg, cwd):
        with original_lock(corpus_arg, cwd) as authority:
            current_spec = corpus.latest_spec()
            assert current_spec is not None
            decision = _partition._latest_impl_per_slice(  # noqa: SLF001
                corpus,
                {"decision"},
                refs=current_spec.implementation_refs,
            )["decision"]
            transcript = corpus.put_artifact(
                b'[{"event":"newer same-byte authority"}]',
                kind="transcripts",
            )
            corpus.commit(implementations=(replace(
                decision,
                transcript_ref=transcript,
            ),))
            yield authority

    monkeypatch.setattr(
        handoff_workflow,
        "recovery_workspace_mutation_lock",
        append_same_bytes_authority_after_lock,
    )

    assert handoff_main(_conditional_absence_args(repo, spec_dir)) == 0
    latest = corpus.latest_spec()
    assert latest is not None
    partition = _partition.partition_for_required_verification(
        corpus,
        latest.tier_dag.tiers[1],
        spec=latest,
    )
    assert set(partition.verified_retained) == {"rust", "cpp"}


def test_conditional_absence_rejects_selector_symlink(tmp_path):
    repo, corpus, spec_dir = _conditional_absence_env(
        tmp_path,
        selector_symlink=True,
    )

    with pytest.raises(SystemExit) as exc:
        handoff_main(_conditional_absence_args(repo, spec_dir))

    assert exc.value.code == 2
    assert "cpp" not in _latest_impls(corpus)


def test_bound_conditional_selector_rejects_ancestor_swap(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import handoff as handoff_workflow
    from crucible.workflows._mutation_journal import workspace_mutation_lock

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    selector_dir = repo / "configuration"
    selector_dir.mkdir()
    payload = b'{"selection":{"language":"rust"}}\n'
    (selector_dir / "decision.json").write_bytes(payload)
    original_read = handoff_workflow.os.read
    swapped = False

    def swap_ancestor_after_read(descriptor, size):
        nonlocal swapped
        result = original_read(descriptor, size)
        if not swapped and result:
            swapped = True
            selector_dir.rename(repo / "detached-configuration")
            selector_dir.mkdir()
            (selector_dir / "decision.json").write_bytes(payload)
        return result

    with workspace_mutation_lock(repo) as authority:
        monkeypatch.setattr(
            handoff_workflow.os,
            "read",
            swap_ancestor_after_read,
        )
        with pytest.raises(ValueError, match="ancestor changed"):
            handoff_workflow._read_bound_conditional_selector(  # noqa: SLF001
                authority.root_descriptor,
                "configuration/decision.json",
            )

    assert swapped is True


def test_conditional_absence_rejects_ignored_dangling_output_symlink(
    tmp_path,
    capsys,
):
    repo, _corpus, spec_dir = _conditional_absence_env(tmp_path)
    git_dir = Path(subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip())
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    (git_dir / "info" / "exclude").write_text("native/runtime-cpp\n")
    (repo / "native/runtime-cpp").symlink_to(
        "missing-runtime-cpp",
        target_is_directory=True,
    )

    rc = handoff_main(_conditional_absence_args(repo, spec_dir))

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["targets"][0]["reason"] == "conditional output path exists"


def test_conditional_absence_rejects_output_ancestor_symlink(
    tmp_path,
    capsys,
):
    repo, corpus, spec_dir = _conditional_absence_env(tmp_path)
    external = tmp_path / "external-output-root"
    external.mkdir()
    (repo / "generated").symlink_to(external, target_is_directory=True)

    rc = handoff_main(_conditional_absence_args(repo, spec_dir))

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["targets"][0]["reason"] == "conditional output path exists"
    assert "generated/runtime-cpp" in summary["targets"][0]["present_outputs"]
    assert "cpp" not in _latest_impls(corpus)


def test_conditional_absence_reproved_inside_implementation_commit(
    tmp_path,
    monkeypatch,
    capsys,
):
    repo, corpus, spec_dir = _conditional_absence_env(tmp_path)
    external = tmp_path / "external-output-root"
    external.mkdir()
    original_commit = Corpus.commit
    raced = False

    def race_conditional_implementation(self, **kwargs):
        nonlocal raced
        implementations = tuple(kwargs.get("implementations", ()))
        if not raced and any(item.slice_id == "cpp" for item in implementations):
            raced = True
            (repo / "generated").symlink_to(
                external,
                target_is_directory=True,
            )
        return original_commit(self, **kwargs)

    monkeypatch.setattr(Corpus, "commit", race_conditional_implementation)

    assert handoff_main(_conditional_absence_args(repo, spec_dir)) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["targets"][0]["classification"] == (
        "hard_fail_concurrent_spec_change"
    )
    latest = corpus.latest_spec()
    assert latest is not None
    assert all(
        corpus.get_record(ref, as_type=Implementation).slice_id != "cpp"
        for ref in latest.implementation_refs
    )
    assert raced is True


def test_conditional_absence_reproved_inside_lifecycle_commit(
    tmp_path,
    monkeypatch,
    capsys,
):
    repo, corpus, spec_dir = _conditional_absence_env(tmp_path)
    external = tmp_path / "external-output-root"
    external.mkdir()
    original_commit = Corpus.commit
    raced = False

    def race_conditional_lifecycle(self, **kwargs):
        nonlocal raced
        if not raced and kwargs.get("lifecycle_event_factory") is not None:
            raced = True
            (repo / "generated").symlink_to(
                external,
                target_is_directory=True,
            )
        return original_commit(self, **kwargs)

    monkeypatch.setattr(Corpus, "commit", race_conditional_lifecycle)

    assert handoff_main(_conditional_absence_args(repo, spec_dir)) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["landed"] is False
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.lifecycle_event_refs == ()
    assert raced is True


def test_tier_resume_rejects_present_conditional_output_before_backend(
    tmp_path,
    monkeypatch,
    capsys,
):
    from crucible.workflows import tier as tier_workflow

    repo, _corpus, spec_dir = _conditional_absence_env(tmp_path)
    assert handoff_main(_conditional_absence_args(repo, spec_dir)) == 0
    capsys.readouterr()
    (repo / "native/runtime-cpp").mkdir(parents=True)
    (repo / "native/runtime-cpp/ambient.cpp").write_text("// ambient\n")

    def reject_resolution(*args, **kwargs):
        pytest.fail("invalid conditional authority must not resolve a backend")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        reject_resolution,
    )

    rc = tier_workflow.run_tier(
        spec=spec_dir,
        tier_id="runtime-tier",
        cwd=str(repo),
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
        no_reap=True,
    )

    assert rc == 2
    run_dir = sorted((spec_dir / "runs").glob("*_tier_*"))[-1]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == tier_workflow.STATUS_SETUP_ERROR
    assert summary["error_kind"] == "conditional_absence_authority_invalid"
    assert summary["costs"]["total_usd"] == 0.0


def _run_conditional_consumer_tier(repo: Path, spec_dir: Path) -> int:
    from crucible.workflows import tier as tier_workflow

    return tier_workflow.run_tier(
        spec=spec_dir,
        tier_id="consumer-tier",
        cwd=str(repo),
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
        no_reap=True,
    )


def test_tier_rejects_stale_conditional_dependency_before_backend(
    tmp_path,
    monkeypatch,
    capsys,
):
    from crucible.workflows import _partition

    repo, corpus, spec_dir = _conditional_absence_env(tmp_path)
    assert handoff_main(_conditional_absence_args(repo, spec_dir)) == 0
    capsys.readouterr()
    current_spec = corpus.latest_spec()
    assert current_spec is not None
    decision = _partition._latest_impl_per_slice(  # noqa: SLF001
        corpus,
        {"decision"},
        refs=current_spec.implementation_refs,
    )["decision"]
    transcript = corpus.put_artifact(
        b'[{"event":"superseding decision authority"}]',
        kind="transcripts",
    )
    corpus.commit(implementations=(replace(
        decision,
        transcript_ref=transcript,
    ),))

    def reject_resolution(*args, **kwargs):
        pytest.fail("stale dependency authority must not resolve a backend")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        reject_resolution,
    )

    assert _run_conditional_consumer_tier(repo, spec_dir) == 2
    run_dir = sorted((spec_dir / "runs").glob("*_tier_*"))[-1]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "dependency_implementation_closure_invalid"
    assert summary["costs"]["total_usd"] == 0.0


def test_tier_rejects_present_conditional_dependency_output_before_backend(
    tmp_path,
    monkeypatch,
    capsys,
):
    repo, _corpus, spec_dir = _conditional_absence_env(tmp_path)
    assert handoff_main(_conditional_absence_args(repo, spec_dir)) == 0
    capsys.readouterr()
    (repo / "native/runtime-cpp").mkdir(parents=True)
    (repo / "native/runtime-cpp/ambient.cpp").write_text("// ambient\n")

    def reject_resolution(*args, **kwargs):
        pytest.fail("present conditional dependency must not resolve a backend")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        reject_resolution,
    )

    assert _run_conditional_consumer_tier(repo, spec_dir) == 2
    run_dir = sorted((spec_dir / "runs").glob("*_tier_*"))[-1]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "dependency_implementation_closure_invalid"
    assert summary["costs"]["total_usd"] == 0.0


def test_tier_rejects_conditional_dependency_ancestor_symlink_before_backend(
    tmp_path,
    monkeypatch,
    capsys,
):
    repo, _corpus, spec_dir = _conditional_absence_env(tmp_path)
    assert handoff_main(_conditional_absence_args(repo, spec_dir)) == 0
    capsys.readouterr()
    external = tmp_path / "external-output-root"
    external.mkdir()
    (repo / "generated").symlink_to(external, target_is_directory=True)

    def reject_resolution(*args, **kwargs):
        pytest.fail("aliased conditional dependency must not resolve a backend")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        reject_resolution,
    )

    assert _run_conditional_consumer_tier(repo, spec_dir) == 2
    run_dir = sorted((spec_dir / "runs").glob("*_tier_*"))[-1]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == "dependency_implementation_closure_invalid"
    assert summary["costs"]["total_usd"] == 0.0


def test_verification_only_slice_does_not_capture_ambient_dirt(
    tmp_path,
    capsys,
):
    repo, corpus, spec_dir = _verification_only_env(tmp_path)
    (repo / "seed.txt").write_text("ambient operator edit\n")

    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "verify-only",
                "--cwd",
                str(repo),
                "--skip-verify",
                "--allow-empty",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    (row,) = summary["targets"]
    assert row["files_modified"] == []
    assert row["diff_bytes"] == 0
    assert row["out_of_scope_changes"] == ["seed.txt"]
    impl = _latest_impls(corpus)["verify-only"]
    assert impl.files_modified == ()
    assert corpus.get_artifact(impl.diff_ref) == b""


@pytest.mark.parametrize(
    ("classification", "expected_rc", "expected_action"),
    [
        ("success", 0, "recorded"),
        ("verify_failed", 1, "failed"),
    ],
)
def test_verification_only_slice_runs_default_gate(
    tmp_path,
    capsys,
    monkeypatch,
    classification,
    expected_rc,
    expected_action,
):
    from crucible.workflows import handoff as handoff_mod

    repo, corpus, spec_dir = _verification_only_env(tmp_path)
    calls = []

    def fake_gate(*args, **kwargs):
        calls.append((args, kwargs))
        return _gate_outcome(classification)

    monkeypatch.setattr(handoff_mod, "build_and_verify", fake_gate)

    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "verify-only",
                "--cwd",
                str(repo),
                "--allow-empty",
            ]
        )
        == expected_rc
    )

    assert len(calls) == 1
    summary = json.loads(capsys.readouterr().out)
    (row,) = summary["targets"]
    assert row["action"] == expected_action
    if expected_rc == 0:
        assert row["verify"] == "success"
        assert _latest_impls(corpus)["verify-only"].classification == (
            "soft_fail_no_changes"
        )
    else:
        assert _latest_impls(corpus) == {}


def test_verification_only_slice_skips_breach_without_a_seed(
    tmp_path,
    capsys,
    monkeypatch,
):
    import crucible.workflows.handoff as handoff_mod

    repo, corpus, spec_dir = _verification_only_env(tmp_path)
    (repo / "seed.txt").write_text("ambient operator edit\n")

    def unexpected_breach(argv):
        pytest.fail(f"empty diff was dispatched to Breach: {argv!r}")

    monkeypatch.setattr(handoff_mod, "breach_main", unexpected_breach)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "verify-only",
        "--cwd", str(repo), "--skip-verify", "--allow-empty",
        "--with-breach",
    ])

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["gates"] == [{
        "slice": "verify-only",
        "gate": "breach",
        "rc": 0,
        "status": "skipped_empty_diff",
    }]
    assert summary["targets"][0]["classification"] == (
        "soft_fail_no_changes"
    )
    assert (repo / "seed.txt").read_text() == "ambient operator edit\n"
    attached = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert attached.classification == "soft_fail_no_changes"


def test_with_breach_invalid_binding_stops_before_capture_and_child(
    env,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handoff validates exact observations before recording operator work."""

    from crucible.workflows import handoff as handoff_mod

    repo, corpus, spec_dir = env
    spec = corpus.latest_spec()
    assert spec is not None
    context = replace(
        spec.project_context,
        runners={
            "build": "crucible.common.runners.polyglot",
            "verify": "crucible.common.runners.polyglot",
        },
        runner_parameters={
            "test_command": ["bash", "project-gate.sh", "test"],
        },
    )
    corpus.put_spec(spec._next_version(project_context=context))  # noqa: SLF001
    (repo / "seed.txt").write_text("unrecorded operator work\n")
    monkeypatch.setattr(
        handoff_mod,
        "capture_workspace_delta",
        lambda *_args, **_kwargs: pytest.fail(
            "capture ran before breach observation preflight"
        ),
    )
    monkeypatch.setattr(
        handoff_mod,
        "breach_main",
        lambda *_args, **_kwargs: pytest.fail(
            "Breach child ran before observation preflight"
        ),
    )

    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
                "--skip-verify",
                "--with-breach",
            ]
        )
        == 2
    )
    assert corpus.latest_spec().implementation_refs == ()
    assert "observation binding" in capsys.readouterr().err


def test_runner_missing_cannot_satisfy_required_handoff_landing(env, capsys):
    from crucible.workflows._gate_evidence import (
        HANDOFF_VERIFICATION_ARTIFACT,
        load_verification_evidence,
    )

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate without a configured runner\n")

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0",
        "--cwd", str(repo), "--land",
    ])

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["landed"] is False
    assert summary["verification_landing_blockers"] == ["s0"]
    assert summary["targets"][0]["action"] == "verification-pending"
    assert corpus.latest_spec().lifecycle_event_refs == ()
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    evidence = load_verification_evidence(
        corpus,
        str(implementation.artifacts[HANDOFF_VERIFICATION_ARTIFACT]),
    )
    assert evidence.disposition == "unavailable"
    assert not evidence.satisfies_required_gate


def test_land_recaptures_prior_handoff_when_live_candidate_changed(env, capsys):
    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate v1\n")
    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
                "--skip-verify",
            ]
        )
        == 0
    )
    capsys.readouterr()

    (repo / "seed.txt").write_text("candidate v2\n")
    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
                "--skip-verify",
                "--land",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)

    assert summary["targets"][0]["action"] == "recorded"
    assert summary["landed"] is True
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    diff = corpus.get_artifact(implementation.diff_ref)
    assert b"candidate v2" in diff
    assert b"candidate v1" not in diff


def test_no_surface_required_verification_needs_tier_closure_before_land(
    tmp_path,
    capsys,
    monkeypatch,
):
    import crucible.workflows.handoff as handoff_mod

    repo, corpus, spec_dir = _verification_only_env(tmp_path)
    monkeypatch.setattr(
        handoff_mod,
        "build_and_verify",
        lambda *args, **kwargs: _gate_outcome("success"),
    )

    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "verify-only",
                "--cwd",
                str(repo),
                "--allow-empty",
                "--land",
            ]
        )
        == 1
    )
    summary = json.loads(capsys.readouterr().out)

    assert summary["landed"] is False
    assert summary["verification_landing_blockers"] == ["verify-only"]
    assert corpus.latest_spec().lifecycle_event_refs == ()


def test_land_stamps_lifecycle_event(env, capsys):
    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("v2\n")
    rc = handoff_main([
        "--spec", str(spec_dir), "--tier", "t0",
        "--cwd", str(repo), "--skip-verify", "--allow-empty", "--land",
        "--operator", "rfuoco",
    ])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["landed"] is True
    final = corpus.latest_spec()
    assert len(final.lifecycle_event_refs) == 1
    from assay.types.artifacts import LifecycleEvent
    ev = corpus.get_record(
        final.lifecycle_event_refs[0], as_type=LifecycleEvent
    )
    assert ev.kind == "landed"
    assert "rfuoco" in ev.reason


def test_unknown_slice_exits_2(env, capsys):
    """Usage errors must exit 2 (documented contract) — review finding:
    SystemExit("msg") exits 1, so the workflow prints to stderr and
    raises SystemExit(2)."""
    repo, corpus, spec_dir = env
    with pytest.raises(SystemExit) as exc:
        handoff_main([
            "--spec", str(spec_dir), "--slice", "nope", "--cwd", str(repo),
        ])
    assert exc.value.code == 2
    assert "not found" in capsys.readouterr().err


def test_land_event_cites_the_persisted_version(env, capsys):
    """Review finding: with_lifecycle_event bumps the version and only
    the FINAL version is written — the event must cite that on-disk
    version, not the pre-bump one (which never persists)."""
    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("v3\n")
    assert (
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--tier",
                "t0",
                "--cwd",
                str(repo),
                "--skip-verify",
                "--allow-empty",
                "--land",
            ]
        )
        == 0
    )
    capsys.readouterr()
    from assay.types.artifacts import LifecycleEvent

    final = corpus.latest_spec()
    ev = corpus.get_record(final.lifecycle_event_refs[0], as_type=LifecycleEvent)
    assert tuple(ev.affected_versions) == (final.version,)
    # And that version is genuinely retrievable from disk.
    assert corpus.get_spec(final.version).version == final.version


def test_with_breach_and_review_chain_after_persist(env, capsys, monkeypatch):
    """--with-breach/--with-review invoke the gate workflows on each
    recorded diff_ref, AFTER the implementations are persisted (the
    gates read latest_spec() from disk), forwarding budget/backend."""
    import crucible.workflows.handoff as handoff_mod

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("gated edit\n")
    calls: list[tuple[str, list[str]]] = []

    def fake_breach(argv):
        # Ordering contract: our Implementation must already be on disk.
        latest = corpus.latest_spec()
        assert len(latest.implementation_refs) == 1
        initial = corpus.get_record(
            latest.implementation_refs[-1],
            as_type=Implementation,
        )
        from crucible.workflows import breach as breach_mod
        from crucible.workflows._mutation_journal import (
            validate_live_parent_transaction,
        )

        authority_path = Path(argv[argv.index("--authority-file") + 1])
        authority_digest = argv[argv.index("--authority-digest") + 1]
        diff_bytes = corpus.get_artifact(initial.diff_ref)
        authority = breach_mod._load_breach_authority(
            authority_file=authority_path,
            authority_digest_value=authority_digest,
            issue_text=None,
            diff_files=breach_mod._touched_files_from_diff(
                diff_bytes,
                cwd=str(repo),
            ),
        )
        assert authority is not None
        breach_mod._validate_authority_canonical_tier(authority, latest)
        validate_live_parent_transaction(
            argv[argv.index("--parent-mutation-transaction-id") + 1],
            cwd=repo,
        )
        (repo / "seed.txt").write_text("gated edit\naccepted defender remediation\n")
        (repo / "tests").mkdir(exist_ok=True)
        (repo / "tests" / "test_breach.py").write_text(
            "def test_retained_attack():\n    assert True\n"
        )
        _settle_fake_breach_writer_epoch(
            argv,
            phase="test_handoff_breach_child",
        )
        calls.append(("breach", argv))
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="proceed",
        )
        return 0

    def fake_review(argv):
        calls.append(("review", argv))
        _publish_child_outcome(
            corpus,
            workflow="review",
            status="pass",
        )
        return 0

    monkeypatch.setattr(handoff_mod, "breach_main", fake_breach)
    monkeypatch.setattr(handoff_mod, "review_main", fake_review)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
        "--skip-verify", "--with-breach", "--with-review",
        "--review-rubric", "tier_completeness", "--budget-usd", "25",
    ])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    impl = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    initial_impl = corpus.get_record(
        corpus.latest_spec().implementation_refs[0],
        as_type=Implementation,
    )

    (b,) = [a for (g, a) in calls if g == "breach"]
    assert b[b.index("--diff-hash") + 1] == str(initial_impl.diff_ref)
    assert b[b.index("--diff-base-treeish") + 1] == initial_impl.base_commit
    assert b[b.index("--budget-usd") + 1] == "25.0"
    authority_path = Path(b[b.index("--authority-file") + 1])
    authority_bytes = authority_path.read_bytes()
    assert b[b.index("--authority-digest") + 1] == (
        "sha256:" + hashlib.sha256(authority_bytes).hexdigest()
    )
    assert b[b.index("--parent-mutation-transaction-id") + 1]
    authority = json.loads(authority_bytes)
    assert authority["schema_version"] == "crucible.breach-authority.v3"
    assert authority["authority_kind"] == "selected_slices"
    assert [item["slice_id"] for item in authority["slices"]] == ["s0"]
    assert "s1" not in authority_bytes.decode()
    (r,) = [a for (g, a) in calls if g == "review"]
    assert "--artifact-hash" not in r
    review_artifact = Path(r[r.index("--artifact") + 1]).read_text()
    assert r[r.index("--rubric") + 1] == "tier_completeness"
    assert r[r.index("--slice-id") + 1] == "s0"
    assert r[r.index("--response-retry-budget") + 1] == "2"
    assert "crucible.tier-completeness-evidence.v1" in review_artifact
    assert "### slice `s0`" in review_artifact
    assert "### slice `s1`" not in review_artifact
    assert "do s0" in review_artifact
    assert "seed.txt" in review_artifact
    assert "gated edit" in review_artifact
    assert "accepted defender remediation" in review_artifact
    assert "tests/test_breach.py" not in review_artifact
    assert "test_retained_attack" not in review_artifact
    assert "WORKFLOW-RECORDED VERIFICATION EVIDENCE" in review_artifact
    assert '"source": "handoff"' in review_artifact
    assert "--workflow-generated-path" not in r
    child_gates = [
        gate for gate in summary["gates"]
        if gate["gate"] in {"breach", "review:tier_completeness"}
    ]
    assert [gate["rc"] for gate in child_gates] == [0, 0]
    assert [gate["terminal_class"] for gate in child_gates] == [
        "succeeded",
        "succeeded",
    ]
    assert [gate["decision"] for gate in child_gates] == ["proceed", "pass"]
    assert [gate["accounting_debt"] for gate in child_gates] == [[], []]
    assert [gate["cost_usd_is_lower_bound"] for gate in child_gates] == [False, False]
    assert summary["cost_usd_is_lower_bound"] is False
    assert summary["accounting_debt"] is False
    assert summary["accounting_debt_by_gate"] == {}
    assert b"accepted defender remediation" in corpus.get_artifact(impl.diff_ref)
    assert b"test_retained_attack" not in corpus.get_artifact(impl.diff_ref)
    assert not (repo / "tests" / "test_breach.py").exists()
    from crucible.workflows.review import _scope_context

    _, scope_context = _scope_context(
        corpus,
        slice_id="s0",
        artifact_ref=impl.diff_ref,
    )
    assert "tests/test_breach.py" not in scope_context["creates_paths"]


def test_selected_slice_handoff_reaches_real_breach(
    env,
    capsys,
    monkeypatch,
):
    from crucible.workflows import breach as breach_mod
    from .test_workflows_breach import (
        ScriptedBackend,
        _breaker_clean,
        _patch_backend,
        _patch_successful_shipping_gate,
    )

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("real gated edit\n")
    scripted = ScriptedBackend([_breaker_clean()])
    _patch_backend(monkeypatch, scripted)
    _patch_successful_shipping_gate(monkeypatch)
    monkeypatch.setattr(
        breach_mod,
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )

    rc = handoff_main([
        "--spec",
        str(spec_dir),
        "--slice",
        "s0",
        "--cwd",
        str(repo),
        "--skip-verify",
        "--with-breach",
    ])

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert len(scripted.calls) == 1
    breach_gate = next(
        gate for gate in summary["gates"] if gate["gate"] == "breach"
    )
    assert breach_gate["rc"] == 0
    assert breach_gate["terminal_class"] == "succeeded"
    assert breach_gate["decision"] == "no_attacks"
    assert breach_gate["run_dir"]
    assert summary["targets"][0]["classification"] == "success"
    assert not (repo / "tests" / "test_breach.py").exists()
    assert corpus.latest_spec().lifecycle_event_refs == ()


@pytest.mark.parametrize(
    ("child_status", "child_rc", "expected_handoff_rc", "terminal_class"),
    (
        ("no_attacks", 0, 0, "succeeded"),
        ("proceed", 0, 0, "succeeded"),
        ("inconclusive_authority", 3, 1, "inconclusive"),
        ("budget_exhausted", 1, 1, "budget_exhausted"),
        ("brand_new_unclassified", 1, 1, "contract_error"),
    ),
)
def test_typed_breach_outcome_controls_handoff_gate(
    env,
    capsys,
    monkeypatch,
    child_status,
    child_rc,
    expected_handoff_rc,
    terminal_class,
):
    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._partition import _latest_impl_per_slice

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text(f"candidate for {child_status}\n")

    def typed_breach(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status=child_status,
        )
        return child_rc

    monkeypatch.setattr(handoff_mod, "breach_main", typed_breach)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
        "--skip-verify", "--with-breach",
    ])

    assert rc == expected_handoff_rc
    summary = json.loads(capsys.readouterr().out)
    breach_gate = next(
        gate for gate in summary["gates"] if gate["gate"] == "breach"
    )
    assert breach_gate["rc"] == child_rc
    assert breach_gate["terminal_class"] == terminal_class
    assert breach_gate["decision"] == child_status
    assert breach_gate["error_kind"] is None
    assert breach_gate["run_dir"]
    terminal = _latest_impl_per_slice(corpus, {"s0"})["s0"]
    assert terminal.classification == (
        "success" if expected_handoff_rc == 0 else "hard_fail_gate"
    )


@pytest.mark.parametrize("child_outcome_mode", ("missing", "corrupt", "undercount"))
def test_missing_typed_breach_outcome_fails_closed(
    env,
    capsys,
    monkeypatch,
    child_outcome_mode,
):
    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._gate_evidence import (
        HANDOFF_GATE_LEDGER_ARTIFACT,
        load_handoff_gate_ledger,
    )
    from crucible.workflows._partition import _latest_impl_per_slice

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate with missing child outcome\n")

    def untrusted_breach(argv):
        del argv
        if child_outcome_mode in {"corrupt", "undercount"}:
            _publish_child_outcome(
                corpus,
                workflow="breach",
                status="proceed",
                cost_usd=7.25,
            )
            (parent_saga,) = (spec_dir / "runs").glob("*_handoff_*/saga.json")
            saga = json.loads(parent_saga.read_text(encoding="utf-8"))
            copied_outcome = saga["children"]["handoff_0_breach"]["outcome"]
            if child_outcome_mode == "corrupt":
                del copied_outcome["cost_usd"]
            else:
                copied_outcome["cost_usd"] = 0.0
            parent_saga.write_text(
                json.dumps(saga, sort_keys=True),
                encoding="utf-8",
            )
        return 0

    def forbidden_review(argv):
        del argv
        pytest.fail("unknown child accounting must stop review transport")

    monkeypatch.setattr(handoff_mod, "breach_main", untrusted_breach)
    monkeypatch.setattr(handoff_mod, "review_main", forbidden_review)

    rc = handoff_main(
        [
            "--spec",
            str(spec_dir),
            "--slice",
            "s0",
            "--cwd",
            str(repo),
            "--skip-verify",
            "--with-breach",
            "--with-review",
            "--land",
        ]
    )

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    breach_gate = next(gate for gate in summary["gates"] if gate["gate"] == "breach")
    assert breach_gate["rc"] == 0
    assert breach_gate["terminal_class"] == "contract_error"
    assert breach_gate["decision"] == "child_outcome_contract_error"
    assert breach_gate["error_kind"] == "child_outcome_contract_error"
    assert breach_gate["run_dir"] is None
    assert breach_gate["accounting_debt"] == [
        "token_usage_complete",
        "cost_complete",
    ]
    assert breach_gate["cost_usd_is_lower_bound"] is True
    review_gate = next(
        gate for gate in summary["gates"] if gate["gate"] == "review:diff_scope"
    )
    assert review_gate["rc"] is None
    assert review_gate["terminal_class"] == "skipped"
    assert review_gate["decision"] == "skipped_incomplete_accounting"
    assert review_gate["error_kind"] == "paid_gate_authority_revoked"
    assert summary["status"] == "incomplete_accounting"
    assert summary["cost_usd"] == 0.0
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt"] is True
    assert summary["accounting_debt_by_gate"] == {
        "s0:breach": ["cost_complete", "token_usage_complete"],
    }
    assert summary["landed"] is False
    terminal = _latest_impl_per_slice(corpus, {"s0"})["s0"]
    assert terminal.classification == "hard_fail_gate"
    ledger = load_handoff_gate_ledger(
        corpus,
        str(terminal.artifacts[HANDOFF_GATE_LEDGER_ARTIFACT]),
    )
    assert [entry.decision for entry in ledger.entries[-2:]] == [
        "child_outcome_contract_error",
        "skipped_incomplete_accounting",
    ]
    assert ledger.entries[-2].workflow_outcome is None


@pytest.mark.parametrize(
    ("child_status", "child_rc"),
    (("no_attacks", 1), ("reject", 0)),
)
def test_breach_exit_code_and_typed_outcome_mismatch_fails_closed(
    env,
    capsys,
    monkeypatch,
    child_status,
    child_rc,
):
    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._partition import _latest_impl_per_slice

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate with contradictory child result\n")

    def contradictory_breach(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status=child_status,
        )
        return child_rc

    monkeypatch.setattr(handoff_mod, "breach_main", contradictory_breach)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
        "--skip-verify", "--with-breach",
    ])

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    breach_gate = next(
        gate for gate in summary["gates"] if gate["gate"] == "breach"
    )
    assert breach_gate["decision"] == child_status
    assert breach_gate["rc"] == child_rc
    assert breach_gate["error_kind"] == "child_exit_outcome_mismatch"
    assert _latest_impl_per_slice(corpus, {"s0"})["s0"].classification == (
        "hard_fail_gate"
    )


def test_review_pass_is_preserved_when_breach_is_inconclusive(
    env,
    capsys,
    monkeypatch,
):
    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._gate_evidence import (
        HANDOFF_GATE_LEDGER_ARTIFACT,
        load_handoff_gate_ledger,
    )
    from crucible.workflows._partition import _latest_impl_per_slice

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("independently reviewable candidate\n")

    def inconclusive_breach(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="inconclusive_post_strip_verify",
        )
        return 3

    def passing_review(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="review",
            status="pass",
        )
        return 0

    monkeypatch.setattr(handoff_mod, "breach_main", inconclusive_breach)
    monkeypatch.setattr(handoff_mod, "review_main", passing_review)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
        "--skip-verify", "--with-breach", "--with-review", "--land",
    ])

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    breach_gate = next(
        gate for gate in summary["gates"] if gate["gate"] == "breach"
    )
    review_gate = next(
        gate for gate in summary["gates"] if gate["gate"] == "review:diff_scope"
    )
    assert breach_gate["terminal_class"] == "inconclusive"
    assert breach_gate["decision"] == "inconclusive_post_strip_verify"
    assert review_gate["terminal_class"] == "succeeded"
    assert review_gate["decision"] == "pass"
    assert summary["landed"] is False

    terminal = _latest_impl_per_slice(corpus, {"s0"})["s0"]
    assert terminal.classification == "hard_fail_gate"
    ledger_ref = terminal.artifacts[HANDOFF_GATE_LEDGER_ARTIFACT]
    ledger = load_handoff_gate_ledger(corpus, str(ledger_ref))
    typed_children = {
        entry.gate: entry.workflow_outcome for entry in ledger.entries
        if entry.workflow_outcome is not None
    }
    assert typed_children["breach"].terminal_class.value == "inconclusive"
    assert typed_children["breach"].detail_status == (
        "inconclusive_post_strip_verify"
    )
    assert typed_children["review:diff_scope"].terminal_class.value == (
        "succeeded"
    )
    assert typed_children["review:diff_scope"].detail_status == "pass"


def test_incomplete_accounting_breach_revokes_all_later_paid_gates(
    env,
    capsys,
    monkeypatch,
):
    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._gate_evidence import (
        HANDOFF_GATE_LEDGER_ARTIFACT,
        load_handoff_gate_ledger,
    )
    from crucible.workflows._partition import _latest_impl_per_slice

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("first candidate\n")
    (repo / "other.txt").write_text("second candidate\n")
    breach_calls = 0

    def incomplete_breach(argv):
        nonlocal breach_calls
        del argv
        breach_calls += 1
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="incomplete_accounting",
            cost_usd=0.375,
        )
        return 3

    def forbidden_review(argv):
        del argv
        pytest.fail("accounting debt must stop before review transport")

    monkeypatch.setattr(handoff_mod, "breach_main", incomplete_breach)
    monkeypatch.setattr(handoff_mod, "review_main", forbidden_review)

    rc = handoff_main(
        [
            "--spec",
            str(spec_dir),
            "--all",
            "--cwd",
            str(repo),
            "--skip-verify",
            "--with-breach",
            "--with-review",
            "--land",
        ]
    )

    assert rc == 1
    assert breach_calls == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete_accounting"
    assert summary["cost_usd"] == pytest.approx(0.375)
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt"] is True
    assert summary["accounting_debt_by_gate"] == {
        "s0:breach": ["cost_complete", "token_usage_complete"],
    }
    assert summary["landed"] is False
    assert [
        (gate["gate"], gate["terminal_class"], gate["decision"])
        for gate in summary["gates"]
    ] == [
        ("breach", "inconclusive", "incomplete_accounting"),
        (
            "review:diff_scope",
            "skipped",
            "skipped_incomplete_accounting",
        ),
    ]

    latest = _latest_impl_per_slice(corpus, {"s0", "s1"})
    assert latest["s0"].classification == "hard_fail_gate"
    assert latest["s1"].classification == "pending_gates"
    ledger = load_handoff_gate_ledger(
        corpus,
        str(latest["s0"].artifacts[HANDOFF_GATE_LEDGER_ARTIFACT]),
    )
    assert [(entry.gate, entry.decision) for entry in ledger.entries[-2:]] == [
        ("breach", "incomplete_accounting"),
        ("review:diff_scope", "skipped_incomplete_accounting"),
    ]
    assert ledger.entries[-2].workflow_outcome is not None
    assert ledger.entries[-2].workflow_outcome.cost_usd == pytest.approx(0.375)


def test_failed_breach_child_can_rebind_exact_parent_candidate_inode(
    env,
    capsys,
    monkeypatch,
):
    """A nested rollback may replace a plan-owned inode with exact bytes."""

    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._mutation_journal import (
        MutationTransaction,
        discover_mutation_journals,
    )

    repo, corpus, spec_dir = env
    candidate = "candidate restored by nested Breach\n"
    (repo / "seed.txt").write_text(candidate, encoding="utf-8")
    before = (repo / "seed.txt").stat()

    def incomplete_breach(argv):
        parent_id = argv[
            argv.index("--parent-mutation-transaction-id") + 1
        ]
        nested = MutationTransaction.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "nested-breach-mutations"
            ),
            slice_id="nested-breach",
            scope=("seed.txt",),
            sequence=991,
            spec=corpus.latest_spec(),
            parent_transaction_id=parent_id,
        )
        with (repo / "seed.txt").open("rb") as held_candidate:
            held = os.fstat(held_candidate.fileno())
            assert (held.st_dev, held.st_ino) == (
                before.st_dev,
                before.st_ino,
            )
            _settle_fake_breach_writer_epoch(
                argv,
                phase="test_handoff_nested_breach_full_rollback",
            )
            # A full nested rollback reconstructs every dirty baseline root
            # even when Breach did not change it.  Holding the old inode open
            # prevents immediate inode-number reuse and proves the production
            # A -> B -> C failure shape deterministically.
            assert nested.restore(full=True)
            after = (repo / "seed.txt").stat()
            assert (after.st_dev, after.st_ino) != (
                before.st_dev,
                before.st_ino,
            )
        _settle_fake_breach_writer_epoch(
            argv,
            phase="test_handoff_exact_parent_candidate_rebind",
        )
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="incomplete_accounting",
        )
        return 3

    monkeypatch.setattr(handoff_mod, "breach_main", incomplete_breach)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
        "--skip-verify", "--with-breach",
    ])

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete_accounting"
    assert (repo / "seed.txt").read_text(encoding="utf-8") == candidate
    assert discover_mutation_journals(corpus, workspace=repo) == ()


def test_incomplete_accounting_review_revokes_later_slices_and_retains_cost(
    env,
    capsys,
    monkeypatch,
):
    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._gate_evidence import (
        HANDOFF_GATE_LEDGER_ARTIFACT,
        load_handoff_gate_ledger,
    )
    from crucible.workflows._partition import _latest_impl_per_slice

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("first review candidate\n")
    (repo / "other.txt").write_text("second review candidate\n")
    review_calls = 0

    def incomplete_review(argv):
        nonlocal review_calls
        del argv
        review_calls += 1
        if review_calls > 1:
            pytest.fail("accounting debt must stop later review transports")
        _publish_child_outcome(
            corpus,
            workflow="review",
            status="incomplete_accounting",
            cost_usd=1.25,
        )
        return 3

    monkeypatch.setattr(handoff_mod, "review_main", incomplete_review)

    rc = handoff_main(
        [
            "--spec",
            str(spec_dir),
            "--all",
            "--cwd",
            str(repo),
            "--skip-verify",
            "--with-review",
            "--land",
        ]
    )

    assert rc == 1
    assert review_calls == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete_accounting"
    assert summary["cost_usd"] == pytest.approx(1.25)
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt"] is True
    assert summary["accounting_debt_by_gate"] == {
        "s0:review:diff_scope": [
            "cost_complete",
            "token_usage_complete",
        ],
    }
    assert summary["landed"] is False
    assert corpus.latest_spec().lifecycle_event_refs == ()
    assert [
        (gate["slice"], gate["gate"], gate["terminal_class"], gate["decision"])
        for gate in summary["gates"]
    ] == [
        (
            "s0",
            "review:diff_scope",
            "inconclusive",
            "incomplete_accounting",
        ),
    ]

    latest = _latest_impl_per_slice(corpus, {"s0", "s1"})
    assert latest["s0"].classification == "hard_fail_gate"
    assert latest["s1"].classification == "pending_gates"
    ledger = load_handoff_gate_ledger(
        corpus,
        str(latest["s0"].artifacts[HANDOFF_GATE_LEDGER_ARTIFACT]),
    )
    review_entry = ledger.entries[-1]
    assert (review_entry.gate, review_entry.decision) == (
        "review:diff_scope",
        "incomplete_accounting",
    )
    assert review_entry.workflow_outcome is not None
    assert review_entry.workflow_outcome.detail_status == "incomplete_accounting"
    assert review_entry.workflow_outcome.cost_usd == pytest.approx(1.25)

    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    run_result = json.loads((handoff_run / "result.json").read_text())
    parent_outcome = json.loads((handoff_run / "outcome.json").read_text())
    assert run_result["status"] == "incomplete_accounting"
    assert run_result["cost_usd"] == pytest.approx(1.25)
    assert parent_outcome["detail_status"] == "incomplete_accounting"
    assert parent_outcome["cost_usd"] == pytest.approx(1.25)


def test_cleanup_exception_after_paid_children_retains_aggregate_cost(
    env,
    monkeypatch,
):
    """A parent cleanup failure must not erase terminal child costs."""

    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._mutation_journal import (
        MutationTransaction,
        discover_mutation_journals,
    )

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate with paid gates\n")

    def paid_breach(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="proceed",
            cost_usd=5.251292,
        )
        return 0

    def paid_review(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="review",
            status="pass",
            cost_usd=2.726243,
        )
        return 0

    monkeypatch.setattr(handoff_mod, "breach_main", paid_breach)
    monkeypatch.setattr(handoff_mod, "review_main", paid_review)
    original_settle_or_restore = MutationTransaction.settle_or_restore

    def fail_gate_settlement(self, target_corpus, *, full=None):
        if self.slice_id == "handoff_s0_gates":
            return False
        return original_settle_or_restore(
            self,
            target_corpus,
            full=full,
        )

    def post_child_failure(*_args, **_kwargs):
        raise RuntimeError("post-child integrity probe failed")

    with monkeypatch.context() as unwind:
        unwind.setattr(
            handoff_mod,
            "_candidate_integrity_error",
            post_child_failure,
        )
        unwind.setattr(
            MutationTransaction,
            "settle_or_restore",
            fail_gate_settlement,
        )
        with pytest.raises(
            RuntimeError,
            match="failed to restore Handoff gate transaction",
        ):
            handoff_main([
                "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
                "--skip-verify", "--with-breach", "--with-review",
            ])

    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    summary = json.loads((handoff_run / "summary.json").read_text())
    result = json.loads((handoff_run / "result.json").read_text())
    outcome = json.loads((handoff_run / "outcome.json").read_text())
    saga = json.loads((handoff_run / "saga.json").read_text())
    children = saga["children"]
    assert children["handoff_0_breach"]["state"] == "terminal"
    assert children["handoff_0_review"]["state"] == "terminal"
    assert children["handoff_0_breach"]["outcome"]["cost_usd"] == (
        pytest.approx(5.251292)
    )
    assert children["handoff_0_review"]["outcome"]["cost_usd"] == (
        pytest.approx(2.726243)
    )
    for terminal in (summary, result, outcome):
        assert terminal["cost_usd"] == pytest.approx(7.977535)
    assert summary["status"] == "failed"
    assert "accounting_debt" not in summary
    assert "cost_usd_is_lower_bound" not in summary
    assert result["status"] == "failed"
    assert result["error_kind"] == "RuntimeError"
    assert result["error_message"] == ("failed to restore Handoff gate transaction")
    assert outcome["terminal_class"] == "permanent_failure"

    retained = discover_mutation_journals(corpus, workspace=repo)
    assert len(retained) == 1
    assert retained[0].restore()


def test_incomplete_child_cleanup_exception_preserves_accounting_debt(
    env,
    monkeypatch,
):
    """Rollback failure cannot flatten a child's incomplete accounting."""

    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._mutation_journal import (
        MutationTransaction,
        discover_mutation_journals,
    )

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate with accounting debt\n")

    def incomplete_breach(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="incomplete_accounting",
            cost_usd=0.625,
        )
        return 3

    monkeypatch.setattr(handoff_mod, "breach_main", incomplete_breach)
    original_restore = MutationTransaction.restore

    def fail_gate_restore(self, *args, **kwargs):
        if self.slice_id == "handoff_s0_gates":
            return False
        return original_restore(self, *args, **kwargs)

    with monkeypatch.context() as unwind:
        unwind.setattr(MutationTransaction, "restore", fail_gate_restore)
        with pytest.raises(
            RuntimeError,
            match="failed to restore Handoff gate transaction",
        ):
            handoff_main(
                [
                    "--spec",
                    str(spec_dir),
                    "--slice",
                    "s0",
                    "--cwd",
                    str(repo),
                    "--skip-verify",
                    "--with-breach",
                ]
            )

    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    summary = json.loads((handoff_run / "summary.json").read_text())
    result = json.loads((handoff_run / "result.json").read_text())
    outcome = json.loads((handoff_run / "outcome.json").read_text())
    assert summary["status"] == "incomplete_accounting"
    assert summary["cost_usd"] == pytest.approx(0.625)
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt"] is True
    assert summary["accounting_debt_by_gate"] == {
        "s0:breach": ["cost_complete", "token_usage_complete"],
    }
    assert summary["error_kind"] == "RuntimeError"
    assert summary["error_message"] == ("failed to restore Handoff gate transaction")
    assert result["status"] == "incomplete_accounting"
    assert result["cost_usd"] == pytest.approx(0.625)
    assert result["error_kind"] == "RuntimeError"
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["cost_usd"] == pytest.approx(0.625)

    retained = discover_mutation_journals(corpus, workspace=repo)
    assert len(retained) == 1
    assert retained[0].restore()


@pytest.mark.parametrize("escaped_kind", ("cleanup_owner", "keyboard_interrupt"))
def test_escaped_child_terminal_preserves_accounting_debt(
    env,
    monkeypatch,
    escaped_kind,
):
    """A child terminal is harvested before preserving escaped control flow."""

    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate with escaped child debt\n")
    escaped: BaseException | None = None

    def incomplete_then_escape(argv):
        nonlocal escaped
        del argv
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="proceed",
            cost_usd=0.625,
        )
        if escaped_kind == "cleanup_owner":
            cleanup_error = RuntimeError("provider descendants remain live")
            escaped = ProviderSettlementCleanupError(
                ProviderSettlementExitStack(),
                primary_error=cleanup_error,
                cleanup_error=cleanup_error,
            )
        else:
            escaped = KeyboardInterrupt("operator cancelled after child terminal")
        raise escaped

    monkeypatch.setattr(handoff_mod, "breach_main", incomplete_then_escape)

    with pytest.raises(BaseException) as captured:
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
                "--skip-verify",
                "--with-breach",
            ]
        )

    assert escaped is not None
    assert captured.value is escaped
    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    summary = json.loads((handoff_run / "summary.json").read_text())
    result = json.loads((handoff_run / "result.json").read_text())
    outcome = json.loads((handoff_run / "outcome.json").read_text())
    saga = json.loads((handoff_run / "saga.json").read_text())
    assert saga["children"]["handoff_0_breach"]["outcome"]["cost_usd"] == (
        pytest.approx(0.625)
    )
    assert summary["status"] == "incomplete_accounting"
    assert summary["cost_usd"] == pytest.approx(0.625)
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt_by_gate"] == {
        "s0:breach": ["cost_complete", "token_usage_complete"],
    }
    assert result["status"] == "incomplete_accounting"
    assert result["cost_usd"] == pytest.approx(0.625)
    assert result["error_kind"] == type(escaped).__name__
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["cost_usd"] == pytest.approx(0.625)
    if isinstance(escaped, ProviderSettlementCleanupError):
        escaped.retry_cleanup()


@pytest.mark.parametrize("outer_kind", ("cleanup_owner", "mutation_terminal"))
def test_child_provider_owner_retains_parent_cleanup_in_order(
    env,
    monkeypatch,
    outer_kind,
):
    """Child quiescence precedes parent rollback and workspace release."""

    import crucible.workflows.handoff as handoff_mod
    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows._mutation_journal import MutationTransaction
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate with retained provider cleanup\n")
    events: list[str] = []
    original_lock = handoff_mod.recovery_workspace_mutation_lock
    original_settle = MutationTransaction.settle_or_restore
    fail_parent_restore = True
    cleanup_owner: ProviderSettlementCleanupError | None = None
    outer_terminal: BaseException | None = None

    class RetryProviderClose(RuntimeError):
        def retry_cleanup(self) -> None:
            events.append("child-provider-quiesced")

    class PendingProvider:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            raise RetryProviderClose("provider descendants remain live")

    @contextmanager
    def tracking_lock(*args, **kwargs):
        with original_lock(*args, **kwargs) as authority:
            yield authority
        events.append("workspace-released")

    def tracked_settle(transaction, *args, **kwargs):
        nonlocal fail_parent_restore
        if transaction.slice_id.endswith("_gates"):
            events.append("parent-gate-rollback")
            if fail_parent_restore:
                return False
        return original_settle(transaction, *args, **kwargs)

    def incomplete_then_escape(argv):
        nonlocal cleanup_owner, outer_terminal
        del argv
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="proceed",
            cost_usd=0.625,
        )
        provider_owner = ProviderSettlementExitStack()
        provider_owner.__enter__()
        provider_owner.callback(events.append, "child-callback")
        provider_owner.enter_read_only_provider_context(PendingProvider())
        try:
            provider_owner.__exit__(None, None, None)
        except ProviderSettlementCleanupError as exc:
            cleanup_owner = exc
        assert cleanup_owner is not None
        if outer_kind == "mutation_terminal":
            outer_terminal = ToolMutationIndeterminate(
                "child mutation settlement is unknown"
            )
            raise outer_terminal from cleanup_owner
        outer_terminal = cleanup_owner
        raise cleanup_owner

    monkeypatch.setattr(
        handoff_mod,
        "recovery_workspace_mutation_lock",
        tracking_lock,
    )
    monkeypatch.setattr(
        MutationTransaction,
        "settle_or_restore",
        tracked_settle,
    )
    monkeypatch.setattr(handoff_mod, "breach_main", incomplete_then_escape)

    try:
        with pytest.raises(BaseException) as captured:
            handoff_main([
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
                "--skip-verify",
                "--with-breach",
            ])

        assert outer_terminal is not None
        assert captured.value is outer_terminal
        assert cleanup_owner is not None
        assert events == []

        with pytest.raises(ProviderSettlementCleanupError) as retry_failed:
            cleanup_owner.retry_cleanup()
        assert retry_failed.value is cleanup_owner
        assert events == [
            "child-provider-quiesced",
            "child-callback",
            "parent-gate-rollback",
        ]

        fail_parent_restore = False
        cleanup_owner.retry_cleanup()
        assert events == [
            "child-provider-quiesced",
            "child-callback",
            "parent-gate-rollback",
            "parent-gate-rollback",
            "workspace-released",
        ]
        assert cleanup_owner.owner.cleanup_complete
    finally:
        fail_parent_restore = False
        if cleanup_owner is not None and not cleanup_owner.owner.cleanup_complete:
            cleanup_owner.retry_cleanup()


@pytest.mark.parametrize("escaped_kind", ("cleanup_owner", "keyboard_interrupt"))
def test_child_operation_exit_failure_preserves_accounting_debt(
    env,
    monkeypatch,
    escaped_kind,
):
    """A paid child is harvested before its operation scope error escapes."""

    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate with scope-exit debt\n")
    original_child_operation = handoff_mod._outcome.child_operation
    cleanup_error = RuntimeError("child operation cleanup failed")
    if escaped_kind == "cleanup_owner":
        escaped: BaseException = ProviderSettlementCleanupError(
            ProviderSettlementExitStack(),
            primary_error=cleanup_error,
            cleanup_error=cleanup_error,
        )
    else:
        escaped = KeyboardInterrupt("child operation cleanup interrupted")

    @contextmanager
    def failing_child_operation(*args, **kwargs):
        with original_child_operation(*args, **kwargs):
            yield
        raise escaped

    def successful_breach(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="proceed",
            cost_usd=0.625,
        )
        return 0

    monkeypatch.setattr(
        handoff_mod._outcome,
        "child_operation",
        failing_child_operation,
    )
    monkeypatch.setattr(handoff_mod, "breach_main", successful_breach)

    with pytest.raises(BaseException) as captured:
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
                "--skip-verify",
                "--with-breach",
            ]
        )

    assert captured.value is escaped
    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    summary = json.loads((handoff_run / "summary.json").read_text())
    result = json.loads((handoff_run / "result.json").read_text())
    assert summary["status"] == "incomplete_accounting"
    assert summary["cost_usd"] == pytest.approx(0.625)
    assert summary["accounting_debt_by_gate"] == {
        "s0:breach": ["cost_complete", "token_usage_complete"],
    }
    assert result["status"] == "incomplete_accounting"
    assert result["error_kind"] == type(escaped).__name__
    if isinstance(escaped, ProviderSettlementCleanupError):
        escaped.retry_cleanup()


def test_corrupt_child_run_binding_is_unknown_accounting(
    env,
    monkeypatch,
):
    """An unbound saga copy cannot provide even a trustworthy lower bound."""

    import crucible.workflows.handoff as handoff_mod

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate with corrupt child binding\n")

    def corrupt_breach_binding(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="proceed",
            cost_usd=7.25,
        )
        (parent_saga,) = (spec_dir / "runs").glob("*_handoff_*/saga.json")
        loop = parent_saga.parent / "child-run-loop"
        loop.symlink_to(loop.name)
        saga = json.loads(parent_saga.read_text(encoding="utf-8"))
        saga["children"]["handoff_0_breach"]["run_dir"] = str(loop)
        parent_saga.write_text(
            json.dumps(saga, sort_keys=True),
            encoding="utf-8",
        )
        return 0

    def forbidden_review(argv):
        del argv
        pytest.fail("corrupt child accounting must stop review transport")

    monkeypatch.setattr(handoff_mod, "breach_main", corrupt_breach_binding)
    monkeypatch.setattr(handoff_mod, "review_main", forbidden_review)

    rc = handoff_main(
        [
            "--spec",
            str(spec_dir),
            "--slice",
            "s0",
            "--cwd",
            str(repo),
            "--skip-verify",
            "--with-breach",
            "--with-review",
        ]
    )

    assert rc == 1
    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    summary = json.loads((handoff_run / "summary.json").read_text())
    result = json.loads((handoff_run / "result.json").read_text())
    outcome = json.loads((handoff_run / "outcome.json").read_text())
    assert summary["status"] == "incomplete_accounting"
    assert summary["cost_usd"] == 0.0
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt_by_gate"] == {
        "s0:breach": ["cost_complete", "token_usage_complete"],
    }
    breach_gate = next(gate for gate in summary["gates"] if gate["gate"] == "breach")
    assert breach_gate["decision"] == "child_outcome_contract_error"
    assert breach_gate["terminal_class"] == "contract_error"
    assert breach_gate["error_kind"] == "child_outcome_contract_error"
    assert result["status"] == "incomplete_accounting"
    assert result["cost_usd"] == 0.0
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["cost_usd"] == 0.0


@pytest.mark.parametrize("child_outcome_mode", ("missing", "corrupt"))
def test_unknown_child_persistence_exception_preserves_accounting_debt(
    env,
    monkeypatch,
    child_outcome_mode,
):
    """Unknown usage survives a later terminal-record persistence failure."""

    import crucible.workflows.handoff as handoff_mod

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("candidate with unknown child usage\n")

    def untrusted_breach(argv):
        del argv
        if child_outcome_mode == "corrupt":
            _publish_child_outcome(
                corpus,
                workflow="breach",
                status="proceed",
                cost_usd=9.75,
            )
            (parent_saga,) = (spec_dir / "runs").glob("*_handoff_*/saga.json")
            saga = json.loads(parent_saga.read_text(encoding="utf-8"))
            del saga["children"]["handoff_0_breach"]["outcome"]["cost_usd"]
            parent_saga.write_text(
                json.dumps(saga, sort_keys=True),
                encoding="utf-8",
            )
        return 0

    def forbidden_review(argv):
        del argv
        pytest.fail("unknown child accounting must stop review transport")

    original_persist = handoff_mod._persist_implementation_guarded
    persist_calls = 0

    def fail_terminal_persist(*args, **kwargs):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            raise RuntimeError("terminal persistence failed")
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(handoff_mod, "breach_main", untrusted_breach)
    monkeypatch.setattr(handoff_mod, "review_main", forbidden_review)
    monkeypatch.setattr(
        handoff_mod,
        "_persist_implementation_guarded",
        fail_terminal_persist,
    )

    with pytest.raises(RuntimeError, match="terminal persistence failed"):
        handoff_main(
            [
                "--spec",
                str(spec_dir),
                "--slice",
                "s0",
                "--cwd",
                str(repo),
                "--skip-verify",
                "--with-breach",
                "--with-review",
            ]
        )

    (handoff_run,) = (spec_dir / "runs").glob("*_handoff_*")
    summary = json.loads((handoff_run / "summary.json").read_text())
    result = json.loads((handoff_run / "result.json").read_text())
    outcome = json.loads((handoff_run / "outcome.json").read_text())
    assert summary["status"] == "incomplete_accounting"
    assert summary["cost_usd"] == 0.0
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt"] is True
    assert summary["accounting_debt_by_gate"] == {
        "s0:breach": ["cost_complete", "token_usage_complete"],
    }
    assert summary["error_kind"] == "RuntimeError"
    assert summary["error_message"] == "terminal persistence failed"
    assert [gate["decision"] for gate in summary["gates"][-2:]] == [
        "child_outcome_contract_error",
        "skipped_incomplete_accounting",
    ]
    assert result["status"] == "incomplete_accounting"
    assert result["cost_usd"] == 0.0
    assert result["error_kind"] == "RuntimeError"
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["cost_usd"] == 0.0


def test_gate_failure_marks_failed_and_suppresses_land(env, capsys, monkeypatch):
    import crucible.workflows.handoff as handoff_mod

    repo, corpus, spec_dir = env
    (repo / "seed.txt").write_text("rejected edit\n")

    def rejected_breach(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="reject",
        )
        return 2

    monkeypatch.setattr(handoff_mod, "breach_main", rejected_breach)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
        "--skip-verify", "--with-breach", "--land",
    ])
    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "failed"
    assert summary["landed"] is False
    assert corpus.latest_spec().lifecycle_event_refs == ()
    # The initial capture remains as provenance, but a later terminal record
    # makes the rejected handoff non-skip-worthy for resume.
    from crucible.workflows._partition import _latest_impl_per_slice

    terminal = _latest_impl_per_slice(corpus, {"s0"})["s0"]
    assert terminal.classification == "hard_fail_gate"
    assert len(corpus.latest_spec().implementation_refs) == 2


def test_concurrent_tier_dag_change_rejects_terminal_and_land(
    env,
    capsys,
    monkeypatch,
):
    import crucible.workflows.handoff as handoff_mod

    repo, corpus, spec_dir = env
    operator_bytes = "operator candidate\n"
    (repo / "seed.txt").write_text(operator_bytes)
    replacement_dag = TierDAG(tiers=(
        Tier(
            tier_id="replacement",
            depends_on=(),
            slices=(_slice("replacement", ("replacement.txt",)),),
        ),
    ))

    def review_that_changes_execution_authority(argv):
        del argv
        corpus.commit(
            transform=lambda candidate: replace(
                candidate,
                tier_dag=replacement_dag,
            )
        )
        _publish_child_outcome(
            corpus,
            workflow="review",
            status="pass",
        )
        return 0

    monkeypatch.setattr(
        handoff_mod,
        "review_main",
        review_that_changes_execution_authority,
    )

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
        "--skip-verify", "--with-review", "--land",
    ])

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "failed"
    assert summary["landed"] is False
    assert summary["concurrent_structure_error"] == (
        "Spec execution authority changed during handoff"
    )
    assert summary["targets"][0]["classification"] == (
        "hard_fail_concurrent_spec_change"
    )
    assert summary["gates"][-1] == {
        "slice": "s0",
        "gate": "execution_structure_cas",
        "rc": 1,
        "error": "Spec execution authority changed during handoff",
    }
    assert (repo / "seed.txt").read_text() == operator_bytes

    final = corpus.latest_spec()
    assert final.tier_dag == replacement_dag
    assert final.lifecycle_event_refs == ()
    assert len(final.implementation_refs) == 1
    attached = corpus.get_record(
        final.implementation_refs[0],
        as_type=Implementation,
    )
    assert attached.slice_id == "s0"
    assert attached.classification == "pending_gates"


def test_review_failure_restores_accepted_defender_bytes(
    env,
    capsys,
    monkeypatch,
):
    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._partition import _latest_impl_per_slice

    repo, corpus, spec_dir = env
    operator_bytes = "operator candidate\n"
    (repo / "seed.txt").write_text(operator_bytes)

    def fake_breach(argv):
        (repo / "seed.txt").write_text(
            operator_bytes + "unlanded defender bytes\n"
        )
        _settle_fake_breach_writer_epoch(
            argv,
            phase="test_handoff_rejected_breach_child",
        )
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="proceed",
        )
        return 0

    def fake_review(argv):
        del argv
        _publish_child_outcome(
            corpus,
            workflow="review",
            status="revise",
        )
        return 1

    monkeypatch.setattr(handoff_mod, "breach_main", fake_breach)
    monkeypatch.setattr(handoff_mod, "review_main", fake_review)

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
        "--skip-verify", "--with-breach", "--with-review",
    ])

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert (repo / "seed.txt").read_text() == operator_bytes
    terminal = _latest_impl_per_slice(corpus, {"s0"})["s0"]
    assert terminal.classification == "hard_fail_gate"
    assert b"unlanded defender bytes" not in corpus.get_artifact(
        terminal.diff_ref
    )
    assert summary["targets"][0]["diff_ref"] == str(terminal.diff_ref)


def test_mutating_post_breach_verifier_restores_operator_bytes(
    env,
    capsys,
    monkeypatch,
):
    import crucible.workflows.handoff as handoff_mod
    from crucible.workflows._gate_evidence import (
        HANDOFF_GATE_LEDGER_ARTIFACT,
        load_handoff_gate_ledger,
        load_verification_evidence,
    )
    from crucible.workflows._partition import _latest_impl_per_slice

    repo, corpus, spec_dir = env
    operator_bytes = "operator candidate\n"
    (repo / "seed.txt").write_text(operator_bytes)
    verify_calls = 0

    def fake_breach(argv):
        (repo / "seed.txt").write_text(operator_bytes + "defender bytes\n")
        _settle_fake_breach_writer_epoch(
            argv,
            phase="test_handoff_post_verify_breach_child",
        )
        _publish_child_outcome(
            corpus,
            workflow="breach",
            status="proceed",
        )
        return 0

    def fake_verify(*args, **kwargs):
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            (repo / "seed.txt").write_text(
                operator_bytes + "defender bytes\nverifier mutation\n"
            )
        return _gate_outcome("success")

    monkeypatch.setattr(handoff_mod, "breach_main", fake_breach)
    monkeypatch.setattr(
        handoff_mod,
        "build_and_verify",
        _contract_aware_gate_stub(fake_verify),
    )

    rc = handoff_main([
        "--spec", str(spec_dir), "--slice", "s0", "--cwd", str(repo),
        "--with-breach",
    ])

    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert verify_calls == 2
    assert (repo / "seed.txt").read_text() == operator_bytes
    post_verify = next(
        gate for gate in summary["gates"]
        if gate["gate"] == "post_breach_verify"
    )
    assert post_verify["rc"] == 1
    assert "mutated source" in post_verify["integrity_error"]
    terminal = _latest_impl_per_slice(corpus, {"s0"})["s0"]
    assert terminal.classification == "hard_fail_gate"
    assert b"defender bytes" not in corpus.get_artifact(terminal.diff_ref)
    ledger = load_handoff_gate_ledger(
        corpus,
        str(terminal.artifacts[HANDOFF_GATE_LEDGER_ARTIFACT]),
    )
    post_breach_entry = next(
        entry for entry in ledger.entries
        if entry.gate == "post_breach_verify"
    )
    evidence = load_verification_evidence(
        corpus,
        post_breach_entry.verification_evidence_ref,
    )
    assert evidence.classification == "hard_fail_gate_mutation"
    assert evidence.disposition == "failed"
    assert not evidence.satisfies_required_gate
