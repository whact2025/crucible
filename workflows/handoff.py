# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``handoff`` — hand-complete a slice / tier / spec from operator work.

Capture operator-completed work as scoped, verified Corpus records. The work
may come from manual editing or an external agent. Recording it gives status
and ``tier --resume`` the same completion evidence as a workflow-owned
implementation.

What it does, per target slice (in tier topo order):

1. **Capture** the working-tree delta scoped to the slice's
   ``files_in_scope ∪ creates_paths`` via assay's single-sourced
   ``capture_workspace_delta`` (same plumbing as ``assay.implement`` —
   new-file visibility, index cleanup, path normalization).
2. **Verify** (default on, ``--skip-verify`` to bypass): the spec's
   build/verify runners via ``workflows._build_verify.build_and_verify``
   with the slice's directive — the same gate the tier loop applies.
3. **Record** an ``Implementation`` with ``classification="success"``
   (``soft_fail_no_changes`` under ``--allow-empty`` for deliberately
   diffless targets), or ``pending_gates`` until optional breach/review gates
   finish. Gate rejection appends a superseding ``hard_fail_gate`` attempt.
   Records carry ``execution.recipe="handoff"`` and
   ``backend=None`` so attribution is unambiguous (a human / external
   agent, not a crucible backend), plus a transcript artifact carrying
   the operator note.
4. **Stamp** ``LifecycleEvent(kind="landed")`` with ``--land`` once
   every targeted slice records as complete.

Out-of-scope edits in the tree are REPORTED (summary + warning) and never
captured — the operator consciously owns the tree. Their content is never
fatal. The report itself captures the whole tree, so an entry Git cannot
capture (for example an untracked nested repository without a commit) ends
the run as a setup error even when it lies outside every targeted scope.

The interactive loop lives one level up (the ``crucible-handoff``
skill / Claude Code): edit → ``handoff --dry-run`` to preview exactly
what would be captured → ``handoff`` to record. ``breach`` / ``review``
remain available as follow-on gates on the recorded diff.

Exit codes: 0 = all targets recorded; 1 = some target failed
(empty diff without ``--allow-empty``, a directory entry in the scoped
capture, verification or optional-gate failure) or landing was blocked
(for example a recorded candidate Git could not re-capture); 2 = usage or
setup error (unknown slice/tier, no tier_dag, a working-tree delta Git could
not capture before recording).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, replace
from functools import partial
import json
import logging
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, TypeVar

from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus
from assay.canonical import hash_record
from assay.primitives.implement import GitCaptureError, capture_workspace_delta
from assay.types.artifacts import Implementation, LifecycleEvent
from assay.types.execution import ExecutionContext
from assay.types.identity import ContentHash

from ._args import add_spec_arg, positive_budget_arg
from ._build_verify import (
    DEFAULT_BUILD_TIMEOUT_S,
    DEFAULT_VERIFY_TIMEOUT_S,
    BuildVerifyOutcome,
    _settle_no_process_writer_epoch,
    build_and_verify,
)
from ._gate_evidence import (
    CONDITIONAL_ABSENCE_ARTIFACT,
    GateLedgerEntry,
    HANDOFF_AUTHORIZING_FINALIZATION_ARTIFACT,
    HANDOFF_GATE_LEDGER_ARTIFACT,
    HANDOFF_VERIFICATION_ARTIFACT,
    bound_workspace_path_absent,
    handoff_gate_ledger_matches,
    persist_conditional_absence_evidence,
    persist_handoff_authoring_finalization_evidence,
    persist_handoff_gate_ledger,
    persist_verification_evidence,
    validate_conditional_selector,
)
from ._mutation_journal import (
    MutationTransaction,
    RetainedRecoveryWorkspaceMutationLock,
    discover_mutation_journals,
    live_writer_epoch_family,
    recover_mutation_journals,
    recovery_workspace_mutation_lock,
)
from ._provider_settlement import ProviderSettlementCleanupError
from . import _outcome, _partition
from ._run import FailureTerminal, open_run_with_capture, record_handled_terminal
from .breach import (
    breach_authority_digest,
    encode_breach_authority,
    main as breach_main,
    preflight_breach_observation_binding,
    resolve_breach_observation_language,
    resolve_breach_test_target,
)
from .review import main as review_main


# Handoff's final review is an autonomous gate, so it receives the same
# bounded semantic-response repair allowance as Tier. Under Assay's chunk
# protocol this authorizes retries only for the currently rejected map; it
# never replays an accepted prefix or a reducer.
_HANDOFF_REVIEW_RESPONSE_RETRY_BUDGET = 2

logger = logging.getLogger(__name__)

_CONDITIONAL_SELECTOR_MAX_BYTES = 1_048_576

_CleanupT = TypeVar("_CleanupT")


class _RetryableCleanupOwner(Protocol):
    """Minimal retry contract shared by provider and composed owners."""

    @property
    def cleanup_complete(self) -> bool: ...

    def retry_cleanup(self) -> None: ...


class _OneShotHandoffContextExit:
    """Never report a consumed generic context exit as a later success."""

    def __init__(self, context: AbstractContextManager[Any]) -> None:
        self._context = context
        self._attempted = False
        self._failure: BaseException | None = None

    def __call__(self) -> None:
        if self._attempted:
            if self._failure is not None:
                raise self._failure.with_traceback(self._failure.__traceback__)
            return
        self._attempted = True
        try:
            self._context.__exit__(None, None, None)
        except BaseException as exc:  # noqa: BLE001 - exact failed release
            self._failure = exc
            raise


class _RetainedCleanupStack:
    """LIFO cleanup whose failing callback remains the next retry target."""

    def __init__(self, *, retain_failures_on_exit: bool = False) -> None:
        self._callbacks: list[
            tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]
        ] = []
        self._retain_failures_on_exit = retain_failures_on_exit
        self._provider_delegation: (
            _DeferHandoffCleanupToProviderOwner | None
        ) = None

    @property
    def cleanup_complete(self) -> bool:
        return not self._callbacks

    def __enter__(self) -> "_RetainedCleanupStack":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: object | None,
    ) -> bool:
        delegation = self._provider_delegation
        if delegation is not None and delegation.defers_lexical_cleanup:
            # Delegation is fail-closed from construction through provider
            # owner publication.  Lexical unwind cannot outrun a child whose
            # settlement authority is still being recognized or transferred.
            retained = delegation.retain_pending_transfer_error(_exc)
            if retained is not None:
                raise retained from _exc
            return False
        if self._retain_failures_on_exit:
            self.retry_cleanup()
        else:
            pending_error: BaseException | None = None
            while self._callbacks:
                callback, args, kwargs = self._callbacks.pop()
                try:
                    callback(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - finish unwind
                    pending_error = exc
            if pending_error is not None:
                raise pending_error
        return False

    def callback(
        self,
        callback: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        self._callbacks.append((callback, args, kwargs))
        return callback

    def enter_context(
        self,
        context: AbstractContextManager[_CleanupT],
    ) -> _CleanupT:
        if isinstance(context, RetainedRecoveryWorkspaceMutationLock):
            # Recovery workspace locks publish their acquisition lease inside
            # the held-lock source scope immediately before yield.  Retain that
            # lease owner before entry: it is inert if callback publication is
            # interrupted before ``__enter__`` starts, and live before an
            # acquired authority can cross back into this frame.
            self.callback(context.retry_cleanup)
            return context.__enter__()

        # Generic context managers provide no source-owned acquisition proof.
        # Preserve standard ``with`` semantics by registering their exit only
        # after a successful entry; critical workspace authority uses the
        # retained lease path above.
        entered = context.__enter__()
        exit_callback = _OneShotHandoffContextExit(context)
        self.callback(exit_callback)
        return entered

    def bind_provider_delegation(
        self,
        delegation: "_DeferHandoffCleanupToProviderOwner",
    ) -> None:
        """Bind the pre-dispatch shared delegation state exactly once."""

        if (
            self._provider_delegation is not None
            and self._provider_delegation is not delegation
        ):
            raise RuntimeError("cleanup stack already has a provider delegation")
        self._provider_delegation = delegation

    def pop_all(self) -> "_RetainedCleanupStack":
        transferred = type(self)(retain_failures_on_exit=True)
        transferred._callbacks = self._callbacks
        self._callbacks = []
        return transferred

    def close(self) -> None:
        delegation = self._provider_delegation
        if delegation is not None and delegation.defers_lexical_cleanup:
            # Acquisition guards share the lexical stack's close surface and
            # must observe the same fail-closed provider transfer state.
            return
        self.retry_cleanup()

    def retry_cleanup(self) -> None:
        while self._callbacks:
            callback, args, kwargs = self._callbacks[-1]
            callback(*args, **kwargs)
            self._callbacks.pop()


class _HandoffProviderCleanupOwner:
    """Extend one child owner through parent rollback and lock release."""

    def __init__(
        self,
        child: _RetryableCleanupOwner | None,
        rollback: _RetainedCleanupStack,
        workspace_release: _RetainedCleanupStack,
    ) -> None:
        self._child = child
        self._rollback = rollback
        self._workspace_release = workspace_release
        self._source_bound = False
        self._source_error: BaseException | None = None
        self._provider_error: ProviderSettlementCleanupError | None = None
        self._provider_absent = False
        self._published = False
        self._deferred_interrupt: KeyboardInterrupt | None = None

    def bind_source_error(self, error: BaseException | None) -> None:
        """Retain the escaping chain before any interruptible recognition."""

        if self._source_bound and self._source_error is not error:
            raise RuntimeError("handoff cleanup owner received a new source error")
        self._source_error = error
        self._source_bound = True

    def bind_provider_error(
        self,
        error: ProviderSettlementCleanupError,
    ) -> None:
        """Bind exactly one child cleanup error before reading its owner."""

        if self._provider_error is not None and self._provider_error is not error:
            raise RuntimeError("handoff cleanup owner received a new provider error")
        self._provider_error = error

    @property
    def has_source_error(self) -> bool:
        return self._source_bound

    def publish_provider_owner(self) -> None:
        """Idempotently splice this precomposed owner into the child error."""

        provider_error = self._provider_error
        if provider_error is None:
            provider_error = _provider_cleanup_error_in_cause_chain(
                self._source_error
            )
            if (
                provider_error is None
                and isinstance(self._source_error, KeyboardInterrupt)
            ):
                # A main-thread interrupt raised by this transfer's own frame
                # carries the displaced body error as its direct context.
                # Classify only that explicit displaced chain; general context
                # traversal could revive an incidental/stale provider owner.
                provider_error = _provider_cleanup_error_in_cause_chain(
                    self._source_error.__context__
                )
                if provider_error is None:
                    self._provider_absent = True
                    return
            if provider_error is None and self._source_bound:
                self._provider_absent = True
                return
            if provider_error is None:
                raise RuntimeError(
                    "handoff cleanup owner cannot resolve child provider authority"
                )
            self.bind_provider_error(provider_error)
        if self._child is None:
            child = provider_error.owner
            if child is self:
                raise RuntimeError(
                    "handoff provider owner was published without its child"
                )
            self._child = child
        if not self._published:
            provider_error.owner = self  # type: ignore[assignment]
            self._published = True

    @property
    def provider_owner_published(self) -> bool:
        return self._published

    def defer_interrupt(self, interrupt: KeyboardInterrupt) -> None:
        """Replay transfer-time cancellation only after complete settlement."""

        self._deferred_interrupt = self._deferred_interrupt or interrupt

    @property
    def cleanup_complete(self) -> bool:
        child = self._child
        if child is None and not self._provider_absent:
            return False
        return bool(
            (child is None or child.cleanup_complete)
            and self._rollback.cleanup_complete
            and self._workspace_release.cleanup_complete
        )

    def retry_cleanup(self) -> None:
        while self._child is None and not self._provider_absent:
            try:
                self.publish_provider_owner()
            except KeyboardInterrupt as exc:
                self.defer_interrupt(exc)
        child = self._child
        if child is not None and not child.cleanup_complete:
            try:
                child.retry_cleanup()
            except KeyboardInterrupt as exc:
                if not child.cleanup_complete:
                    raise
                self._deferred_interrupt = self._deferred_interrupt or exc
        if not self._rollback.cleanup_complete:
            self._rollback.retry_cleanup()
        if not self._workspace_release.cleanup_complete:
            self._workspace_release.retry_cleanup()
        interrupt = self._deferred_interrupt
        self._deferred_interrupt = None
        if interrupt is not None:
            raise interrupt


def _provider_cleanup_error_in_cause_chain(
    error: BaseException | None,
) -> ProviderSettlementCleanupError | None:
    """Find an explicit provider retry owner without following context."""

    current = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderSettlementCleanupError):
            return current
        cause = current.__cause__
        current = cause if isinstance(cause, BaseException) else None
    return None


class _DeferHandoffCleanupToProviderOwner:
    """Move parent cleanup behind a child provider's live retry owner."""

    def __init__(
        self,
        rollback: _RetainedCleanupStack,
        workspace_release: _RetainedCleanupStack,
    ) -> None:
        self._cleanup_owner = _HandoffProviderCleanupOwner(
            None,
            rollback,
            workspace_release,
        )
        pending_error = RuntimeError(
            "handoff provider cleanup owner transfer remains pending"
        )
        self._pending_transfer_error = ProviderSettlementCleanupError(
            self._cleanup_owner,  # type: ignore[arg-type]
            primary_error=pending_error,
            cleanup_error=pending_error,
        )
        self._lexical_cleanup_enabled = False
        # Both lexical stacks consult this pre-dispatch fail-closed state. The
        # complete parent owner exists before a child can run; provider
        # recognition only fills its child slot and publishes that same owner.
        rollback.bind_provider_delegation(self)
        workspace_release.bind_provider_delegation(self)

    @property
    def cleanup_owner(self) -> _HandoffProviderCleanupOwner | None:
        if self._lexical_cleanup_enabled:
            return None
        return self._cleanup_owner

    @property
    def defers_lexical_cleanup(self) -> bool:
        return not self._lexical_cleanup_enabled

    def retain_pending_transfer_error(
        self,
        error: BaseException | None,
    ) -> ProviderSettlementCleanupError | None:
        """Expose the precomposed owner if transfer control escapes early."""

        if self._cleanup_owner.provider_owner_published:
            return None
        retained = self._pending_transfer_error
        if error is retained:
            return None
        if error is not None:
            if not self._cleanup_owner.has_source_error:
                self._cleanup_owner.bind_source_error(error)
            self._pending_transfer_error.primary_error = error
            self._pending_transfer_error.cleanup_error = error
            if isinstance(error, KeyboardInterrupt):
                self._cleanup_owner.defer_interrupt(error)
        return retained

    def __enter__(self) -> "_DeferHandoffCleanupToProviderOwner":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: object | None,
    ) -> bool:
        deferred_interrupt: KeyboardInterrupt | None = None
        cleanup_error: ProviderSettlementCleanupError | None = None
        self._cleanup_owner.bind_source_error(error)
        while True:
            try:
                cleanup_error = _provider_cleanup_error_in_cause_chain(error)
                if cleanup_error is None:
                    self._lexical_cleanup_enabled = True
                    break
                self._cleanup_owner.bind_provider_error(cleanup_error)
                self._cleanup_owner.publish_provider_owner()
                if deferred_interrupt is not None:
                    self._cleanup_owner.defer_interrupt(deferred_interrupt)
                return False
            except KeyboardInterrupt as interrupt:
                deferred_interrupt = deferred_interrupt or interrupt
            except BaseException as transfer_error:  # noqa: BLE001
                # Parent cleanup remains fail-closed, but the escaping error
                # exposes the already-published composite retry owner. A later
                # retry can repeat recognition/publication before settling the
                # child, rollback, and workspace in order.
                primary_error = cleanup_error or error or transfer_error
                retained = self._pending_transfer_error
                retained.primary_error = primary_error
                retained.cleanup_error = transfer_error
                raise retained from primary_error
        if deferred_interrupt is not None:
            raise deferred_interrupt
        return False


def _mutation_journal_inventory(
    corpus: Corpus,
    cwd: str | Path,
) -> list[dict[str, str]]:
    """Describe retained journals without applying startup recovery."""

    runs_root = Path(corpus.spec_dir).resolve() / "runs"
    rows: list[dict[str, str]] = []
    for journal in discover_mutation_journals(corpus, workspace=cwd):
        relative = journal.journal_dir.relative_to(runs_root)
        rows.append({
            "run_id": relative.parts[0],
            "journal_id": journal.journal_dir.name,
            "journal_manifest": str(journal.journal_dir / "manifest.json"),
            "transaction_id": journal.transaction_id,
            "slice_id": journal.slice_id,
            "state": journal.state,
        })
    return sorted(
        rows,
        key=lambda row: (
            row["run_id"],
            row["journal_id"],
            row["transaction_id"],
        ),
    )


def _read_bound_conditional_selector(
    root_descriptor: int,
    relative_path: str,
) -> bytes:
    """Read one stable in-workspace regular selector without symlink traversal."""

    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("conditional selector path is not repository-relative")
    descriptor = os.dup(root_descriptor)
    leaf_descriptor: int | None = None
    ancestor_identities: list[tuple[str, tuple[int, int, int, int]]] = []
    try:
        for component in path.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_DIRECTORY", 0),
                dir_fd=descriptor,
            )
            child_stat = os.fstat(child)
            ancestor_identities.append((
                component,
                (
                    child_stat.st_dev,
                    child_stat.st_ino,
                    child_stat.st_mode,
                    child_stat.st_nlink,
                ),
            ))
            os.close(descriptor)
            descriptor = child
        leaf = path.parts[-1]
        before = os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _CONDITIONAL_SELECTOR_MAX_BYTES
        ):
            raise ValueError(
                "conditional selector must be one bounded regular file"
            )
        leaf_descriptor = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=descriptor,
        )
        opened = os.fstat(leaf_descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError("conditional selector changed while opening")
        payload = bytearray()
        while True:
            chunk = os.read(
                leaf_descriptor,
                min(
                    1024 * 1024,
                    _CONDITIONAL_SELECTOR_MAX_BYTES + 1 - len(payload),
                ),
            )
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > _CONDITIONAL_SELECTOR_MAX_BYTES:
                raise ValueError("conditional selector exceeds its byte ceiling")
        finished = os.fstat(leaf_descriptor)
        stable_fields = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
            "st_mtime_ns", "st_ctime_ns",
        )
        if any(
            getattr(opened, field) != getattr(finished, field)
            for field in stable_fields
        ):
            raise ValueError("conditional selector changed during read")

        # A parent directory can be renamed after it is opened, leaving its
        # descriptor valid but detached from the workspace pathname. Re-walk
        # the complete chain from the held root and require every ancestor and
        # the leaf to resolve to the same kernel objects used for the read.
        validation_descriptor = os.dup(root_descriptor)
        try:
            for component, expected_identity in ancestor_identities:
                child = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=validation_descriptor,
                )
                child_stat = os.fstat(child)
                observed_identity = (
                    child_stat.st_dev,
                    child_stat.st_ino,
                    child_stat.st_mode,
                    child_stat.st_nlink,
                )
                if observed_identity != expected_identity:
                    os.close(child)
                    raise ValueError(
                        "conditional selector ancestor changed during read"
                    )
                os.close(validation_descriptor)
                validation_descriptor = child
            current = os.stat(
                leaf,
                dir_fd=validation_descriptor,
                follow_symlinks=False,
            )
            if any(
                getattr(opened, field) != getattr(current, field)
                for field in stable_fields
            ):
                raise ValueError("conditional selector changed during read")
        finally:
            os.close(validation_descriptor)
        return bytes(payload)
    finally:
        if leaf_descriptor is not None:
            os.close(leaf_descriptor)
        os.close(descriptor)


class _ConcurrentSpecChange(RuntimeError):
    """Canonical execution authority changed during Handoff composition."""


@dataclass(frozen=True, slots=True)
class _TypedChildGate:
    """One synchronous child dispatch plus its durable typed outcome."""

    workflow: str
    exit_code: int | None
    outcome: _outcome.WorkflowOutcome | None
    run_dir: Path | None
    error_kind: str | None
    error_message: str | None
    escaped_terminal: bool = False

    @property
    def succeeded(self) -> bool:
        if self.error_kind is not None or self.outcome is None:
            return False
        if (
            self.outcome.workflow != self.workflow
            or self.outcome.error_kind is not None
            or self.outcome.error_message is not None
        ):
            return False
        if self.exit_code != 0 or (
            self.outcome.terminal_class is not _outcome.TerminalClass.SUCCEEDED
        ):
            return False
        if self.workflow == "breach":
            return self.outcome.detail_status in {"no_attacks", "proceed"}
        if self.workflow == "review":
            return self.outcome.detail_status == "pass"
        return False

    @property
    def revokes_paid_authority(self) -> bool:
        """Return whether this child made every later paid gate unsafe."""

        return bool(self.accounting_debt)

    @property
    def accounting_debt(self) -> tuple[str, ...]:
        """Return unknown usage dimensions for this paid child dispatch.

        A missing or corrupt child outcome cannot prove that transport never
        occurred.  Its numeric zero is therefore only a lower bound, exactly
        like a typed ``incomplete_accounting`` outcome.  Both cases revoke
        later paid authority even though only the latter retains a trustworthy
        child-reported partial cost.
        """

        if (
            self.escaped_terminal
            or self.outcome is None
            or self.error_kind == "child_outcome_contract_error"
            or self.outcome.terminal_class
            in {
                _outcome.TerminalClass.CONTRACT_ERROR,
                _outcome.TerminalClass.INTERRUPTED,
            }
            or self.outcome.detail_status.strip().lower()
            in {
                "budget_accounting_incomplete",
                "cost_incomplete",
                "incomplete_accounting",
                "process_lost",
            }
        ):
            return ("token_usage_complete", "cost_complete")
        return ()

    def ledger_entry(
        self,
        *,
        gate: str,
        candidate_diff_sha256: str,
    ) -> GateLedgerEntry:
        if self.outcome is None or self.error_kind == "child_outcome_contract_error":
            terminal_class = _outcome.TerminalClass.CONTRACT_ERROR.value
            decision = "child_outcome_contract_error"
        else:
            terminal_class = self.outcome.terminal_class.value
            decision = self.outcome.detail_status
        ledger_outcome = (
            None if self.error_kind == "child_outcome_contract_error" else self.outcome
        )
        return GateLedgerEntry(
            gate=gate,
            candidate_diff_sha256=candidate_diff_sha256,
            terminal_class=terminal_class,
            decision=decision,
            exit_code=self.exit_code,
            workflow_outcome=ledger_outcome,
            verification_evidence_ref=None,
            error_kind=self.error_kind,
            error_message=self.error_message,
        )


def _paid_gate_accounting_fields(
    debt_by_gate: dict[str, set[str]],
) -> dict[str, object]:
    """Project known child-accounting debt without upgrading lower bounds."""

    return {
        "cost_usd_is_lower_bound": any(
            "cost_complete" in missing for missing in debt_by_gate.values()
        ),
        "accounting_debt": bool(debt_by_gate),
        "accounting_debt_by_gate": {
            gate: sorted(missing) for gate, missing in sorted(debt_by_gate.items())
        },
    }


def _invoke_typed_child_gate(
    *,
    parent_run_dir: Path,
    operation_id: str,
    workflow: str,
    invoke: Callable[[], int],
    on_escaped_terminal: Callable[[_TypedChildGate], None] | None = None,
) -> _TypedChildGate:
    """Dispatch one child and require its saga-bound ``WorkflowOutcome``."""

    exit_code: int | None = None
    raised: BaseException | None = None
    escaped: BaseException | None = None
    escaped_traceback = None
    operation_stack = ExitStack()
    operation_stack.enter_context(
        _outcome.child_operation(
            parent_run_dir,
            operation_id=operation_id,
            workflow=workflow,
        )
    )
    try:
        returned = invoke()
        if isinstance(returned, bool) or not isinstance(returned, int):
            raise TypeError(f"{workflow} child returned non-integer exit code")
        exit_code = returned
    except BaseException as exc:  # noqa: BLE001 - classify control flow
        raised = exc
        if isinstance(exc, SystemExit) and isinstance(exc.code, int):
            exit_code = exc.code
        if _provider_cleanup_error_in_cause_chain(exc) is not None or not (
            isinstance(exc, (Exception, SystemExit))
        ):
            # Preserve cleanup ownership and control-flow identity, but defer
            # the re-raise until saga-bound child accounting is harvested.
            escaped = exc
            escaped_traceback = exc.__traceback__
        else:
            logger.exception(
                "handoff: child %s operation %s raised %s",
                workflow,
                operation_id,
                type(exc).__name__,
            )

    operation_scope_error: BaseException | None = None
    try:
        operation_stack.close()
    except BaseException as scope_exc:  # noqa: BLE001 - preserve retry owner
        operation_scope_error = scope_exc
        if escaped is None:
            escaped = scope_exc
            escaped_traceback = scope_exc.__traceback__
            raised = scope_exc
        else:
            escaped.add_note(
                "secondary child-operation cleanup failure: "
                f"{type(scope_exc).__name__}: {str(scope_exc)[:1000]}"
            )

    outcome = None
    run_dir = None
    contract_error: BaseException | None = operation_scope_error
    try:
        outcome = _outcome.child_outcome(parent_run_dir, operation_id)
    except BaseException as exc:  # noqa: BLE001 - preserve retry owner
        if escaped is not None:
            escaped.add_note(
                "secondary child-outcome read failure: "
                f"{type(exc).__name__}: {str(exc)[:1000]}"
            )
            contract_error = contract_error or exc
        elif isinstance(
            exc, (OSError, OverflowError, RuntimeError, TypeError, ValueError)
        ) and _provider_cleanup_error_in_cause_chain(exc) is None:
            contract_error = exc
        else:
            escaped = exc
            escaped_traceback = exc.__traceback__
            raised = exc
            contract_error = exc
    if outcome is not None and contract_error is None:
        try:
            run_dir = _outcome.child_run_dir(parent_run_dir, operation_id)
        except BaseException as exc:  # noqa: BLE001 - preserve retry owner
            if escaped is not None:
                escaped.add_note(
                    "secondary child-run binding failure: "
                    f"{type(exc).__name__}: {str(exc)[:1000]}"
                )
                contract_error = exc
            elif isinstance(
                exc, (OSError, OverflowError, RuntimeError, TypeError, ValueError)
            ) and _provider_cleanup_error_in_cause_chain(exc) is None:
                contract_error = exc
            else:
                escaped = exc
                escaped_traceback = exc.__traceback__
                raised = exc
                contract_error = exc

    error_kind = type(raised).__name__ if raised is not None else None
    error_message = str(raised)[:2000] if raised is not None else None
    if contract_error is not None:
        error_kind = "child_outcome_contract_error"
        error_message = str(contract_error)[:2000]
        run_dir = None
    elif outcome is not None:
        typed_success = outcome.terminal_class in {
            _outcome.TerminalClass.SUCCEEDED,
            _outcome.TerminalClass.NO_WORK,
        }
        exit_success = exit_code == 0
        if raised is None and typed_success != exit_success:
            error_kind = "child_exit_outcome_mismatch"
            error_message = (
                f"exit_code={exit_code!r} disagrees with terminal_class="
                f"{outcome.terminal_class.value!r}"
            )

    gate = _TypedChildGate(
        workflow=workflow,
        exit_code=exit_code,
        outcome=outcome,
        run_dir=run_dir,
        error_kind=error_kind,
        error_message=error_message,
        escaped_terminal=escaped is not None,
    )
    if escaped is not None:
        if on_escaped_terminal is not None:
            try:
                on_escaped_terminal(gate)
            except BaseException as sink_error:  # noqa: BLE001
                escaped.add_note(
                    "could not retain escaped child terminal accounting: "
                    f"{type(sink_error).__name__}: {str(sink_error)[:1000]}"
                )
        raise escaped.with_traceback(escaped_traceback)
    return gate


def _execution_structure_guard(
    expected_structure_hash: str,
    publication_guard: Callable[[], None] | None = None,
):
    """Return a Corpus transform that admits evidence-only concurrent refs."""

    def guard(candidate):
        observed = _partition.spec_execution_structure_hash(candidate)
        if observed != expected_structure_hash:
            raise _ConcurrentSpecChange(
                "Spec execution authority changed during handoff"
            )
        if publication_guard is not None:
            publication_guard()
        return candidate

    return guard


def _execution_authority_guard(
    expected_structure_hash: str,
    expected_implementation_refs: tuple[ContentHash, ...] | None = None,
    publication_guard: Callable[[], None] | None = None,
):
    """Admit safe evidence rebases while rejecting stale execution lineage."""

    structure_guard = _execution_structure_guard(
        expected_structure_hash,
        publication_guard,
    )

    def guard(candidate):
        candidate = structure_guard(candidate)
        if (
            expected_implementation_refs is not None
            and tuple(candidate.implementation_refs)
            != expected_implementation_refs
        ):
            raise _ConcurrentSpecChange(
                "Spec Implementation authority changed during handoff"
            )
        return candidate

    return guard


def _persist_implementation_guarded(
    corpus: Corpus,
    implementation: Implementation,
    *,
    expected_structure_hash: str,
    expected_implementation_refs: tuple[ContentHash, ...] | None = None,
    publication_guard: Callable[[], None] | None = None,
):
    expected_candidate_refs = None
    if expected_implementation_refs is not None:
        implementation_ref = ContentHash(hash_record(implementation))
        expected_candidate_refs = expected_implementation_refs
        if implementation_ref not in expected_candidate_refs:
            expected_candidate_refs += (implementation_ref,)
    result = corpus.commit(
        implementations=(implementation,),
        transform=_execution_authority_guard(
            expected_structure_hash,
            expected_candidate_refs,
            publication_guard,
        ),
    )
    return result.spec, result.refs.implementation_refs[0]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="crucible handoff",
        description=(
            "Record operator-completed work (manual or agent-assisted) "
            "as proper Implementation records on the spec."
        ),
    )
    add_spec_arg(p)
    target = p.add_mutually_exclusive_group()
    target.add_argument("--slice", help="hand-complete one slice id")
    target.add_argument("--tier", help="hand-complete every slice in one tier id")
    target.add_argument(
        "--all", action="store_true",
        help="hand-complete every pending slice in the spec",
    )
    p.add_argument(
        "--cwd", default=None,
        help="repo root the work was done in (default: process cwd)",
    )
    p.add_argument(
        "--operator", default="operator",
        help="who/what produced the work (recorded in the transcript note)",
    )
    p.add_argument("--note", default="", help="free-text completion note")
    p.add_argument(
        "--skip-verify", action="store_true",
        help="skip the build/verify gate (record capture only)",
    )
    p.add_argument(
        "--allow-empty", action="store_true",
        help="record an empty-diff target as soft_fail_no_changes instead of failing",
    )
    p.add_argument(
        "--conditional-absence-selector",
        help="repository-relative JSON decision artifact proving this slice is unselected",
    )
    p.add_argument(
        "--conditional-absence-json-pointer",
        help="strict RFC 6901 pointer selecting the deciding scalar",
    )
    p.add_argument(
        "--conditional-absence-equals",
        help="expected selector value encoded as a JSON scalar",
    )
    p.add_argument(
        "--conditional-absence-decision-slice",
        help="accepted dependency slice that owns the selector artifact",
    )
    p.add_argument(
        "--conditional-absence-selected-slice",
        help="accepted sibling slice selected by the decision",
    )
    p.add_argument(
        "--redo", action="store_true",
        help="re-record slices that already have a completed Implementation",
    )
    p.add_argument(
        "--land", action="store_true",
        help="stamp LifecycleEvent(kind='landed') when all targets complete",
    )
    p.add_argument(
        "--with-breach", action="store_true",
        help="after recording, run the breach gate (Breaker→Defender→Judge) "
             "on each recorded diff — LLM-backed; uses --budget-usd/--backend-config",
    )
    p.add_argument(
        "--with-review", action="store_true",
        help="after recording, run the review gate on each recorded diff "
             "— LLM-backed; rubric via --review-rubric",
    )
    p.add_argument(
        "--review-rubric", default="diff_scope",
        help="rubric for --with-review (default: diff_scope)",
    )
    p.add_argument(
        "--budget-usd", type=positive_budget_arg, default=None,
        help="budget forwarded to each chained gate invocation",
    )
    p.add_argument(
        "--backend-config", default=None,
        help="backend routing YAML forwarded to chained gates",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="preview the capture per slice; write nothing to the corpus",
    )
    return p.parse_args(argv)


def _usage_exit(msg: str) -> SystemExit:
    """Usage error → rc=2. ``SystemExit(str)`` exits 1; print + exit(2)."""
    print(f"crucible handoff: {msg}", file=sys.stderr)
    return SystemExit(2)


def _resolve_targets(spec, args) -> list:
    """Targeted (tier_id, slice) pairs in topo order, or SystemExit(2)."""
    if spec.tier_dag is None:
        raise _usage_exit("spec has no tier_dag — nothing to hand-complete")
    pairs = [
        (tier.tier_id, s)
        for tier in spec.tier_dag.topo_order()
        for s in tier.slices
    ]
    if args.slice:
        pairs = [(t, s) for (t, s) in pairs if s.id == args.slice]
        if not pairs:
            raise _usage_exit(f"slice {args.slice!r} not found in tier_dag")
    elif args.tier:
        pairs = [(t, s) for (t, s) in pairs if t == args.tier]
        if not pairs:
            raise _usage_exit(f"tier {args.tier!r} not found in tier_dag")
    elif not args.all:
        raise _usage_exit("pick a target: --slice <id>, --tier <id>, or --all")
    return pairs


def _handoff_execution(cwd: str) -> ExecutionContext:
    import datetime as _dt
    import socket as _sk
    return ExecutionContext(
        timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds"),
        machine=_sk.gethostname(),
        cwd=cwd,
        toolchain={},
        backend=None,   # deliberately None: a human / external agent did this
        model=None,
        recipe="handoff",
        recipe_commit=None,
        gpu_model=None,
        run_id=None,
    )


def _run_readonly_gate(
    operation,
    *,
    expected_candidate: tuple[bytes, tuple[str, ...], str],
    cwd: str,
    journal_root: Path,
    scope: tuple[str, ...],
    sequence: int,
    spec,
    parent_transaction_id: str | None = None,
):
    """Run a deterministic Handoff gate without accepting its writes."""

    checkpoint = MutationTransaction.begin(
        cwd=cwd,
        journal_root=journal_root,
        slice_id=f"handoff_gate_{sequence}",
        scope=scope,
        sequence=sequence,
        spec=spec,
        parent_transaction_id=parent_transaction_id,
    )
    try:
        before = (
            capture_workspace_delta(cwd=cwd, path_scope=scope)
            if scope
            else (b"", (), spec.project_context.base_commit)
        )
        if before != expected_candidate:
            _settle_no_process_writer_epoch(
                live_writer_epoch_family(checkpoint),
                phase="handoff_candidate_changed_before_gate",
                succeeded=False,
            )
            checkpoint.commit()
            return None, (
                "live workspace changed after candidate capture and before "
                "deterministic verification"
            )
        outcome = operation(
            writer_epoch_members=live_writer_epoch_family(checkpoint),
        )
        checkpoint.prepare_git_capture()
        after = (
            capture_workspace_delta(cwd=cwd, path_scope=scope)
            if scope
            else (b"", (), spec.project_context.base_commit)
        )
        integrity_error: str | None = None
        try:
            escapes = checkpoint.detected_scope_escapes()
            if escapes:
                raise RuntimeError(f"scope escapes: {list(escapes)!r}")
            checkpoint.assert_head_unchanged()
            checkpoint.assert_repository_metadata_unchanged()
            checkpoint.assert_index_unchanged()
        except Exception as exc:  # noqa: BLE001 - journal integrity boundary
            integrity_error = str(exc)
        if after != before or integrity_error is not None:
            if not checkpoint.restore():
                raise RuntimeError(
                    "failed to restore mutating deterministic Handoff gate"
                )
            return outcome, (
                "deterministic gate mutated source or Git authority; "
                f"integrity_error={integrity_error!r}"
            )
        checkpoint.commit()
        return outcome, None
    finally:
        if checkpoint.active and not checkpoint.restore():
            raise RuntimeError(
                "failed to settle deterministic Handoff gate journal"
            )


def _authoring_finalization_bindings(spec):
    """Reuse Tier's closed constraint validation for operator handoff."""

    from .tier import _authoring_finalization_bindings as resolve  # noqa: PLC0415

    return resolve(spec)



def _authoring_materialization_state(*args, **kwargs):
    """Apply Tier's descriptor-bound output classifier."""

    from .tier import _authoring_materialization_state as classify  # noqa: PLC0415

    return classify(*args, **kwargs)



def _restore_authoring_finalization_family(
    parent: MutationTransaction,
    child: MutationTransaction | None,
) -> None:
    """Restore a failed Handoff finalization children-first."""

    if child is not None and child.active and not child.restore(full=True):
        raise RuntimeError(
            "failed to restore Handoff authoring-finalization child"
        )
    if parent.active and not parent.restore(full=True):
        raise RuntimeError(
            "failed to restore Handoff authoring-finalization parent"
        )


def _run_authoring_finalization_gate(
    operation,
    *,
    expected_candidate: tuple[bytes, tuple[str, ...], str],
    cwd: str,
    journal_root: Path,
    scope: tuple[str, ...],
    materializes_paths: tuple[str, ...],
    parent_sequence: int,
    child_sequence: int,
    spec,
    workspace_root_descriptor: int,
) -> tuple[
    BuildVerifyOutcome | None,
    MutationTransaction | None,
    tuple[tuple[str, str], ...],
    str | None,
]:
    """Run one exact output-producing gate under a nested rollback family."""

    parent = MutationTransaction.begin(
        cwd=cwd,
        journal_root=journal_root,
        slice_id=f"handoff_authoring_parent_{parent_sequence}",
        scope=scope,
        sequence=parent_sequence,
        spec=spec,
    )
    child: MutationTransaction | None = None
    try:
        before = capture_workspace_delta(cwd=cwd, path_scope=scope)
        if before != expected_candidate:
            _settle_no_process_writer_epoch(
                live_writer_epoch_family(parent),
                phase="handoff_candidate_changed_before_finalization",
                succeeded=False,
            )
            _restore_authoring_finalization_family(parent, child)
            return None, None, (), (
                "live workspace changed after candidate capture and before "
                "authoring finalization"
            )
        child = MutationTransaction.begin(
            cwd=cwd,
            journal_root=journal_root,
            slice_id=f"handoff_authoring_child_{child_sequence}",
            scope=materializes_paths,
            sequence=child_sequence,
            spec=spec,
            parent_transaction_id=parent.transaction_id,
        )
        outcome = operation(live_writer_epoch_family(child))
        child.prepare_git_capture()
        child_escapes = child.detected_scope_escapes()
        child.assert_head_unchanged()
        child.assert_repository_metadata_unchanged()
        child.assert_index_unchanged()
        changed_paths = child.locally_changed_paths(materializes_paths)
        present, missing, invalid, output_sha256 = (
            _authoring_materialization_state(
                workspace_root_descriptor,
                materializes_paths,
            )
        )
        parent.prepare_git_capture()
        parent_escapes = parent.detected_scope_escapes()
        parent.assert_head_unchanged()
        parent.assert_repository_metadata_unchanged()
        parent.assert_index_unchanged()
        if not outcome.classification.startswith("success"):
            _restore_authoring_finalization_family(parent, child)
            return outcome, None, (), (
                "authoring-finalization gate failed: "
                f"{outcome.classification}"
            )
        if (
            child_escapes
            or parent_escapes
            or changed_paths != materializes_paths
            or present != materializes_paths
            or missing
            or invalid
        ):
            error = (
                "authoring-finalization gate did not materialize only its "
                "exact authorized output set"
            )
            _restore_authoring_finalization_family(parent, child)
            return outcome, None, (), error
        child.commit()
        return (
            outcome,
            parent,
            tuple(output_sha256.items()),
            None,
        )
    except BaseException:
        _restore_authoring_finalization_family(parent, child)
        raise


def _candidate_integrity_error(
    transaction: MutationTransaction,
    *,
    corpus: Corpus,
    cwd: str,
    scope: tuple[str, ...],
    spec,
    implementation: Implementation,
) -> str | None:
    """Return why live gate output cannot bind the terminal Implementation."""

    try:
        transaction.prepare_git_capture()
        observed = (
            capture_workspace_delta(cwd=cwd, path_scope=scope)
            if scope
            else (b"", (), spec.project_context.base_commit)
        )
        expected = (
            corpus.get_artifact(implementation.diff_ref),
            implementation.files_modified,
            implementation.base_commit,
        )
        if observed != expected:
            return "live workspace does not match the candidate diff binding"
        escapes = transaction.detected_scope_escapes()
        if escapes:
            return f"gate wrote outside selected slice authority: {list(escapes)!r}"
        transaction.assert_head_unchanged()
        transaction.assert_repository_metadata_unchanged()
        transaction.assert_index_unchanged()
    except Exception as exc:  # noqa: BLE001 - journal integrity boundary
        return str(exc)
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    corpus = Corpus(args.spec)
    spec = corpus.latest_spec()
    if spec is None:
        message = f"no spec manifest under {args.spec}"
        record_handled_terminal(
            corpus, "handoff", status="setup_error",
            message=message, error_kind="missing_spec",
            provenance_root=None,
        )
        return 2
    expected_structure_hash = _partition.spec_execution_structure_hash(spec)
    cwd = args.cwd or str(Path.cwd())

    pairs = _resolve_targets(spec, args)
    prior = _partition._latest_impl_per_slice(  # noqa: SLF001
        corpus,
        {s.id for _, s in pairs},
        refs=spec.implementation_refs,
    )
    tiers_by_id = {tier.tier_id: tier for tier in spec.tier_dag.tiers}
    slice_locations = {
        slice_.id: (tier, slice_)
        for tier in spec.tier_dag.tiers
        for slice_ in tier.slices
    }
    conditional_values = (
        args.conditional_absence_selector,
        args.conditional_absence_json_pointer,
        args.conditional_absence_equals,
        args.conditional_absence_decision_slice,
        args.conditional_absence_selected_slice,
    )
    conditional_mode = any(value is not None for value in conditional_values)
    conditional_expected: object | None = None
    conditional_authority_resolver: (
        Callable[
            [int],
            tuple[
                Implementation,
                Implementation,
                bytes,
                tuple[ContentHash, ...],
            ],
        ]
        | None
    ) = None
    if conditional_mode:
        if any(value is None for value in conditional_values):
            raise _usage_exit("all --conditional-absence-* options are required together")
        if not args.slice or len(pairs) != 1:
            raise _usage_exit("conditional absence requires exactly one --slice target")
        if not args.allow_empty:
            raise _usage_exit("conditional absence requires --allow-empty")
        if args.skip_verify or args.with_breach or args.with_review:
            raise _usage_exit(
                "conditional absence cannot be combined with --skip-verify or child gates"
            )
        try:
            conditional_expected = json.loads(args.conditional_absence_equals)
        except json.JSONDecodeError as exc:
            raise _usage_exit(
                "--conditional-absence-equals must be valid JSON"
            ) from exc
        if isinstance(conditional_expected, (dict, list)):
            raise _usage_exit(
                "--conditional-absence-equals must encode a JSON scalar"
            )
        target_tier_id, target_slice = pairs[0]
        if target_slice.directive.files_in_scope:
            raise _usage_exit(
                "conditional absence requires an empty files_in_scope declaration"
            )
        if not target_slice.directive.creates_paths:
            raise _usage_exit(
                "conditional absence requires at least one declared creates_path"
            )
        decision_location = slice_locations.get(
            args.conditional_absence_decision_slice
        )
        selected_location = slice_locations.get(
            args.conditional_absence_selected_slice
        )
        if decision_location is None or selected_location is None:
            raise _usage_exit("conditional authority slice id was not found")
        decision_tier, decision_slice = decision_location
        selected_tier, selected_slice = selected_location
        dependency_ids: set[str] = set()
        pending_dependencies = list(tiers_by_id[target_tier_id].depends_on)
        while pending_dependencies:
            dependency_id = pending_dependencies.pop()
            if dependency_id in dependency_ids or dependency_id not in tiers_by_id:
                continue
            dependency_ids.add(dependency_id)
            pending_dependencies.extend(tiers_by_id[dependency_id].depends_on)
        if decision_tier.tier_id not in dependency_ids:
            raise _usage_exit(
                "decision slice must belong to a transitive dependency tier"
            )
        if (
            selected_tier.tier_id != target_tier_id
            or selected_slice.id == target_slice.id
        ):
            raise _usage_exit("selected slice must be a distinct sibling of the target")
        if args.conditional_absence_selector not in {
            *decision_slice.directive.files_in_scope,
            *decision_slice.directive.creates_paths,
        }:
            raise _usage_exit("decision slice does not own the selector path")

        def resolve_conditional_authority(
            root_descriptor: int,
        ) -> tuple[
            Implementation,
            Implementation,
            bytes,
            tuple[ContentHash, ...],
        ]:
            """Capture dynamic conditional authority while holding the workspace lock."""

            nonlocal spec
            authority_spec = corpus.latest_spec()
            if (
                authority_spec is None
                or _partition.spec_execution_structure_hash(authority_spec)
                != expected_structure_hash
            ):
                raise _usage_exit(
                    "Spec execution authority changed during conditional handoff"
                )
            spec = authority_spec
            latest_authorities = _partition._latest_impl_per_slice(  # noqa: SLF001
                corpus,
                {decision_slice.id, selected_slice.id},
                refs=authority_spec.implementation_refs,
            )
            decision_impl = latest_authorities.get(decision_slice.id)
            selected_impl = latest_authorities.get(selected_slice.id)
            authority_bindings = (
                ("decision", decision_tier, decision_slice, decision_impl),
                ("selected", selected_tier, selected_slice, selected_impl),
            )

            def require_exact_live_authority() -> None:
                for (
                    label,
                    owner_tier,
                    owner_slice,
                    implementation,
                ) in authority_bindings:
                    if (
                        implementation is None
                        or not _partition._is_skip_worthy(  # noqa: SLF001
                            implementation.classification,
                            owner_slice,
                            implementation,
                        )
                        or not _partition._identity_matches(  # noqa: SLF001
                            corpus,
                            authority_spec,
                            owner_tier,
                            owner_slice,
                            implementation,
                        )
                        or (
                            _partition._is_handoff_candidate(  # noqa: SLF001
                                implementation
                            )
                            and not (
                                _partition._verification_matches_handoff_authority(  # noqa: SLF001
                                    corpus,
                                    implementation,
                                    spec=authority_spec,
                                    tier_id=owner_tier.tier_id,
                                    slice_id=owner_slice.id,
                                    spec_execution_authority_hash=(
                                        expected_structure_hash
                                    ),
                                    require_gate=True,
                                )
                            )
                        )
                    ):
                        raise _usage_exit(
                            f"{label} slice lacks an exact required-pass implementation"
                        )
                    authority_scope = (
                        *owner_slice.directive.files_in_scope,
                        *owner_slice.directive.creates_paths,
                    )
                    try:
                        live_authority = capture_workspace_delta(
                            cwd=cwd,
                            path_scope=authority_scope,
                        )
                        recorded_authority = (
                            corpus.get_artifact(implementation.diff_ref),
                            implementation.files_modified,
                            implementation.base_commit,
                        )
                    except (KeyError, OSError, TypeError, ValueError) as exc:
                        raise _usage_exit(
                            f"{label} authority candidate is unavailable: {exc}"
                        ) from exc
                    if live_authority != recorded_authority:
                        raise _usage_exit(
                            f"live {label} authority bytes do not match its "
                            "exact retained implementation"
                        )

            require_exact_live_authority()
            try:
                selector_bytes = _read_bound_conditional_selector(
                    root_descriptor,
                    args.conditional_absence_selector,
                )
            except (OSError, ValueError) as exc:
                raise _usage_exit(
                    f"conditional selector is unavailable: {exc}"
                ) from exc
            # Re-prove both complete authority projections after the selector
            # read. This closes the useful uncoordinated-mutation window
            # without trusting pathnames or a pre-lock Corpus snapshot.
            require_exact_live_authority()

            final_spec = corpus.latest_spec()
            if (
                final_spec is None
                or _partition.spec_execution_structure_hash(final_spec)
                != expected_structure_hash
                or tuple(final_spec.implementation_refs)
                != tuple(authority_spec.implementation_refs)
            ):
                raise _usage_exit(
                    "Spec Implementation authority changed while binding "
                    "conditional handoff evidence"
                )
            assert decision_impl is not None
            assert selected_impl is not None
            return (
                decision_impl,
                selected_impl,
                selector_bytes,
                tuple(authority_spec.implementation_refs),
            )

        conditional_authority_resolver = resolve_conditional_authority

    # Out-of-scope detection, computed ONCE against the union of every
    # targeted slice's scope (review finding): per-slice unscoped
    # captures both re-walked the whole tree N times and misreported a
    # sibling slice's in-scope edits as "out of scope" in --tier/--all
    # batches. Files listed here are modified but inside NO targeted
    # slice's lane — informational, never captured, never fatal.
    union_scope: set[str] = set()
    for _, s in pairs:
        union_scope |= set(s.directive.files_in_scope)
        union_scope |= set(s.directive.creates_paths)
    rows: list[dict] = []
    gate_rows: list[dict] = []
    recorded_impls: dict[str, tuple[Implementation, str]] = {}
    initial_recorded_impls: dict[str, tuple[Implementation, str]] = {}
    skipped_impls_for_land: dict[str, Implementation] = {}
    slices_by_id = {sl.id: sl for _, sl in pairs}
    tier_id_by_slice = {sl.id: tier_id for tier_id, sl in pairs}
    failed = False
    paid_gate_authority_revoked = False
    paid_gate_cost_usd = 0.0
    paid_gate_accounting_debt_by_gate: dict[str, set[str]] = {}
    concurrent_structure_error: str | None = None
    journal_sequence = 0
    authoring_finalizations: dict[str, object] = {}

    def next_journal_sequence() -> int:
        nonlocal journal_sequence
        journal_sequence += 1
        return journal_sequence

    def present_or_unsafe_conditional_outputs(
        root_descriptor: int,
        paths: tuple[str, ...],
    ) -> list[str]:
        rejected: list[str] = []
        for path in paths:
            try:
                absent = bound_workspace_path_absent(
                    root_descriptor,
                    path,
                )
            except (OSError, ValueError):
                absent = False
            if not absent:
                rejected.append(path)
        return rejected

    def require_conditional_outputs_absent(
        paths: tuple[str, ...],
    ) -> None:
        rejected = present_or_unsafe_conditional_outputs(
            workspace_authority.root_descriptor,
            paths,
        )
        if rejected:
            raise _ConcurrentSpecChange(
                f"conditional output authority changed during publication: {rejected!r}"
            )

    def paid_gate_failure_terminal(
        _terminal_error: BaseException,
    ) -> FailureTerminal | None:
        """Select debt-aware failure only after every rollback has run."""

        if not paid_gate_accounting_debt_by_gate:
            return None
        return FailureTerminal(
            status="incomplete_accounting",
            summary_fields={
                **_paid_gate_accounting_fields(paid_gate_accounting_debt_by_gate),
                "targets": list(rows),
                "gates": list(gate_rows),
                "landed": False,
            },
        )

    workspace_acquisition = _RetainedCleanupStack()
    with workspace_acquisition:
        workspace_authority = workspace_acquisition.enter_context(
            recovery_workspace_mutation_lock(corpus, cwd)
        )
        workspace_release = workspace_acquisition.pop_all()
    workspace_acquisition_guard = ExitStack()
    workspace_acquisition_guard.callback(workspace_release.close)
    rollback_stack = _RetainedCleanupStack()

    with (
        workspace_acquisition_guard,
        open_run_with_capture(
            corpus,
            "handoff",
            provenance_root=Path(cwd),
            failure_cost_usd=lambda: paid_gate_cost_usd,
            failure_terminal=paid_gate_failure_terminal,
        ) as handle,
        workspace_release,
        rollback_stack,
        _DeferHandoffCleanupToProviderOwner(
            rollback_stack,
            workspace_release,
        ),
    ):
        if args.dry_run:
            try:
                mutation_journals = _mutation_journal_inventory(corpus, cwd)
            except Exception as exc:  # noqa: BLE001 - inventory must fail closed
                message = (
                    "mutation journal inventory failed before dry-run; "
                    f"no recovery was attempted: {type(exc).__name__}: {exc}"
                )
                summary = {
                    "status": "setup_error",
                    "error_kind": "mutation_journal_inventory_failed",
                    "error": message,
                    "mutation_journals": [],
                }
                handle.write_summary(summary)
                print(json.dumps(summary, indent=2))
                handle.finalize_elapsed(RunResult(status="setup_error"))
                logger.error("handoff: %s", message)
                return 2
            if mutation_journals:
                message = (
                    "dry-run refused because retained mutation journals require "
                    "recovery or cleanup; no recovery was attempted"
                )
                summary = {
                    "status": "setup_error",
                    "error_kind": "mutation_recovery_required",
                    "error": message,
                    "mutation_journals": mutation_journals,
                }
                handle.write_summary(summary)
                print(json.dumps(summary, indent=2))
                handle.finalize_elapsed(RunResult(status="setup_error"))
                logger.error("handoff: %s", message)
                return 2
        else:
            recovery = recover_mutation_journals(corpus, cwd)
            if recovery.failures:
                message = "mutation recovery failed closed: " + "; ".join(
                    recovery.failures
                )
                handle.write_summary({"status": "setup_error", "error": message})
                handle.finalize_elapsed(RunResult(status="setup_error"))
                logger.error("handoff: %s", message)
                return 2

        locked_spec = corpus.latest_spec()
        if (
            locked_spec is None
            or _partition.spec_execution_structure_hash(locked_spec)
            != expected_structure_hash
        ):
            message = "Spec execution authority changed before handoff lock"
            handle.write_summary({
                "status": "setup_error",
                "error_kind": "concurrent_spec_change",
                "error": message,
            })
            handle.finalize_elapsed(RunResult(status="setup_error"))
            logger.error("handoff: %s", message)
            return 2
        # Implementation refs are execution authority too. Refresh them only
        # after acquiring the workspace lock so skip/conditional decisions do
        # not bind a pre-lock Corpus snapshot.
        spec = locked_spec
        pairs = _resolve_targets(spec, args)
        try:
            configured_finalizations = _authoring_finalization_bindings(spec)
        except ValueError as exc:
            message = f"authoring finalization authority is invalid: {exc}"
            handle.write_summary({
                "status": "setup_error",
                "error_kind": "invalid_authoring_finalization",
                "error": message,
            })
            handle.finalize_elapsed(RunResult(status="setup_error"))
            logger.error("handoff: %s", message)
            return 2
        authoring_finalizations = {
            slice_.id: configured_finalizations[slice_.id]
            for _tier_id, slice_ in pairs
            if slice_.id in configured_finalizations
        }
        if authoring_finalizations and args.skip_verify:
            message = (
                "--skip-verify is incompatible with Spec-bound authoring "
                "finalization for slices "
                f"{sorted(authoring_finalizations)!r}"
            )
            handle.write_summary({
                "status": "setup_error",
                "error_kind": "invalid_authoring_finalization",
                "error": message,
            })
            handle.finalize_elapsed(RunResult(status="setup_error"))
            logger.error("handoff: %s", message)
            return 2
        if authoring_finalizations and (args.with_breach or args.with_review):
            requested_gates = tuple(
                gate
                for gate, requested in (
                    ("--with-breach", args.with_breach),
                    ("--with-review", args.with_review),
                )
                if requested
            )
            message = (
                "follow-on gates are incompatible with Spec-bound authoring "
                "finalization; run them after Handoff through Tier: "
                f"{list(requested_gates)!r}"
            )
            handle.write_summary({
                "status": "setup_error",
                "error_kind": "invalid_authoring_finalization",
                "error": message,
            })
            handle.finalize_elapsed(RunResult(status="setup_error"))
            logger.error("handoff: %s", message)
            return 2
        if authoring_finalizations and conditional_mode:
            message = (
                "conditional absence is incompatible with Spec-bound "
                "authoring finalization"
            )
            handle.write_summary({
                "status": "setup_error",
                "error_kind": "invalid_authoring_finalization",
                "error": message,
            })
            handle.finalize_elapsed(RunResult(status="setup_error"))
            logger.error("handoff: %s", message)
            return 2
        if args.with_breach:
            try:
                for _tier_id, selected_slice in pairs:
                    if not (
                        selected_slice.directive.files_in_scope
                        or selected_slice.directive.creates_paths
                    ):
                        # Verification-only slices cannot seed Breach: their
                        # scoped candidate is necessarily empty and the child
                        # gate is deterministically skipped.
                        continue
                    preflight_breach_observation_binding(
                        spec,
                        language=resolve_breach_observation_language(
                            spec,
                            selected_slice.directive,
                        ),
                        gate="verify",
                        slice_id=selected_slice.id,
                        cwd=cwd,
                    )
            except (TypeError, ValueError) as exc:
                message = f"selected breach observation binding is unavailable: {exc}"
                handle.write_summary(
                    {
                        "status": "setup_error",
                        "error_kind": "breach_observation_binding",
                        "error": message,
                    }
                )
                handle.finalize_elapsed(RunResult(status="setup_error"))
                logger.error("handoff: %s", message)
                return 2
        prior = _partition._latest_impl_per_slice(  # noqa: SLF001
            corpus,
            {slice_.id for _, slice_ in pairs},
            refs=spec.implementation_refs,
        )
        tiers_by_id = {
            tier.tier_id: tier for tier in spec.tier_dag.tiers
        }
        slices_by_id = {slice_.id: slice_ for _, slice_ in pairs}
        tier_id_by_slice = {
            slice_.id: tier_id for tier_id, slice_ in pairs
        }

        def workspace_capture_failed(
            exc: GitCaptureError,
            *,
            tier_id: str | None = None,
            slice_id: str | None = None,
            out_of_scope_changes: list[str] | None = None,
        ) -> int:
            """Publish the setup terminal for a delta Git could not capture."""

            target = (
                "the working tree" if slice_id is None else f"slice {slice_id!r}"
            )
            message = (
                f"cannot capture the working-tree delta of {target} in "
                f"{cwd!r}: {exc}"
            )
            if slice_id is not None:
                rows.append({
                    "tier": tier_id, "slice": slice_id, "action": "failed",
                    "reason": f"capture error: {exc}",
                    "out_of_scope_changes": out_of_scope_changes or [],
                })
            # Targets recorded earlier in this run keep their own evidence;
            # the rows keep the typed terminal from reading as "nothing ran".
            summary = {
                "status": "setup_error",
                "error_kind": "workspace_capture_failed",
                "error": message,
                "targets": rows,
                "landed": False,
            }
            handle.write_summary(summary)
            print(json.dumps(summary, indent=2))
            handle.finalize_elapsed(RunResult(status="setup_error"))
            logger.error("handoff: %s", message)
            return 2

        try:
            _, all_modified, _ = capture_workspace_delta(
                cwd=cwd, path_scope=None,
            )
        except GitCaptureError as exc:
            return workspace_capture_failed(exc)
        out_of_scope = sorted(set(all_modified) - union_scope)
        if out_of_scope:
            logger.warning(
                "handoff: %d modified file(s) lie outside every targeted "
                "slice's scope and will NOT be captured: %s",
                len(out_of_scope), out_of_scope[:5],
            )
        for tier_id, sl in pairs:
            d = sl.directive
            scope = tuple(d.files_in_scope) + tuple(d.creates_paths)
            already = prior.get(sl.id)
            try:
                prior_candidate_matches = bool(
                    already is not None
                    and (
                        (
                            capture_workspace_delta(
                                cwd=cwd,
                                path_scope=scope,
                            )
                            if scope
                            else (
                                b"",
                                (),
                                spec.project_context.base_commit,
                            )
                        )
                        == (
                            corpus.get_artifact(already.diff_ref),
                            already.files_modified,
                            already.base_commit,
                        )
                    )
                )
            except (GitCaptureError, KeyError, OSError, TypeError, ValueError):
                # An uncapturable live candidate cannot prove it matches the
                # prior record. The scoped capture below owns the terminal.
                prior_candidate_matches = False
            prior_verification_admitted = bool(
                already is not None
                and (args.skip_verify or bool(scope))
                and (
                    (
                        conditional_mode
                        and sl.id in _partition.partition_for_required_verification(
                            corpus,
                            tiers_by_id[tier_id],
                            spec=spec,
                        ).verified_retained
                    )
                    or not _partition._is_handoff_candidate(  # noqa: SLF001
                        already
                    )
                    or (
                        _partition._verification_matches_handoff_authority(  # noqa: SLF001
                            corpus,
                            already,
                            spec=spec,
                            tier_id=tier_id,
                            slice_id=sl.id,
                            spec_execution_authority_hash=(
                                expected_structure_hash
                            ),
                            require_gate=not args.skip_verify,
                        )
                    )
                )
            )
            prior_requested_gates_admitted = bool(
                already is not None
                and handoff_gate_ledger_matches(
                    corpus,
                    already,
                    tier_id=tier_id,
                    slice_id=sl.id,
                    spec_execution_authority_hash=expected_structure_hash,
                    require_breach=args.with_breach,
                    review_rubric=(
                        args.review_rubric if args.with_review else None
                    ),
                )
            )
            if (
                already is not None
                and _partition._is_skip_worthy(  # noqa: SLF001
                    already.classification, sl,
                )
                and _partition._identity_matches(  # noqa: SLF001
                    corpus, spec, tiers_by_id[tier_id], sl, already,
                )
                and prior_verification_admitted
                and prior_requested_gates_admitted
                and prior_candidate_matches
                and not args.redo
            ):
                skipped_impls_for_land[sl.id] = already
                rows.append({
                    "tier": tier_id, "slice": sl.id, "action": "skipped",
                    "reason": f"already {already.classification} (use --redo)",
                })
                continue

            if scope:
                try:
                    diff_bytes, files_modified, base_commit = (
                        capture_workspace_delta(cwd=cwd, path_scope=scope)
                    )
                except GitCaptureError as exc:
                    return workspace_capture_failed(
                        exc,
                        tier_id=tier_id,
                        slice_id=sl.id,
                        out_of_scope_changes=out_of_scope,
                    )
            else:
                # ``None`` means unscoped capture. An explicitly empty slice
                # is verification-only and must never absorb ambient dirt.
                diff_bytes = b""
                files_modified = ()
                base_commit = spec.project_context.base_commit

            directory_entries = [
                path for path in files_modified if path.endswith("/")
            ]
            if directory_entries:
                # Before any gate, preview row, or record: the entry would
                # otherwise be journaled, previewed, or recorded as content.
                rows.append({
                    "tier": tier_id, "slice": sl.id, "action": "failed",
                    "reason": (
                        "scoped capture lists directory entries "
                        f"{directory_entries!r}: Git reports an untracked "
                        "nested repository as one path and the captured "
                        "diff cannot carry its content"
                    ),
                    "out_of_scope_changes": out_of_scope,
                })
                failed = True
                continue

            if not files_modified and not args.allow_empty:
                rows.append({
                    "tier": tier_id, "slice": sl.id, "action": "failed",
                    "reason": "empty diff in slice scope (pass --allow-empty "
                              "if the slice is deliberately diffless)",
                    "out_of_scope_changes": out_of_scope,
                })
                failed = True
                continue

            authoring_finalization = authoring_finalizations.get(sl.id)
            finalization_transaction: MutationTransaction | None = None
            finalization_output_sha256: tuple[tuple[str, str], ...] = ()
            note_blob = json.dumps({
                "mode": "handoff",
                "operator": args.operator,
                "note": args.note,
                "tier": tier_id,
                "slice": sl.id,
            }, indent=2).encode("utf-8")
            transcript_ref: ContentHash | None = None
            if authoring_finalization is not None and not args.dry_run:
                transcript_ref = corpus.put_artifact(
                    note_blob,
                    kind="transcripts",
                    name_hint=f"handoff_{sl.id}.json",
                )

            verify_outcome: BuildVerifyOutcome | None = None
            verify_summary = None
            if not args.skip_verify and not conditional_mode and not args.dry_run:
                if authoring_finalization is None:
                    verify_outcome, gate_integrity_error = _run_readonly_gate(
                        partial(
                            build_and_verify,
                            spec,
                            slice_id=sl.id,
                            cwd=cwd,
                            log_dir=handle.run_dir,
                            slice_directive=d,
                        ),
                        expected_candidate=(
                            diff_bytes,
                            files_modified,
                            base_commit,
                        ),
                        cwd=cwd,
                        journal_root=handle.run_dir / "mutations",
                        scope=tuple(dict.fromkeys(scope)),
                        sequence=next_journal_sequence(),
                        spec=spec,
                    )
                else:
                    persisted_spec, _ = corpus.get_spec_with_digest(
                        spec.version
                    )
                    if persisted_spec != spec:
                        raise RuntimeError(
                            "locked Spec differs from its persisted authority"
                        )
                    assert transcript_ref is not None
                    parent_sequence = next_journal_sequence()
                    child_sequence = next_journal_sequence()

                    def run_finalizing_gate(
                        writer_epoch_members,
                        *,
                        authoring_spec=spec,
                        authoring_slice_id=sl.id,
                        finalization_constraint=authoring_finalization,
                        slice_directive=d,
                    ) -> BuildVerifyOutcome:
                        return build_and_verify(
                            authoring_spec,
                            slice_id=authoring_slice_id,
                            cwd=cwd,
                            log_dir=handle.run_dir,
                            slice_directive=slice_directive,
                            gate=finalization_constraint.gate,
                            writer_epoch_members=writer_epoch_members,
                        )

                    (
                        verify_outcome,
                        finalization_transaction,
                        finalization_output_sha256,
                        gate_integrity_error,
                    ) = _run_authoring_finalization_gate(
                        run_finalizing_gate,
                        expected_candidate=(
                            diff_bytes,
                            files_modified,
                            base_commit,
                        ),
                        cwd=cwd,
                        journal_root=handle.run_dir / "mutations",
                        scope=tuple(dict.fromkeys(scope)),
                        materializes_paths=tuple(
                            authoring_finalization.materializes_paths
                        ),
                        parent_sequence=parent_sequence,
                        child_sequence=child_sequence,
                        spec=spec,
                        workspace_root_descriptor=(
                            workspace_authority.root_descriptor
                        ),
                    )
                    if finalization_transaction is not None:
                        def settle_finalization_transaction(
                            transaction=finalization_transaction,
                        ) -> None:
                            if (
                                transaction.active
                                and not transaction.settle_or_restore(corpus)
                            ):
                                raise RuntimeError(
                                    "failed to settle Handoff authoring "
                                    "finalization"
                                )

                        rollback_stack.callback(
                            settle_finalization_transaction
                        )
                if gate_integrity_error is not None:
                    rows.append({
                        "tier": tier_id, "slice": sl.id, "action": "failed",
                        "reason": gate_integrity_error,
                        "out_of_scope_changes": out_of_scope,
                    })
                    failed = True
                    continue
                if verify_outcome is None:
                    raise RuntimeError(
                        "deterministic Handoff gate returned no outcome"
                    )
                verify_summary = verify_outcome.classification
                if verify_outcome.classification == "runner_missing":
                    # Spec declares no build/verify runners (docs-only
                    # specs, external-runner setups). Verification is
                    # unavailable rather than failed — warn + record.
                    logger.warning(
                        "handoff %s: no build/verify runner configured — "
                        "recording without a verify gate",
                        sl.id,
                    )
                elif not verify_outcome.classification.startswith("success"):
                    rows.append(
                        {
                            "tier": tier_id,
                            "slice": sl.id,
                            "action": "failed",
                            "reason": (
                                f"build/verify gate: {verify_outcome.classification}"
                            ),
                            "out_of_scope_changes": out_of_scope,
                        }
                    )
                    failed = True
                    continue
                if authoring_finalization is not None:
                    diff_bytes, files_modified, base_commit = (
                        capture_workspace_delta(cwd=cwd, path_scope=scope)
                    )

            terminal_classification = (
                "success" if files_modified else "soft_fail_no_changes"
            )
            classification = (
                "pending_gates"
                if args.with_breach or args.with_review
                else terminal_classification
            )
            row = {
                "tier": tier_id, "slice": sl.id,
                "action": "would-record" if args.dry_run else "recorded",
                "classification": classification,
                "files_modified": list(files_modified),
                "diff_bytes": len(diff_bytes),
                "verify": (
                    "conditional_absence"
                    if conditional_mode
                    else verify_summary or ("skipped" if args.skip_verify else None)
                ),
                "out_of_scope_changes": out_of_scope,
            }
            conditional_parent_implementation_refs: tuple[ContentHash, ...] | None = (
                None
            )
            if authoring_finalization is not None:
                row["authoring_finalization"] = {
                    "gate": authoring_finalization.gate,
                    "materializes_paths": list(
                        authoring_finalization.materializes_paths
                    ),
                    "status": (
                        "would-run" if args.dry_run else "succeeded"
                    ),
                }
            if conditional_mode:
                present_outputs = present_or_unsafe_conditional_outputs(
                    workspace_authority.root_descriptor,
                    tuple(d.creates_paths),
                )
                if present_outputs:
                    rows.append(
                        {
                            **row,
                            "action": "failed",
                            "reason": "conditional output path exists",
                            "present_outputs": present_outputs,
                        }
                    )
                    failed = True
                    continue
                assert conditional_authority_resolver is not None
                (
                    conditional_decision_impl,
                    conditional_selected_impl,
                    conditional_selector_bytes,
                    conditional_parent_implementation_refs,
                ) = conditional_authority_resolver(
                    workspace_authority.root_descriptor
                )
                present_outputs = present_or_unsafe_conditional_outputs(
                    workspace_authority.root_descriptor,
                    tuple(d.creates_paths),
                )
                if present_outputs:
                    rows.append({
                        **row,
                        "action": "failed",
                        "reason": "conditional output path exists",
                        "present_outputs": present_outputs,
                    })
                    failed = True
                    continue
                try:
                    validate_conditional_selector(
                        conditional_selector_bytes,
                        json_pointer=args.conditional_absence_json_pointer,
                        expected_value=conditional_expected,
                    )
                except ValueError as exc:
                    rows.append({**row, "action": "failed", "reason": str(exc)})
                    failed = True
                    continue
            if args.dry_run:
                rows.append(row)
                continue

            diff_ref = corpus.put_artifact(
                diff_bytes, kind="diffs", name_hint=f"{sl.id}.diff",
            )
            if conditional_mode:
                assert conditional_selector_bytes is not None
                assert conditional_decision_impl is not None
                assert conditional_selected_impl is not None
                try:
                    conditional_ref = persist_conditional_absence_evidence(
                        corpus,
                        tier_id=tier_id,
                        slice_id=sl.id,
                        spec_execution_authority_hash=expected_structure_hash,
                        candidate_diff_ref=diff_ref,
                        base_commit=base_commit,
                        files_modified=files_modified,
                        creates_paths=tuple(d.creates_paths),
                        selector_path=args.conditional_absence_selector,
                        selector_bytes=conditional_selector_bytes,
                        json_pointer=args.conditional_absence_json_pointer,
                        expected_value=conditional_expected,
                        decision_slice_id=args.conditional_absence_decision_slice,
                        decision_implementation=conditional_decision_impl,
                        selected_slice_id=args.conditional_absence_selected_slice,
                        selected_implementation=conditional_selected_impl,
                    )
                except ValueError as exc:
                    rows.append({**row, "action": "failed", "reason": str(exc)})
                    failed = True
                    continue
                verification_ref = None
                row["conditional_absence_evidence_ref"] = str(conditional_ref)
            else:
                verification_ref = persist_verification_evidence(
                    corpus,
                    tier_id=tier_id,
                    slice_id=sl.id,
                    spec_execution_authority_hash=expected_structure_hash,
                    candidate_diff_ref=diff_ref,
                    base_commit=base_commit,
                    files_modified=files_modified,
                    outcome=verify_outcome,
                    skipped_by_policy=args.skip_verify,
                    build_timeout_s=DEFAULT_BUILD_TIMEOUT_S,
                    verify_timeout_s=DEFAULT_VERIFY_TIMEOUT_S,
                    gate=(
                        authoring_finalization.gate
                        if authoring_finalization is not None
                        else "verify"
                    ),
                    name_hint=f"handoff_{tier_id}_{sl.id}_initial",
                )
            finalization_ref: ContentHash | None = None
            if authoring_finalization is not None and not args.dry_run:
                assert transcript_ref is not None
                assert verification_ref is not None
                finalization_ref = (
                    persist_handoff_authoring_finalization_evidence(
                        corpus,
                        tier_id=tier_id,
                        slice_id=sl.id,
                        spec_execution_authority_hash=expected_structure_hash,
                        candidate_diff_ref=diff_ref,
                        base_commit=base_commit,
                        files_modified=files_modified,
                        gate=authoring_finalization.gate,
                        output_sha256=finalization_output_sha256,
                        transcript_ref=transcript_ref,
                        verification_evidence_ref=verification_ref,
                    )
                )
            # The chaining handle for the follow-on gates:
            #   crucible breach --spec <dir> --diff-hash <diff_ref>
            #   crucible review --spec <dir> --rubric diff_scope --artifact-hash <diff_ref>
            row["diff_ref"] = str(diff_ref)
            if verification_ref is not None:
                row["verification_evidence_ref"] = str(verification_ref)
            if transcript_ref is None:
                transcript_ref = corpus.put_artifact(
                    note_blob,
                    kind="transcripts",
                    name_hint=f"handoff_{sl.id}.json",
                )
            impl = Implementation(
                slice_id=sl.id,
                base_commit=base_commit,
                result_commit=None,
                diff_ref=diff_ref,
                transcript_ref=transcript_ref,
                files_modified=files_modified,
                classification=classification,
                execution=_handoff_execution(cwd),
                scope_escapes=(),  # reported above, deliberately not fatal
                artifacts=(
                    {CONDITIONAL_ABSENCE_ARTIFACT: conditional_ref}
                    if conditional_mode
                    else {
                        HANDOFF_VERIFICATION_ARTIFACT: verification_ref,
                        **(
                            {
                                HANDOFF_AUTHORIZING_FINALIZATION_ARTIFACT: (
                                    finalization_ref
                                )
                            }
                            if finalization_ref is not None
                            else {}
                        ),
                    }
                ),
                cost_usd=0.0,
            )
            impl = _partition.attach_resume_identity(
                corpus, spec, tiers_by_id[tier_id], sl, impl,
            )
            if finalization_transaction is not None:
                finalization_transaction.prepare(impl, keep=True)
            try:
                spec, implementation_ref = _persist_implementation_guarded(
                    corpus,
                    impl,
                    expected_structure_hash=expected_structure_hash,
                    expected_implementation_refs=(
                        conditional_parent_implementation_refs
                    ),
                    publication_guard=(
                        partial(
                            require_conditional_outputs_absent,
                            tuple(d.creates_paths),
                        )
                        if conditional_mode
                        else None
                    ),
                )
            except _ConcurrentSpecChange as exc:
                concurrent_structure_error = str(exc)
                if (
                    finalization_transaction is not None
                    and finalization_transaction.active
                    and not finalization_transaction.settle_or_restore(corpus)
                ):
                    raise RuntimeError(
                        "failed to restore Handoff authoring finalization "
                        "after concurrent Spec change"
                    ) from exc
                row.update({
                    "action": "failed",
                    "classification": "hard_fail_concurrent_spec_change",
                    "reason": concurrent_structure_error,
                })
                rows.append(row)
                failed = True
                break
            if finalization_transaction is not None:
                finalization_transaction.commit(attached=True)
            row["implementation_ref"] = str(implementation_ref)
            recorded_impls[sl.id] = (impl, terminal_classification)
            initial_recorded_impls[sl.id] = (impl, terminal_classification)
            handle.write_event({"slice": sl.id, "classification": classification})
            rows.append(row)

        # Convenience gate chaining (--with-breach / --with-review): run
        # the adversarial gates on each recorded diff, exactly as the
        # documented manual chain would. Each gate is its own workflow
        # invocation (own run folder, own budget) — composition, not
        # re-implementation. A non-zero gate rc marks the handoff failed
        # (and suppresses --land) but ordinary remaining gates still run so
        # the operator sees the full picture. Typed incomplete accounting is
        # different: known debt cannot authorize another paid dispatch.
        gate_evidence_by_slice: dict[str, list[GateLedgerEntry]] = {}
        gate_failed_by_slice: dict[str, bool] = {}
        if (
            (args.with_breach or args.with_review)
            and not args.dry_run
            and concurrent_structure_error is None
        ):
            forwarded: list[str] = []
            if args.budget_usd is not None:
                forwarded += ["--budget-usd", str(args.budget_usd)]
            if args.backend_config is not None:
                forwarded += ["--backend-config", str(args.backend_config)]
            for gate_index, row in enumerate(rows):
                ref = row.get("diff_ref")
                if not ref:
                    continue  # skipped / failed / dry rows have no diff
                slice_id = row["slice"]
                selected_slice = slices_by_id[slice_id]
                slice_gate_evidence = gate_evidence_by_slice.setdefault(
                    slice_id,
                    [],
                )
                initial_verification_decision = str(row.get("verify") or "unavailable")
                slice_gate_evidence.append(
                    GateLedgerEntry(
                        gate="initial_verify",
                        candidate_diff_sha256=str(ref),
                        terminal_class=(
                            _outcome.TerminalClass.NO_WORK.value
                            if initial_verification_decision
                            in {"skipped", "runner_missing"}
                            else _outcome.TerminalClass.SUCCEEDED.value
                        ),
                        decision=initial_verification_decision,
                        exit_code=0,
                        workflow_outcome=None,
                        verification_evidence_ref=str(row["verification_evidence_ref"]),
                        error_kind=None,
                        error_message=None,
                    )
                )

                def record_paid_child_gate(
                    child: _TypedChildGate,
                    *,
                    gate_name: str,
                    debt_gate_name: str,
                    candidate_diff_sha256: str,
                    target_slice_id: str,
                    target_gate_evidence: list[GateLedgerEntry],
                ) -> GateLedgerEntry:
                    """Retain one child before normal return or escaped unwind."""

                    nonlocal failed
                    nonlocal paid_gate_authority_revoked
                    nonlocal paid_gate_cost_usd

                    if child.outcome is not None:
                        paid_gate_cost_usd += child.outcome.cost_usd
                    debt = child.accounting_debt
                    if debt:
                        paid_gate_accounting_debt_by_gate.setdefault(
                            debt_gate_name,
                            set(),
                        ).update(debt)
                    entry = child.ledger_entry(
                        gate=gate_name,
                        candidate_diff_sha256=candidate_diff_sha256,
                    )
                    target_gate_evidence.append(entry)
                    gate_rows.append(
                        {
                            "slice": target_slice_id,
                            "gate": gate_name,
                            "rc": child.exit_code,
                            "terminal_class": entry.terminal_class,
                            "decision": entry.decision,
                            "run_dir": (
                                str(child.run_dir)
                                if child.run_dir is not None
                                else None
                            ),
                            "error_kind": child.error_kind,
                            "error_message": child.error_message,
                            "accounting_debt": list(debt),
                            "cost_usd_is_lower_bound": ("cost_complete" in debt),
                        }
                    )
                    if not child.succeeded:
                        failed = True
                        gate_failed_by_slice[target_slice_id] = True
                        paid_gate_authority_revoked = (
                            paid_gate_authority_revoked or child.revokes_paid_authority
                        )
                    return entry

                directive_scope = tuple(
                    dict.fromkeys(
                        (
                            *selected_slice.directive.files_in_scope,
                            *selected_slice.directive.creates_paths,
                        )
                    )
                )
                initial_impl, _ = initial_recorded_impls[slice_id]
                run_breach = bool(
                    args.with_breach and corpus.get_artifact(initial_impl.diff_ref)
                )
                breach_test_path = None
                if run_breach:
                    breach_test_path = resolve_breach_test_target(
                        spec=spec,
                        cwd=cwd,
                        diff_bytes=corpus.get_artifact(initial_impl.diff_ref),
                        base_treeish=initial_impl.base_commit,
                    )
                transaction_scope = tuple(dict.fromkeys((
                    *directive_scope,
                    *((breach_test_path,) if breach_test_path else ()),
                )))
                test_is_plan_owned = bool(
                    breach_test_path
                    and any(
                        breach_test_path == path
                        or breach_test_path.startswith(path.rstrip("/") + "/")
                        for path in directive_scope
                    )
                )
                gate_transaction = MutationTransaction.begin(
                    cwd=cwd,
                    journal_root=handle.run_dir / "mutations",
                    slice_id=f"handoff_{slice_id}_gates",
                    scope=transaction_scope,
                    sequence=next_journal_sequence(),
                    spec=spec,
                )

                def restore_gate_transaction(
                    transaction=gate_transaction,
                ) -> None:
                    if transaction.active and not transaction.settle_or_restore(corpus):
                        raise RuntimeError("failed to restore Handoff gate transaction")

                rollback_stack.callback(restore_gate_transaction)
                breach_test_transaction = None
                if breach_test_path is not None and not test_is_plan_owned:
                    breach_test_transaction = MutationTransaction.begin(
                        cwd=cwd,
                        journal_root=handle.run_dir / "mutations",
                        slice_id=f"handoff_{slice_id}_breach_test",
                        scope=(breach_test_path,),
                        sequence=next_journal_sequence(),
                        spec=spec,
                        parent_transaction_id=gate_transaction.transaction_id,
                    )

                    def restore_breach_test_transaction(
                        transaction=breach_test_transaction,
                        parent_paths=directive_scope,
                    ) -> None:
                        if not transaction.active:
                            return
                        # A nested Breach may atomically restore or accept a
                        # parent-owned production path while this narrower
                        # journal owns only the workflow-generated attack.
                        # Transfer those exact live postimages back to the
                        # still-active gate parent before removing the attack;
                        # the outer transaction remains the sole failure-
                        # rollback authority for production bytes.
                        parent_handoff = (
                            transaction.capture_active_parent_postimage_handoff(
                                parent_paths
                            )
                            if parent_paths
                            else None
                        )
                        if not transaction.restore(
                            full=False,
                            active_parent_postimage_handoff=parent_handoff,
                        ):
                            raise RuntimeError(
                                "failed to restore non-plan Handoff breach test"
                            )

                    rollback_stack.callback(restore_breach_test_transaction)
                post_breach_ready = True
                if args.with_breach and not run_breach:
                    # Breaker needs a concrete candidate patch. Passing an
                    # empty diff can make workspace fallback inspect unrelated
                    # ambient dirt, while a clean tree can only yield a
                    # workspace-mismatch result. Tier applies the
                    # same no-seed rule to verification-only execution.
                    gate_rows.append({
                        "slice": slice_id,
                        "gate": "breach",
                        "rc": 0,
                        "status": "skipped_empty_diff",
                    })
                    slice_gate_evidence.append(GateLedgerEntry(
                        gate="breach",
                        candidate_diff_sha256=str(ref),
                        terminal_class="skipped",
                        decision="skipped_empty_diff",
                        exit_code=0,
                        workflow_outcome=None,
                        verification_evidence_ref=None,
                        error_kind=None,
                        error_message=None,
                    ))
                if run_breach:
                    # Breach must receive the exact current slice's approved
                    # execution authority. Binding every sibling in the owner
                    # tier would let one handoff diff be judged against work it
                    # neither contains nor has authority to mutate.
                    authority_bytes, authority_digest = encode_breach_authority(
                        tier_id=row["tier"],
                        issue_text=spec.intent or selected_slice.directive.intent,
                        slice_directives=(selected_slice.directive,),
                        authority_kind="selected_slices",
                    )
                    authority_path = (
                        handle.run_dir / f"breach_authority_{gate_index}.json"
                    )
                    authority_path.write_bytes(authority_bytes)
                    if (
                        breach_authority_digest(authority_path.read_bytes())
                        != authority_digest
                    ):
                        raise RuntimeError(
                            "handoff breach authority changed after encoding"
                        )
                    breach_argv = [
                        "--spec",
                        str(args.spec),
                        "--diff-hash",
                        ref,
                        "--diff-base-treeish",
                        initial_impl.base_commit,
                        "--cwd",
                        cwd,
                        "--authority-file",
                        str(authority_path),
                        "--authority-digest",
                        authority_digest,
                        "--parent-mutation-transaction-id",
                        (
                            breach_test_transaction.transaction_id
                            if breach_test_transaction is not None
                            else gate_transaction.transaction_id
                        ),
                        *forwarded,
                    ]
                    breach_child = _invoke_typed_child_gate(
                        parent_run_dir=handle.run_dir,
                        operation_id=f"handoff_{gate_index}_breach",
                        workflow="breach",
                        invoke=partial(breach_main, breach_argv),
                        on_escaped_terminal=partial(
                            record_paid_child_gate,
                            gate_name="breach",
                            debt_gate_name=f"{slice_id}:breach",
                            candidate_diff_sha256=str(ref),
                            target_slice_id=slice_id,
                            target_gate_evidence=slice_gate_evidence,
                        ),
                    )
                    record_paid_child_gate(
                        breach_child,
                        gate_name="breach",
                        debt_gate_name=f"{slice_id}:breach",
                        candidate_diff_sha256=str(ref),
                        target_slice_id=slice_id,
                        target_gate_evidence=slice_gate_evidence,
                    )
                    if breach_child.succeeded:
                        # A successful Breach may include an accepted Defender
                        # mutation. Rebind every downstream gate and the final
                        # Implementation to the exact resulting bytes instead
                        # of silently reviewing the pre-Breach handoff diff.
                        # A generated attack is workflow evidence rather than
                        # shipping authority unless the plan owns its path.
                        if (
                            breach_test_transaction is not None
                            and breach_test_transaction.active
                        ):
                            restore_breach_test_transaction()
                        if directive_scope:
                            (
                                post_breach_diff,
                                post_breach_files,
                                post_breach_base,
                            ) = capture_workspace_delta(
                                cwd=cwd,
                                path_scope=directive_scope,
                            )
                        else:
                            post_breach_diff = b""
                            post_breach_files = ()
                            post_breach_base = spec.project_context.base_commit

                        post_breach_ref = corpus.put_artifact(
                            post_breach_diff,
                            kind="diffs",
                            name_hint=(f"{slice_id}_post_breach.diff"),
                        )
                        post_breach_outcome: BuildVerifyOutcome | None = None
                        post_breach_integrity_error: str | None = None
                        post_breach_verify = "skipped"
                        if not args.skip_verify:
                            post_breach_log_dir = (
                                handle.run_dir / f"post_breach_verify_{gate_index}"
                            )
                            post_breach_log_dir.mkdir(parents=True, exist_ok=True)
                            (
                                post_breach_outcome,
                                post_breach_integrity_error,
                            ) = _run_readonly_gate(
                                partial(
                                    build_and_verify,
                                    spec,
                                    slice_id=slice_id,
                                    cwd=cwd,
                                    log_dir=post_breach_log_dir,
                                    slice_directive=selected_slice.directive,
                                ),
                                expected_candidate=(
                                    post_breach_diff,
                                    post_breach_files,
                                    post_breach_base,
                                ),
                                cwd=cwd,
                                journal_root=handle.run_dir / "mutations",
                                scope=directive_scope,
                                sequence=next_journal_sequence(),
                                spec=spec,
                                parent_transaction_id=(
                                    gate_transaction.transaction_id
                                ),
                            )
                            if post_breach_outcome is None:
                                post_breach_outcome = BuildVerifyOutcome(
                                    classification=(
                                        "hard_fail_candidate_changed"
                                    ),
                                    build=None,
                                    verify=None,
                                )
                            post_breach_verify = post_breach_outcome.classification
                            if post_breach_integrity_error is not None:
                                post_breach_outcome = BuildVerifyOutcome(
                                    classification="hard_fail_gate_mutation",
                                    build=post_breach_outcome.build,
                                    verify=post_breach_outcome.verify,
                                    dev_verify=post_breach_outcome.dev_verify,
                                )
                                post_breach_verify = post_breach_outcome.classification
                            post_breach_ready = (
                                post_breach_integrity_error is None
                                and (
                                    post_breach_verify == "runner_missing"
                                    or post_breach_verify.startswith("success")
                                )
                            )
                            gate_rows.append(
                                {
                                    "slice": slice_id,
                                    "gate": "post_breach_verify",
                                    "rc": 0 if post_breach_ready else 1,
                                    "classification": post_breach_verify,
                                    "integrity_error": (post_breach_integrity_error),
                                }
                            )
                            if not post_breach_ready:
                                failed = True
                                gate_failed_by_slice[slice_id] = True

                        post_breach_verification_ref = persist_verification_evidence(
                            corpus,
                            tier_id=row["tier"],
                            slice_id=slice_id,
                            spec_execution_authority_hash=(expected_structure_hash),
                            candidate_diff_ref=post_breach_ref,
                            base_commit=post_breach_base,
                            files_modified=post_breach_files,
                            outcome=post_breach_outcome,
                            skipped_by_policy=args.skip_verify,
                            build_timeout_s=DEFAULT_BUILD_TIMEOUT_S,
                            verify_timeout_s=DEFAULT_VERIFY_TIMEOUT_S,
                            name_hint=(f"handoff_{row['tier']}_{slice_id}_post_breach"),
                        )
                        if args.skip_verify:
                            post_verify_terminal = _outcome.TerminalClass.NO_WORK.value
                        elif not post_breach_ready:
                            post_verify_terminal = (
                                _outcome.TerminalClass.PERMANENT_FAILURE.value
                            )
                        elif post_breach_verify == "runner_missing":
                            post_verify_terminal = _outcome.TerminalClass.NO_WORK.value
                        else:
                            post_verify_terminal = (
                                _outcome.TerminalClass.SUCCEEDED.value
                            )
                        slice_gate_evidence.append(
                            GateLedgerEntry(
                                gate="post_breach_verify",
                                candidate_diff_sha256=str(post_breach_ref),
                                terminal_class=post_verify_terminal,
                                decision=post_breach_verify,
                                exit_code=0 if post_breach_ready else 1,
                                workflow_outcome=None,
                                verification_evidence_ref=str(
                                    post_breach_verification_ref
                                ),
                                error_kind=(
                                    "deterministic_gate_mutation"
                                    if post_breach_integrity_error is not None
                                    else None
                                ),
                                error_message=post_breach_integrity_error,
                            )
                        )
                        if args.skip_verify:
                            gate_rows.append(
                                {
                                    "slice": slice_id,
                                    "gate": "post_breach_verify",
                                    "rc": 0,
                                    "classification": post_breach_verify,
                                    "integrity_error": None,
                                    "verification_evidence_ref": str(
                                        post_breach_verification_ref
                                    ),
                                }
                            )
                        else:
                            gate_rows[-1]["verification_evidence_ref"] = str(
                                post_breach_verification_ref
                            )

                        if post_breach_ready:
                            prior_impl, _ = recorded_impls[slice_id]
                            if (
                                post_breach_diff
                                != corpus.get_artifact(prior_impl.diff_ref)
                                or post_breach_files != prior_impl.files_modified
                                or post_breach_base != prior_impl.base_commit
                                or prior_impl.artifacts.get(
                                    HANDOFF_VERIFICATION_ARTIFACT
                                ) != post_breach_verification_ref
                            ):
                                post_breach_note_ref = corpus.put_artifact(
                                    json.dumps(
                                        {
                                            "mode": "handoff_post_breach",
                                            "operator": args.operator,
                                            "note": args.note,
                                            "tier": row["tier"],
                                            "slice": slice_id,
                                            "breach_authority_digest": (
                                                authority_digest
                                            ),
                                            "post_breach_verify": (post_breach_verify),
                                        },
                                        indent=2,
                                    ).encode("utf-8"),
                                    kind="transcripts",
                                    name_hint=(f"handoff_post_breach_{slice_id}.json"),
                                )
                                success_classification = (
                                    "success"
                                    if post_breach_files
                                    else "soft_fail_no_changes"
                                )
                                amended = replace(
                                    prior_impl,
                                    base_commit=post_breach_base,
                                    diff_ref=post_breach_ref,
                                    transcript_ref=post_breach_note_ref,
                                    files_modified=post_breach_files,
                                    classification="pending_gates",
                                    execution=_handoff_execution(cwd),
                                    artifacts={
                                        **prior_impl.artifacts,
                                        HANDOFF_VERIFICATION_ARTIFACT: (
                                            post_breach_verification_ref
                                        ),
                                    },
                                )
                                amended = _partition.attach_resume_identity(
                                    corpus,
                                    spec,
                                    tiers_by_id[row["tier"]],
                                    selected_slice,
                                    amended,
                                )
                                try:
                                    spec, amended_ref = _persist_implementation_guarded(
                                        corpus,
                                        amended,
                                        expected_structure_hash=(
                                            expected_structure_hash
                                        ),
                                    )
                                except _ConcurrentSpecChange as exc:
                                    concurrent_structure_error = str(exc)
                                    failed = True
                                    gate_failed_by_slice[slice_id] = True
                                    post_breach_ready = False
                                    gate_rows.append(
                                        {
                                            "slice": slice_id,
                                            "gate": "execution_structure_cas",
                                            "rc": 1,
                                            "error": concurrent_structure_error,
                                        }
                                    )
                                else:
                                    recorded_impls[slice_id] = (
                                        amended,
                                        success_classification,
                                    )
                                    ref = str(post_breach_ref)
                                    row["diff_ref"] = ref
                                    row["post_breach_implementation_ref"] = str(
                                        amended_ref
                                    )
                                    row["files_modified"] = list(
                                        post_breach_files
                                    )
                                    row["diff_bytes"] = len(post_breach_diff)
                                    row["verify"] = post_breach_verify
                                    row["verification_evidence_ref"] = str(
                                        post_breach_verification_ref
                                    )
                selected_tier = replace(
                    tiers_by_id[row["tier"]],
                    slices=(selected_slice,),
                    depends_on=(),
                )
                if (
                    args.with_review
                    and post_breach_ready
                    and not paid_gate_authority_revoked
                ):
                    if args.review_rubric == "tier_completeness":
                        # Reuse Tier's evidence contract instead of reviewing
                        # a bare diff with none of the rubric's directive or
                        # verification context.
                        from .tier import (  # noqa: PLC0415
                            _render_tier_completeness_artifact,
                        )

                        provisional, success_classification = recorded_impls[slice_id]
                        review_artifact = _render_tier_completeness_artifact(
                            selected_tier,
                            corpus.get_artifact(provisional.diff_ref),
                            verification_evidence=(
                                {
                                    "slice_id": slice_id,
                                    "classification": success_classification,
                                    "build_verify": row.get("verify"),
                                    "diff_ref": str(provisional.diff_ref),
                                    "candidate_diff_sha256": str(provisional.diff_ref),
                                    "verification_evidence_ref": row.get(
                                        "verification_evidence_ref"
                                    ),
                                    "source": "handoff",
                                },
                            ),
                        )
                        review_path = (
                            handle.run_dir / f"tier_completeness_{gate_index}.md"
                        )
                        review_path.write_bytes(review_artifact)
                        review_source_args = ["--artifact", str(review_path)]
                    else:
                        review_source_args = ["--artifact-hash", ref]
                    review_argv = [
                        "--spec",
                        str(args.spec),
                        "--rubric",
                        args.review_rubric,
                        *review_source_args,
                        "--slice-id",
                        slice_id,
                        "--response-retry-budget",
                        str(_HANDOFF_REVIEW_RESPONSE_RETRY_BUDGET),
                        *forwarded,
                    ]
                    review_child = _invoke_typed_child_gate(
                        parent_run_dir=handle.run_dir,
                        operation_id=f"handoff_{gate_index}_review",
                        workflow="review",
                        invoke=partial(review_main, review_argv),
                        on_escaped_terminal=partial(
                            record_paid_child_gate,
                            gate_name=f"review:{args.review_rubric}",
                            debt_gate_name=(f"{slice_id}:review:{args.review_rubric}"),
                            candidate_diff_sha256=str(ref),
                            target_slice_id=slice_id,
                            target_gate_evidence=slice_gate_evidence,
                        ),
                    )
                    record_paid_child_gate(
                        review_child,
                        gate_name=f"review:{args.review_rubric}",
                        debt_gate_name=(f"{slice_id}:review:{args.review_rubric}"),
                        candidate_diff_sha256=str(ref),
                        target_slice_id=slice_id,
                        target_gate_evidence=slice_gate_evidence,
                    )
                elif args.with_review and paid_gate_authority_revoked:
                    skipped_review = GateLedgerEntry(
                        gate=f"review:{args.review_rubric}",
                        candidate_diff_sha256=str(ref),
                        terminal_class="skipped",
                        decision="skipped_incomplete_accounting",
                        exit_code=None,
                        workflow_outcome=None,
                        verification_evidence_ref=None,
                        error_kind="paid_gate_authority_revoked",
                        error_message=(
                            "a prior paid child has incomplete or unknown accounting"
                        ),
                    )
                    slice_gate_evidence.append(skipped_review)
                    gate_rows.append(
                        {
                            "slice": slice_id,
                            "gate": f"review:{args.review_rubric}",
                            "rc": None,
                            "terminal_class": skipped_review.terminal_class,
                            "decision": skipped_review.decision,
                            "run_dir": None,
                            "error_kind": skipped_review.error_kind,
                            "error_message": skipped_review.error_message,
                        }
                    )

                provisional, success_classification = recorded_impls[slice_id]
                gate_failed = gate_failed_by_slice.get(slice_id, False)
                if not gate_failed:
                    integrity_error = _candidate_integrity_error(
                        gate_transaction,
                        corpus=corpus,
                        cwd=cwd,
                        scope=directive_scope,
                        spec=spec,
                        implementation=provisional,
                    )
                    if integrity_error is not None:
                        gate_rows.append({
                            "slice": slice_id,
                            "gate": "candidate_integrity",
                            "rc": 1,
                            "error": integrity_error,
                        })
                        failed = True
                        gate_failed = True
                        gate_failed_by_slice[slice_id] = True
                        slice_gate_evidence.append(GateLedgerEntry(
                            gate="candidate_integrity",
                            candidate_diff_sha256=str(provisional.diff_ref),
                            terminal_class=(
                                _outcome.TerminalClass.PERMANENT_FAILURE.value
                            ),
                            decision="candidate_integrity_failed",
                            exit_code=1,
                            workflow_outcome=None,
                            verification_evidence_ref=None,
                            error_kind="candidate_integrity_error",
                            error_message=integrity_error,
                        ))

                # Failed gates restore the exact operator-authored baseline.
                # The terminal failure record consequently binds that baseline
                # instead of an unlanded Defender or mutating-verifier image.
                if gate_failed:
                    if (
                        breach_test_transaction is not None
                        and breach_test_transaction.active
                    ):
                        restore_breach_test_transaction()
                    if not gate_transaction.restore():
                        raise RuntimeError(
                            "failed to restore rejected Handoff gate bytes"
                        )
                    provisional, success_classification = initial_recorded_impls[
                        slice_id
                    ]
                    recorded_impls[slice_id] = (
                        provisional,
                        success_classification,
                    )
                    baseline_diff = corpus.get_artifact(provisional.diff_ref)
                    row["diff_ref"] = str(provisional.diff_ref)
                    row["files_modified"] = list(provisional.files_modified)
                    row["diff_bytes"] = len(baseline_diff)
                    row["verification_evidence_ref"] = str(
                        provisional.artifacts[HANDOFF_VERIFICATION_ARTIFACT]
                    )

                if concurrent_structure_error is not None:
                    row["classification"] = "hard_fail_concurrent_spec_change"
                    row["action"] = "gate-failed"
                    row["reason"] = concurrent_structure_error
                    handle.write_event(
                        {
                            "slice": slice_id,
                            "classification": ("hard_fail_concurrent_spec_change"),
                            "error": concurrent_structure_error,
                        }
                    )
                    break

                # A provisional record is intentionally not skip-worthy.
                # Stamp the terminal attempt only after this slice's requested
                # gates and byte-integrity checks have all returned.
                slice_gates = [g for g in gate_rows if g["slice"] == slice_id]
                terminal_classification = (
                    "hard_fail_gate" if gate_failed else success_classification
                )
                gate_ledger_ref = persist_handoff_gate_ledger(
                    corpus,
                    tier_id=row["tier"],
                    slice_id=slice_id,
                    spec_execution_authority_hash=expected_structure_hash,
                    terminal_candidate_diff_ref=provisional.diff_ref,
                    entries=tuple(slice_gate_evidence),
                )
                gate_blob = json.dumps(
                    {
                        "mode": "handoff_gate_result",
                        "operator": args.operator,
                        "note": args.note,
                        "slice": slice_id,
                        "gates": slice_gates,
                        "gate_ledger_ref": str(gate_ledger_ref),
                        "classification": terminal_classification,
                    },
                    indent=2,
                ).encode("utf-8")
                terminal_transcript_ref = corpus.put_artifact(
                    gate_blob,
                    kind="transcripts",
                    name_hint=f"handoff_gates_{slice_id}.json",
                )
                terminal = replace(
                    provisional,
                    classification=terminal_classification,
                    transcript_ref=terminal_transcript_ref,
                    execution=_handoff_execution(cwd),
                    artifacts={
                        **provisional.artifacts,
                        HANDOFF_GATE_LEDGER_ARTIFACT: gate_ledger_ref,
                    },
                )
                if not gate_failed:
                    gate_transaction.prepare(terminal, keep=True)
                try:
                    spec, terminal_ref = _persist_implementation_guarded(
                        corpus,
                        terminal,
                        expected_structure_hash=expected_structure_hash,
                    )
                except _ConcurrentSpecChange as exc:
                    concurrent_structure_error = str(exc)
                    failed = True
                    if (
                        breach_test_transaction is not None
                        and breach_test_transaction.active
                    ):
                        restore_breach_test_transaction()
                    if gate_transaction.active and not gate_transaction.restore():
                        raise RuntimeError(
                            "failed to restore Handoff gate bytes after "
                            "concurrent Spec change"
                        ) from exc
                    gate_rows.append(
                        {
                            "slice": slice_id,
                            "gate": "execution_structure_cas",
                            "rc": 1,
                            "error": concurrent_structure_error,
                        }
                    )
                    row["classification"] = "hard_fail_concurrent_spec_change"
                    row["action"] = "gate-failed"
                    row["reason"] = concurrent_structure_error
                    handle.write_event(
                        {
                            "slice": slice_id,
                            "classification": ("hard_fail_concurrent_spec_change"),
                            "error": concurrent_structure_error,
                        }
                    )
                    break
                if not gate_failed:
                    gate_transaction.commit(attached=True)
                row["classification"] = terminal_classification
                row["terminal_implementation_ref"] = str(terminal_ref)
                row["gate_ledger_ref"] = str(gate_ledger_ref)
                row["action"] = "gate-failed" if gate_failed else "recorded"
                handle.write_event({
                    "slice": slice_id,
                    "classification": terminal_classification,
                    "gates": slice_gates,
                })
                if paid_gate_authority_revoked:
                    break

        candidate_landing_blockers: list[str] = []
        candidate_capture_errors: dict[str, str] = {}
        if args.land and not failed and not args.dry_run:
            landing_candidates = {
                **skipped_impls_for_land,
                **{
                    slice_id: implementation
                    for slice_id, (implementation, _) in recorded_impls.items()
                },
            }
            for slice_id, implementation in landing_candidates.items():
                directive = slices_by_id[slice_id].directive
                scope = tuple(
                    (*directive.files_in_scope, *directive.creates_paths)
                )
                try:
                    observed = (
                        capture_workspace_delta(cwd=cwd, path_scope=scope)
                        if scope
                        else (b"", (), spec.project_context.base_commit)
                    )
                except GitCaptureError as exc:
                    # A live candidate Git cannot capture cannot prove it
                    # still is the recorded one, so it blocks landing. Nothing
                    # was compared: the evidence keeps the capture error as
                    # the cause instead of reporting a mismatch.
                    logger.warning(
                        "handoff %s: live candidate could not be re-captured "
                        "before landing: %s",
                        slice_id, exc,
                    )
                    candidate_capture_errors[slice_id] = str(exc)
                    candidate_landing_blockers.append(slice_id)
                    continue
                try:
                    expected = (
                        corpus.get_artifact(implementation.diff_ref),
                        implementation.files_modified,
                        implementation.base_commit,
                    )
                except (KeyError, OSError, TypeError, ValueError):
                    candidate_landing_blockers.append(slice_id)
                    continue
                if observed != expected:
                    candidate_landing_blockers.append(slice_id)
            if candidate_landing_blockers:
                failed = True
                for row in rows:
                    if row.get("slice") in candidate_landing_blockers:
                        capture_error = candidate_capture_errors.get(
                            row["slice"]
                        )
                        row["action"] = "candidate-mismatch"
                        row["reason"] = (
                            "live workspace no longer matches the terminal candidate"
                            if capture_error is None
                            else (
                                "live candidate could not be re-captured "
                                f"before landing: {capture_error}"
                            )
                        )
                handle.write_event(
                    {
                        "phase": "land",
                        "classification": "candidate_mismatch",
                        "slices": candidate_landing_blockers,
                        **(
                            {"capture_errors": candidate_capture_errors}
                            if candidate_capture_errors
                            else {}
                        ),
                    }
                )

        verification_landing_blockers: list[str] = []
        if args.land and not args.skip_verify and not failed and not args.dry_run:
            for slice_id, (implementation, _) in recorded_impls.items():
                directive = slices_by_id[slice_id].directive
                if not directive.files_in_scope and not directive.creates_paths:
                    verification_landing_blockers.append(slice_id)
                    continue
                if conditional_mode:
                    latest_for_land = corpus.latest_spec()
                    if latest_for_land is None or slice_id not in (
                        _partition.partition_for_required_verification(
                            corpus,
                            tiers_by_id[tier_id_by_slice[slice_id]],
                            spec=latest_for_land,
                        ).verified_retained
                    ):
                        verification_landing_blockers.append(slice_id)
                    continue
                if not _partition._verification_matches_handoff_authority(  # noqa: SLF001
                    corpus,
                    implementation,
                    spec=spec,
                    tier_id=tier_id_by_slice[slice_id],
                    slice_id=slice_id,
                    spec_execution_authority_hash=expected_structure_hash,
                    require_gate=True,
                ):
                    verification_landing_blockers.append(slice_id)
            if verification_landing_blockers:
                failed = True
                for row in rows:
                    if row.get("slice") in verification_landing_blockers:
                        row["action"] = "verification-pending"
                        row["reason"] = (
                            "required build/verify evidence is unavailable or "
                            "does not bind the terminal candidate"
                        )
                handle.write_event(
                    {
                        "phase": "land",
                        "classification": "verification_pending",
                        "slices": verification_landing_blockers,
                    }
                )

        landed = False
        if args.land and not failed and not args.dry_run:
            try:
                committed = corpus.commit(
                    lifecycle_event_factory=lambda committed_version: LifecycleEvent(
                        kind="landed",
                        reason=(
                            "hand-completed via handoff by "
                            f"{args.operator}" + (f": {args.note}" if args.note else "")
                        ),
                        affected_versions=(committed_version,),
                        execution=_handoff_execution(cwd),
                    ),
                    transform=_execution_structure_guard(
                        expected_structure_hash,
                        (
                            partial(
                                require_conditional_outputs_absent,
                                tuple(pairs[0][1].directive.creates_paths),
                            )
                            if conditional_mode
                            else None
                        ),
                    ),
                )
            except _ConcurrentSpecChange as exc:
                concurrent_structure_error = str(exc)
                failed = True
            else:
                spec = committed.spec
                landed = True

        status = (
            "incomplete_accounting"
            if paid_gate_authority_revoked
            else "failed"
            if failed
            else "dry-run"
            if args.dry_run
            else "complete"
        )
        final_spec = corpus.latest_spec()
        summary = {
            "status": status,
            "cost_usd": paid_gate_cost_usd,
            **_paid_gate_accounting_fields(paid_gate_accounting_debt_by_gate),
            "spec_version": final_spec.version if final_spec is not None else None,
            "targets": rows,
            "gates": gate_rows,
            "out_of_scope_changes": out_of_scope,
            "landed": landed,
            "candidate_landing_blockers": candidate_landing_blockers,
            "verification_landing_blockers": verification_landing_blockers,
            "concurrent_structure_error": concurrent_structure_error,
        }
        handle.write_summary(summary)
        # CLI-result emit (same carve-out as compare/swebench --list).
        print(json.dumps(summary, indent=2))
        handle.finalize_elapsed(RunResult(
            status=status,
            cost_usd=paid_gate_cost_usd,
        ))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
