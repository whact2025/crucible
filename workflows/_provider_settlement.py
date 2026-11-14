# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Order provider teardown before mutation-journal settlement.

Write-capable backend adapters may own dispatch workers, subprocesses, or
other descendants that outlive one synchronous model call.  A mutation
journal must therefore remain armed until every provider owner has closed.
This helper couples those two cleanup families and retains retry authority
when either family cannot settle on the first attempt.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from contextlib import AbstractContextManager, ExitStack
from functools import partial
from threading import Lock
from types import TracebackType
from typing import Any, Callable, NoReturn, TypeVar

from crucible.backends.capabilities import (
    CapabilityProfile,
    normalize_capability,
    toolset_write_names,
    validate_toolset_capability,
)
from crucible.tools._dispatch_budget import ToolMutationIndeterminate

from ._backends import BackendSettlementInterrupted

from ._writer_epoch import (
    WriterEpochArmError,
    WriterEpochBlockedError,
    WriterEpochBoundary,
    WriterEpochLease,
    WriterEpochQuiescenceError,
    WriterEpochStorage,
    WriterSettlementOutcome,
    capture_writer_epoch_owner,
    prepare_writer_epoch_arm,
    provider_settlement_receipt,
    quiesce_writer_epoch_family,
    require_writer_epoch_quiescence,
    retry_writer_epoch_arm,
)


_T = TypeVar("_T")
_MAX_SETTLEMENT_INTERRUPTS = 2
_CLEANUP_ERROR_OWNER_TOKEN_ATTR = (
    "_crucible_provider_settlement_owner_token"
)


class _ReadOnlyProviderView:
    """Capability-ceiling proxy for an intentionally unarmed pool."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def get(
        self,
        role: str,
        *,
        toolset: Any | None = None,
        capability: str | None = None,
        **kwargs: Any,
    ) -> Any:
        resolved_capability: CapabilityProfile
        if capability is None:
            if toolset is None:
                resolved_capability = "text"
            else:
                writes = toolset_write_names(toolset)
                if writes is None:
                    raise ValueError(
                        "read-only provider toolset capability is unclassified"
                    )
                resolved_capability = "workspace_write" if writes else "read_only"
        else:
            resolved_capability = normalize_capability(capability)
        validate_toolset_capability(
            resolved_capability,
            toolset,
        )
        if resolved_capability == "workspace_write":
            raise RuntimeError(
                "read-only provider generation cannot resolve a "
                "workspace-write backend; close it and open an armed "
                "generation"
            )
        return self._provider.get(
            role,
            toolset=toolset,
            capability=resolved_capability,
            **kwargs,
        )

    def new_exclusive_writer(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "read-only provider generation cannot create an exclusive writer; "
            "close it and open an armed generation"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


class ProviderSettlementCleanupError(RuntimeError):
    """Live retry owner for provider-first mutation cleanup.

    ``retry_cleanup`` always retries provider teardown first.  Mutation
    callbacks remain retained, in LIFO order, until provider teardown has
    positively completed.  A persistent provider close failure therefore
    cannot race a journal restore.
    """

    def __init__(
        self,
        owner: "ProviderSettlementExitStack",
        *,
        primary_error: BaseException,
        cleanup_error: BaseException,
    ) -> None:
        self.owner = owner
        self.primary_error = primary_error
        self.cleanup_error = cleanup_error
        super().__init__(
            "provider-first mutation cleanup remains pending: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )

    def retry_cleanup(self) -> None:
        """Retry the exact retained provider and mutation cleanup owners."""

        try:
            self.owner.retry_cleanup()
        except KeyboardInterrupt as exc:
            # The owner only replays a deferred interrupt after every retained
            # provider, writer epoch, and mutation callback has settled. Do
            # not turn that successful cancellation boundary back into a live
            # cleanup error.
            if self.owner.cleanup_complete:
                raise
            self.cleanup_error = exc
            raise self from exc
        except BaseException as exc:  # noqa: BLE001 - preserve live owner
            self.cleanup_error = exc
            raise self from exc


class _UnprovenProviderSettlementInterrupt(RuntimeError):
    """A provider context raised SIGINT without positive close evidence."""


class ProviderSettlementExitStack:
    """A narrow ExitStack whose providers always close before callbacks.

    Provider contexts are registered with :meth:`enter_provider_context`.
    Mutation settlement callbacks use the familiar :meth:`callback` surface.
    Unlike a plain ``ExitStack``, a provider close failure does not continue
    into callbacks: the provider error and exact callback list are retained in
    :class:`ProviderSettlementCleanupError` for an ordered retry.
    """

    def __init__(self) -> None:
        self._providers = ExitStack()
        self._callbacks: list[
            tuple[object, Callable[..., Any], tuple[Any, ...], dict[str, Any]]
        ] = []
        self._provider_cleanup_error: BaseException | None = None
        self._provider_context_active = False
        self._provider_context_enter_failed = False
        self._provider_close_completed = False
        self._provider_context_owner: object | None = None
        self._provider_epoch_lease: WriterEpochLease | None = None
        self._provider_epoch_arm_error: WriterEpochArmError | None = None
        self._provider_owner_ids: tuple[str, ...] = ()
        self._provider_worker_ids: tuple[str, ...] = ()
        self._provider_evidence_status: str | None = None
        self._provider_epoch_outcome: WriterSettlementOutcome = "succeeded"
        self._deferred_interrupt: KeyboardInterrupt | None = None
        self._terminal_precedence: ToolMutationIndeterminate | None = None
        # Every typed cleanup error created by this stack carries this private
        # identity token.  Callers can recognize their own retained authority
        # without invoking an overridable ``ProviderSettlementCleanupError.owner``
        # getter while asynchronous cancellation is already in flight.
        self._cleanup_error_owner_token = object()
        self._entered = False
        self._complete = False
        self._lock = Lock()

    @property
    def cleanup_complete(self) -> bool:
        with self._lock:
            return self._complete

    @property
    def settlement_pending(self) -> bool:
        with self._lock:
            return bool(
                self._callbacks
                or self._provider_context_active
                or self._provider_cleanup_error is not None
                or self._provider_epoch_lease is not None
                or self._provider_epoch_arm_error is not None
            )

    def _new_cleanup_error(
        self,
        *,
        primary_error: BaseException,
        cleanup_error: BaseException,
    ) -> ProviderSettlementCleanupError:
        """Create one cleanup error bound to this stack by object identity."""

        retained = ProviderSettlementCleanupError(
            self,
            primary_error=primary_error,
            cleanup_error=cleanup_error,
        )
        setattr(
            retained,
            _CLEANUP_ERROR_OWNER_TOKEN_ATTR,
            self._cleanup_error_owner_token,
        )
        return retained

    def owns_cleanup_error(
        self,
        error: ProviderSettlementCleanupError,
    ) -> bool:
        """Recognize this stack's error without consulting public owner hooks."""

        return bool(
            vars(error).get(_CLEANUP_ERROR_OWNER_TOKEN_ATTR)
            is self._cleanup_error_owner_token
        )

    def retained_cleanup_error(
        self,
        *,
        primary_error: BaseException,
        cleanup_error: BaseException,
    ) -> ProviderSettlementCleanupError:
        """Publish this prebound stack as a higher-level cleanup owner.

        Workflow composition uses this only to place an already-live child
        cleanup dependency ahead of this stack.  It does not run or otherwise
        settle either owner.
        """

        return self._new_cleanup_error(
            primary_error=primary_error,
            cleanup_error=cleanup_error,
        )

    def __enter__(self) -> "ProviderSettlementExitStack":
        with self._lock:
            if self._entered:
                raise RuntimeError("provider settlement stack is not reusable")
            self._entered = True
        self._providers.__enter__()
        return self

    def _call_deferring_interrupts(
        self,
        callback: Callable[[], _T],
    ) -> _T:
        """Retry one idempotent settlement step through main-thread SIGINT."""

        interrupt_count = 0
        while True:
            try:
                return callback()
            except BackendSettlementInterrupted as exc:
                self._accept_backend_settlement_interrupt(exc)
                interrupt_count += 1
                if interrupt_count >= _MAX_SETTLEMENT_INTERRUPTS:
                    raise WriterEpochBlockedError(
                        "repeated SIGINT prevented provider-first settlement; "
                        "exact cleanup authority is retained"
                    ) from exc
                continue
            except KeyboardInterrupt as exc:
                self._defer_interrupt(exc)
                interrupt_count += 1
                if interrupt_count >= _MAX_SETTLEMENT_INTERRUPTS:
                    raise WriterEpochBlockedError(
                        "repeated SIGINT prevented provider-first settlement; "
                        "exact cleanup authority is retained"
                    ) from exc
                continue
            except BaseException as exc:
                if self._defer_wrapped_settlement_interrupt(exc):
                    interrupt_count += 1
                    if interrupt_count >= _MAX_SETTLEMENT_INTERRUPTS:
                        raise WriterEpochBlockedError(
                            "repeated SIGINT prevented provider-first "
                            "settlement; exact cleanup authority is retained"
                        ) from exc
                    continue
                with self._lock:
                    interrupted = self._deferred_interrupt is not None
                if interrupted:
                    exc.add_note(
                        "SIGINT was deferred while provider-first settlement "
                        "remained unproven"
                    )
                raise

    def _defer_wrapped_settlement_interrupt(
        self,
        error: BaseException,
        *,
        validate_provider_owner: bool = True,
    ) -> bool:
        """Retain cancellation wrapped by an exact writer-epoch retry owner."""

        if not isinstance(
            error,
            (WriterEpochArmError, WriterEpochQuiescenceError),
        ):
            return False
        nested = error.error
        if isinstance(nested, BackendSettlementInterrupted):
            if validate_provider_owner:
                self._accept_backend_settlement_interrupt(nested)
            else:
                self._defer_interrupt(nested.interrupt)
            return True
        if isinstance(nested, KeyboardInterrupt):
            self._defer_interrupt(nested)
            return True
        return False

    def _defer_interrupt(self, interrupt: KeyboardInterrupt) -> None:
        """Retain the first cancellation for replay after full settlement."""

        with self._lock:
            self._deferred_interrupt = self._deferred_interrupt or interrupt

    def _accept_backend_settlement_interrupt(
        self,
        interrupted: BackendSettlementInterrupted,
    ) -> None:
        """Accept only a cancellation bound to the entered provider owner."""

        with self._lock:
            expected_owner = self._provider_context_owner
        if interrupted.owner is not expected_owner or expected_owner is None:
            raise _UnprovenProviderSettlementInterrupt(
                "backend settlement interrupt is not bound to the entered "
                "provider owner"
            ) from interrupted
        self._defer_interrupt(interrupted.interrupt)

    def _replay_deferred_interrupt(
        self,
        *,
        primary_error: BaseException | None = None,
    ) -> None:
        """Replay cancellation only beyond the durable settlement boundary."""

        with self._lock:
            interrupt = self._deferred_interrupt
            self._deferred_interrupt = None
        if interrupt is None:
            return
        if primary_error is not None:
            primary_error.add_note(
                "SIGINT was deferred until provider-first settlement completed"
            )
            return
        raise interrupt

    def enter_provider_context(
        self,
        context: AbstractContextManager[_T],
    ) -> _T:
        """Acquire one provider owner whose exit precedes all callbacks."""

        with self._lock:
            if not self._entered or self._complete:
                raise RuntimeError("provider settlement stack is not active")
            if self._provider_cleanup_error is not None:
                raise RuntimeError("provider cleanup retry is pending")
            if self._provider_context_active:
                raise RuntimeError(
                    "one provider generation is already active"
                )
            if self._provider_epoch_arm_error is not None:
                raise RuntimeError("provider epoch arm retry is pending")
            if self._provider_epoch_lease is None:
                raise RuntimeError(
                    "writer-capable provider context requires an exact armed "
                    "writer epoch"
                )
        return self._enter_provider_context(context)

    def enter_read_only_provider_context(
        self,
        context: AbstractContextManager[_T],
    ) -> _ReadOnlyProviderView:
        """Acquire an explicitly unarmed provider under a read-only ceiling."""

        with self._lock:
            if not self._entered or self._complete:
                raise RuntimeError("provider settlement stack is not active")
            if self._provider_cleanup_error is not None:
                raise RuntimeError("provider cleanup retry is pending")
            if self._provider_context_active:
                raise RuntimeError(
                    "one provider generation is already active"
                )
            if self._provider_epoch_arm_error is not None:
                raise RuntimeError("provider epoch arm retry is pending")
            if self._provider_epoch_lease is not None:
                raise RuntimeError(
                    "armed provider generations must use enter_provider_context"
                )
        entered = self._enter_provider_context(context)
        return _ReadOnlyProviderView(entered)

    def _enter_provider_context(
        self,
        context: AbstractContextManager[_T],
    ) -> _T:
        """Enter one already-admitted writer or read-only provider context."""

        try:
            entered = self._providers.enter_context(context)
        except BaseException as cleanup_error:
            with self._lock:
                self._provider_context_enter_failed = True
                self._provider_cleanup_error = cleanup_error
            raise self._new_cleanup_error(
                primary_error=cleanup_error,
                cleanup_error=cleanup_error,
            ) from cleanup_error
        with self._lock:
            self._provider_context_active = True
            self._provider_context_enter_failed = False
            self._provider_close_completed = False
            self._provider_context_owner = entered
            self._provider_evidence_status = None
            self._provider_epoch_outcome = "succeeded"
        return entered

    def arm_provider_generation(
        self,
        members: tuple[WriterEpochStorage, ...],
    ) -> WriterEpochLease:
        """Durably arm one exact journal family before provider creation."""

        with self._lock:
            if not self._entered or self._complete:
                raise RuntimeError("provider settlement stack is not active")
            if (
                self._provider_context_active
                or self._provider_epoch_lease is not None
                or self._provider_cleanup_error is not None
            ):
                raise RuntimeError(
                    "prior provider generation has not positively settled"
                )
        generation_id = f"provider-{secrets.token_hex(16)}"
        owner_ids = (f"{generation_id}:backend-pool",)
        worker_ids = (f"{generation_id}:provider-descendants",)
        lease = prepare_writer_epoch_arm(
            members,
            owner=capture_writer_epoch_owner(
                "provider_generation",
                generation_id,
            ),
            boundary=WriterEpochBoundary(
                "provider_generation",
                generation_id,
                "provider_close_drain",
                owner_ids,
                worker_ids,
            ),
        )
        provisional_error = WriterEpochArmError(
            lease,
            WriterEpochBlockedError(
                "provider writer-epoch arm completion is not yet proved"
            ),
        )
        try:
            # Publish complete retry authority before the first durable write.
            # Publication shares this catch/defer transaction with the arm so
            # cancellation cannot land between those lifecycle boundaries.
            self._publish_prepared_provider_epoch(
                lease,
                provisional_error,
                owner_ids=owner_ids,
                worker_ids=worker_ids,
            )
            retry_writer_epoch_arm(lease)
            with self._lock:
                self._provider_epoch_arm_error = None
            # Keep return inside the protected region: cancellation after the
            # durable arm but before delivery restores the exact retry owner.
            return lease
        except BaseException as caught:
            error = (
                caught
                if isinstance(caught, WriterEpochArmError)
                else WriterEpochArmError(lease, caught)
            )
            with self._lock:
                self._provider_owner_ids = owner_ids
                self._provider_worker_ids = worker_ids
                self._provider_close_completed = False
                self._provider_evidence_status = None
                self._provider_epoch_outcome = "succeeded"
                self._provider_epoch_lease = lease
                self._provider_epoch_arm_error = error
            # The writer-epoch layer intentionally wraps storage failures with
            # its exact retry lease. Preserve a wrapped cancellation so partial
            # arm cleanup can replay it after proving no provider was dispatched.
            self._defer_wrapped_settlement_interrupt(
                error,
                validate_provider_owner=False,
            )
            raise self._new_cleanup_error(
                primary_error=error,
                cleanup_error=error,
            ) from error

    def _publish_prepared_provider_epoch(
        self,
        lease: WriterEpochLease,
        arm_error: WriterEpochArmError,
        *,
        owner_ids: tuple[str, ...],
        worker_ids: tuple[str, ...],
    ) -> None:
        """Publish exact retry authority before the first durable arm write."""

        with self._lock:
            self._provider_owner_ids = owner_ids
            self._provider_worker_ids = worker_ids
            self._provider_close_completed = False
            self._provider_evidence_status = None
            self._provider_epoch_outcome = "succeeded"
            # Publish the self-contained error first: if SIGINT lands before
            # the redundant lease field, retry still recovers ``error.lease``.
            self._provider_epoch_arm_error = arm_error
            self._provider_epoch_lease = lease

    def _settle_provider_epoch(self) -> None:
        with self._lock:
            lease = self._provider_epoch_lease
            owner_ids = self._provider_owner_ids
            worker_ids = self._provider_worker_ids
            evidence_status = self._provider_evidence_status
            outcome = self._provider_epoch_outcome
        if lease is None:
            with self._lock:
                if self._provider_close_completed:
                    self._provider_context_owner = None
            return
        if evidence_status is None:
            raise WriterEpochBlockedError(
                "provider epoch has no positive close/drain evidence"
            )
        evidence = json.dumps(
            {
                "epoch_id": lease.epoch_id,
                "generation": lease.generation,
                "provider_context_settlement": evidence_status,
                "schema_version": "crucible.provider-settlement.v1",
                "settled_owner_ids": list(owner_ids),
                "joined_worker_ids": list(worker_ids),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        receipt = provider_settlement_receipt(
            lease,
            settled_owner_ids=owner_ids,
            joined_worker_ids=worker_ids,
            evidence_digest=(
                "sha256:" + hashlib.sha256(evidence).hexdigest()
            ),
            outcome=outcome,
        )
        quiesce_writer_epoch_family(lease, receipt)
        require_writer_epoch_quiescence(
            lease.members,
            allow_never_armed=False,
        )
        with self._lock:
            if self._provider_epoch_lease is lease:
                self._provider_epoch_lease = None
                self._provider_owner_ids = ()
                self._provider_worker_ids = ()
                self._provider_evidence_status = None
                self._provider_epoch_outcome = "succeeded"
                self._provider_context_owner = None

    def _retry_partial_arm(self) -> None:
        """Finish a partial arm, then prove that dispatch never began."""

        with self._lock:
            arm_error = self._provider_epoch_arm_error
        if arm_error is not None:
            try:
                lease = retry_writer_epoch_arm(arm_error.lease)
            except WriterEpochArmError as retry_error:
                with self._lock:
                    self._provider_epoch_lease = retry_error.lease
                    self._provider_epoch_arm_error = retry_error
                raise
            with self._lock:
                self._provider_epoch_lease = lease
                self._provider_epoch_arm_error = None
                self._provider_context_active = False
                self._provider_close_completed = True
                self._provider_evidence_status = "provider_not_dispatched"
                self._provider_epoch_outcome = "failed"
        # If cancellation interrupted the quiescence step after the arm retry,
        # a repeated call observes no arm error but must still settle the exact
        # retained lease before any callbacks or cancellation replay.
        self._settle_provider_epoch()

    def callback(
        self,
        callback: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Register one idempotent settlement callback in LIFO order."""

        self._register_callback(callback, args, kwargs)
        return callback

    def retain_terminal_precedence(
        self,
        error: ToolMutationIndeterminate,
    ) -> ToolMutationIndeterminate:
        """Retain one exact TMI for cleanup-failure precedence at exit.

        A workflow can finish its ordinary typed terminal bookkeeping and
        return normally when settlement succeeds.  If provider close or a
        retained mutation callback fails during that return, the same terminal
        object remains outermost and the cleanup owner becomes its direct
        cause.  Explicit exceptions leaving the body still take precedence
        over this latent fallback.
        """

        if not isinstance(error, ToolMutationIndeterminate):
            raise TypeError(
                "terminal precedence requires ToolMutationIndeterminate"
            )
        with self._lock:
            if not self._entered or self._complete:
                raise RuntimeError("provider settlement stack is not active")
            if self._terminal_precedence is not None:
                raise RuntimeError(
                    "provider settlement terminal precedence is already retained"
                )
            self._terminal_precedence = error
        return error

    def _register_callback(
        self,
        callback: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> object:
        token = object()
        with self._lock:
            if not self._entered or self._complete:
                raise RuntimeError("provider settlement stack is not active")
            self._callbacks.append((token, callback, args, kwargs))
        return token

    def _run_registered_callback(self, token: object) -> Any:
        with self._lock:
            if not self._callbacks or self._callbacks[-1][0] is not token:
                raise RuntimeError("settlement callback ordering changed")
            _, callback, args, kwargs = self._callbacks[-1]
        result = callback(*args, **kwargs)
        with self._lock:
            if not self._callbacks or self._callbacks[-1][0] is not token:
                raise RuntimeError("settlement callback ordering changed")
            self._callbacks.pop()
        return result

    def _close_providers(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
        *,
        retry_pending: bool = False,
    ) -> bool:
        return self._call_deferring_interrupts(
            lambda: self._close_providers_once(
                exc_type,
                exc,
                traceback,
                retry_pending=retry_pending,
            )
        )

    def _close_providers_once(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
        *,
        retry_pending: bool = False,
    ) -> bool:
        """Advance one retryable provider/epoch settlement transaction."""

        with self._lock:
            pending_error = self._provider_cleanup_error
            context_active = self._provider_context_active
            close_completed = self._provider_close_completed
            epoch_lease = self._provider_epoch_lease
            arm_error = self._provider_epoch_arm_error
        if arm_error is not None:
            if not retry_pending:
                raise arm_error
            self._call_deferring_interrupts(self._retry_partial_arm)
            return False
        if pending_error is not None:
            if not retry_pending:
                raise pending_error
            if close_completed and epoch_lease is None:
                # The provider and exact writer lease already crossed their
                # durable boundary before cancellation landed. Clear only the
                # incidental bounded-retry error; deferred SIGINT is replayed
                # after retained callbacks finish below.
                with self._lock:
                    self._provider_cleanup_error = None
                return False
            if close_completed and epoch_lease is not None:
                # Provider close evidence is already retained; the failure
                # occurred while publishing/quiescing the writer receipt.
                # Retry that exact lease transition instead of requiring the
                # incidental exception type to carry its own retry callback.
                try:
                    self._call_deferring_interrupts(
                        self._settle_provider_epoch
                    )
                except BaseException as cleanup_error:
                    with self._lock:
                        self._provider_cleanup_error = cleanup_error
                    raise
                with self._lock:
                    self._provider_cleanup_error = None
                return False
            retry = getattr(pending_error, "retry_cleanup", None)
            if retry is None:
                raise pending_error
            self._call_deferring_interrupts(retry)
            with self._lock:
                self._provider_cleanup_error = None
                close_completed = self._provider_close_completed
            if not close_completed:
                with self._lock:
                    self._provider_context_active = False
                    self._provider_close_completed = True
                    self._provider_evidence_status = (
                        "provider_enter_failure_cleanup_succeeded"
                        if self._provider_context_enter_failed
                        else "provider_close_retry_succeeded"
                    )
                    self._provider_epoch_outcome = (
                        "failed"
                        if self._provider_context_enter_failed
                        else "succeeded"
                    )
                    self._provider_context_enter_failed = False
            self._call_deferring_interrupts(self._settle_provider_epoch)
            return False
        if not context_active:
            if epoch_lease is not None:
                if close_completed:
                    self._call_deferring_interrupts(
                        self._settle_provider_epoch
                    )
                    return False
                raise WriterEpochBlockedError(
                    "provider epoch was armed but provider settlement did not "
                    "complete"
                )
            return False
        try:
            suppressed = self._providers.__exit__(exc_type, exc, traceback)
        except BackendSettlementInterrupted as interrupted:
            # This typed KeyboardInterrupt is emitted only after the backend
            # pool's exact cleanup owners all returned success. ExitStack has
            # consumed that provider callback, so it is positive close evidence.
            try:
                self._accept_backend_settlement_interrupt(interrupted)
            except BaseException as cleanup_error:
                with self._lock:
                    self._provider_cleanup_error = cleanup_error
                raise
            suppressed = False
        except KeyboardInterrupt as interrupted:
            unproven_error = _UnprovenProviderSettlementInterrupt(
                "provider context raised SIGINT without typed positive close "
                "evidence"
            )
            with self._lock:
                self._provider_cleanup_error = unproven_error
            raise unproven_error from interrupted
        except BaseException as cleanup_error:  # noqa: BLE001 - retained owner
            with self._lock:
                self._provider_cleanup_error = cleanup_error
            raise
        with self._lock:
            self._provider_context_active = False
            self._provider_close_completed = True
            self._provider_evidence_status = "provider_close_succeeded"
            self._provider_epoch_outcome = "succeeded"
        try:
            self._call_deferring_interrupts(self._settle_provider_epoch)
        except BaseException as cleanup_error:  # noqa: BLE001 - retained owner
            with self._lock:
                self._provider_cleanup_error = cleanup_error
            raise
        return bool(suppressed)

    def quiesce_providers(self) -> None:
        """Close providers now, retaining callbacks if close is incomplete.

        Workflows call this at their last paid/provider-backed frontier before
        an in-body restore or commit.  A failure is deliberately not retried by
        ``__exit__``; the raised typed owner is the sole explicit retry path.
        """

        try:
            self._close_providers()
        except BaseException as cleanup_error:  # noqa: BLE001 - typed owner
            raise self._new_cleanup_error(
                primary_error=cleanup_error,
                cleanup_error=cleanup_error,
            ) from cleanup_error
        self._replay_deferred_interrupt()

    def run_after_provider_quiescence(
        self,
        callback: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Close providers, then run one retained mutation action exactly.

        The action is registered before close begins.  If close fails, the
        typed retry owner therefore retains this exact restore/commit action
        ahead of the stack's broader unwind callbacks.  If the action itself
        fails, it likewise remains the next explicit retry target.
        """

        token = self._register_callback(callback, args, kwargs)
        try:
            self._close_providers()
            result = self._call_deferring_interrupts(
                lambda: self._run_registered_callback(token)
            )
        except BaseException as cleanup_error:  # noqa: BLE001 - typed owner
            with self._lock:
                retained_terminal = self._terminal_precedence
            if retained_terminal is not None:
                self._raise_retained_cleanup_owner(
                    primary_error=retained_terminal,
                    cleanup_error=cleanup_error,
                )
            if isinstance(cleanup_error, ProviderSettlementCleanupError):
                raise
            self._raise_retained_cleanup_owner(
                primary_error=None,
                cleanup_error=cleanup_error,
            )
        self._replay_deferred_interrupt()
        return result

    def _run_settlement_callbacks(self) -> None:
        while True:
            with self._lock:
                if not self._callbacks:
                    self._complete = True
                    return
                token = self._callbacks[-1][0]
            # Pop only after success so retry authority remains exact.
            self._call_deferring_interrupts(
                partial(self._run_registered_callback, token)
            )

    def _owned_cleanup_error_in_cause_chain(
        self,
        error: BaseException | None,
    ) -> ProviderSettlementCleanupError | None:
        """Find this stack's explicit retained owner without implicit retry."""

        current = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if (
                isinstance(current, ProviderSettlementCleanupError)
                and self.owns_cleanup_error(current)
            ):
                return current
            cause = current.__cause__
            current = cause if isinstance(cause, BaseException) else None
        return None

    def _raise_retained_cleanup_owner(
        self,
        *,
        primary_error: BaseException | None,
        cleanup_error: BaseException,
    ) -> NoReturn:
        """Publish cleanup authority without masking an exact TMI terminal."""

        retained_primary = (
            primary_error if primary_error is not None else cleanup_error
        )
        cleanup_owner = self._new_cleanup_error(
            primary_error=retained_primary,
            cleanup_error=cleanup_error,
        )
        if isinstance(retained_primary, ToolMutationIndeterminate):
            # Tool mutation uncertainty is the terminal business outcome.
            # Cleanup failure is still the direct, retryable cause, and the
            # underlying close/rollback error remains discoverable beneath it.
            cleanup_owner.__cause__ = cleanup_error
            cleanup_owner.__suppress_context__ = True
            raise retained_primary from cleanup_owner
        raise cleanup_owner from cleanup_error

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        with self._lock:
            retained_terminal = self._terminal_precedence
        # The latch is a cleanup-failure fallback, not a synthetic body
        # exception. An exception explicitly leaving the body retains the
        # existing precedence contract.
        cleanup_primary_error = (
            exc if exc is not None else retained_terminal
        )
        if self._owned_cleanup_error_in_cause_chain(exc) is not None:
            # A direct quiescence/settlement attempt already published this
            # exact retry owner. It may be the primary exception or the
            # explicit cause of a higher-priority mutation terminal. Do not
            # turn lexical unwind into an implicit retry or run broader
            # mutation callbacks in either shape.
            return False
        try:
            suppressed = self._close_providers(exc_type, exc, traceback)
        except BaseException as cleanup_error:  # noqa: BLE001 - typed owner
            if (
                cleanup_error is exc
                and self._owned_cleanup_error_in_cause_chain(cleanup_error)
                is not None
            ):
                # A nested retained action can publish the same TMI object
                # that was already leaving the body, now with this stack's
                # exact cleanup owner as its direct cause. Re-wrapping that
                # object would create TMI -> owner -> TMI cause cycle.
                raise
            self._raise_retained_cleanup_owner(
                primary_error=cleanup_primary_error,
                cleanup_error=cleanup_error,
            )
        try:
            self._run_settlement_callbacks()
        except BaseException as cleanup_error:  # noqa: BLE001 - typed owner
            if (
                cleanup_error is exc
                and self._owned_cleanup_error_in_cause_chain(cleanup_error)
                is not None
            ):
                # ``run_after_provider_quiescence`` already established the
                # desired exact TMI -> owner chain inside this callback.
                raise
            self._raise_retained_cleanup_owner(
                primary_error=cleanup_primary_error,
                cleanup_error=cleanup_error,
            )
        self._replay_deferred_interrupt(
            primary_error=(exc if not suppressed else None),
        )
        return suppressed

    def retry_cleanup(self) -> None:
        """Retry retained provider ownership, then retained callbacks."""

        with self._lock:
            if self._complete:
                return
        self._close_providers(retry_pending=True)
        self._run_settlement_callbacks()
        self._replay_deferred_interrupt()


__all__ = [
    "ProviderSettlementCleanupError",
    "ProviderSettlementExitStack",
]
