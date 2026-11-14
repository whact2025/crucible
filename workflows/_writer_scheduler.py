# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Disposable, backend-neutral writer lanes with serial integration.

WRK-01 deliberately separates speculative authoring from landing authority:

1. require an exact clean immutable base commit;
2. create one detached disposable worktree, ToolContext, ToolSet, backend,
   debug directory, and full mutation journal per lane;
3. run only the backend invocations concurrently;
4. drain and join every child, validate its exact scope and Git authority,
   capture a bounded patch, and restore every writer worktree to baseline;
5. apply successful patches in declaration order inside one new disposable
   integration worktree;
6. run one caller-owned union gate, capture one bounded aggregate patch, then
   restore and remove the integration worktree; and
7. return that patch to the owning workflow, which may land it only through
   its existing mutation journal and Corpus acceptance boundary.

The operator checkout is never an integration target.  Any dirty base,
scope escape, index/ref/object/config mutation, conflict, incomplete backend
evidence, cancellation, journal failure, gate failure, or cleanup failure
withholds the aggregate patch.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import threading
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, cast

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    Invocation,
    InvocationResult,
    Usage,
    UsagePricingEvidence,
)
from assay.canonical import hash_record
from assay.primitives.implement import capture_workspace_delta

from crucible.backends.capabilities import validate_toolset_capability
from crucible.tools import ToolContext, ToolMutationIndeterminate

from ._backends import BackendCleanupError
from ._mutation_journal import (
    MutationJournalCleanupError,
    MutationJournalError,
    WorkspaceMutationJournal,
    _git_argv,
    _sanitized_git_env,
    live_writer_epoch_family,
    workspace_mutation_lock,
)
from ._outcome import (
    TOOL_MUTATION_INDETERMINATE_STOP_REASON,
    is_tool_mutation_indeterminate_result,
)
from ._run_policy import WriterLanePolicy
from ._worktree import (
    WorktreeCleanupError,
    WorktreeResolver,
    _recover_or_refuse_mutation_registry,
)
from ._writer_epoch import (
    WriterEpochArmError,
    WriterEpochBoundary,
    WriterEpochLease,
    WriterEpochQuiescenceError,
    WriterSettlementReceipt,
    capture_writer_epoch_owner,
    arm_writer_epoch_family,
    quiesce_writer_epoch_family,
    require_writer_epoch_quiescence,
    scheduler_settlement_receipt,
    synchronous_callback_settlement_receipt,
)


_LANE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_SHA256_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_POLL_SECONDS = 0.025

WriterLaneStatus = Literal[
    "succeeded",
    "failed",
    "cancelled",
    "incomplete",
    "cleanup_failed",
]
WriterAggregateStatus = Literal[
    "succeeded",
    "failed",
    "cancelled",
    "integration_failed",
    "cleanup_failed",
]


class WriterSchedulerError(RuntimeError):
    """Base class for pre-transport writer-scheduler admission failures."""


class WriterProviderGenerationError(WriterSchedulerError):
    """Provider-generation setup/teardown retained an exact retry owner."""

    def __init__(
        self,
        owner: "_ManagedProviderGeneration",
        *,
        primary_error: BaseException,
    ) -> None:
        if owner.closed:
            raise ValueError(
                "WriterProviderGenerationError requires a live provider owner"
            )
        self.owner = owner
        self.primary_error = primary_error
        super().__init__(
            "writer provider generation retained cleanup authority after "
            f"{type(primary_error).__name__}: {primary_error}"
        )

    def retry_cleanup(self) -> None:
        """Retry only the exact provider owner retained by this failure."""

        self.owner.close()
        if not self.owner.closed:
            raise WriterProviderGenerationError(
                self.owner,
                primary_error=self.owner.close_error or self,
            ) from self


class WriterIsolationError(WriterSchedulerError):
    """A factory shared mutable child state across isolated writer lanes."""


class WriterIntegrationConflict(WriterSchedulerError):
    """Declaration-ordered patch integration could not apply exactly."""


class WriterCleanupError(WriterSchedulerError):
    """Cleanup failed while one or more retryable resources stay owned."""

    def __init__(
        self,
        *,
        primary_error: BaseException,
        failures: tuple[tuple[str, BaseException], ...],
        retained_resolvers: tuple[
            tuple[str, WorktreeResolver],
            ...,
        ] = (),
        retained_journals: tuple[
            tuple[str, WorkspaceMutationJournal],
            ...,
        ] = (),
        retained_backends: tuple[
            tuple[str, _ManagedBackend],
            ...,
        ] = (),
        retained_provider_generation: _ManagedProviderGeneration | None = None,
        retained_epoch_lease: WriterEpochLease | None = None,
        epoch_arm_error: WriterEpochArmError | None = None,
        epoch_quiescence_error: WriterEpochQuiescenceError | None = None,
        epoch_settlement_factory: (Callable[[], WriterSettlementReceipt] | None) = None,
        partial_result: WriterFanoutResult | None = None,
    ) -> None:
        if not failures:
            raise ValueError("WriterCleanupError requires a failure")
        retained_resolvers = tuple(
            (label, resolver)
            for label, resolver in retained_resolvers
            if resolver.owned
        )
        retained_journals = tuple(
            (label, journal)
            for label, journal in retained_journals
            if journal.cleanup_pending
        )
        retained_backends = tuple(
            (label, managed)
            for label, managed in retained_backends
            if not managed.closed
        )
        if (
            retained_provider_generation is not None
            and retained_provider_generation.closed
        ):
            retained_provider_generation = None
        if retained_epoch_lease is not None and not any(
            member.active is True for member in retained_epoch_lease.members
        ):
            retained_epoch_lease = None
            epoch_arm_error = None
            epoch_quiescence_error = None
            epoch_settlement_factory = None
        if epoch_arm_error is not None and (
            retained_epoch_lease is None
            or epoch_arm_error.lease is not retained_epoch_lease
        ):
            raise ValueError("epoch arm retry owner must bind the retained lease")
        if epoch_quiescence_error is not None and (
            retained_epoch_lease is None
            or epoch_quiescence_error.lease is not retained_epoch_lease
        ):
            raise ValueError(
                "epoch quiescence retry owner must bind the retained lease"
            )
        if epoch_arm_error is not None and epoch_quiescence_error is not None:
            raise ValueError(
                "writer cleanup cannot retain arm and quiescence retries "
                "for the same generation"
            )
        if epoch_settlement_factory is not None and retained_epoch_lease is None:
            raise ValueError("epoch settlement factory requires a retained lease")
        if not (
            retained_resolvers
            or retained_journals
            or retained_backends
            or retained_provider_generation is not None
            or retained_epoch_lease is not None
        ):
            raise ValueError(
                "WriterCleanupError requires live retryable cleanup authority"
            )
        self.primary_error = primary_error
        self.failures = failures
        self.retained_resolvers = retained_resolvers
        self._retained_journals = retained_journals
        self._retained_backends = retained_backends
        self._retained_provider_generation = retained_provider_generation
        self._retained_epoch_lease = retained_epoch_lease
        self._epoch_arm_error = epoch_arm_error
        self._epoch_quiescence_error = epoch_quiescence_error
        self._epoch_settlement_factory = epoch_settlement_factory
        self.partial_result = partial_result
        retained_journal_ids = (
            ", ".join(label for label, _journal in retained_journals) or "none"
        )
        retained_worktrees = (
            ", ".join(
                f"{label}(worktree={resolver.worktree_path}, "
                f"lease={resolver.lease_path})"
                for label, resolver in retained_resolvers
            )
            or "none"
        )
        retained_backend_ids = (
            ", ".join(label for label, _managed in retained_backends) or "none"
        )
        retained_provider = (
            "active" if retained_provider_generation is not None else "none"
        )
        retained_epoch_id = (
            retained_epoch_lease.epoch_id
            if retained_epoch_lease is not None
            else "none"
        )
        details = "; ".join(
            f"{label}: {type(error).__name__}: {error}" for label, error in failures
        )
        super().__init__(
            f"writer lifecycle cleanup failed after "
            f"{type(primary_error).__name__}: {primary_error}; "
            f"retained_journals={retained_journal_ids}; "
            f"retained_worktrees={retained_worktrees}; "
            f"retained_backends={retained_backend_ids}; failures={details}"
            f"; retained_provider_generation={retained_provider}"
            f"; retained_writer_epoch={retained_epoch_id}"
        )

    @property
    def retained_journal_ids(self) -> tuple[str, ...]:
        """Stable labels whose mutation journals remain active."""

        return tuple(
            label
            for label, journal in self._retained_journals
            if journal.cleanup_pending
        )

    @property
    def retained_backend_ids(self) -> tuple[str, ...]:
        """Stable lane IDs whose exclusive backend remains owned."""

        return tuple(
            label for label, managed in self._retained_backends if not managed.closed
        )

    @property
    def retained_epoch_id(self) -> str | None:
        """Exact durable writer generation still blocking workspace access."""

        lease = self._retained_epoch_lease
        if lease is None or not any(member.active is True for member in lease.members):
            return None
        return lease.epoch_id

    @property
    def provider_generation_retained(self) -> bool:
        """Whether the exact route/pool owner still needs cleanup."""

        owner = self._retained_provider_generation
        return owner is not None and not owner.closed

    def retry_cleanup(self) -> WriterFanoutResult | None:
        """Retry retained resources and return partial evidence on settlement."""

        failures: list[tuple[str, BaseException]] = []
        retained_journals: list[tuple[str, WorkspaceMutationJournal]] = []
        retained_resolvers: list[tuple[str, WorktreeResolver]] = []
        retained_backends: list[tuple[str, _ManagedBackend]] = []
        # A backend may own descendant processes or transports that can still
        # mutate its checkout after the invocation future itself has joined.
        # Close every retained backend before touching journal or worktree
        # authority.  If any close remains incomplete, preserve the complete
        # resource family for a later retry instead of racing rollback against
        # a live writer.
        for label, managed in self._retained_backends:
            if managed.closed:
                continue
            managed.close()
            if not managed.closed:
                retained_backends.append((label, managed))
                failures.append(
                    (
                        label,
                        managed.close_error
                        or RuntimeError("backend retained ownership after close"),
                    )
                )
        partial_result = self.partial_result
        if retained_backends:
            retained_journals = [
                (label, journal)
                for label, journal in self._retained_journals
                if journal.cleanup_pending
            ]
            retained_resolvers = [
                (label, resolver)
                for label, resolver in self.retained_resolvers
                if resolver.owned
            ]
            failures.append(
                (
                    "workspace resources",
                    MutationJournalError(
                        "mutation journal and worktree cleanup is deferred until "
                        "every retained backend closes"
                    ),
                )
            )
            raise WriterCleanupError(
                primary_error=self,
                failures=tuple(failures),
                retained_journals=tuple(retained_journals),
                retained_resolvers=tuple(retained_resolvers),
                retained_backends=tuple(retained_backends),
                retained_provider_generation=(
                    self._retained_provider_generation
                ),
                retained_epoch_lease=self._retained_epoch_lease,
                epoch_arm_error=self._epoch_arm_error,
                epoch_quiescence_error=self._epoch_quiescence_error,
                epoch_settlement_factory=self._epoch_settlement_factory,
                partial_result=partial_result,
            ) from self
        provider_generation = self._retained_provider_generation
        if provider_generation is not None and not provider_generation.closed:
            provider_generation.close()
        if provider_generation is not None and not provider_generation.closed:
            retained_journals = [
                (label, journal)
                for label, journal in self._retained_journals
                if journal.cleanup_pending
            ]
            retained_resolvers = [
                (label, resolver)
                for label, resolver in self.retained_resolvers
                if resolver.owned
            ]
            provider_error = (
                provider_generation.close_error
                or RuntimeError("provider generation retained ownership")
            )
            failures.extend(
                (
                    ("provider generation", provider_error),
                    (
                        "workspace resources",
                        MutationJournalError(
                            "writer epoch and workspace cleanup are deferred "
                            "until the provider generation closes"
                        ),
                    ),
                )
            )
            raise WriterCleanupError(
                primary_error=self,
                failures=tuple(failures),
                retained_journals=tuple(retained_journals),
                retained_resolvers=tuple(retained_resolvers),
                retained_provider_generation=provider_generation,
                retained_epoch_lease=self._retained_epoch_lease,
                epoch_arm_error=self._epoch_arm_error,
                epoch_quiescence_error=self._epoch_quiescence_error,
                epoch_settlement_factory=self._epoch_settlement_factory,
                partial_result=partial_result,
            ) from self
        epoch_lease = self._retained_epoch_lease
        epoch_arm_error = self._epoch_arm_error
        epoch_retry_error = self._epoch_quiescence_error
        epoch_settlement_factory = self._epoch_settlement_factory
        if epoch_lease is not None:
            if epoch_arm_error is not None:
                try:
                    epoch_lease = epoch_arm_error.retry_arm()
                except WriterEpochArmError as error:
                    failures.append(("writer epoch arm", error))
                    retained_journals = [
                        (label, journal)
                        for label, journal in self._retained_journals
                        if journal.cleanup_pending
                    ]
                    retained_resolvers = [
                        (label, resolver)
                        for label, resolver in self.retained_resolvers
                        if resolver.owned
                    ]
                    raise WriterCleanupError(
                        primary_error=self,
                        failures=tuple(failures),
                        retained_journals=tuple(retained_journals),
                        retained_resolvers=tuple(retained_resolvers),
                        retained_epoch_lease=epoch_lease,
                        epoch_arm_error=error,
                        partial_result=partial_result,
                    ) from self
                receipt = scheduler_settlement_receipt(
                    epoch_lease,
                    settled_owner_ids=(epoch_lease.boundary.expected_owner_ids),
                    joined_worker_ids=(epoch_lease.boundary.expected_worker_ids),
                    evidence_digest=hash_record(
                        {
                            "schema_version": (
                                "crucible.writer-scheduler-not-dispatched.v1"
                            ),
                            "epoch_id": epoch_lease.epoch_id,
                            "reason": (
                                "epoch arm failed before its writer boundary dispatched"
                            ),
                        }
                    ),
                    outcome="failed",
                )
                try:
                    quiesce_writer_epoch_family(epoch_lease, receipt)
                except WriterEpochQuiescenceError as error:
                    epoch_retry_error = error
                else:
                    epoch_settlement_factory = None
            elif epoch_retry_error is None and epoch_settlement_factory is not None:
                try:
                    receipt = epoch_settlement_factory()
                    quiesce_writer_epoch_family(epoch_lease, receipt)
                except WriterEpochQuiescenceError as error:
                    epoch_retry_error = error
                    epoch_settlement_factory = None
                except BaseException as error:  # noqa: BLE001 - keep armed
                    failures.append(("writer epoch settlement", error))
                    retained_journals = [
                        (label, journal)
                        for label, journal in self._retained_journals
                        if journal.cleanup_pending
                    ]
                    retained_resolvers = [
                        (label, resolver)
                        for label, resolver in self.retained_resolvers
                        if resolver.owned
                    ]
                    raise WriterCleanupError(
                        primary_error=self,
                        failures=tuple(failures),
                        retained_journals=tuple(retained_journals),
                        retained_resolvers=tuple(retained_resolvers),
                        retained_epoch_lease=epoch_lease,
                        epoch_settlement_factory=epoch_settlement_factory,
                        partial_result=partial_result,
                    ) from self
                else:
                    epoch_settlement_factory = None
            if epoch_retry_error is not None:
                try:
                    epoch_retry_error.retry_cleanup()
                except WriterEpochQuiescenceError as error:
                    failures.append(("writer epoch quiescence", error))
                    retained_journals = [
                        (label, journal)
                        for label, journal in self._retained_journals
                        if journal.cleanup_pending
                    ]
                    retained_resolvers = [
                        (label, resolver)
                        for label, resolver in self.retained_resolvers
                        if resolver.owned
                    ]
                    raise WriterCleanupError(
                        primary_error=self,
                        failures=tuple(failures),
                        retained_journals=tuple(retained_journals),
                        retained_resolvers=tuple(retained_resolvers),
                        retained_epoch_lease=epoch_lease,
                        epoch_quiescence_error=error,
                        partial_result=partial_result,
                    ) from self
            try:
                require_writer_epoch_quiescence(
                    epoch_lease.members,
                    allow_never_armed=False,
                )
            except BaseException as error:  # noqa: BLE001 - retain authority
                failures.append(("writer epoch quiescence", error))
                retained_journals = [
                    (label, journal)
                    for label, journal in self._retained_journals
                    if journal.cleanup_pending
                ]
                retained_resolvers = [
                    (label, resolver)
                    for label, resolver in self.retained_resolvers
                    if resolver.owned
                ]
                raise WriterCleanupError(
                    primary_error=self,
                    failures=tuple(failures),
                    retained_journals=tuple(retained_journals),
                    retained_resolvers=tuple(retained_resolvers),
                    retained_epoch_lease=epoch_lease,
                    partial_result=partial_result,
                ) from self
        if partial_result is not None:
            partial_result = replace(
                partial_result,
                cancellation_drained=True,
            )
        # Journals own rollback and registry-marker authority. They must settle
        # before a resolver may remove the checkout referenced by that marker.
        pending_journal_owners = tuple(
            (label, journal)
            for label, journal in self._retained_journals
            if journal.cleanup_pending
        )
        owned_resolver_owners = tuple(
            (label, resolver)
            for label, resolver in self.retained_resolvers
            if resolver.owned
        )

        def retry_workspace_resources() -> None:
            lane_journal_failed = False
            for label, journal in self._retained_journals:
                if not journal.cleanup_pending:
                    continue
                if lane_journal_failed:
                    failures.append(
                        (
                            label,
                            MutationJournalError(
                                "mutation journal cleanup is deferred until every "
                                "later-created lane journal settles"
                            ),
                        )
                    )
                    retained_journals.append((label, journal))
                    continue
                try:
                    restored = journal.retry_cleanup()
                except BaseException as error:
                    failures.append((label, error))
                    if journal.cleanup_pending:
                        retained_journals.append((label, journal))
                    if journal.cleanup_pending and label.startswith("lane "):
                        lane_journal_failed = True
                else:
                    if restored is not True or journal.cleanup_pending:
                        postcondition_error = MutationJournalError(
                            "mutation journal cleanup retry did not report "
                            "exact settlement"
                        )
                        failures.append((label, postcondition_error))
                        if journal.cleanup_pending:
                            retained_journals.append((label, journal))
                        if journal.cleanup_pending and label.startswith("lane "):
                            lane_journal_failed = True
            journals_block_resolver_cleanup = any(
                journal.cleanup_pending for _label, journal in retained_journals
            )
            for label, resolver in self.retained_resolvers:
                if not resolver.owned:
                    continue
                if journals_block_resolver_cleanup:
                    failures.append(
                        (
                            label,
                            MutationJournalError(
                                "worktree cleanup is deferred until every retained "
                                "mutation journal settles"
                            ),
                        )
                    )
                    retained_resolvers.append((label, resolver))
                    continue
                try:
                    resolver.close()
                except BaseException as error:
                    failures.append((label, error))
                    retained_resolvers.append((label, resolver))
                else:
                    if resolver.owned:
                        worktree_postcondition_error = RuntimeError(
                            "worktree retained ownership after cleanup retry"
                        )
                        failures.append((label, worktree_postcondition_error))
                        retained_resolvers.append((label, resolver))

        if pending_journal_owners or owned_resolver_owners:
            retry_root = (
                owned_resolver_owners[0][1].repo_root
                if owned_resolver_owners
                else Path(str(pending_journal_owners[0][1]._manifest["cwd"]))
            )
            try:
                # Marker settlement and checkout teardown are one topology
                # transition. No other workflow may baseline or mutate the
                # repository between those two steps.
                with workspace_mutation_lock(retry_root):
                    retry_workspace_resources()
            except BaseException as error:
                failures.append(("workspace mutation cleanup lock", error))
                for journal_owner in self._retained_journals:
                    if (
                        journal_owner[1].cleanup_pending
                        and journal_owner not in retained_journals
                    ):
                        retained_journals.append(journal_owner)
                for resolver_owner in self.retained_resolvers:
                    if (
                        resolver_owner[1].owned
                        and resolver_owner not in retained_resolvers
                    ):
                        retained_resolvers.append(resolver_owner)
        if failures:
            live_journals = tuple(
                owned for owned in retained_journals if owned[1].cleanup_pending
            )
            live_resolvers = tuple(
                owned for owned in retained_resolvers if owned[1].owned
            )
            live_backends = tuple(
                owned for owned in retained_backends if not owned[1].closed
            )
            if not (live_journals or live_resolvers or live_backends):
                details = "; ".join(
                    f"{label}: {type(error).__name__}: {error}"
                    for label, error in failures
                )
                raise WriterSchedulerError(
                    "writer cleanup retry encountered recovered failures "
                    f"after all authority settled: {details}"
                ) from self
            raise WriterCleanupError(
                primary_error=self,
                failures=tuple(failures),
                retained_journals=live_journals,
                retained_resolvers=live_resolvers,
                retained_backends=live_backends,
                retained_epoch_lease=(
                    epoch_lease
                    if epoch_lease is not None
                    and any(member.active for member in epoch_lease.members)
                    else None
                ),
                partial_result=partial_result,
            ) from self
        return partial_result


@dataclass(frozen=True, slots=True)
class WriterLaneRequest:
    """One isolated writer using an already-qualified parent route role."""

    lane_id: str
    role: str
    route_role: str
    scope: tuple[str, ...]
    invocation: Invocation

    def __post_init__(self) -> None:
        if (
            not isinstance(self.lane_id, str)
            or _LANE_ID_RE.fullmatch(self.lane_id) is None
        ):
            raise ValueError(
                "WriterLaneRequest.lane_id must be a portable 1-64 character identifier"
            )
        for field_name in ("role", "route_role"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"WriterLaneRequest.{field_name} must be non-empty")
        if (
            not isinstance(self.scope, tuple)
            or not self.scope
            or any(not isinstance(path, str) or not path.strip() for path in self.scope)
        ):
            raise ValueError(
                "WriterLaneRequest.scope must be a non-empty tuple of paths"
            )
        if len(self.scope) != len(set(self.scope)):
            raise ValueError("WriterLaneRequest.scope must not contain duplicates")
        if not isinstance(self.invocation, Invocation):
            raise TypeError("WriterLaneRequest.invocation must be assay.Invocation")
        if self.invocation.tools is not None:
            raise ValueError(
                "writer lanes use only their attached classified ToolSet; "
                "Invocation.tools must be None"
            )


@dataclass(frozen=True, slots=True)
class IntegrationGateResult:
    """Redacted terminal result from the workflow-owned union gate."""

    success: bool
    evidence_digest: str
    error_kind: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise TypeError("IntegrationGateResult.success must be bool")
        if (
            not isinstance(self.evidence_digest, str)
            or _SHA256_DIGEST_RE.fullmatch(self.evidence_digest) is None
        ):
            raise ValueError(
                "IntegrationGateResult.evidence_digest must be a sha256 digest"
            )
        for field_name in ("error_kind", "error_message"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"IntegrationGateResult.{field_name} must be str or None"
                )
        if self.success and (self.error_kind is not None or self.error_message):
            raise ValueError("successful IntegrationGateResult cannot carry an error")
        if not self.success and not self.error_kind:
            raise ValueError("failed IntegrationGateResult requires error_kind")


@dataclass(frozen=True, slots=True)
class WriterLaneOutcome:
    lane_id: str
    role: str
    route_role: str
    scope: tuple[str, ...]
    status: WriterLaneStatus
    result: InvocationResult | None
    backend_name: str | None
    model: str | None
    worktree: str
    debug_dir: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    token_usage_complete: bool
    cost_complete: bool
    terminal_complete: bool
    trace_complete: bool
    backend_cleanup_complete: bool
    journal_cleanup_complete: bool
    worktree_cleanup_complete: bool
    patch_bytes: int = 0
    patch_digest: str | None = None
    files_modified: tuple[str, ...] = ()
    error_kind: str | None = None
    error_message: str | None = None
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    pricing_evidence: UsagePricingEvidence | None = None

    @property
    def evidence_complete(self) -> bool:
        return (
            self.token_usage_complete
            and self.cost_complete
            and self.terminal_complete
            and self.trace_complete
            and self.backend_cleanup_complete
            and self.journal_cleanup_complete
            and self.worktree_cleanup_complete
        )

    @property
    def successful(self) -> bool:
        return (
            self.status == "succeeded"
            and self.evidence_complete
            and self.patch_digest is not None
        )

    @property
    def has_extended_pricing_evidence(self) -> bool:
        return (
            self.cache_creation_tokens != 0
            or self.cache_read_tokens != 0
            or self.pricing_evidence is not None
        )

    def to_summary(
        self,
        *,
        include_extended_pricing: bool | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "lane_id": self.lane_id,
            "role": self.role,
            "route_role": self.route_role,
            "scope": list(self.scope),
            "status": self.status,
            "backend": self.backend_name,
            "model": self.model,
            "worktree": self.worktree,
            "debug_dir": self.debug_dir,
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "token_usage_complete": self.token_usage_complete,
            "cost_complete": self.cost_complete,
            "terminal_complete": self.terminal_complete,
            "trace_complete": self.trace_complete,
            "backend_cleanup_complete": self.backend_cleanup_complete,
            "journal_cleanup_complete": self.journal_cleanup_complete,
            "worktree_cleanup_complete": self.worktree_cleanup_complete,
            "patch_bytes": self.patch_bytes,
            "patch_digest": self.patch_digest,
            "files_modified": list(self.files_modified),
            "error_kind": self.error_kind,
            "error_message": self.error_message,
        }
        if include_extended_pricing is None:
            include_extended_pricing = self.has_extended_pricing_evidence
        if include_extended_pricing:
            summary["cache_creation_tokens"] = self.cache_creation_tokens
            summary["cache_read_tokens"] = self.cache_read_tokens
            summary["pricing_evidence"] = _pricing_evidence_summary(
                self.pricing_evidence
            )
        return summary


@dataclass(frozen=True, slots=True)
class WriterFanoutResult:
    """Hash-redactable outcome plus the only patch eligible for landing."""

    children: tuple[WriterLaneOutcome, ...]
    usage: Usage
    trace: BackendExecutionTrace
    status: WriterAggregateStatus
    policy_digest: str
    base_commit: str
    integrated_patch: bytes
    integrated_files: tuple[str, ...]
    integration_gate: IntegrationGateResult | None
    all_joined: bool
    cancellation_drained: bool
    base_unchanged: bool
    integration_cleanup_complete: bool
    error_kind: str | None = None
    error_message: str | None = None

    @property
    def integrated_patch_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.integrated_patch).hexdigest()

    @property
    def evidence_complete(self) -> bool:
        return (
            self.all_joined
            and self.cancellation_drained
            and self.base_unchanged
            and self.integration_cleanup_complete
            and all(child.evidence_complete for child in self.children)
        )

    @property
    def successful(self) -> bool:
        return (
            self.status == "succeeded"
            and self.evidence_complete
            and bool(self.integrated_patch)
            and self.integration_gate is not None
            and self.integration_gate.success
        )

    def to_summary(self) -> dict[str, Any]:
        has_extended_pricing = any(
            child.has_extended_pricing_evidence for child in self.children
        )
        summary = {
            "schema_version": (
                "crucible.writer-fanout-result.v2"
                if has_extended_pricing
                else "crucible.writer-fanout-result.v1"
            ),
            "status": self.status,
            "policy_digest": self.policy_digest,
            "base_commit": self.base_commit,
            "lane_count": len(self.children),
            "cost_usd": round(self.usage.cost_usd, 6),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "all_joined": self.all_joined,
            "cancellation_drained": self.cancellation_drained,
            "base_unchanged": self.base_unchanged,
            "integration_cleanup_complete": self.integration_cleanup_complete,
            "evidence_complete": self.evidence_complete,
            "integrated_patch_bytes": len(self.integrated_patch),
            "integrated_patch_digest": self.integrated_patch_digest,
            "integrated_files": list(self.integrated_files),
            "integration_gate": (
                {
                    "success": self.integration_gate.success,
                    "evidence_digest": self.integration_gate.evidence_digest,
                    "error_kind": self.integration_gate.error_kind,
                    "error_message": self.integration_gate.error_message,
                }
                if self.integration_gate is not None
                else None
            ),
            "error_kind": self.error_kind,
            "error_message": self.error_message,
            "events": [
                {
                    "sequence": event.sequence,
                    "kind": event.kind,
                    "name": event.name,
                    "parent_id": event.parent_id,
                    "child_id": event.child_id,
                }
                for event in self.trace.events
            ],
            "children": [
                child.to_summary(
                    include_extended_pricing=has_extended_pricing,
                )
                for child in self.children
            ],
        }
        if has_extended_pricing:
            summary["cache_creation_tokens"] = self.usage.cache_creation_tokens
            summary["cache_read_tokens"] = self.usage.cache_read_tokens
        return summary


def _pricing_evidence_summary(
    pricing_evidence: UsagePricingEvidence | None,
) -> dict[str, Any] | None:
    """Project typed pricing evidence without accepting arbitrary keys."""

    if pricing_evidence is None:
        return None
    return {
        "segments": [
            {
                "input_tokens": segment.input_tokens,
                "output_tokens": segment.output_tokens,
                "cache_creation_tokens": segment.cache_creation_tokens,
                "cache_read_tokens": segment.cache_read_tokens,
                "cost_usd": segment.cost_usd,
            }
            for segment in pricing_evidence.segments
        ],
        "complete": pricing_evidence.complete,
    }


ToolSetFactory = Callable[[WriterLaneRequest, ToolContext], Any]
BackendFactory = Callable[[WriterLaneRequest, Any, Path, Path], Any]
ProviderGenerationFactory = Callable[
    [tuple[WorkspaceMutationJournal, ...]],
    Any,
]
IntegrationGate = Callable[
    [Path, tuple[str, ...], tuple[WriterLaneRequest, ...]],
    IntegrationGateResult,
]


@dataclass(frozen=True, slots=True)
class _LaneCandidate:
    outcome: WriterLaneOutcome
    patch: bytes = b""


class _ManagedProviderGeneration:
    """Own one provider context from epoch arm through close/drain proof.

    The context must yield the exact ``BackendFactory`` used by every lane.
    If setup or teardown reports a retryable cleanup exception, its bound
    retry callback is retained instead of attempting to re-enter a consumed
    generator context manager. An untyped setup/teardown failure retains the
    exact object and workspace epoch indefinitely; a raise is never treated
    as evidence that provider resources self-cleaned.
    """

    def __init__(
        self,
        factory: ProviderGenerationFactory,
        members: tuple[WorkspaceMutationJournal, ...],
    ) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._entered = False
        self._exit_attempted = False
        self._cleanup_in_progress = False
        self._backend_factory: BackendFactory | None = None
        self._retry_cleanup: Callable[[], None] | None = None
        self._close_errors: list[BaseException] = []
        self._factory = factory
        self._members = members
        self._factory_called = False
        self._context: Any | None = None
        self._enter: Callable[[], Any] | None = None
        self._exit: Callable[[Any, Any, Any], Any] | None = None

    def _record_setup_failure(self, error: BaseException) -> None:
        retry = self._bind_retry(error)
        with self._lock:
            self._close_errors.append(error)
            self._retry_cleanup = retry
            # Construction and __enter__ are untrusted provider code. A raise
            # is not proof that a transport, subprocess, or SDK client was
            # never published. Without an explicit retry callback there is no
            # safe cleanup operation, so retain the exact wrapper and armed
            # workspace epoch indefinitely rather than racing rollback.
            self._closed = False

    def _construct_context(self) -> None:
        with self._lock:
            if self._factory_called:
                raise WriterSchedulerError(
                    "provider generation factory cannot be invoked twice"
                )
            self._factory_called = True
        try:
            context = self._factory(self._members)
        except BaseException as error:  # noqa: BLE001 - retain setup owner
            self._record_setup_failure(error)
            raise WriterProviderGenerationError(
                self,
                primary_error=error,
            ) from error
        # Retain the exact returned object before touching dynamic protocol
        # attributes. Their descriptors are untrusted and may raise after the
        # object has already acquired provider resources.
        self._context = context
        invalid_context_error: TypeError | None = None
        try:
            enter = context.__enter__
            exit_context = context.__exit__
        except BaseException as exc:  # noqa: BLE001 - reject unsafe owner
            invalid_context_error = TypeError(
                "provider_generation_factory must return a context manager"
            )
            self._record_setup_failure(exc)
            raise WriterProviderGenerationError(
                self,
                primary_error=invalid_context_error,
            ) from exc
        if not callable(enter) or not callable(exit_context):
            invalid_context_error = TypeError(
                "provider_generation_factory must return a context manager"
            )
            self._record_setup_failure(invalid_context_error)
            raise WriterProviderGenerationError(
                self,
                primary_error=invalid_context_error,
            ) from invalid_context_error
        self._enter = enter
        self._exit = exit_context

    @staticmethod
    def _bind_retry(error: BaseException) -> Callable[[], None] | None:
        try:
            retry = getattr(error, "retry_cleanup", None)
        except BaseException:  # noqa: BLE001 - error object is untrusted
            return None
        return retry if callable(retry) else None

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def backend_factory(self) -> BackendFactory:
        with self._lock:
            backend_factory = self._backend_factory
        if backend_factory is None:
            raise WriterSchedulerError(
                "writer provider generation has no admitted backend factory"
            )
        return backend_factory

    @property
    def close_error(self) -> BaseException | None:
        with self._lock:
            errors = tuple(self._close_errors)
        if not errors:
            return None
        if len(errors) == 1:
            return errors[0]
        return BackendCleanupError(
            tuple(
                (f"provider generation close attempt {index}", error)
                for index, error in enumerate(errors, start=1)
            ),
            retry_cleanup=self.close,
        )

    def open(self) -> BackendFactory:
        with self._lock:
            if self._entered or self._closed:
                raise WriterSchedulerError(
                    "provider generation context cannot be entered twice"
                )
        self._construct_context()
        enter = self._enter
        if enter is None:
            raise WriterSchedulerError(
                "provider generation context has no admitted enter operation"
            )
        try:
            candidate = enter()
        except BaseException as error:  # noqa: BLE001 - retain retry owner
            self._record_setup_failure(error)
            raise WriterProviderGenerationError(
                self,
                primary_error=error,
            ) from error
        with self._lock:
            self._entered = True
        if not callable(candidate):
            invalid_factory_error = TypeError(
                "provider generation context must yield a BackendFactory"
            )
            self.close()
            if not self.closed:
                raise WriterProviderGenerationError(
                    self,
                    primary_error=invalid_factory_error,
                ) from invalid_factory_error
            raise invalid_factory_error
        with self._lock:
            self._backend_factory = candidate
        return candidate

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._cleanup_in_progress:
                return
            retry = self._retry_cleanup
            entered = self._entered
            if retry is None and entered:
                if self._exit_attempted:
                    return
                self._exit_attempted = True
            self._cleanup_in_progress = True
        try:
            if retry is not None:
                retry()
            elif entered:
                exit_context = self._exit
                if exit_context is None:
                    raise WriterSchedulerError(
                        "provider generation retained no exit operation"
                    )
                exit_context(None, None, None)
            else:
                raise WriterSchedulerError(
                    "provider generation retained no cleanup operation"
                )
        except BaseException as error:  # noqa: BLE001 - retain exact owner
            with self._lock:
                self._close_errors.append(error)
                replacement = self._bind_retry(error)
                if replacement is not None:
                    self._retry_cleanup = replacement
                self._cleanup_in_progress = False
            return
        with self._lock:
            self._closed = True
            self._backend_factory = None
            self._retry_cleanup = None
            self._context = None
            self._enter = None
            self._exit = None
            self._cleanup_in_progress = False


class _ManagedBackend:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._lock = threading.Lock()
        self._closed = False
        self._dispatched = False
        self._close_errors: list[BaseException] = []
        try:
            close = getattr(backend, "close", None)
        except BaseException as exc:  # noqa: BLE001 - retain exact owner
            close = None
            self._close_errors.append(exc)
        self._close = close if callable(close) else None

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def close_error(self) -> BaseException | None:
        with self._lock:
            errors = tuple(self._close_errors)
        if not errors:
            return None
        if len(errors) == 1:
            return errors[0]
        return BackendCleanupError(
            tuple(
                (f"backend close attempt {index}", error)
                for index, error in enumerate(errors, start=1)
            ),
            retry_cleanup=self.retry_cleanup,
        )

    @property
    def quiescence_supported(self) -> bool:
        """Whether this writer exposes an explicit settlement operation."""

        with self._lock:
            return self._close is not None

    def mark_dispatched(self) -> None:
        """Freeze cleanup authority before workspace-write dispatch."""

        with self._lock:
            if self._closed:
                raise WriterSchedulerError("closed writer backend cannot be dispatched")
            if self._close is None:
                raise WriterSchedulerError(
                    "writer backend must expose callable close before "
                    "workspace-write dispatch"
                )
            self._dispatched = True

    def retry_cleanup(self) -> None:
        """Retry close and raise an actionable aggregate if still owned."""

        self.close()
        if self.closed:
            return
        error = self.close_error
        if isinstance(error, BackendCleanupError):
            raise error
        raise BackendCleanupError(
            (("backend close retry", error or RuntimeError("close failed")),),
            retry_cleanup=self.retry_cleanup,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            close = self._close
            if close is None and not self._dispatched:
                # Construction itself may have acquired resources.  Before a
                # backend is ever dispatched, an explicit caller-side repair
                # may supply the missing settlement operation.  Once dispatch
                # begins, however, the exact pre-bound callable is immutable.
                try:
                    candidate = getattr(self.backend, "close", None)
                except BaseException as exc:  # noqa: BLE001 - retained owner
                    self._close_errors.append(exc)
                    candidate = None
                if callable(candidate):
                    self._close = candidate
                    close = candidate
            if close is None:
                if not self._close_errors:
                    self._close_errors.append(
                        WriterSchedulerError(
                            "writer backend does not expose callable close; "
                            "workspace quiescence cannot be proved"
                        )
                    )
                return
            try:
                close()
            except BaseException as exc:  # noqa: BLE001 - normalized evidence
                self._close_errors.append(exc)
            else:
                self._closed = True


def _run_git(cwd: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        _git_argv(*args),
        cwd=cwd,
        env=_sanitized_git_env(GIT_OPTIONAL_LOCKS="0"),
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise WriterSchedulerError(
            f"git {' '.join(args)} failed in {cwd}: {detail[:2000]}"
        )
    return result.stdout


def _exact_head(cwd: Path) -> str:
    return (
        _run_git(cwd, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    )


def _status(cwd: Path) -> bytes:
    return _run_git(
        cwd,
        "status",
        "--porcelain=v2",
        "-z",
        "--untracked-files=all",
    )


def _assert_exact_clean_base(repo_root: Path, base_commit: str) -> None:
    if _exact_head(repo_root) != base_commit:
        raise WriterSchedulerError(
            "writer scheduler base commit does not match checkout HEAD"
        )
    dirty = _status(repo_root)
    if dirty:
        raise WriterSchedulerError(
            "writer scheduler requires an exact clean base checkout"
        )


def _normalize_scope(repo_root: Path, scope: tuple[str, ...]) -> tuple[str, ...]:
    from ._mutation_journal import normalize_scope_path  # noqa: PLC0415

    normalized = tuple(
        dict.fromkeys(normalize_scope_path(repo_root, path) for path in scope)
    )
    if not normalized:
        raise WriterSchedulerError("writer lane scope normalized to empty")
    return normalized


def _scopes_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return any(
        a == b or a.startswith(b.rstrip("/") + "/") or b.startswith(a.rstrip("/") + "/")
        for a in left
        for b in right
    )


def _validate_writer_toolset(toolset: Any) -> None:
    validate_toolset_capability("workspace_write", toolset)
    read_names = getattr(toolset, "read_only_names", None)
    write_names = getattr(toolset, "write_names", None)
    if read_names is None or write_names is None:
        raise WriterSchedulerError(
            "writer ToolSet must expose complete read_only_names and "
            "write_names metadata"
        )
    reads = {str(name) for name in read_names}
    writes = {str(name) for name in write_names}
    if not writes:
        raise WriterSchedulerError("writer ToolSet exposes no mutating tools")
    if reads & writes:
        raise WriterSchedulerError("writer ToolSet read/write classifications overlap")
    descriptors = toolset.to_anthropic_tools()
    if not isinstance(descriptors, list):
        raise WriterSchedulerError("writer ToolSet descriptors must be a list")
    try:
        names = {entry["name"] for entry in descriptors}
    except (KeyError, TypeError) as exc:
        raise WriterSchedulerError("writer ToolSet descriptors require names") from exc
    if (
        any(not isinstance(name, str) or not name for name in names)
        or names != reads | writes
    ):
        raise WriterSchedulerError(
            "writer ToolSet classification does not exactly match descriptors"
        )


def _usage_values(result: InvocationResult) -> tuple[float, int, int, int, int]:
    usage = result.usage
    if (
        isinstance(usage.cost_usd, bool)
        or not isinstance(usage.cost_usd, (int, float))
        or not math.isfinite(usage.cost_usd)
        or usage.cost_usd < 0.0
    ):
        raise ValueError("lane usage.cost_usd is not finite and non-negative")
    for field_name in (
        "input_tokens",
        "output_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
    ):
        value = getattr(usage, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"lane usage.{field_name} is not an exact non-negative integer"
            )
    if usage.cache_creation_tokens + usage.cache_read_tokens > usage.input_tokens:
        raise ValueError("lane usage cache-token subsets exceed input_tokens")
    return (
        float(usage.cost_usd),
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_tokens,
        usage.cache_read_tokens,
    )


def _reserved_invocation(
    request: WriterLaneRequest,
    *,
    policy: WriterLanePolicy,
    parent_budget_usd: float | None,
    lane_count: int,
) -> Invocation:
    budget = request.invocation.budget_usd
    if parent_budget_usd is not None:
        total_slot = parent_budget_usd * policy.max_total_cost_share / lane_count
        lane_slot = parent_budget_usd * policy.max_lane_cost_share
        reserved = min(total_slot, lane_slot)
        budget = reserved if budget is None else min(budget, reserved)
    return replace(
        request.invocation,
        max_turns=min(request.invocation.max_turns, policy.max_turns),
        budget_usd=budget,
    )


def _zero_outcome(
    request: WriterLaneRequest,
    *,
    scope: tuple[str, ...],
    worktree: Path,
    debug_dir: Path,
    status: WriterLaneStatus,
    error_kind: str,
    error_message: str,
) -> WriterLaneOutcome:
    cancelled = status == "cancelled"
    return WriterLaneOutcome(
        lane_id=request.lane_id,
        role=request.role,
        route_role=request.route_role,
        scope=scope,
        status=status,
        result=None,
        backend_name=None,
        model=None,
        worktree=str(worktree),
        debug_dir=str(debug_dir),
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        token_usage_complete=True,
        cost_complete=True,
        terminal_complete=True,
        trace_complete=cancelled,
        backend_cleanup_complete=True,
        journal_cleanup_complete=True,
        worktree_cleanup_complete=True,
        error_kind=error_kind,
        error_message=error_message[:2000],
    )


def _fatal_outcome(
    request: WriterLaneRequest,
    *,
    scope: tuple[str, ...],
    worktree: Path,
    debug_dir: Path,
    backend: Any | None,
    error: BaseException,
    backend_cleanup_complete: bool,
    journal_cleanup_complete: bool,
    worktree_cleanup_complete: bool,
) -> WriterLaneOutcome:
    """Represent an unstructured child unwind without inventing usage."""

    cleanup_complete = (
        backend_cleanup_complete
        and journal_cleanup_complete
        and worktree_cleanup_complete
    )
    return WriterLaneOutcome(
        lane_id=request.lane_id,
        role=request.role,
        route_role=request.route_role,
        scope=scope,
        status="failed" if cleanup_complete else "cleanup_failed",
        result=None,
        backend_name=(
            str(getattr(backend, "name", "unknown")) if backend is not None else None
        ),
        model=(
            str(backend.model)
            if backend is not None and getattr(backend, "model", None) is not None
            else None
        ),
        worktree=str(worktree),
        debug_dir=str(debug_dir),
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        token_usage_complete=False,
        cost_complete=False,
        terminal_complete=False,
        trace_complete=False,
        backend_cleanup_complete=backend_cleanup_complete,
        journal_cleanup_complete=journal_cleanup_complete,
        worktree_cleanup_complete=worktree_cleanup_complete,
        error_kind=type(error).__name__,
        error_message=str(error)[:2000],
    )


def _result_outcome(
    request: WriterLaneRequest,
    *,
    scope: tuple[str, ...],
    worktree: Path,
    debug_dir: Path,
    backend: Any,
    result: InvocationResult,
    policy: WriterLanePolicy,
    cleanup_error: BaseException | None,
) -> WriterLaneOutcome:
    mutation_indeterminate = is_tool_mutation_indeterminate_result(result)
    usage_error: BaseException | None = None
    try:
        (
            cost,
            input_tokens,
            output_tokens,
            cache_creation_tokens,
            cache_read_tokens,
        ) = _usage_values(result)
    except (TypeError, ValueError) as exc:
        if not mutation_indeterminate:
            raise
        usage_error = exc
        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        cache_creation_tokens = 0
        cache_read_tokens = 0
    completeness = result.completeness
    error = result.error
    error_kind = getattr(error, "kind", None)
    error_message = getattr(error, "message", None)
    if mutation_indeterminate:
        # Backend adapters must express this through a transport-shaped Assay
        # error for compatibility.  Retain the workflow-owned causal subtype
        # in the durable lane summary and keep the lane permanently failed.
        error_kind = TOOL_MUTATION_INDETERMINATE_STOP_REASON
        if error_message is None:
            error_message = "source mutation settlement is indeterminate"
        if usage_error is not None:
            error_message = (
                f"{error_message}; provider accounting is an unknown lower "
                f"bound after {type(usage_error).__name__}: {usage_error}"
            )
    if cleanup_error is not None:
        status: WriterLaneStatus = "cleanup_failed"
        if mutation_indeterminate:
            error_message = (
                f"{error_message or 'source mutation settlement is indeterminate'}; "
                "provider cleanup remains pending: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        else:
            error_kind = type(cleanup_error).__name__
            error_message = str(cleanup_error)
    elif mutation_indeterminate:
        status = "failed"
    elif error_kind == "cancellation":
        status = "cancelled"
    elif error is not None:
        status = "failed"
    elif len(result.final_message.encode("utf-8")) > policy.max_lane_output_bytes:
        status = "failed"
        error_kind = "LaneOutputLimitExceeded"
        error_message = (
            "writer lane final message exceeds max_lane_output_bytes="
            f"{policy.max_lane_output_bytes}"
        )
    elif not completeness.complete:
        status = "incomplete"
        error_kind = "IncompleteLaneEvidence"
        error_message = "writer lane lacks complete token/cost/terminal/trace evidence"
    else:
        status = "succeeded"
    return WriterLaneOutcome(
        lane_id=request.lane_id,
        role=request.role,
        route_role=request.route_role,
        scope=scope,
        status=status,
        result=result,
        backend_name=str(getattr(backend, "name", "unknown")),
        model=(
            str(backend.model) if getattr(backend, "model", None) is not None else None
        ),
        worktree=str(worktree),
        debug_dir=str(debug_dir),
        cost_usd=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_usage_complete=(
            completeness.token_usage_complete and usage_error is None
        ),
        cost_complete=completeness.cost_complete and usage_error is None,
        terminal_complete=completeness.terminal_complete,
        trace_complete=completeness.trace_complete,
        backend_cleanup_complete=cleanup_error is None,
        journal_cleanup_complete=True,
        worktree_cleanup_complete=True,
        error_kind=(str(error_kind) if error_kind is not None else None),
        error_message=(
            str(error_message)[:2000] if error_message is not None else None
        ),
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        pricing_evidence=(
            result.pricing_evidence if usage_error is None else None
        ),
    )


def _failed_outcome(
    outcome: WriterLaneOutcome,
    *,
    error: BaseException,
    status: WriterLaneStatus = "failed",
) -> WriterLaneOutcome:
    mutation_indeterminate = (
        outcome.error_kind == TOOL_MUTATION_INDETERMINATE_STOP_REASON
    )
    return replace(
        outcome,
        status=status,
        patch_bytes=0,
        patch_digest=None,
        files_modified=(),
        error_kind=(
            TOOL_MUTATION_INDETERMINATE_STOP_REASON
            if mutation_indeterminate
            else type(error).__name__
        ),
        error_message=(
            (
                f"{outcome.error_message or 'source mutation settlement is indeterminate'}; "
                f"secondary {type(error).__name__}: {error}"
            )
            if mutation_indeterminate
            else str(error)
        )[:2000],
    )


def _apply_patch(worktree: Path, patch: bytes, *, lane_id: str) -> None:
    for check_only in (True, False):
        args = ["apply", "--recount", "--whitespace=nowarn"]
        if check_only:
            args.append("--check")
        result = subprocess.run(
            _git_argv(*args),
            cwd=worktree,
            env=_sanitized_git_env(GIT_OPTIONAL_LOCKS="0"),
            input=patch,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise WriterIntegrationConflict(
                f"declaration-ordered integration rejected lane {lane_id!r}: "
                f"{detail[:2000]}"
            )


def _state_root(path: Path, repo_root: Path) -> Path:
    candidate = path.expanduser().resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise WriterSchedulerError(
            "writer scheduler state_root must be outside the target repository"
        )
    if candidate.exists():
        raise WriterSchedulerError(
            f"writer scheduler state_root already exists: {candidate}"
        )
    if not any(parent.name == "runs" for parent in candidate.parents):
        raise WriterSchedulerError(
            "writer scheduler state_root must be beneath a Corpus runs directory"
        )
    candidate.mkdir(parents=True, mode=0o700)
    return candidate


def run_isolated_writer_fanout(
    requests: tuple[WriterLaneRequest, ...],
    *,
    policy: WriterLanePolicy,
    repo_root: Path,
    base_commit: str,
    backend_factory: BackendFactory,
    toolset_factory: ToolSetFactory,
    state_root: Path,
    integration_gate: IntegrationGate | None,
    parent_budget_usd: float | None = None,
    cancel_event: threading.Event | None = None,
    provider_generation_factory: ProviderGenerationFactory | None = None,
) -> WriterFanoutResult:
    """Run isolated writers and return one gate-qualified aggregate patch.

    Child/backend failures are normalized into the returned result. Invalid
    authority, policy, base, state-root, factory, or scope configuration fails
    before transport with :class:`WriterSchedulerError`. If an exclusive
    backend or disposable worktree remains owned after internal cleanup
    retries, :class:`WriterCleanupError` carries the partial cost/trace result
    and the only caller-authorized ``retry_cleanup()`` handle.
    """

    if not isinstance(policy, WriterLanePolicy) or not policy.enabled:
        raise WriterSchedulerError(
            "writer scheduler requires WriterLanePolicy.mode='isolated_writer'"
        )
    if not requests:
        raise WriterSchedulerError("writer scheduler requires at least one lane")
    if len(requests) > policy.max_lanes:
        raise WriterSchedulerError(
            f"requested {len(requests)} lanes exceeds max_lanes={policy.max_lanes}"
        )
    lane_ids = tuple(request.lane_id for request in requests)
    if len(lane_ids) != len(set(lane_ids)):
        raise WriterSchedulerError("writer lane IDs must be unique")
    for request in requests:
        if request.route_role not in policy.eligible_roles:
            raise WriterSchedulerError(
                f"writer route role {request.route_role!r} is not eligible"
            )
    if policy.require_union_gate and integration_gate is None:
        raise WriterSchedulerError(
            "writer policy requires a workflow-owned union integration gate"
        )
    if not isinstance(base_commit, str) or _OBJECT_ID_RE.fullmatch(base_commit) is None:
        raise WriterSchedulerError(
            "base_commit must be an exact lowercase 40- or 64-hex object ID"
        )
    if parent_budget_usd is not None and (
        isinstance(parent_budget_usd, bool)
        or not isinstance(parent_budget_usd, (int, float))
        or not math.isfinite(parent_budget_usd)
        or parent_budget_usd < 0.0
    ):
        raise WriterSchedulerError("parent_budget_usd must be finite and non-negative")
    if parent_budget_usd is None and any(
        request.invocation.budget_usd is None for request in requests
    ):
        raise WriterSchedulerError(
            "writer lanes require a finite parent budget or explicit budget "
            "on every invocation"
        )

    try:
        repo = repo_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise WriterSchedulerError(f"writer repository is unavailable: {exc}") from exc
    if not repo.is_dir():
        raise WriterSchedulerError("writer repository is not a directory")
    normalized_scopes = {
        request.lane_id: _normalize_scope(repo, request.scope) for request in requests
    }
    if policy.require_disjoint_scopes:
        for index, left in enumerate(requests):
            for right in requests[index + 1 :]:
                if _scopes_overlap(
                    normalized_scopes[left.lane_id],
                    normalized_scopes[right.lane_id],
                ):
                    raise WriterSchedulerError(
                        "writer lane scopes overlap under disjoint-scope policy: "
                        f"{left.lane_id!r} and {right.lane_id!r}"
                    )

    with workspace_mutation_lock(repo):
        _recover_or_refuse_mutation_registry(repo)
        _assert_exact_clean_base(repo, base_commit)
        run_state = _state_root(state_root, repo)
        debug_root = run_state / "debug"
        debug_root.mkdir(mode=0o700)
        journal_root = run_state / "journals"
        journal_root.mkdir(mode=0o700)
        debug_by_id = {
            request.lane_id: debug_root / request.lane_id for request in requests
        }
        for path in debug_by_id.values():
            path.mkdir(mode=0o700)

        resolvers: dict[str, WorktreeResolver] = {}
        worktrees: dict[str, Path] = {}
        lane_journals: dict[str, WorkspaceMutationJournal] = {}
        parent_journal: WorkspaceMutationJournal | None = None
        outcomes_by_id: dict[str, WriterLaneOutcome] = {}
        tool_mutation_terminal_by_id: dict[
            str,
            ToolMutationIndeterminate,
        ] = {}
        patches_by_id: dict[str, bytes] = {}
        active: dict[str, _ManagedBackend] = {}
        state_lock = threading.Lock()
        claimed_backends: list[Any] = []
        claimed_toolsets: list[Any] = []
        claimed_contexts: list[ToolContext] = []
        internal_cancel = threading.Event()
        cancellation_started = False
        all_joined = False
        integration_cleanup_complete = True
        base_unchanged = False
        integration_gate_result: IntegrationGateResult | None = None
        aggregate_error: BaseException | None = None
        integrated_patch = b""
        integrated_files: tuple[str, ...] = ()
        integration_resolver: WorktreeResolver | None = None
        futures_by_id: dict[str, Future[WriterLaneOutcome]] = {}
        retained_journals: dict[
            int,
            tuple[str, WorkspaceMutationJournal],
        ] = {}
        writer_epoch_lease: WriterEpochLease | None = None
        writer_epoch_quiescence_error: WriterEpochQuiescenceError | None = None
        writer_epoch_quiesced = True
        integration_epoch_lease: WriterEpochLease | None = None
        integration_epoch_arm_error: WriterEpochArmError | None = None
        integration_epoch_quiescence_error: WriterEpochQuiescenceError | None = None
        integration_epoch_quiesced = True
        factory_state_by_id = {lane_id: "not_started" for lane_id in lane_ids}
        managed_backends_by_id: dict[str, _ManagedBackend] = {}
        provider_generation: _ManagedProviderGeneration | None = None
        effective_backend_factory = backend_factory

        def cancellation_requested() -> bool:
            return internal_cancel.is_set() or (
                cancel_event is not None and cancel_event.is_set()
            )

        def claim_unique(bucket: list[Any], value: Any, *, kind: str) -> None:
            with state_lock:
                if any(prior is value for prior in bucket):
                    raise WriterIsolationError(
                        f"{kind} object was shared across writer lanes"
                    )
                bucket.append(value)

        def retain_journal(
            label: str,
            journal: WorkspaceMutationJournal,
        ) -> None:
            """Keep exact rollback authority only while its journal is active."""

            if journal.cleanup_pending:
                retained_journals[id(journal)] = (label, journal)
            else:
                retained_journals.pop(id(journal), None)

        def retained_journal_snapshot() -> tuple[
            tuple[str, WorkspaceMutationJournal],
            ...,
        ]:
            return tuple(
                owned
                for owned in retained_journals.values()
                if owned[1].cleanup_pending
            )

        def build_writer_epoch_settlement_receipt() -> WriterSettlementReceipt:
            """Build proof only after all factories and owners are settled."""

            lease = writer_epoch_lease
            if lease is None:
                raise WriterSchedulerError(
                    "writer epoch settlement has no exact armed lease"
                )
            if (
                provider_generation is not None
                and not provider_generation.closed
            ):
                raise WriterSchedulerError(
                    "writer epoch cannot settle before its provider generation"
                )
            with state_lock:
                states = tuple(
                    (
                        lane_id,
                        (
                            "settled"
                            if factory_state_by_id[lane_id] == "owned"
                            and lane_id in managed_backends_by_id
                            and managed_backends_by_id[lane_id].closed
                            else "settled"
                            if (
                                factory_state_by_id[lane_id]
                                == "provider_factory_failed"
                                and provider_generation is not None
                                and provider_generation.closed
                            )
                            else factory_state_by_id[lane_id]
                        ),
                    )
                    for lane_id in sorted(factory_state_by_id)
                )
                live_backends = tuple(
                    lane_id
                    for lane_id, managed in managed_backends_by_id.items()
                    if not managed.closed
                )
            if live_backends or any(
                state not in {"not_started", "settled"} for _lane_id, state in states
            ):
                raise WriterSchedulerError(
                    "writer epoch cannot settle a live or unowned factory "
                    f"boundary: live={live_backends!r} states={states!r}"
                )
            if any(not future.done() for future in futures_by_id.values()):
                raise WriterSchedulerError(
                    "writer epoch cannot settle before every submitted "
                    "future is terminal"
                )
            outcome_statuses = tuple(
                (
                    lane_id,
                    outcomes_by_id[lane_id].status
                    if lane_id in outcomes_by_id
                    else "not_started",
                )
                for lane_id in sorted(factory_state_by_id)
            )
            return scheduler_settlement_receipt(
                lease,
                settled_owner_ids=lane_ids,
                joined_worker_ids=lane_ids,
                evidence_digest=hash_record(
                    {
                        "schema_version": ("crucible.writer-scheduler-settlement.v1"),
                        "epoch_id": lease.epoch_id,
                        "factory_states": states,
                        "outcome_statuses": outcome_statuses,
                        "all_submitted_futures_terminal": all(
                            future.done() for future in futures_by_id.values()
                        ),
                        "provider_generation_settled": (
                            provider_generation is None
                            or provider_generation.closed
                        ),
                    }
                ),
                outcome=(
                    "succeeded"
                    if outcome_statuses
                    and all(
                        status == "succeeded" for _lane_id, status in outcome_statuses
                    )
                    else "failed"
                ),
            )

        def settle_writer_epoch_if_safe() -> bool:
            """Persist exact close+join proof before any workspace read."""

            nonlocal writer_epoch_quiescence_error
            nonlocal writer_epoch_quiesced
            lease = writer_epoch_lease
            if lease is None:
                return writer_epoch_quiesced
            try:
                receipt = build_writer_epoch_settlement_receipt()
                quiesce_writer_epoch_family(lease, receipt)
                require_writer_epoch_quiescence(
                    lease.members,
                    allow_never_armed=False,
                )
            except WriterEpochQuiescenceError as error:
                writer_epoch_quiescence_error = error
                writer_epoch_quiesced = False
                return False
            except WriterSchedulerError:
                writer_epoch_quiesced = False
                return False
            writer_epoch_quiescence_error = None
            writer_epoch_quiesced = True
            return True

        def settle_provider_generation_if_safe(*, attempts: int = 2) -> bool:
            """Close the pool/route owner after lanes join, before any read."""

            owner = provider_generation
            if owner is None:
                return True
            for _attempt in range(attempts):
                if owner.closed:
                    break
                owner.close()
            return owner.closed

        try:
            for request in requests:
                resolver = WorktreeResolver(repo, base_ref=base_commit)
                # Arm ownership before creation. _ensure_worktree_locked()
                # can publish a Git worktree and durable lease before a later
                # validation or lease-publication failure is raised.
                resolvers[request.lane_id] = resolver
                worktree = resolver._ensure_worktree_locked()
                if _exact_head(worktree) != base_commit or _status(worktree):
                    raise WriterSchedulerError(
                        f"writer worktree {request.lane_id!r} did not start "
                        "at the exact clean base"
                    )
                worktrees[request.lane_id] = worktree
            # Detect an operator/base race immediately before durable capture.
            _assert_exact_clean_base(repo, base_commit)
            peer_roots = tuple(str(path) for path in worktrees.values())
            try:
                parent_journal = WorkspaceMutationJournal.begin(
                    cwd=repo,
                    journal_root=journal_root / "parent",
                    slice_id="writer-fanout-parent",
                    scope=(),
                    sequence=0,
                    spec=None,
                    _registered_peer_roots=peer_roots,
                    _suppress_peer_guards=True,
                )
            except MutationJournalCleanupError as cleanup_error:
                # begin() can publish its registry marker before a later setup
                # failure.  Preserve that exact owner even though assignment
                # of the successful return value never happened.
                parent_journal = cleanup_error.owner
                raise
            if parent_journal._manifest.get("baseline_dirty"):
                raise WriterSchedulerError(
                    "base became dirty while writer authority was captured"
                )
            for sequence, request in enumerate(requests, start=1):
                worktree = worktrees[request.lane_id]
                other_roots = tuple(
                    str(path)
                    for path in (repo, *worktrees.values())
                    if path != worktree
                )
                try:
                    journal = WorkspaceMutationJournal.begin(
                        cwd=worktree,
                        journal_root=journal_root / "lanes" / request.lane_id,
                        slice_id=f"writer-lane::{request.lane_id}",
                        scope=normalized_scopes[request.lane_id],
                        sequence=sequence,
                        spec=None,
                        parent_transaction_id=parent_journal.transaction_id,
                        _registered_peer_roots=other_roots,
                        _suppress_peer_guards=True,
                    )
                except MutationJournalCleanupError as cleanup_error:
                    # As above, the exception itself is the only in-memory
                    # handle for retrying a partially-created lane journal.
                    lane_journals[request.lane_id] = cleanup_error.owner
                    raise
                if journal._manifest.get("baseline_dirty"):
                    raise WriterSchedulerError(
                        f"writer worktree {request.lane_id!r} was dirty at "
                        "journal baseline"
                    )
                lane_journals[request.lane_id] = journal

            epoch_members = (
                parent_journal,
                *(lane_journals[request.lane_id] for request in requests),
            )
            writer_epoch_quiesced = False
            try:
                writer_epoch_lease = arm_writer_epoch_family(
                    epoch_members,
                    owner=capture_writer_epoch_owner(
                        "writer_scheduler",
                        parent_journal.transaction_id,
                    ),
                    boundary=WriterEpochBoundary(
                        kind="writer_scheduler_fanout",
                        identifier=parent_journal.transaction_id,
                        settlement_kind="scheduler_join",
                        expected_owner_ids=lane_ids,
                        expected_worker_ids=lane_ids,
                    ),
                )
            except WriterEpochArmError as error:
                writer_epoch_lease = error.lease
                raise

            if provider_generation_factory is not None:
                provider_generation = _ManagedProviderGeneration(
                    provider_generation_factory,
                    cast(
                        tuple[WorkspaceMutationJournal, ...],
                        writer_epoch_lease.members,
                    ),
                )
                effective_backend_factory = provider_generation.open()

            def run_lane(request: WriterLaneRequest) -> WriterLaneOutcome:
                lane_id = request.lane_id
                worktree = worktrees[lane_id]
                debug_dir = debug_by_id[lane_id]
                scope = normalized_scopes[lane_id]
                if cancellation_requested():
                    return _zero_outcome(
                        request,
                        scope=scope,
                        worktree=worktree,
                        debug_dir=debug_dir,
                        status="cancelled",
                        error_kind="SchedulerCancellation",
                        error_message="cancelled before backend construction",
                    )
                backend: Any | None = None
                managed: _ManagedBackend | None = None
                result: InvocationResult | None = None
                mutation_terminal = False
                cleanup_attempted = False
                try:
                    context = ToolContext(
                        workspace_root=worktree,
                        repo_root=worktree,
                        context_budget=policy.max_context_bytes,
                        write_scope=scope,
                        mutation_journal_dir=(
                            lane_journals[lane_id].journal_dir
                        ),
                        mutation_journal_transaction_id=(
                            lane_journals[lane_id].transaction_id
                        ),
                        path_resolver=resolvers[lane_id],
                    )
                    claim_unique(claimed_contexts, context, kind="ToolContext")
                    with state_lock:
                        factory_state_by_id[lane_id] = "toolset_entered"
                    toolset = toolset_factory(request, context)
                    _validate_writer_toolset(toolset)
                    claim_unique(claimed_toolsets, toolset, kind="ToolSet")
                    with state_lock:
                        factory_state_by_id[lane_id] = "factory_entered"
                    backend = effective_backend_factory(
                        request,
                        toolset,
                        debug_dir,
                        worktree,
                    )
                    managed = _ManagedBackend(backend)
                    with state_lock:
                        active[lane_id] = managed
                        managed_backends_by_id[lane_id] = managed
                        factory_state_by_id[lane_id] = "owned"
                    claim_unique(claimed_backends, backend, kind="backend")
                    if not managed.quiescence_supported:
                        raise WriterSchedulerError(
                            "writer backend must expose callable close before "
                            "workspace-write dispatch"
                        )
                    if cancellation_requested():
                        managed.close()
                        return _zero_outcome(
                            request,
                            scope=scope,
                            worktree=worktree,
                            debug_dir=debug_dir,
                            status="cancelled",
                            error_kind="SchedulerCancellation",
                            error_message="cancelled before backend invocation",
                        )
                    managed.mark_dispatched()
                    invocation = _reserved_invocation(
                        request,
                        policy=policy,
                        parent_budget_usd=parent_budget_usd,
                        lane_count=len(requests),
                    )
                    candidate = backend.invoke(invocation)
                    if not isinstance(candidate, InvocationResult):
                        raise TypeError(
                            "writer backend returned a non-InvocationResult"
                        )
                    result = candidate
                    mutation_terminal = is_tool_mutation_indeterminate_result(
                        result
                    )
                    if mutation_terminal:
                        with state_lock:
                            tool_mutation_terminal_by_id[lane_id] = (
                                ToolMutationIndeterminate(
                                    "backend returned indeterminate source "
                                    "mutation settlement"
                                )
                            )
                except Exception as exc:  # noqa: BLE001 - normalized terminal
                    mutation_terminal = isinstance(
                        exc,
                        ToolMutationIndeterminate,
                    )
                    if managed is not None:
                        managed.close()
                        cleanup_attempted = True
                    cleanup_error = managed.close_error if managed is not None else None
                    if backend is None:
                        with state_lock:
                            if (
                                provider_generation is not None
                                and factory_state_by_id[lane_id]
                                == "factory_entered"
                            ):
                                factory_state_by_id[lane_id] = (
                                    "provider_factory_failed"
                                )
                        return _zero_outcome(
                            request,
                            scope=scope,
                            worktree=worktree,
                            debug_dir=debug_dir,
                            status=(
                                "cancelled" if cancellation_requested() else "failed"
                            ),
                            error_kind=type(exc).__name__,
                            error_message=str(exc),
                        )
                    if mutation_terminal:
                        with state_lock:
                            tool_mutation_terminal_by_id[lane_id] = exc
                    return WriterLaneOutcome(
                        lane_id=lane_id,
                        role=request.role,
                        route_role=request.route_role,
                        scope=scope,
                        status=(
                            "cleanup_failed"
                            if cleanup_error is not None
                            else "cancelled"
                            if cancellation_requested()
                            else "failed"
                        ),
                        result=result,
                        backend_name=str(getattr(backend, "name", "unknown")),
                        model=(
                            str(backend.model)
                            if getattr(backend, "model", None) is not None
                            else None
                        ),
                        worktree=str(worktree),
                        debug_dir=str(debug_dir),
                        cost_usd=0.0,
                        input_tokens=0,
                        output_tokens=0,
                        token_usage_complete=False,
                        cost_complete=False,
                        terminal_complete=True,
                        trace_complete=False,
                        backend_cleanup_complete=cleanup_error is None,
                        journal_cleanup_complete=True,
                        worktree_cleanup_complete=True,
                        error_kind=(
                            TOOL_MUTATION_INDETERMINATE_STOP_REASON
                            if mutation_terminal
                            else type(cleanup_error).__name__
                            if cleanup_error is not None
                            else type(exc).__name__
                        ),
                        error_message=(
                            (
                                f"{str(exc)}; provider cleanup remains pending: "
                                f"{type(cleanup_error).__name__}: {cleanup_error}"
                            )
                            if mutation_terminal and cleanup_error is not None
                            else str(exc)
                            if mutation_terminal
                            else str(
                                cleanup_error
                                if cleanup_error is not None
                                else exc
                            )
                        )[:2000],
                    )
                finally:
                    if managed is not None:
                        if mutation_terminal and not cleanup_attempted:
                            managed.close()
                        elif not mutation_terminal:
                            managed.close()
                            if not managed.closed:
                                managed.close()
                        if managed.closed:
                            with state_lock:
                                active.pop(lane_id, None)
                                factory_state_by_id[lane_id] = "settled"
                assert backend is not None
                assert managed is not None
                assert result is not None
                try:
                    return _result_outcome(
                        request,
                        scope=scope,
                        worktree=worktree,
                        debug_dir=debug_dir,
                        backend=backend,
                        result=result,
                        policy=policy,
                        cleanup_error=managed.close_error,
                    )
                except Exception as exc:  # noqa: BLE001 - malformed evidence
                    return WriterLaneOutcome(
                        lane_id=lane_id,
                        role=request.role,
                        route_role=request.route_role,
                        scope=scope,
                        status=(
                            "cleanup_failed" if not managed.closed else "failed"
                        ),
                        result=result,
                        backend_name=str(getattr(backend, "name", "unknown")),
                        model=(
                            str(backend.model)
                            if getattr(backend, "model", None) is not None
                            else None
                        ),
                        worktree=str(worktree),
                        debug_dir=str(debug_dir),
                        cost_usd=0.0,
                        input_tokens=0,
                        output_tokens=0,
                        token_usage_complete=False,
                        cost_complete=False,
                        terminal_complete=True,
                        trace_complete=False,
                        backend_cleanup_complete=managed.closed,
                        journal_cleanup_complete=True,
                        worktree_cleanup_complete=True,
                        error_kind=(
                            TOOL_MUTATION_INDETERMINATE_STOP_REASON
                            if mutation_terminal
                            else type(exc).__name__
                        ),
                        error_message=(
                            (
                                "source mutation settlement is indeterminate; "
                                f"secondary {type(exc).__name__}: {exc}"
                            )
                            if mutation_terminal
                            else str(exc)
                        )[:2000],
                    )

            executor = ThreadPoolExecutor(
                max_workers=min(policy.max_concurrency, len(requests)),
                thread_name_prefix="crucible-writer",
            )
            all_futures: tuple[Future[WriterLaneOutcome], ...] = ()

            def cancel_and_drain() -> None:
                nonlocal cancellation_started
                cancellation_started = True
                internal_cancel.set()
                for future in all_futures:
                    future.cancel()
                with state_lock:
                    active_snapshot = tuple(active.items())
                for lane_id, managed in active_snapshot:
                    if lane_id in tool_mutation_terminal_by_id:
                        continue
                    managed.close()

            def retry_active_cleanup() -> None:
                with state_lock:
                    active_snapshot = tuple(active.items())
                for lane_id, managed in active_snapshot:
                    if lane_id in tool_mutation_terminal_by_id:
                        continue
                    managed.close()
                    if managed.closed:
                        with state_lock:
                            if active.get(lane_id) is managed:
                                active.pop(lane_id, None)

            try:
                for request in requests:
                    futures_by_id[request.lane_id] = executor.submit(
                        run_lane,
                        request,
                    )
                all_futures = tuple(futures_by_id.values())
                pending: set[Future[WriterLaneOutcome]] = set(all_futures)
                request_by_id = {request.lane_id: request for request in requests}
                id_by_future = {
                    future: lane_id for lane_id, future in futures_by_id.items()
                }
                while pending:
                    if (
                        not cancellation_started
                        and cancel_event is not None
                        and cancel_event.is_set()
                    ):
                        cancel_and_drain()
                    done, still_pending = wait(
                        pending,
                        timeout=_POLL_SECONDS,
                        return_when=FIRST_COMPLETED,
                    )
                    pending = set(still_pending)
                    for future in done:
                        lane_id = id_by_future[future]
                        request = request_by_id[lane_id]
                        if future.cancelled():
                            outcome = _zero_outcome(
                                request,
                                scope=normalized_scopes[lane_id],
                                worktree=worktrees[lane_id],
                                debug_dir=debug_by_id[lane_id],
                                status="cancelled",
                                error_kind="SchedulerCancellation",
                                error_message="cancelled before worker start",
                            )
                        else:
                            outcome = future.result()
                        outcomes_by_id[lane_id] = outcome
                        if (
                            policy.fail_fast
                            and outcome.status != "succeeded"
                            and not cancellation_started
                        ):
                            cancel_and_drain()
            except BaseException:
                cancel_and_drain()
                wait(all_futures)
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
                all_joined = True
                retry_active_cleanup()
            with state_lock:
                backends_drained = not active
            for request in requests:
                if request.lane_id in outcomes_by_id:
                    continue
                future = futures_by_id[request.lane_id]
                if future.cancelled():
                    outcomes_by_id[request.lane_id] = _zero_outcome(
                        request,
                        scope=normalized_scopes[request.lane_id],
                        worktree=worktrees[request.lane_id],
                        debug_dir=debug_by_id[request.lane_id],
                        status="cancelled",
                        error_kind="SchedulerCancellation",
                        error_message="cancelled before worker start",
                    )
                else:
                    outcomes_by_id[request.lane_id] = future.result()

            # A cleanup retry may have positively settled an exception-only
            # mutation terminal after its lane returned.  Preserve the exact
            # TMI cause while making the recovered cleanup state authoritative.
            for request in requests:
                lane_id = request.lane_id
                outcome = outcomes_by_id[lane_id]
                managed = managed_backends_by_id.get(lane_id)
                if (
                    outcome.error_kind
                    == TOOL_MUTATION_INDETERMINATE_STOP_REASON
                    and managed is not None
                    and managed.closed
                    and not outcome.backend_cleanup_complete
                ):
                    outcomes_by_id[lane_id] = replace(
                        outcome,
                        status="failed",
                        backend_cleanup_complete=True,
                    )

            provider_generation_settled = (
                backends_drained and settle_provider_generation_if_safe()
            )
            writers_quiesced = (
                provider_generation_settled
                and settle_writer_epoch_if_safe()
            )
            if not writers_quiesced:
                mutation_primary = next(
                    (
                        tool_mutation_terminal_by_id[request.lane_id]
                        for request in requests
                        if request.lane_id in tool_mutation_terminal_by_id
                    ),
                    None,
                )
                aggregate_error = (
                    mutation_primary
                    or aggregate_error
                    or WriterSchedulerError(
                        "writer backend/provider close and join did not produce "
                        "durable epoch quiescence"
                    )
                )

            # Cost shares are authoritative even when a child later fails
            # scope or integration validation.
            for request in requests:
                outcome = outcomes_by_id[request.lane_id]
                reserved = _reserved_invocation(
                    request,
                    policy=policy,
                    parent_budget_usd=parent_budget_usd,
                    lane_count=len(requests),
                ).budget_usd
                if reserved is not None and outcome.cost_usd > reserved + 1e-12:
                    budget_message = (
                        f"lane cost ${outcome.cost_usd:.6f} exceeded "
                        f"reserved ${reserved:.6f}"
                    )
                    outcome = replace(
                        outcome,
                        status=(
                            outcome.status
                            if outcome.status == "cleanup_failed"
                            else "failed"
                        ),
                        error_kind=(
                            TOOL_MUTATION_INDETERMINATE_STOP_REASON
                            if outcome.error_kind
                            == TOOL_MUTATION_INDETERMINATE_STOP_REASON
                            else "LaneBudgetShareExceeded"
                        ),
                        error_message=(
                            f"{outcome.error_message}; secondary {budget_message}"
                            if outcome.error_kind
                            == TOOL_MUTATION_INDETERMINATE_STOP_REASON
                            else budget_message
                        )[:2000],
                    )
                    outcomes_by_id[request.lane_id] = outcome

            # Executor threads have joined, but an adapter can still own child
            # processes or transports until close succeeds.  Do not inspect or
            # mutate any writer workspace while such ownership remains live.
            if writers_quiesced:
                for request in requests:
                    lane_id = request.lane_id
                    outcome = outcomes_by_id[lane_id]
                    journal = lane_journals[lane_id]
                    try:
                        escapes = journal.detected_scope_escapes()
                        journal.assert_head_unchanged()
                        journal.assert_repository_metadata_unchanged()
                        journal.assert_index_unchanged()
                        if escapes:
                            raise MutationJournalError(
                                f"writer lane scope escape: {list(escapes)!r}"
                            )
                        patch, files, observed_base = capture_workspace_delta(
                            cwd=str(worktrees[lane_id]),
                            path_scope=normalized_scopes[lane_id],
                            max_diff_bytes=policy.max_lane_patch_bytes,
                        )
                        if observed_base != base_commit:
                            raise MutationJournalError(
                                "writer lane capture changed the immutable base"
                            )
                        if (
                            outcome.status == "succeeded"
                            and policy.require_nonempty_patch
                            and not patch
                        ):
                            raise WriterSchedulerError(
                                "writer lane produced an empty patch"
                            )
                        if outcome.status == "succeeded":
                            digest = "sha256:" + hashlib.sha256(patch).hexdigest()
                            outcome = replace(
                                outcome,
                                patch_bytes=len(patch),
                                patch_digest=digest,
                                files_modified=files,
                            )
                            patches_by_id[lane_id] = patch
                    except BaseException as exc:  # noqa: BLE001 - fail closed
                        outcome = _failed_outcome(outcome, error=exc)
                    outcomes_by_id[lane_id] = outcome
            else:
                aggregate_error = aggregate_error or WriterSchedulerError(
                    "writer workspace remains blocked by live or unproved "
                    "writer ownership"
                )

            # Always restore every speculative lane. A failed restoration
            # leaves its registry marker and worktree lease in place so later
            # recovery refuses to baseline or remove it.
            lane_restore_ok: dict[str, bool] = {}
            lane_cleanup_settled: dict[str, bool] = {}
            lane_restore_blocked = not writers_quiesced
            for request in reversed(requests):
                journal = lane_journals[request.lane_id]
                if lane_restore_blocked:
                    restored = False
                else:
                    try:
                        restored = journal.restore()
                    except BaseException:
                        restored = False
                lane_restore_ok[request.lane_id] = (
                    restored is True and not journal.cleanup_pending
                )
                lane_cleanup_settled[request.lane_id] = not journal.cleanup_pending
                if not lane_cleanup_settled[request.lane_id]:
                    # Each earlier-created lane captured this later lane's
                    # registry marker as topology. Preserve those markers until
                    # the failed later lane can retry its exact rollback.
                    lane_restore_blocked = True
                retain_journal(
                    f"lane {request.lane_id!r} mutation journal",
                    journal,
                )
            parent_restore_ok = False
            parent_cleanup_settled = parent_journal is None
            if parent_journal is not None:
                if all(lane_cleanup_settled.values()):
                    try:
                        parent_escapes = parent_journal.detected_scope_escapes()
                        if parent_escapes:
                            aggregate_error = MutationJournalError(
                                "base checkout changed during writer fanout: "
                                f"{list(parent_escapes)!r}"
                            )
                        parent_journal.assert_head_unchanged()
                        parent_journal.assert_repository_metadata_unchanged()
                        parent_journal.assert_index_unchanged()
                    except BaseException as exc:  # noqa: BLE001
                        aggregate_error = aggregate_error or exc
                    try:
                        restored = parent_journal.restore()
                    except BaseException:
                        restored = False
                    parent_restore_ok = (
                        restored is True and not parent_journal.cleanup_pending
                    )
                    parent_cleanup_settled = not parent_journal.cleanup_pending
                retain_journal("parent mutation journal", parent_journal)

            for request in requests:
                lane_id = request.lane_id
                if not lane_restore_ok.get(lane_id, False) or not parent_restore_ok:
                    outcomes_by_id[lane_id] = replace(
                        outcomes_by_id[lane_id],
                        status="cleanup_failed",
                        journal_cleanup_complete=False,
                        patch_bytes=0,
                        patch_digest=None,
                        files_modified=(),
                        error_kind=(
                            outcomes_by_id[lane_id].error_kind
                            or "MutationJournalCleanupFailed"
                        ),
                        error_message=(
                            outcomes_by_id[lane_id].error_message
                            or "writer mutation journal did not restore exactly"
                        ),
                    )
                    patches_by_id.pop(lane_id, None)

            if parent_cleanup_settled and all(lane_cleanup_settled.values()):
                for request in reversed(requests):
                    lane_id = request.lane_id
                    resolver = resolvers[lane_id]
                    close_failures: list[tuple[str, BaseException]] = []
                    for attempt in range(1, 3):
                        if not resolver.owned:
                            break
                        try:
                            resolver._close_locked()
                        except BaseException as cleanup_error:
                            close_failures.append(
                                (
                                    f"lane {lane_id!r} worktree attempt {attempt}",
                                    cleanup_error,
                                )
                            )
                    if close_failures:
                        if resolver.owned:
                            cleanup_summary: BaseException = WriterCleanupError(
                                primary_error=close_failures[0][1],
                                failures=tuple(close_failures),
                                retained_resolvers=((lane_id, resolver),),
                            )
                        else:
                            failure_detail = "; ".join(
                                f"{label}: {type(error).__name__}: {error}"
                                for label, error in close_failures
                            )
                            cleanup_summary = WriterSchedulerError(
                                "writer worktree cleanup encountered recovered "
                                f"failures and failed closed: {failure_detail}"
                            )
                        outcome = outcomes_by_id[lane_id]
                        outcomes_by_id[lane_id] = replace(
                            outcome,
                            status="cleanup_failed",
                            worktree_cleanup_complete=False,
                            patch_bytes=0,
                            patch_digest=None,
                            files_modified=(),
                            error_kind=(
                                TOOL_MUTATION_INDETERMINATE_STOP_REASON
                                if outcome.error_kind
                                == TOOL_MUTATION_INDETERMINATE_STOP_REASON
                                else type(cleanup_summary).__name__
                            ),
                            error_message=(
                                (
                                    f"{outcome.error_message}; secondary "
                                    f"{type(cleanup_summary).__name__}: "
                                    f"{cleanup_summary}"
                                )
                                if outcome.error_kind
                                == TOOL_MUTATION_INDETERMINATE_STOP_REASON
                                else str(cleanup_summary)
                            )[:2000],
                        )
                        patches_by_id.pop(lane_id, None)
            else:
                for request in requests:
                    lane_id = request.lane_id
                    outcomes_by_id[lane_id] = replace(
                        outcomes_by_id[lane_id],
                        status="cleanup_failed",
                        worktree_cleanup_complete=False,
                        patch_bytes=0,
                        patch_digest=None,
                        files_modified=(),
                    )
                    patches_by_id.pop(lane_id, None)

            if writers_quiesced:
                try:
                    _assert_exact_clean_base(repo, base_commit)
                    base_unchanged = True
                except BaseException as exc:  # noqa: BLE001
                    aggregate_error = aggregate_error or exc
                    base_unchanged = False
            else:
                base_unchanged = False

            ready_to_integrate = (
                not cancellation_started
                and aggregate_error is None
                and base_unchanged
                and all(
                    outcomes_by_id[request.lane_id].successful for request in requests
                )
                and all(not resolvers[request.lane_id].owned for request in requests)
            )
            if ready_to_integrate:
                integration_resolver = WorktreeResolver(
                    repo,
                    base_ref=base_commit,
                )
                integration_worktree: Path | None = None
                integration_journal: WorkspaceMutationJournal | None = None
                try:
                    integration_worktree = (
                        integration_resolver._ensure_worktree_locked()
                    )
                    integration_journal = WorkspaceMutationJournal.begin(
                        cwd=integration_worktree,
                        journal_root=journal_root / "integration",
                        slice_id="writer-serial-integration",
                        scope=tuple(
                            dict.fromkeys(
                                path
                                for request in requests
                                for path in normalized_scopes[request.lane_id]
                            )
                        ),
                        sequence=len(requests) + 1,
                        spec=None,
                    )
                    for request in requests:
                        _apply_patch(
                            integration_worktree,
                            patches_by_id[request.lane_id],
                            lane_id=request.lane_id,
                        )
                    integration_scope = tuple(
                        dict.fromkeys(
                            path
                            for request in requests
                            for path in normalized_scopes[request.lane_id]
                        )
                    )
                    pre_gate_patch, pre_gate_files, observed_base = (
                        capture_workspace_delta(
                            cwd=str(integration_worktree),
                            path_scope=integration_scope,
                            max_diff_bytes=policy.max_integrated_patch_bytes,
                        )
                    )
                    if observed_base != base_commit:
                        raise MutationJournalError(
                            "integration capture changed the immutable base"
                        )
                    if not pre_gate_patch:
                        raise WriterSchedulerError(
                            "serial integration produced an empty aggregate patch"
                        )
                    if integration_gate is None:
                        integration_gate_result = IntegrationGateResult(
                            success=True,
                            evidence_digest=hash_record(
                                {
                                    "schema_version": (
                                        "crucible.integration-gate-skipped.v1"
                                    ),
                                    "policy_digest": policy.digest,
                                    "base_commit": base_commit,
                                }
                            ),
                        )
                    else:
                        integration_epoch_quiesced = False
                        try:
                            integration_epoch_lease = arm_writer_epoch_family(
                                live_writer_epoch_family(integration_journal),
                                owner=capture_writer_epoch_owner(
                                    "writer_scheduler",
                                    integration_journal.transaction_id,
                                ),
                                boundary=WriterEpochBoundary(
                                    kind="writer_integration_gate",
                                    identifier=(integration_journal.transaction_id),
                                    settlement_kind="synchronous_callback",
                                    expected_owner_ids=(),
                                    expected_worker_ids=("integration_gate",),
                                ),
                            )
                        except WriterEpochArmError as error:
                            integration_epoch_lease = error.lease
                            integration_epoch_arm_error = error
                            raise
                        gate_candidate = integration_gate(
                            integration_worktree,
                            integration_scope,
                            requests,
                        )
                        if not isinstance(gate_candidate, IntegrationGateResult):
                            raise TypeError(
                                "integration_gate returned a non-IntegrationGateResult"
                            )
                        callback_receipt = synchronous_callback_settlement_receipt(
                            integration_epoch_lease,
                            settled_owner_ids=(),
                            joined_worker_ids=("integration_gate",),
                            evidence_digest=hash_record(
                                {
                                    "schema_version": (
                                        "crucible.writer-integration-gate-settlement.v1"
                                    ),
                                    "epoch_id": (integration_epoch_lease.epoch_id),
                                    "success": gate_candidate.success,
                                    "evidence_digest": (gate_candidate.evidence_digest),
                                    "error_kind": gate_candidate.error_kind,
                                }
                            ),
                            outcome=(
                                "succeeded" if gate_candidate.success else "failed"
                            ),
                        )
                        try:
                            quiesce_writer_epoch_family(
                                integration_epoch_lease,
                                callback_receipt,
                            )
                            require_writer_epoch_quiescence(
                                integration_epoch_lease.members,
                                allow_never_armed=False,
                            )
                        except WriterEpochQuiescenceError as error:
                            integration_epoch_quiescence_error = error
                            raise
                        integration_epoch_quiescence_error = None
                        integration_epoch_quiesced = True
                        integration_gate_result = gate_candidate
                    escapes = integration_journal.detected_scope_escapes()
                    integration_journal.assert_head_unchanged()
                    integration_journal.assert_repository_metadata_unchanged()
                    integration_journal.assert_index_unchanged()
                    if escapes:
                        raise MutationJournalError(
                            f"integration lane scope escape: {list(escapes)!r}"
                        )
                    if not integration_gate_result.success:
                        raise WriterSchedulerError(
                            "union integration gate failed: "
                            f"{integration_gate_result.error_kind}: "
                            f"{integration_gate_result.error_message or ''}"
                        )
                    integrated_patch, integrated_files, observed_base = (
                        capture_workspace_delta(
                            cwd=str(integration_worktree),
                            path_scope=integration_scope,
                            max_diff_bytes=policy.max_integrated_patch_bytes,
                        )
                    )
                    if observed_base != base_commit:
                        raise MutationJournalError(
                            "integration capture changed the immutable base"
                        )
                    if (
                        integrated_patch != pre_gate_patch
                        or integrated_files != pre_gate_files
                    ):
                        raise MutationJournalError(
                            "union integration gate mutated candidate source"
                        )
                except BaseException as exc:  # noqa: BLE001 - normalized result
                    if isinstance(exc, MutationJournalCleanupError):
                        integration_journal = exc.owner
                    aggregate_error = exc
                    integrated_patch = b""
                    integrated_files = ()
                finally:
                    creation_cleanup_error = (
                        aggregate_error
                        if isinstance(aggregate_error, WorktreeCleanupError)
                        else None
                    )
                    integration_failures: list[tuple[str, BaseException]] = []
                    journal_allows_worktree_removal = True
                    if isinstance(aggregate_error, WorktreeCleanupError):
                        integration_failures.extend(
                            (
                                "integration worktree creation cleanup",
                                cleanup_failure,
                            )
                            for cleanup_failure in aggregate_error.cleanup_errors
                        )
                    if integration_journal is not None:
                        if not integration_epoch_quiesced:
                            integration_failures.append(
                                (
                                    "integration mutation journal",
                                    integration_epoch_quiescence_error
                                    or MutationJournalError(
                                        "integration callback writer epoch is "
                                        "armed or lacks exact settlement"
                                    ),
                                )
                            )
                        else:
                            try:
                                journal_restored = integration_journal.restore()
                            except BaseException as cleanup_failure:
                                integration_failures.append(
                                    (
                                        "integration mutation journal",
                                        cleanup_failure,
                                    )
                                )
                            else:
                                if (
                                    journal_restored is not True
                                    or integration_journal.cleanup_pending
                                ):
                                    integration_failures.append(
                                        (
                                            "integration mutation journal",
                                            MutationJournalError(
                                                "integration mutation journal did "
                                                "not restore exactly"
                                            ),
                                        )
                                    )
                        journal_allows_worktree_removal = (
                            integration_epoch_quiesced
                            and not integration_journal.cleanup_pending
                        )
                        retain_journal(
                            "integration mutation journal",
                            integration_journal,
                        )
                    # ``_ensure_worktree_locked`` can create and retain its
                    # checkout before raising (for example, when publishing
                    # the post-add durable lease fails and its first removal
                    # attempt also fails).  In that case assignment to
                    # ``integration_worktree`` never happens, so the
                    # resolver's ownership state is the cleanup authority.
                    if integration_resolver.owned and journal_allows_worktree_removal:
                        for attempt in range(1, 3):
                            if not integration_resolver.owned:
                                break
                            try:
                                integration_resolver._close_locked()
                            except BaseException as cleanup_failure:
                                integration_failures.append(
                                    (
                                        f"integration worktree attempt {attempt}",
                                        cleanup_failure,
                                    )
                                )
                    if integration_resolver.owned and not integration_failures:
                        integration_failures.append(
                            (
                                "integration worktree",
                                RuntimeError(
                                    "integration resolver retained ownership "
                                    "without reporting a cleanup failure"
                                ),
                            )
                        )
                    if (
                        isinstance(
                            aggregate_error,
                            MutationJournalCleanupError,
                        )
                        and not aggregate_error.owner.cleanup_pending
                    ) or (
                        isinstance(
                            aggregate_error,
                            WorktreeCleanupError,
                        )
                        and not aggregate_error.owner.owned
                    ):
                        # Cleanup settled inside this lock scope, so do not
                        # leak a stale retry-owner exception to the caller.
                        aggregate_error = aggregate_error.primary_error
                    if integration_failures:
                        retained_integration_journals = retained_journal_snapshot()
                        if (
                            creation_cleanup_error is not None
                            and integration_resolver.owned
                            and journal_allows_worktree_removal
                        ):
                            integration_cleanup_summary: BaseException = (
                                WorktreeCleanupError(
                                    owner=integration_resolver,
                                    primary_error=(
                                        creation_cleanup_error.primary_error
                                    ),
                                    cleanup_errors=tuple(
                                        error for _, error in integration_failures
                                    ),
                                )
                            )
                        elif (
                            retained_integration_journals or integration_resolver.owned
                        ):
                            integration_cleanup_summary = WriterCleanupError(
                                primary_error=(
                                    aggregate_error
                                    if aggregate_error is not None
                                    else integration_failures[0][1]
                                ),
                                failures=tuple(integration_failures),
                                retained_journals=retained_integration_journals,
                                retained_resolvers=(
                                    (("integration", integration_resolver),)
                                    if integration_resolver.owned
                                    else ()
                                ),
                            )
                        else:
                            failure_detail = "; ".join(
                                f"{label}: {type(error).__name__}: {error}"
                                for label, error in integration_failures
                            )
                            integration_cleanup_summary = WriterSchedulerError(
                                "integration cleanup encountered recovered "
                                f"failures and failed closed: {failure_detail}"
                            )
                        aggregate_error = integration_cleanup_summary
                        integration_cleanup_complete = False
                    else:
                        integration_cleanup_complete = not integration_resolver.owned
                    if not integration_cleanup_complete:
                        integrated_patch = b""
                        integrated_files = ()
            elif cancellation_started:
                aggregate_error = aggregate_error or WriterSchedulerError(
                    "writer fanout cancelled before integration"
                )
            elif aggregate_error is None:
                aggregate_error = WriterSchedulerError(
                    "one or more writer lanes failed before integration"
                )
        except BaseException as exc:
            # Admission after worktree creation still has to unwind every
            # armed transaction and disposable checkout before surfacing.
            aggregate_error = aggregate_error or exc
            cleanup_failures: list[tuple[str, BaseException]] = []
            if isinstance(exc, WorktreeCleanupError):
                cleanup_failures.extend(
                    (
                        "worktree creation cleanup",
                        cleanup_error,
                    )
                    for cleanup_error in exc.cleanup_errors
                )
            internal_cancel.set()
            epoch_arm_error: WriterEpochArmError | None = None
            if isinstance(exc, WriterEpochArmError):
                writer_epoch_lease = exc.lease
                try:
                    writer_epoch_lease = exc.retry_arm()
                except WriterEpochArmError as error:
                    epoch_arm_error = error
                    cleanup_failures.append(("writer epoch arm", error))
            with state_lock:
                active_snapshot = tuple(active.items())
            for lane_id, managed in active_snapshot:
                managed.close()
                if not managed.closed:
                    managed.close()
                if managed.close_error is not None:
                    cleanup_failures.append(
                        (
                            f"lane {lane_id!r} backend",
                            managed.close_error,
                        )
                    )
            with state_lock:
                settled_ids = tuple(
                    lane_id for lane_id, managed in active.items() if managed.closed
                )
                for lane_id in settled_ids:
                    active.pop(lane_id, None)
                    factory_state_by_id[lane_id] = "settled"
                retained_backend_snapshot = tuple(active.items())
            backends_drained = not retained_backend_snapshot
            provider_generation_settled = (
                backends_drained and settle_provider_generation_if_safe()
            )
            if not provider_generation_settled:
                cleanup_failures.append(
                    (
                        "provider generation",
                        (
                            provider_generation.close_error
                            if provider_generation is not None
                            else None
                        )
                        or RuntimeError(
                            "provider generation retained cleanup authority"
                        ),
                    )
                )
            writer_authority_settled = (
                provider_generation_settled
                and epoch_arm_error is None
                and settle_writer_epoch_if_safe()
            )
            if not writer_authority_settled and writer_epoch_lease is not None:
                cleanup_failures.append(
                    (
                        "writer epoch quiescence",
                        writer_epoch_quiescence_error
                        or epoch_arm_error
                        or MutationJournalError(
                            "writer epoch lacks exact close/join settlement"
                        ),
                    )
                )
            lane_journals_settled = writer_authority_settled
            for lane_id, journal in reversed(tuple(lane_journals.items())):
                label = f"lane {lane_id!r} mutation journal"
                if not writer_authority_settled:
                    cleanup_failures.append(
                        (
                            label,
                            MutationJournalError(
                                "lane mutation journal cleanup is deferred until "
                                "every writer backend closes, every worker joins, "
                                "and epoch quiescence is durable"
                            ),
                        )
                    )
                elif not lane_journals_settled:
                    cleanup_failures.append(
                        (
                            label,
                            MutationJournalError(
                                "lane mutation journal cleanup is deferred until "
                                "every later-created lane journal settles"
                            ),
                        )
                    )
                else:
                    try:
                        restored = journal.restore()
                    except BaseException as cleanup_error:
                        cleanup_failures.append(
                            (
                                label,
                                cleanup_error,
                            )
                        )
                        if journal.cleanup_pending:
                            lane_journals_settled = False
                    else:
                        if restored is not True or journal.cleanup_pending:
                            cleanup_failures.append(
                                (
                                    label,
                                    MutationJournalError(
                                        "lane mutation journal did not restore exactly"
                                    ),
                                )
                            )
                            if journal.cleanup_pending:
                                lane_journals_settled = False
                retain_journal(label, journal)
            if parent_journal is not None:
                if not writer_authority_settled:
                    cleanup_failures.append(
                        (
                            "parent mutation journal",
                            MutationJournalError(
                                "parent mutation journal cleanup is deferred until "
                                "every writer backend closes, every worker joins, "
                                "and epoch quiescence is durable"
                            ),
                        )
                    )
                elif lane_journals_settled:
                    try:
                        restored = parent_journal.restore()
                    except BaseException as cleanup_error:
                        cleanup_failures.append(
                            (
                                "parent mutation journal",
                                cleanup_error,
                            )
                        )
                    else:
                        if restored is not True or parent_journal.cleanup_pending:
                            cleanup_failures.append(
                                (
                                    "parent mutation journal",
                                    MutationJournalError(
                                        "parent mutation journal did not restore "
                                        "exactly"
                                    ),
                                )
                            )
                else:
                    cleanup_failures.append(
                        (
                            "parent mutation journal",
                            MutationJournalError(
                                "parent mutation journal cleanup is deferred until "
                                "every lane journal settles"
                            ),
                        )
                    )
                retain_journal("parent mutation journal", parent_journal)
            retained_journal_owners = retained_journal_snapshot()
            retained_resolvers: list[tuple[str, WorktreeResolver]] = []
            for lane_id, resolver in reversed(tuple(resolvers.items())):
                if retained_journal_owners:
                    if resolver.owned:
                        retained_resolvers.append((lane_id, resolver))
                    else:
                        resolvers.pop(lane_id, None)
                    continue
                for attempt in range(1, 3):
                    if not resolver.owned:
                        break
                    try:
                        resolver._close_locked()
                    except BaseException as cleanup_error:
                        cleanup_failures.append(
                            (
                                f"lane {lane_id!r} worktree attempt {attempt}",
                                cleanup_error,
                            )
                        )
                if not resolver.owned:
                    resolvers.pop(lane_id, None)
                else:
                    retained_resolvers.append((lane_id, resolver))
            lifecycle_cleanup_error: WriterCleanupError | None = None
            live_cleanup_ownership = bool(
                retained_journal_owners
                or retained_resolvers
                or retained_backend_snapshot
                or (
                    provider_generation is not None
                    and not provider_generation.closed
                )
                or (
                    writer_epoch_lease is not None
                    and any(
                        member.active is True for member in writer_epoch_lease.members
                    )
                )
            )
            unwind_primary_error = (
                exc.primary_error
                if (
                    (
                        isinstance(exc, MutationJournalCleanupError)
                        and not exc.owner.cleanup_pending
                    )
                    or (isinstance(exc, WorktreeCleanupError) and not exc.owner.owned)
                    or (
                        isinstance(exc, WriterProviderGenerationError)
                        and exc.owner.closed
                    )
                )
                else exc
            )
            if live_cleanup_ownership:
                if not cleanup_failures:
                    cleanup_failures.append(
                        (
                            "writer outer unwind",
                            RuntimeError("writer lifecycle retained cleanup ownership"),
                        )
                    )
                lifecycle_cleanup_error = WriterCleanupError(
                    primary_error=unwind_primary_error,
                    failures=tuple(cleanup_failures),
                    retained_journals=retained_journal_owners,
                    retained_resolvers=tuple(retained_resolvers),
                    retained_backends=retained_backend_snapshot,
                    retained_provider_generation=provider_generation,
                    retained_epoch_lease=writer_epoch_lease,
                    epoch_arm_error=epoch_arm_error,
                    epoch_quiescence_error=(writer_epoch_quiescence_error),
                    epoch_settlement_factory=(
                        build_writer_epoch_settlement_receipt
                        if (
                            writer_epoch_lease is not None
                            and epoch_arm_error is None
                            and not writer_epoch_quiesced
                        )
                        else None
                    ),
                )
                aggregate_error = lifecycle_cleanup_error
                integration_cleanup_complete = False
            managed_by_id = dict(active_snapshot)
            if futures_by_id:
                # Harvest every joined future before constructing a retained
                # cleanup owner. FIRST_COMPLETED returns an unordered set, so a
                # fatal child must not erase a sibling's terminal usage merely
                # because it happened to be visited first.
                for request in requests:
                    lane_id = request.lane_id
                    if lane_id in outcomes_by_id:
                        continue
                    missing_future = futures_by_id.get(lane_id)
                    fatal_error: BaseException = exc
                    if missing_future is not None:
                        if missing_future.cancelled():
                            outcomes_by_id[lane_id] = _zero_outcome(
                                request,
                                scope=normalized_scopes[lane_id],
                                worktree=worktrees[lane_id],
                                debug_dir=debug_by_id[lane_id],
                                status="cancelled",
                                error_kind="SchedulerCancellation",
                                error_message=("cancelled during fatal sibling unwind"),
                            )
                            continue
                        try:
                            completed_outcome = missing_future.result()
                        except BaseException as future_error:
                            fatal_error = future_error
                        else:
                            if isinstance(
                                completed_outcome,
                                WriterLaneOutcome,
                            ):
                                outcomes_by_id[lane_id] = completed_outcome
                                continue
                            fatal_error = TypeError(
                                "writer worker returned a non-WriterLaneOutcome"
                            )
                    fatal_managed = managed_by_id.get(lane_id)
                    lane_journal = lane_journals.get(lane_id)
                    outcomes_by_id[lane_id] = _fatal_outcome(
                        request,
                        scope=normalized_scopes[lane_id],
                        worktree=worktrees[lane_id],
                        debug_dir=debug_by_id[lane_id],
                        backend=(
                            fatal_managed.backend if fatal_managed is not None else None
                        ),
                        error=fatal_error,
                        backend_cleanup_complete=(
                            fatal_managed is None or fatal_managed.closed
                        ),
                        journal_cleanup_complete=(
                            (lane_journal is None or not lane_journal.cleanup_pending)
                            and (
                                parent_journal is None
                                or not parent_journal.cleanup_pending
                            )
                        ),
                        worktree_cleanup_complete=(
                            lane_id not in resolvers or not resolvers[lane_id].owned
                        ),
                    )

            # Fully settled setup/fatal failures retain their original exception
            # type. Only live cleanup ownership is converted into the typed
            # caller-retry contract; submitted child evidence is attached below.
            if lifecycle_cleanup_error is None:
                if unwind_primary_error is not exc:
                    raise unwind_primary_error from exc
                raise
            if not futures_by_id:
                raise lifecycle_cleanup_error from exc

        parent_journal_clean = (
            parent_journal is None or not parent_journal.cleanup_pending
        )
        for request in requests:
            lane_id = request.lane_id
            outcome = outcomes_by_id[lane_id]
            lane_journal = lane_journals.get(lane_id)
            active_managed = active.get(lane_id)
            backend_cleanup_complete = outcome.backend_cleanup_complete and (
                active_managed is None or active_managed.closed
            )
            journal_cleanup_complete = (
                outcome.journal_cleanup_complete
                and parent_journal_clean
                and (lane_journal is None or not lane_journal.cleanup_pending)
            )
            outcome_resolver = resolvers.get(lane_id)
            worktree_cleanup_complete = outcome.worktree_cleanup_complete and (
                outcome_resolver is None or not outcome_resolver.owned
            )
            if (
                backend_cleanup_complete
                and journal_cleanup_complete
                and worktree_cleanup_complete
            ):
                continue
            outcomes_by_id[lane_id] = replace(
                outcome,
                status="cleanup_failed",
                backend_cleanup_complete=backend_cleanup_complete,
                journal_cleanup_complete=journal_cleanup_complete,
                worktree_cleanup_complete=worktree_cleanup_complete,
                patch_bytes=0,
                patch_digest=None,
                files_modified=(),
                error_kind=(outcome.error_kind or "WriterLifecycleCleanupIncomplete"),
                error_message=(
                    outcome.error_message or "writer lane retained cleanup authority"
                ),
            )
        ordered_outcomes = tuple(
            outcomes_by_id[request.lane_id] for request in requests
        )
        with state_lock:
            backends_drained = not active
        provider_generation_settled = (
            provider_generation is None or provider_generation.closed
        )
        if (
            backends_drained
            and provider_generation_settled
            and writer_epoch_quiesced
            and integration_epoch_quiesced
        ):
            try:
                _assert_exact_clean_base(repo, base_commit)
                base_unchanged = True
            except BaseException as exc:  # noqa: BLE001
                aggregate_error = aggregate_error or exc
                base_unchanged = False
                integrated_patch = b""
                integrated_files = ()
        else:
            base_unchanged = False
            integrated_patch = b""
            integrated_files = ()

        if (
            aggregate_error is None
            and integrated_patch
            and integration_cleanup_complete
            and base_unchanged
            and all(outcome.successful for outcome in ordered_outcomes)
        ):
            aggregate_status: WriterAggregateStatus = "succeeded"
        elif not integration_cleanup_complete or any(
            outcome.status == "cleanup_failed" for outcome in ordered_outcomes
        ):
            aggregate_status = "cleanup_failed"
            integrated_patch = b""
            integrated_files = ()
        elif cancellation_started or any(
            outcome.status == "cancelled" for outcome in ordered_outcomes
        ):
            aggregate_status = "cancelled"
            integrated_patch = b""
            integrated_files = ()
        elif all(outcome.status == "succeeded" for outcome in ordered_outcomes):
            aggregate_status = "integration_failed"
            integrated_patch = b""
            integrated_files = ()
        else:
            aggregate_status = "failed"
            integrated_patch = b""
            integrated_files = ()

        events: list[BackendExecutionEvent] = []
        for outcome in ordered_outcomes:
            events.append(
                BackendExecutionEvent(
                    sequence=len(events),
                    kind="child_start",
                    name=outcome.role,
                    parent_id=outcome.route_role,
                    child_id=outcome.lane_id,
                )
            )
            events.append(
                BackendExecutionEvent(
                    sequence=len(events),
                    kind="child_terminal",
                    name=outcome.status,
                    parent_id=outcome.route_role,
                    child_id=outcome.lane_id,
                )
            )
        events.append(
            BackendExecutionEvent(
                sequence=len(events),
                kind="terminal",
                name=aggregate_status,
            )
        )
        usage = Usage(
            input_tokens=sum(outcome.input_tokens for outcome in ordered_outcomes),
            output_tokens=sum(outcome.output_tokens for outcome in ordered_outcomes),
            cost_usd=sum(outcome.cost_usd for outcome in ordered_outcomes),
            cache_creation_tokens=sum(
                outcome.cache_creation_tokens for outcome in ordered_outcomes
            ),
            cache_read_tokens=sum(
                outcome.cache_read_tokens for outcome in ordered_outcomes
            ),
        )
        fanout_result = WriterFanoutResult(
            children=ordered_outcomes,
            usage=usage,
            trace=BackendExecutionTrace(
                backend="crucible_writer_scheduler",
                requested_model="backend-neutral",
                events=tuple(events),
            ),
            status=aggregate_status,
            policy_digest=policy.digest,
            base_commit=base_commit,
            integrated_patch=integrated_patch,
            integrated_files=integrated_files,
            integration_gate=integration_gate_result,
            all_joined=all_joined,
            cancellation_drained=(
                not active
                and provider_generation_settled
                and writer_epoch_quiesced
                and integration_epoch_quiesced
            ),
            base_unchanged=base_unchanged,
            integration_cleanup_complete=integration_cleanup_complete,
            error_kind=(
                type(aggregate_error).__name__ if aggregate_error is not None else None
            ),
            error_message=(
                str(aggregate_error)[:2000] if aggregate_error is not None else None
            ),
        )
        with state_lock:
            retained_backends = tuple(active.items())
        final_retained_resolvers: list[tuple[str, WorktreeResolver]] = []
        seen_resolvers: set[int] = set()
        for lane_id, resolver in resolvers.items():
            if resolver.owned:
                final_retained_resolvers.append((lane_id, resolver))
                seen_resolvers.add(id(resolver))
        if (
            integration_resolver is not None
            and integration_resolver.owned
            and id(integration_resolver) not in seen_resolvers
        ):
            final_retained_resolvers.append(
                (
                    "integration",
                    integration_resolver,
                )
            )
        final_retained_journals = retained_journal_snapshot()
        retained_epoch_lease: WriterEpochLease | None = None
        retained_epoch_arm_error: WriterEpochArmError | None = None
        retained_epoch_quiescence_error: WriterEpochQuiescenceError | None = None
        retained_epoch_settlement_factory: (
            Callable[[], WriterSettlementReceipt] | None
        ) = None
        for (
            candidate_lease,
            candidate_arm_error,
            candidate_quiescence_error,
            candidate_settlement_factory,
        ) in (
            (
                integration_epoch_lease,
                integration_epoch_arm_error,
                integration_epoch_quiescence_error,
                None,
            ),
            (
                writer_epoch_lease,
                None,
                writer_epoch_quiescence_error,
                (
                    build_writer_epoch_settlement_receipt
                    if not writer_epoch_quiesced
                    else None
                ),
            ),
        ):
            if candidate_lease is not None and any(
                member.active is True for member in candidate_lease.members
            ):
                retained_epoch_lease = candidate_lease
                retained_epoch_arm_error = candidate_arm_error
                retained_epoch_quiescence_error = candidate_quiescence_error
                retained_epoch_settlement_factory = candidate_settlement_factory
                break
        if (
            retained_backends
            or not provider_generation_settled
            or final_retained_journals
            or final_retained_resolvers
            or retained_epoch_lease is not None
        ):
            mutation_primary = next(
                (
                    tool_mutation_terminal_by_id[request.lane_id]
                    for request in requests
                    if request.lane_id in tool_mutation_terminal_by_id
                ),
                None,
            )
            if mutation_primary is not None:
                aggregate_error = mutation_primary
            final_cleanup_failures: list[tuple[str, BaseException]] = []
            for lane_id, managed in retained_backends:
                final_cleanup_failures.append(
                    (
                        f"lane {lane_id!r} backend",
                        managed.close_error
                        or RuntimeError(
                            "backend retained ownership after cleanup retries"
                        ),
                    )
                )
            if not provider_generation_settled:
                final_cleanup_failures.append(
                    (
                        "provider generation",
                        (
                            provider_generation.close_error
                            if provider_generation is not None
                            else None
                        )
                        or RuntimeError(
                            "provider generation retained cleanup authority"
                        ),
                    )
                )
            if retained_epoch_lease is not None:
                final_cleanup_failures.append(
                    (
                        "writer epoch",
                        retained_epoch_quiescence_error
                        or retained_epoch_arm_error
                        or MutationJournalError(
                            "writer epoch retained active authority without "
                            "durable settlement"
                        ),
                    )
                )
            for label, _journal in final_retained_journals:
                final_cleanup_failures.append(
                    (
                        label,
                        RuntimeError("mutation journal retained cleanup authority"),
                    )
                )
            for label, resolver in final_retained_resolvers:
                retained_outcome = outcomes_by_id.get(label)
                final_cleanup_failures.append(
                    (
                        f"{label} worktree",
                        (
                            aggregate_error
                            if isinstance(
                                aggregate_error,
                                (WorktreeCleanupError, WriterCleanupError),
                            )
                            else RuntimeError(retained_outcome.error_message)
                            if (
                                retained_outcome is not None
                                and retained_outcome.error_message
                            )
                            else RuntimeError(
                                "worktree retained ownership after cleanup retries: "
                                f"worktree={resolver.worktree_path} "
                                f"intent={resolver.intent_path} "
                                f"lease={resolver.lease_path}"
                            )
                        ),
                    )
                )
            ownership_error = WriterCleanupError(
                primary_error=(
                    aggregate_error
                    if aggregate_error is not None
                    else final_cleanup_failures[0][1]
                ),
                failures=tuple(final_cleanup_failures),
                retained_journals=final_retained_journals,
                retained_resolvers=tuple(final_retained_resolvers),
                retained_backends=retained_backends,
                retained_provider_generation=provider_generation,
                retained_epoch_lease=retained_epoch_lease,
                epoch_arm_error=retained_epoch_arm_error,
                epoch_quiescence_error=(retained_epoch_quiescence_error),
                epoch_settlement_factory=(retained_epoch_settlement_factory),
                partial_result=fanout_result,
            )
            if isinstance(aggregate_error, ToolMutationIndeterminate):
                raise aggregate_error from ownership_error
            raise ownership_error from aggregate_error
        return fanout_result


__all__ = [
    "BackendFactory",
    "IntegrationGate",
    "IntegrationGateResult",
    "ProviderGenerationFactory",
    "ToolSetFactory",
    "WriterAggregateStatus",
    "WriterFanoutResult",
    "WriterIntegrationConflict",
    "WriterIsolationError",
    "WriterLaneOutcome",
    "WriterLaneRequest",
    "WriterLaneStatus",
    "WriterCleanupError",
    "WriterProviderGenerationError",
    "WriterSchedulerError",
    "run_isolated_writer_fanout",
]
