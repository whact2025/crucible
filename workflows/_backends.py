# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Per-workflow backend lifecycle helper.

``backend_pool(args, log_dir)`` is the canonical pattern every workflow
``main()`` uses to acquire backends — multi-role workflows (breach,
goal, scaling, summary, solutioning, regression, architect) ask the
pool for each role in turn, and the pool guarantees lifecycle cleanup
on context exit (no per-role try/finally boilerplate, no leaks on
early return).

Packages the whole lifecycle:

- Loads routing once (cache hit for every subsequent role).
- Lazily constructs + caches per-role backends on first ``.get(role)``.
- Optionally attaches a toolset at construction. The cache key also tracks
  capability and native-shell policy, so a confined container role cannot
  reuse a host-shell-enabled instance.
- Closes every constructed backend on context exit, unconditionally.

Usage::

    with open_run_with_capture(corpus, "<workflow>") as handle:
        with backend_pool(args, handle.run_dir) as pool:
            red    = pool.get("goal_red")
            blue   = pool.get("goal_blue")
            judge  = pool.get("goal_judge")
            ...

For workflows that need per-instance backends (swebench's
``--implement`` loop builds one backend per repo workspace with a
different cwd each time), don't use the pool — the cache-by-role model
would return the wrong backend after the first instance. Build via
``resolve_for_role`` directly for that case and document why.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, cast

from assay import ConversationMode

from crucible.backends.config import (
    load_routing,
    resolve_for_role,
    routing_feature_fingerprint,
    routing_fingerprint,
    routing_source_fingerprint,
)
from crucible.backends.capabilities import (
    CapabilityProfile,
    normalize_capability,
    toolset_write_names,
    validate_toolset_capability,
)
from crucible.backends.features import FeatureRequirement
from crucible.backends.limits import (
    INVOCATION_LIMIT_LEDGER_LEAF,
    InvocationLimitContractError,
    InvocationLimitLedger,
    InvocationLimitMode,
    InvocationLimitPolicy,
    InvocationLimitRequest,
    InvocationLimitState,
)
from crucible.workflows import _outcome
from crucible.workflows._development_budget import DevelopmentBudgetLedger


_MAX_SETTLEMENT_INTERRUPTS = 2


def _conversation_feature_requirements(
    mode: ConversationMode,
) -> tuple[FeatureRequirement, ...]:
    """Translate a requested handle mode to pre-transport route admission."""

    if mode == "none":
        return ()
    if mode == "retry_only":
        return (
            FeatureRequirement(
                "session_continuity",
                frozenset({"retry_only", "workflow_bound"}),
            ),
        )
    if mode == "workflow_bound":
        return (
            FeatureRequirement(
                "session_continuity",
                frozenset({"workflow_bound"}),
            ),
        )
    raise ValueError(f"unsupported conversation mode {mode!r}")


class BackendCleanupError(RuntimeError):
    """One or more owned backends could not be released."""

    def __init__(
        self,
        failures: tuple[tuple[str, BaseException], ...],
        *,
        retry_cleanup: Callable[[], None],
    ) -> None:
        if not failures:
            raise ValueError("BackendCleanupError requires a failure")
        if not callable(retry_cleanup):
            raise TypeError(
                "BackendCleanupError requires a retry cleanup callback"
            )
        self.failures = failures
        self._retry_cleanup = retry_cleanup
        details = "; ".join(
            f"{label}: {type(error).__name__}: {error}"
            for label, error in failures
        )
        super().__init__(
            f"backend lifecycle cleanup failed ({len(failures)}): {details}"
        )

    def retry_cleanup(self) -> None:
        """Retry the still-owned backend cleanup or raise its replacement."""

        self._retry_cleanup()


class BackendSettlementInterrupted(KeyboardInterrupt):
    """Cancellation replay proving that every backend already settled."""

    def __init__(
        self,
        owner: object,
        interrupt: KeyboardInterrupt,
    ) -> None:
        self.owner = owner
        self.interrupt = interrupt
        super().__init__(*interrupt.args)


class BackendLifecycleError(RuntimeError):
    """A write-capable backend cannot prove descendant settlement."""


class _BackendPool:
    """Lazy per-role backend cache with collective teardown.

    Constructed and torn down by :func:`backend_pool`; callers
    shouldn't instantiate directly.
    """

    def __init__(
        self,
        routing: Any,
        *,
        log_dir: Path | str | None,
        cwd: str | None,
        limit_policy: InvocationLimitPolicy | None = None,
        limit_state: InvocationLimitState | None = None,
        limit_ledger: InvocationLimitLedger | None = None,
        development_budget_ledger: DevelopmentBudgetLedger | None = None,
    ) -> None:
        self.routing = routing
        self._log_dir = log_dir
        self._cwd = cwd
        self._limit_policy = limit_policy
        if limit_state is not None and limit_policy is None:
            raise ValueError("limit_state requires limit_policy")
        self._limit_state = (
            limit_state
            if limit_state is not None
            else (InvocationLimitState() if limit_policy is not None else None)
        )
        if limit_ledger is not None and type(limit_ledger) is not InvocationLimitLedger:
            raise TypeError("limit_ledger must be InvocationLimitLedger or None")
        self._limit_ledger = limit_ledger
        self._development_budget_ledger = development_budget_ledger
        # Toolset-attached/plain, capability, and native-shell variants for
        # the same role must remain distinct instances. In particular, a
        # container-confined role must never reuse a host-shell-enabled
        # backend from another invocation.
        self._cache: dict[
            tuple[
                str,
                bool,
                CapabilityProfile,
                bool | None,
                tuple[tuple[str, tuple[str, ...]], ...],
                ConversationMode,
                InvocationLimitRequest | None,
            ],
            Any,
        ] = {}
        self._lifecycle_lock = threading.RLock()
        self._cleanup_authorities: dict[
            int,
            tuple[str, Callable[[], None] | None],
        ] = {}
        # Cleanup binding alone need not retain the backend (a custom adapter
        # may expose a static callable).  Keep the exact object from the first
        # successful bind until cache ownership or caller ownership is
        # positively established, and retain it across every failed unwind.
        self._provisional_backends: dict[int, Any] = {}
        self._cache_owner_ids: dict[
            tuple[
                str,
                bool,
                CapabilityProfile,
                bool | None,
                tuple[tuple[str, tuple[str, ...]], ...],
                ConversationMode,
                InvocationLimitRequest | None,
            ],
            int,
        ] = {}
        self._pending_cleanup: dict[
            int,
            tuple[str, Callable[[], None]],
        ] = {}
        # A backend exists as soon as route resolution returns, before its
        # cleanup descriptor has been proven safe to bind.  Retain that exact
        # object when descriptor binding fails: otherwise provider-context
        # unwind could see an apparently empty pool, quiesce the writer epoch,
        # and run mutation callbacks while a constructed writer resource has
        # no positive settlement proof.
        self._unbound_cleanup: dict[
            int,
            tuple[str, Any, bool],
        ] = {}
        self._cleanup_errors: dict[
            int,
            tuple[str, BaseException],
        ] = {}
        self._deferred_interrupt: KeyboardInterrupt | None = None

    def _raise_if_cleanup_pending(self) -> None:
        with self._lifecycle_lock:
            failures = tuple(self._cleanup_errors.values())
        if failures:
            raise BackendCleanupError(
                failures,
                retry_cleanup=lambda: self.close(_retry=True),
            )

    def _cleanup_failed_setup(
        self,
        backend: Any,
        *,
        label: str,
        primary_error: BaseException,
        close: Callable[[], None] | None,
    ) -> None:
        backend_id = id(backend)
        if close is None:
            with self._lifecycle_lock:
                self._provisional_backends.pop(backend_id, None)
                self._cleanup_authorities.pop(backend_id, None)
                self._pending_cleanup.pop(backend_id, None)
                self._cleanup_errors.pop(backend_id, None)
            return
        try:
            close()
        except KeyboardInterrupt as cleanup_error:
            # Setup has already transferred this backend into provisional
            # ownership. Retain the exact close callable and cancellation so
            # the surrounding pool context can retry settlement immediately
            # and replay SIGINT only after that retry succeeds.
            cleanup_error.add_note(
                f"{label} setup also failed: "
                f"{type(primary_error).__name__}: {primary_error}"
            )
            deferred_interrupt = (
                cleanup_error.interrupt
                if isinstance(
                    cleanup_error,
                    BackendSettlementInterrupted,
                )
                else cleanup_error
            )
            with self._lifecycle_lock:
                self._pending_cleanup[backend_id] = (label, close)
                self._cleanup_errors[backend_id] = (
                    label,
                    cleanup_error,
                )
                self._deferred_interrupt = (
                    self._deferred_interrupt or deferred_interrupt
                )
            raise
        except BaseException as cleanup_error:
            with self._lifecycle_lock:
                self._pending_cleanup[backend_id] = (label, close)
                self._cleanup_errors[backend_id] = (
                    label,
                    cleanup_error,
                )
            raise BackendCleanupError(
                (
                    (f"{label} setup", primary_error),
                    (f"{label} cleanup", cleanup_error),
                ),
                retry_cleanup=lambda: self.close(_retry=True),
            ) from primary_error
        with self._lifecycle_lock:
            self._provisional_backends.pop(backend_id, None)
            self._cleanup_authorities.pop(backend_id, None)
            self._pending_cleanup.pop(backend_id, None)
            self._cleanup_errors.pop(backend_id, None)

    def _own_bound_backend_setup(
        self,
        backend: Any,
        *,
        label: str,
        close: Callable[[], None] | None,
    ) -> None:
        """Retain a just-bound backend before any adapter descriptor access."""

        backend_id = id(backend)
        with self._lifecycle_lock:
            self._provisional_backends[backend_id] = backend
            self._cleanup_authorities[backend_id] = (label, close)

    def _relinquish_bound_backend_setup(self, backend: Any) -> None:
        """Transfer an exclusive backend to its caller after setup succeeds."""

        backend_id = id(backend)
        with self._lifecycle_lock:
            self._provisional_backends.pop(backend_id, None)
            self._cleanup_authorities.pop(backend_id, None)
            self._pending_cleanup.pop(backend_id, None)
            self._cleanup_errors.pop(backend_id, None)

    @staticmethod
    def _bind_backend_cleanup(
        backend: Any,
        *,
        label: str,
        required: bool,
    ) -> Callable[[], None] | None:
        """Bind cleanup authority once, before ToolSet attach or dispatch.

        For a write-capable route, ``close`` is Crucible's explicit adapter
        contract: it must terminate, drain, and join every provider/tool
        descendant, and raise while any descendant remains live.  Caching the
        bound callable here prevents a later mutable/raising descriptor from
        escaping lifecycle ownership during cleanup.
        """

        try:
            close = getattr(backend, "close", None)
        except BaseException as exc:  # noqa: BLE001 - reject unsafe descriptor
            raise BackendLifecycleError(
                f"{label} close authority could not be bound before dispatch: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if callable(close):
            return close
        if required:
            raise BackendLifecycleError(
                f"{label} resolved a workspace-write backend without callable "
                "close(); Crucible cannot prove provider/tool descendant "
                "settlement and refuses dispatch"
            )
        return None

    def _bind_or_retain_backend_cleanup(
        self,
        backend: Any,
        *,
        label: str,
        required: bool,
    ) -> Callable[[], None] | None:
        """Bind cleanup or poison this pool with the exact backend object.

        A failed initial binding is retryable only through :meth:`close`.
        Keeping the object itself (rather than merely its ``id`` or the
        binding exception) prevents collection from masquerading as resource
        settlement and blocks provider-generation quiescence until a later
        binding plus cleanup succeeds.
        """

        try:
            return self._bind_backend_cleanup(
                backend,
                label=label,
                required=required,
            )
        except BaseException as error:  # noqa: BLE001 - retain exact owner
            backend_id = id(backend)
            with self._lifecycle_lock:
                self._unbound_cleanup[backend_id] = (
                    label,
                    backend,
                    required,
                )
                self._cleanup_errors[backend_id] = (label, error)
            raise

    def get(
        self,
        role: str,
        *,
        toolset: Any | None = None,
        context_budget: int | None = None,
        capability: str | None = None,
        native_shell: bool | None = None,
        feature_requirements: tuple[FeatureRequirement, ...] = (),
        conversation_mode: ConversationMode = "none",
        limit_request: InvocationLimitRequest | None = None,
    ) -> Any:
        """Resolve (and cache) a backend for ``role``.

        ``toolset`` (optional) is attached via ``backend.attach_toolset``
        on first construction. Subsequent ``get(role, toolset=...)`` calls
        with the same role return the same backend — the toolset is not
        re-attached. If you need to swap toolsets mid-workflow, build a
        second backend via a second pool, or call attach_toolset directly
        on the returned backend (its lifecycle is the pool's still).

        ``native_shell=False`` disables adapter-native shell tools while
        retaining a ``workspace_write`` capability for path-confined MCP
        mutation tools. Containerized authoring uses this split so workflow
        code owns every executable command.

        ``feature_requirements`` declares any workflow-specific guarantees
        beyond the role and capability defaults. Requirements participate in
        the cache key and are admitted before backend construction.

        ``conversation_mode`` is likewise admitted before construction.
        ``workflow_bound`` accepts only a route whose concrete feature profile
        declares that exact continuity strength; transport-retry continuity
        never silently widens into a workflow correction session.

        ``limit_request`` binds this cache entry to one exact UIR-03
        invocation envelope. A different request necessarily constructs a
        different adapter; callers must not reuse either entry for an
        invocation whose turn or USD values differ.
        """
        self._raise_if_cleanup_pending()
        if self._development_budget_ledger is not None:
            # Allocation admission precedes route resolution and therefore
            # any provider construction or transport.
            self._development_budget_ledger.require_role(role)
        if capability is None:
            if toolset is None:
                resolved_capability: CapabilityProfile = "text"
            else:
                writes = toolset_write_names(toolset)
                if writes is None:
                    raise ValueError(
                        "toolset capability is unclassified; pass an explicit "
                        "capability and add write_names metadata"
                    )
                resolved_capability = (
                    "workspace_write" if writes else "read_only"
                )
        else:
            resolved_capability = normalize_capability(capability)
        effective_requirements = (
            *feature_requirements,
            *_conversation_feature_requirements(conversation_mode),
        )
        requirements_key = tuple(
            (requirement.field, tuple(sorted(requirement.accepted)))
            for requirement in effective_requirements
        )
        key = (
            role,
            toolset is not None,
            resolved_capability,
            native_shell,
            requirements_key,
            conversation_mode,
            limit_request,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        resolve_kwargs: dict[str, Any] = {
            "routing": self.routing,
            "log_dir": self._log_dir,
            "cwd": self._cwd,
            "capability": resolved_capability,
            "native_shell": native_shell,
        }
        if resolved_capability == "workspace_write":
            # Writer workflows supply their task context explicitly through
            # the confined toolset. Never let an adapter also ingest mutable
            # repository instructions outside that workflow-owned boundary.
            resolve_kwargs["allow_trusted_project_instructions"] = False
        if effective_requirements:
            resolve_kwargs["feature_requirements"] = effective_requirements
        if limit_request is not None:
            resolve_kwargs["limit_request"] = limit_request
            if self._limit_ledger is not None:
                resolve_kwargs["limit_ledger"] = self._limit_ledger
        elif self._limit_policy is not None:
            resolve_kwargs["limit_policy"] = self._limit_policy
            resolve_kwargs["limit_state"] = self._limit_state
            if self._limit_ledger is not None:
                resolve_kwargs["limit_ledger"] = self._limit_ledger
        backend = resolve_for_role(role, **resolve_kwargs)
        cleanup = self._bind_or_retain_backend_cleanup(
            backend,
            label=f"role {role!r}",
            required=resolved_capability == "workspace_write",
        )
        label = f"role {role!r}"
        self._own_bound_backend_setup(
            backend,
            label=label,
            close=cleanup,
        )
        try:
            if toolset is not None:
                attach = getattr(backend, "attach_toolset", None)
                if attach is None:
                    # Every supported tool-using adapter implements this
                    # method.  Do not inspect another backend descriptor while
                    # reporting the mismatch: it may raise before settlement.
                    raise RuntimeError(
                        f"backend routed for {label} does not implement "
                        "attach_toolset; route this role to a tool-capable "
                        "backend or drop the toolset argument."
                    )
                attach(toolset, context_budget=context_budget)
            exposed_backend = (
                self._development_budget_ledger.wrap(role, backend)
                if self._development_budget_ledger is not None
                else backend
            )
            owner_id = id(backend)
            with self._lifecycle_lock:
                self._cache[key] = exposed_backend
                self._cache_owner_ids[key] = owner_id
                self._provisional_backends.pop(owner_id, None)
        except BaseException as error:
            with self._lifecycle_lock:
                self._cache.pop(key, None)
                self._cache_owner_ids.pop(key, None)
            self._cleanup_failed_setup(
                backend,
                label=label,
                primary_error=error,
                close=cleanup,
            )
            raise
        return exposed_backend

    def new_exclusive(
        self,
        role: str,
        *,
        toolset: Any,
        log_dir: Path | str,
        cwd: str,
        context_budget: int | None = None,
        feature_requirements: tuple[FeatureRequirement, ...] = (),
        limit_request: InvocationLimitRequest | None = None,
        capability: CapabilityProfile = "read_only",
    ) -> Any:
        """Construct one uncached text or read-only analytical backend lease.

        The returned backend is owned by the caller and MUST be closed.  This
        method deliberately bypasses the role cache: adapters retain mutable
        ToolSet, context-budget, session, cancellation, and debug state, so a
        cached role instance is never a valid concurrent-child lease.

        Text reviewers must pass ``toolset=None``; they never gain an MCP
        surface merely because they need an independent session.
        ORC-01 keeps this surface narrow.  Writer leases belong to WRK-01
        after disposable worktrees and deterministic integration are
        available.
        """

        self._raise_if_cleanup_pending()
        if capability not in {"text", "read_only"}:
            raise ValueError("exclusive analytical leases require text or read_only capability")
        validate_toolset_capability(capability, toolset)
        if self._development_budget_ledger is not None:
            self._development_budget_ledger.require_role(role)
        writes = set() if toolset is None and capability == "text" else toolset_write_names(toolset)
        if writes is None:
            raise ValueError(
                "exclusive child ToolSet is unclassified; read_only_names "
                "and write_names are required"
            )
        if writes:
            raise ValueError(
                "exclusive ORC-01 backend leases cannot attach mutating "
                "tools: " + ", ".join(sorted(writes))
            )
        resolve_kwargs: dict[str, Any] = {
            "routing": self.routing,
            "log_dir": log_dir,
            "cwd": cwd,
            "capability": capability,
            "native_shell": False,
            "allow_trusted_project_instructions": False,
        }
        # ORC-01 is scheduler-owned. A future route that advertises qualified
        # provider-native delegation must still pass the separate NAT-01 E9
        # admission boundary; it cannot silently change this scheduler path.
        resolve_kwargs["feature_requirements"] = (
            *feature_requirements,
            FeatureRequirement(
                "native_delegation",
                frozenset({"none"}),
            ),
        )
        if limit_request is not None:
            resolve_kwargs["limit_request"] = limit_request
            if self._limit_ledger is not None:
                resolve_kwargs["limit_ledger"] = self._limit_ledger
        elif self._limit_policy is not None:
            resolve_kwargs["limit_policy"] = self._limit_policy
            resolve_kwargs["limit_state"] = self._limit_state
            if self._limit_ledger is not None:
                resolve_kwargs["limit_ledger"] = self._limit_ledger
        backend = resolve_for_role(role, **resolve_kwargs)
        cleanup = self._bind_or_retain_backend_cleanup(
            backend,
            label=f"exclusive role {role!r}",
            required=False,
        )
        label = f"exclusive role {role!r}"
        self._own_bound_backend_setup(
            backend,
            label=label,
            close=cleanup,
        )
        try:
            if toolset is not None:
                attach = getattr(backend, "attach_toolset", None)
                if attach is None:
                    raise RuntimeError(
                        f"backend routed for {label} does not implement "
                        "attach_toolset"
                    )
                attach(toolset, context_budget=context_budget)
            exposed_backend = (
                self._development_budget_ledger.wrap(role, backend)
                if self._development_budget_ledger is not None
                else backend
            )
        except BaseException as error:
            self._cleanup_failed_setup(
                backend,
                label=label,
                primary_error=error,
                close=cleanup,
            )
            raise
        self._relinquish_bound_backend_setup(backend)
        return exposed_backend

    def new_exclusive_writer(
        self,
        role: str,
        *,
        toolset: Any,
        log_dir: Path | str,
        cwd: str,
        context_budget: int | None = None,
        feature_requirements: tuple[FeatureRequirement, ...] = (),
        limit_request: InvocationLimitRequest | None = None,
    ) -> Any:
        """Construct one uncached, path-confined writer backend lease.

        WRK-01 callers create one instance per disposable worktree and own
        teardown.  Native shell and provider-native delegation remain off;
        mutation authority comes only from the attached, exactly classified
        Crucible ToolSet.  The isolated-writer scheduler separately verifies
        that no backend, ToolSet, ToolContext, worktree, or debug state is
        shared across lanes.
        """

        self._raise_if_cleanup_pending()
        if self._development_budget_ledger is not None:
            self._development_budget_ledger.require_role(role)
        writes = toolset_write_names(toolset)
        if writes is None:
            raise ValueError(
                "exclusive writer ToolSet is unclassified; read_only_names "
                "and write_names are required"
            )
        if not writes:
            raise ValueError(
                "exclusive writer backend requires at least one classified "
                "mutating tool"
            )
        resolve_kwargs: dict[str, Any] = {
            "routing": self.routing,
            "log_dir": log_dir,
            "cwd": cwd,
            "capability": "workspace_write",
            "native_shell": False,
            "allow_trusted_project_instructions": False,
            "feature_requirements": (
                *feature_requirements,
                FeatureRequirement(
                    "native_delegation",
                    frozenset({"none"}),
                ),
            ),
        }
        if limit_request is not None:
            resolve_kwargs["limit_request"] = limit_request
            if self._limit_ledger is not None:
                resolve_kwargs["limit_ledger"] = self._limit_ledger
        elif self._limit_policy is not None:
            resolve_kwargs["limit_policy"] = self._limit_policy
            resolve_kwargs["limit_state"] = self._limit_state
            if self._limit_ledger is not None:
                resolve_kwargs["limit_ledger"] = self._limit_ledger
        backend = resolve_for_role(role, **resolve_kwargs)
        cleanup = self._bind_or_retain_backend_cleanup(
            backend,
            label=f"exclusive writer role {role!r}",
            required=True,
        )
        label = f"exclusive writer role {role!r}"
        self._own_bound_backend_setup(
            backend,
            label=label,
            close=cleanup,
        )
        try:
            attach = getattr(backend, "attach_toolset", None)
            if attach is None:
                raise RuntimeError(
                    f"backend routed for {label} does not implement "
                    "attach_toolset"
                )
            attach(toolset, context_budget=context_budget)
            exposed_backend = (
                self._development_budget_ledger.wrap(role, backend)
                if self._development_budget_ledger is not None
                else backend
            )
        except BaseException as error:
            self._cleanup_failed_setup(
                backend,
                label=label,
                primary_error=error,
                close=cleanup,
            )
            raise
        self._relinquish_bound_backend_setup(backend)
        return exposed_backend

    def close(self, *, _retry: bool = False) -> None:
        """Retry the exact cleanup transaction through cancellation."""

        interrupt_count = 0
        while True:
            try:
                self._close_once(_retry=_retry)
                with self._lifecycle_lock:
                    deferred_interrupt = self._deferred_interrupt
                    self._deferred_interrupt = None
                if deferred_interrupt is not None:
                    raise BackendSettlementInterrupted(
                        self,
                        deferred_interrupt
                    ) from deferred_interrupt
                return
            except BackendSettlementInterrupted as error:
                if error.owner is self:
                    raise
                interrupt_count += 1
                with self._lifecycle_lock:
                    self._deferred_interrupt = (
                        self._deferred_interrupt or error.interrupt
                    )
                if interrupt_count >= _MAX_SETTLEMENT_INTERRUPTS:
                    raise BackendCleanupError(
                        (("backend pool cancellation settlement", error),),
                        retry_cleanup=lambda: self.close(_retry=True),
                    ) from error
                _retry = True
                continue
            except KeyboardInterrupt as error:
                interrupt_count += 1
                with self._lifecycle_lock:
                    self._deferred_interrupt = (
                        self._deferred_interrupt or error
                    )
                if interrupt_count >= _MAX_SETTLEMENT_INTERRUPTS:
                    raise BackendCleanupError(
                        (("backend pool cancellation settlement", error),),
                        retry_cleanup=lambda: self.close(_retry=True),
                    ) from error
                _retry = True
                continue
            except BaseException as error:
                with self._lifecycle_lock:
                    interrupted = self._deferred_interrupt is not None
                if interrupted:
                    error.add_note(
                        "SIGINT was deferred while backend settlement "
                        "remained unproven"
                    )
                raise

    def _close_once(self, *, _retry: bool) -> None:
        """Close every backend constructed by this pool.

        Every owned backend is attempted. Successfully settled objects are
        removed immediately; failed objects and their errors remain owned so
        a later ``close()`` retries only those objects. Failures are reported
        together after the full teardown pass.
        """
        with self._lifecycle_lock:
            pending_failures = tuple(self._cleanup_errors.values())
        if (
            not _retry
            and pending_failures
            and isinstance(sys.exception(), BackendCleanupError)
        ):
            # A setup unwind already attempted every newly owned cleanup and
            # is propagating its retry handle.  A surrounding context manager
            # must retain that owner, not immediately perform a second close
            # in the same exception propagation.  Explicit retry handles use
            # ``_retry=True`` and therefore always make progress.
            raise BackendCleanupError(
                pending_failures,
                retry_cleanup=lambda: self.close(_retry=True),
            )
        with self._lifecycle_lock:
            # Cleanup authority is bound before ToolSet attachment or model
            # dispatch.  Never rediscover it here: a mutable or raising
            # descriptor must not be able to revoke settlement after a
            # backend has acquired writer authority.
            owners = dict(self._cleanup_authorities)
            unbound = dict(self._unbound_cleanup)
            for backend_id, owned in self._pending_cleanup.items():
                owners.setdefault(backend_id, owned)

        failures: list[tuple[str, BaseException]] = []
        settled_ids: set[int] = set()
        for backend_id, (label, backend, required) in unbound.items():
            try:
                close = self._bind_backend_cleanup(
                    backend,
                    label=label,
                    required=required,
                )
            except KeyboardInterrupt as error:
                with self._lifecycle_lock:
                    self._cleanup_errors[backend_id] = (label, error)
                raise
            except BaseException as error:  # noqa: BLE001 - retained owner
                failures.append((label, error))
                with self._lifecycle_lock:
                    self._cleanup_errors[backend_id] = (label, error)
                continue
            with self._lifecycle_lock:
                # Cleanup authority is now positively bound.  Retain that
                # callable for all later retries and stop consulting a
                # potentially mutable descriptor.
                self._unbound_cleanup.pop(backend_id, None)
                self._provisional_backends[backend_id] = backend
                self._cleanup_authorities[backend_id] = (label, close)
                self._cleanup_errors.pop(backend_id, None)
            owners[backend_id] = (label, close)
        for backend_id, (label, close) in owners.items():
            if close is None:
                settled_ids.add(backend_id)
                continue
            try:
                close()
            except KeyboardInterrupt as error:
                # The outer transaction retains every exact owner and retries
                # from current pool state before cancellation can escape.
                with self._lifecycle_lock:
                    self._cleanup_errors[backend_id] = (label, error)
                self._release_settled_backend_ids(settled_ids)
                raise
            except BaseException as error:
                failures.append((label, error))
                with self._lifecycle_lock:
                    self._cleanup_errors[backend_id] = (label, error)
            else:
                settled_ids.add(backend_id)

        self._release_settled_backend_ids(settled_ids)

        if failures:
            raise BackendCleanupError(
                tuple(failures),
                retry_cleanup=lambda: self.close(_retry=True),
            )

    def _release_settled_backend_ids(self, settled_ids: set[int]) -> None:
        """Forget only owners whose exact close callable returned success."""

        with self._lifecycle_lock:
            if settled_ids:
                settled_keys = {
                    key
                    for key, owner_id in self._cache_owner_ids.items()
                    if owner_id in settled_ids
                }
                for key in settled_keys:
                    self._cache.pop(key, None)
                    self._cache_owner_ids.pop(key, None)
                for backend_id in settled_ids:
                    self._cleanup_authorities.pop(backend_id, None)
                    self._provisional_backends.pop(backend_id, None)
                    self._pending_cleanup.pop(backend_id, None)
                    self._unbound_cleanup.pop(backend_id, None)
                    self._cleanup_errors.pop(backend_id, None)


@contextlib.contextmanager
def backend_pool(
    args: argparse.Namespace,
    log_dir: Path | str | None,
    *,
    cwd: str | None = None,
    development_budget_ledger: DevelopmentBudgetLedger | None = None,
) -> Iterator[_BackendPool]:
    """Load routing once, hand back a per-role backend pool, teardown on exit.

    ``args`` is the workflow's parsed ``argparse.Namespace``; the pool
    reads ``args.backend_config`` and ``args.spec`` for routing
    discovery — the canonical kwargs every workflow now exposes via
    :func:`workflows._args.add_backend_config_arg` +
    :func:`workflows._args.add_spec_arg`.

    ``log_dir`` is typically ``handle.run_dir`` from
    :func:`workflows._run.open_run_with_capture`. ``cwd`` (optional)
    is forwarded to ``resolve_for_role`` so backends with filesystem
    surfaces sandbox their tools at the given path — pass
    ``str(workspace)`` for thread-safe fan-outs that cannot rely on process
    cwd.

    Workflows that need per-instance backends (e.g. swebench's
    per-instance ``ClaudeBackend(cwd=workspace)``) build via
    ``resolve_for_role`` directly; the pool's cache-by-role model would
    return a stale backend across instances.
    """
    routing = load_routing(
        config_path=args.backend_config, spec_dir=args.spec,
    )
    limit_mode = getattr(args, "backend_limit_mode", None)
    limit_deadline_seconds = getattr(
        args,
        "backend_limit_deadline_seconds",
        None,
    )
    if (limit_mode is None) != (limit_deadline_seconds is None):
        raise ValueError(
            "backend_limit_mode and backend_limit_deadline_seconds must be "
            "provided together"
        )
    from ._diagnostic_cost import validate_diagnostic_cost_policy

    diagnostic_cost_policy = getattr(args, "diagnostic_cost_policy", None)
    validate_diagnostic_cost_policy(
        diagnostic_cost_policy,
        limit_mode=limit_mode,
        budget_usd=getattr(args, "budget_usd", None),
    )
    limit_policy = (
        InvocationLimitPolicy(
            mode=cast(InvocationLimitMode, limit_mode),
            deadline_seconds=float(cast(float, limit_deadline_seconds)),
            require_budget_usd=getattr(args, "budget_usd", None) is not None,
            diagnostic_cost_policy=diagnostic_cost_policy,
        )
        if limit_mode is not None
        else None
    )
    limit_state: InvocationLimitState | None = None
    limit_ledger: InvocationLimitLedger | None = None
    if limit_policy is not None:
        if log_dir is None:
            raise ValueError(
                "backend invocation-limit policy requires an exact run "
                "directory for durable evidence"
            )
        ledger_path = (
            Path(log_dir).resolve(strict=False)
            / INVOCATION_LIMIT_LEDGER_LEAF
        )
        ledger_key = str(ledger_path)
        existing_states = getattr(
            args,
            "_crucible_invocation_limit_states",
            None,
        )
        if existing_states is None:
            existing_states = {}
            args._crucible_invocation_limit_states = existing_states
        if type(existing_states) is not dict:
            raise TypeError(
                "args._crucible_invocation_limit_states has an invalid type"
            )
        existing_limit_state = existing_states.get(ledger_key)
        if existing_limit_state is None:
            existing_limit_state = InvocationLimitState()
            existing_states[ledger_key] = existing_limit_state
        if type(existing_limit_state) is not InvocationLimitState:
            raise TypeError(
                "args._crucible_invocation_limit_states contains an invalid "
                "state"
            )
        existing_limit_state.bind(
            run_key=ledger_key,
            policy=limit_policy,
        )
        limit_state = existing_limit_state
        existing_ledgers = getattr(
            args,
            "_crucible_invocation_limit_ledgers",
            None,
        )
        if existing_ledgers is None:
            existing_ledgers = {}
            args._crucible_invocation_limit_ledgers = existing_ledgers
        if type(existing_ledgers) is not dict:
            raise TypeError(
                "args._crucible_invocation_limit_ledgers has an invalid type"
            )
        existing_ledger = existing_ledgers.get(ledger_key)
        if existing_ledger is None:
            existing_ledger = InvocationLimitLedger(ledger_path)
            existing_ledgers[ledger_key] = existing_ledger
        if type(existing_ledger) is not InvocationLimitLedger:
            raise TypeError(
                "args._crucible_invocation_limit_ledgers contains an invalid "
                "ledger"
            )
        if existing_ledger.path != ledger_path:
            raise InvocationLimitContractError(
                "cached invocation-limit ledger path does not match its "
                "exact run-directory key"
            )
        existing_ledger.bind_policy(limit_policy)
        limit_ledger = existing_ledger
    if log_dir is not None:
        _outcome.record_provenance_inputs(
            log_dir,
            backend_routing=routing_fingerprint(routing),
            backend_features=routing_feature_fingerprint(routing),
            backend_routing_source_sha256=(
                routing_source_fingerprint(routing)
            ),
        )
    pool = _BackendPool(
        routing,
        log_dir=log_dir,
        cwd=cwd,
        limit_policy=limit_policy,
        limit_state=limit_state,
        limit_ledger=limit_ledger,
        development_budget_ledger=development_budget_ledger,
    )
    try:
        yield pool
    finally:
        pool.close()


__all__ = [
    "BackendCleanupError",
    "BackendSettlementInterrupted",
    "BackendLifecycleError",
    "backend_pool",
]
