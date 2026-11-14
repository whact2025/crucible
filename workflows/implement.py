# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tiny ``implement`` exercise workflow.

Composes ``assay.primitives.implement`` with
``crucible.backends.config.resolve_for_role("implementer")`` and a ``Corpus`` to drive a single
slice end-to-end. This is the smoke check that the assay/crucible
boundary is real and the protocol round-trips.

Usage:

    uv run python -m crucible.workflows.implement \\
        --spec <path/to/spec/dir> \\
        --slice <slice_id>

The spec dir must be a ``Corpus``-shaped folder produced by an earlier
``assay`` workflow (a ``versions/`` directory with at least one
manifest). The slice id must resolve to a slice in the spec's
``tier_dag``.

This workflow is intentionally minimal — no build/verify, no retries,
no DAG iteration. A implementer path is the ``architect`` / ``breach`` workflows wrapping ``workflows._build_verify``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, TypeVar

from assay import UsagePricingEvidence
from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus
from assay.primitives.implement import implement

from crucible.common.cost import CostAccumulator, CostLimitExceeded
from crucible.tools import ToolMutationIndeterminate
from crucible.tools._context import ToolContext
from crucible.tools.profiles import build_autonomous_implementer_profile

from ._args import (
    IMPLEMENTER_BUILTIN_TOOLS,
    add_backend_config_arg, add_budget_arg, add_max_turns_arg, add_spec_arg,
)
from ._atomic_persistence import persist_and_attach
from ._backends import backend_pool
from ._implementer_completion import normalize_implementer_completion
from ._mutation_journal import (
    MutationTransaction,
    RetainedRecoveryWorkspaceMutationLock,
    recover_mutation_journals,
    recovery_workspace_mutation_lock,
)
from . import _outcome
from ._outcome import (
    TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
    checkpoint_settled_tool_mutation_terminal,
    checkpoint_unsettled_tool_mutation_terminal,
    is_tool_mutation_indeterminate_result,
    preserve_tool_mutation_indeterminate,
    synthesize_tool_mutation_indeterminate_implementation,
    unknown_tool_mutation_usage,
)
from ._provider_settlement import (
    ProviderSettlementCleanupError,
    ProviderSettlementExitStack,
)
from ._run import (
    FailureTerminal,
    exec_ctx_now,
    open_run_with_capture,
    record_handled_terminal,
)

logger = logging.getLogger(__name__)


_CleanupT = TypeVar("_CleanupT")
_WORKSPACE_CLEANUP_OWNER_ATTR = (
    "_crucible_implement_workspace_cleanup_owner"
)
_WORKSPACE_CLEANUP_ERROR_ATTR = (
    "_crucible_implement_workspace_cleanup_error"
)


class _OneShotContextExit:
    """Never report a failed generator-context exit as a later success."""

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
    """LIFO cleanup whose next failing callback remains retry authority."""

    def __init__(self) -> None:
        self._callbacks: list[
            tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]
        ] = []

    @property
    def cleanup_complete(self) -> bool:
        return not self._callbacks

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
            # The recovery-lock source publishes this owner's acquisition
            # lease immediately before yielding authority.  Pre-registering
            # retry_cleanup is therefore inert before entry and live before
            # the acquired lock can cross back into this frame.
            self.callback(context.retry_cleanup)
            return context.__enter__()

        # A generic context manager has no source-published acquisition proof.
        # Follow normal ``with`` semantics and never invoke ``__exit__`` when
        # ``__enter__`` did not return successfully.
        entered = context.__enter__()
        self.callback(_OneShotContextExit(context))
        return entered

    def close(self) -> None:
        self.retry_cleanup()

    def retry_cleanup(self) -> None:
        while self._callbacks:
            callback, args, kwargs = self._callbacks[-1]
            callback(*args, **kwargs)
            self._callbacks.pop()


class _ImplementSettlementStack(ProviderSettlementExitStack):
    """Precompose provider/journal cleanup with workspace release.

    The workspace owner is bound before workspace acquisition. Every cleanup
    error created by :class:`ProviderSettlementExitStack` therefore already
    names the complete retry owner; an asynchronous interruption never has to
    race a later ``ProviderSettlementCleanupError.owner`` replacement.
    """

    def __init__(
        self,
        workspace_release: _RetainedCleanupStack,
    ) -> None:
        super().__init__()
        self._workspace_release = workspace_release
        self._deferred_interrupt: KeyboardInterrupt | None = None

    @property
    def cleanup_complete(self) -> bool:
        return bool(
            super().cleanup_complete
            and self._workspace_release.cleanup_complete
        )

    def retry_cleanup(self) -> None:
        if not super().cleanup_complete:
            try:
                super().retry_cleanup()
            except KeyboardInterrupt as exc:
                if not super().cleanup_complete:
                    raise
                self._deferred_interrupt = self._deferred_interrupt or exc
        if not self._workspace_release.cleanup_complete:
            self._workspace_release.retry_cleanup()
        interrupt = self._deferred_interrupt
        self._deferred_interrupt = None
        if interrupt is not None:
            raise interrupt


def _owned_provider_cleanup_error(
    error: BaseException | None,
    expected_owner: ProviderSettlementExitStack,
) -> ProviderSettlementCleanupError | None:
    """Find this invocation's provider owner through explicit causes only."""

    current = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            isinstance(current, ProviderSettlementCleanupError)
            and current.owner is expected_owner
        ):
            return current
        cause = current.__cause__
        current = cause if isinstance(cause, BaseException) else None
    return None


def _retain_workspace_cleanup_owner(
    error: BaseException,
    workspace_release: _RetainedCleanupStack,
    cleanup_error: BaseException,
) -> None:
    """Bind one failed workspace release to its escaping primary error."""

    setattr(error, _WORKSPACE_CLEANUP_OWNER_ATTR, workspace_release)
    setattr(error, _WORKSPACE_CLEANUP_ERROR_ATTR, cleanup_error)


@dataclass(frozen=True)
class _ObservedInvocationUsage:
    """Cost-ledger projection retained before Assay post-processing."""

    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    accounting_debt: tuple[str, ...]
    pricing_evidence: Any
    projection_error: BaseException | None = None


def _observed_invocation_usage(result: Any) -> _ObservedInvocationUsage:
    """Project independent usage dimensions without discarding known spend.

    Provider telemetry is untrusted boundary data.  In particular, a broken
    token or completeness accessor must not flatten an already returned,
    independently valid numeric cost to ``$0``.  Retain the first exact
    projection error for terminal precedence, annotate later errors beneath it,
    and return a CostAccumulator-safe partial projection with debt only for the
    dimensions that could not be established.
    """

    debt: set[str] = set()
    projection_error: BaseException | None = None

    def retain_projection_error(
        error: BaseException,
        *missing: str,
    ) -> None:
        nonlocal projection_error
        debt.update(missing)
        if projection_error is None:
            projection_error = error
            return
        if projection_error is not error:
            projection_error.add_note(
                "secondary Implement usage projection failure: "
                f"{type(error).__name__}: {str(error)[:1000]}"
            )

    try:
        usage = getattr(result, "usage", None)
    except BaseException as error:  # noqa: BLE001 - exact telemetry owner
        usage = None
        retain_projection_error(
            error,
            "token_usage_complete",
            "cost_complete",
        )

    cost_usd = 0.0
    input_tokens = 0
    output_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    if usage is None:
        debt.update(("token_usage_complete", "cost_complete"))
    else:
        try:
            cost_usd = float(getattr(usage, "cost_usd", 0.0) or 0.0)
            if not math.isfinite(cost_usd) or cost_usd < 0.0:
                raise ValueError(
                    "backend usage cost_usd must be finite and non-negative"
                )
        except BaseException as error:  # noqa: BLE001 - exact telemetry owner
            cost_usd = 0.0
            retain_projection_error(error, "cost_complete")

        token_values: dict[str, int] = {}
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
        ):
            try:
                value = int(getattr(usage, field_name, 0) or 0)
                if value < 0:
                    raise ValueError(
                        f"backend usage {field_name} must be non-negative"
                    )
            except BaseException as error:  # noqa: BLE001 - exact telemetry
                value = 0
                retain_projection_error(error, "token_usage_complete")
            token_values[field_name] = value
        input_tokens = token_values["input_tokens"]
        output_tokens = token_values["output_tokens"]
        cache_creation_tokens = token_values["cache_creation_tokens"]
        cache_read_tokens = token_values["cache_read_tokens"]
        if cache_creation_tokens + cache_read_tokens > input_tokens:
            retain_projection_error(
                ValueError(
                    "backend cache-token subsets must not exceed input tokens"
                ),
                "token_usage_complete",
            )
            # Keep the independently valid aggregate token counters while
            # replacing only the inconsistent cache subsets with safe zeros.
            cache_creation_tokens = 0
            cache_read_tokens = 0

    try:
        completeness = getattr(result, "completeness", None)
    except BaseException as error:  # noqa: BLE001 - exact telemetry owner
        completeness = None
        retain_projection_error(
            error,
            "token_usage_complete",
            "cost_complete",
        )
    if completeness is None:
        debt.update(("token_usage_complete", "cost_complete"))
    else:
        for field_name, dimension in (
            ("token_usage_complete", "token_usage_complete"),
            ("cost_complete", "cost_complete"),
        ):
            try:
                raw_complete = getattr(completeness, field_name, None)
                if type(raw_complete) is not bool:
                    raise TypeError(
                        "backend completeness "
                        f"{field_name} must be an exact boolean"
                    )
                complete = raw_complete
            except BaseException as error:  # noqa: BLE001 - exact telemetry
                complete = False
                retain_projection_error(error, dimension)
            if not complete:
                debt.add(dimension)

    try:
        pricing_evidence = getattr(result, "pricing_evidence", None)
    except BaseException as error:  # noqa: BLE001 - exact telemetry owner
        pricing_evidence = None
        retain_projection_error(error, "cost_complete")
    if (
        pricing_evidence is not None
        and type(pricing_evidence) is not UsagePricingEvidence
    ):
        retain_projection_error(
            TypeError(
                "backend pricing_evidence must be a UsagePricingEvidence or None"
            ),
            "cost_complete",
        )
        # CostAccumulator accepts only typed pricing provenance.  Dropping the
        # malformed projection while retaining independently validated usage
        # records the useful numeric lower bound without laundering the
        # untrusted evidence into proof of completeness.
        pricing_evidence = None

    return _ObservedInvocationUsage(
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        accounting_debt=tuple(
            dimension
            for dimension in (
                "token_usage_complete",
                "cost_complete",
            )
            if dimension in debt
        ),
        pricing_evidence=pricing_evidence,
        projection_error=projection_error,
    )


class _TerminalPrecedenceBackend:
    """Observe one exact mutation terminal before Assay's Git callback.

    ``assay.implement`` invokes ``before_capture`` only after the backend has
    returned.  That callback must close a workspace-writing provider before it
    touches Git, but a close failure would otherwise hide a returned
    ``tool_mutation_indeterminate`` result.  This transparent proxy retains one
    workflow-owned terminal object at the invocation boundary, early enough
    for provider-first cleanup to preserve its precedence.
    """

    def __init__(
        self,
        backend: Any,
        settlement_stack: ProviderSettlementExitStack,
    ) -> None:
        self._backend = backend
        self._settlement_stack = settlement_stack
        self.terminal_error: ToolMutationIndeterminate | None = None
        self.terminal_result: Any | None = None
        self.invocation_started = False
        self.invocation_result: Any | None = None

    @property
    def name(self) -> str:
        return str(self._backend.name)

    @property
    def model(self) -> str | None:
        model = self._backend.model
        return str(model) if model is not None else None

    def _retain(
        self,
        error: ToolMutationIndeterminate,
    ) -> ToolMutationIndeterminate:
        if self.terminal_error is None:
            self._settlement_stack.retain_terminal_precedence(error)
            self.terminal_error = error
            return error
        if self.terminal_error is not error:
            raise RuntimeError(
                "backend reported more than one mutation-indeterminate "
                "terminal for one invocation"
            )
        return error

    def invoke(self, invocation: Any) -> Any:
        if self.invocation_started:
            raise RuntimeError(
                "standalone Implement dispatched more than one provider call"
            )
        self.invocation_started = True
        try:
            result = self._backend.invoke(invocation)
        except ToolMutationIndeterminate as error:
            self._retain(error)
            raise
        # Retain exact usage before Assay's post-invocation provider close,
        # Git capture, transcript persistence, or Implementation construction.
        self.invocation_result = result
        if is_tool_mutation_indeterminate_result(result):
            self.terminal_result = result
            provider_error = getattr(result, "error", None)
            provider_message = getattr(provider_error, "message", None)
            message = (
                str(provider_message)
                if provider_message
                else "backend returned tool_mutation_indeterminate"
            )
            self._retain(ToolMutationIndeterminate(message))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


@dataclass
class _ImplementFailureState:
    """In-memory cost and cleanup evidence available during run unwind."""

    costs: CostAccumulator
    observed_backend: _TerminalPrecedenceBackend | None = None
    usage_recorded: bool = False
    terminal_accounting_error: BaseException | None = None
    terminal_accounting_debt: tuple[str, ...] = ()
    provider_cleanup_pending: bool = False
    provider_cleanup_settled: bool = False
    mutation_cleanup_pending: bool = False
    workspace_cleanup_pending: bool = False
    evidence_publication_incomplete: bool = False

    def mark_cleanup(
        self,
        *,
        provider: bool = False,
        mutation: bool = False,
        workspace: bool = False,
    ) -> None:
        self.provider_cleanup_pending = bool(
            self.provider_cleanup_pending or provider
        )
        self.mutation_cleanup_pending = bool(
            self.mutation_cleanup_pending or mutation
        )
        self.workspace_cleanup_pending = bool(
            self.workspace_cleanup_pending or workspace
        )

    def failure_terminal(
        self,
        error: BaseException,
    ) -> FailureTerminal | None:
        """Return a debt terminal only when exact failure evidence requires it."""

        costs = self.costs.to_dict()
        debt_by_gate: dict[str, list[str]] = {}
        backend = self.observed_backend
        if (
            backend is not None
            and backend.invocation_started
            and not self.usage_recorded
        ):
            debt_by_gate["implement_invocation"] = [
                "cost_complete",
                "token_usage_complete",
            ]
        if bool(costs.get("accounting_debt")):
            debt_by_role = costs.get("accounting_debt_by_role")
            role_debt = (
                debt_by_role.get("implementer", ())
                if isinstance(debt_by_role, dict)
                else ()
            )
            debt_by_gate["implement_usage"] = sorted(
                {
                    str(dimension)
                    for dimension in role_debt
                    if str(dimension) in {
                        "cost_complete",
                        "token_usage_complete",
                    }
                }
                or {"cost_complete", "token_usage_complete"}
            )
        if self.terminal_accounting_error is not None:
            debt_by_gate["implement_accounting"] = sorted(
                set(self.terminal_accounting_debt)
                or {"cost_complete", "token_usage_complete"}
            )
        if self.provider_cleanup_pending:
            debt_by_gate["implement_provider_cleanup"] = [
                "cost_complete",
                "token_usage_complete",
            ]
        if self.workspace_cleanup_pending or (
            isinstance(
                getattr(error, _WORKSPACE_CLEANUP_OWNER_ATTR, None),
                _RetainedCleanupStack,
            )
            and not getattr(
                error,
                _WORKSPACE_CLEANUP_OWNER_ATTR,
            ).cleanup_complete
        ):
            debt_by_gate["implement_workspace_cleanup"] = [
                "cost_complete",
                "token_usage_complete",
            ]
        if self.evidence_publication_incomplete:
            debt_by_gate["implement_terminal_publication"] = [
                "cost_complete",
                "token_usage_complete",
            ]
        if not debt_by_gate:
            return None
        fields: dict[str, object] = {
            "cost_usd_is_lower_bound": True,
            "accounting_debt": True,
            "accounting_debt_by_gate": debt_by_gate,
            "provider_cleanup_pending": self.provider_cleanup_pending,
            "mutation_cleanup_pending": self.mutation_cleanup_pending,
            "journal_cleanup_pending": self.mutation_cleanup_pending,
            "workspace_cleanup_pending": self.workspace_cleanup_pending,
            "cleanup_pending": bool(
                self.provider_cleanup_pending
                or self.mutation_cleanup_pending
                or self.workspace_cleanup_pending
            ),
            "accounting_evidence": "incomplete",
            "costs": costs,
        }
        debt_by_role = costs.get("accounting_debt_by_role")
        if isinstance(debt_by_role, dict):
            fields["accounting_debt_by_role"] = dict(debt_by_role)
        if self.terminal_accounting_error is not None:
            fields.update({
                "accounting_error_kind": type(
                    self.terminal_accounting_error
                ).__name__,
                "accounting_error_message": str(
                    self.terminal_accounting_error
                )[:2000],
            })
        workspace_error = getattr(
            error,
            _WORKSPACE_CLEANUP_ERROR_ATTR,
            None,
        )
        if isinstance(workspace_error, BaseException):
            fields.update({
                "workspace_cleanup_error_kind": type(
                    workspace_error
                ).__name__,
                "workspace_cleanup_error_message": str(
                    workspace_error
                )[:2000],
            })
        return FailureTerminal(
            status="incomplete_accounting",
            summary_fields=fields,
        )


class _DeferredImplementRunHandle:
    """Stage immutable results until workspace authority has been released."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._pending_result: RunResult | None = None
        self._pending_terminal_error: BaseException | None = None
        self._measure_elapsed = True

    @property
    def finalized(self) -> bool:
        return self._pending_result is not None or bool(self._handle.finalized)

    @property
    def pending_terminal_error(self) -> BaseException | None:
        return self._pending_terminal_error

    def bind_terminal_error(self, error: BaseException) -> None:
        if (
            self._pending_terminal_error is not None
            and self._pending_terminal_error is not error
        ):
            raise RuntimeError(
                "Implement terminal is already bound to another error"
            )
        self._pending_terminal_error = error

    def pending_terminal_matches(self, error: BaseException) -> bool:
        result = self._pending_result
        if result is None or self._pending_terminal_error is not error:
            return False
        try:
            terminal_class = _outcome.classify_terminal_status(result.status)
        except (TypeError, ValueError):
            return False
        return bool(
            terminal_class
            not in {
                _outcome.TerminalClass.SUCCEEDED,
                _outcome.TerminalClass.NO_WORK,
            }
            and result.error_kind == type(error).__name__
            and result.error_message == str(error)[:2000]
        )

    def finalize_elapsed(self, result: RunResult) -> None:
        self._stage(result, measure_elapsed=True)

    def finalize(self, result: RunResult) -> None:
        self._stage(result, measure_elapsed=False)

    def _stage(self, result: RunResult, *, measure_elapsed: bool) -> None:
        if self.finalized:
            raise RuntimeError("Implement run result is already staged")
        self._pending_result = result
        self._measure_elapsed = measure_elapsed

    def publish(self) -> None:
        result = self._pending_result
        if result is None:
            return
        if self._measure_elapsed:
            self._handle.finalize_elapsed(result)
        else:
            self._handle.finalize(result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


class _ImplementRunBoundary:
    """Order workspace acquisition/release around one deferred run envelope."""

    def __init__(
        self,
        *,
        corpus: Corpus,
        cwd: Path,
        failure_state: _ImplementFailureState,
    ) -> None:
        self._corpus = corpus
        self._cwd = cwd
        self._failure_state = failure_state
        self._workspace_release = _RetainedCleanupStack()
        # Compose the complete retry owner before ``__enter__`` can acquire
        # workspace authority. Provider/journal failure therefore never needs
        # a later, interruptible owner replacement to retain the workspace.
        self._settlement_stack = _ImplementSettlementStack(
            self._workspace_release,
        )
        self._run_context: AbstractContextManager[Any] | None = None
        self._deferred: _DeferredImplementRunHandle | None = None
        self._boundary_entered = False
        self._open_started = False
        self._run_entry_started = False
        self._workspace_release_attempted = False
        self._preyield_cleanup_error: BaseException | None = None

    @property
    def settlement_stack(self) -> _ImplementSettlementStack:
        return self._settlement_stack

    @property
    def workspace_release(self) -> _RetainedCleanupStack:
        return self._workspace_release

    def _release_workspace(self) -> None:
        self._workspace_release_attempted = True
        self._workspace_release.close()

    def _bind_workspace_failure(
        self,
        primary: BaseException,
        cleanup_error: BaseException,
    ) -> None:
        self._failure_state.mark_cleanup(workspace=True)
        _retain_workspace_cleanup_owner(
            primary,
            self._workspace_release,
            cleanup_error,
        )
        if primary is cleanup_error:
            return
        if primary.__cause__ is None:
            primary.__cause__ = cleanup_error
            primary.__suppress_context__ = True
        else:
            primary.add_note(
                "secondary workspace release failure: "
                f"{type(cleanup_error).__name__}: "
                f"{str(cleanup_error)[:1000]}"
            )

    def _failure_terminal(
        self,
        error: BaseException,
    ) -> FailureTerminal | None:
        if self._deferred is None and not self._workspace_release_attempted:
            try:
                self._release_workspace()
            except BaseException as cleanup_error:  # noqa: BLE001
                self._preyield_cleanup_error = cleanup_error
                self._bind_workspace_failure(cleanup_error, cleanup_error)
                if cleanup_error.__cause__ is None:
                    cleanup_error.__cause__ = error
                    cleanup_error.__suppress_context__ = True
                terminal = self._failure_state.failure_terminal(cleanup_error)
                if terminal is None:
                    raise RuntimeError(
                        "Implement pre-yield cleanup lost its debt terminal"
                    ) from cleanup_error
                return terminal
        return self._failure_state.failure_terminal(error)

    def __enter__(self) -> "_ImplementRunBoundary":
        if self._boundary_entered:
            raise RuntimeError("Implement run boundary is not reusable")
        # Install this boundary's __exit__ before ``open`` can acquire the
        # workspace or advance the generator-backed run context to its yield.
        # The empty entry itself owns no resource, so an interruption while the
        # outer ``with`` records this return has nothing to leak.
        self._boundary_entered = True
        return self

    def open(self) -> _DeferredImplementRunHandle:
        if not self._boundary_entered:
            raise RuntimeError(
                "Implement run boundary must be entered before open()"
            )
        if self._open_started:
            raise RuntimeError("Implement run boundary is not reusable")
        self._open_started = True
        try:
            self._workspace_release.enter_context(
                recovery_workspace_mutation_lock(
                    self._corpus,
                    self._cwd,
                )
            )
            self._run_context = open_run_with_capture(
                self._corpus,
                "implement",
                provenance_root=Path(self._cwd),
                failure_cost_usd=lambda: self._failure_state.costs.total_cost_usd,
                failure_terminal=self._failure_terminal,
            )
            # Publish the exact generator exit owner before ``__enter__`` can
            # yield a live RunHandle.  If SIGINT lands during the return-value
            # transfer, the already-entered outer boundary still closes capture
            # and publishes typed terminal evidence.
            self._run_entry_started = True
            raw_handle = self._run_context.__enter__()
            self._deferred = _DeferredImplementRunHandle(raw_handle)
            return self._deferred
        except BaseException as error:  # noqa: BLE001 - preserve exact owner
            failure = self._preyield_cleanup_error or error
            if failure is not error and failure.__cause__ is None:
                failure.__cause__ = error
                failure.__suppress_context__ = True
            raise failure.with_traceback(  # noqa: B904 - preserve exact owner
                failure.__traceback__
            )

    def _exit_run_with_error(
        self,
        error: BaseException,
        traceback: object | None,
        *,
        body_error: bool,
    ) -> bool:
        if self._run_context is None or not self._run_entry_started:
            if body_error:
                return False
            raise error.with_traceback(error.__traceback__)
        suppressed = self._run_context.__exit__(
            type(error),
            error,
            traceback,
        )
        if suppressed:
            return True
        if body_error:
            return False
        raise error.with_traceback(error.__traceback__)

    def _publish_bound_terminal(
        self,
        error: BaseException,
        traceback: object | None,
    ) -> None:
        assert self._deferred is not None
        try:
            self._deferred.publish()
        except BaseException as publication_error:  # noqa: BLE001
            self._failure_state.evidence_publication_incomplete = True
            if error.__cause__ is None:
                error.__cause__ = publication_error
                error.__suppress_context__ = True
            else:
                error.add_note(
                    "secondary Implement terminal publication failure: "
                    f"{type(publication_error).__name__}: "
                    f"{str(publication_error)[:1000]}"
                )
            self._exit_run_with_error(
                error,
                traceback,
                body_error=False,
            )
            raise AssertionError(
                "unreachable terminal publication path"
            ) from publication_error
        assert self._run_context is not None
        try:
            self._run_context.__exit__(None, None, None)
        except BaseException as outcome_error:  # noqa: BLE001
            self._failure_state.evidence_publication_incomplete = True
            if error.__cause__ is None:
                error.__cause__ = outcome_error
                error.__suppress_context__ = True
            else:
                error.add_note(
                    "secondary Implement outcome publication failure: "
                    f"{type(outcome_error).__name__}: "
                    f"{str(outcome_error)[:1000]}"
                )
        raise error.with_traceback(error.__traceback__)

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: object | None,
    ) -> bool:
        if error is not None:
            try:
                cleanup_owner = _owned_provider_cleanup_error(
                    error,
                    self._settlement_stack,
                )
                if cleanup_owner is not None:
                    self._failure_state.mark_cleanup(
                        provider=(
                            not self._failure_state.provider_cleanup_settled
                        ),
                        mutation=True,
                        workspace=True,
                    )
            except BaseException as recognition_error:  # noqa: BLE001
                if recognition_error is error:
                    raise
                # Recognition is diagnostic only: the provider error already
                # names the precomposed provider/journal/workspace owner.  A
                # SIGINT at any bytecode in the cause-chain walk must still
                # cross the real run/capture exit. Recover the two canonical
                # direct shapes without replaying the interrupted walk.
                direct_cause = error.__cause__
                retained_owner = (
                    error
                    if (
                        isinstance(error, ProviderSettlementCleanupError)
                        and error.owner is self._settlement_stack
                    )
                    else (
                        direct_cause
                        if (
                            isinstance(
                                direct_cause,
                                ProviderSettlementCleanupError,
                            )
                            and direct_cause.owner is self._settlement_stack
                        )
                        else None
                    )
                )
                if retained_owner is not None:
                    # Avoid another interruptible helper dispatch here. These
                    # monotonic flags describe the already-stable owner used as
                    # the cancellation's direct cause.
                    self._failure_state.provider_cleanup_pending = bool(
                        self._failure_state.provider_cleanup_pending
                        or not self._failure_state.provider_cleanup_settled
                    )
                    self._failure_state.mutation_cleanup_pending = True
                    self._failure_state.workspace_cleanup_pending = True
                    recognition_error.__cause__ = retained_owner
                    recognition_error.__suppress_context__ = True
                else:
                    # No provider authority was discoverable in either normal
                    # direct shape. Release the standalone workspace owner
                    # before publishing the recognition interruption.
                    try:
                        self._release_workspace()
                    except BaseException as cleanup_error:  # noqa: BLE001
                        self._bind_workspace_failure(
                            recognition_error,
                            cleanup_error,
                        )
                    if recognition_error.__cause__ is None:
                        recognition_error.__cause__ = error
                        recognition_error.__suppress_context__ = True
                return self._exit_run_with_error(
                    recognition_error,
                    recognition_error.__traceback__,
                    body_error=False,
                )
            if cleanup_owner is not None:
                return self._exit_run_with_error(
                    error,
                    traceback,
                    body_error=True,
                )
            try:
                self._release_workspace()
            except BaseException as cleanup_error:  # noqa: BLE001
                self._bind_workspace_failure(error, cleanup_error)
                return self._exit_run_with_error(
                    error,
                    traceback,
                    body_error=True,
                )
            if (
                self._deferred is not None
                and self._deferred.pending_terminal_matches(error)
            ):
                self._publish_bound_terminal(error, traceback)
                raise AssertionError("unreachable bound terminal path")
            return self._exit_run_with_error(
                error,
                traceback,
                body_error=True,
            )

        if self._deferred is None:
            incomplete_open = RuntimeError(
                "Implement run boundary exited without an opened run handle"
            )
            try:
                self._release_workspace()
            except BaseException as cleanup_error:  # noqa: BLE001
                self._bind_workspace_failure(incomplete_open, cleanup_error)
            return self._exit_run_with_error(
                incomplete_open,
                incomplete_open.__traceback__,
                body_error=False,
            )

        try:
            self._release_workspace()
        except BaseException as cleanup_error:  # noqa: BLE001
            terminal_error = self._deferred.pending_terminal_error
            primary_error = terminal_error or cleanup_error
            self._bind_workspace_failure(primary_error, cleanup_error)
            return self._exit_run_with_error(
                primary_error,
                primary_error.__traceback__,
                body_error=False,
            )
        try:
            self._deferred.publish()
        except BaseException as publication_error:  # noqa: BLE001
            self._failure_state.evidence_publication_incomplete = True
            terminal_error = self._deferred.pending_terminal_error
            if terminal_error is not None:
                if terminal_error.__cause__ is None:
                    terminal_error.__cause__ = publication_error
                    terminal_error.__suppress_context__ = True
                else:
                    terminal_error.add_note(
                        "secondary Implement terminal publication failure: "
                        f"{type(publication_error).__name__}: "
                        f"{str(publication_error)[:1000]}"
                    )
                return self._exit_run_with_error(
                    terminal_error,
                    terminal_error.__traceback__,
                    body_error=False,
                )
            return self._exit_run_with_error(
                publication_error,
                publication_error.__traceback__,
                body_error=False,
            )
        try:
            return bool(self._run_context.__exit__(None, None, None))
        except BaseException as outcome_error:  # noqa: BLE001
            terminal_error = self._deferred.pending_terminal_error
            if terminal_error is None:
                raise
            # A settled mutation-indeterminate result remains the business
            # terminal even when the later outcome projection fails.  The
            # immutable result is already published; retain the projection
            # error as secondary evidence without replacing terminal identity.
            self._failure_state.evidence_publication_incomplete = True
            if terminal_error.__cause__ is None:
                terminal_error.__cause__ = outcome_error
                terminal_error.__suppress_context__ = True
            else:
                terminal_error.add_note(
                    "secondary Implement outcome publication failure: "
                    f"{type(outcome_error).__name__}: "
                    f"{str(outcome_error)[:1000]}"
                )
            raise terminal_error.with_traceback(  # noqa: B904 - preserve terminal
                terminal_error.__traceback__
            )


def _find_slice(spec, slice_id: str):
    if spec.tier_dag is None:
        raise SystemExit(
            f"spec v{spec.version} has no tier_dag — nothing to implement"
        )
    for tier in spec.tier_dag.tiers:
        for s in tier.slices:
            if s.id == slice_id:
                return s
    raise SystemExit(
        f"slice {slice_id!r} not found in tier_dag "
        f"(tiers: {[t.tier_id for t in spec.tier_dag.tiers]})"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="crucible-workflow-implement")
    add_spec_arg(p)
    p.add_argument(
        "--slice",
        dest="slice_id",
        required=True,
        help="Slice ID to implement (must exist in the spec's tier_dag).",
    )
    add_backend_config_arg(p, "implementer")
    add_max_turns_arg(
        p, default=50,
        scope="for the implementer's single backend call",
    )
    add_budget_arg(p, kind="per_call")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    corpus = Corpus(args.spec)
    spec = corpus.latest_spec()
    if spec is None:
        message = f"no spec manifest found under {args.spec / 'versions'}"
        record_handled_terminal(
            corpus, "implement", status="setup_error",
            message=message, error_kind="missing_spec",
            provenance_root=None,
        )
        return 2

    try:
        slice_ = _find_slice(spec, args.slice_id)
    except SystemExit as exc:
        record_handled_terminal(
            corpus,
            "implement",
            status="setup_error",
            message=str(exc),
            error_kind="usage",
            provenance_root=None,
        )
        return 2

    # Attach the implementer profile (source + write + build) as an
    # adapter bridge so the agent can navigate source and perform only
    # directive-scoped canonical source mutation, including exact-root
    # deletion. Build/test execution and binary artifact materialization remain
    # workflow-owned. The SDK-native allowlist
    # below is empty, so this path-confined profile is the entire authoring
    # surface.
    cwd = Path.cwd()
    write_scope = tuple(dict.fromkeys((
        *slice_.directive.files_in_scope,
        *slice_.directive.creates_paths,
    )))
    ctx = ToolContext(
        workspace_root=cwd,
        repo_root=cwd,
        write_scope=write_scope,
    )
    # The boundary acquires the workspace before opening run provenance, then
    # stages immutable RunResults until provider/journal cleanup and workspace
    # release have completed. Its pre-yield failure selector also closes the
    # workspace before ``open_run_with_capture`` can publish a capture-entry
    # failure.
    costs = CostAccumulator(max_cost_usd=args.budget_usd)
    failure_state = _ImplementFailureState(costs=costs)
    run_boundary = _ImplementRunBoundary(
        corpus=corpus,
        cwd=cwd,
        failure_state=failure_state,
    )
    settlement_stack = run_boundary.settlement_stack

    with run_boundary:
        handle = run_boundary.open()
        recovery = recover_mutation_journals(corpus, cwd)
        if recovery.failures:
            message = (
                "mutation recovery failed closed: "
                + "; ".join(recovery.failures)
            )
            handle.write_summary({"status": "setup_error", "error": message})
            handle.finalize_elapsed(RunResult(status="setup_error"))
            logger.error("implement: %s", message)
            return 2

        with settlement_stack:
            transaction = MutationTransaction.begin(
                cwd=cwd,
                journal_root=handle.run_dir / "mutations",
                slice_id=args.slice_id,
                scope=write_scope,
                sequence=1,
                spec=spec,
            )
            ctx.mutation_journal_dir = transaction.journal_dir
            ctx.mutation_journal_transaction_id = transaction.transaction_id
            settled = False
            settlement_action: str | None = None

            def _settle_transaction() -> None:
                nonlocal settled
                if not transaction.active:
                    settled = True
                    return
                if settlement_action == "commit":
                    transaction.commit(attached=True)
                    settled = True
                    return
                if settlement_action == "restore":
                    settled = transaction.restore()
                    if not settled:
                        raise RuntimeError(
                            "implement mutation rollback failed; journal retained"
                        )
                    return
                settled = transaction.settle_or_restore(corpus)
                if not settled:
                    raise RuntimeError(
                        "implement mutation settlement/rollback failed; "
                        "journal retained"
                    )

            settlement_stack.callback(_settle_transaction)

            def _mark_provider_cleanup_settled() -> None:
                failure_state.provider_cleanup_settled = True

            # ProviderSettlementExitStack closes the provider before callbacks
            # and executes callbacks in LIFO order. This marker therefore
            # distinguishes a later journal callback failure from a still-live
            # provider without inspecting the stack's private state.
            settlement_stack.callback(_mark_provider_cleanup_settled)

            # Durable rollback ownership precedes route construction, ToolSet
            # attachment, and every provider dispatch. Arm the exact journal
            # generation before even entering the provider pool so no SDK or
            # tool worker can predate its durable settlement authority.
            settlement_stack.arm_provider_generation((transaction,))
            pool = settlement_stack.enter_provider_context(
                backend_pool(args, handle.run_dir, cwd=str(cwd))
            )
            implementer_profile = build_autonomous_implementer_profile(ctx)
            backend = pool.get(
                "implementer",
                toolset=implementer_profile,
                native_shell=False,
            )
            observed_backend = _TerminalPrecedenceBackend(
                backend,
                settlement_stack,
            )
            backend_name = observed_backend.name
            backend_model = observed_backend.model
            failure_state.observed_backend = observed_backend

            def _prepare_capture_after_provider_quiescence() -> None:
                settlement_stack.quiesce_providers()
                failure_state.provider_cleanup_settled = True
                transaction.prepare_git_capture()

            budget_exceeded = False

            def _record_invocation_usage(
                *,
                fallback: Any | None = None,
                owner_error: BaseException | None = None,
            ) -> None:
                """Commit one call's usage before any later failure can hide it."""

                nonlocal budget_exceeded
                if failure_state.usage_recorded:
                    return
                if (
                    observed_backend.invocation_result is None
                    and fallback is None
                    and not observed_backend.invocation_started
                ):
                    return
                # Never replay accounting after an ambiguous storage error;
                # publish the latch before projection because malformed
                # result accessors are themselves ambiguous accounting work.
                failure_state.usage_recorded = True

                def retain_accounting_failure(
                    accounting_error: BaseException,
                    missing: tuple[str, ...],
                ) -> None:
                    current = failure_state.terminal_accounting_error
                    if current is None:
                        failure_state.terminal_accounting_error = (
                            accounting_error
                        )
                    elif current is not accounting_error:
                        current.add_note(
                            "secondary Implement accounting failure: "
                            f"{type(accounting_error).__name__}: "
                            f"{str(accounting_error)[:1000]}"
                        )
                    failure_state.terminal_accounting_debt = tuple(sorted(
                        set(failure_state.terminal_accounting_debt)
                        | set(missing)
                    ))
                    if owner_error is not None:
                        owner_error.add_note(
                            "could not record Implement invocation usage: "
                            f"{type(accounting_error).__name__}: "
                            f"{str(accounting_error)[:1000]}"
                        )

                projection_error: BaseException | None = None
                projection_debt: tuple[str, ...] = ()
                try:
                    if observed_backend.invocation_result is not None:
                        usage = _observed_invocation_usage(
                            observed_backend.invocation_result
                        )
                    elif fallback is not None:
                        usage = fallback
                    else:
                        usage = unknown_tool_mutation_usage()
                    if isinstance(usage, _ObservedInvocationUsage):
                        projection_error = usage.projection_error
                        projection_debt = usage.accounting_debt
                except BaseException as accounting_error:  # noqa: BLE001
                    retain_accounting_failure(
                        accounting_error,
                        ("cost_complete", "token_usage_complete"),
                    )
                    return

                try:
                    costs.record(
                        role="implementer",
                        usage=usage,
                        backend_name=backend_name,
                        model=backend_model,
                    )
                except CostLimitExceeded:
                    # CostAccumulator commits the completed call first.
                    budget_exceeded = True
                except BaseException as accounting_error:  # noqa: BLE001
                    retained_error = projection_error or accounting_error
                    if projection_error is not None:
                        projection_error.add_note(
                            "secondary Implement cost-ledger failure: "
                            f"{type(accounting_error).__name__}: "
                            f"{str(accounting_error)[:1000]}"
                        )
                    retain_accounting_failure(
                        retained_error,
                        tuple(sorted(
                            set(projection_debt)
                            | {"cost_complete", "token_usage_complete"}
                        )),
                    )
                    return
                if projection_error is not None:
                    retain_accounting_failure(
                        projection_error,
                        projection_debt
                        or ("cost_complete", "token_usage_complete"),
                    )

            def _terminal_accounting_evidence() -> str:
                return (
                    "incomplete"
                    if (
                        failure_state.terminal_accounting_error is not None
                        or costs.to_dict()["accounting_debt"]
                    )
                    else "complete"
                )

            def _terminal_checkpoint_summary() -> dict[str, Any]:
                summary: dict[str, Any] = {
                    "slice_id": args.slice_id,
                    "costs": costs.to_dict(),
                }
                if failure_state.terminal_accounting_error is not None:
                    summary.update({
                        "accounting_error_kind": type(
                            failure_state.terminal_accounting_error
                        ).__name__,
                        "accounting_error_message": str(
                            failure_state.terminal_accounting_error
                        )[:2000],
                    })
                return summary

            def _checkpoint_pending_terminal(
                terminal_error: ToolMutationIndeterminate,
                cleanup_owner: ProviderSettlementCleanupError,
                *,
                provider_cleanup_pending: bool,
            ) -> None:
                """Best-effort checkpoint without weakening live authority."""

                try:
                    handle.bind_terminal_error(terminal_error)
                    if provider_cleanup_pending:
                        checkpoint_unsettled_tool_mutation_terminal(
                            handle=handle,
                            summary=_terminal_checkpoint_summary(),
                            cost_usd=costs.total_cost_usd,
                            terminal_error=terminal_error,
                            cleanup_error=cleanup_owner,
                            accounting_evidence=(
                                _terminal_accounting_evidence()
                            ),
                            mutation_cleanup_pending=True,
                        )
                    else:
                        checkpoint_settled_tool_mutation_terminal(
                            handle=handle,
                            summary=_terminal_checkpoint_summary(),
                            cost_usd=costs.total_cost_usd,
                            terminal_error=terminal_error,
                            secondary_error=cleanup_owner.cleanup_error,
                            accounting_evidence=(
                                _terminal_accounting_evidence()
                            ),
                            mutation_cleanup_pending=True,
                            journal_cleanup_pending=True,
                        )
                except BaseException as checkpoint_error:  # noqa: BLE001
                    failure_state.evidence_publication_incomplete = True
                    terminal_error.add_note(
                        "could not checkpoint retained mutation cleanup: "
                        f"{type(checkpoint_error).__name__}: "
                        f"{str(checkpoint_error)[:1000]}"
                    )

            def _raise_pending_terminal(
                terminal_error: ToolMutationIndeterminate,
                escaped: BaseException,
                *,
                provider_cleanup_pending: bool,
            ) -> None:
                """Re-publish exact TMI over its direct live cleanup owner."""

                if escaped is terminal_error:
                    cause = escaped.__cause__
                    if (
                        isinstance(cause, ProviderSettlementCleanupError)
                        and cause.owner is settlement_stack
                        and cause.primary_error is terminal_error
                    ):
                        _checkpoint_pending_terminal(
                            terminal_error,
                            cause,
                            provider_cleanup_pending=(
                                provider_cleanup_pending
                            ),
                        )
                        raise terminal_error from cause
                    terminal_error.add_note(
                        "provider settlement violated the exact retained "
                        "cleanup-owner identity contract"
                    )
                    # ``escaped`` is the exact terminal object; chaining it
                    # to itself would create a cause cycle. Preserve its
                    # actual cause and identity unchanged.
                    raise terminal_error
                terminal_error.add_note(
                    "provider settlement violated exact terminal identity: "
                    f"{type(escaped).__name__}: {str(escaped)[:1000]}"
                )
                raise terminal_error from escaped

            def _terminal_provider_barrier(
                terminal_error: ToolMutationIndeterminate,
            ) -> None:
                try:
                    # Register the post-close frontier before teardown starts.
                    # On a close failure the exact no-op and the broader
                    # journal callback stay behind the live retry owner.
                    settlement_stack.run_after_provider_quiescence(lambda: None)
                    failure_state.provider_cleanup_settled = True
                except BaseException as escaped:  # noqa: BLE001
                    _raise_pending_terminal(
                        terminal_error,
                        escaped,
                        provider_cleanup_pending=True,
                    )

            def _restore_terminal_transaction() -> None:
                nonlocal settled
                if not transaction.active:
                    settled = True
                    return
                restored = transaction.restore()
                if not restored:
                    raise RuntimeError(
                        "implement mutation rollback failed; journal retained"
                    )
                settled = True

            def _restore_terminal_or_raise(
                terminal_error: ToolMutationIndeterminate,
            ) -> None:
                try:
                    # ``False`` is converted to an exception inside the
                    # retained action. The provider stack therefore exposes
                    # one explicit retry owner and lexical unwind cannot make
                    # a second implicit rollback attempt.
                    settlement_stack.run_after_provider_quiescence(
                        _restore_terminal_transaction,
                    )
                except BaseException as escaped:  # noqa: BLE001
                    _raise_pending_terminal(
                        terminal_error,
                        escaped,
                        provider_cleanup_pending=False,
                    )

            def _raise_terminal_after_secondary(
                terminal_error: ToolMutationIndeterminate,
                secondary_error: BaseException,
            ) -> None:
                """Restore exactly, checkpoint both causes, then raise TMI."""

                _restore_terminal_or_raise(terminal_error)
                try:
                    handle.bind_terminal_error(terminal_error)
                    checkpoint_settled_tool_mutation_terminal(
                        handle=handle,
                        summary=_terminal_checkpoint_summary(),
                        cost_usd=costs.total_cost_usd,
                        terminal_error=terminal_error,
                        secondary_error=secondary_error,
                        accounting_evidence=_terminal_accounting_evidence(),
                    )
                except BaseException as checkpoint_error:  # noqa: BLE001
                    failure_state.evidence_publication_incomplete = True
                    terminal_error.add_note(
                        "could not checkpoint settled mutation terminal: "
                        f"{type(checkpoint_error).__name__}: "
                        f"{str(checkpoint_error)[:1000]}"
                    )
                raise terminal_error from secondary_error

            raw_terminal = False
            terminal_assay_error: BaseException | None = None
            impl = None
            try:
                impl = implement(
                    spec,
                    args.slice_id,
                    slice_.directive,
                    backend=observed_backend,
                    corpus=corpus,
                    max_turns=args.max_turns,
                    budget_usd=args.budget_usd,
                    allowed_tools=IMPLEMENTER_BUILTIN_TOOLS,
                    prompt_toolset=implementer_profile,
                    cwd=str(cwd),
                    before_capture=_prepare_capture_after_provider_quiescence,
                )
            except ToolMutationIndeterminate as terminal_error:
                raw_terminal = True
                retained_terminal = observed_backend.terminal_error
                if retained_terminal is not terminal_error:
                    # The observing boundary must retain the exact object
                    # before Assay can touch Git. Repair a missing latch with
                    # that same object; a conflicting latch remains noted as
                    # an invariant violation without replacing either owner.
                    if retained_terminal is None:
                        observed_backend._retain(terminal_error)
                    else:
                        terminal_error.add_note(
                            "backend terminal identity changed before the "
                            "standalone workflow caught it"
                        )
                _record_invocation_usage(owner_error=terminal_error)
            except BaseException as assay_error:  # noqa: BLE001
                _record_invocation_usage(owner_error=assay_error)
                if observed_backend.terminal_error is None:
                    # A malformed usage/completeness projection is secondary
                    # evidence. The exact Assay/control-flow owner remains the
                    # exception leaving this invocation.
                    raise
                # The provider already returned a typed TMI. Preserve its
                # exact result/usage when Assay's later capture or artifact
                # processing fails before it can construct an Implementation.
                terminal_assay_error = assay_error
            else:
                impl = preserve_tool_mutation_indeterminate(impl)
                _record_invocation_usage(
                    fallback=impl,
                    owner_error=observed_backend.terminal_error,
                )
                if (
                    failure_state.terminal_accounting_error is not None
                    and observed_backend.terminal_error is None
                ):
                    # No stronger mutation terminal owns this failure. Raise
                    # the exact one-shot projection/accounting error before any
                    # later validation or persistence can replace it.
                    raise failure_state.terminal_accounting_error

            terminal_error = observed_backend.terminal_error
            is_terminal_mutation = (
                terminal_error is not None
                or (
                    impl is not None
                    and impl.classification
                    == TOOL_MUTATION_INDETERMINATE_CLASSIFICATION
                )
                if not raw_terminal
                else True
            )
            if is_terminal_mutation:
                if terminal_error is None:
                    # Defensive workflow-owned identity for an Implementation
                    # injected below the observing backend boundary.
                    terminal_error = ToolMutationIndeterminate(
                        "Implementation retained "
                        "tool_mutation_indeterminate without its invocation "
                        "terminal"
                    )
                    settlement_stack.retain_terminal_precedence(terminal_error)
                settlement_action = "restore"

                try:
                    _terminal_provider_barrier(terminal_error)
                    if failure_state.terminal_accounting_error is not None:
                        _raise_terminal_after_secondary(
                            terminal_error,
                            failure_state.terminal_accounting_error,
                        )
                    if terminal_assay_error is not None:
                        _raise_terminal_after_secondary(
                            terminal_error,
                            terminal_assay_error,
                        )
                    if raw_terminal:
                        transaction.prepare_git_capture()
                        impl = (
                            synthesize_tool_mutation_indeterminate_implementation(
                                corpus=corpus,
                                cwd=cwd,
                                slice_id=args.slice_id,
                                path_scope=write_scope,
                                error=terminal_error,
                                execution=exec_ctx_now(
                                    "implement_tool_mutation_indeterminate",
                                    run_id=handle.run_dir.name,
                                ),
                                provider_quiesced=True,
                                base_commit=spec.project_context.base_commit,
                            )
                        )
                    if impl is None:
                        raise RuntimeError(
                            "mutation terminal has no Implementation after "
                            "post-provider processing"
                        )
                    impl = preserve_tool_mutation_indeterminate(impl)
                    impl = transaction.validate_implementation(impl)
                    # Completion evidence remains useful attribution, but it
                    # cannot replace the stronger mutation terminal.
                    impl, completion_disposition = (
                        normalize_implementer_completion(corpus, impl)
                    )
                    transaction.prepare(impl, keep=False)
                    committed_spec, implementation_ref = persist_and_attach(
                        corpus,
                        impl,
                        kind="implementations",
                        attachment="implementation",
                    )
                    _restore_terminal_or_raise(terminal_error)

                    terminal_summary = {
                        "slice_id": impl.slice_id,
                        "classification": impl.classification,
                        "base_commit": impl.base_commit,
                        "result_commit": impl.result_commit,
                        "files_modified": list(impl.files_modified),
                        "diff_ref": impl.diff_ref,
                        "transcript_ref": impl.transcript_ref,
                        "implementation_ref": implementation_ref,
                        "spec_version": committed_spec.version,
                        "execution": asdict(impl.execution),
                        "implementer_terminal_disposition": (
                            completion_disposition
                        ),
                        "accounting_evidence": (
                            _terminal_accounting_evidence()
                        ),
                        "costs": costs.to_dict(),
                    }
                    handle.write_summary(terminal_summary)
                    handle.bind_terminal_error(terminal_error)
                    handle.finalize_elapsed(RunResult(
                        status=TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
                        cost_usd=costs.total_cost_usd,
                        error_kind=type(terminal_error).__name__,
                        error_message=str(terminal_error)[:2000],
                    ))
                except ToolMutationIndeterminate as escaped_terminal:
                    if escaped_terminal is terminal_error:
                        raise
                    _raise_terminal_after_secondary(
                        terminal_error,
                        escaped_terminal,
                    )
                except BaseException as secondary_error:  # noqa: BLE001
                    _raise_terminal_after_secondary(
                        terminal_error,
                        secondary_error,
                    )
                # This normal terminal return is reachable only after exact
                # rollback and immutable run finalization both succeeded.
                return 1

            # Provider/tool descendants must be gone before validation captures
            # final bytes or any path can settle the durable transaction.
            settlement_stack.quiesce_providers()
            impl = transaction.validate_implementation(impl)
            if failure_state.terminal_accounting_error is not None:
                raise failure_state.terminal_accounting_error
            if budget_exceeded:
                # The completed call was recorded before this point; preserve
                # its evidence but never classify an over-budget attempt as
                # successful.
                impl = replace(
                    impl,
                    classification="hard_fail_budget_exhausted",
                )
            if impl.accounting_debt or costs.to_dict()["accounting_debt"]:
                # Missing provider token/cost evidence makes the observed
                # spend only a lower bound. Preserve the diagnostic
                # Implementation, but never let its nominal classification
                # authorize retaining workspace mutations.
                impl = replace(
                    impl,
                    classification="hard_fail_incomplete_evidence",
                )
            impl, completion_disposition = normalize_implementer_completion(
                corpus,
                impl,
            )
            keep_workspace = impl.classification.startswith("success")
            transaction.prepare(impl, keep=keep_workspace)
            committed_spec, implementation_ref = persist_and_attach(
                corpus,
                impl,
                kind="implementations",
                attachment="implementation",
            )
            settlement_action = "commit" if keep_workspace else "restore"

        summary = {
            "slice_id": impl.slice_id,
            "classification": impl.classification,
            "base_commit": impl.base_commit,
            "result_commit": impl.result_commit,
            "files_modified": list(impl.files_modified),
            "diff_ref": impl.diff_ref,
            "transcript_ref": impl.transcript_ref,
            "implementation_ref": implementation_ref,
            "spec_version": committed_spec.version,
            "execution": asdict(impl.execution),
            "implementer_terminal_disposition": completion_disposition,
            "costs": costs.to_dict(),
        }
        handle.write_summary(summary)
        logger.info(
            "implement: slice=%s classification=%s",
            impl.slice_id, impl.classification,
        )
        handle.finalize_elapsed(RunResult(
            status=impl.classification,
            cost_usd=costs.total_cost_usd,
        ))
        # Exit non-zero on hard fail so shells / job runners can branch.
        return 0 if impl.classification.startswith("success") else 1


if __name__ == "__main__":
    sys.exit(main())
