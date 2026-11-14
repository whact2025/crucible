# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Typed workflow outcomes, provenance manifests, and parent/child sagas.

This module makes workflow ownership and terminal meaning durable:

* every run publishes a private ``run_manifest.json`` before work begins;
* every terminal run publishes a schema-checked ``outcome.json``;
* parent workflows register a child operation before dispatch, and the child
  binds its run directory to that operation through a ``ContextVar``;
* startup reconciliation marks manifests whose owning process disappeared as
  interrupted instead of leaving an apparently-running run forever.

Detailed workflow statuses remain useful diagnostics.  ``TerminalClass`` is
the cross-workflow control-flow authority and has a deliberately small,
exhaustive vocabulary.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import datetime as dt
import enum
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import secrets
import socket
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


_MANIFEST_SCHEMA = "crucible.run-manifest.v1"
_OUTCOME_SCHEMA = "crucible.workflow-outcome.v1"
_SAGA_SCHEMA = "crucible.workflow-saga.v1"
_PROVENANCE_ROOT_UNSET = object()

TOOL_MUTATION_INDETERMINATE_STOP_REASON = "tool_mutation_indeterminate"
TOOL_MUTATION_INDETERMINATE_CLASSIFICATION = "hard_fail_tool_mutation_indeterminate"
_TOOL_MUTATION_EVIDENCE_MAX_DIFF_BYTES = 8 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class UnknownToolMutationUsage:
    """Conservative lower-bound accounting for an exception-only call.

    A provider can prove the mutation terminal while its exception path has
    no trustworthy token or pricing envelope.  Zeroes below are lower bounds,
    never claims that the paid call was free; ``accounting_debt`` is the
    machine-readable authority that prevents later paid work.
    """

    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    accounting_debt: tuple[str, ...] = (
        "token_usage_complete",
        "cost_complete",
    )
    pricing_evidence: None = None


def unknown_tool_mutation_usage() -> UnknownToolMutationUsage:
    """Return explicit incomplete accounting for one terminal provider call."""

    return UnknownToolMutationUsage()


def _is_content_addressed_ref(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(char in "0123456789abcdef" for char in value[7:])
    )


class TerminalClass(str, enum.Enum):
    """Exhaustive cross-workflow terminal taxonomy."""

    SUCCEEDED = "succeeded"
    NO_WORK = "no_work"
    INCONCLUSIVE = "inconclusive"
    RETRYABLE_FAILURE = "retryable_failure"
    PERMANENT_FAILURE = "permanent_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    CONTRACT_ERROR = "contract_error"


_SUCCESS = frozenset({
    "accepted",
    "complete",
    "completed",
    "conclusive",
    "deepswe_complete",
    "dry-run",
    "dry_run_complete",
    "implement_complete",
    "landed",
    "no_attacks",
    "ok",
    "pass",
    "passed",
    "proceed",
    "resolved",
    "success",
})
_NO_WORK = frozenset({
    "no_instances_matched",
    "no_measurements",
    "no_tasks_matched",
    "no_work",
    "skipped_disabled",
    "skipped_empty",
})
_INCONCLUSIVE = frozenset({
    "budget_accounting_incomplete",
    "cost_incomplete",
    "critic_failures_exhausted",
    "critic_infrastructure_failure",
    "critic_protocol_failure",
    "incomplete_accounting",
    "inconclusive",
    "inconclusive_breaker",
    "inconclusive_authority",
    "inconclusive_judge",
    "inconclusive_post_strip_verify",
    "inconclusive_transport_failed",
    "inconclusive_workspace_mismatch",
    "insufficient_input",
    "judge_failures_exhausted",
    "judge_infrastructure_failure",
    "judge_protocol_failure",
    "lifecycle_output_invalid",
    "no_mode_selected",
    "partial_stopped",
    "planner_protocol_failure",
})
_BUDGET = frozenset({
    "aggregate_budget_exceeded",
    "aggregate_budget_reached",
    "budget_exhausted",
    "hard_fail_budget_exhausted",
})
_CANCELLED = frozenset({"cancelled", "canceled"})
_RETRYABLE = frozenset({
    "all_slices_transport_failed",
    "backend_error",
    "concurrent_spec_change",
    "hard_fail_timeout",
    "hard_fail_transport",
    "incomplete_transport_failed",
    "retryable_tasks_failed",
    "retryable_tier_failure",
    "task_lease_busy",
    "task_lease_error",
    "transport_failed",
})
_PERMANENT = frozenset({
    "bench_error",
    "breach_exception",
    "cascade_error",
    "cleanup_failed",
    "container_preflight_failed",
    "container_runtime_required",
    "dataset_invalid",
    "dataset_missing",
    "dataset_unavailable",
    "empty_union_diff",
    "failed",
    "hard_fail_content_rejected",
    "hard_fail_max_turns",
    "hard_fail_mutation_scope",
    "hard_fail_storage_quota",
    TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
    "harness_failed",
    "incomplete",
    "inflation_veto",
    "judge_rejected",
    "max_rounds_exhausted",
    "missing_dep",
    "no_candidate",
    "partial_failed",
    "partial_failure",
    "patch_apply_failed",
    "persistence_veto",
    "planner_contract_failure",
    "reject",
    "rejected",
    "review_failed",
    "revise",
    "rounds_exhausted",
    "runner_error",
    "setup_error",
    "setup_failed",
    "slice_failed",
    "smoke_regression",
    "storage_quota_exceeded",
    "tasks_failed",
    "tier_exception",
    "unresolved",
    "validate_blocked",
    "verification_failed",
    "weighted_severity_veto",
})


def is_tool_mutation_indeterminate_result(result: object) -> bool:
    """Return whether a backend reported the terminal mutation subtype."""

    return (
        getattr(result, "stop_reason", None)
        == TOOL_MUTATION_INDETERMINATE_STOP_REASON
    )


def preserve_tool_mutation_indeterminate(implementation: Any) -> Any:
    """Restore Crucible's permanent subtype after Assay transport mapping.

    Assay deliberately maps every ``TransportError`` to
    ``hard_fail_transport``.  Crucible mutation adapters additionally retain
    the non-forgeable workflow-owned stop reason when source settlement is
    indeterminate.  Workflow retry decisions must use that more specific
    cause instead of treating the call as an ordinary transient transport
    failure.
    """

    if not is_tool_mutation_indeterminate_result(implementation):
        return implementation
    if (
        getattr(implementation, "classification", None)
        == TOOL_MUTATION_INDETERMINATE_CLASSIFICATION
    ):
        return implementation
    return dataclasses.replace(
        implementation,
        classification=TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
    )


def synthesize_tool_mutation_indeterminate_implementation(
    *,
    corpus: Any,
    cwd: str | Path,
    slice_id: str,
    path_scope: tuple[str, ...],
    error: BaseException,
    execution: Any,
    provider_quiesced: bool,
    base_commit: str | None = None,
) -> Any:
    """Capture a durable terminal Implementation after positive quiescence.

    Exception-only mutation terminals have no Assay ``Implementation`` to
    persist.  The workflow may synthesize one only after the exact provider
    generation has positively quiesced.  The bounded capture is therefore a
    stable postimage, not a race with an adapter or tool worker that may still
    be mutating the checkout.
    """

    if provider_quiesced is not True:
        raise ValueError(
            "tool-mutation terminal evidence requires positive provider quiescence"
        )

    from assay import ExecutionCompleteness, Implementation
    from assay.primitives.implement import GitCaptureError, capture_workspace_delta

    capture_error: BaseException | None = None
    if path_scope:
        try:
            diff, files, captured_base = capture_workspace_delta(
                cwd=str(cwd),
                path_scope=path_scope,
                max_diff_bytes=_TOOL_MUTATION_EVIDENCE_MAX_DIFF_BYTES,
            )
        except GitCaptureError as exc:
            # The exact mutation terminal outranks secondary Git evidence
            # failure.  Persist an explicit empty lower-bound capture and let
            # the durable journal restore the actual bytes on unwind.
            capture_error = exc
            diff, files, captured_base = (
                b"",
                (),
                base_commit or "unknown",
            )
    else:
        # An empty mutation authority must not widen into Assay's unscoped
        # Git capture.  No provider should normally reach this path, but the
        # fail-closed record remains well-formed if one does.
        diff, files, captured_base = b"", (), base_commit or "unknown"

    message = str(error)[:2000]
    cause = error.__cause__
    transcript = {
        "schema_version": "crucible.tool-mutation-indeterminate.v1",
        "slice_id": slice_id,
        "classification": TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
        "stop_reason": TOOL_MUTATION_INDETERMINATE_STOP_REASON,
        "provider_quiesced": True,
        "error_kind": type(error).__name__,
        "error_message": message,
        "cause_kind": type(cause).__name__ if cause is not None else None,
        "capture_error_kind": (
            type(capture_error).__name__
            if capture_error is not None
            else None
        ),
        "capture_error_message": (
            str(capture_error)[:2000]
            if capture_error is not None
            else None
        ),
        "accounting_evidence": {
            "token_usage_complete": False,
            "cost_complete": False,
            "pricing_evidence": None,
            "numeric_values_are_lower_bounds": True,
        },
    }
    diff_ref = corpus.put_artifact(
        diff,
        kind="diffs",
        name_hint=f"{slice_id}_tool_mutation_indeterminate.diff",
    )
    transcript_ref = corpus.put_artifact(
        json.dumps(
            transcript,
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8"),
        kind="transcripts",
        name_hint=f"{slice_id}_tool_mutation_indeterminate.json",
    )
    return Implementation(
        slice_id=slice_id,
        base_commit=captured_base,
        result_commit=None,
        diff_ref=diff_ref,
        transcript_ref=transcript_ref,
        files_modified=files,
        classification=TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
        execution=execution,
        scope_escapes=(),
        artifacts={
            "tool_mutation_indeterminate": transcript_ref,
        },
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        stop_reason=TOOL_MUTATION_INDETERMINATE_STOP_REASON,
        execution_completeness=ExecutionCompleteness(
            token_usage_complete=False,
            cost_complete=False,
            terminal_complete=True,
            trace_complete=False,
        ),
        pricing_evidence=None,
    )


def _validate_tool_mutation_accounting_evidence(value: str) -> None:
    """Reject ambiguous accounting claims at the terminal boundary."""

    if value not in {"complete", "incomplete"}:
        raise ValueError(
            "accounting_evidence must be exactly 'complete' or 'incomplete'"
        )


def checkpoint_unsettled_tool_mutation_terminal(
    *,
    handle: Any,
    summary: Mapping[str, Any],
    cost_usd: float,
    terminal_error: BaseException,
    cleanup_error: BaseException,
    accounting_evidence: str = "incomplete",
    mutation_cleanup_pending: bool = True,
) -> None:
    """Commit the exact terminal cause while cleanup authority stays live.

    This helper deliberately performs no Git inspection and creates no
    Implementation.  The caller must re-raise the terminal error with the
    retained provider cleanup owner as its cause; only that owner can retry
    the exact generation and eventually release journal rollback callbacks.
    """

    _validate_tool_mutation_accounting_evidence(accounting_evidence)
    if getattr(handle, "finalized", False):
        return
    from assay.corpus.runs import RunResult

    message = str(terminal_error)[:2000]
    payload = dict(summary)
    payload.update({
        "status": TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
        "classification": TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
        "stop_reason": TOOL_MUTATION_INDETERMINATE_STOP_REASON,
        "error_kind": type(terminal_error).__name__,
        "error_message": message,
        "provider_cleanup_pending": True,
        "mutation_cleanup_pending": mutation_cleanup_pending,
        "journal_cleanup_pending": mutation_cleanup_pending,
        "cleanup_pending": True,
        "provider_cleanup_error_kind": type(cleanup_error).__name__,
        "accounting_evidence": accounting_evidence,
        "cost_usd": cost_usd,
    })
    handle.write_summary(payload)
    handle.finalize_elapsed(RunResult(
        status=TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
        cost_usd=cost_usd,
        error_kind=type(terminal_error).__name__,
        error_message=message,
    ))


def checkpoint_settled_tool_mutation_terminal(
    *,
    handle: Any,
    summary: Mapping[str, Any],
    cost_usd: float,
    terminal_error: BaseException,
    secondary_error: BaseException | None = None,
    accounting_evidence: str = "incomplete",
    mutation_cleanup_pending: bool = False,
    journal_cleanup_pending: bool = False,
) -> None:
    """Commit exact TMI precedence after positive provider quiescence.

    Git capture, validation, artifact persistence, and journal settlement are
    secondary evidence/cleanup operations once a provider call has reached the
    non-retryable mutation-indeterminate terminal.  A failure in any of those
    operations must remain visible without replacing that terminal or implying
    another paid call is safe.  Unlike the unsettled checkpoint above, this
    record proves that the provider generation is no longer live.
    """

    _validate_tool_mutation_accounting_evidence(accounting_evidence)
    if getattr(handle, "finalized", False):
        return
    from assay.corpus.runs import RunResult

    message = str(terminal_error)[:2000]
    payload = dict(summary)
    payload.update({
        "status": TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
        "classification": TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
        "stop_reason": TOOL_MUTATION_INDETERMINATE_STOP_REASON,
        "error_kind": type(terminal_error).__name__,
        "error_message": message,
        "provider_cleanup_pending": False,
        "mutation_cleanup_pending": mutation_cleanup_pending,
        "journal_cleanup_pending": journal_cleanup_pending,
        "cleanup_pending": (
            mutation_cleanup_pending
            or journal_cleanup_pending
        ),
        "secondary_error_kind": (
            type(secondary_error).__name__
            if secondary_error is not None
            else None
        ),
        "secondary_error_message": (
            str(secondary_error)[:2000]
            if secondary_error is not None
            else None
        ),
        "accounting_evidence": accounting_evidence,
        "cost_usd": cost_usd,
    })
    handle.write_summary(payload)
    handle.finalize_elapsed(RunResult(
        status=TOOL_MUTATION_INDETERMINATE_CLASSIFICATION,
        cost_usd=cost_usd,
        error_kind=type(terminal_error).__name__,
        error_message=message,
    ))


def classify_terminal_status(status: str, *, error_kind: str | None = None) -> TerminalClass:
    """Map a detailed status to the exhaustive control-flow taxonomy.

    Unknown values are a contract error, never silently treated as success or
    as an ordinary failure.  This makes newly introduced statuses visible in
    tests and run postmortems until they are deliberately classified here.
    """

    normalized = str(status).strip().lower()
    if normalized in _SUCCESS:
        return TerminalClass.SUCCEEDED
    if normalized in _NO_WORK:
        return TerminalClass.NO_WORK
    if normalized in _INCONCLUSIVE:
        return TerminalClass.INCONCLUSIVE
    if normalized in _BUDGET:
        return TerminalClass.BUDGET_EXHAUSTED
    if normalized in _CANCELLED:
        return TerminalClass.CANCELLED
    if normalized in _RETRYABLE or normalized.startswith("hard_fail_transport"):
        return TerminalClass.RETRYABLE_FAILURE
    if normalized in _PERMANENT or normalized.startswith("hard_fail_"):
        return TerminalClass.PERMANENT_FAILURE
    if error_kind and error_kind.lower() in {
        "timeouterror", "connectionerror", "transporterror",
    }:
        return TerminalClass.RETRYABLE_FAILURE
    return TerminalClass.CONTRACT_ERROR


@dataclasses.dataclass(frozen=True)
class WorkflowOutcome:
    schema_version: str
    workflow: str
    run_id: str
    terminal_class: TerminalClass
    detail_status: str
    cost_usd: float
    spec_version: int | None
    refs: tuple[str, ...]
    error_kind: str | None
    error_message: str | None
    parent_run_id: str | None
    operation_id: str | None
    completed_at: str
    implementation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != _OUTCOME_SCHEMA:
            raise ValueError("unsupported WorkflowOutcome schema")
        if not self.workflow or not self.run_id or not self.detail_status:
            raise ValueError("workflow, run_id, and detail_status are required")
        if any(separator in self.run_id for separator in ("/", "\\")):
            raise ValueError("run_id must be a path-free identifier")
        if bool(self.parent_run_id) != bool(self.operation_id):
            raise ValueError("parent_run_id and operation_id must be set together")
        if not math.isfinite(self.cost_usd) or self.cost_usd < 0:
            raise ValueError("cost_usd must be finite and non-negative")
        if self.spec_version is not None and self.spec_version < 0:
            raise ValueError("spec_version cannot be negative")
        if not isinstance(self.implementation_refs, tuple):
            raise ValueError("WorkflowOutcome.implementation_refs must be a tuple")
        for ref in (*self.refs, *self.implementation_refs):
            if not _is_content_addressed_ref(ref):
                raise ValueError(f"invalid content-addressed ref: {ref!r}")
        if len(set(self.implementation_refs)) != len(
            self.implementation_refs
        ):
            raise ValueError("WorkflowOutcome.implementation_refs has duplicates")
        try:
            completed = dt.datetime.fromisoformat(self.completed_at)
        except ValueError as exc:
            raise ValueError("completed_at must be an ISO-8601 timestamp") from exc
        if completed.tzinfo is None:
            raise ValueError("completed_at must include a timezone")

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["terminal_class"] = self.terminal_class.value
        value["refs"] = list(self.refs)
        value["implementation_refs"] = list(self.implementation_refs)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkflowOutcome":
        required = {
            "schema_version", "workflow", "run_id", "terminal_class",
            "detail_status", "cost_usd", "spec_version", "refs", "error_kind",
            "error_message", "parent_run_id", "operation_id", "completed_at",
        }
        allowed = required | {"implementation_refs"}
        if not required.issubset(value) or not set(value).issubset(allowed):
            raise ValueError(
                "WorkflowOutcome fields differ: "
                f"missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - allowed)}"
            )
        refs = value["refs"]
        if not isinstance(refs, list) or not all(isinstance(item, str) for item in refs):
            raise ValueError("WorkflowOutcome.refs must be a string list")
        implementation_refs = value.get("implementation_refs", [])
        if not isinstance(implementation_refs, list) or not all(
            isinstance(item, str) for item in implementation_refs
        ):
            raise ValueError(
                "WorkflowOutcome.implementation_refs must be a string list"
            )
        required_strings = ("schema_version", "workflow", "run_id", "detail_status")
        for field_name in required_strings:
            if not isinstance(value[field_name], str):
                raise ValueError(f"WorkflowOutcome.{field_name} must be a string")
        for field_name in (
            "error_kind",
            "error_message",
            "parent_run_id",
            "operation_id",
        ):
            if value[field_name] is not None and not isinstance(value[field_name], str):
                raise ValueError(
                    f"WorkflowOutcome.{field_name} must be a string or null"
                )
        if not isinstance(value["terminal_class"], str):
            raise ValueError("WorkflowOutcome.terminal_class must be a string")
        if isinstance(value["cost_usd"], bool) or not isinstance(
            value["cost_usd"], (int, float)
        ):
            raise ValueError("WorkflowOutcome.cost_usd must be numeric")
        if value["spec_version"] is not None and (
            isinstance(value["spec_version"], bool)
            or not isinstance(value["spec_version"], int)
        ):
            raise ValueError("WorkflowOutcome.spec_version must be an integer or null")
        if not isinstance(value["completed_at"], str):
            raise ValueError("WorkflowOutcome.completed_at must be a string")
        return cls(
            schema_version=value["schema_version"],
            workflow=value["workflow"],
            run_id=value["run_id"],
            terminal_class=TerminalClass(value["terminal_class"]),
            detail_status=value["detail_status"],
            cost_usd=float(value["cost_usd"]),
            spec_version=value["spec_version"],
            refs=tuple(refs),
            error_kind=value["error_kind"],
            error_message=value["error_message"],
            parent_run_id=value["parent_run_id"],
            operation_id=value["operation_id"],
            completed_at=value["completed_at"],
            implementation_refs=tuple(implementation_refs),
        )


@dataclasses.dataclass(frozen=True)
class ChildOperation:
    parent_run_dir: Path
    parent_run_id: str
    operation_id: str
    workflow: str


_CHILD_OPERATION: contextvars.ContextVar[ChildOperation | None] = (
    contextvars.ContextVar("crucible_child_operation", default=None)
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / (
        f".tmp_{path.name}_{os.getpid()}_{secrets.token_hex(4)}"
    )
    data = json.dumps(value, indent=2, sort_keys=True, default=str).encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _locked_json(path: Path, *, default: dict[str, Any]) -> Iterator[dict[str, Any]]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"{path} must contain a JSON object")
            value = loaded
        else:
            value = default
        yield value
        _atomic_json(path, value)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _pid_start_token(pid: int) -> str | None:
    """Return Linux's immutable process start-tick identity for ``pid``.

    ``/proc/<pid>/stat`` encloses the command name in parentheses and that name
    may itself contain spaces.  Splitting the complete line shifts field 22
    and can make a reused PID look like the original owner, so parse after the
    final closing parenthesis and index relative to field 3 instead.
    """

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        command_end = raw.rfind(")")
        if command_end < 0:
            return None
        fields_after_command = raw[command_end + 1 :].split()
        # The suffix starts at field 3 (state); starttime is field 22.
        token = fields_after_command[19]
        int(token)
        return token
    except (OSError, IndexError, ValueError):
        return None


def _process_is_owner(pid: int, start_token: str | None) -> bool:
    if pid <= 0 or start_token is None:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return _pid_start_token(pid) == start_token


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata(
    root: Path,
    *,
    bind_workspace_state: bool = False,
    workspace_state_excluded_roots: Iterable[Path | str] = (),
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "root": str(root),
        "git_root": None,
        "commit": None,
        "dirty": None,
    }
    try:
        resolved_root = root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        if bind_workspace_state:
            metadata.update({
                "workspace_state_sha256": None,
                "workspace_state_complete": False,
            })
        return metadata
    metadata["root"] = str(resolved_root)

    from ._mutation_journal import (  # noqa: PLC0415
        MutationJournalError,
        _git_argv,
        _reject_git_redirect_env,
        _sanitized_git_env,
    )

    try:
        _reject_git_redirect_env()
    except MutationJournalError:
        if bind_workspace_state:
            metadata.update({
                "workspace_state_sha256": None,
                "workspace_state_complete": False,
            })
        return metadata
    command = _git_argv()
    git_env = _sanitized_git_env()

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                [*command, *args],
                cwd=resolved_root,
                env=git_env,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    git_root = run("rev-parse", "--show-toplevel")
    if git_root:
        try:
            metadata["git_root"] = str(Path(git_root).resolve(strict=True))
        except (OSError, RuntimeError):
            metadata["git_root"] = None
    commit = run("rev-parse", "--verify", "HEAD^{commit}")
    status = run("status", "--porcelain=v1", "--untracked-files=no")
    metadata.update({
        "commit": commit.lower() if commit else None,
        "dirty": bool(status) if status is not None else None,
    })
    if bind_workspace_state:
        try:
            digest, complete = _workspace_state_digest(
                resolved_root,
                excluded_roots=workspace_state_excluded_roots,
            )
        except ValueError:
            # Run-envelope provenance is diagnostic and must remain writable
            # when a workflow rejects hostile ambient Git identity controls.
            # The authoritative Plan admission boundary calls the public
            # helper and still fails closed; this projection records only
            # that workspace evidence was unavailable.
            digest, complete = None, False
        metadata["workspace_state_sha256"] = digest
        metadata["workspace_state_complete"] = complete
    return metadata


def _workspace_state_digest(
    root: Path,
    *,
    excluded_roots: Iterable[Path | str] = (),
) -> tuple[str | None, bool]:
    """Hash tracked diffs and every untracked file without retaining content."""

    from ._mutation_journal import (  # noqa: PLC0415
        MutationJournalError,
        _git_argv,
        _reject_git_redirect_env,
        _sanitized_git_env,
    )

    try:
        _reject_git_redirect_env()
    except MutationJournalError as exc:
        raise ValueError(str(exc)) from exc
    command = _git_argv()
    git_env = _sanitized_git_env()
    try:
        workspace = root.expanduser().resolve(strict=True)
        top = subprocess.run(
            [*command, "rev-parse", "--show-toplevel"],
            cwd=workspace,
            env=git_env,
            capture_output=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        repo = Path(os.fsdecode(top)).resolve(strict=True)
        exclusions: list[Path] = []
        for raw_exclusion in excluded_roots:
            exclusion = Path(raw_exclusion).expanduser()
            if not exclusion.is_absolute():
                exclusion = workspace / exclusion
            # Keep the lexical path until admission is complete. Resolving
            # first would let an operator-controlled symlink outside the
            # checkout alias an arbitrary in-repository path into the
            # exclusion set.
            try:
                relative_exclusion = exclusion.relative_to(repo)
            except ValueError:
                try:
                    resolved_exclusion = exclusion.resolve(strict=False)
                except (OSError, RuntimeError) as exc:
                    raise ValueError(
                        "workspace exclusion cannot be resolved safely"
                    ) from exc
                try:
                    resolved_exclusion.relative_to(repo)
                except ValueError:
                    # State outside this Git repository does not affect its
                    # mutable-state digest and needs no Git pathspec.
                    continue
                raise ValueError(
                    "workspace exclusion aliases into the repository through a symlink"
                ) from None
            # Parent traversal is ambiguous after crossing a symlink. It is
            # irrelevant for a proven out-of-repository state root, but an
            # in-repository governed root must remain purely lexical.
            if ".." in relative_exclusion.parts:
                raise ValueError("workspace exclusion contains a parent traversal")
            if not relative_exclusion.parts:
                raise ValueError("workspace exclusion cannot cover the repository root")
            current = repo
            for index, part in enumerate(relative_exclusion.parts):
                current /= part
                try:
                    info = current.lstat()
                except FileNotFoundError:
                    break
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError(
                        f"workspace exclusion resolves through a symlink: {current}"
                    )
                if not stat.S_ISDIR(info.st_mode):
                    kind = (
                        "root"
                        if index == len(relative_exclusion.parts) - 1
                        else "ancestor"
                    )
                    raise ValueError(
                        f"workspace exclusion {kind} is not a directory: {current}"
                    )
            exclusions.append(relative_exclusion)
        exclusions = sorted(set(exclusions), key=lambda path: path.parts)

        indexed = subprocess.run(
            [*command, "ls-files", "--stage", "-z"],
            cwd=repo,
            env=git_env,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout.split(b"\0")
        index_paths: list[Path] = []
        for record in (item for item in indexed if item):
            _metadata, separator, relative_bytes = record.partition(b"\t")
            if not separator or not relative_bytes:
                raise ValueError("Git index returned a malformed path record")
            relative = Path(os.fsdecode(relative_bytes))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Git index returned an unsafe path")
            index_paths.append(relative)
        for exclusion in exclusions:
            overlap = next(
                (
                    indexed_path
                    for indexed_path in index_paths
                    if (
                        indexed_path == exclusion
                        or exclusion in indexed_path.parents
                        or indexed_path in exclusion.parents
                    )
                ),
                None,
            )
            if overlap is not None:
                raise ValueError(
                    f"workspace exclusion overlaps tracked/index content: {overlap}"
                )

        diff_command = [
            *command,
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
            ".",
        ]
        for exclusion in exclusions:
            exclusion_pathspec = exclusion.as_posix()
            # ``literal`` prevents `[`, `?`, or `*` in a governed directory
            # name from widening into a Git wildcard. ``top`` makes the
            # exclusion repository-root relative even when ``root`` is not.
            diff_command.append(
                f":(top,exclude,literal){exclusion_pathspec}"
            )
        tracked_diff = subprocess.run(
            diff_command,
            cwd=repo,
            env=git_env,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
        untracked = subprocess.run(
            [*command, "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo,
            env=git_env,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout.split(b"\0")
        hasher = hashlib.sha256()
        hasher.update(b"tracked-diff\0")
        hasher.update(hashlib.sha256(tracked_diff).digest())
        for relative_bytes in sorted(item for item in untracked if item):
            relative = Path(os.fsdecode(relative_bytes))
            if relative.is_absolute() or ".." in relative.parts:
                return None, False
            if any(
                relative == exclusion
                or exclusion in relative.parents
                for exclusion in exclusions
            ):
                continue
            candidate = repo / relative
            info = candidate.lstat()
            hasher.update(b"untracked\0")
            hasher.update(len(relative_bytes).to_bytes(8, "big"))
            hasher.update(relative_bytes)
            hasher.update(stat.S_IFMT(info.st_mode).to_bytes(8, "big"))
            hasher.update((info.st_mode & 0o111).to_bytes(2, "big"))
            if stat.S_ISLNK(info.st_mode):
                target = os.fsencode(os.readlink(candidate))
                hasher.update(len(target).to_bytes(8, "big"))
                hasher.update(target)
            elif stat.S_ISREG(info.st_mode):
                hasher.update(info.st_size.to_bytes(8, "big"))
                with candidate.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        hasher.update(chunk)
            else:
                # Git can retain symlinks and regular files. Other untracked
                # node types are not reproducible workflow inputs.
                return None, False
        return "sha256:" + hasher.hexdigest(), True
    except (OSError, subprocess.SubprocessError):
        return None, False


def workspace_state_digest(
    root: Path | str,
    *,
    excluded_roots: Iterable[Path | str] = (),
) -> str:
    """Return a complete digest of planner-visible mutable Git state.

    The digest covers the tracked diff from ``HEAD`` plus every non-ignored
    untracked path and its content.  Callers that use it for reuse admission
    must bind the checked-out commit separately; a clean tree has the same
    mutable-state digest at every commit. In-repository exclusions are
    admitted only when their lexical path has no symlink component and their
    entire subtree is absent from the Git index.
    """

    digest, complete = _workspace_state_digest(
        Path(root),
        excluded_roots=excluded_roots,
    )
    if not complete or digest is None:
        raise ValueError(
            "workspace state could not be fingerprinted completely"
        )
    return digest


def _sha256_if_file(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _spec_input_identities(spec: Any | None) -> dict[str, Any]:
    if spec is None:
        return {
            "spec": None,
            "spec_version": None,
            "spec_root": None,
            "directives": {},
        }
    from assay.canonical import hash_record  # noqa: PLC0415

    directives: dict[str, str] = {}
    dag = getattr(spec, "tier_dag", None)
    for tier in getattr(dag, "tiers", ()) if dag is not None else ():
        for slice_ in tier.slices:
            directives[f"{tier.tier_id}/{slice_.id}"] = hash_record(
                slice_.directive
            )
    root = getattr(spec, "root", None)
    return {
        "spec": hash_record(spec),
        "spec_version": getattr(spec, "version", None),
        "spec_root": str(root) if root is not None else None,
        "directives": directives,
    }


def _target_workspace_provenance(
    provenance_root: Path | str | None | object,
    *,
    excluded_roots: Iterable[Path | str] = (),
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve one target checkout without treating ambient cwd as authority."""

    try:
        ambient = Path.cwd().resolve(strict=True)
    except (OSError, RuntimeError):
        ambient = Path.cwd().absolute()
    if provenance_root is _PROVENANCE_ROOT_UNSET:
        binding = "implicit_ambient_legacy"
        target: Path | None = ambient
    elif provenance_root is None:
        binding = "none"
        target = None
    else:
        if not isinstance(provenance_root, (str, Path)):
            raise TypeError("provenance_root must be a path, string, or None")
        binding = "explicit"
        target = Path(provenance_root)

    if target is None:
        workspace: dict[str, Any] = {
            "binding": binding,
            "root": None,
            "git_root": None,
            "commit": None,
            "dirty": None,
            "workspace_state_sha256": None,
            "workspace_state_complete": False,
            "identity_complete": False,
        }
    else:
        workspace = _git_metadata(
            target,
            bind_workspace_state=True,
            workspace_state_excluded_roots=excluded_roots,
        )
        digest = workspace.get("workspace_state_sha256")
        commit = workspace.get("commit")
        workspace["binding"] = binding
        workspace["identity_complete"] = bool(
            binding == "explicit"
            and workspace.get("workspace_state_complete") is True
            and isinstance(digest, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            and isinstance(commit, str)
            and re.fullmatch(r"[0-9a-f]{40,64}", commit)
        )
    return workspace, {"cwd": str(ambient)}


def _provenance(
    spec: Any | None = None,
    *,
    provenance_root: Path | str | None | object = _PROVENANCE_ROOT_UNSET,
    excluded_roots: Iterable[Path | str] = (),
) -> dict[str, Any]:
    crucible_root = Path(__file__).resolve().parents[1]
    assay_root = crucible_root.parent / "assay"
    workspace, ambient = _target_workspace_provenance(
        provenance_root,
        excluded_roots=excluded_roots,
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "packages": {
            "crucible": _package_version("crucible"),
            "assay": _package_version("assay"),
        },
        "source": {
            "crucible": _git_metadata(crucible_root),
            "assay": _git_metadata(assay_root) if assay_root.is_dir() else None,
            "workspace": workspace,
            "ambient": ambient,
        },
        "inputs": {
            "uv_lock": _sha256_if_file(crucible_root / "uv.lock"),
            "rubric_manifest": _sha256_if_file(
                crucible_root / "rubrics" / "_manifest.yaml"
            ),
            "backend_config": _sha256_if_file(Path(value).expanduser())
            if (value := os.environ.get("CRUCIBLE_BACKEND_CONFIG")) else None,
            **_spec_input_identities(spec),
        },
    }


def _empty_saga(run_id: str, workflow: str) -> dict[str, Any]:
    return {
        "schema_version": _SAGA_SCHEMA,
        "run_id": run_id,
        "workflow": workflow,
        "children": {},
    }


def _same_terminal_outcome_mapping(
    left: object,
    right: object,
) -> bool:
    """Compare current and legacy typed outcome payloads semantically.

    ``implementation_refs`` was added compatibly to the v1 outcome schema and
    defaults to an empty tuple when absent.  Reconciliation must not rewrite a
    terminal saga merely because parsing an older payload materializes that
    empty default.  Malformed payloads remain unequal and fail through the
    ordinary immutable-terminal guard.
    """

    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return left == right
    try:
        return WorkflowOutcome.from_mapping(left) == WorkflowOutcome.from_mapping(right)
    except (TypeError, ValueError):
        return left == right


def _update_parent_child(operation: ChildOperation, **updates: Any) -> None:
    saga_path = operation.parent_run_dir / "saga.json"
    if operation.parent_run_dir.name != operation.parent_run_id:
        raise ValueError("parent run directory does not match parent_run_id")
    if not saga_path.is_file():
        raise ValueError("parent has no durable workflow saga")
    with _locked_json(
        saga_path,
        default={},
    ) as saga:
        if saga.get("schema_version") != _SAGA_SCHEMA:
            raise ValueError("unsupported workflow saga schema")
        if saga.get("run_id") != operation.parent_run_id:
            raise ValueError("workflow saga run_id does not match parent")
        children = saga.setdefault("children", {})
        if not isinstance(children, dict):
            raise ValueError("workflow saga children must be a mapping")
        child_exists = operation.operation_id in children
        child = children.setdefault(operation.operation_id, {})
        if not isinstance(child, dict):
            raise ValueError("workflow saga child entry must be a mapping")
        requested_state = updates.get("state")
        if not child_exists and requested_state != "intent_recorded":
            raise ValueError("child operation has no recorded intent")
        current_state = child.get("state")
        allowed_transitions = {
            None: {"intent_recorded"},
            "intent_recorded": {"running", "terminal"},
            "running": {"running", "terminal"},
            "terminal": {"terminal"},
        }
        if requested_state not in allowed_transitions.get(current_state, set()):
            raise ValueError(
                f"invalid child operation transition: "
                f"{current_state!r} -> {requested_state!r}"
            )
        existing_workflow = child.get("workflow")
        if existing_workflow not in {None, operation.workflow}:
            raise ValueError("child operation workflow identity changed")
        if current_state == "terminal":
            if child.get("run_dir") != updates.get("run_dir", child.get("run_dir")):
                raise ValueError("terminal child run_dir is immutable")
            if not _same_terminal_outcome_mapping(
                child.get("outcome"),
                updates.get("outcome", child.get("outcome")),
            ):
                raise ValueError("terminal child outcome is immutable")
        child.update(updates)


@contextlib.contextmanager
def child_operation(
    parent_run_dir: Path | str,
    *,
    operation_id: str,
    workflow: str,
) -> Iterator[None]:
    """Register and scope exactly one synchronous child dispatch."""

    parent = Path(parent_run_dir).resolve()
    if not operation_id or "/" in operation_id or "\\" in operation_id:
        raise ValueError("operation_id must be a non-empty path-free identifier")
    operation = ChildOperation(
        parent_run_dir=parent,
        parent_run_id=parent.name,
        operation_id=operation_id,
        workflow=workflow,
    )
    _update_parent_child(
        operation,
        workflow=workflow,
        state="intent_recorded",
        intent_recorded_at=_utc_now(),
        run_dir=None,
        outcome=None,
    )
    token = _CHILD_OPERATION.set(operation)
    try:
        yield
    finally:
        _CHILD_OPERATION.reset(token)


def begin_run(
    run_dir: Path | str,
    *,
    workflow: str,
    spec: Any | None = None,
    provenance_root: Path | str | None | object = _PROVENANCE_ROOT_UNSET,
) -> dict[str, Any]:
    """Publish the running manifest and bind it to a parent operation."""

    path = Path(run_dir).resolve()
    operation = _CHILD_OPERATION.get()
    if operation is not None and operation.workflow != workflow:
        raise ValueError(
            f"child operation expected workflow {operation.workflow!r}, got {workflow!r}"
        )
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "workflow": workflow,
        "run_id": path.name,
        "state": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "owner": {
            "pid": os.getpid(),
            "pid_start_token": _pid_start_token(os.getpid()),
        },
        "parent_run_id": operation.parent_run_id if operation else None,
        # Parent and child corpora do not necessarily share a ``runs``
        # directory (regression remediation is the canonical example), so a
        # parent id alone cannot safely reconstruct the saga location.
        "parent_run_dir": str(operation.parent_run_dir) if operation else None,
        "operation_id": operation.operation_id if operation else None,
        "provenance": _provenance(
            spec,
            provenance_root=provenance_root,
            excluded_roots=(path,),
        ),
    }
    _atomic_json(path / "run_manifest.json", manifest)
    _atomic_json(path / "saga.json", _empty_saga(path.name, workflow))
    if operation is not None:
        _update_parent_child(
            operation,
            state="running",
            run_dir=str(path),
            started_at=manifest["started_at"],
        )
    return manifest


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_valid_outcome(path: Path, manifest: Mapping[str, Any]) -> WorkflowOutcome | None:
    raw = _read_json(path / "outcome.json")
    if raw is None:
        return None
    try:
        outcome = WorkflowOutcome.from_mapping(raw)
    except (TypeError, ValueError):
        return None
    if (
        outcome.run_id != manifest.get("run_id")
        or outcome.workflow != manifest.get("workflow")
        or outcome.parent_run_id != manifest.get("parent_run_id")
        or outcome.operation_id != manifest.get("operation_id")
    ):
        return None
    return outcome


def _terminalize_manifest(
    path: Path,
    manifest: dict[str, Any],
    outcome: WorkflowOutcome,
) -> None:
    terminal_state = (
        "interrupted"
        if outcome.terminal_class is TerminalClass.INTERRUPTED
        else "terminal"
    )
    terminal_fields = {
        "state": terminal_state,
        "completed_at": outcome.completed_at,
        "terminal_class": outcome.terminal_class.value,
        "detail_status": outcome.detail_status,
    }
    # Reconciliation visits every run in a Corpus before allocating the next
    # one.  A fully published terminal run is immutable history, not recovery
    # work: avoid replacing its manifest with byte-identical JSON on every
    # later invocation.  Apart from preserving artifact identity, this keeps
    # reruns from making an old child look newly modified.
    if all(manifest.get(key) == value for key, value in terminal_fields.items()):
        return
    manifest.update(terminal_fields)
    _atomic_json(path / "run_manifest.json", manifest)


def _publish_parent_outcome(
    path: Path,
    manifest: Mapping[str, Any],
    outcome: WorkflowOutcome,
) -> None:
    if not outcome.parent_run_id and not outcome.operation_id:
        return
    if not outcome.parent_run_id or not outcome.operation_id:
        raise ValueError("child outcome has an incomplete parent identity")
    parent_run_dir = manifest.get("parent_run_dir")
    if not isinstance(parent_run_dir, str) or not parent_run_dir:
        raise ValueError("child run manifest is missing parent_run_dir")
    operation = ChildOperation(
        parent_run_dir=Path(parent_run_dir),
        parent_run_id=outcome.parent_run_id,
        operation_id=outcome.operation_id,
        workflow=outcome.workflow,
    )
    # ``Corpus.spec_dir`` may be relative even though ``begin_run`` binds a
    # child using an absolute path.  Reconciliation must use that same
    # canonical representation or a later ``--rerun`` will appear to be
    # changing the prior terminal child's run_dir.
    canonical_path = path.resolve()
    canonical_run_dir = str(canonical_path)
    outcome_payload = outcome.to_dict()

    # Parent sagas are a derived view of the immutable child outcome.  When
    # that view is already complete, leave the old terminal saga untouched.
    # A non-terminal or incomplete entry still flows through the locked state
    # transition below so crash recovery remains durable and race-safe.
    saga = _read_json(operation.parent_run_dir / "saga.json")
    if (
        saga is not None
        and saga.get("schema_version") == _SAGA_SCHEMA
        and saga.get("run_id") == operation.parent_run_id
    ):
        children = saga.get("children")
        child = (
            children.get(operation.operation_id)
            if isinstance(children, dict)
            else None
        )
        if (
            isinstance(child, dict)
            and child.get("workflow") == operation.workflow
            and child.get("state") == "terminal"
            and child.get("run_dir") == canonical_run_dir
            and child.get("completed_at") == outcome.completed_at
            and _same_terminal_outcome_mapping(
                child.get("outcome"), outcome_payload,
            )
        ):
            return
    _update_parent_child(
        operation,
        state="terminal",
        run_dir=canonical_run_dir,
        completed_at=outcome.completed_at,
        outcome=outcome_payload,
    )


def _same_outcome_payload(
    left: WorkflowOutcome,
    right: WorkflowOutcome,
) -> bool:
    """Compare immutable outcome content while ignoring commit time."""

    left_payload = left.to_dict()
    right_payload = right.to_dict()
    left_payload.pop("completed_at")
    right_payload.pop("completed_at")
    return left_payload == right_payload


def record_provenance_inputs(
    run_dir: Path | str,
    *,
    backend_routing: str,
    backend_routing_source_sha256: str | None,
    backend_features: str | None = None,
) -> None:
    """Attach the late-resolved routing identity to a running manifest.

    Backend routing is discovered only after the run folder opens, so the
    initial manifest cannot know which CLI/spec/workspace config won.  This
    locked update records the effective route identity and, separately, the
    exact selected file digest without persisting secret expansions or making
    mutable summaries authoritative. The closed argument surface prevents
    callers from accidentally recording expanded credentials.
    """

    if backend_routing != "builtin:crucible.backend-routing.v1":
        if (
            not backend_routing.startswith("sha256:")
            or len(backend_routing) != 71
            or any(
                char not in "0123456789abcdef"
                for char in backend_routing[7:]
            )
        ):
            raise ValueError("backend_routing must be a SHA-256 identity or builtin ID")
    if backend_features is not None and (
        not backend_features.startswith("sha256:")
        or len(backend_features) != 71
        or any(
            char not in "0123456789abcdef"
            for char in backend_features[7:]
        )
    ):
        raise ValueError("backend_features must be a SHA-256 identity")
    if backend_routing_source_sha256 is not None and (
        not backend_routing_source_sha256.startswith("sha256:")
        or len(backend_routing_source_sha256) != 71
        or any(
            char not in "0123456789abcdef"
            for char in backend_routing_source_sha256[7:]
        )
    ):
        raise ValueError(
            "backend_routing_source_sha256 must be a SHA-256 identity"
        )
    manifest_path = Path(run_dir).resolve() / "run_manifest.json"
    with _locked_json(manifest_path, default={}) as manifest:
        if manifest.get("schema_version") != _MANIFEST_SCHEMA:
            raise ValueError("run has no valid provenance manifest")
        if manifest.get("state") != "running":
            raise ValueError("cannot change provenance after run terminal")
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("run provenance must be a mapping")
        retained = provenance.get("inputs")
        if not isinstance(retained, dict):
            raise ValueError("run provenance inputs must be a mapping")
        updates: dict[str, str | None] = {
            "backend_routing": backend_routing,
            # ``None`` is authoritative for the built-in/no-file route. Keep
            # it explicit so a later caller cannot retrofit different source
            # bytes into an already-open run.
            "backend_routing_source_sha256": backend_routing_source_sha256,
        }
        if backend_features is not None:
            updates["backend_features"] = backend_features
        for key, value in updates.items():
            if key in retained and retained[key] != value:
                raise ValueError(
                    f"cannot change immutable provenance input {key!r}"
                )
            retained[key] = value


def record_execution_policy_input(
    run_dir: Path | str,
    *,
    execution_policy: str,
) -> None:
    """Bind a late-resolved workflow execution policy into run provenance.

    Execution policy is intentionally separate from semantic Spec authority.
    The value is a secret-free canonical SHA-256 identity; callers cannot
    persist expanded prompts, credentials, or mutable scheduler state here.
    Repeated equal writes are idempotent and conflicting writes fail closed.
    """

    if (
        not execution_policy.startswith("sha256:")
        or len(execution_policy) != 71
        or any(
            char not in "0123456789abcdef"
            for char in execution_policy[7:]
        )
    ):
        raise ValueError("execution_policy must be a SHA-256 identity")
    manifest_path = Path(run_dir).resolve() / "run_manifest.json"
    with _locked_json(manifest_path, default={}) as manifest:
        if manifest.get("schema_version") != _MANIFEST_SCHEMA:
            raise ValueError("run has no valid provenance manifest")
        if manifest.get("state") != "running":
            raise ValueError("cannot change provenance after run terminal")
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("run provenance must be a mapping")
        retained = provenance.get("inputs")
        if not isinstance(retained, dict):
            raise ValueError("run provenance inputs must be a mapping")
        existing = retained.get("execution_policy")
        if existing is not None and existing != execution_policy:
            raise ValueError(
                "cannot change immutable provenance input 'execution_policy'"
            )
        retained["execution_policy"] = execution_policy


def finalize_run(
    run_dir: Path | str,
    *,
    spec_version: int | None = None,
    trust_summary: bool = True,
) -> WorkflowOutcome:
    """Project immutable result/summary files into a typed outcome."""

    path = Path(run_dir).resolve()
    manifest = _read_json(path / "run_manifest.json")
    if manifest is None or manifest.get("schema_version") != _MANIFEST_SCHEMA:
        raise ValueError(f"run {path} has no valid run manifest")
    result = _read_json(path / "result.json") or {}
    if not isinstance(trust_summary, bool):
        raise TypeError("trust_summary must be a boolean")
    summary = (_read_json(path / "summary.json") or {}) if trust_summary else {}
    detail_status = str(
        result.get("status") or summary.get("status") or "contract_error"
    )
    error_kind = result.get("error_kind") or summary.get("error_kind")
    error_message = result.get("error_message") or summary.get("error_message")
    try:
        cost = float(result.get("cost_usd") or 0.0)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("cost is not finite and non-negative")
    except (TypeError, ValueError):
        cost = 0.0
        error_kind = error_kind or "invalid_cost"
        error_message = error_message or (
            "terminal cost accounting was invalid; success is not authoritative"
        )
        detail_status = "cost_incomplete"
    refs: list[str] = []
    for key, value in summary.items():
        if key.endswith("_ref") and isinstance(value, str):
            if _is_content_addressed_ref(value):
                refs.append(value)
        if key.endswith("_refs"):
            if isinstance(value, str) and _is_content_addressed_ref(value):
                refs.append(value)
            elif isinstance(value, list):
                refs.extend(
                    item for item in value
                    if _is_content_addressed_ref(item)
                )
    raw_implementation_refs = summary.get("implementation_refs", [])
    implementation_refs: tuple[str, ...]
    if (
        not isinstance(raw_implementation_refs, list)
        or not all(
            _is_content_addressed_ref(item)
            for item in raw_implementation_refs
        )
        or len(set(raw_implementation_refs)) != len(raw_implementation_refs)
    ):
        implementation_refs = ()
        error_kind = error_kind or "invalid_implementation_refs"
        error_message = error_message or (
            "terminal implementation_refs must be a unique canonical "
            "content-addressed string list"
        )
        detail_status = "contract_error"
    else:
        implementation_refs = tuple(raw_implementation_refs)
    outcome = WorkflowOutcome(
        schema_version=_OUTCOME_SCHEMA,
        workflow=str(manifest["workflow"]),
        run_id=str(manifest["run_id"]),
        terminal_class=classify_terminal_status(
            detail_status,
            error_kind=str(error_kind) if error_kind else None,
        ),
        detail_status=detail_status,
        cost_usd=cost,
        spec_version=spec_version,
        refs=tuple(dict.fromkeys(refs)),
        error_kind=str(error_kind) if error_kind else None,
        error_message=str(error_message)[:2000] if error_message else None,
        parent_run_id=manifest.get("parent_run_id"),
        operation_id=manifest.get("operation_id"),
        completed_at=_utc_now(),
        implementation_refs=implementation_refs,
    )
    existing = _read_valid_outcome(path, manifest)
    if (path / "outcome.json").exists():
        if existing is None:
            raise ValueError("existing workflow outcome is corrupt or misbound")
        if not _same_outcome_payload(existing, outcome):
            raise ValueError("terminal workflow outcome is immutable")
        outcome = existing
    else:
        # The outcome is the terminal commit record.  Recovery can roll the
        # manifest and parent saga forward if the process dies after this
        # atomic write but before either derived view is updated.
        _atomic_json(path / "outcome.json", outcome.to_dict())
    _terminalize_manifest(path, manifest, outcome)
    _publish_parent_outcome(path, manifest, outcome)
    return outcome


def _bound_child_terminal(
    parent_run_dir: Path | str,
    operation_id: str,
) -> tuple[WorkflowOutcome, Path]:
    """Return one child terminal only when both durable sides agree.

    A parent's saga is a mutable derived index, not accounting authority.  Its
    copied outcome must exactly match the immutable child ``outcome.json`` and
    the child manifest must bind back to this parent operation.  This prevents
    a schema-valid edit to a saga copy (for example lowering ``cost_usd``)
    from authorizing later paid work.
    """

    parent = Path(parent_run_dir).resolve()
    if not operation_id or "/" in operation_id or "\\" in operation_id:
        raise ValueError("operation_id must be a non-empty path-free identifier")
    saga = _read_json(parent / "saga.json")
    if (
        saga is None
        or saga.get("schema_version") != _SAGA_SCHEMA
        or saga.get("run_id") != parent.name
    ):
        raise ValueError("parent has no valid workflow saga")
    children = saga.get("children")
    child = children.get(operation_id) if isinstance(children, dict) else None
    if not isinstance(child, dict) or child.get("state") != "terminal":
        raise ValueError(f"child operation {operation_id!r} is not terminal")
    workflow = child.get("workflow")
    if not isinstance(workflow, str) or not workflow:
        raise ValueError(f"child operation {operation_id!r} has no workflow")
    saga_outcome = child.get("outcome")
    if not isinstance(saga_outcome, dict):
        raise ValueError(f"child operation {operation_id!r} has no typed outcome")
    # Parse before filesystem traversal so malformed copied terminals always
    # fail as contract errors even if their path happens to be unavailable.
    WorkflowOutcome.from_mapping(saga_outcome)

    raw_path = child.get("run_dir")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"child operation {operation_id!r} has no run_dir")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise ValueError(f"child operation {operation_id!r} run_dir is not absolute")
    path = candidate.resolve()
    if not path.is_dir():
        raise ValueError(f"child operation {operation_id!r} run_dir is unavailable")
    manifest = _read_json(path / "run_manifest.json")
    if manifest is None or manifest.get("schema_version") != _MANIFEST_SCHEMA:
        raise ValueError(f"child operation {operation_id!r} has no valid manifest")
    committed = _read_valid_outcome(path, manifest)
    if committed is None:
        raise ValueError(
            f"child operation {operation_id!r} has no valid committed outcome"
        )
    parent_binding = manifest.get("parent_run_dir")
    if not isinstance(parent_binding, str) or not parent_binding:
        raise ValueError(f"child operation {operation_id!r} has no parent run binding")
    if (
        path.name != committed.run_id
        or committed.workflow != workflow
        or committed.parent_run_id != parent.name
        or committed.operation_id != operation_id
        or Path(parent_binding).resolve() != parent
    ):
        raise ValueError(
            f"child operation {operation_id!r} committed identity is misbound"
        )
    if not _same_terminal_outcome_mapping(saga_outcome, committed.to_dict()):
        raise ValueError(
            f"child operation {operation_id!r} saga outcome differs from its "
            "committed outcome"
        )
    if child.get("completed_at") != committed.completed_at:
        raise ValueError(
            f"child operation {operation_id!r} completion identity differs"
        )
    return committed, path


def child_outcome(parent_run_dir: Path | str, operation_id: str) -> WorkflowOutcome:
    outcome, _ = _bound_child_terminal(parent_run_dir, operation_id)
    return outcome


def child_run_dir(parent_run_dir: Path | str, operation_id: str) -> Path:
    """Return the saga-bound child directory without filesystem guessing."""
    _, path = _bound_child_terminal(parent_run_dir, operation_id)
    return path


def _validate_recoverable_result(path: Path) -> None:
    """Validate an immutable RunResult before stale-run roll-forward.

    ``finalize_run`` intentionally accepts legacy partial documents during
    ordinary publication.  Crash recovery has a different authority boundary:
    it may treat a pre-existing ``result.json`` as terminal only when the full
    create-once RunResult contract is present.  ``summary.json`` is a mutable
    checkpoint and may legitimately be stale when result publication wins a
    crash race, so recovery neither trusts nor rejects based on that document.
    """

    result_path = path / "result.json"
    try:
        result_stat = result_path.lstat()
    except OSError as exc:
        raise ValueError(f"run {path} has no durable terminal result") from exc
    if not stat.S_ISREG(result_stat.st_mode) or result_stat.st_nlink != 1:
        raise ValueError(f"run {path} terminal result is not a private file")
    result = _read_json(result_path)
    expected = {
        "status",
        "duration_s",
        "cost_usd",
        "error_kind",
        "error_message",
    }
    if result is None or set(result) != expected:
        raise ValueError(f"run {path} has no valid immutable terminal result")
    status = result["status"]
    if (
        not isinstance(status, str)
        or not status
        or status != status.strip()
        or len(status) > 128
        or status.lower() in {"pending", "running", "started"}
    ):
        raise ValueError(f"run {path} result status is not terminal")
    for field_name in ("duration_s", "cost_usd"):
        number = result[field_name]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0.0
        ):
            raise ValueError(f"run {path} result {field_name} is invalid")
    for field_name in ("error_kind", "error_message"):
        if result[field_name] is not None and not isinstance(result[field_name], str):
            raise ValueError(f"run {path} result {field_name} is invalid")


def _manifest_spec_version(manifest: Mapping[str, Any]) -> int | None:
    provenance = manifest.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, dict) else None
    value = inputs.get("spec_version") if isinstance(inputs, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _recover_uncommitted_result(
    path: Path,
    manifest: dict[str, Any],
    *,
    error_kind: str = "process_lost_after_result",
    boundary_message: str = (
        "owning process exited after publishing result but before the "
        "cleanup-complete outcome boundary"
    ),
) -> WorkflowOutcome:
    """Conservatively terminalize a dead owner that published only result.

    ``result.json`` can be written inside provider and mutation contexts, so
    it is not proof that cleanup completed.  Retain its numeric cost only as a
    lower bound and never carry success or mutable-summary references across
    that missing boundary.
    """

    result = _read_json(path / "result.json")
    if result is None:  # guarded by _validate_recoverable_result
        raise ValueError(f"run {path} has no recoverable result")
    original_error = ""
    if result["error_kind"] or result["error_message"]:
        original_error = (
            f"; recorded error={result['error_kind']!r}: {result['error_message']!r}"
        )
    outcome = WorkflowOutcome(
        schema_version=_OUTCOME_SCHEMA,
        workflow=str(manifest.get("workflow") or "unknown"),
        run_id=path.name,
        terminal_class=TerminalClass.INCONCLUSIVE,
        detail_status="incomplete_accounting",
        cost_usd=float(result["cost_usd"]),
        spec_version=_manifest_spec_version(manifest),
        refs=(),
        error_kind=error_kind,
        error_message=f"{boundary_message}{original_error}"[:2000],
        parent_run_id=manifest.get("parent_run_id"),
        operation_id=manifest.get("operation_id"),
        completed_at=_utc_now(),
        implementation_refs=(),
    )
    outcome_path = path / "outcome.json"
    if outcome_path.exists():
        raise ValueError(f"run {path} has a corrupt or misbound outcome")
    _atomic_json(outcome_path, outcome.to_dict())
    _terminalize_manifest(path, manifest, outcome)
    _publish_parent_outcome(path, manifest, outcome)
    return outcome


def _finalize_uncommitted_result_as_incomplete(
    run_dir: Path | str,
    *,
    cause: BaseException,
) -> WorkflowOutcome:
    """Publish debt when cleanup fails after a caller pre-finalized result."""

    path = Path(run_dir).resolve()
    manifest = _read_json(path / "run_manifest.json")
    if manifest is None or manifest.get("schema_version") != _MANIFEST_SCHEMA:
        raise ValueError(f"run {path} has no valid run manifest")
    if manifest.get("state") != "running":
        raise ValueError(f"run {path} is not awaiting terminal projection")
    _validate_recoverable_result(path)
    return _recover_uncommitted_result(
        path,
        manifest,
        error_kind=type(cause).__name__,
        boundary_message=(
            "workflow cleanup raised after result publication and before the "
            f"cleanup-complete outcome boundary: {str(cause)[:1000]}"
        ),
    )


def reconcile_stale_runs(corpus: Any) -> tuple[str, ...]:
    """Roll forward committed outcomes and interrupt exact dead owners.

    ``outcome.json`` is the terminal commit record.  If a process dies after
    publishing it but before updating the manifest or parent saga, recovery
    finishes those derived writes instead of replacing success with an
    interruption.  A run without a committed outcome is interrupted only when
    its exact PID/start-token identity is no longer live.
    """

    runs = Path(corpus.spec_dir) / "runs"
    if not runs.is_dir():
        return ()
    recovered: list[str] = []
    for candidate in runs.iterdir():
        path = candidate.resolve()
        manifest = _read_json(path / "run_manifest.json") if path.is_dir() else None
        if manifest is None or manifest.get("schema_version") != _MANIFEST_SCHEMA:
            continue
        committed = _read_valid_outcome(path, manifest)
        if committed is not None:
            changed = manifest.get("state") not in {"terminal", "interrupted"}
            _terminalize_manifest(path, manifest, committed)
            _publish_parent_outcome(path, manifest, committed)
            if changed:
                recovered.append(path.name)
            continue
        if manifest.get("state") != "running":
            continue
        owner = manifest.get("owner")
        if not isinstance(owner, dict):
            continue
        try:
            pid = int(str(owner.get("pid", "")))
        except (TypeError, ValueError):
            pid = -1
        token = owner.get("pid_start_token")
        if _process_is_owner(pid, str(token) if token is not None else None):
            continue
        result_path = path / "result.json"
        if os.path.lexists(result_path):
            _validate_recoverable_result(path)
            _recover_uncommitted_result(path, manifest)
            recovered.append(path.name)
            continue
        outcome = WorkflowOutcome(
            schema_version=_OUTCOME_SCHEMA,
            workflow=str(manifest.get("workflow") or "unknown"),
            run_id=path.name,
            terminal_class=TerminalClass.INTERRUPTED,
            detail_status="process_lost",
            cost_usd=0.0,
            spec_version=None,
            refs=(),
            error_kind="process_lost",
            error_message="owning process exited without publishing a terminal result",
            parent_run_id=manifest.get("parent_run_id"),
            operation_id=manifest.get("operation_id"),
            completed_at=_utc_now(),
            implementation_refs=(),
        )
        outcome_path = path / "outcome.json"
        if outcome_path.exists():
            raise ValueError(f"run {path} has a corrupt or misbound outcome")
        _atomic_json(outcome_path, outcome.to_dict())
        _terminalize_manifest(path, manifest, outcome)
        _publish_parent_outcome(path, manifest, outcome)
        recovered.append(path.name)
    return tuple(recovered)


__all__ = [
    "TOOL_MUTATION_INDETERMINATE_CLASSIFICATION",
    "TOOL_MUTATION_INDETERMINATE_STOP_REASON",
    "TerminalClass",
    "UnknownToolMutationUsage",
    "WorkflowOutcome",
    "begin_run",
    "child_operation",
    "child_outcome",
    "child_run_dir",
    "classify_terminal_status",
    "checkpoint_settled_tool_mutation_terminal",
    "checkpoint_unsettled_tool_mutation_terminal",
    "finalize_run",
    "is_tool_mutation_indeterminate_result",
    "preserve_tool_mutation_indeterminate",
    "record_execution_policy_input",
    "record_provenance_inputs",
    "reconcile_stale_runs",
    "synthesize_tool_mutation_indeterminate_implementation",
    "unknown_tool_mutation_usage",
    "workspace_state_digest",
]
