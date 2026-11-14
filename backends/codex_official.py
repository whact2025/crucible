# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Optional Codex backend transport over the official OpenAI Python SDK."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from assay import (
    BackendExecutionEvent,
    BackendPolicy,
    BudgetExhausted,
    ExternalCancellation,
    Invocation,
    InvocationResult,
    Message,
    StaleConversation,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)

from .codex import (
    _CODEX_RESUME_NUDGE,
    _COST_UPPER_BOUND_EVENT_NAME,
    _MCP_URL_UNSET,
    CodexBackend,
    _append_event,
    _codex_usage_cost_upper_bound,
    _conversation_outcome,
    _resolve_codex_auth_path,
)
from .codex_official_worker import (
    CodexOfficialSDKQualificationError,
    CodexOfficialSDKUnavailable,
    CodexOfficialWorkerInvocationError,
    resolve_worker_python,
    run_official_worker,
    worker_environment,
)

_PROCESS_STOP_SECONDS = 2.0
_PROCESS_POLL_SECONDS = 0.01


def _official_process_group_exists(process_group_id: int) -> bool:
    """Return whether an isolated official worker group still exists."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        raise CodexOfficialSDKQualificationError(
            "could not inspect a retained official Codex worker process group "
            f"({type(exc).__name__})"
        ) from exc
    return True


def _official_process_exists(process_id: int) -> bool:
    """Return whether a registered process-group leader still exists."""

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        raise CodexOfficialSDKQualificationError(
            "could not inspect a retained official Codex worker leader "
            f"({type(exc).__name__})"
        ) from exc
    return True


class OfficialCodexBackend(CodexBackend):
    """Rollback-safe official-SDK pilot with legacy adapter parity.

    The inherited logical-invocation controller owns retry, deadline, budget,
    mutation-host, MCP, evidence, and debug behavior. This subclass replaces
    only one physical app-server attempt with an exact-locked
    ``openai-codex`` worker process.
    """

    transport = "official_sdk"

    def __init__(
        self,
        *,
        worker_python: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if self._native_shell:
            raise ValueError(
                "official_sdk transport is not qualified for native_shell; "
                "use native_shell=False with explicit Crucible tools or "
                "select legacy_app_server"
            )
        self._official_worker_python = resolve_worker_python(worker_python)
        self._official_process_condition = threading.Condition()
        self._official_process_ids: set[int] = set()
        self._official_closed = threading.Event()

    def _register_official_process(self, process_id: int) -> None:
        with self._official_process_condition:
            if self._official_closed.is_set():
                with contextlib.suppress(OSError, ProcessLookupError):
                    os.killpg(process_id, signal.SIGKILL)
                raise asyncio.CancelledError(
                    "OfficialCodexBackend closed before worker registration"
                )
            self._official_process_ids.add(process_id)
            self._official_process_condition.notify_all()

    def _unregister_official_process(self, process_id: int) -> None:
        with self._official_process_condition:
            self._official_process_ids.discard(process_id)
            self._official_process_condition.notify_all()

    def _signal_official_process_groups(
        self,
        process_ids: tuple[int, ...],
        sig: signal.Signals,
    ) -> None:
        for process_id in process_ids:
            try:
                os.killpg(process_id, sig)
            except ProcessLookupError:
                self._unregister_official_process(process_id)
            except OSError:
                # Settlement is proven below rather than inferred from signal
                # delivery. Retain the ID so close can fail closed or retry.
                continue

    def _prune_settled_official_process_groups(self) -> tuple[int, ...]:
        with self._official_process_condition:
            process_ids = tuple(self._official_process_ids)
        for process_id in process_ids:
            if not _official_process_group_exists(process_id):
                self._unregister_official_process(process_id)
        with self._official_process_condition:
            return tuple(self._official_process_ids)

    def _wait_for_official_process_groups(
        self,
        *,
        timeout: float,
        return_on_missing_leader: bool = False,
    ) -> tuple[int, ...]:
        deadline = time.monotonic() + timeout
        while True:
            process_ids = self._prune_settled_official_process_groups()
            if not process_ids:
                return ()
            if return_on_missing_leader and any(
                not _official_process_exists(process_id)
                for process_id in process_ids
            ):
                # A residual group whose leader exited no longer has a worker
                # capable of coordinating safe descendant shutdown.
                return process_ids
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return process_ids
            with self._official_process_condition:
                self._official_process_condition.wait(
                    timeout=min(_PROCESS_POLL_SECONDS, remaining)
                )

    def close(self) -> None:
        """Cancel/reap official worker groups, then release inherited state."""

        self._official_closed.set()
        with self._official_process_condition:
            process_ids = tuple(self._official_process_ids)
        settlement_error: BaseException | None = None
        try:
            self._signal_official_process_groups(
                process_ids,
                signal.SIGTERM,
            )
            process_ids = self._wait_for_official_process_groups(
                timeout=_PROCESS_STOP_SECONDS,
                return_on_missing_leader=True,
            )
            if process_ids:
                self._signal_official_process_groups(
                    process_ids,
                    signal.SIGKILL,
                )
                process_ids = self._wait_for_official_process_groups(
                    timeout=_PROCESS_STOP_SECONDS
                )
            if process_ids:
                raise CodexOfficialSDKQualificationError(
                    "official Codex worker lifecycle did not settle after "
                    f"SIGKILL ({len(process_ids)} retained process group(s))"
                )
        except BaseException as exc:
            settlement_error = exc

        try:
            super().close()
        except BaseException as exc:
            if settlement_error is not None:
                raise settlement_error from exc
            raise
        if settlement_error is not None:
            raise settlement_error

    def _attached_tool_names(self) -> frozenset[str]:
        with self._mcp_lock:
            toolset = self._mcp_toolset
        if toolset is None:
            return frozenset()
        try:
            descriptors = toolset.to_anthropic_tools()
        except (AttributeError, TypeError) as exc:
            raise CodexOfficialSDKQualificationError(
                "attached ToolSet cannot produce canonical descriptors"
            ) from exc
        names: set[str] = set()
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                raise CodexOfficialSDKQualificationError(
                    "attached ToolSet returned a malformed descriptor"
                )
            name = descriptor.get("name")
            if not isinstance(name, str) or not name:
                raise CodexOfficialSDKQualificationError(
                    "attached ToolSet returned a descriptor without a name"
                )
            names.add(name)
        return frozenset(names)

    async def _invoke_once(
        self,
        invocation: Invocation,
        policy: BackendPolicy,
        *,
        resume_thread_id: str | None = None,
        thread_id_holder: list[str] | None = None,
        resume_failed_holder: list[bool] | None = None,
        external_resume: bool = False,
        recovery: bool = False,
        budget_usd_override: float | None = None,
        mcp_url_override: Any = _MCP_URL_UNSET,
        interrupted_evidence: list[InvocationResult] | None = None,
    ) -> InvocationResult:
        """Run one qualified physical attempt through the isolated worker.

        The worker protocol publishes usage only in its terminal payload, so
        cancellation cannot yield a qualified partial result for the inherited
        controller's diagnostic sink. Accepting the sink still keeps this
        transport substitutable for :class:`CodexBackend`; it deliberately
        remains empty until a complete worker payload has been received.
        """

        auth_path = _resolve_codex_auth_path()
        if not auth_path.is_file():
            raise RuntimeError(
                f"Codex auth file not found at {auth_path}. Run `codex login` "
                "(or set $CODEX_HOME) before invoking the codex backend."
            )
        if self._official_closed.is_set():
            raise asyncio.CancelledError("OfficialCodexBackend is closed")
        mcp_url = (
            self._ensure_mcp_server_running()
            if mcp_url_override is _MCP_URL_UNSET
            else mcp_url_override
        )
        subprocess_env = self._confined_subprocess_env(auth_path)
        if subprocess_env is None:
            raise CodexOfficialSDKQualificationError(
                "official_sdk transport requires a confined private "
                "CODEX_HOME"
            )
        codex_home = subprocess_env.get("CODEX_HOME")
        if not codex_home:
            raise CodexOfficialSDKQualificationError(
                "confined Codex environment omitted CODEX_HOME"
            )
        output_schema = (
            invocation.response_schema
            if invocation.response_schema is not None
            else self._output_schema
        )
        turn_input = (
            _CODEX_RESUME_NUDGE.format(task=invocation.user)
            if recovery
            else invocation.user
        )
        request = {
            "protocol": "crucible.codex-worker.v1",
            "cwd": self._cwd,
            "model": self.model,
            "effort": self._effort,
            "system": invocation.system,
            "user": turn_input,
            "output_schema": (
                None if output_schema is None else deepcopy(output_schema)
            ),
            "mcp_url": mcp_url,
            "native_shell": False,
            "trusted_project_instructions": (
                self._trusted_project_instructions
            ),
            "resume_thread_id": resume_thread_id,
        }
        events: list[BackendExecutionEvent] = []
        _append_event(events, "request", name="codex_official_sdk")
        holder_size = 0 if thread_id_holder is None else len(thread_id_holder)

        def attempt_conversation():
            captured_id = (
                thread_id_holder[-1]
                if thread_id_holder is not None
                and len(thread_id_holder) > holder_size
                else resume_thread_id
            )
            return _conversation_outcome(
                invocation,
                opaque_id=captured_id,
            )

        try:
            payload = await run_official_worker(
                request,
                worker_python=self._official_worker_python,
                env=worker_environment(
                    codex_home=codex_home,
                    inherited=subprocess_env,
                ),
                # The inherited controller owns the one absolute deadline.
                # Cancellation of this await kills/reaps the process group.
                timeout_seconds=None,
                thread_id_holder=thread_id_holder,
                on_process_start=self._register_official_process,
                on_process_stop=self._unregister_official_process,
            )
        except asyncio.CancelledError:
            raise
        except (
            CodexOfficialSDKQualificationError,
            CodexOfficialSDKUnavailable,
        ) as exc:
            if (
                resume_thread_id is not None
                and resume_failed_holder is not None
                and (
                    thread_id_holder is None
                    or len(thread_id_holder) == holder_size
                )
            ):
                resume_failed_holder.append(True)
            error: Any
            if self._official_closed.is_set():
                error = ExternalCancellation(
                    cause="backend_close",
                    message="official Codex worker cancelled by backend close",
                    retriable=True,
                )
                _append_event(events, "cancellation", name="backend_close")
                stop_reason = "cancelled"
            else:
                error = TransportError(
                    http_status=None,
                    message=f"{type(exc).__name__}: {exc}",
                    retriable=False,
                )
                stop_reason = "error"
            return self._result_with_evidence(
                final_message="",
                transcript=(),
                usage=Usage(),
                stop_reason=stop_reason,
                error=error,
                token_usage_complete=False,
                cost_complete=False,
                terminal_complete=False,
                events=events,
                tool_authority_digest=self._attached_tool_authority_digest(),
                trusted_context_digest=self._trusted_context_digest(),
                conversation=attempt_conversation(),
            )
        except CodexOfficialWorkerInvocationError as exc:
            if (
                resume_thread_id is not None
                and resume_failed_holder is not None
                and (
                    thread_id_holder is None
                    or len(thread_id_holder) == holder_size
                )
            ):
                resume_failed_holder.append(True)
            if external_resume and exc.error_type in {
                "CodexRpcError",
                "InvalidParamsError",
                "JsonRpcError",
            } and (
                thread_id_holder is None
                or len(thread_id_holder) == holder_size
            ):
                return self._result_with_evidence(
                    final_message="",
                    transcript=(),
                    usage=Usage(),
                    stop_reason="stale_conversation",
                    error=StaleConversation(
                        message=(
                            "official Codex provider rejected the requested "
                            "conversation handle"
                        )
                    ),
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=False,
                    events=events,
                    tool_authority_digest=(
                        self._attached_tool_authority_digest()
                    ),
                    trusted_context_digest=self._trusted_context_digest(),
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=None,
                        state="stale",
                    ),
                )
            error = self._classify_exception(exc, policy)
            if error is None:
                raise
            return self._result_with_evidence(
                final_message="",
                transcript=(),
                usage=Usage(),
                stop_reason="error",
                error=error,
                token_usage_complete=False,
                cost_complete=False,
                terminal_complete=False,
                events=events,
                tool_authority_digest=self._attached_tool_authority_digest(),
                trusted_context_digest=self._trusted_context_digest(),
                conversation=attempt_conversation(),
            )
        except Exception as exc:  # provider/transport failure is retryable
            if (
                resume_thread_id is not None
                and resume_failed_holder is not None
                and (
                    thread_id_holder is None
                    or len(thread_id_holder) == holder_size
                )
            ):
                resume_failed_holder.append(True)
            error = self._classify_exception(exc, policy)
            if error is None:
                raise
            return self._result_with_evidence(
                final_message="",
                transcript=(),
                usage=Usage(),
                stop_reason="error",
                error=error,
                token_usage_complete=False,
                cost_complete=False,
                terminal_complete=False,
                events=events,
                tool_authority_digest=self._attached_tool_authority_digest(),
                trusted_context_digest=self._trusted_context_digest(),
                conversation=attempt_conversation(),
            )

        allowed_names = self._attached_tool_names()
        for item in payload["tool_items"]:
            if mcp_url is None or item["tool"] not in allowed_names:
                return self._result_with_evidence(
                    final_message="",
                    transcript=(),
                    usage=Usage(),
                    stop_reason="error",
                    error=TransportError(
                        http_status=None,
                        message=(
                            "official Codex worker reported a tool outside "
                            "the attached authority"
                        ),
                        retriable=False,
                    ),
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=True,
                    events=events,
                    provider_request_id=payload["turn_id"],
                    tool_authority_digest=(
                        self._attached_tool_authority_digest()
                    ),
                    trusted_context_digest=self._trusted_context_digest(),
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=payload["thread_id"],
                    ),
                )
            _append_event(
                events,
                "tool_call",
                name=item["tool"],
                parent_id="codex_official_sdk",
                child_id=item["call_id"],
            )
            _append_event(
                events,
                "tool_result",
                name=item["tool"],
                parent_id="codex_official_sdk",
                child_id=item["call_id"],
            )

        usage_payload = payload["usage"]
        usage = Usage()
        usage_complete = False
        cost_complete = False
        pricing_evidence = UsagePricingEvidence(segments=(), complete=False)
        if usage_payload is not None:
            # The exact-locked public SDK ABI exposes cache reads but not
            # OpenAI's GPT-5.6 cache-write counter. Preserve observed cache
            # reads while pricing a budget-safe upper bound. The token
            # partition is complete only when all input was read from cache,
            # which arithmetically forces cache writes to zero.
            usage_complete = (
                usage_payload["input_tokens"]
                == usage_payload["cached_input_tokens"]
            )
            cost_usd, cost_complete = _codex_usage_cost_upper_bound(
                model=self.model,
                input_tokens=usage_payload["input_tokens"],
                output_tokens=usage_payload["output_tokens"],
                cache_read_tokens=usage_payload["cached_input_tokens"],
            )

            usage = Usage(
                input_tokens=usage_payload["input_tokens"],
                output_tokens=usage_payload["output_tokens"],
                cost_usd=cost_usd,
                cache_read_tokens=usage_payload["cached_input_tokens"],
            )
            if usage_complete and cost_complete:
                pricing_evidence = UsagePricingEvidence(
                    segments=(
                        UsagePricingSegment(
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            cache_creation_tokens=0,
                            cache_read_tokens=usage.cache_read_tokens,
                            cost_usd=usage.cost_usd,
                        ),
                    ),
                    complete=True,
                )
            _append_event(
                events,
                "usage",
                name=(
                    "complete"
                    if usage_complete and cost_complete
                    else _COST_UPPER_BOUND_EVENT_NAME
                ),
            )

        final_message = payload["final_response"]
        transcript = (
            (Message(role="assistant", content=final_message),)
            if final_message
            else ()
        )
        status = payload["status"]
        error_payload = payload["error"]
        error = None
        stop_reason = "end_turn"
        if status == "interrupted":
            error = ExternalCancellation(
                cause="provider_interrupt",
                message=(
                    "official Codex turn was interrupted"
                    if error_payload is None
                    else error_payload["message"]
                ),
                retriable=True,
            )
            _append_event(events, "cancellation", name="provider_interrupt")
            stop_reason = "cancelled"
        elif status == "failed":
            assert error_payload is not None
            error = TransportError(
                http_status=None,
                message=error_payload["message"],
                retriable=False,
            )
            stop_reason = "error"

        budget_limit = (
            invocation.budget_usd
            if budget_usd_override is None
            else budget_usd_override
        )
        if (
            error is None
            and budget_limit is not None
            and usage.cost_usd > budget_limit
        ):
            _append_event(events, "limit", name="budget_usd")
            error = BudgetExhausted(
                spent_usd=usage.cost_usd,
                limit_usd=budget_limit,
                message=(
                    f"official Codex turn spent ${usage.cost_usd:.4f}, "
                    f"exceeding remaining budget ${budget_limit:.4f}"
                ),
            )
            stop_reason = "budget_exhausted"
        elif (
            error is None
            and not final_message
            and usage.output_tokens == 0
            and not invocation.tools
            and not allowed_names
        ):
            error = TransportError(
                http_status=None,
                message=(
                    "official Codex SDK returned a completed but "
                    "content-free turn"
                ),
                retriable=True,
            )
            stop_reason = "error"

        return self._result_with_evidence(
            final_message=final_message,
            transcript=transcript,
            usage=usage,
            pricing_evidence=pricing_evidence,
            stop_reason=stop_reason,
            error=error,
            token_usage_complete=usage_complete,
            cost_complete=cost_complete,
            terminal_complete=True,
            events=events,
            provider_request_id=payload["turn_id"],
            tool_authority_digest=self._attached_tool_authority_digest(),
            trusted_context_digest=self._trusted_context_digest(),
            conversation=_conversation_outcome(
                invocation,
                opaque_id=payload["thread_id"],
            ),
        )


__all__ = ["OfficialCodexBackend"]
