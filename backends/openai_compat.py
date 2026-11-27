# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""OpenAIBackend — assay.Backend over an OpenAI-compatible HTTP endpoint.

Wraps any server that speaks the OpenAI ``/v1/chat/completions`` wire
protocol: local inference servers (vLLM, llama.cpp, Ollama, TGI,
lm-studio) AND hosted gateways like the **NVIDIA Inference Hub**
(`https://inference-api.nvidia.com`). The wire protocol is the same;
only ``base_url`` + ``api_key`` differ.

Hosted inference routes use this adapter's OpenAI tool-calling loop. They do
not pass through ``ClaudeBackend`` even when a gateway also exposes an
Anthropic-compatible endpoint, because Crucible requires structured OpenAI
``tool_calls`` and continuation messages on this route.

Capabilities:

- ``invoke()`` — single-shot completion, OR a multi-turn tool-use loop
  when a ToolSet has been attached via ``attach_toolset(...)``. The
  loop converts the crucible ToolSet to OpenAI function-calling tools,
  dispatches ``tool_calls``, feeds results back, and iterates until the
  model stops calling tools (bounded by ``max_turns`` + ``budget_usd``).
- ``attach_toolset()`` — stores the ToolSet so ``invoke()`` can expose
  it as OpenAI tools. (assay.plan / assay.implement attach out-of-band
  then call ``invoke()``; the tool loop lives inside ``invoke()``.)

URL + key plumbing follows env vars (constructor kwargs override):

- ``CRUCIBLE_LOCAL_URL`` / ``base_url=``    — endpoint base URL.
- ``CRUCIBLE_LOCAL_API_KEY`` / ``api_key=`` — bearer token. For the
  NVIDIA Inference Hub, point ``base_url`` at the hub and pass the
  hub key (operators commonly export it as ``NVIDIA_API_KEY`` and
  thread it into the routing YAML's ``api_key`` — but the key value
  itself is what this adapter holds, not the env-var name).

The ``model`` kwarg is required (the server exposes a model id you
must know to route correctly).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import os
import threading
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace as _replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendPolicy,
    BudgetExhausted,
    ContentRejected,
    ExecutionCompleteness,
    ExternalCancellation,
    IncompleteEvidence,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    Message,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from crucible.tools._dispatch_budget import (
    TerminalToolDispatchError,
    ToolDispatchBudgetExceeded,
    ToolMutationIndeterminate,
)

from ._classify import ClassifiedError, RETRIABLE_HTTP_STATUSES
from ._sdk_debug import SdkDebugSink
from .capabilities import toolset_write_names
from .pricing import compute_cost

logger = logging.getLogger(__name__)

# Most local servers do not validate a key, but the OpenAI client requires a
# non-empty value. Explicit hosted routes must provide their real credential.
_DEFAULT_API_KEY = "not-needed"
_DEFAULT_MODEL = ""  # No safe default — caller must pass a model id.

_SOURCE_EDIT_TOOL_NAMES = frozenset({
    "apply_patch",
    "delete_source",
    "write_source",
})
_CONTROL_POLL_SECONDS = 0.05

# One wall for a whole call: every request of a tool loop, its retries and
# backoff, the tool dispatches between them and the forced finalization.
# Ten minutes is less than a multi-round planner or writer call may need,
# so the default is the 45 minutes of the Codex adapter. A route sets
# another wall with ``timeout_seconds``; a policy on the Invocation still
# decides the wall of that call. The wall is checked before and after each
# request, and what is left of it is handed to the SDK as the timeout of
# the request, which its HTTP client applies to the connect, to each read,
# to each write and to the wait for a connection separately. Nothing
# shorter bounds a single request, and a reply that keeps arriving is not
# cut at the wall.
_DEFAULT_TIMEOUT_SECONDS = 2700.0

_DEFAULT_POLICY = BackendPolicy(
    retry_budget=3,
    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    on_cancellation="convert_to_backend_error",
    on_max_turns="raise",
)


@dataclass(frozen=True)
class _ForcedFinalOutcome:
    """Provider evidence retained from one no-tools finalization call."""

    text: str
    input_tokens: int
    output_tokens: int
    served_model: str | None
    provider_request_id: str | None
    model_identity_error: IncompleteEvidence | None
    # Whether the reply itself reported both token counters, and whether an
    # earlier attempt of this finalization was discarded before it arrived.
    usage_complete: bool
    retried: bool


class _InvocationControl:
    """One absolute wall deadline plus backend-lifetime cancellation.

    OpenAI-compatible tool handlers are synchronous Python calls, so Crucible
    cannot safely preempt one halfway through a mutation.  The shared control
    is nevertheless checked before and after every provider request, retry
    delay, tool dispatch, and forced-finalization request.  That makes the
    route hard-deadline-aware for HTTP and backoff while retaining the honest
    ``boundary_only`` cancellation feature declaration for tool execution.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float | None,
        close_event: threading.Event,
    ) -> None:
        if timeout_seconds is None:
            self.deadline: float | None = None
        else:
            if (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(float(timeout_seconds))
                or float(timeout_seconds) <= 0
            ):
                raise ValueError(
                    "OpenAIBackend timeout_seconds must be finite and positive"
                )
            self.deadline = time.monotonic() + float(timeout_seconds)
        self._close_event = close_event
        self.events: list[BackendExecutionEvent] = []

    def record(
        self,
        kind: str,
        *,
        name: str | None = None,
        parent_id: str | None = None,
        child_id: str | None = None,
    ) -> None:
        self.events.append(
            BackendExecutionEvent(
                sequence=len(self.events),
                kind=kind,
                name=name,
                parent_id=parent_id,
                child_id=child_id,
            )
        )

    def record_stop(self, stopped: str) -> None:
        kind = "cancellation" if stopped == "cancellation" else "limit"
        name = "close" if stopped == "cancellation" else "deadline"
        if self.events and (
            self.events[-1].kind == kind and self.events[-1].name == name
        ):
            return
        self.record(kind, name=name)

    def stopped(self) -> str | None:
        if self._close_event.is_set():
            return "cancellation"
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return "deadline"
        return None

    def request_timeout(self) -> float | None:
        """Return the remaining provider-request timeout.

        Callers must first check :meth:`stopped`; the tiny positive floor
        avoids handing a compatible SDK a zero timeout due solely to clock
        movement between the two calls.
        """

        if self.deadline is None:
            return None
        return max(self.deadline - time.monotonic(), 1e-6)

    def wait(self, delay: float) -> str | None:
        """Wait interruptibly without extending the invocation deadline."""

        remaining_delay = max(0.0, delay)
        while remaining_delay > 0:
            stopped = self.stopped()
            if stopped is not None:
                return stopped
            chunk = min(_CONTROL_POLL_SECONDS, remaining_delay)
            if self.deadline is not None:
                chunk = min(chunk, max(0.0, self.deadline - time.monotonic()))
                if chunk <= 0:
                    return "deadline"
            # Short polling bounds close() latency while retaining the
            # adapter's replaceable ``time.sleep`` seam for deterministic
            # retry tests.
            time.sleep(chunk)
            remaining_delay -= chunk
        return self.stopped()


def _mark_prior_attempts_incomplete(
    result: InvocationResult,
) -> InvocationResult:
    """Conservatively account for a discarded physical retry attempt."""

    return _replace(
        result,
        completeness=_replace(
            result.completeness,
            token_usage_complete=False,
            cost_complete=False,
        ),
        pricing_evidence=(
            None
            if result.pricing_evidence is None
            else _replace(result.pricing_evidence, complete=False)
        ),
    )


def _endpoint_identity(base_url: str) -> str | None:
    """Return a credential- and query-free endpoint identity."""

    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if not parsed.scheme or not hostname:
        return None
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = rendered_host if port is None else f"{rendered_host}:{port}"
    return f"{parsed.scheme}://{netloc}{parsed.path}"


def _provider_request_id(response: Any) -> str | None:
    """Return one exact non-empty provider request identifier."""

    response_id = getattr(response, "id", None)
    if isinstance(response_id, str) and response_id.strip():
        return response_id.strip()
    return None


class OpenAIBackend:
    """assay.Backend over an OpenAI-compatible HTTP server (local or hosted)."""

    name = "openai_compat"

    @property
    def features(self) -> Any:
        """Return the observed profile for this concrete HTTP route."""

        from .features import feature_profile_for_route

        return feature_profile_for_route(
            self.name,
            model=self.model,
            base_url=self._base_url,
        )

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        cwd: str | None = None,
        log_dir: Path | str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        expected_served_model: str | None = None,
        capability: str = "text",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        from .capabilities import normalize_capability

        self.capability = normalize_capability(capability)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "OpenAIBackend timeout_seconds must be a finite positive number"
            )
        self._default_policy = _replace(
            _DEFAULT_POLICY,
            timeout_seconds=float(timeout_seconds),
        )
        if not model:
            env_model = os.environ.get("CRUCIBLE_LOCAL_MODEL", "").strip()
            if not env_model:
                raise ValueError(
                    "OpenAIBackend requires a model id — pass model=... "
                    "or set CRUCIBLE_LOCAL_MODEL"
                )
            model = env_model
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAIBackend model must be a non-empty string")
        self.model = model
        if expected_served_model is not None and (
            type(expected_served_model) is not str
            or not expected_served_model
            or expected_served_model != expected_served_model.strip()
        ):
            raise ValueError(
                "OpenAIBackend expected_served_model must be an exact "
                "non-empty string"
            )
        self._expected_served_model = expected_served_model
        # ``INFER_BASE_URL`` / ``INFER_AUTH_TOKEN`` are the workspace's
        # neutral NVIDIA-gateway vars (~/.config/crucible/env), shared with
        # the claude backend. They sit below explicit kwargs and the
        # CRUCIBLE_LOCAL_* overrides but above the localhost default, so the
        # hub is reached by default while a local vLLM/Ollama run can still
        # override via kwarg or CRUCIBLE_LOCAL_*.
        self._base_url = (
            base_url
            or os.environ.get("CRUCIBLE_LOCAL_URL", "").strip()
            or os.environ.get("INFER_BASE_URL", "").strip()
            or "http://localhost:8000/v1"
        )
        self._api_key = (
            api_key
            or os.environ.get("CRUCIBLE_LOCAL_API_KEY", "").strip()
            or os.environ.get("INFER_AUTH_TOKEN", "").strip()
            or _DEFAULT_API_KEY
        )
        # cwd is accepted for symmetry with ClaudeBackend / CodexBackend
        # so the resolve() factory can pass it uniformly. The OpenAI tool
        # loop dispatches through the attached ToolSet's own ToolContext
        # (which carries repo_root); this cwd is recorded for postmortem.
        self._cwd = cwd or os.getcwd()
        self._sdk_sink = SdkDebugSink(
            Path(log_dir) if log_dir is not None else None,
            backend_name="openai_compat",
        )
        self._client: Any = None
        self._client_lock = threading.RLock()
        self._close_event = threading.Event()
        # Reasoning-effort knob. The hub honors the top-level
        # ``reasoning_effort`` param (confirmed: low→high raises the
        # reasoning_content length for both deepseek-v4-pro + kimi-k2.6;
        # the nested ``reasoning: {...}`` block 400s as unsupported). When
        # set, it's sent on every chat-completions call so hosted
        # reasoning models run at the configured effort rather than the
        # hub default. Routed in from ``effort:`` in backends.yaml.
        self._reasoning_effort = reasoning_effort
        # Output-token cap. Reasoning models spend this budget on the
        # thinking block FIRST, then the visible answer — a small cap
        # truncates the answer (or yields empty content: kimi-k2.6 at
        # max_tokens=256/effort=high emits 756 reasoning chars and an
        # EMPTY visible message). Default generously so think+answer+
        # tool-args all fit.
        self._max_tokens = max_tokens if max_tokens is not None else 16384
        # Set by attach_toolset(...). When present, invoke() runs the
        # multi-turn tool-use loop instead of a single-shot completion.
        self._toolset: Any = None
        self._context_budget: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, invocation: Invocation) -> InvocationResult:
        if invocation.conversation is not None:
            raise NotImplementedError(
                "OpenAIBackend does not support provider conversation "
                "continuity"
            )
        from .pricing import pricing_for_model

        pricing_for_model(self.model)
        policy = invocation.policy or self._default_policy
        control = _InvocationControl(
            timeout_seconds=policy.timeout_seconds,
            close_event=self._close_event,
        )
        for attempt in range(max(1, policy.retry_budget)):
            stopped = control.stopped()
            if stopped is not None:
                return self._control_result(
                    stopped,
                    policy,
                    transcript=(),
                    usage=Usage(),
                    control=control,
                )
            started_iso = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds")
            start_mono = time.monotonic()
            try:
                result = self._invoke_once(invocation, policy, control)
            except asyncio.CancelledError as exc:
                if policy.on_cancellation == "raise":
                    raise
                result = self._control_result(
                    "cancellation",
                    policy,
                    transcript=(),
                    usage=Usage(),
                    message=f"OpenAI-compatible invocation cancelled: {exc}",
                    control=control,
                )
            except BaseException as exc:  # noqa: BLE001
                stopped = control.stopped()
                if stopped is not None:
                    result = self._control_result(
                        stopped,
                        policy,
                        transcript=(),
                        usage=Usage(),
                        control=control,
                    )
                    self._sdk_sink.dump(
                        invocation=invocation,
                        result=result,
                        started_iso=started_iso,
                        duration_s=time.monotonic() - start_mono,
                        attempt=attempt,
                    )
                    return result
                err = self._classify(exc)
                if err is None:
                    self._sdk_sink.dump(
                        invocation=invocation, result=None,
                        started_iso=started_iso,
                        duration_s=time.monotonic() - start_mono,
                        re_raised=True, attempt=attempt,
                    )
                    raise
                result = self._result_with_evidence(
                    final_message="",
                    transcript=(),
                    usage=Usage(),
                    stop_reason="error",
                    error=err,
                    usage_complete=False,
                    events=control.events,
                )
                if not err.retriable or attempt + 1 >= policy.retry_budget:
                    self._sdk_sink.dump(
                        invocation=invocation, result=result,
                        started_iso=started_iso,
                        duration_s=time.monotonic() - start_mono,
                        attempt=attempt,
                    )
                    return result
                self._sdk_sink.dump(
                    invocation=invocation, result=result,
                    started_iso=started_iso,
                    duration_s=time.monotonic() - start_mono,
                    attempt=attempt,
                )
                backoff = min(2 ** attempt, 16) + (time.monotonic() % 1.0)
                logger.info(
                    "OpenAIBackend transient error on attempt %d (%s); "
                    "retrying in %.1fs",
                    attempt + 1, type(exc).__name__, backoff,
                )
                control.record(
                    "retry",
                    name=getattr(err, "kind", type(exc).__name__),
                )
                stopped = control.wait(backoff)
                if stopped is not None:
                    return self._control_result(
                        stopped,
                        policy,
                        transcript=(),
                        usage=Usage(),
                        control=control,
                    )
                continue
            self._sdk_sink.dump(
                invocation=invocation, result=result,
                started_iso=started_iso,
                duration_s=time.monotonic() - start_mono,
                attempt=attempt,
            )
            # Single-shot calls can return a structured retriable error (the
            # empty-completion guard below is the current producer) without
            # raising. Honor the same retry budget as exception-driven
            # failures. Tool loops are excluded: they already retry each HTTP
            # completion in-place to preserve the accumulated conversation,
            # and retrying here would multiply the budget and restart it.
            if (
                self._toolset is None
                and result.error is not None
                and getattr(result.error, "retriable", False)
                and attempt + 1 < policy.retry_budget
            ):
                backoff = min(2 ** attempt, 16) + (time.monotonic() % 1.0)
                logger.info(
                    "OpenAIBackend retriable error in result on attempt "
                    "%d (%s); retrying in %.1fs",
                    attempt + 1,
                    getattr(result.error, "kind", type(result.error).__name__),
                    backoff,
                )
                control.record(
                    "retry",
                    name=getattr(
                        result.error,
                        "kind",
                        type(result.error).__name__,
                    ),
                )
                stopped = control.wait(backoff)
                if stopped is not None:
                    return self._control_result(
                        stopped,
                        policy,
                        transcript=result.transcript,
                        usage=result.usage,
                        pricing_segments=(
                            ()
                            if result.pricing_evidence is None
                            else result.pricing_evidence.segments
                        ),
                        control=control,
                    )
                continue
            return (
                _mark_prior_attempts_incomplete(result)
                if attempt > 0
                else result
            )
        return self._result_with_evidence(
            final_message="",
            transcript=(),
            usage=Usage(),
            stop_reason="error",
            error=TransportError(
                message="retry loop exhausted with no result",
                retriable=False,
            ),
            usage_complete=False,
            events=control.events,
        )

    def attach_toolset(self, toolset: Any, *, context_budget: int | None = None) -> None:
        """Store a ToolSet so ``invoke()`` exposes it as OpenAI tools.

        ``assay.plan`` / ``assay.implement`` attach the toolset
        out-of-band, then call ``invoke()``; the multi-turn tool loop
        lives inside ``invoke()`` (see ``_invoke_with_tools``). Passing
        ``None`` detaches. ``context_budget`` is a hard per-invocation UTF-8
        byte ceiling on tool results fed back into the provider conversation.
        """
        from .capabilities import validate_toolset_capability

        validate_toolset_capability(self.capability, toolset)
        if context_budget is not None and (
            isinstance(context_budget, bool)
            or not isinstance(context_budget, int)
            or context_budget <= 0
        ):
            raise ValueError("context_budget must be a positive integer")
        self._toolset = toolset
        self._context_budget = context_budget if toolset is not None else None
        if toolset is not None:
            tool_count = "?"
            try:
                tool_count = str(len(toolset.to_openai_tools()))
            except (AttributeError, TypeError):
                pass
            logger.info(
                "OpenAIBackend: attached toolset %r exposing %s OpenAI tool(s)",
                getattr(toolset, "domain", "?"), tool_count,
            )

    def _tool_authority_digest(
        self,
        toolset: Any,
        *,
        context_budget: int | None,
    ) -> str:
        from .tool_authority import (
            APPROVAL_POLICY_PER_CALL_EXACT,
            TOOL_AUTHORITY_BRIDGE_PROTOCOL,
            TOOL_AUTHORITY_BRIDGE_VERSION,
            tool_authority_digest,
        )

        return tool_authority_digest(
            toolset,
            capability=self.capability,
            approval_policy=APPROVAL_POLICY_PER_CALL_EXACT,
            bridge_protocol=TOOL_AUTHORITY_BRIDGE_PROTOCOL,
            bridge_version=TOOL_AUTHORITY_BRIDGE_VERSION,
            context_budget=context_budget,
        )

    def chat_with_tools(
        self,
        invocation: Invocation,
        toolset: Any,
        *,
        cache: Any = None,
        max_rounds: int = 10,
        max_response_tokens: int = 16_000,
        retry_budget: int = 3,
        strict: bool = False,
    ) -> InvocationResult:
        """Direct OpenAI tool-use loop with an explicitly-passed toolset.

        Symmetric with ``ClaudeBackend.chat_with_tools``: drives the
        same multi-turn loop as ``invoke()``-with-attached-toolset, but
        takes the toolset as an argument rather than from a prior
        ``attach_toolset``. Provided so workflows that call
        ``chat_with_tools`` on a routed backend get a real tool loop on
        the OpenAI path instead of a silent fallback.

        ``max_rounds`` is an additional ceiling and can never widen the
        invocation's turn budget. Unsupported cache semantics are rejected
        before transport instead of being silently ignored.
        """
        if invocation.conversation is not None:
            raise NotImplementedError(
                "OpenAIBackend does not support provider conversation "
                "continuity"
            )
        if toolset is None:
            if strict:
                raise NotImplementedError(
                    "OpenAIBackend.chat_with_tools called with no toolset"
                )
            return self.invoke(invocation)
        from .capabilities import validate_toolset_capability

        validate_toolset_capability(self.capability, toolset)
        if cache is not None:
            raise NotImplementedError(
                "OpenAIBackend does not yet support ToolCallCache on the "
                "direct chat_with_tools path"
            )
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        if retry_budget <= 0:
            raise ValueError("retry_budget must be positive")
        policy = invocation.policy or self._default_policy
        policy = _replace(
            policy,
            retry_budget=min(policy.retry_budget, retry_budget),
        )
        control = _InvocationControl(
            timeout_seconds=policy.timeout_seconds,
            close_event=self._close_event,
        )
        stopped = control.stopped()
        if stopped is not None:
            return self._control_result(
                stopped,
                policy,
                transcript=(),
                usage=Usage(),
                control=control,
            )
        max_turns = min(max(1, invocation.max_turns or 1), max_rounds)
        return self._invoke_with_tools(
            invocation,
            policy,
            toolset,
            max_turns=max_turns,
            max_response_tokens=min(
                self._max_tokens,
                max_response_tokens,
            ),
            control=control,
        )

    def close(self) -> None:
        """Release any held HTTP client state, retaining failed owners."""
        self._close_event.set()
        with self._client_lock:
            client = self._client
            if client is None:
                return
            close_fn = getattr(client, "close", None)
            if not callable(close_fn):
                raise RuntimeError(
                    "OpenAI HTTP client does not expose callable close; "
                    "transport settlement is unproved"
                )
            # A failed close is not a settlement receipt. Retain this exact
            # client so the workflow can retry provider cleanup.
            close_fn()
            if self._client is client:
                self._client = None

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        with self._client_lock:
            if self._close_event.is_set():
                raise RuntimeError("OpenAIBackend is closed")
            if self._client is not None:
                return self._client
            try:
                import openai
            except ImportError as exc:  # pragma: no cover — install-time
                raise RuntimeError(
                    "OpenAIBackend requires the `openai` package. Install "
                    "via `uv add openai` or `pip install crucible[local]`."
                ) from exc
            self._client = openai.OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                # Crucible owns the retry budget and records every attempt.
                # SDK retries would multiply one recorded attempt into three
                # unobservable HTTP requests.
                max_retries=0,
            )
            return self._client

    def _served_model_identity(
        self,
        response: Any,
    ) -> tuple[str | None, IncompleteEvidence | None]:
        """Return and, when configured, admit one exact served model id.

        Generic operator routes retain their historical observe-only behavior.
        Governed routes opt in with ``expected_served_model``; on those routes
        every physical response must report that exact identity before its
        content or tool calls can be admitted.
        """

        raw_model = getattr(response, "model", None)
        served_model = (
            raw_model.strip()
            if isinstance(raw_model, str) and raw_model.strip()
            else None
        )
        expected = self._expected_served_model
        if expected is None:
            return served_model, None
        if (
            type(raw_model) is not str
            or not raw_model
            or raw_model != raw_model.strip()
        ):
            return None, IncompleteEvidence(
                missing=("execution_trace.served_model",),
                message=(
                    "OpenAI-compatible provider response omitted an exact "
                    "served-model identity required by this route"
                ),
                retriable=False,
            )
        if raw_model != expected:
            return raw_model, IncompleteEvidence(
                missing=(
                    "execution_trace.served_model_matches_expected",
                ),
                message=(
                    "OpenAI-compatible provider served a different model "
                    "than this route expected"
                ),
                retriable=False,
            )
        return raw_model, None

    def _invoke_once(
        self,
        invocation: Invocation,
        policy: BackendPolicy,
        control: _InvocationControl,
    ) -> InvocationResult:
        # When a ToolSet is attached, run the multi-turn tool loop so the
        # planner / implementer / defender roles get real tool use on the
        # OpenAI path. Otherwise a single completion (judge / breaker
        # single-shot roles).
        if self._toolset is not None:
            return self._invoke_with_tools(
                invocation, policy, self._toolset,
                max_turns=max(1, invocation.max_turns or 1),
                control=control,
            )

        stopped = control.stopped()
        if stopped is not None:
            return self._control_result(
                stopped,
                policy,
                transcript=(),
                usage=Usage(),
                control=control,
            )
        client = self._get_client()
        messages = [
            {"role": "system", "content": invocation.system},
            {"role": "user", "content": invocation.user},
        ]
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "timeout": control.request_timeout(),
            **self._reasoning_kwargs(),
            **self._structured_output_kwargs(invocation.response_schema),
        }

        control.record("request", name="single_shot")
        response = client.chat.completions.create(**create_kwargs)
        in_tokens, out_tokens = _usage_tokens(response)
        usage = self._usage(in_tokens, out_tokens)
        pricing_segments = _pricing_segments(
            _usage_is_complete(response),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=usage.cost_usd,
        )
        served_model, model_identity_error = self._served_model_identity(
            response
        )
        provider_request_id = _provider_request_id(response)
        control.record(
            "usage",
            name="complete" if _usage_is_complete(response) else "incomplete",
        )
        stopped = control.stopped()
        if stopped is not None:
            return self._control_result(
                stopped,
                policy,
                transcript=(Message(role="user", content=invocation.user),),
                usage=usage,
                pricing_segments=pricing_segments,
                served_model=served_model,
                provider_request_id=provider_request_id,
                control=control,
            )

        if model_identity_error is not None:
            control.record("limit", name="served_model_identity")
            return self._result_with_evidence(
                final_message="",
                transcript=(Message(role="user", content=invocation.user),),
                usage=usage,
                stop_reason=(
                    "missing_model_identity"
                    if served_model is None
                    else "model_mismatch"
                ),
                error=model_identity_error,
                usage_complete=False,
                pricing_segments=pricing_segments,
                served_model=served_model,
                provider_request_id=provider_request_id,
                events=control.events,
            )

        choice = response.choices[0]
        text = _message_text(choice.message)
        finish_reason = choice.finish_reason or "stop"
        transcript = (
            Message(role="user", content=invocation.user),
            Message(role="assistant", content=text),
        )

        if (
            invocation.budget_usd is not None
            and usage.cost_usd > invocation.budget_usd
        ):
            control.record("limit", name="budget_usd")
            return self._result_with_evidence(
                final_message=text,
                transcript=transcript,
                usage=usage,
                stop_reason="budget_exhausted",
                error=BudgetExhausted(
                    spent_usd=usage.cost_usd,
                    limit_usd=invocation.budget_usd,
                    message=(
                        f"Invocation spent ${usage.cost_usd:.4f} exceeding "
                        f"budget ${invocation.budget_usd:.4f}"
                    ),
                ),
                response=response,
                pricing_segments=pricing_segments,
                events=control.events,
            )

        if finish_reason == "length" and policy.on_max_turns == "raise":
            control.record("limit", name="max_turns")
            return self._result_with_evidence(
                final_message=text,
                transcript=transcript,
                usage=usage,
                stop_reason="max_turns",
                error=MaxTurnsExceeded(
                    turns_used=invocation.max_turns,
                    turns_limit=invocation.max_turns,
                    message="server returned finish_reason=length",
                ),
                response=response,
                pricing_segments=pricing_segments,
                events=control.events,
            )

        # Stalled / empty completion guard — mirrors ClaudeBackend /
        # CodexBackend. A clean ``finish_reason=stop`` with empty visible
        # content (kimi-k2.6 et al. can burn the whole token budget on the
        # hidden reasoning block and emit an empty answer) is a
        # content-free stall: returned as-is it looks like success, the
        # retry loop never re-tries it, and downstream JSON-parsing
        # consumers (assay.review / assay.plan) fail the gate on a
        # parse_failure. Convert to a retriable TransportError. This is
        # the non-tool single-shot path only (judge / review / plan); the
        # tool loop is handled in ``_invoke_with_tools``. Not gated on
        # output_tokens — reasoning models legitimately spend tokens on
        # the hidden think block while leaving visible text empty.
        if not text and finish_reason not in ("length",):
            return self._result_with_evidence(
                final_message="",
                transcript=transcript,
                usage=usage,
                stop_reason=finish_reason,
                error=TransportError(
                    http_status=None,
                    message=(
                        f"empty completion (finish_reason={finish_reason}) "
                        "— server returned a clean but content-free message"
                    ),
                    retriable=True,
                ),
                response=response,
                pricing_segments=pricing_segments,
                events=control.events,
            )

        return self._result_with_evidence(
            final_message=text,
            transcript=transcript,
            usage=usage,
            stop_reason="max_turns" if finish_reason == "length" else "end_turn",
            response=response,
            pricing_segments=pricing_segments,
            events=control.events,
        )

    def _control_result(
        self,
        stopped: str,
        policy: BackendPolicy,
        *,
        transcript: tuple[Message, ...],
        usage: Usage,
        final_message: str = "",
        message: str | None = None,
        pricing_segments: tuple[UsagePricingSegment, ...] = (),
        served_model: str | None = None,
        provider_request_id: str | None = None,
        tool_authority_digest: str | None = None,
        control: _InvocationControl | None = None,
    ) -> InvocationResult:
        """Normalize one deadline/cancellation terminal at a safe boundary."""

        error, stop_reason = self._control_error(stopped, policy, message=message)
        if control is not None:
            control.record_stop(stopped)
        return self._result_with_evidence(
            final_message=final_message,
            transcript=transcript,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            usage_complete=False,
            pricing_segments=pricing_segments,
            served_model=served_model,
            provider_request_id=provider_request_id,
            tool_authority_digest=tool_authority_digest,
            events=None if control is None else control.events,
        )

    def _control_error(
        self,
        stopped: str,
        policy: BackendPolicy,
        *,
        message: str | None = None,
    ) -> tuple[TransportError | ExternalCancellation, str]:
        """Build a policy-aware interruption error and stop reason."""

        if stopped == "cancellation":
            detail = message or "OpenAI-compatible backend closed during invocation"
            if policy.on_cancellation == "raise":
                raise asyncio.CancelledError(detail)
            return ExternalCancellation(
                cause="OpenAIBackend.close",
                message=detail,
                retriable=True,
            ), "cancelled"
        if stopped == "deadline":
            return TransportError(
                http_status=None,
                message=(
                    message
                    or "OpenAI-compatible invocation exhausted its cumulative "
                    "wall deadline"
                ),
                retriable=False,
            ), "timeout"
        raise AssertionError(  # pragma: no cover - internal invariant
            f"unknown invocation control state {stopped!r}"
        )

    def _reasoning_kwargs(self) -> dict[str, Any]:
        """Per-call extra kwargs carrying the reasoning-effort signal.

        Empty when no effort configured (the hub then runs its default
        effort). The hub accepts the top-level ``reasoning_effort`` param
        for hosted reasoning models; the nested ``reasoning:`` block is
        rejected as unsupported.
        """
        if self._reasoning_effort:
            return {"reasoning_effort": self._reasoning_effort}
        return {}

    def _structured_output_kwargs(self, response_schema: Any) -> dict[str, Any]:
        """Translate Assay's vendor-neutral schema to OpenAI strict JSON."""

        if response_schema is None:
            return {}
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "assay_structured_response",
                    "strict": True,
                    "schema": deepcopy(response_schema),
                },
            }
        }

    def _invoke_with_tools(
        self,
        invocation: Invocation,
        policy: BackendPolicy,
        toolset: Any,
        *,
        max_turns: int,
        max_response_tokens: int | None = None,
        control: _InvocationControl,
    ) -> InvocationResult:
        """Multi-turn OpenAI function-calling loop over ``toolset``.

        Each round POSTs the running message list + the toolset's
        OpenAI tool catalog. If the model returns ``tool_calls`` we
        dispatch each through ``toolset.dispatch(name, args)``, append
        the results as ``role=tool`` messages, and iterate. The loop
        ends when the model stops calling tools, on ``finish_reason=
        length``, when ``max_turns`` is hit, or when accumulated cost
        exceeds ``invocation.budget_usd``.
        """
        tool_authority_digest = self._tool_authority_digest(
            toolset,
            context_budget=(
                self._context_budget
                if toolset is self._toolset
                else None
            ),
        )
        stopped = control.stopped()
        if stopped is not None:
            return self._control_result(
                stopped,
                policy,
                transcript=(),
                usage=Usage(),
                tool_authority_digest=tool_authority_digest,
                control=control,
            )
        # Validate the complete, explicitly classified authority before even
        # constructing an SDK client. Missing or inconsistent ToolSet metadata
        # therefore fails closed before any provider transport can begin.
        client = self._get_client()
        response_token_limit = (
            self._max_tokens
            if max_response_tokens is None
            else min(self._max_tokens, max_response_tokens)
        )
        openai_tools = toolset.to_openai_tools()
        write_names = toolset_write_names(toolset) or set()
        source_edit_names = write_names & _SOURCE_EDIT_TOOL_NAMES
        write_tools = [
            descriptor
            for descriptor in openai_tools
            if descriptor.get("function", {}).get("name") in source_edit_names
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": invocation.system},
            {"role": "user", "content": invocation.user},
        ]
        transcript: list[Message] = [
            Message(role="user", content=invocation.user),
        ]
        total_in = 0
        total_out = 0
        total_cost_usd = 0.0
        # One segment per provider response that reported its usage, in order.
        pricing_segments: list[UsagePricingSegment] = []
        final_text = ""
        stop_reason = "end_turn"
        # True until the latest source-edit obligation is satisfied.  A later
        # rejected edit re-opens the obligation even if an earlier patch
        # succeeded; otherwise one partial write can mask an unresolved safety
        # rejection and let the model end in prose.
        source_edit_unresolved = bool(write_tools)
        source_edit_failure_recovery_used = False
        length_write_continuation_used = False
        write_only_continuation_pending = False
        implementation_phase_announced = False
        # Mutation-capable invocations promise the model a bounded
        # investigation window.  Enforce that boundary in the transport
        # instead of relying on prompt compliance: after the first third of
        # turns, retaining read tools causes every subsequent request to
        # replay an ever-growing body of source results and lets a partial
        # edit reopen unbounded exploration.
        implementation_phase_turn = max(1, (max_turns + 2) // 3)
        loop_exhausted = False
        context_result_bytes = 0
        saw_response = False
        all_usage_complete = True
        served_model: str | None = None
        provider_request_id: str | None = None
        next_request_phase = "tool"

        def stopped_result(stopped: str) -> InvocationResult:
            return self._control_result(
                stopped,
                policy,
                transcript=tuple(transcript),
                usage=self._usage(
                    total_in,
                    total_out,
                    cost_usd=total_cost_usd,
                ),
                final_message=final_text,
                pricing_segments=tuple(pricing_segments),
                served_model=served_model,
                provider_request_id=provider_request_id,
                tool_authority_digest=tool_authority_digest,
                control=control,
            )

        def retain_forced_evidence(
            outcome: _ForcedFinalOutcome | None,
            error: ClassifiedError | None,
        ) -> None:
            """Retain what one detached finalization physically consumed."""

            nonlocal all_usage_complete
            nonlocal provider_request_id
            nonlocal served_model
            nonlocal total_cost_usd
            nonlocal total_in
            nonlocal total_out

            if outcome is None:
                # A classified failure can occur after the provider accepted
                # and billed the request but before its response envelope
                # reached Crucible. The prior aggregate is then only a known
                # lower bound. A budget skip returns neither outcome nor error
                # and therefore does not poison evidence.
                if error is not None:
                    all_usage_complete = False
                return
            # The finalization is one more provider response: the aggregate
            # stays complete only if its reply reported its usage and no
            # attempt of it was discarded.
            all_usage_complete = (
                all_usage_complete
                and outcome.usage_complete
                and not outcome.retried
            )
            outcome_cost_usd = compute_cost(
                self.model,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
            )
            total_in += outcome.input_tokens
            total_out += outcome.output_tokens
            total_cost_usd += outcome_cost_usd
            pricing_segments.extend(
                _pricing_segments(
                    outcome.usage_complete,
                    input_tokens=outcome.input_tokens,
                    output_tokens=outcome.output_tokens,
                    cost_usd=outcome_cost_usd,
                )
            )
            if (
                self._expected_served_model is not None
                or outcome.served_model is not None
            ):
                served_model = outcome.served_model
            if (
                self._expected_served_model is not None
                or outcome.provider_request_id is not None
            ):
                provider_request_id = outcome.provider_request_id

        for turn in range(max_turns + 1):
            stopped = control.stopped()
            if stopped is not None:
                return stopped_result(stopped)
            if turn == max_turns and not write_only_continuation_pending:
                loop_exhausted = True
                break
            # A generic no-tools finalization can summarize an analytical
            # answer, but it cannot rescue an implementation that spent every
            # turn reading. Reserve the final in-budget tool turn for scoped
            # mutators when the role has not completed a write yet. Read-only
            # roles retain the historical all-tools loop and detached final
            # answer below.
            write_only_continuation_turn = write_only_continuation_pending
            write_only_continuation_pending = False
            request_phase = next_request_phase
            next_request_phase = "tool"
            require_tool_call = write_only_continuation_turn
            turn_tools = (
                write_tools if write_only_continuation_turn else openai_tools
            )
            implementation_phase_active = bool(
                write_tools
                and turn >= implementation_phase_turn
            )
            if (
                implementation_phase_active
                and not write_only_continuation_turn
            ):
                request_phase = "implementation_phase"
                turn_tools = write_tools
                # Once the latest edit obligation is satisfied, keep the
                # scoped write catalog but allow the model either to make
                # additional edits or to end. Until then, prose cannot consume
                # the implementation phase.
                require_tool_call = source_edit_unresolved
                if not implementation_phase_announced:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Implementation phase: the bounded investigation "
                            "window is complete. Available tools are now "
                            "limited to scoped source edits. Apply the "
                            "requested implementation now; do not request "
                            "additional reads or test runs. You may make "
                            "multiple targeted edits, then return a concise "
                            "final answer. Deterministic workflow verification "
                            "will run afterward."
                        ),
                    })
                    implementation_phase_announced = True
            if (
                not write_only_continuation_turn
                and turn == max_turns - 1
                and write_tools
                and source_edit_unresolved
            ):
                request_phase = "final_write"
                turn_tools = write_tools
                require_tool_call = True
                messages.append({
                    "role": "user",
                    "content": (
                        "Final implementation turn: investigation is over. "
                        "Use one of the available scoped write tools NOW to "
                        "apply your best targeted source edit. Do not request "
                        "another read or test run; the workflow will verify "
                        "the resulting diff after this turn."
                    ),
                })
            # Per-completion transport retry — IN-PLACE so the conversation
            # context is never reset. For OpenAI the conversation IS the
            # client-side ``messages`` list (system + user + every prior
            # turn's assistant/tool messages); a transport blip on round N's
            # create() must NOT unwind to invoke() and restart the whole
            # loop with a fresh ``messages`` (which would discard rounds
            # 0..N-1 of exploration + tool results). Instead we re-issue the
            # SAME create() with the SAME messages, so the model continues
            # from full context. Only a non-retriable error, or exhausting
            # the budget, returns — carrying accumulated usage so cost
            # rollups stay accurate. Mirrors chat_with_tools' inner retry.
            response = None
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": turn_tools,
                "max_tokens": response_token_limit,
                "timeout": control.request_timeout(),
                **self._reasoning_kwargs(),
            }
            if require_tool_call:
                # These turns expose only scoped source mutators. Requiring a
                # tool call prevents a write-only completion from ending in
                # planning prose without changing the workspace.
                request_kwargs["tool_choice"] = "required"
            for attempt in range(1, max(1, policy.retry_budget) + 1):
                stopped = control.stopped()
                if stopped is not None:
                    return stopped_result(stopped)
                request_kwargs["timeout"] = control.request_timeout()
                request_name = (
                    f"{request_phase}_round_{turn + 1}_attempt_{attempt}"
                )
                control.record("request", name=request_name)
                try:
                    response = client.chat.completions.create(**request_kwargs)
                    break
                except BaseException as exc:  # noqa: BLE001
                    stopped = control.stopped()
                    if stopped is not None:
                        return stopped_result(stopped)
                    err = self._classify(exc)
                    if err is None:
                        raise
                    if getattr(err, "retriable", False) and attempt < policy.retry_budget:
                        # A transport fault can occur after the provider has
                        # accepted/billed the request but before usage reaches
                        # us. The successful retry cannot make that discarded
                        # attempt's token/cost evidence complete.
                        all_usage_complete = False
                        backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1.0)
                        logger.info(
                            "OpenAIBackend transient error on turn %d attempt "
                            "%d (%s); retrying in %.1fs — conversation "
                            "preserved (%d messages, no restart)",
                            turn, attempt,
                            getattr(err, "kind", type(exc).__name__),
                            backoff, len(messages),
                        )
                        control.record(
                            "retry",
                            name=getattr(err, "kind", type(exc).__name__),
                        )
                        stopped = control.wait(backoff)
                        if stopped is not None:
                            return stopped_result(stopped)
                        continue
                    return self._result_with_evidence(
                        pricing_segments=tuple(pricing_segments),
                        final_message=final_text,
                        transcript=tuple(transcript),
                        usage=self._usage(
                            total_in,
                            total_out,
                            cost_usd=total_cost_usd,
                        ),
                        stop_reason="error",
                        error=err,
                        usage_complete=False,
                        served_model=served_model,
                        provider_request_id=provider_request_id,
                        events=control.events,
                        tool_authority_digest=tool_authority_digest,
                    )
            saw_response = True
            all_usage_complete = (
                all_usage_complete and _usage_is_complete(response)
            )
            observed_served_model, model_identity_error = (
                self._served_model_identity(response)
            )
            if (
                self._expected_served_model is not None
                or observed_served_model is not None
            ):
                served_model = observed_served_model
            observed_request_id = _provider_request_id(response)
            if (
                self._expected_served_model is not None
                or observed_request_id is not None
            ):
                provider_request_id = observed_request_id
            in_tok, out_tok = _usage_tokens(response)
            response_cost_usd = compute_cost(
                self.model,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            total_in += in_tok
            total_out += out_tok
            total_cost_usd += response_cost_usd
            pricing_segments.extend(
                _pricing_segments(
                    _usage_is_complete(response),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=response_cost_usd,
                )
            )
            control.record(
                "usage",
                name=(
                    "complete"
                    if _usage_is_complete(response)
                    else "incomplete"
                ),
            )
            stopped = control.stopped()
            if stopped is not None:
                return stopped_result(stopped)
            if model_identity_error is not None:
                # Account for the billed response, but admit neither its
                # content nor its tool calls when the configured route's
                # exact physical model identity is unproved.
                control.record("limit", name="served_model_identity")
                return self._result_with_evidence(
                    pricing_segments=tuple(pricing_segments),
                    final_message=final_text,
                    transcript=tuple(transcript),
                    usage=self._usage(
                        total_in,
                        total_out,
                        cost_usd=total_cost_usd,
                    ),
                    stop_reason=(
                        "missing_model_identity"
                        if served_model is None
                        else "model_mismatch"
                    ),
                    error=model_identity_error,
                    usage_complete=False,
                    served_model=served_model,
                    provider_request_id=provider_request_id,
                    events=control.events,
                    tool_authority_digest=tool_authority_digest,
                )

            choice = response.choices[0]
            msg = choice.message
            finish_reason = choice.finish_reason or "stop"
            text = msg.content or ""
            if text:
                final_text = text
                transcript.append(Message(role="assistant", content=text))

            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                if require_tool_call:
                    # Some compatible gateways ignore ``tool_choice=required``.
                    # Preserve their prose for causal context, but do not
                    # report a successful end turn while a write obligation is
                    # unresolved. Normal implementation-phase turns may try
                    # again within the existing budget; the last/special turn
                    # receives at most one bounded write-only recovery.
                    messages.append(_assistant_text_message(msg, text))
                    running = total_cost_usd
                    if (
                        invocation.budget_usd is not None
                        and running >= invocation.budget_usd
                    ):
                        stop_reason = "budget_exhausted"
                        control.record("limit", name="budget_usd")
                        logger.info(
                            "OpenAIBackend required-write retry skipped at "
                            "budget cap $%.4f >= $%.4f on turn %d",
                            running,
                            invocation.budget_usd,
                            turn,
                        )
                        break
                    if (
                        not write_only_continuation_turn
                        and turn < max_turns - 1
                    ):
                        messages.append({
                            "role": "user",
                            "content": (
                                "A scoped source edit is still required. The "
                                "previous response did not call an available "
                                "write tool. Use one of the currently exposed "
                                "source-edit tools now; do not return prose or "
                                "request another read."
                            ),
                        })
                        next_request_phase = "required_write_nudge"
                        stop_reason = "max_turns"
                        continue
                    if not source_edit_failure_recovery_used:
                        messages.append({
                            "role": "user",
                            "content": (
                                "Bounded source-edit recovery: the required "
                                "write tool was not called. Use one available "
                                "scoped write tool NOW. Do not read, test, or "
                                "return planning prose."
                            ),
                        })
                        source_edit_failure_recovery_used = True
                        write_only_continuation_pending = True
                        next_request_phase = "write_recovery"
                        stop_reason = "max_turns"
                        continue
                    stop_reason = "max_turns"
                    break
                if (
                    finish_reason == "length"
                    and write_tools
                    and source_edit_unresolved
                    and not length_write_continuation_used
                    and turn < max_turns - 1
                ):
                    running = total_cost_usd
                    if (
                        invocation.budget_usd is not None
                        and running >= invocation.budget_usd
                    ):
                        stop_reason = "budget_exhausted"
                        control.record("limit", name="budget_usd")
                        logger.info(
                            "OpenAIBackend early-length write continuation "
                            "skipped at budget cap $%.4f >= $%.4f on turn %d",
                            running,
                            invocation.budget_usd,
                            turn,
                        )
                        break
                    # Continue the exact client-side conversation. The partial
                    # assistant text is retained before the workflow-owned
                    # nudge, and every prior assistant tool_call/tool result
                    # remains in ``messages`` with its original tool_call_id.
                    messages.append(_assistant_text_message(msg, text))
                    messages.append({
                        "role": "user",
                        "content": (
                            "Bounded implementation completion: the previous "
                            "response hit its output limit before applying a "
                            "write. Investigation is over. Use one available "
                            "scoped write tool NOW to complete the requested "
                            "implementation. Do not read, test, or ask for "
                            "another turn."
                        ),
                    })
                    length_write_continuation_used = True
                    write_only_continuation_pending = True
                    next_request_phase = "length_write_recovery"
                    stop_reason = "max_turns"
                    continue
                if write_only_continuation_turn:
                    # The sole same-context completion did not write. Keep the
                    # causal max-turn signal so a workflow may use its own
                    # bounded, journaled fallback rather than treating model
                    # prose as a completed implementation.
                    stop_reason = "max_turns"
                    break
                stop_reason = (
                    "max_turns" if finish_reason == "length" else "end_turn"
                )
                # Some OpenAI-compatible gateways (notably NVIDIA GLM-5.2)
                # suppress tool calls when ``tools`` and a strict
                # ``response_format`` are sent together.  Tool-enabled turns
                # therefore run without the response schema.  Once exploration
                # ends naturally, make one no-tools completion that turns the
                # draft into the schema-constrained final response.
                if invocation.response_schema is not None:
                    final_messages = messages + [
                        _assistant_text_message(msg, text),
                    ]
                    forced, final_error = self._force_final_answer(
                        client,
                        final_messages,
                        policy,
                        total_cost_usd,
                        budget_usd=invocation.budget_usd,
                        response_schema=invocation.response_schema,
                        max_response_tokens=response_token_limit,
                        nudge=(
                            "Tool exploration is complete. Produce the final "
                            "answer now in the exact structured format the "
                            "system prompt requires. Return only that final "
                            "structured response."
                        ),
                        control=control,
                    )
                    retain_forced_evidence(forced, final_error)
                    if final_error is not None:
                        stopped = control.stopped()
                        if stopped is not None:
                            return stopped_result(stopped)
                        return self._result_with_evidence(
                            pricing_segments=tuple(pricing_segments),
                            final_message=final_text,
                            transcript=tuple(transcript),
                            usage=self._usage(
                                total_in,
                                total_out,
                                cost_usd=total_cost_usd,
                            ),
                            stop_reason="error",
                            error=final_error,
                            usage_complete=False,
                            served_model=served_model,
                            provider_request_id=provider_request_id,
                            events=control.events,
                            tool_authority_digest=tool_authority_digest,
                        )
                    if forced is not None:
                        if forced.model_identity_error is not None:
                            control.record(
                                "limit", name="served_model_identity"
                            )
                            return self._result_with_evidence(
                                pricing_segments=tuple(pricing_segments),
                                final_message="",
                                transcript=tuple(transcript),
                                usage=self._usage(
                                    total_in,
                                    total_out,
                                    cost_usd=total_cost_usd,
                                ),
                                stop_reason=(
                                    "missing_model_identity"
                                    if forced.served_model is None
                                    else "model_mismatch"
                                ),
                                error=forced.model_identity_error,
                                usage_complete=False,
                                served_model=forced.served_model,
                                provider_request_id=provider_request_id,
                                events=control.events,
                                tool_authority_digest=tool_authority_digest,
                            )
                        if forced.text:
                            final_text = forced.text
                            transcript.append(Message(
                                role="assistant", content=forced.text,
                            ))
                break

            # Echo the assistant's tool-call turn back verbatim so the
            # follow-up tool messages reference valid tool_call_ids.
            messages.append(_assistant_tool_call_message(msg, tool_calls))
            allowed_tool_names = {
                descriptor.get("function", {}).get("name")
                for descriptor in turn_tools
            }

            for tc in tool_calls:
                stopped = control.stopped()
                if stopped is not None:
                    return stopped_result(stopped)
                name = tc.function.name
                control.record(
                    "tool_call",
                    name=name,
                    parent_id=request_name,
                    child_id=tc.id,
                )
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
                if name not in allowed_tool_names:
                    # Descriptor filtering is not an authorization boundary:
                    # compatible gateways can emit a stale or hallucinated
                    # function name. Enforce the per-turn catalog before
                    # dispatch so read tools cannot cross into write phase.
                    result_str = (
                        f"ERROR tool {name!r} is unavailable in the current "
                        "workflow phase; use one of the exposed tools"
                    )
                else:
                    try:
                        result_str = toolset.dispatch(name, args)
                    except TerminalToolDispatchError as exc:
                        mutation_indeterminate = isinstance(
                            exc,
                            ToolMutationIndeterminate,
                        )
                        budget_exhausted = isinstance(
                            exc,
                            ToolDispatchBudgetExceeded,
                        )
                        if budget_exhausted:
                            control.record(
                                "limit",
                                name="tool_dispatch_budget",
                            )
                        return self._result_with_evidence(
                            pricing_segments=tuple(pricing_segments),
                            final_message=final_text,
                            transcript=tuple(transcript),
                            usage=self._usage(
                                total_in,
                                total_out,
                                cost_usd=total_cost_usd,
                            ),
                            stop_reason=(
                                "tool_mutation_indeterminate"
                                if mutation_indeterminate
                                else (
                                    "tool_budget_exhausted"
                                    if budget_exhausted
                                    else "tool_dispatch_terminal"
                                )
                            ),
                            error=(
                                MaxTurnsExceeded(
                                    turns_used=turn + 1,
                                    turns_limit=max_turns,
                                    message=(
                                        "OpenAI tool loop stopped at a "
                                        f"terminal dispatch boundary: {exc}"
                                    ),
                                )
                                if budget_exhausted
                                else TransportError(
                                    http_status=None,
                                    message=(
                                        "OpenAI tool loop terminated at a "
                                        "workflow-owned dispatch boundary: "
                                        f"{exc}"
                                    ),
                                    retriable=False,
                                )
                            ),
                            usage_complete=(
                                saw_response and all_usage_complete
                            ),
                            served_model=served_model,
                            provider_request_id=provider_request_id,
                            events=control.events,
                            tool_authority_digest=tool_authority_digest,
                        )
                    except Exception as exc:  # noqa: BLE001 — feed back
                        result_str = f"ERROR dispatching {name}: {exc}"
                result_bytes = len(str(result_str).encode("utf-8"))
                if (
                    self._context_budget is not None
                    and context_result_bytes + result_bytes
                    > self._context_budget
                ):
                    control.record("limit", name="context_bytes")
                    return self._result_with_evidence(
                        pricing_segments=tuple(pricing_segments),
                        final_message=final_text,
                        transcript=tuple(transcript),
                        usage=self._usage(
                            total_in,
                            total_out,
                            cost_usd=total_cost_usd,
                        ),
                        stop_reason="context_limit",
                        error=TransportError(
                            http_status=None,
                            message=(
                                "tool-result context budget exceeded: "
                                f"{context_result_bytes + result_bytes} > "
                                f"{self._context_budget} UTF-8 bytes"
                            ),
                            retriable=False,
                        ),
                        usage_complete=saw_response and all_usage_complete,
                        served_model=served_model,
                        provider_request_id=provider_request_id,
                        events=control.events,
                        tool_authority_digest=tool_authority_digest,
                    )
                context_result_bytes += result_bytes
                if name in source_edit_names and name in allowed_tool_names:
                    edit_failed = str(result_str).lstrip().lower().startswith(
                        "error"
                    )
                    source_edit_unresolved = edit_failed
                transcript.append(
                    Message(role="assistant", content=f"[tool_call] {name}({args})")
                )
                transcript.append(
                    Message(role="tool", content=str(result_str))
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result_str),
                })
                control.record(
                    "tool_result",
                    name=name,
                    parent_id=request_name,
                    child_id=tc.id,
                )
                stopped = control.stopped()
                if stopped is not None:
                    return stopped_result(stopped)

            if (
                not write_only_continuation_turn
                and turn == max_turns - 1
                and source_edit_unresolved
                and not source_edit_failure_recovery_used
            ):
                messages.append({
                    "role": "user",
                    "content": (
                        "Bounded source-edit recovery: your final scoped write "
                        "was rejected by the tool's safety contract. Preserve "
                        "all prior context and use one available write tool "
                        "NOW to make the targeted edit. Follow the exact tool "
                        "error: prefer apply_patch for an existing file, or "
                        "supply the required complete-read hash. Do not read, "
                        "test, or return planning prose."
                    ),
                })
                source_edit_failure_recovery_used = True
                write_only_continuation_pending = True
                next_request_phase = "rejected_write_recovery"

            if (
                write_only_continuation_turn
                and source_edit_unresolved
                and not source_edit_failure_recovery_used
            ):
                # A rejected early-length continuation still has the same
                # single bounded safety-recovery allowance as a rejected final
                # turn.  Do not abandon unused invocation turns merely because
                # the first forced edit failed validation.
                messages.append({
                    "role": "user",
                    "content": (
                        "Bounded source-edit recovery: the scoped write was "
                        "rejected by the tool's safety contract. Follow the "
                        "exact tool error and use one available write tool NOW. "
                        "Do not read, test, or return planning prose."
                    ),
                })
                source_edit_failure_recovery_used = True
                write_only_continuation_pending = True
                next_request_phase = "rejected_write_recovery"

            # Soft cost cap between rounds.
            if invocation.budget_usd is not None:
                running = total_cost_usd
                if running >= invocation.budget_usd:
                    stop_reason = "budget_exhausted"
                    control.record("limit", name="budget_usd")
                    logger.info(
                        "OpenAIBackend tool loop hit budget cap $%.4f >= $%.4f "
                        "at turn %d", running, invocation.budget_usd, turn,
                    )
                    break
            if (
                write_only_continuation_turn
                and not write_only_continuation_pending
            ):
                # Exactly one same-conversation write-only completion is
                # permitted. Its tool calls have been dispatched and retained
                # in the transcript; deterministic build/verify now owns the
                # result, whether the write succeeded or failed.
                stop_reason = "max_turns"
                break
        else:
            loop_exhausted = True

        if loop_exhausted:
            # Loop exhausted ``max_turns`` while the model was STILL
            # calling tools (the for-else fires only if we never hit a
            # `break`, i.e. every turn ended in tool_calls). Some models
            # — observed: deepseek-v4-pro in the planner role — never
            # stop investigating and so never emit their structured
            # answer; the loop ends mid-tool-call and `final_text` is
            # whatever scrap of visible text trailed the last call
            # (often empty), which fails every downstream JSON parser.
            # Force ONE final completion with tools removed: the model
            # physically cannot call a tool, so it must convert its
            # accumulated investigation into a final answer. Budget
            # permitting; on any fault we fall back to the partial.
            stop_reason = "max_turns"
            control.record("limit", name="max_turns")
            forced, final_error = self._force_final_answer(
                client, messages, policy, total_cost_usd,
                budget_usd=invocation.budget_usd,
                response_schema=invocation.response_schema,
                max_response_tokens=response_token_limit,
                control=control,
            )
            retain_forced_evidence(forced, final_error)
            if final_error is not None:
                stopped = control.stopped()
                if stopped is not None:
                    return stopped_result(stopped)
            if forced is not None:
                if forced.model_identity_error is not None:
                    control.record("limit", name="served_model_identity")
                    return self._result_with_evidence(
                        pricing_segments=tuple(pricing_segments),
                        final_message="",
                        transcript=tuple(transcript),
                        usage=self._usage(
                            total_in,
                            total_out,
                            cost_usd=total_cost_usd,
                        ),
                        stop_reason=(
                            "missing_model_identity"
                            if forced.served_model is None
                            else "model_mismatch"
                        ),
                        error=forced.model_identity_error,
                        usage_complete=False,
                        served_model=forced.served_model,
                        provider_request_id=provider_request_id,
                        events=control.events,
                        tool_authority_digest=tool_authority_digest,
                    )
                if forced.text:
                    final_text = forced.text
                    transcript.append(
                        Message(role="assistant", content=forced.text)
                    )
                    logger.info(
                        "OpenAIBackend: forced no-tools finalization turn "
                        "after max_turns produced %d chars", len(forced.text),
                    )

        final_usage = self._usage(
            total_in,
            total_out,
            cost_usd=total_cost_usd,
        )
        result_error: Any = None
        if invocation.budget_usd is not None and (
            stop_reason == "budget_exhausted"
            or final_usage.cost_usd > invocation.budget_usd
        ):
            stop_reason = "budget_exhausted"
            if not any(
                event.kind == "limit" and event.name == "budget_usd"
                for event in control.events
            ):
                control.record("limit", name="budget_usd")
            result_error = BudgetExhausted(
                spent_usd=final_usage.cost_usd,
                limit_usd=invocation.budget_usd,
                message=(
                    f"Invocation spent ${final_usage.cost_usd:.4f} exceeding "
                    f"budget ${invocation.budget_usd:.4f}"
                ),
            )
        elif stop_reason == "max_turns" and policy.on_max_turns == "raise":
            if not any(
                event.kind == "limit" and event.name == "max_turns"
                for event in control.events
            ):
                control.record("limit", name="max_turns")
            result_error = MaxTurnsExceeded(
                turns_used=max_turns,
                turns_limit=max_turns,
                message=(
                    "OpenAI-compatible tool loop exhausted its turn budget"
                ),
            )

        return self._result_with_evidence(
            pricing_segments=tuple(pricing_segments),
            final_message=final_text,
            transcript=tuple(transcript),
            usage=final_usage,
            stop_reason=stop_reason,
            error=result_error,
            usage_complete=saw_response and all_usage_complete,
            trace_complete=True,
            served_model=served_model,
            provider_request_id=provider_request_id,
            tool_authority_digest=tool_authority_digest,
            events=control.events,
        )

    def _force_final_answer(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        policy: BackendPolicy,
        spent_usd: float,
        *,
        budget_usd: float | None,
        response_schema: Any = None,
        nudge: str | None = None,
        max_response_tokens: int | None = None,
        control: _InvocationControl,
    ) -> tuple[_ForcedFinalOutcome | None, ClassifiedError | None]:
        """One final completion with NO tools after the loop maxes out.

        Returns one response/evidence snapshot and no transport error on
        success. The caller must reject ``model_identity_error`` before
        admitting the returned text.
        A budget skip returns ``(None, None)``; a classified transport
        failure returns ``(None, error)`` after Crucible's retry budget is
        exhausted.  Appends an explicit user nudge so the model knows the
        investigation phase is over and it must answer now.
        """
        if budget_usd is not None:
            if spent_usd >= budget_usd:
                return None, None
        final_messages = messages + [{
            "role": "user",
            "content": nudge or (
                "Turn budget exhausted — no more tool calls are "
                "available. Based on everything you have already "
                "investigated, produce your final answer NOW in the "
                "exact format the system prompt requires. Do not ask "
                "for more information; commit to your best answer."
            ),
        }]
        response = None
        retried = False
        for attempt in range(1, max(1, policy.retry_budget) + 1):
            stopped = control.stopped()
            if stopped is not None:
                error, _stop_reason = self._control_error(stopped, policy)
                return None, error
            control.record(
                "request",
                name=f"forced_finalization_attempt_{attempt}",
            )
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=final_messages,
                    max_tokens=(
                        self._max_tokens
                        if max_response_tokens is None
                        else min(self._max_tokens, max_response_tokens)
                    ),
                    timeout=control.request_timeout(),
                    **self._reasoning_kwargs(),
                    **self._structured_output_kwargs(response_schema),
                )
                break
            except BaseException as exc:  # noqa: BLE001
                stopped = control.stopped()
                if stopped is not None:
                    error, _stop_reason = self._control_error(stopped, policy)
                    return None, error
                err = self._classify(exc)
                if err is None:
                    # A programming or unknown SDK-contract fault must not be
                    # made indistinguishable from the no-request budget skip.
                    # Propagate it through invoke()'s normal unclassified-error
                    # boundary instead of returning a plausibly complete
                    # lower-bound accounting result.
                    raise
                if err.retriable and attempt < policy.retry_budget:
                    backoff = min(2 ** (attempt - 1), 16) + (
                        time.monotonic() % 1.0
                    )
                    logger.info(
                        "OpenAIBackend transient error on no-tools "
                        "finalization attempt %d (%s); retrying in %.1fs",
                        attempt,
                        type(exc).__name__,
                        backoff,
                    )
                    control.record(
                        "retry",
                        name=getattr(err, "kind", type(exc).__name__),
                    )
                    # As in the tool loop: the provider may have billed the
                    # discarded attempt, and its usage never arrived.
                    retried = True
                    stopped = control.wait(backoff)
                    if stopped is not None:
                        error, _stop_reason = self._control_error(stopped, policy)
                        return None, error
                    continue
                logger.warning(
                    "OpenAIBackend: forced finalization turn faulted: %s", exc,
                )
                return None, err
        assert response is not None
        fin_tok, fout_tok = _usage_tokens(response)
        served_model, model_identity_error = self._served_model_identity(
            response
        )
        provider_request_id = _provider_request_id(response)
        usage_complete = _usage_is_complete(response)
        control.record(
            "usage",
            name="complete" if usage_complete else "incomplete",
        )
        evidence = _ForcedFinalOutcome(
            text="",
            input_tokens=fin_tok,
            output_tokens=fout_tok,
            served_model=served_model,
            provider_request_id=provider_request_id,
            model_identity_error=model_identity_error,
            usage_complete=usage_complete,
            retried=retried,
        )
        stopped = control.stopped()
        if stopped is not None:
            # Retain the completed physical response's accounting and route
            # identity, but do not extract or admit content after the absolute
            # invocation deadline/cancellation boundary.
            error, _stop_reason = self._control_error(stopped, policy)
            return evidence, error
        if model_identity_error is not None:
            # Exact-model routes must reject before even extracting a response
            # body that could otherwise flow into a transcript or final answer.
            return evidence, None
        choice = response.choices[0]
        return _ForcedFinalOutcome(
            text=_message_text(choice.message),
            input_tokens=fin_tok,
            output_tokens=fout_tok,
            served_model=served_model,
            provider_request_id=provider_request_id,
            model_identity_error=model_identity_error,
            usage_complete=usage_complete,
            retried=retried,
        ), None

    def _usage(
        self,
        in_tokens: int,
        out_tokens: int,
        *,
        cost_usd: float | None = None,
    ) -> Usage:
        """Build a Usage with cost from the pricing table.

        Every model has an explicit positive or zero-rate pricing entry.
        Missing entries fail before transport, so a typo cannot disable the
        workflow's budget gate. Multi-response callers pass the sum of costs
        computed independently for each provider response so nonlinear price
        tiers are not selected from aggregate invocation tokens.
        """
        return Usage(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=(
                compute_cost(
                    self.model,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
                if cost_usd is None
                else cost_usd
            ),
        )

    def _result_with_evidence(
        self,
        *,
        final_message: str,
        transcript: tuple[Message, ...],
        usage: Usage,
        stop_reason: str,
        error: Any = None,
        response: Any = None,
        usage_complete: bool | None = None,
        trace_complete: bool = True,
        pricing_segments: tuple[UsagePricingSegment, ...] = (),
        served_model: str | None = None,
        provider_request_id: str | None = None,
        tool_authority_digest: str | None = None,
        events: list[BackendExecutionEvent] | None = None,
    ) -> InvocationResult:
        """Build one normalized terminal result without raw reasoning."""

        if usage_complete is None:
            usage_complete = _usage_is_complete(response)
        if served_model is None and response is not None:
            observed_model = getattr(response, "model", None)
            if isinstance(observed_model, str) and observed_model.strip():
                served_model = observed_model.strip()
        if provider_request_id is None and response is not None:
            observed_id = getattr(response, "id", None)
            if isinstance(observed_id, str) and observed_id.strip():
                provider_request_id = observed_id.strip()

        trace_events = (
            list(events)
            if events is not None
            else [BackendExecutionEvent(sequence=0, kind="request")]
        )
        if events is None and usage_complete:
            trace_events.append(
                BackendExecutionEvent(sequence=len(trace_events), kind="usage")
            )
        trace_events.append(
            BackendExecutionEvent(
                sequence=len(trace_events),
                kind="terminal",
                name=stop_reason,
            )
        )
        endpoint_identity = _endpoint_identity(self._base_url)
        from .features import feature_profile_digest

        return InvocationResult(
            final_message=final_message,
            transcript=transcript,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            completeness=ExecutionCompleteness(
                token_usage_complete=usage_complete,
                cost_complete=usage_complete,
                terminal_complete=True,
                trace_complete=trace_complete,
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                served_model=served_model,
                requested_effort=self._reasoning_effort,
                # OpenAI-compatible responses do not standardize an applied
                # reasoning-effort field. Do not infer it from the request.
                served_effort=None,
                endpoint=endpoint_identity,
                provider_request_id=provider_request_id,
                feature_profile_digest=feature_profile_digest(
                    self.features
                ),
                tool_authority_digest=tool_authority_digest,
                events=tuple(trace_events),
            ),
            # Complete exactly where the accounting flags are: every response
            # of the call reported its usage and has its segment.
            pricing_evidence=UsagePricingEvidence(
                segments=tuple(pricing_segments),
                complete=usage_complete,
            ),
        )

    def _classify(
        self, exc: BaseException,
    ) -> TransportError | ContentRejected | ExternalCancellation | None:
        cls_name = type(exc).__name__
        msg = str(exc)
        # Match the SDK base behavior as well as concrete class names.  The
        # OpenAI SDK raises subclasses such as ``InternalServerError`` rather
        # than an object whose exact class is ``APIStatusError``.
        mro_names = {base.__name__ for base in type(exc).__mro__}
        if (
            isinstance(exc, (ConnectionError, TimeoutError, OSError))
            or mro_names.intersection({"APIConnectionError", "APITimeoutError"})
        ):
            return TransportError(
                http_status=None,
                message=f"{cls_name}: {msg}",
                retriable=True,
            )
        status = _exception_http_status(exc)
        if status is not None or mro_names.intersection({
            "APIStatusError", "BadRequestError", "AuthenticationError",
        }):
            status = status or 0
            return TransportError(
                http_status=status,
                message=f"{cls_name}: {msg}",
                retriable=status in RETRIABLE_HTTP_STATUSES,
            )
        if "refusal" in msg.lower():
            return ContentRejected(message=msg, retriable=False)
        return None


def _exception_http_status(exc: BaseException) -> int | None:
    """Extract a real HTTP status from SDK or response-shaped exceptions."""

    candidates = [getattr(exc, "status_code", None)]
    response = getattr(exc, "response", None)
    if response is not None:
        candidates.append(getattr(response, "status_code", None))
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        try:
            status = int(candidate)
        except (TypeError, ValueError):
            continue
        if 100 <= status <= 599:
            return status
    return None


def _usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, Mapping):
        raw_in = usage.get("prompt_tokens")
        raw_out = usage.get("completion_tokens")
    else:
        raw_in = getattr(usage, "prompt_tokens", None)
        raw_out = getattr(usage, "completion_tokens", None)
    return (
        raw_in if type(raw_in) is int and raw_in >= 0 else 0,
        raw_out if type(raw_out) is int and raw_out >= 0 else 0,
    )


def _usage_is_complete(response: Any) -> bool:
    """Return whether both provider token counters were actually present."""

    if response is None:
        return False
    usage = getattr(response, "usage", None)
    if usage is None:
        return False
    if isinstance(usage, Mapping):
        raw_in = usage.get("prompt_tokens")
        raw_out = usage.get("completion_tokens")
    else:
        raw_in = getattr(usage, "prompt_tokens", None)
        raw_out = getattr(usage, "completion_tokens", None)
    return (
        type(raw_in) is int
        and raw_in >= 0
        and type(raw_out) is int
        and raw_out >= 0
    )


def _pricing_segments(
    usage_complete: bool,
    *,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> tuple[UsagePricingSegment, ...]:
    """Return the pricing segment of one provider response, if it has one.

    Only a response that reported both token counters is pricing evidence.
    Its cost is what the price row of the route gives for those counters; a
    zero-rate row prices it at zero. This adapter reads the two token
    counters of a reply and no cache counter, so a segment records none:
    input that a provider reports as read from its cache is priced, here as
    in the usage of the result, at the input rate of the row and not at its
    cache rate.
    """

    if not usage_complete:
        return ()
    return (
        UsagePricingSegment(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            cost_usd=cost_usd,
        ),
    )


def _message_text(message: Any) -> str:
    """Return a completion's visible text; reasoning is never an answer."""

    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _assistant_tool_call_message(msg: Any, tool_calls: Any) -> dict[str, Any]:
    """Render the assistant's tool-call turn as an OpenAI message dict.

    The follow-up ``role=tool`` messages must reference the
    ``tool_call_id``s from this turn, so we echo the assistant message
    (content + tool_calls) back into the running list. Reasoning-capable
    providers also require their opaque ``reasoning_content`` to be replayed
    on the assistant turn before tool results (observed with Kimi K2.6 on
    the NVIDIA hub; Moonshot documents the same rule for the K3 series).
    Keep it in the provider conversation without exposing it in Crucible's
    user-visible transcript. Assistant turns without tool calls go back
    through ``_assistant_text_message``, which carries it the same way.
    """
    message = {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in tool_calls
        ],
    }
    reasoning_content = _message_reasoning_content(msg)
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return message


def _assistant_text_message(msg: Any, text: str) -> dict[str, Any]:
    """Render an assistant turn without tool calls for provider replay.

    Moonshot (K3 series) and DeepSeek document that the complete assistant
    message, including ``reasoning_content``, is sent back on the next
    request for turns without a tool call too.  The key is omitted when the
    provider sent no reasoning.  The reasoning stays in the provider
    conversation and never enters Crucible's transcript.
    """
    message: dict[str, Any] = {"role": "assistant", "content": text}
    reasoning_content = _message_reasoning_content(msg)
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return message


def _message_reasoning_content(message: Any) -> str | None:
    """Read opaque reasoning needed only for provider conversation replay."""

    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str):
        return reasoning_content

    provider_fields = getattr(message, "provider_specific_fields", None)
    if isinstance(provider_fields, Mapping):
        reasoning_content = provider_fields.get("reasoning_content")
    elif provider_fields is not None:
        reasoning_content = getattr(provider_fields, "reasoning_content", None)
    return reasoning_content if isinstance(reasoning_content, str) else None
