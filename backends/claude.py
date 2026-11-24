# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""ClaudeBackend — assay.Backend implementation over claude-agent-sdk.

Exposes a single class that takes an ``assay.Invocation`` and returns
an ``assay.InvocationResult``, with the failure-mode handling assay
§6.5 requires (B31 max-turns, B32 cancellation, transport retry,
timeout).

The agent loop (tool use, multi-turn) is owned by the SDK; this
adapter provides translation, configuration isolation, capability policy, and
failure handling around ``query()``.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import os
import shutil
import threading
import time
import traceback as _tb
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace as _replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import urlsplit

import anyio
from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendPolicy,
    BudgetExhausted,
    ContentRejected,
    ConversationTransportOutcome,
    ExecutionCompleteness,
    ExternalCancellation,
    IncompleteEvidence,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    Message,
    StaleConversation,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)

from ._classify import (
    RETRIABLE_HTTP_STATUSES,
    TERMINAL_EXCEPTION_NAMES,
    ClassifiedError,
)
from ._sdk_debug import SdkDebugSink, redact_sdk_lines, redact_sdk_text

logger = logging.getLogger(__name__)

# Adapter-local default for callers that explicitly construct a Claude
# backend without a model. The neutral INFER_* environment redirects this
# adapter to the organization gateway, whose catalog exposes the fully
# qualified route below rather than the public first-party alias. Direct
# Anthropic callers can still select another model explicitly. The global
# routing default remains Codex/Astra/Max.
_DEFAULT_MODEL = "aws/anthropic/bedrock-claude-fable-5-1"

# Closed credential authority used by the first-party organization-session
# route.  The endpoint is a logical transport identity: Claude Code owns the
# provider HTTP endpoint and authenticates from its state under HOME.
CLAUDE_SESSION_CREDENTIAL_SOURCE = "claude_session"
CLAUDE_SESSION_ENDPOINT = "claude-cli://organization-session"
CLAUDE_SESSION_MODEL = "claude-fable-5-1"
CLAUDE_SESSION_MODELS = frozenset({
    CLAUDE_SESSION_MODEL,
    "claude-mythos-5-1",
    "claude-opus-5",
})
CLAUDE_NVIDIA_ORG_ID = "76f2dd92-7d56-4550-afaf-abd363e3235c"
CLAUDE_NVIDIA_ORG_NAME = "NVIDIA Corporation - Power Users"
_CLAUDE_CREDENTIAL_SOURCES = frozenset({
    "environment",
    CLAUDE_SESSION_CREDENTIAL_SOURCE,
})

# Closed provider terminal spelling documented by the pinned Agent SDK.  Do
# not infer budget exhaustion from arbitrary model/error prose: the terminal
# subtype is the provider-owned discriminator.
_CLAUDE_MAX_BUDGET_RESULT_SUBTYPES = frozenset({
    "error_max_budget_usd",
})

# Closed provider terminal spelling named by the pinned Agent SDK for a session
# that used up ``max_turns``.  Claude Code reports it as an error terminal
# whose ``stop_reason`` is the last API stop reason, then exits non-zero, so
# the subtype is the only provider-owned statement of the turn limit.
_CLAUDE_MAX_TURNS_RESULT_SUBTYPES = frozenset({
    "error_max_turns",
})

# The pinned Agent SDK exposes these closed Claude Code API-error categories
# on ``AssistantMessage.error``. Claude Code uses ``model="<synthetic>"`` for
# that local, non-model envelope before emitting the authoritative
# ``ResultMessage``. Only this exact category+placeholder pair may defer the
# normal immediate served-model tripwire.
_CLAUDE_AGENT_API_ERROR_CATEGORIES = frozenset({
    "authentication_failed",
    "billing_error",
    "rate_limit",
    "invalid_request",
    "server_error",
    "unknown",
})
_CLAUDE_AGENT_API_ERROR_MODEL = "<synthetic>"

# Claude Code can add locally typed API-error categories before the pinned
# Agent SDK updates its narrower ``AssistantMessage.error`` Literal. Keep the
# SDK's reviewed six-category set unchanged and admit an extension only when
# the organization-session transport proves the exact model/CLI pair whose
# wire schema was reviewed. Unknown models, versions, and categories remain
# fail-closed.
_CLAUDE_SESSION_API_ERROR_CATEGORY_EXTENSIONS = MappingProxyType({
    ("claude-mythos-5-1", "2.1.263"): frozenset({"model_not_found"}),
})

# Provider and organization routes use different names for the same Claude
# model.  Only these documented spellings are aliases; arbitrary suffix
# matching would let an untrusted gateway label evade an exact route check.
_CLAUDE_MODEL_ALIAS_PREFIXES = (
    "aws/anthropic/bedrock-",
    "azure/anthropic/",
    "gateway/",
    "anthropic.",
)
_CLAUDE_TERMINAL_METADATA_MAX_CHARS = 256


def _adaptive_thinking() -> dict[str, str]:
    """Return a fresh adaptive-thinking request for one provider call."""
    return {"type": "adaptive"}


def _json_output_format(schema: Any) -> dict[str, Any]:
    """Translate and copy a vendor-neutral schema for Claude APIs."""

    return {"type": "json_schema", "schema": deepcopy(schema)}


def _valid_token_counter(value: object) -> bool:
    return type(value) is int and value >= 0


def _valid_cost(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _valid_http_status(value: object) -> bool:
    return type(value) is int and 100 <= value <= 599


def _claude_agent_api_error_category(
    message: object,
    *,
    extensions: frozenset[str] = frozenset(),
) -> str | None:
    """Return one exact SDK API-error category when present."""

    category = getattr(message, "error", None)
    if (
        type(category) is str
        and (
            category in _CLAUDE_AGENT_API_ERROR_CATEGORIES
            or category in extensions
        )
    ):
        return category
    return None


def _synthetic_claude_api_error_category(
    message: object,
    *,
    extensions: frozenset[str] = frozenset(),
) -> str | None:
    """Return the category only for the CLI's exact local error envelope."""

    category = _claude_agent_api_error_category(
        message,
        extensions=extensions,
    )
    model = getattr(message, "model", None)
    if (
        category is not None
        and type(model) is str
        and model == _CLAUDE_AGENT_API_ERROR_MODEL
    ):
        return category
    return None


def _claude_session_api_error_category_extensions(
    *,
    requested_model: object,
    validated_cli_version: object,
) -> frozenset[str]:
    """Return only the extension reviewed for one exact session runtime."""

    if type(requested_model) is not str:
        return frozenset()
    if type(validated_cli_version) is not str:
        return frozenset()
    return _CLAUDE_SESSION_API_ERROR_CATEGORY_EXTENSIONS.get(
        (requested_model, validated_cli_version),
        frozenset(),
    )


def _valid_synthetic_api_error_model_usage(value: object) -> bool:
    """Accept only terminal model-usage shapes that cannot name a real model."""

    if value is None:
        return True
    if type(value) is not dict:
        return False
    if not value:
        return True
    if len(value) != 1:
        return False
    model, details = next(iter(value.items()))
    return (
        type(model) is str
        and model == _CLAUDE_AGENT_API_ERROR_MODEL
        and type(details) is dict
    )


def _usage_value(usage: object, name: str) -> object:
    if isinstance(usage, dict):
        return usage.get(name)
    return getattr(usage, name, None)


def _validated_timeout_seconds(policy: BackendPolicy) -> float | None:
    value = policy.timeout_seconds
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(
            "BackendPolicy.timeout_seconds must be None or a finite number > 0"
        )
    return float(value)


def _conversation_outcome(
    invocation: Invocation,
    *,
    opaque_id: str | None,
    state: str = "active",
) -> ConversationTransportOutcome | None:
    """Bind provider continuity to the exact admitted logical operation."""

    request = invocation.conversation
    if request is None:
        return None
    if state == "active" and opaque_id is None:
        return None
    return ConversationTransportOutcome(
        state=state,  # type: ignore[arg-type]
        opaque_id=opaque_id,
        identity_fingerprint=request.identity_fingerprint,
        operation_digest=request.operation_digest,
    )


_CLI_STDERR_MAX_ENTRIES = 200
_CLI_STDERR_MAX_BYTES = 64 * 1024
_RESUME_REJECTED_EXACT = 1
_RESUME_REJECTED_SESSION_ID = 2
_RESUME_REJECTED_LOOSE = 4


def _resume_rejection_marks(text: str) -> int:
    """Name the rejected-resume markers that one piece of evidence holds."""

    lowered = text.lower()
    marks = 0
    if any(
        marker in lowered
        for marker in (
            "no conversation found",
            "conversation not found",
            "no session found",
            "session not found",
        )
    ):
        marks |= _RESUME_REJECTED_EXACT
    if "session id" in lowered:
        marks |= _RESUME_REJECTED_SESSION_ID
    if any(
        marker in lowered
        for marker in ("does not exist", "not found", "unknown", "invalid")
    ):
        marks |= _RESUME_REJECTED_LOOSE
    return marks


def _claude_resume_was_rejected(
    exc: BaseException,
    stderr_lines: list[str],
    stderr_marks: int,
) -> bool:
    """Recognize a missing/rejected session without treating outage as stale."""

    leaf = _unwrap_exception_group(exc)
    marks = stderr_marks | _resume_rejection_marks(
        "\n".join([str(leaf), *stderr_lines])
    )
    if marks & _RESUME_REJECTED_EXACT:
        return True
    loose = _RESUME_REJECTED_SESSION_ID | _RESUME_REJECTED_LOOSE
    return marks & loose == loose


class _CliStderrWindow:
    """The end of one attempt's CLI stderr, bounded in entries and in bytes.

    ``entries`` holds the last entries that fit both bounds, ``notes`` flags
    those that are the window's own size note for an entry over the byte
    bound, and ``received`` counts every entry handed over.
    """

    def __init__(self, *, note_resume_markers: bool) -> None:
        self.entries: list[str] = []
        self.received = 0
        self._entry_bytes = 0
        self._sizes: list[int] = []
        self._marks: list[int] = []
        self.notes: list[bool] = []
        self._note_resume_markers = note_resume_markers

    def add(self, line: str) -> None:
        """Keep one entry. Called by the SDK; never raises for a ``str``."""

        self.received += 1
        if self._note_resume_markers:
            if len(self._marks) >= _CLI_STDERR_MAX_ENTRIES:
                del self._marks[0]
            self._marks.append(_resume_rejection_marks(line))
        size = (
            len(line)
            if line.isascii()
            else len(line.encode("utf-8", "surrogatepass"))
        )
        noted = size > _CLI_STDERR_MAX_BYTES
        if noted:
            line = f"[crucible: one stderr line of {size} bytes not kept]"
            size = len(line)
        self.entries.append(line)
        self._sizes.append(size)
        self.notes.append(noted)
        self._entry_bytes += size
        while len(self.entries) > 1 and (
            len(self.entries) > _CLI_STDERR_MAX_ENTRIES
            or self.kept_bytes > _CLI_STDERR_MAX_BYTES
        ):
            self._entry_bytes -= self._sizes.pop(0)
            del self.entries[0]
            del self.notes[0]

    @property
    def kept_bytes(self) -> int:
        """UTF-8 size of the kept entries joined by line breaks."""

        return self._entry_bytes + max(len(self.entries) - 1, 0)

    def resume_marks(self) -> int:
        """Markers of the last 200 entries received, kept or not."""

        marks = 0
        for entry_marks in self._marks:
            marks |= entry_marks
        return marks


def _usage_sum(left: Usage, right: Usage) -> Usage:
    """Add usage observed on two physical attempts."""

    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cost_usd=left.cost_usd + right.cost_usd,
        cache_creation_tokens=(
            left.cache_creation_tokens + right.cache_creation_tokens
        ),
        cache_read_tokens=left.cache_read_tokens + right.cache_read_tokens,
    )


def _usage_conservative_envelope(
    current: Usage,
    observed: Usage | None,
) -> Usage:
    """Retain a valid conservative envelope for conflicting usage views."""

    if observed is None:
        return current
    cache_creation_tokens = max(
        current.cache_creation_tokens,
        observed.cache_creation_tokens,
    )
    cache_read_tokens = max(
        current.cache_read_tokens,
        observed.cache_read_tokens,
    )
    ordinary_input_tokens = max(
        current.input_tokens
        - current.cache_creation_tokens
        - current.cache_read_tokens,
        observed.input_tokens
        - observed.cache_creation_tokens
        - observed.cache_read_tokens,
    )
    return Usage(
        input_tokens=(
            ordinary_input_tokens
            + cache_creation_tokens
            + cache_read_tokens
        ),
        output_tokens=max(current.output_tokens, observed.output_tokens),
        cost_usd=max(current.cost_usd, observed.cost_usd),
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )


def _pricing_segment_from_usage(
    usage: object,
    *,
    model: str,
) -> UsagePricingSegment | None:
    """Parse one exact Claude response usage record into pricing inputs."""

    counters = {
        name: _usage_value(usage, name)
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }
    if not all(_valid_token_counter(value) for value in counters.values()):
        return None
    ordinary_input = counters["input_tokens"]
    output_tokens = counters["output_tokens"]
    cache_creation_tokens = counters["cache_creation_input_tokens"]
    cache_read_tokens = counters["cache_read_input_tokens"]
    assert isinstance(ordinary_input, int)
    assert isinstance(output_tokens, int)
    assert isinstance(cache_creation_tokens, int)
    assert isinstance(cache_read_tokens, int)
    total_input = (
        ordinary_input + cache_creation_tokens + cache_read_tokens
    )
    from .pricing import compute_cost

    return UsagePricingSegment(
        input_tokens=total_input,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        cost_usd=compute_cost(
            model,
            input_tokens=total_input,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        ),
    )


def _terminal_model_usage_pricing_segment(
    model_usage: object,
) -> UsagePricingSegment | None:
    """Strictly parse the pinned CLI's one-model terminal cost envelope."""

    if not isinstance(model_usage, Mapping):
        return None
    try:
        entries = tuple(model_usage.items())
    except Exception:  # noqa: BLE001 - untrusted terminal mapping
        return None
    if len(entries) != 1:
        return None
    model_key, raw_envelope = entries[0]
    if not _valid_terminal_metadata_string(model_key):
        return None
    if not isinstance(raw_envelope, Mapping):
        return None
    token_fields = {
        name: raw_envelope.get(name)
        for name in (
            "inputTokens",
            "outputTokens",
            "cacheCreationInputTokens",
            "cacheReadInputTokens",
            "webSearchRequests",
            "contextWindow",
            "maxOutputTokens",
        )
    }
    if not all(
        _valid_token_counter(value) for value in token_fields.values()
    ):
        return None
    if (
        token_fields["contextWindow"] == 0
        or token_fields["maxOutputTokens"] == 0
    ):
        return None
    canonical_model = raw_envelope.get("canonicalModel")
    provider = raw_envelope.get("provider")
    if (
        not _valid_terminal_metadata_string(canonical_model)
        or not _valid_terminal_metadata_string(provider)
    ):
        return None
    assert isinstance(model_key, str)
    assert isinstance(canonical_model, str)
    if not claude_model_identities_match(model_key, canonical_model):
        return None
    raw_cost = raw_envelope.get("costUSD")
    if not _valid_cost(raw_cost):
        return None
    # The local pricing table accounts only for tokens. A nonzero server-tool
    # charge cannot be reconstructed from the token fields and therefore
    # cannot support complete pricing evidence.
    if token_fields["webSearchRequests"] != 0:
        return None
    ordinary_input = token_fields["inputTokens"]
    output_tokens = token_fields["outputTokens"]
    cache_creation_tokens = token_fields["cacheCreationInputTokens"]
    cache_read_tokens = token_fields["cacheReadInputTokens"]
    assert isinstance(ordinary_input, int)
    assert isinstance(output_tokens, int)
    assert isinstance(cache_creation_tokens, int)
    assert isinstance(cache_read_tokens, int)
    return UsagePricingSegment(
        input_tokens=(
            ordinary_input + cache_creation_tokens + cache_read_tokens
        ),
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        cost_usd=float(raw_cost),
    )


def _valid_terminal_metadata_string(value: object) -> bool:
    """Validate bounded, exact CLI pricing-envelope identity metadata."""

    return (
        type(value) is str
        and 0 < len(value) <= _CLAUDE_TERMINAL_METADATA_MAX_CHARS
        and value == value.strip()
        and value.isprintable()
    )


def _pricing_segments_reconcile(
    segments: tuple[UsagePricingSegment, ...],
    terminal: UsagePricingSegment,
) -> bool:
    """Return whether visible physical responses equal a terminal total."""

    aggregate = UsagePricingEvidence(
        segments=segments,
        complete=False,
    ).usage
    return (
        aggregate.input_tokens == terminal.input_tokens
        and aggregate.output_tokens == terminal.output_tokens
        and aggregate.cache_creation_tokens
        == terminal.cache_creation_tokens
        and aggregate.cache_read_tokens == terminal.cache_read_tokens
        and math.isclose(
            aggregate.cost_usd,
            terminal.cost_usd,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )


def _pricing_fragment_within_terminal(
    fragment: UsagePricingSegment,
    terminal: UsagePricingSegment,
) -> bool:
    """Return whether one raw terminal fragment fits its aggregate.

    ``ResultMessage.usage`` on an exact max-budget terminal can describe only
    the last response fragment while ``model_usage`` describes the fresh
    invocation aggregate.  Every token dimension of a genuine fragment must
    therefore be contained by that aggregate.  Cost is deliberately excluded:
    the ResultMessage cost field is itself the aggregate, not fragment cost.
    """

    fragment_ordinary_input = (
        fragment.input_tokens
        - fragment.cache_creation_tokens
        - fragment.cache_read_tokens
    )
    terminal_ordinary_input = (
        terminal.input_tokens
        - terminal.cache_creation_tokens
        - terminal.cache_read_tokens
    )
    return (
        fragment_ordinary_input <= terminal_ordinary_input
        and fragment.output_tokens <= terminal.output_tokens
        and fragment.cache_creation_tokens
        <= terminal.cache_creation_tokens
        and fragment.cache_read_tokens <= terminal.cache_read_tokens
    )


def _combine_pricing_evidence(
    usage: Usage,
    *records: UsagePricingEvidence | None,
) -> UsagePricingEvidence | None:
    """Combine ordered physical evidence without promoting a mismatch."""

    present = tuple(record for record in records if record is not None)
    if not present:
        return None
    combined = UsagePricingEvidence(
        segments=tuple(
            segment
            for record in present
            for segment in record.segments
        ),
        complete=False,
    )
    return UsagePricingEvidence(
        segments=combined.segments,
        complete=(
            len(present) == len(records)
            and all(record.complete for record in present)
            and combined.reconciles(usage)
        ),
    )


def _usage_observed(result: InvocationResult) -> bool:
    return (
        result.completeness.token_usage_complete
        or result.completeness.cost_complete
        or result.usage.input_tokens != 0
        or result.usage.output_tokens != 0
        or result.usage.cache_creation_tokens != 0
        or result.usage.cache_read_tokens != 0
        or result.usage.cost_usd != 0
    )


def _append_event(
    events: list[BackendExecutionEvent],
    kind: str,
    *,
    name: str | None = None,
    parent_id: str | None = None,
    child_id: str | None = None,
) -> None:
    events.append(
        BackendExecutionEvent(
            sequence=len(events),
            kind=kind,
            name=name,
            parent_id=parent_id,
            child_id=child_id,
        )
    )


def _merge_nonterminal_events(
    destination: list[BackendExecutionEvent],
    result: InvocationResult,
    *,
    request_name: str,
) -> None:
    """Merge one physical-attempt trace without its wrapper terminal."""

    source = (
        result.execution_trace.events
        if result.execution_trace is not None
        else ()
    )
    nonterminal = [event for event in source if event.kind != "terminal"]
    if not any(event.kind == "request" for event in nonterminal):
        _append_event(destination, "request", name=request_name)
    for event in nonterminal:
        _append_event(
            destination,
            event.kind,
            name=event.name,
            parent_id=event.parent_id,
            child_id=event.child_id,
        )
    if (
        not any(event.kind == "usage" for event in nonterminal)
        and _usage_observed(result)
    ):
        _append_event(
            destination,
            "usage",
            name=(
                "complete"
                if (
                    result.completeness.token_usage_complete
                    and result.completeness.cost_complete
                )
                else "partial"
            ),
        )


def _validated_credential_source(value: object) -> str:
    if type(value) is not str or value not in _CLAUDE_CREDENTIAL_SOURCES:
        raise ValueError(
            "Claude credential_source must be one of "
            f"{sorted(_CLAUDE_CREDENTIAL_SOURCES)}"
        )
    return value


def _effective_base_url(
    *,
    prefer_infer: bool = False,
    credential_source: str = "environment",
) -> str | None:
    """Resolve the runtime endpoint without letting ambient vars win a route."""

    source = _validated_credential_source(credential_source)
    if source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
        return None
    if prefer_infer:
        value = os.environ.get("INFER_BASE_URL", "").strip()
    else:
        value = (
            os.environ.get("ANTHROPIC_BASE_URL", "").strip()
            or os.environ.get("INFER_BASE_URL", "").strip()
        )
    return value or None


def _endpoint_identity(base_url: str | None) -> str | None:
    """Return a credential- and query-free endpoint identity."""

    if base_url is None:
        return None
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if parsed.scheme not in {"http", "https"} or not hostname:
        return None
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = rendered_host if port is None else f"{rendered_host}:{port}"
    return f"{parsed.scheme}://{netloc}{parsed.path}".rstrip("/")


def _validated_endpoint_identity(
    base_url: str | None,
    *,
    setting: str,
) -> str | None:
    if base_url is None:
        return None
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting} must be a valid http(s) endpoint") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(
            character.isspace()
            or ord(character) < 0x20
            or ord(character) == 0x7F
            for character in base_url
        )
        or (
            parsed.scheme == "http"
            and (
                parsed.hostname != "127.0.0.1"
                or port is None
                or parsed.netloc != f"127.0.0.1:{port}"
            )
        )
    ):
        raise ValueError(
            f"{setting} must be an HTTPS endpoint without embedded "
            "credentials, whitespace, query parameters, or fragments; "
            "plain HTTP is allowed only for exact "
            "http://127.0.0.1:<port> tests"
        )
    identity = _endpoint_identity(base_url)
    if identity is None:  # pragma: no cover - guarded above
        raise ValueError(f"{setting} must be a valid http(s) endpoint")
    return identity


def _record_served_model(
    requested_model: str,
    served_model: object,
    *,
    transport: str,
    observed: set[str],
    exact: bool = False,
) -> str | None:
    """Log each provider-reported model identity once per invocation.

    ``InvocationResult`` currently has no provider metadata field and the
    shared SDK debug serializer has no extension surface. Keep the identity
    observable without mutating transcript semantics; a future shared trace
    field can consume the same requested/served pair.
    """
    if (
        type(served_model) is not str
        or not served_model
        or served_model != served_model.strip()
    ):
        return None
    served = served_model
    if served in observed:
        return served
    observed.add(served)
    log = (
        logger.info
        if (
            requested_model == served
            if exact
            else claude_model_identities_match(requested_model, served)
        )
        else logger.warning
    )
    log(
        "ClaudeBackend %s model identity: requested=%s served=%s",
        transport,
        requested_model,
        served,
    )
    return served


def _canonical_claude_model_identity(model: str) -> str:
    """Return the provider-neutral identity for a known Claude alias.

    Unknown identifiers remain exact-match-only.  In particular, this
    deliberately does not search for a ``claude-*`` suffix inside an
    arbitrary route string.
    """

    candidate = model.strip()
    if candidate.startswith("claude-"):
        return candidate
    for prefix in _CLAUDE_MODEL_ALIAS_PREFIXES:
        if candidate.startswith(prefix):
            unwrapped = candidate[len(prefix):]
            if unwrapped.startswith("claude-"):
                return unwrapped
    return candidate


def claude_model_identities_match(
    requested_model: str,
    served_model: str,
) -> bool:
    """Whether two exact or documented-alias Claude identities agree."""

    return (
        _canonical_claude_model_identity(requested_model)
        == _canonical_claude_model_identity(served_model)
    )


def _served_model_mismatch(
    requested_model: str,
    served_model: str | None,
    *,
    transport: str,
    exact: bool = False,
) -> IncompleteEvidence | None:
    """Build a typed, non-retriable route failure for observed drift."""

    if served_model is None or (
        requested_model == served_model
        if exact
        else claude_model_identities_match(
            requested_model,
            served_model,
        )
    ):
        return None
    logger.error(
        "ClaudeBackend %s refused served-model drift: requested=%s served=%s",
        transport,
        requested_model,
        served_model,
    )
    return IncompleteEvidence(
        missing=("execution_trace.served_model_matches_requested",),
        message=(
            "Claude provider served a different model than requested: "
            f"requested={requested_model!r}, served={served_model!r}"
        ),
        retriable=False,
    )


def _result_model_usage_identity(
    requested_model: str,
    model_usage: object,
    *,
    observed: set[str],
    exact: bool = False,
) -> tuple[str | None, IncompleteEvidence | None]:
    """Validate the Agent SDK's terminal model-usage identity.

    The terminal envelope must name exactly one model. Multiple raw keys are
    ambiguous even when they canonicalize to the same alias because the
    normalized trace has one ``served_model`` field and this confined route
    has no legitimate multi-model/fallback behavior.
    """

    if not isinstance(model_usage, Mapping):
        return None, IncompleteEvidence(
            missing=("execution_trace.served_model_matches_requested",),
            message=(
                "Claude Agent SDK ResultMessage.model_usage must be a "
                "non-empty string-keyed mapping when present"
            ),
            retriable=False,
        )
    try:
        model_keys = tuple(model_usage.keys())
    except Exception as exc:  # noqa: BLE001 - untrusted terminal shape
        return None, IncompleteEvidence(
            missing=("execution_trace.served_model_matches_requested",),
            message=(
                "Claude Agent SDK ResultMessage.model_usage keys could not "
                f"be read safely ({type(exc).__name__})"
            ),
            retriable=False,
        )
    if len(model_keys) != 1:
        qualifier = "empty" if not model_keys else "ambiguous"
        return None, IncompleteEvidence(
            missing=("execution_trace.served_model_matches_requested",),
            message=(
                "Claude Agent SDK ResultMessage.model_usage is "
                f"{qualifier}; expected exactly one served-model key"
            ),
            retriable=False,
        )
    model_key = model_keys[0]
    if (
        type(model_key) is not str
        or not model_key
        or model_key != model_key.strip()
    ):
        return None, IncompleteEvidence(
            missing=("execution_trace.served_model_matches_requested",),
            message=(
                "Claude Agent SDK ResultMessage.model_usage must have one "
                "exact non-empty string model key"
            ),
            retriable=False,
        )
    served_model = _record_served_model(
        requested_model,
        model_key,
        transport="agent_sdk.model_usage",
        observed=observed,
        exact=exact,
    )
    return served_model, _served_model_mismatch(
        requested_model,
        served_model,
        transport="agent_sdk.model_usage",
        exact=exact,
    )


def _missing_served_model_identity() -> IncompleteEvidence:
    """Typed failure for a nominal Claude response with no model proof."""

    return IncompleteEvidence(
        missing=("execution_trace.served_model",),
        message=(
            "Claude provider nominal success provided no valid served-model "
            "identity in its response, AssistantMessage.model, or "
            "ResultMessage.model_usage"
        ),
        retriable=False,
    )


def _build_sdk_env(
    *,
    prefer_infer: bool = False,
    credential_source: str = "environment",
) -> dict[str, str]:
    """Build explicit adapter variables for a Claude Agent SDK invocation.

    Map the workspace's neutral gateway vars
    (INFER_BASE_URL / INFER_AUTH_TOKEN, exported by
    ~/.config/crucible/env) onto the ANTHROPIC_* vars the bundled CLI
    reads — injected into the SUBPROCESS env only, so an interactive
    `claude` shell that merely sourced the env file is NOT redirected.

    The stock SDK merges this mapping over its inherited environment.  For
    ``claude_session``, :class:`SanitizedClaudeSessionTransport` applies this
    mapping as a real process boundary instead: only the operational allowlist
    reaches the auth probe, version probe, or model process.
    """
    safe_names = {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NODE_EXTRA_CA_CERTS",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
        "TZ",
    }
    sdk_env = {
        name: value
        for name, value in os.environ.items()
        if name in safe_names
    }
    # Claude Code can select a one-hour prompt-cache TTL through remote or
    # ambient configuration.  Crucible's reviewed pricing profile assumes
    # the five-minute write rate, so bind that TTL at the SDK child boundary
    # instead of inheriting an operator value.
    sdk_env["FORCE_PROMPT_CACHING_5M"] = "1"
    source = _validated_credential_source(credential_source)
    if source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
        # Claude Code reads its organization session beneath HOME.  Do not
        # inject any ambient provider endpoint or credential into that child:
        # sourcing ~/.config/crucible/env must not redirect this route.
        return sdk_env
    if base_url := _effective_base_url(
        prefer_infer=prefer_infer,
        credential_source=source,
    ):
        sdk_env["ANTHROPIC_BASE_URL"] = base_url
    auth_token = (
        os.environ.get("INFER_AUTH_TOKEN", "").strip()
        if prefer_infer
        else (
            os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
            or os.environ.get("INFER_AUTH_TOKEN", "").strip()
        )
    )
    if auth_token:
        sdk_env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    elif not prefer_infer and (
        api_key := os.environ.get("ANTHROPIC_API_KEY")
    ):
        sdk_env["ANTHROPIC_API_KEY"] = api_key
    return sdk_env


def _direct_client_kwargs(
    *,
    prefer_infer: bool = False,
    credential_source: str = "environment",
) -> dict[str, Any]:
    """Map the neutral gateway environment onto the direct SDK client."""

    source = _validated_credential_source(credential_source)
    if source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
        raise NotImplementedError(
            "Claude credential_source='claude_session' is available only "
            "through the Claude Agent SDK/CLI transport; the direct Messages "
            "API cannot authenticate with Claude Code session state"
        )
    kwargs: dict[str, Any] = {
        # Crucible records and bounds every retry itself. The Anthropic SDK
        # otherwise performs two invisible retries per adapter attempt.
        "max_retries": 0,
    }
    base_url = _effective_base_url(
        prefer_infer=prefer_infer,
        credential_source=source,
    )
    auth_token = (
        os.environ.get("INFER_AUTH_TOKEN", "").strip()
        if prefer_infer
        else (
            os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
            or os.environ.get("INFER_AUTH_TOKEN", "").strip()
        )
    )
    api_key = (
        ""
        if prefer_infer
        else os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )
    if base_url is not None:
        kwargs["base_url"] = base_url
    # The organization inference gateway authenticates with a bearer token.
    # Prefer that explicit route over an unrelated ambient first-party key.
    if auth_token:
        kwargs["auth_token"] = auth_token
    elif api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _system_cli_path() -> str | None:
    """Resolve the system-installed ``claude`` CLI, bypassing the SDK's bundle.

    The claude_agent_sdk wheel ships a *bundled* CLI that the SDK's
    ``_find_cli`` prefers over any system install. That bundled build
    (2.1.92) gates adaptive thinking on a model-name allow-list that
    predates newer Opus identifiers, so for the NVIDIA-gateway model group it falls
    back to the deprecated ``thinking.type=enabled`` request shape — which
    the gateway rejects with a 400 ("Use thinking.type.adaptive and
    output_config.effort"). The system CLI (2.1.152+) flipped that gate to
    a deny-list, so current Opus models correctly negotiate adaptive thinking.
    Pinning ``cli_path`` here forces the working system binary regardless
    of what the wheel bundles. Returns None if no system ``claude`` is on
    PATH, leaving the SDK to its own resolution.
    """
    return shutil.which("claude")


def _validated_session_cli_path(value: object) -> str:
    """Return one canonical executable CLI path for session authentication."""

    requirement = (
        "credential_source='claude_session' requires cli_path to resolve "
        "to an absolute existing executable Claude CLI"
    )
    if type(value) is not str or not value.strip():
        raise ValueError(requirement)
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"{requirement}; relative paths are not allowed")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{requirement}; path does not exist") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{requirement}; path is not an executable file")
    return str(resolved)


def _validated_session_org_identity(
    *,
    expected_org_id: object,
    expected_org_name: object,
) -> tuple[str, str | None]:
    """Validate an explicit, non-secret principal binding for session auth."""

    if (
        type(expected_org_id) is not str
        or not expected_org_id
        or expected_org_id != expected_org_id.strip()
    ):
        raise ValueError(
            "credential_source='claude_session' requires an exact non-empty "
            "expected_org_id"
        )
    if expected_org_name is not None and (
        type(expected_org_name) is not str
        or not expected_org_name
        or expected_org_name != expected_org_name.strip()
    ):
        raise ValueError(
            "expected_org_name must be an exact non-empty string when set"
        )
    return expected_org_id, expected_org_name


# Default policy when Invocation.policy is absent. The 45-minute deadline
# allows repository investigation, visible-test iteration, and multi-file
# changes while bounding any invocation that stops making progress.
_DEFAULT_POLICY = BackendPolicy(
    retry_budget=3,
    timeout_seconds=2700.0,  # 45 min wall — see comment above
    on_cancellation="convert_to_backend_error",
    on_max_turns="raise",
)

# Sent as the prompt when RESUMING a session after a transport drop. The
# original task + the model's prior (interrupted) progress are already in
# the resumed session, so we nudge it to continue rather than re-issuing
# the full prompt (which would duplicate it in the conversation).
_CLAUDE_RESUME_NUDGE = (
    "A previous attempt at the task below was interrupted by a transport "
    "error before it finished. Any progress is preserved in this session and "
    "any edits are already on disk. FIRST review the current state (what's "
    "already changed), THEN continue and complete the task — do not blindly "
    "restart or duplicate work you've already done.\n\n"
    "--- TASK ---\n{task}"
)


def _unwrap_exception_group(exc: BaseException) -> BaseException:
    """Drill through ``BaseExceptionGroup`` nesting to the leaf cause.

    Python 3.11's ``BaseExceptionGroup`` wraps the asyncio cancellation
    chain; we want the inner cause to classify correctly.
    """
    while isinstance(exc, BaseExceptionGroup):  # noqa: E501
        if not exc.exceptions:
            break
        exc = exc.exceptions[0]
    return exc


def _is_external_cancellation(exc: BaseException) -> bool:
    """Return True if ``exc`` is an asyncio.CancelledError NOT caused by
    a wall-timeout we ourselves set.

    Per memory project_crucible_asyncio_cancel_mid_fix_loop.md (B32):
    external cancellation must convert to a backend error so retry
    loops downstream can recover; internal wall-timeout cancellations
    should propagate as MaxTurnsExceeded/timeout instead.
    """
    inner = _unwrap_exception_group(exc)
    return isinstance(inner, asyncio.CancelledError)


class _InvocationDeadlineExceeded(TimeoutError):
    """The adapter's logical wall deadline, not a provider timeout."""


async def _await_with_deadline(
    awaitable: Any,
    timeout: float | None,
) -> Any:
    """Await without confusing a provider ``TimeoutError`` with our wall."""

    if timeout is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    try:
        done, _pending = await asyncio.wait(
            {task},
            timeout=max(0.0, timeout),
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise
    if task in done:
        return await task
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    raise _InvocationDeadlineExceeded


async def _iterate_until_tool_dispatch_denial(
    messages: Any,
    worker: Any | None,
):
    """Yield SDK messages while making a bridge budget denial terminal.

    The pinned Claude Agent SDK converts exceptions raised by an in-process
    MCP handler into JSON-RPC error responses for the bundled CLI.  It does
    not re-raise them through ``query()``.  Race message receipt against the
    bridge's invocation-scoped denial signal so the adapter cancels the
    physical query before the CLI can continue its internal tool loop.
    """

    wait_for_denial = getattr(
        worker,
        "wait_for_terminal_denial",
        None,
    )
    if not callable(wait_for_denial):
        async for message in messages:
            yield message
        return

    iterator = messages.__aiter__()
    while True:
        denial = getattr(worker, "terminal_denial", None)
        if denial is not None:
            raise denial

        message_task = asyncio.ensure_future(anext(iterator))
        denial_task = asyncio.ensure_future(wait_for_denial())
        try:
            done, _pending = await asyncio.wait(
                {message_task, denial_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if denial_task in done:
                denial = denial_task.result()
                if not message_task.done():
                    message_task.cancel()
                await asyncio.gather(message_task, return_exceptions=True)
                close = getattr(iterator, "aclose", None)
                if callable(close):
                    try:
                        await close()
                    except BaseException:  # noqa: BLE001
                        # The cancelled query iterator already performs its
                        # own teardown; an explicit close is best effort.
                        pass
                raise denial

            denial_task.cancel()
            await asyncio.gather(denial_task, return_exceptions=True)
            try:
                message = message_task.result()
            except StopAsyncIteration:
                return
        finally:
            pending = tuple(
                task
                for task in (message_task, denial_task)
                if not task.done()
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        yield message


_CLAUDE_SESSION_MODEL_ADMISSION_TIMEOUT_SECONDS = 10.0


async def _query_claude_session_with_model_preflight(
    *,
    prompt: str,
    options: Any,
    transport: Any,
):
    """Admit the exact local model before writing the first user frame."""

    from claude_agent_sdk import ClaudeSDKClient  # noqa: PLC0415
    from crucible.backends._claude_session_transport import (  # noqa: PLC0415
        CLAUDE_SESSION_FRONTIER_MODELS,
        ClaudeSessionIdentityError,
    )

    model = options.model
    if type(model) is not str or model not in CLAUDE_SESSION_FRONTIER_MODELS:
        raise ClaudeSessionIdentityError(
            "Claude organization-session admission requires an exact reviewed model"
        )

    client = ClaudeSDKClient(options=options, transport=transport)
    primary_error: BaseException | None = None
    try:
        # connect(None) starts the CLI, its message reader, and the SDK
        # initialize control exchange without writing a user message.
        await client.connect()
        try:
            with anyio.fail_after(
                _CLAUDE_SESSION_MODEL_ADMISSION_TIMEOUT_SECONDS
            ):
                await client.set_model(model)
        except Exception as exc:
            # The pinned SDK represents an explicit correlated
            # ``control_response{subtype=error}`` as exactly ``Exception``.
            # Preserve typed timeouts, process exits, and connection failures:
            # no user frame has been written, so those indeterminate transport
            # failures remain safe for the backend's bounded retry policy.
            if type(exc) is not Exception:
                raise
            raise ClaudeSessionIdentityError(
                "Claude organization-session exact model was rejected before prompt"
            ) from exc
        await client.query(prompt, session_id="")
        # Preserve query()'s complete stream/EOF behavior. The backend owns
        # ResultMessage admission and explicitly closes this generator on all
        # early returns.
        async for message in client.receive_messages():
            yield message
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            await client.disconnect()
        except BaseException:
            if primary_error is None:
                raise
            logger.debug(
                "Claude session cleanup failed after a primary error",
                exc_info=True,
            )


_USER_MESSAGE_CACHE_MARKER: dict = {"type": "ephemeral"}
_MCP_ATTACHMENT_UNSET = object()
_BUDGET_OVERRIDE_UNSET = object()


def _process_hosted_toolset(
    toolset: Any,
    *,
    invocation_deadline: float | None,
    register_host: Any | None = None,
    unregister_host: Any | None = None,
    ensure_open: Any | None = None,
) -> tuple[Any, Any | None]:
    """Qualify mutators and bind them to one reapable child process."""

    write_names = frozenset(
        str(name) for name in getattr(toolset, "write_names", ())
    )
    if not write_names:
        return toolset, None
    if invocation_deadline is None:
        raise ValueError(
            "workspace-write Claude invocations require a finite wall deadline"
        )
    from crucible.tools._isolated_mutation_host import (  # noqa: PLC0415
        IsolatedMutationHost,
        ProcessHostedToolSet,
        build_isolated_toolset_spec,
    )

    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=write_names,
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=invocation_deadline,
    )
    if register_host is not None:
        register_host(host)
    try:
        host.start()
        if ensure_open is not None:
            ensure_open()
        hosted = ProcessHostedToolSet(toolset, host)
    except BaseException:
        host.stop()
        if unregister_host is not None:
            unregister_host(host)
        raise
    return hosted, host


def _set_user_message_cache_marker(messages: list[dict]) -> None:
    """Place an Anthropic ``cache_control: ephemeral`` marker on the
    last user message's terminal content block, stripping any stale
    markers from earlier user messages.

    The Anthropic API caches everything up to and including the latest
    ``cache_control`` marker in the request. Across a multi-round
    ``chat_with_tools`` loop, refreshing the marker before every
    ``messages.create`` causes round N to read rounds 1..N-1 from
    cache and write only the new content as a fresh cache entry —
    amortizing the otherwise-quadratic input billing.

    Anthropic limits a request to 4 cache breakpoints total
    (system+tools+messages combined). We keep exactly one
    message-level marker; combined with the system block's marker we
    use 2 of 4, leaving headroom for future per-tool caching.

    Messages with non-list ``content`` (the initial user prompt
    starts as a bare string) are promoted to a one-element block
    list so cache_control has a block to attach to. The promotion
    is API-compatible: ``{"role": "user", "content": "x"}`` and
    ``{"role": "user", "content": [{"type": "text", "text": "x"}]}``
    are equivalent server-side.
    """
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_idx = i
    if last_user_idx < 0:
        return

    # Strip cache_control from every OTHER user message's content
    # blocks so we don't accumulate stale markers across rounds.
    for i, msg in enumerate(messages):
        if msg.get("role") != "user" or i == last_user_idx:
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and "cache_control" in blk:
                    blk.pop("cache_control", None)

    last_msg = messages[last_user_idx]
    content = last_msg.get("content")
    if isinstance(content, str):
        # Promote bare string to a one-element block list so we can
        # hang cache_control off the block. Equivalent to the str
        # form server-side.
        last_msg["content"] = [{
            "type": "text",
            "text": content,
            "cache_control": dict(_USER_MESSAGE_CACHE_MARKER),
        }]
        return
    if isinstance(content, list) and content:
        last_block = content[-1]
        if isinstance(last_block, dict):
            last_block["cache_control"] = dict(_USER_MESSAGE_CACHE_MARKER)


def _classify_max_turns(stop_reason: str | None, turns_used: int, turns_limit: int) -> bool:
    """Detect whether the SDK ended due to max-turns being hit.

    Per memory project_crucible_breach_judge_max_turns_crash.md (B31):
    trust the SDK's canonical ``stop_reason`` exclusively. The
    previous ``turns_used >= 95% of turns_limit`` fallback fired
    spuriously when the SDK reported a non-canonical ``stop_reason``
    (e.g. ``"error"`` or empty) for an invocation that happened to
    exhaust most of its turn budget — misclassifying a legitimate
    completion as a max-turns failure. The SDK's own counter is the
    source of truth; if it didn't say ``max_turns``, the invocation
    wasn't a max-turns event. The provider's turn-limit terminal
    subtype (``error_max_turns``) is normalised to the canonical stop
    reason before this check.
    """
    return stop_reason in ("max_turns", "max-turns")


class ClaudeBackend:
    """assay.Backend implementation over claude-agent-sdk."""

    name = "claude"

    @property
    def features(self) -> Any:
        """Return the observed profile for this concrete Claude route."""

        from .features import feature_profile_for_route

        return feature_profile_for_route(self.name, model=self.model)

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        cwd: str | None = None,
        cli_path: str | None = None,
        effort: str = "high",
        log_dir: Path | str | None = None,
        capability: str = "text",
        native_shell: bool | None = None,
        expected_base_url: str | None = None,
        prefer_infer_credentials: bool = False,
        credential_source: str = "environment",
        expected_org_id: str | None = None,
        expected_org_name: str | None = None,
    ) -> None:
        from .capabilities import normalize_capability

        self.capability = normalize_capability(capability)
        if native_shell is True and self.capability != "workspace_write":
            raise ValueError(
                "native_shell requires workspace_write capability"
            )
        self._native_shell = (
            self.capability == "workspace_write"
            if native_shell is None
            else native_shell
        )
        if (
            type(model) is not str
            or not model
            or model != model.strip()
        ):
            raise ValueError("ClaudeBackend model must be a non-empty string")
        # ``model`` is part of the assay Backend protocol (public attr);
        # callers and ``assay.implement`` read it to populate
        # ExecutionContext.model.
        self.model = model
        source = _validated_credential_source(credential_source)
        if type(prefer_infer_credentials) is not bool:
            raise TypeError("prefer_infer_credentials must be a bool")
        if source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
            if model not in CLAUDE_SESSION_MODELS:
                raise ValueError(
                    "credential_source='claude_session' requires exact model "
                    f"from reviewed set {sorted(CLAUDE_SESSION_MODELS)!r}"
                )
            if expected_base_url is not None:
                raise ValueError(
                    "credential_source='claude_session' cannot be combined "
                    "with expected_base_url"
                )
            if prefer_infer_credentials:
                raise ValueError(
                    "credential_source='claude_session' cannot be combined "
                    "with prefer_infer_credentials"
                )
            (
                validated_org_id,
                validated_org_name,
            ) = _validated_session_org_identity(
                expected_org_id=expected_org_id,
                expected_org_name=expected_org_name,
            )
        else:
            if expected_org_id is not None or expected_org_name is not None:
                raise ValueError(
                    "expected_org_id/expected_org_name require "
                    "credential_source='claude_session'"
                )
            validated_org_id = None
            validated_org_name = None
        self._credential_source = source
        self._expected_org_id = validated_org_id
        self._expected_org_name = validated_org_name
        self._expected_endpoint = _validated_endpoint_identity(
            expected_base_url,
            setting="ClaudeBackend expected_base_url",
        )
        self._prefer_infer_credentials = prefer_infer_credentials
        self._cwd = cwd or os.getcwd()
        # Prefer an explicit cli_path; otherwise pin the system CLI to
        # dodge the bundled build's stale adaptive-thinking model gate
        # (see ``_system_cli_path``). Falls back to the SDK's own
        # resolution only when no system ``claude`` is installed.
        self._session_cli_pin: Any | None = None
        if source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
            selected_cli_path = (
                cli_path if cli_path is not None else _system_cli_path()
            )
            self._cli_path = _validated_session_cli_path(selected_cli_path)
            from ._claude_session_transport import (  # noqa: PLC0415
                PinnedClaudeCLI,
            )

            self._session_cli_pin = PinnedClaudeCLI.capture(self._cli_path)
        else:
            self._cli_path = cli_path or _system_cli_path()
        # Claude CLI accepts ``low | medium | high | max``. Default to
        # "high" so workflow-driven planning / implementation calls don't
        # silently underperform on harder Specs.
        if effort not in ("low", "medium", "high", "max"):
            raise ValueError(
                f"effort must be one of low/medium/high/max; got {effort!r}"
            )
        self._effort = effort
        # ``log_dir``: per-invocation SDK debug dumps land here as
        # ``sdk_<seq>_claude.json``. Workflows typically pass
        # ``handle.run_dir`` from ``corpus.open_run(...)`` so the dumps
        # sit alongside ``events.jsonl`` / ``result.json`` /
        # ``build.log`` in the run folder. When None, no debug files
        # are written (the workflow either doesn't care or hasn't
        # opened a run folder yet).
        self._sdk_sink = SdkDebugSink(
            Path(log_dir) if log_dir is not None else None,
            backend_name="claude",
        )
        # In-memory MCP attachment. Set via ``attach_toolset(...)`` —
        # ``_mcp_attachment`` carries ``server`` (the McpSdkServerConfig
        # the SDK accepts as ``mcp_servers["crucible"]``),
        # ``tool_names`` (the ``mcp__crucible__<name>`` strings appended
        # to ``allowed_tools``), and ``worker`` (the dispatch thread
        # pool to shut down on ``close()``). None means "no custom
        # tools attached; the bundled CLI uses only its built-ins +
        # whatever ``Invocation.tools`` names".
        self._mcp_attachment: dict[str, Any] | None = None
        self._lifecycle_lock = threading.RLock()
        self._active_mutation_hosts: dict[int, Any] = {}
        self._active_dispatch_workers: dict[int, Any] = {}
        self._closed = False

        # Partial usage published by ``_invoke_once`` as each turn's
        # ``AssistantMessage`` streams in. The wall-timeout handler in
        # ``_invoke_with_retries`` reads this to attribute the cost a
        # cancelled call already incurred: the authoritative cost only
        # arrives on the final ``ResultMessage``, which never lands when
        # ``asyncio.wait_for`` cancels a long implementer turn — so
        # without this the timed-out call would report ``cost_usd=0``
        # despite burning tens of minutes of tool-loop tokens. Reset at
        # the top of every ``_invoke_once`` pass.
        self._partial_usage: Usage | None = None
        # Exact visible response segments observed before a terminal. This
        # stays incomplete until ResultMessage reconciles the full envelope.
        self._partial_pricing_evidence: UsagePricingEvidence | None = None
        # Canonical ResultMessage.num_turns for the most recent physical
        # attempt, or the observed AssistantMessage lower bound while a stream
        # is incomplete. The retry wrapper uses these to keep the logical
        # turn ceiling cumulative rather than resetting it per reconnect.
        self._last_num_turns: int | None = None
        self._partial_turns = 0

    def _validate_session_cli_path(self) -> None:
        if self._credential_source != CLAUDE_SESSION_CREDENTIAL_SOURCE:
            return
        if "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT" in os.environ:
            raise ValueError(
                "credential_source='claude_session' rejects ambient "
                "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT because the pinned Agent "
                "SDK reads it in the parent process"
            )
        if self._session_cli_pin is None:  # pragma: no cover - constructor invariant
            raise ValueError(
                "credential_source='claude_session' has no pinned Claude CLI"
            )
        self._session_cli_pin.revalidate()

    def _trace_endpoint_identity(self) -> str | None:
        if self._credential_source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
            return CLAUDE_SESSION_ENDPOINT
        return _endpoint_identity(
            _effective_base_url(
                prefer_infer=self._prefer_infer_credentials,
                credential_source=self._credential_source,
            )
        )

    def _validate_expected_endpoint(self) -> str | None:
        """Bind an operator route to the endpoint used at invocation time."""

        if self._credential_source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
            self._validate_session_cli_path()
            return CLAUDE_SESSION_ENDPOINT
        effective = _validated_endpoint_identity(
            _effective_base_url(
                prefer_infer=self._prefer_infer_credentials,
                credential_source=self._credential_source,
            ),
            setting="effective Claude base URL",
        )
        if (
            self._expected_endpoint is not None
            and effective != self._expected_endpoint
        ):
            raise ValueError(
                "ClaudeBackend endpoint drift: configured route expects "
                f"{self._expected_endpoint!r}, effective environment is "
                f"{effective!r}"
            )
        if (
            self._prefer_infer_credentials
            and not os.environ.get("INFER_AUTH_TOKEN", "").strip()
        ):
            raise ValueError(
                "ClaudeBackend routed through the inference gateway "
                "requires INFER_AUTH_TOKEN at invocation time"
            )
        return effective

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach_toolset(
        self,
        toolset: Any,
        *,
        context_budget: int | None = None,
        mcp_max_workers: int = 2,
    ) -> None:
        """Expose ``toolset`` as an in-process MCP server for the bundled CLI.

        After attachment, every ``invoke()`` call adds
        ``mcp_servers={"crucible": <sdk_config>}`` to ``ClaudeAgentOptions``
        and appends ``mcp__crucible__<name>`` entries to ``allowed_tools``.
        The agent SDK dispatches tool calls back into this Python process —
        no subprocess, no stdio JSON-RPC.

        Args:
          context_budget: Optional ceiling on total tool-result bytes
            per ``invoke()`` call. The tracker is reset between invokes
            (per-invocation cap, not per-attachment).
          mcp_max_workers: Concurrent in-flight tool dispatch slots.
            Default 2 (read + build slot historically). Bump for
            workflows that fan out many concurrent tool calls.

        Replaces any prior attachment. Pass ``None`` to detach (released
        on next call).
        """
        from .capabilities import validate_toolset_capability

        if context_budget is not None and (
            isinstance(context_budget, bool)
            or not isinstance(context_budget, int)
            or context_budget <= 0
        ):
            raise ValueError("context_budget must be a positive integer")
        if type(mcp_max_workers) is not int or mcp_max_workers <= 0:
            raise ValueError("mcp_max_workers must be a positive integer")
        validate_toolset_capability(self.capability, toolset)

        # Lazy import: avoid pulling claude-agent-sdk into module-import
        # graph when no toolset is in play.
        from crucible.tools._mcp_bridge import build_mcp_server

        # Tear down a stale attachment if present.
        self._detach_toolset()
        if toolset is None:
            return
        write_names = frozenset(
            str(name) for name in getattr(toolset, "write_names", ())
        )
        if write_names:
            # Qualify the complete mutating surface before publishing the
            # attachment. The process itself is invocation-scoped because its
            # kill boundary must share that invocation's absolute deadline.
            from crucible.tools._isolated_mutation_host import (  # noqa: PLC0415
                build_isolated_toolset_spec,
            )

            build_isolated_toolset_spec(
                toolset,
                expected_write_names=write_names,
            )
            descriptors = toolset.to_anthropic_tools()
            mcp_tool_names = [
                f"mcp__crucible__{descriptor['name']}"
                for descriptor in descriptors
            ]
            server = None
            worker = None
            budget_tracker = None
        else:
            server, mcp_tool_names, worker, budget_tracker = build_mcp_server(
                toolset,
                context_budget=context_budget,
                max_workers=mcp_max_workers,
            )
            self._register_dispatch_worker(worker)
        with self._lifecycle_lock:
            if self._closed:
                if worker is not None:
                    worker.shutdown()
                    self._unregister_dispatch_worker(worker)
                raise RuntimeError("ClaudeBackend is closed")
            self._mcp_attachment = {
                "server": server,
                "tool_names": mcp_tool_names,
                "worker": worker,
                "budget_tracker": budget_tracker,
                "toolset": toolset,
                "context_budget": context_budget,
                "max_workers": mcp_max_workers,
            }
        logger.info(
            "ClaudeBackend: attached toolset %r exposing %d MCP tools "
            "(max_workers=%d, context_budget=%s)",
            getattr(toolset, "domain", "?"),
            len(mcp_tool_names),
            mcp_max_workers,
            f"{context_budget:,}" if context_budget else "off",
        )

    def _detach_toolset(self) -> None:
        attachment = self._mcp_attachment
        if attachment is None:
            return
        worker = attachment.get("worker")
        if worker is not None:
            worker.shutdown()
            self._unregister_dispatch_worker(worker)
        if self._mcp_attachment is attachment:
            self._mcp_attachment = None

    def _register_mutation_host(self, host: Any) -> None:
        with self._lifecycle_lock:
            closed = self._closed
            self._active_mutation_hosts[id(host)] = host
        if closed:
            try:
                host.stop()
            except BaseException:
                raise
            else:
                self._unregister_mutation_host(host)
            raise RuntimeError("ClaudeBackend is closed")

    def _ensure_lifecycle_open(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("ClaudeBackend is closed")

    def _unregister_mutation_host(self, host: Any) -> None:
        with self._lifecycle_lock:
            if self._active_mutation_hosts.get(id(host)) is host:
                del self._active_mutation_hosts[id(host)]

    def _register_dispatch_worker(self, worker: Any) -> None:
        with self._lifecycle_lock:
            closed = self._closed
            self._active_dispatch_workers[id(worker)] = worker
        if closed:
            try:
                worker.shutdown()
            except BaseException:
                raise
            else:
                self._unregister_dispatch_worker(worker)
            raise RuntimeError("ClaudeBackend is closed")

    def _unregister_dispatch_worker(self, worker: Any) -> None:
        with self._lifecycle_lock:
            if self._active_dispatch_workers.get(id(worker)) is worker:
                del self._active_dispatch_workers[id(worker)]

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

    def close(self) -> None:
        """Positively settle and retain every invocation-owned resource."""

        with self._lifecycle_lock:
            self._closed = True
        errors: list[BaseException] = []
        try:
            self._detach_toolset()
        except BaseException as exc:
            errors.append(exc)
        dispatch_settled = not errors

        with self._lifecycle_lock:
            hosts = tuple(self._active_mutation_hosts.values())
            workers = tuple(self._active_dispatch_workers.values())
        for worker in workers:
            try:
                worker.shutdown()
            except BaseException as exc:
                errors.append(exc)
                dispatch_settled = False
            else:
                self._unregister_dispatch_worker(worker)
        if dispatch_settled:
            for host in hosts:
                try:
                    host.stop()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    self._unregister_mutation_host(host)

        with self._lifecycle_lock:
            resources_remain = bool(
                self._active_mutation_hosts
                or self._active_dispatch_workers
                or self._mcp_attachment is not None
            )
        for cleanup_error in errors:
            if isinstance(cleanup_error, KeyboardInterrupt):
                # Preserve cancellation even when an earlier owner also failed;
                # the backend pool will retain and retry this adapter before
                # replaying Ctrl-C at the provider settlement boundary.
                raise cleanup_error
        if errors:
            raise errors[0]
        if resources_remain:
            raise RuntimeError(
                "ClaudeBackend.close could not confirm all resources settled"
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
        context_budget: int | None = None,
    ) -> InvocationResult:
        """Run the direct loop with process-isolated mutators when present."""

        if self._credential_source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
            raise NotImplementedError(
                "Claude credential_source='claude_session' cannot use the "
                "direct Messages API; use invoke() through the Claude Agent "
                "SDK/CLI transport"
            )
        if invocation.conversation is not None:
            raise NotImplementedError(
                "ClaudeBackend.chat_with_tools uses the direct Messages API "
                "and cannot resume provider conversations"
            )
        from .capabilities import validate_toolset_capability

        validate_toolset_capability(self.capability, toolset)
        if cache is not None and getattr(toolset, "write_names", ()):
            raise NotImplementedError(
                "ClaudeBackend cannot safely cache a toolset that can mutate "
                "workspace state"
            )
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        if (
            type(max_response_tokens) is not int
            or max_response_tokens <= 0
        ):
            raise ValueError(
                "max_response_tokens must be a positive integer"
            )
        if type(retry_budget) is not int or retry_budget <= 0:
            raise ValueError("retry_budget must be a positive integer")
        if context_budget is not None and (
            type(context_budget) is not int or context_budget <= 0
        ):
            raise ValueError("context_budget must be a positive integer")
        policy = invocation.policy or _DEFAULT_POLICY
        timeout_seconds = _validated_timeout_seconds(policy)
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + timeout_seconds
        )
        hosted_toolset, mutation_host = _process_hosted_toolset(
            toolset,
            invocation_deadline=deadline,
            register_host=self._register_mutation_host,
            unregister_host=self._unregister_mutation_host,
            ensure_open=self._ensure_lifecycle_open,
        )
        try:
            return self._chat_with_tools_direct(
                invocation,
                hosted_toolset,
                cache=cache,
                max_rounds=max_rounds,
                max_response_tokens=max_response_tokens,
                retry_budget=retry_budget,
                context_budget=context_budget,
                _logical_deadline=deadline,
                _mutation_host=mutation_host,
            )
        finally:
            if mutation_host is not None:
                mutation_host.stop()
                self._unregister_mutation_host(mutation_host)

    def _chat_with_tools_direct(
        self,
        invocation: Invocation,
        toolset,
        *,
        cache=None,
        max_rounds: int = 10,
        max_response_tokens: int = 16_000,
        retry_budget: int = 3,
        context_budget: int | None = None,
        _logical_deadline: float | None = None,
        _mutation_host: Any | None = None,
    ) -> InvocationResult:
        """Direct Anthropic Messages-API tool-use loop.

        Bypasses the bundled ``claude`` CLI entirely — uses
        ``anthropic.Anthropic()`` directly. This is the escape hatch
        for hot paths that hit the bundled CLI's 25K-token Read cap
        on large source files (the historical "SDK choke" symptom —
        see ``project_assay_implement_sdk_choke`` memory entry). The
        Messages-API tool-result block has no such cap; arbitrary
        tool returns are bounded only by the model's context window.

        Tool catalog is the caller-supplied ``toolset`` (typically
        built via ``crucible.tools.profiles.build_*_profile(ctx)``).
        On each round:

        1. POST ``messages.create(messages=[...], tools=toolset.to_anthropic_tools())``.
        2. If ``stop_reason`` ∈ {end_turn, stop_sequence, max_tokens,
           refusal}: done.
        3. Otherwise, for each ToolUseBlock dispatch via
           ``toolset.dispatch(name, args)``, feed each result back as
           a ``tool_result`` block, continue.

        Loop bounds:
        - ``max_rounds`` (default 10) — hard ceiling on round count.
        - ``invocation.budget_usd`` — soft cost cap; aborts the loop
          when cumulative cost (computed via
          ``crucible.backends.pricing.compute_cost``) exceeds the cap.
        - ``max_response_tokens`` (default 16_000) — per-round
          response-token ceiling passed to messages.create. Bump for
          long agentic tasks where the model legitimately needs longer
          turns; the per-model API ceiling is the hard upper bound.
        - ``context_budget`` — optional hard cumulative UTF-8 byte ceiling on
          tool results admitted to the provider conversation. It spans all
          rounds and cache hits in this logical invocation.

        ``cache``: optional ``assay.ToolCallCache``. When supplied,
        successful tool dispatches are memoized by (name, canonical
        args); errors are not cached. Hits skip the
        ``toolset.dispatch`` call but still appear in the transcript
        for postmortem.

        ``retry_budget`` is retained for call-site compatibility. Direct
        Messages transport exceptions are terminal because the missing
        response envelope makes billed usage unknowable; exact, successful
        tool rounds may still continue to the next provider request.

        Returns an ``InvocationResult`` shaped identically to
        ``invoke()`` so callers can swap freely.

        Context growth + caching: each round appends the full assistant
        content + tool_results to the running messages list. A 10-round
        loop with verbose tool returns can balloon the request payload
        to tens of KB by round 10. Prompt caching is applied at two
        cache_control breakpoints per round:

        - **System block** — ``cache_control: ephemeral`` on the
          ``invocation.system`` text; constant across rounds, cached
          on the first call, cache_read on every subsequent round.
        - **Most-recent user message** — before each
          ``messages.create``, ``_set_user_message_cache_marker``
          places a single ``cache_control: ephemeral`` marker on the
          last user message's terminal content block, stripping any
          stale markers from earlier user messages. This caches the
          entire conversation prefix through the latest tool_results,
          so round N's billed input shrinks to just the new content
          since round N-1's marker (~5× input-token reduction on long
          tool-heavy loops, in exchange for a single cache-write per
          round).

        Pathological cases (very long tool returns, > 4 tool_uses per
        round) can still hit the model's context window. Cap
        ``max_rounds`` defensively when running large-context loops in
        production.
        """
        from .capabilities import validate_toolset_capability
        from .pricing import pricing_for_model

        if self._credential_source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
            raise NotImplementedError(
                "Claude credential_source='claude_session' cannot use the "
                "direct Messages API"
            )
        pricing_for_model(self.model)
        validate_toolset_capability(self.capability, toolset)
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        if (
            type(max_response_tokens) is not int
            or max_response_tokens <= 0
        ):
            raise ValueError(
                "max_response_tokens must be a positive integer"
            )
        if type(retry_budget) is not int or retry_budget <= 0:
            raise ValueError("retry_budget must be a positive integer")
        if context_budget is not None and (
            type(context_budget) is not int
            or context_budget <= 0
        ):
            raise ValueError("context_budget must be a positive integer")
        tool_authority_digest = self._tool_authority_digest(
            toolset,
            context_budget=context_budget,
        )
        endpoint_identity = self._validate_expected_endpoint()
        policy = invocation.policy or _DEFAULT_POLICY
        retry_budget = min(max(1, policy.retry_budget), retry_budget)
        timeout_seconds = _validated_timeout_seconds(policy)
        deadline = (
            _logical_deadline
            if _logical_deadline is not None
            else (
                None
                if timeout_seconds is None
                else time.monotonic() + timeout_seconds
            )
        )
        # Local imports to keep the optional dependency lazy.
        from anthropic import Anthropic
        from anthropic.types import TextBlock, ToolUseBlock

        from crucible.tools._dispatch_budget import (
            TerminalToolDispatchError,
            ToolDispatchBudgetExceeded,
            ToolMutationIndeterminate,
        )

        from .pricing import compute_cost

        client = Anthropic(
            **_direct_client_kwargs(
                prefer_infer=self._prefer_infer_credentials,
                credential_source=self._credential_source,
            )
        )

        # System + tools are constant across rounds — eligible for
        # prompt caching on the API side. The system block carries
        # cache_control so the API can cache it and charge cache_read
        # rates on subsequent rounds.
        system_block = [{
            "type": "text",
            "text": invocation.system,
            "cache_control": {"type": "ephemeral"},
        }]
        tools = toolset.to_anthropic_tools()
        allowed_tool_names = {tool["name"] for tool in tools}

        messages: list[dict] = [
            {"role": "user", "content": invocation.user},
        ]
        transcript: list[Message] = [
            Message(role="user", content=invocation.user),
        ]

        total_input = 0
        total_output = 0
        total_cache_create = 0
        total_cache_read = 0
        total_cost_usd = 0.0
        pricing_segments: list[UsagePricingSegment] = []
        observed_served_models: set[str] = set()
        served_model: str | None = None
        pricing_identity_complete = True
        provider_request_id: str | None = None
        saw_response = False
        terminal_response_seen = False
        usage_complete = True
        execution_events: list[BackendExecutionEvent] = []
        context_result_bytes = 0
        seen_tool_call_ids: set[str] = set()

        rounds = 0
        round_limit = max(
            1,
            min(max_rounds, max(1, invocation.max_turns)),
        )
        stop_reason = "end_turn"
        final_message_text = ""
        classified_error = None
        tool_uses: list[Any] = []

        while rounds < round_limit:
            rounds += 1
            terminal_response_seen = False
            tool_uses = []
            # Refresh the message-level cache_control marker so the
            # API can match the conversation prefix from cache. Done
            # inside the round loop (not once before) because the
            # previous round appended tool_results that we want to
            # be cached for the next round.
            _set_user_message_cache_marker(messages)
            # One physical provider request per round. ``retry_budget`` is
            # retained for API compatibility, but a transport exception has
            # no authoritative usage envelope, so an automatic paid retry
            # would make logical token/cost accounting permanently ambiguous.
            response = None
            for attempt in range(1, retry_budget + 1):
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    stop_reason = "timeout"
                    classified_error = TransportError(
                        http_status=None,
                        message=(
                            "Claude direct Messages loop exceeded the "
                            "invocation wall deadline"
                        ),
                        retriable=False,
                    )
                    execution_events.append(
                        BackendExecutionEvent(
                            sequence=len(execution_events),
                            kind="limit",
                            name="deadline",
                        )
                    )
                    break
                request_name = (
                    f"messages_round_{rounds}_attempt_{attempt}"
                )
                execution_events.append(
                    BackendExecutionEvent(
                        sequence=len(execution_events),
                        kind="request",
                        name=request_name,
                    )
                )
                try:
                    output_config: dict[str, Any] = {
                        "effort": self._effort,
                    }
                    if invocation.response_schema is not None:
                        output_config["format"] = _json_output_format(
                            invocation.response_schema
                        )
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=max_response_tokens,
                        system=system_block,
                        messages=messages,
                        tools=tools,
                        thinking=_adaptive_thinking(),
                        output_config=output_config,
                        **(
                            {"timeout": remaining}
                            if remaining is not None
                            else {}
                        ),
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    # The request may have reached the provider before the
                    # transport failed, so its billed usage is unknowable
                    # even when this is the final (non-retried) attempt.
                    usage_complete = False
                    classified = self._classify_exception(
                        exc, policy,
                    )
                    if (
                        _mutation_host is not None
                        and _mutation_host.ever_started_write
                        and classified is not None
                        and getattr(classified, "retriable", False)
                    ):
                        classified = _replace(
                            classified,
                            message=(
                                f"{classified.message}; automatic retry "
                                "suppressed after a mutating tool call"
                            ),
                            retriable=False,
                        )
                    if (
                        classified is not None
                        and getattr(classified, "retriable", False)
                    ):
                        classified = _replace(
                            classified,
                            message=(
                                f"{classified.message}; automatic retry "
                                "suppressed because the prior physical "
                                "attempt's token/cost accounting is "
                                "incomplete"
                            ),
                            retriable=False,
                        )
                    # Non-retriable, or retries exhausted.
                    classified_error = classified or TransportError(
                        kind="unclassified",
                        message=f"{type(exc).__name__}: {exc}",
                        retriable=False,
                    )
                    if (
                        getattr(classified_error, "kind", None)
                        == "cancellation"
                    ):
                        _append_event(
                            execution_events,
                            "cancellation",
                            name="external",
                        )
                    logger.warning(
                        "chat_with_tools: round %d raised %s — %s "
                        "(attempt %d/%d)",
                        rounds, type(classified_error).__name__,
                        classified_error.message,
                        attempt, retry_budget,
                    )
                    break
            if response is None:
                # All retries exhausted or non-retriable error.
                if classified_error is not None and stop_reason != "timeout":
                    stop_reason = "error"
                break

            saw_response = True
            terminal_response_seen = True
            served_model = _record_served_model(
                self.model,
                getattr(response, "model", None),
                transport="messages",
                observed=observed_served_models,
            )
            model_identity_error = (
                _missing_served_model_identity()
                if served_model is None
                else _served_model_mismatch(
                    self.model,
                    served_model,
                    transport="messages",
                )
            )
            observed_request_id = getattr(response, "id", None)
            if (
                isinstance(observed_request_id, str)
                and observed_request_id.strip()
            ):
                provider_request_id = observed_request_id.strip()
            usage = getattr(response, "usage", None)
            raw_in = _usage_value(usage, "input_tokens")
            raw_out = _usage_value(usage, "output_tokens")
            raw_cache_create = _usage_value(
                usage, "cache_creation_input_tokens"
            )
            raw_cache_read = _usage_value(
                usage, "cache_read_input_tokens"
            )
            round_usage_complete = (
                usage is not None
                and _valid_token_counter(raw_in)
                and _valid_token_counter(raw_out)
                and (
                    raw_cache_create is None
                    or _valid_token_counter(raw_cache_create)
                )
                and (
                    raw_cache_read is None
                    or _valid_token_counter(raw_cache_read)
                )
            )
            round_pricing_segment = (
                _pricing_segment_from_usage(usage, model=self.model)
                if usage is not None
                else None
            )
            if round_pricing_segment is not None:
                pricing_segments.append(round_pricing_segment)
            if not round_usage_complete:
                usage_complete = False
            if usage is None:
                usage = object()
            in_t = raw_in if _valid_token_counter(raw_in) else 0
            out_t = raw_out if _valid_token_counter(raw_out) else 0
            cc_t = (
                raw_cache_create
                if _valid_token_counter(raw_cache_create)
                else 0
            )
            cr_t = (
                raw_cache_read
                if _valid_token_counter(raw_cache_read)
                else 0
            )
            round_total_input = in_t + cc_t + cr_t
            total_input += round_total_input
            total_output += out_t
            total_cache_create += cc_t
            total_cache_read += cr_t
            total_cost_usd += (
                round_pricing_segment.cost_usd
                if round_pricing_segment is not None
                else compute_cost(
                    self.model,
                    input_tokens=round_total_input,
                    output_tokens=out_t,
                    cache_creation_tokens=cc_t,
                    cache_read_tokens=cr_t,
                )
            )
            if usage is not None:
                _append_event(
                    execution_events,
                    "usage",
                    name=(
                        f"messages_round_{rounds}_"
                        f"{'complete' if round_usage_complete else 'partial'}"
                    ),
                )

            if model_identity_error is not None:
                pricing_identity_complete = False
                # Attribute the provider response's billed usage, but never
                # admit its content or dispatch tool calls without exact route
                # identity proof. The logical terminal evidence remains
                # incomplete because the requested route produced no admitted
                # terminal result.
                stop_reason = (
                    "missing_model_identity"
                    if served_model is None
                    else "model_mismatch"
                )
                classified_error = model_identity_error
                terminal_response_seen = False
                _append_event(
                    execution_events,
                    "limit",
                    name="served_model_mismatch",
                )
                break

            if not round_usage_complete:
                # A provider response without exact usage can contribute only
                # a lower bound.  Do not dispatch its tools: doing so would
                # either strand a local mutation or require a second paid
                # Messages request whose aggregate accounting cannot become
                # exact.
                stop_reason = "incomplete_accounting"
                classified_error = IncompleteEvidence(
                    missing=(
                        "token_usage_complete",
                        "cost_complete",
                    ),
                    message=(
                        "Claude Messages response omitted or malformed "
                        "token/cost accounting; refusing another physical "
                        "request"
                    ),
                )
                terminal_response_seen = False
                _append_event(
                    execution_events,
                    "limit",
                    name="accounting_incomplete",
                )
                break

            # Append the full assistant content (including ToolUseBlocks)
            # to the running message history so the next round sees the
            # tool_use ids it must reply with tool_result blocks for.
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            text_parts: list[str] = []
            tool_uses: list = []
            for block in response.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_uses.append(block)
            round_text = "\n".join(text_parts)
            if round_text:
                transcript.append(Message(role="assistant", content=round_text))
                final_message_text = round_text

            stop_reason = response.stop_reason or "end_turn"

            if (
                invocation.budget_usd is not None
                and total_cost_usd > invocation.budget_usd
            ):
                stop_reason = "budget_exhausted"
                classified_error = BudgetExhausted(
                    spent_usd=total_cost_usd,
                    limit_usd=invocation.budget_usd,
                    message=(
                        f"Invocation spent ${total_cost_usd:.4f} exceeding "
                        f"budget ${invocation.budget_usd:.4f}"
                    ),
                )
                execution_events.append(
                    BackendExecutionEvent(
                        sequence=len(execution_events),
                        kind="limit",
                        name="budget_usd",
                    )
                )
                break

            # Termination conditions.
            if not tool_uses:
                break
            if stop_reason in ("end_turn", "stop_sequence", "max_tokens", "refusal"):
                break
            # Dispatch each tool_use; feed results back as user-role
            # tool_result blocks.
            tool_results: list[dict] = []
            for use in tool_uses:
                raw_call_id = getattr(use, "id", None)
                if (
                    not isinstance(raw_call_id, str)
                    or not raw_call_id.strip()
                    or raw_call_id != raw_call_id.strip()
                    or raw_call_id in seen_tool_call_ids
                ):
                    stop_reason = "error"
                    classified_error = TransportError(
                        http_status=None,
                        message=(
                            "Claude Messages returned a missing, empty, or "
                            "duplicate tool-use id; refusing ambiguous result "
                            "pairing"
                        ),
                        retriable=False,
                    )
                    break
                call_id = raw_call_id
                seen_tool_call_ids.add(call_id)
                tool_name = getattr(use, "name", None)
                if (
                    not isinstance(tool_name, str)
                    or not tool_name
                    or tool_name != tool_name.strip()
                ):
                    stop_reason = "error"
                    classified_error = TransportError(
                        http_status=None,
                        message=(
                            "Claude Messages returned a missing, empty, or "
                            "non-canonical tool name; refusing dispatch"
                        ),
                        retriable=False,
                    )
                    break
                _append_event(
                    execution_events,
                    "tool_call",
                    name=tool_name,
                    parent_id=request_name,
                    child_id=call_id,
                )
                args = use.input if isinstance(use.input, dict) else {}
                result_str: str | None = None
                is_error = False
                if tool_name not in allowed_tool_names:
                    # Provider output is untrusted. A stale or hallucinated
                    # name must never reach dispatch (or a cache entry) unless
                    # it is an exact member of the catalog bound into the
                    # invocation's tool-authority digest.
                    result_str = (
                        f"ERROR: tool {tool_name!r} is not in the exposed "
                        "tool catalog"
                    )
                    is_error = True
                else:
                    # Cache check
                    cache_allowed = (
                        cache is not None
                        and tool_name
                        not in getattr(toolset, "write_names", ())
                    )
                    if cache_allowed:
                        cached = cache.get(tool_name, args)
                        if cached is not None:
                            result_str = cached
                    if result_str is None:
                        try:
                            result_str = toolset.dispatch(tool_name, args)
                        except ToolMutationIndeterminate as exc:
                            stop_reason = "tool_mutation_indeterminate"
                            classified_error = TransportError(
                                http_status=None,
                                message=(
                                    "Claude tool loop terminated after an "
                                    "indeterminate source mutation: "
                                    f"{exc}"
                                ),
                                retriable=False,
                            )
                            terminal_response_seen = False
                            break
                        except ToolDispatchBudgetExceeded as exc:
                            stop_reason = "tool_dispatch_budget_exhausted"
                            classified_error = MaxTurnsExceeded(
                                turns_used=rounds,
                                turns_limit=round_limit,
                                message=(
                                    "Claude tool loop stopped at the first "
                                    f"dispatch-budget denial: {exc}"
                                ),
                            )
                            terminal_response_seen = False
                            _append_event(
                                execution_events,
                                "limit",
                                name="tool_dispatch_budget",
                            )
                            break
                        except TerminalToolDispatchError as exc:
                            stop_reason = "tool_dispatch_terminal"
                            classified_error = TransportError(
                                http_status=None,
                                message=(
                                    "Claude tool loop terminated at a "
                                    "workflow-owned dispatch boundary: "
                                    f"{exc}"
                                ),
                                retriable=False,
                            )
                            terminal_response_seen = False
                            break
                        except Exception as e:  # noqa: BLE001
                            result_str = (
                                f"ERROR: {type(e).__name__}: {e}"
                            )
                            is_error = True
                        if cache_allowed and not is_error:
                            cache.put(tool_name, args, result_str)
                rendered_result = str(result_str)
                result_bytes = len(rendered_result.encode("utf-8"))
                if (
                    context_budget is not None
                    and context_result_bytes + result_bytes > context_budget
                ):
                    stop_reason = "context_limit"
                    classified_error = TransportError(
                        http_status=None,
                        message=(
                            "tool-result context budget exceeded: "
                            f"{context_result_bytes + result_bytes} > "
                            f"{context_budget} UTF-8 bytes"
                        ),
                        retriable=False,
                    )
                    _append_event(
                        execution_events,
                        "limit",
                        name="context_bytes",
                    )
                    break
                context_result_bytes += result_bytes
                tool_result_block: dict = {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": rendered_result,
                }
                if is_error:
                    tool_result_block["is_error"] = True
                tool_results.append(tool_result_block)
                _append_event(
                    execution_events,
                    "tool_result",
                    name=tool_name,
                    parent_id=request_name,
                    child_id=call_id,
                )
                transcript.append(Message(
                    role="tool",
                    content=f"{tool_name}({args}) -> {rendered_result}",
                ))
            if classified_error is not None:
                break
            messages.append({"role": "user", "content": tool_results})

        if (
            rounds >= round_limit
            and classified_error is None
            and tool_uses
        ):
            # Loop hit the hard ceiling with tool calls still pending —
            # surface as max_turns so callers see the same
            # classification ``invoke()`` would emit.
            stop_reason = "max_turns"
            if policy.on_max_turns == "raise":
                classified_error = MaxTurnsExceeded(
                    turns_used=rounds,
                    turns_limit=round_limit,
                    message=(
                        f"Claude direct Messages loop reached {round_limit} "
                        "rounds with tool calls still pending"
                    ),
                )
            execution_events.append(
                BackendExecutionEvent(
                    sequence=len(execution_events),
                    kind="limit",
                    name="max_turns",
                )
            )

        result_usage = Usage(
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=total_cost_usd,
            cache_creation_tokens=total_cache_create,
            cache_read_tokens=total_cache_read,
        )
        pricing_evidence = UsagePricingEvidence(
            segments=tuple(pricing_segments),
            complete=(
                saw_response
                and usage_complete
                and pricing_identity_complete
                and len(pricing_segments) == rounds
            ),
        )
        execution_events.append(
            BackendExecutionEvent(
                sequence=len(execution_events),
                kind="terminal",
                name=stop_reason,
            )
        )
        from .features import feature_profile_digest

        return InvocationResult(
            final_message=final_message_text,
            transcript=tuple(transcript),
            usage=result_usage,
            stop_reason=stop_reason,
            error=classified_error,
            completeness=ExecutionCompleteness(
                token_usage_complete=saw_response and usage_complete,
                cost_complete=(
                    saw_response
                    and usage_complete
                    and pricing_identity_complete
                ),
                terminal_complete=terminal_response_seen,
                trace_complete=True,
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                served_model=served_model,
                requested_effort=self._effort,
                # Effort is a request control; the Messages response does not
                # echo the effective value.
                served_effort=None,
                endpoint=endpoint_identity,
                provider_request_id=provider_request_id,
                feature_profile_digest=feature_profile_digest(self.features),
                tool_authority_digest=tool_authority_digest,
                events=tuple(execution_events),
            ),
            pricing_evidence=pricing_evidence,
        )

    def invoke(self, invocation: Invocation) -> InvocationResult:
        """Synchronous entry point that drives an async SDK query."""
        from .pricing import pricing_for_model

        pricing_for_model(self.model)
        self._validate_expected_endpoint()
        return asyncio.run(self._invoke_with_retries(invocation))

    # ------------------------------------------------------------------
    # Retry + policy enforcement
    # ------------------------------------------------------------------

    def _result_with_evidence(
        self,
        *,
        final_message: str,
        transcript: tuple[Message, ...],
        usage: Usage,
        stop_reason: str,
        error: Any = None,
        token_usage_complete: bool = False,
        cost_complete: bool = False,
        terminal_complete: bool = False,
        events: list[BackendExecutionEvent] | None = None,
        trace_template: BackendExecutionTrace | None = None,
        served_model: str | None = None,
        provider_request_id: str | None = None,
        tool_authority_digest: str | None = None,
        conversation: ConversationTransportOutcome | None = None,
        pricing_evidence: UsagePricingEvidence | None = None,
    ) -> InvocationResult:
        """Build one normalized logical-invocation terminal envelope."""

        trace_events = [
            event
            for event in (events or ())
            if event.kind != "terminal"
        ]
        # Re-number after defensively removing physical-attempt terminals.
        normalized: list[BackendExecutionEvent] = []
        for event in trace_events:
            _append_event(
                normalized,
                event.kind,
                name=event.name,
                parent_id=event.parent_id,
                child_id=event.child_id,
            )
        if not any(event.kind == "request" for event in normalized):
            _append_event(normalized, "request", name="claude_agent_sdk")
        _append_event(normalized, "terminal", name=stop_reason)

        if trace_template is None:
            from .features import feature_profile_digest

            trace_template = BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                served_model=served_model,
                requested_effort=self._effort,
                # Neither Claude transport echoes the provider-applied effort.
                served_effort=None,
                endpoint=self._trace_endpoint_identity(),
                provider_request_id=provider_request_id,
                feature_profile_digest=feature_profile_digest(self.features),
                tool_authority_digest=tool_authority_digest,
            )
        trace = _replace(trace_template, events=tuple(normalized))
        return InvocationResult(
            final_message=final_message,
            transcript=transcript,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            completeness=ExecutionCompleteness(
                token_usage_complete=token_usage_complete,
                cost_complete=cost_complete,
                terminal_complete=terminal_complete,
                # The normalized adapter lifecycle is complete even when a
                # provider ResultMessage is absent. terminal_complete carries
                # that provider-evidence distinction independently.
                trace_complete=True,
            ),
            execution_trace=trace,
            conversation=conversation,
            pricing_evidence=pricing_evidence,
        )

    def _finalize_attempts(
        self,
        result: InvocationResult,
        *,
        events: list[BackendExecutionEvent],
        prior_usage: Usage,
        prior_token_usage_complete: bool,
        prior_cost_complete: bool,
        prior_pricing_evidence: UsagePricingEvidence | None,
    ) -> InvocationResult:
        """Merge physical-attempt evidence into one logical result."""

        _merge_nonterminal_events(
            events,
            result,
            request_name="claude_agent_sdk",
        )
        error_kind = getattr(result.error, "kind", None)
        if (
            error_kind == "budget_exhausted"
            and not any(
                event.kind == "limit" and event.name == "budget_usd"
                for event in events
            )
        ):
            _append_event(events, "limit", name="budget_usd")
        elif (
            error_kind == "max_turns"
            and not any(
                event.kind == "limit" and event.name == "max_turns"
                for event in events
            )
        ):
            _append_event(events, "limit", name="max_turns")
        elif (
            error_kind == "cancellation"
            and not any(event.kind == "cancellation" for event in events)
        ):
            _append_event(events, "cancellation", name="external")
        if (
            result.stop_reason == "timeout"
            and not any(
                event.kind == "limit" and event.name == "deadline"
                for event in events
            )
        ):
            _append_event(events, "limit", name="deadline")

        usage = _usage_sum(prior_usage, result.usage)
        pricing_evidence = (
            result.pricing_evidence
            if prior_pricing_evidence is None and prior_usage == Usage()
            else _combine_pricing_evidence(
                usage,
                prior_pricing_evidence,
                result.pricing_evidence,
            )
        )
        return self._result_with_evidence(
            final_message=result.final_message,
            transcript=result.transcript,
            usage=usage,
            stop_reason=result.stop_reason,
            error=result.error,
            token_usage_complete=(
                prior_token_usage_complete
                and result.completeness.token_usage_complete
            ),
            cost_complete=(
                prior_cost_complete and result.completeness.cost_complete
            ),
            terminal_complete=result.completeness.terminal_complete,
            events=events,
            trace_template=result.execution_trace,
            tool_authority_digest=(
                result.execution_trace.tool_authority_digest
                if result.execution_trace is not None
                else None
            ),
            conversation=result.conversation,
            pricing_evidence=pricing_evidence,
        )

    def _override_with_terminal_dispatch(
        self,
        invocation: Invocation,
        policy: BackendPolicy,
        result: InvocationResult,
        terminal_error: BaseException,
    ) -> InvocationResult:
        """Make a settled late MCP terminal authoritative over provider output."""

        from crucible.tools._dispatch_budget import (  # noqa: PLC0415
            TerminalToolDispatchError,
            ToolDispatchBudgetExceeded,
            ToolMutationIndeterminate,
        )

        if not isinstance(terminal_error, TerminalToolDispatchError):
            return result
        mutation_indeterminate = isinstance(
            terminal_error,
            ToolMutationIndeterminate,
        )
        budget_exhausted = isinstance(
            terminal_error,
            ToolDispatchBudgetExceeded,
        )
        events: list[BackendExecutionEvent] = []
        _merge_nonterminal_events(
            events,
            result,
            request_name="claude_agent_sdk",
        )
        if budget_exhausted:
            _append_event(events, "limit", name="tool_dispatch_budget")
        classified = self._classify_exception(terminal_error, policy)
        if classified is None:  # pragma: no cover - terminal invariant
            raise AssertionError("terminal tool dispatch was not classified")
        return self._result_with_evidence(
            final_message="",
            transcript=result.transcript,
            usage=result.usage,
            stop_reason=(
                "tool_mutation_indeterminate"
                if mutation_indeterminate
                else (
                    "tool_dispatch_budget_exhausted"
                    if budget_exhausted
                    else "tool_dispatch_terminal"
                )
            ),
            error=classified,
            token_usage_complete=result.completeness.token_usage_complete,
            cost_complete=result.completeness.cost_complete,
            terminal_complete=result.completeness.terminal_complete,
            events=events,
            trace_template=result.execution_trace,
            tool_authority_digest=(
                result.execution_trace.tool_authority_digest
                if result.execution_trace is not None
                else None
            ),
            conversation=result.conversation,
            pricing_evidence=result.pricing_evidence,
        )

    async def _invoke_with_retries(
        self,
        invocation: Invocation,
    ) -> InvocationResult:
        policy = invocation.policy or _DEFAULT_POLICY
        timeout_seconds = _validated_timeout_seconds(policy)
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + timeout_seconds
        )
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("ClaudeBackend is closed")
            attachment = self._mcp_attachment
        mutation_host: Any | None = None
        hosted_worker: Any | None = None
        if (
            attachment is not None
            and getattr(attachment["toolset"], "write_names", ())
        ):
            hosted_toolset, mutation_host = _process_hosted_toolset(
                attachment["toolset"],
                invocation_deadline=deadline,
                register_host=self._register_mutation_host,
                unregister_host=self._unregister_mutation_host,
                ensure_open=self._ensure_lifecycle_open,
            )
            from crucible.tools._mcp_bridge import (  # noqa: PLC0415
                build_mcp_server,
            )

            try:
                (
                    server,
                    tool_names,
                    hosted_worker,
                    budget_tracker,
                ) = build_mcp_server(
                    hosted_toolset,
                    context_budget=attachment["context_budget"],
                    max_workers=attachment["max_workers"],
                )
                self._register_dispatch_worker(hosted_worker)
            except BaseException:
                mutation_host.stop()
                self._unregister_mutation_host(mutation_host)
                raise
            attachment = {
                **attachment,
                "server": server,
                "tool_names": tool_names,
                "worker": hosted_worker,
                "budget_tracker": budget_tracker,
                "toolset": hosted_toolset,
            }
        if attachment is not None:
            tracker = attachment.get("budget_tracker")
            if tracker is not None:
                # One reset per logical invocation; physical SDK reconnects
                # must not replenish the cumulative context allowance.
                tracker.reset()
            worker = attachment.get("worker")
            reset_terminal_denial = getattr(
                worker,
                "reset_terminal_denial",
                None,
            )
            if callable(reset_terminal_denial):
                # Like the context budget, this signal spans every physical
                # retry in one logical invocation.
                reset_terminal_denial()
        result: InvocationResult | None = None
        provider_error: BaseException | None = None
        late_terminal: BaseException | None = None
        try:
            try:
                result = await self._invoke_with_retries_core(
                    invocation,
                    policy=policy,
                    timeout_seconds=timeout_seconds,
                    deadline=deadline,
                    mcp_attachment=attachment,
                    mutation_host=mutation_host,
                )
            except BaseException as exc:
                provider_error = exc
        finally:
            cleanup_error: BaseException | None = None
            worker_settled = True
            if hosted_worker is not None:
                try:
                    hosted_worker.shutdown()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                    worker_settled = False
                else:
                    self._unregister_dispatch_worker(hosted_worker)
            if mutation_host is not None and worker_settled:
                try:
                    mutation_host.stop()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                else:
                    self._unregister_mutation_host(mutation_host)
            if hosted_worker is not None:
                observed_terminal = getattr(
                    hosted_worker,
                    "terminal_denial",
                    None,
                )
                if isinstance(observed_terminal, BaseException):
                    late_terminal = observed_terminal
        if late_terminal is not None:
            from crucible.tools._dispatch_budget import (  # noqa: PLC0415
                ToolMutationIndeterminate,
            )

            if isinstance(late_terminal, ToolMutationIndeterminate):
                if (
                    provider_error is not None
                    and provider_error is not late_terminal
                ):
                    late_terminal.add_note(
                        "Claude provider execution also raised "
                        f"{type(provider_error).__name__}; the indeterminate "
                        "mutation remains primary."
                    )
                if cleanup_error is not None:
                    late_terminal.add_note(
                        "Claude retained invocation-scoped cleanup authority; "
                        "ClaudeBackend.close() must settle the exact registered "
                        "dispatch worker and mutation host before journal "
                        "rollback."
                    )
                    if cleanup_error is late_terminal:
                        raise late_terminal
                    raise late_terminal from cleanup_error
                if provider_error is not None:
                    if provider_error is late_terminal:
                        raise late_terminal
                    raise late_terminal from provider_error
        if cleanup_error is not None:
            if provider_error is not None:
                raise cleanup_error from provider_error
            raise cleanup_error
        if provider_error is not None:
            raise provider_error
        if late_terminal is not None:
            assert result is not None
            result = self._override_with_terminal_dispatch(
                invocation,
                policy,
                result,
                late_terminal,
            )
        assert result is not None
        return result

    async def _invoke_with_retries_core(
        self,
        invocation: Invocation,
        *,
        policy: BackendPolicy,
        timeout_seconds: float | None,
        deadline: float | None,
        mcp_attachment: dict[str, Any] | None,
        mutation_host: Any | None,
    ) -> InvocationResult:
        conversation_request = invocation.conversation
        if (
            conversation_request is not None
            and conversation_request.mode == "none"
        ):
            raise NotImplementedError(
                "ClaudeBackend cannot attach provider continuity in "
                "conversation mode 'none'"
            )
        external_session_id = (
            conversation_request.opaque_id
            if (
                conversation_request is not None
                and conversation_request.action == "reuse"
            )
            else None
        )
        # Carried across retries: the session opened by a failed attempt, so
        # a transport retry resumes it (full context) instead of restarting.
        resume_session_id: str | None = external_session_id
        conversation_session_id: str | None = external_session_id
        external_resume_accepted = False
        # True on any retry → the turn uses the task-restating recovery
        # prompt (so a seeded-fresh fallback still carries the task).
        recovery = False
        events: list[BackendExecutionEvent] = []
        prior_usage = Usage()
        prior_token_usage_complete = True
        prior_cost_complete = True
        prior_pricing_evidence: UsagePricingEvidence | None = None
        prior_turns = 0

        for attempt in range(max(1, policy.retry_budget)):
            remaining = (
                None if deadline is None else deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                _append_event(events, "limit", name="deadline")
                return self._result_with_evidence(
                    final_message="",
                    transcript=(),
                    usage=prior_usage,
                    stop_reason="timeout",
                    error=TransportError(
                        http_status=None,
                        message=(
                            "Claude invocation exhausted its cumulative wall "
                            "deadline"
                        ),
                        retriable=False,
                    ),
                    token_usage_complete=prior_token_usage_complete,
                    cost_complete=prior_cost_complete,
                    terminal_complete=False,
                    events=events,
                    pricing_evidence=prior_pricing_evidence,
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=conversation_session_id,
                    ),
                )
            remaining_budget = (
                None
                if invocation.budget_usd is None
                else max(0.0, invocation.budget_usd - prior_usage.cost_usd)
            )
            if remaining_budget is not None and remaining_budget <= 0:
                _append_event(events, "limit", name="budget_usd")
                return self._result_with_evidence(
                    final_message="",
                    transcript=(),
                    usage=prior_usage,
                    stop_reason="budget_exhausted",
                    error=BudgetExhausted(
                        spent_usd=prior_usage.cost_usd,
                        limit_usd=invocation.budget_usd or 0.0,
                        message=(
                            f"Invocation spent ${prior_usage.cost_usd:.4f} "
                            "before a retry could start"
                        ),
                    ),
                    token_usage_complete=prior_token_usage_complete,
                    cost_complete=prior_cost_complete,
                    terminal_complete=False,
                    events=events,
                    pricing_evidence=prior_pricing_evidence,
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=conversation_session_id,
                    ),
                )
            remaining_turns = invocation.max_turns - prior_turns
            if remaining_turns <= 0:
                _append_event(events, "limit", name="max_turns")
                return self._result_with_evidence(
                    final_message="",
                    transcript=(),
                    usage=prior_usage,
                    stop_reason="max_turns",
                    error=MaxTurnsExceeded(
                        turns_used=prior_turns,
                        turns_limit=invocation.max_turns,
                        message=(
                            f"logical invocation exhausted max_turns="
                            f"{invocation.max_turns} before retry"
                        ),
                    ),
                    token_usage_complete=prior_token_usage_complete,
                    cost_complete=prior_cost_complete,
                    terminal_complete=False,
                    events=events,
                    pricing_evidence=prior_pricing_evidence,
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=conversation_session_id,
                    ),
                )
            started_iso = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds")
            start_mono = time.monotonic()
            result: InvocationResult | None = None
            traceback_str: str | None = None
            session_id_holder: list[str] = []
            # Whether THIS attempt tried to resume — if a resume attempt
            # fails, the next retry drops resume (the session may be gone)
            # and falls back to a fresh, context-seeded start.
            attempt_resumed = resume_session_id is not None
            external_resume = (
                external_session_id is not None
                and not external_resume_accepted
                and resume_session_id == external_session_id
            )
            self._partial_usage = None
            self._partial_pricing_evidence = None
            self._last_num_turns = None
            self._partial_turns = 0

            try:
                provider_budget_usd = remaining_budget
                if resume_session_id is not None:
                    # Claude Code restores the session-global spend ledger on
                    # resume.  For an adapter-owned retry, the original
                    # logical budget is therefore the correct cumulative CLI
                    # ceiling; subtracting prior spend a second time would
                    # stop early.  A workflow-supplied session has an unknown
                    # historical baseline, so no honest per-invocation CLI
                    # ceiling can be derived and the post-call evidence stays
                    # explicitly incomplete instead.
                    provider_budget_usd = (
                        invocation.budget_usd
                        if external_session_id is None
                        else None
                    )
                result = await _await_with_deadline(
                    self._invoke_once(
                        invocation, policy,
                        resume_session_id=resume_session_id,
                        session_id_holder=session_id_holder,
                        external_resume=external_resume,
                        recovery=recovery,
                        budget_usd_override=provider_budget_usd,
                        max_turns_override=remaining_turns,
                        mcp_attachment_override=mcp_attachment,
                    ),
                    remaining,
                )
            except _InvocationDeadlineExceeded:
                # Attribute the cost the cancelled call already incurred.
                # ``_invoke_once`` publishes per-turn usage to
                # ``self._partial_usage``; the authoritative final-turn
                # ``ResultMessage`` never arrives when ``wait_for``
                # cancels mid-stream, so without this the timed-out call
                # reports ``cost_usd=0`` despite burning the full wall
                # budget of tool-loop tokens.
                partial = self._partial_usage or Usage()
                logical_partial = _usage_sum(prior_usage, partial)
                err = TransportError(
                    http_status=None,
                    message=(
                        "Claude wall timeout exhausted its cumulative "
                        f"deadline after {timeout_seconds}s"
                    ),
                    retriable=False,
                )
                logger.warning(
                    "ClaudeBackend timeout on attempt %d: %s "
                    "(partial usage: %d in / %d out tokens, $%.4f)",
                    attempt + 1, err.message,
                    partial.input_tokens, partial.output_tokens,
                    partial.cost_usd,
                )
                _append_event(
                    events,
                    "request",
                    name=f"claude_agent_sdk_attempt_{attempt + 1}",
                )
                if (
                    partial.input_tokens
                    or partial.output_tokens
                    or partial.cost_usd
                ):
                    _append_event(events, "usage", name="partial")
                _append_event(events, "limit", name="deadline")
                if session_id_holder:
                    conversation_session_id = session_id_holder[0]
                    if external_session_id is not None:
                        external_resume_accepted = True
                result = self._result_with_evidence(
                    final_message="",
                    transcript=(),
                    usage=logical_partial,
                    stop_reason="timeout",
                    error=err,
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=False,
                    events=events,
                    pricing_evidence=_combine_pricing_evidence(
                        logical_partial,
                        prior_pricing_evidence,
                        self._partial_pricing_evidence,
                    ),
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=conversation_session_id,
                    ),
                )
                self._sdk_sink.dump(
                    invocation=invocation, result=result,
                    started_iso=started_iso,
                    duration_s=time.monotonic() - start_mono,
                    attempt=attempt,
                )
                return result
            except BaseException as exc:  # noqa: BLE001 — we classify below
                traceback_str = _tb.format_exc()
                err = self._classify_exception(exc, policy)
                if err is None:
                    # Unclassified — dump the traceback for postmortem
                    # then propagate. Workflow's outer catch (if any)
                    # sees the original exception.
                    self._sdk_sink.dump(
                        invocation=invocation, result=None,
                        started_iso=started_iso,
                        duration_s=time.monotonic() - start_mono,
                        traceback_str=traceback_str,
                        re_raised=True,
                        attempt=attempt,
                    )
                    raise
                partial = self._partial_usage or Usage()
                local_events: list[BackendExecutionEvent] = []
                _append_event(
                    local_events,
                    "request",
                    name=f"claude_agent_sdk_attempt_{attempt + 1}",
                )
                if (
                    partial.input_tokens
                    or partial.output_tokens
                    or partial.cost_usd
                ):
                    _append_event(local_events, "usage", name="partial")
                if getattr(err, "kind", None) == "cancellation":
                    _append_event(
                        local_events,
                        "cancellation",
                        name="external",
                    )
                result = self._result_with_evidence(
                    final_message="",
                    transcript=(),
                    usage=partial,
                    stop_reason=(
                        "cancelled"
                        if getattr(err, "kind", None) == "cancellation"
                        else "error"
                    ),
                    error=err,
                    terminal_complete=False,
                    events=local_events,
                    pricing_evidence=self._partial_pricing_evidence,
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=(
                            session_id_holder[0]
                            if session_id_holder
                            else conversation_session_id
                        ),
                    ),
                )

            if session_id_holder:
                conversation_session_id = session_id_holder[0]
                if external_session_id is not None:
                    external_resume_accepted = True

            # _invoke_once may package a retriable provider error in the
            # result. A second physical query is admissible only when the
            # prior query supplied exact token and cost envelopes.
            self._sdk_sink.dump(
                invocation=invocation, result=result,
                started_iso=started_iso,
                duration_s=time.monotonic() - start_mono,
                traceback_str=traceback_str,
                attempt=attempt,
            )
            attempt_turns = (
                self._last_num_turns
                if self._last_num_turns is not None
                else self._partial_turns
            )
            retry_requested = (
                result.error is not None
                and getattr(result.error, "retriable", False)
                and getattr(result.error, "kind", None) != "cancellation"
            )
            retry_available = (
                attempt + 1 < policy.retry_budget
                and prior_turns + max(0, attempt_turns) < invocation.max_turns
            )
            if (
                retry_requested
                and retry_available
                and mutation_host is not None
                and mutation_host.ever_started_write
            ):
                result = _replace(
                    result,
                    error=_replace(
                        result.error,
                        message=(
                            f"{result.error.message}; automatic retry "
                            "suppressed after a mutating tool call"
                        ),
                        retriable=False,
                    ),
                )
                retry_requested = False
            if retry_requested and not (
                prior_token_usage_complete
                and prior_cost_complete
                and result.completeness.token_usage_complete
                and result.completeness.cost_complete
            ):
                result = _replace(
                    result,
                    error=_replace(
                        result.error,
                        message=(
                            f"{result.error.message}; automatic retry "
                            "suppressed because the prior physical attempt's "
                            "token/cost accounting is incomplete"
                        ),
                        retriable=False,
                    ),
                )
                retry_requested = False
            retriable = retry_requested and retry_available
            if retriable:
                _merge_nonterminal_events(
                    events,
                    result,
                    request_name=f"claude_agent_sdk_attempt_{attempt + 1}",
                )
                next_prior_usage = _usage_sum(prior_usage, result.usage)
                prior_pricing_evidence = (
                    result.pricing_evidence
                    if prior_pricing_evidence is None and prior_usage == Usage()
                    else _combine_pricing_evidence(
                        next_prior_usage,
                        prior_pricing_evidence,
                        result.pricing_evidence,
                    )
                )
                prior_usage = next_prior_usage
                prior_token_usage_complete = (
                    prior_token_usage_complete
                    and result.completeness.token_usage_complete
                )
                prior_cost_complete = (
                    prior_cost_complete
                    and result.completeness.cost_complete
                )
                # ``num_turns`` bounds the agent loop only; it is not a
                # physical-request counter. Resumed CLI model_usage can also
                # be cumulative, so its pricing evidence remains incomplete
                # unless fresh visible response segments reconcile exactly.
                prior_turns += max(0, attempt_turns)
                recovery = True
                if attempt_resumed:
                    # The resume attempt failed — the session may be gone.
                    # Drop it; the next retry is a fresh, context-seeded
                    # start (recovery prompt restates the task).
                    if external_session_id is None:
                        resume_session_id = None
                elif session_id_holder:
                    resume_session_id = session_id_holder[0]
                backoff = min(2 ** attempt, 16) + (time.monotonic() % 1.0)
                _append_event(
                    events,
                    "retry",
                    name=getattr(
                        result.error,
                        "kind",
                        type(result.error).__name__,
                    ),
                )
                logger.info(
                    "ClaudeBackend retriable error in result on attempt "
                    "%d (%s); retrying in %.1fs%s",
                    attempt + 1,
                    getattr(result.error, "kind", type(result.error).__name__),
                    backoff,
                    f" (resuming session {resume_session_id})" if resume_session_id else "",
                )
                try:
                    if deadline is None:
                        await asyncio.sleep(backoff)
                    else:
                        backoff_remaining = deadline - time.monotonic()
                        if backoff_remaining <= 0:
                            raise asyncio.TimeoutError
                        await asyncio.wait_for(
                            asyncio.sleep(backoff),
                            timeout=backoff_remaining,
                        )
                except asyncio.TimeoutError:
                    _append_event(events, "limit", name="deadline")
                    return self._result_with_evidence(
                        final_message="",
                        transcript=(),
                        usage=prior_usage,
                        stop_reason="timeout",
                        error=TransportError(
                            http_status=None,
                            message=(
                                "Claude retry backoff exhausted the cumulative "
                                "invocation wall deadline"
                            ),
                            retriable=False,
                        ),
                        token_usage_complete=prior_token_usage_complete,
                        cost_complete=prior_cost_complete,
                        terminal_complete=False,
                        events=events,
                        pricing_evidence=prior_pricing_evidence,
                        conversation=_conversation_outcome(
                            invocation,
                            opaque_id=conversation_session_id,
                        ),
                    )
                except asyncio.CancelledError as exc:
                    if policy.on_cancellation == "raise":
                        raise
                    _append_event(events, "cancellation", name="external")
                    return self._result_with_evidence(
                        final_message="",
                        transcript=(),
                        usage=prior_usage,
                        stop_reason="cancelled",
                        error=ExternalCancellation(
                            cause=type(exc).__name__,
                            message=str(exc),
                            retriable=True,
                        ),
                        token_usage_complete=prior_token_usage_complete,
                        cost_complete=prior_cost_complete,
                        terminal_complete=False,
                        events=events,
                        pricing_evidence=prior_pricing_evidence,
                        conversation=_conversation_outcome(
                            invocation,
                            opaque_id=conversation_session_id,
                        ),
                    )
                continue

            logical_usage = _usage_sum(prior_usage, result.usage)
            logical_turns = prior_turns + max(0, attempt_turns)
            if (
                invocation.budget_usd is not None
                and logical_usage.cost_usd > invocation.budget_usd
            ):
                result = _replace(
                    result,
                    stop_reason="budget_exhausted",
                    error=BudgetExhausted(
                        spent_usd=logical_usage.cost_usd,
                        limit_usd=invocation.budget_usd,
                        message=(
                            f"Logical invocation spent "
                            f"${logical_usage.cost_usd:.4f} exceeding budget "
                            f"${invocation.budget_usd:.4f}"
                        ),
                    ),
                )
            # Claude's canonical stop reason is the terminal authority; its
            # reported ``num_turns`` is accounting evidence, not a second
            # terminal classifier.  Cumulative retry admission above still
            # prevents another physical query when no turn authority remains.
            # Do not rewrite an accepted provider terminal from its count.
            elif getattr(result.error, "kind", None) == "max_turns":
                result = _replace(
                    result,
                    error=MaxTurnsExceeded(
                        turns_used=logical_turns,
                        turns_limit=invocation.max_turns,
                        message=f"hit max_turns={invocation.max_turns}",
                    ),
                )
            return self._finalize_attempts(
                result,
                events=events,
                prior_usage=prior_usage,
                prior_token_usage_complete=prior_token_usage_complete,
                prior_cost_complete=prior_cost_complete,
                prior_pricing_evidence=prior_pricing_evidence,
            )

        # Loop exited without returning — shouldn't happen with retry_budget>=1.
        fallback = self._result_with_evidence(
            final_message="",
            transcript=(),
            usage=prior_usage,
            stop_reason="error",
            error=TransportError(
                message="retry loop exhausted with no result",
                retriable=False,
            ),
            token_usage_complete=prior_token_usage_complete,
            cost_complete=prior_cost_complete,
            terminal_complete=False,
            events=events,
            pricing_evidence=prior_pricing_evidence,
            conversation=_conversation_outcome(
                invocation,
                opaque_id=conversation_session_id,
            ),
        )
        return fallback

    # ------------------------------------------------------------------
    # Single SDK invocation
    # ------------------------------------------------------------------

    async def _invoke_once(
        self, invocation: Invocation, policy: BackendPolicy,
        *,
        resume_session_id: str | None = None,
        session_id_holder: list[str] | None = None,
        external_resume: bool = False,
        recovery: bool = False,
        budget_usd_override: float | None | object = _BUDGET_OVERRIDE_UNSET,
        max_turns_override: int | None = None,
        mcp_attachment_override: Any = _MCP_ATTACHMENT_UNSET,
    ) -> InvocationResult:
        """One pass through claude_agent_sdk.query.

        ``resume_session_id``: when set (a transport-retry after a prior
        attempt opened a session), RESUME that session via
        ``ClaudeAgentOptions(resume=...)`` instead of starting fresh. The
        bundled CLI persists each session, so a reconnected query picks the
        conversation back up with full context — exploration, reasoning,
        edits already on disk — intact, rather than re-exploring from zero.

        ``session_id_holder``: a mutable sink the caller passes so the retry
        loop learns the session id even when the turn later fails — it's
        captured from the early ``SystemMessage(subtype="init")``, which
        arrives before any work, so even a near-immediate transport drop
        leaves a session to resume.

        ``external_resume``: distinguishes a workflow-supplied opaque ID from
        an adapter-internal transport retry. An explicit provider missing-
        session response becomes a typed stale outcome instead of a generic
        transport retry or silent fresh start."""
        from claude_agent_sdk import ClaudeAgentOptions, query
        from claude_agent_sdk.types import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
        )

        self._validate_session_cli_path()
        mcp_attachment = (
            self._mcp_attachment
            if mcp_attachment_override is _MCP_ATTACHMENT_UNSET
            else mcp_attachment_override
        )

        if mcp_attachment is not None:
            tool_authority_digest = self._tool_authority_digest(
                mcp_attachment["toolset"],
                context_budget=mcp_attachment["context_budget"],
            )
        else:
            tool_authority_digest = None

        # Translate the assay.Invocation into SDK options. Gateway mapping
        # lives in _build_sdk_env (module-level so tests can pin it).
        sdk_env = _build_sdk_env(
            prefer_infer=self._prefer_infer_credentials,
            credential_source=self._credential_source,
        )

        # Tool availability is enforced below.  Analytical roles never use
        # bypassPermissions; write roles may auto-accept only their explicitly
        # allowed, workspace-scoped surface.
        perm_mode = (
            "bypassPermissions"
            if self.capability == "workspace_write" and os.getuid() != 0
            else "auto"
        )

        # Capture the bundled CLI's stderr so transport-level crashes
        # (e.g. ``Command failed with exit code 1`` from the
        # claude_agent_sdk transport) include the actual error reason
        # instead of the unhelpful "Check stderr output for details"
        # placeholder. Without this callback the SDK silently drops the
        # subprocess's stderr; we'd see ``re_raised: True`` in the
        # debug dump with no diagnostic context. Bounded so that a chatty
        # CLI cannot grow what one attempt keeps: the window holds the last
        # 200 entries and at most 65 536 bytes of UTF-8 over them, in whole
        # entries. For a resumed handle it reads every entry for the reason
        # of a rejected ``--resume`` as the entry arrives, so that reason is
        # found among the last 200 entries received whether or not its text
        # is kept. ``_claude_session_transport.py`` records the limits.
        cli_stderr_window = _CliStderrWindow(note_resume_markers=external_resume)

        effective_max_turns = (
            invocation.max_turns
            if max_turns_override is None
            else max_turns_override
        )
        effective_budget_usd = (
            invocation.budget_usd
            if budget_usd_override is _BUDGET_OVERRIDE_UNSET
            else cast(float | None, budget_usd_override)
        )
        options_kw: dict[str, Any] = {
            "system_prompt": invocation.system,
            "model": self.model,
            "cwd": self._cwd,
            "permission_mode": perm_mode,
            "max_turns": effective_max_turns,
            "thinking": _adaptive_thinking(),
            "effort": self._effort,
            "stderr": cli_stderr_window.add,
            # The workflow supplies the complete MCP surface explicitly.
            # Never inherit project/user/plugin MCP servers, settings, hooks,
            # skills, or plugins from the checkout or operator account.
            "strict_mcp_config": True,
            "setting_sources": [],
            "skills": [],
            "plugins": [],
        }
        if effective_budget_usd is not None:
            # Claude Code enforces this against its session-global ledger.
            # The retry wrapper passes the original logical ceiling for an
            # adapter-owned resume and omits the control for an external
            # session whose historical baseline is unavailable.
            options_kw["max_budget_usd"] = effective_budget_usd
        # Model substitution is a workflow-boundary routing decision. Never
        # configure ``fallback_model`` or smuggle ``--fallback-model`` through
        # ``extra_args``: either mechanism can switch models inside a running
        # agent conversation after tools have already mutated the workspace.
        if not self._native_shell:
            # ``allowed_tools`` controls permission, not availability. The
            # SDK's ``tools=[]`` option is the fail-closed availability gate
            # for every bundled CLI tool; attached MCP tools remain available
            # through ``mcp_servers`` below.
            options_kw["tools"] = []
        # Do not enable the Claude CLI's raw ``--debug-file``.  That file can
        # contain unredacted prompts, tool payloads, and credentials and is
        # written outside our private/redacting SdkDebugSink.  The bounded
        # stderr callback plus the sink's structured, mode-0600 diagnostic is
        # the supported postmortem surface.
        if self._credential_source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
            from ._claude_session_transport import (  # noqa: PLC0415
                CLAUDE_SESSION_REQUIRED_EXTRA_ARGS,
            )

            # Construction and the pre-transport check above make this a
            # canonical absolute executable. Never let the SDK fall back to
            # its bundled CLI for organization-session authentication.
            options_kw["cli_path"] = self._cli_path
            # Public CLI controls suppress prompt-suggestion and slash-command
            # work. The sanitized transport separately installs fixed
            # nonessential-traffic and no-model-fallback environment policy.
            options_kw["extra_args"] = dict(
                CLAUDE_SESSION_REQUIRED_EXTRA_ARGS
            )
        elif self._cli_path:
            options_kw["cli_path"] = self._cli_path
        if sdk_env:
            options_kw["env"] = sdk_env
        if invocation.response_schema is not None:
            options_kw["output_format"] = _json_output_format(
                invocation.response_schema
            )
        # Transport-retry: resume the prior session (full context) instead
        # of starting fresh.
        if resume_session_id is not None:
            options_kw["resume"] = resume_session_id

        # allowed_tools composition:
        # 1. Caller-provided ``Invocation.tools`` names go in first.
        # 2. If a toolset is attached, append ``mcp__crucible__<name>``
        #    entries (and wire ``mcp_servers["crucible"]`` to the in-
        #    process MCP server).
        # Empty list = the SDK applies its default allow-list; an
        # explicit list constrains the bundled CLI to only those names.
        allowed_tools: list[str] = []
        if invocation.tools:
            requested = [t.name for t in invocation.tools]
            if self.capability == "text":
                raise ValueError(
                    "text capability received built-in tools: "
                    + ", ".join(requested)
                )
            if self.capability == "read_only":
                forbidden = sorted(
                    set(requested) - {"Read", "Grep", "Glob"}
                )
                if forbidden:
                    raise ValueError(
                        "read_only capability received mutating/exec tools: "
                        + ", ".join(forbidden)
                    )
            allowed_tools.extend(requested)
        if mcp_attachment is not None:
            allowed_tools.extend(mcp_attachment["tool_names"])
            options_kw["mcp_servers"] = {
                "crucible": mcp_attachment["server"],
            }
        if allowed_tools:
            options_kw["allowed_tools"] = allowed_tools

        # Disallow the bundled CLI's recursive sub-agent spawner.
        # Crucible orchestrates its own agents through the workflow
        # layer; the CLI's built-in ``Task``/``Agent`` tool (which
        # launches ``agent:builtin:Explore`` and friends) is never part
        # of that design and is pure liability here:
        #   * the spawned sub-agent defaults to the small/fast (haiku)
        #     tier, which some gateways forbid outright (NVIDIA
        #     inference-api returns 401 ``key_model_access_denied`` for
        #     haiku) — the CLI then 401-retries 11x and the subprocess
        #     dies with ``exit code 1``;
        #   * even when the fast tier is reachable, an auto-spawned
        #     Explore sub-agent balloons the parent turn's context and
        #     output, which on CLI 2.1.158 produces a second exit-1
        #     transport death on the oversized streamed turn.
        # NOTE: ``allowed_tools`` is a *permission* gate, not an
        # *availability* gate — narrowing it does NOT remove the
        # spawner (verified: the CLI still dispatches ``Agent`` when it
        # is merely absent from ``allowed_tools``). ``disallowed_tools``
        # is the only lever that actually suppresses the spawn.
        disallowed = ["Task", "Agent"]

        # Defense in depth for CLI versions that do not honor ``tools=[]``
        # uniformly. Confined roles remove every known host-native read,
        # mutation, execution, and web tool even when MCP tools are attached.
        if not self._native_shell:
            disallowed += [
                "Read", "Grep", "Glob", "Bash",
                "Edit", "Write", "NotebookEdit",
                "WebFetch", "WebSearch",
            ]

        options_kw["disallowed_tools"] = disallowed

        from .pricing import compute_cost, pricing_for_model

        transcript: list[Message] = []
        final_message = ""
        usage = Usage()
        stop_reason = "end_turn"

        # Per-turn cost accumulation so a wall-timeout (which cancels
        # this coroutine before the authoritative ``ResultMessage``
        # arrives) can still attribute the partial cost. Published to
        # ``self._partial_usage`` after each turn; read by the timeout
        # handler in ``_invoke_with_retries``.
        self._partial_usage = None
        self._partial_pricing_evidence = None
        self._last_num_turns = None
        self._partial_turns = 0
        partial_in = 0
        partial_out = 0
        partial_cache_create = 0
        partial_cache_read = 0
        partial_cost = 0.0
        assistant_pricing_segments: list[UsagePricingSegment] = []
        assistant_pricing_stream_complete = True
        admitted_pre_api_error_assistant_count = 0
        admitted_pre_api_error_usage_seen = False
        synthetic_api_error_marker_count = 0
        synthetic_api_error_marker_usage_zero = True
        pricing_is_flat = (
            pricing_for_model(self.model).long_context_threshold_tokens
            is None
        )
        observed_served_models: set[str] = set()
        served_model: str | None = None
        result_seen = False
        usage_complete = False
        cost_complete = False
        pricing_evidence: UsagePricingEvidence | None = None
        result_error: TransportError | BudgetExhausted | None = None
        pending_api_error_category: str | None = None
        pending_api_error_conflict = False
        pending_api_error_stream_conflict = False
        terminal_fields_complete = True
        provider_request_id: str | None = None
        provider_conversation_id: str | None = resume_session_id
        result_count = 0
        events: list[BackendExecutionEvent] = []
        _append_event(events, "request", name="claude_agent_sdk")

        # A recovery turn (resumed session OR seeded-fresh fallback) gets the
        # task-restating nudge so the model always has the task — even if the
        # prior turn never persisted it, or the session couldn't be resumed
        # and this is a fresh, context-seeded restart. First attempt: verbatim.
        turn_prompt = (
            _CLAUDE_RESUME_NUDGE.format(task=invocation.user) if recovery
            else invocation.user
        )
        sdk_options: Any
        session_transport = None
        if self._credential_source == CLAUDE_SESSION_CREDENTIAL_SOURCE:
            if (
                self._session_cli_pin is None
                or self._expected_org_id is None
            ):  # pragma: no cover - constructor invariant
                raise ValueError(
                    "Claude organization-session transport is not fully bound"
                )
            from ._claude_session_transport import (  # noqa: PLC0415
                FrozenClaudeSessionOptions,
                SanitizedClaudeSessionTransport,
            )

            # Keep both SDK control planes immutable. The SDK re-reads this
            # query-side object after the transport connects, while the
            # transport independently clones and freezes its command-side
            # object before taking an argv snapshot.
            sdk_options = FrozenClaudeSessionOptions.from_kwargs(**options_kw)

            # A fresh transport is constructed for every physical retry.  It
            # snapshots the operational environment and performs the pinned
            # auth/version probes before starting the model process.
            session_transport = SanitizedClaudeSessionTransport(
                prompt=turn_prompt,
                options=sdk_options,
                cli=self._session_cli_pin,
                expected_org_id=self._expected_org_id,
                expected_org_name=self._expected_org_name,
            )
        else:
            sdk_options = ClaudeAgentOptions(**options_kw)
        sdk_messages = None
        message_stream = None
        try:
            sdk_messages = (
                _query_claude_session_with_model_preflight(
                    prompt=turn_prompt,
                    options=sdk_options,
                    transport=session_transport,
                )
                if session_transport is not None
                else query(prompt=turn_prompt, options=sdk_options)
            )
            denial_worker = (
                mcp_attachment.get("worker")
                if mcp_attachment is not None
                else None
            )
            message_stream = _iterate_until_tool_dispatch_denial(
                sdk_messages,
                denial_worker,
            )
            async for sdk_msg in message_stream:
                # Capture the session id ASAP (the init SystemMessage
                # arrives before any work) so a mid-turn transport drop can
                # still resume this exact session.
                if session_id_holder is not None and not session_id_holder:
                    sid = None
                    if isinstance(sdk_msg, SystemMessage):
                        data = getattr(sdk_msg, "data", None)
                        if isinstance(data, dict):
                            sid = data.get("session_id")
                    sid = sid or getattr(sdk_msg, "session_id", None)
                    if sid:
                        provider_conversation_id = str(sid)
                        session_id_holder.append(provider_conversation_id)
                if isinstance(sdk_msg, AssistantMessage):
                    self._partial_turns += 1
                    prior_api_error_category = pending_api_error_category
                    raw_api_error_category = getattr(sdk_msg, "error", None)
                    api_error_category_extensions = (
                        _claude_session_api_error_category_extensions(
                            requested_model=self.model,
                            validated_cli_version=(
                                session_transport.validated_cli_version
                                if session_transport is not None
                                else None
                            ),
                        )
                    )
                    reported_api_error_category = (
                        _claude_agent_api_error_category(
                            sdk_msg,
                            extensions=api_error_category_extensions,
                        )
                    )
                    invalid_api_error_category = (
                        raw_api_error_category is not None
                        and reported_api_error_category is None
                    )
                    api_error_category = (
                        _synthetic_claude_api_error_category(
                            sdk_msg,
                            extensions=api_error_category_extensions,
                        )
                    )
                    misbound_api_error_category = (
                        reported_api_error_category
                        if api_error_category is None
                        else None
                    )
                    if api_error_category is not None:
                        if pending_api_error_category is None:
                            pending_api_error_category = api_error_category
                            # An API error invalidates all content in the same
                            # physical attempt. Nothing from that attempt can
                            # become the logical reply.
                            transcript.clear()
                            final_message = ""
                            observed_served_models.clear()
                            served_model = None
                        elif api_error_category != pending_api_error_category:
                            pending_api_error_conflict = True
                    if (
                        prior_api_error_category is not None
                        and api_error_category != prior_api_error_category
                    ):
                        # After the first exact marker, only an exact duplicate
                        # marker is an admissible AssistantMessage shape.  A
                        # normal model, misbound/unknown category, or malformed
                        # model/error field cannot hide behind sticky content
                        # suppression and a later nominally valid terminal.
                        pending_api_error_stream_conflict = True
                    if pending_api_error_category is None:
                        observed_model = _record_served_model(
                            self.model,
                            getattr(sdk_msg, "model", None),
                            transport="agent_sdk",
                            observed=observed_served_models,
                            exact=(
                                self._credential_source
                                == CLAUDE_SESSION_CREDENTIAL_SOURCE
                            ),
                        )
                        if observed_model is not None:
                            served_model = observed_model
                        model_identity_error = (
                            _missing_served_model_identity()
                            if observed_model is None
                            else _served_model_mismatch(
                                self.model,
                                observed_model,
                                transport="agent_sdk",
                                exact=(
                                    self._credential_source
                                    == CLAUDE_SESSION_CREDENTIAL_SOURCE
                                ),
                            )
                        )
                    else:
                        # Claude Code did not serve the marker's ``<synthetic>``
                        # spelling as a model. Once that typed marker is
                        # pending, no later AssistantMessage in the same
                        # physical attempt may reopen content or model
                        # admission. Keep reading only so terminal status and
                        # accounting can decide the outcome. Any absent or
                        # contradictory terminal remains incomplete below.
                        observed_model = None
                        model_identity_error = None
                    turn_usage = getattr(sdk_msg, "usage", None)
                    turn_pricing_segment: UsagePricingSegment | None = None
                    if turn_usage is not None:
                        counters = {
                            name: _usage_value(turn_usage, name)
                            for name in (
                                "input_tokens",
                                "output_tokens",
                                "cache_creation_input_tokens",
                                "cache_read_input_tokens",
                            )
                        }
                        in_t = (
                            counters["input_tokens"]
                            if _valid_token_counter(counters["input_tokens"])
                            else 0
                        )
                        out_t = (
                            counters["output_tokens"]
                            if _valid_token_counter(counters["output_tokens"])
                            else 0
                        )
                        cc_t = (
                            counters["cache_creation_input_tokens"]
                            if _valid_token_counter(
                                counters["cache_creation_input_tokens"]
                            )
                            else 0
                        )
                        cr_t = (
                            counters["cache_read_input_tokens"]
                            if _valid_token_counter(
                                counters["cache_read_input_tokens"]
                            )
                            else 0
                        )
                        turn_pricing_segment = _pricing_segment_from_usage(
                            turn_usage,
                            model=self.model,
                        )
                        if turn_pricing_segment is not None:
                            assistant_pricing_segments.append(
                                turn_pricing_segment
                            )
                        partial_in += in_t
                        partial_out += out_t
                        partial_cache_create += cc_t
                        partial_cache_read += cr_t
                        partial_cost += (
                            turn_pricing_segment.cost_usd
                            if turn_pricing_segment is not None
                            else compute_cost(
                                self.model,
                                input_tokens=in_t + cc_t + cr_t,
                                output_tokens=out_t,
                                cache_creation_tokens=cc_t,
                                cache_read_tokens=cr_t,
                            )
                        )
                        self._partial_usage = Usage(
                            input_tokens=(
                                partial_in
                                + partial_cache_create
                                + partial_cache_read
                            ),
                            output_tokens=partial_out,
                            cost_usd=partial_cost,
                            cache_creation_tokens=partial_cache_create,
                            cache_read_tokens=partial_cache_read,
                        )
                        self._partial_pricing_evidence = UsagePricingEvidence(
                            segments=tuple(assistant_pricing_segments),
                            complete=False,
                        )
                        _append_event(
                            events,
                            "usage",
                            name="assistant_partial",
                        )
                    if turn_pricing_segment is None:
                        assistant_pricing_stream_complete = False
                    if api_error_category is not None:
                        synthetic_api_error_marker_count += 1
                        if (
                            turn_pricing_segment is None
                            or turn_pricing_segment.input_tokens != 0
                            or turn_pricing_segment.output_tokens != 0
                            or turn_pricing_segment.cache_creation_tokens != 0
                            or turn_pricing_segment.cache_read_tokens != 0
                            or turn_pricing_segment.cost_usd != 0.0
                        ):
                            synthetic_api_error_marker_usage_zero = False
                    elif (
                        prior_api_error_category is None
                        and observed_model is not None
                        and model_identity_error is None
                    ):
                        # A real terminal ledger after a synthetic API-error
                        # marker is accounting evidence only when this attempt
                        # first exposed one priced response on the admitted route.
                        admitted_pre_api_error_assistant_count += 1
                        if (
                            turn_pricing_segment is not None
                            and (
                                turn_pricing_segment.input_tokens > 0
                                or turn_pricing_segment.output_tokens > 0
                            )
                        ):
                            admitted_pre_api_error_usage_seen = True
                    if pending_api_error_category is not None:
                        continue
                    if invalid_api_error_category:
                        # A present error discriminator outside the pinned
                        # SDK's closed category set is an untrusted envelope,
                        # never ordinary assistant content. Do not include the
                        # arbitrary value in evidence because it may contain
                        # provider-controlled detail.
                        return self._result_with_evidence(
                            final_message="",
                            transcript=(),
                            usage=self._partial_usage or Usage(),
                            stop_reason="incomplete_terminal",
                            error=TransportError(
                                http_status=None,
                                message=(
                                    "Claude Agent SDK AssistantMessage.error "
                                    "was present but was not one of the closed "
                                    "API-error categories"
                                ),
                                retriable=False,
                            ),
                            token_usage_complete=False,
                            cost_complete=False,
                            terminal_complete=False,
                            events=events,
                            served_model=served_model,
                            provider_request_id=provider_request_id,
                            tool_authority_digest=tool_authority_digest,
                            pricing_evidence=self._partial_pricing_evidence,
                            conversation=_conversation_outcome(
                                invocation,
                                opaque_id=provider_conversation_id,
                            ),
                        )
                    if model_identity_error is not None:
                        # The SDK owns its internal tool loop, so the first
                        # observable model identity is the earliest adapter
                        # boundary available.  Refuse the content immediately
                        # and do not retry or accept a terminal from this
                        # physical attempt.
                        _append_event(
                            events,
                            "limit",
                            name="served_model_mismatch",
                        )
                        return self._result_with_evidence(
                            final_message="",
                            transcript=(),
                            usage=self._partial_usage or Usage(),
                            stop_reason=(
                                "missing_model_identity"
                                if observed_model is None
                                else "model_mismatch"
                            ),
                            error=model_identity_error,
                            token_usage_complete=False,
                            cost_complete=False,
                            terminal_complete=False,
                            events=events,
                            served_model=served_model,
                            provider_request_id=provider_request_id,
                            tool_authority_digest=tool_authority_digest,
                            pricing_evidence=self._partial_pricing_evidence,
                            conversation=_conversation_outcome(
                                invocation,
                                opaque_id=provider_conversation_id,
                            ),
                        )
                    if misbound_api_error_category is not None:
                        # A recognized API-error category on any model other
                        # than the CLI's exact local placeholder is malformed,
                        # not normal assistant content. Do not wait for a
                        # terminal or let the model spelling make it success.
                        return self._result_with_evidence(
                            final_message="",
                            transcript=(),
                            usage=self._partial_usage or Usage(),
                            stop_reason="incomplete_terminal",
                            error=TransportError(
                                http_status=None,
                                message=(
                                    "Claude Agent SDK API-error category "
                                    f"{misbound_api_error_category!r} was not "
                                    "carried by the exact synthetic envelope"
                                ),
                                retriable=False,
                            ),
                            token_usage_complete=False,
                            cost_complete=False,
                            terminal_complete=False,
                            events=events,
                            served_model=served_model,
                            provider_request_id=provider_request_id,
                            tool_authority_digest=tool_authority_digest,
                            pricing_evidence=self._partial_pricing_evidence,
                            conversation=_conversation_outcome(
                                invocation,
                                opaque_id=provider_conversation_id,
                            ),
                        )
                    text = "".join(
                        block.text for block in sdk_msg.content
                        if isinstance(block, TextBlock)
                    )
                    if text:
                        transcript.append(Message(role="assistant", content=text))
                        final_message = text
                elif isinstance(sdk_msg, ResultMessage):
                    result_count += 1
                    result_seen = True
                    if result_count > 1:
                        terminal_fields_complete = False
                        result_error = TransportError(
                            http_status=None,
                            message=(
                                "Claude Agent SDK emitted multiple terminal "
                                "ResultMessage records"
                            ),
                            retriable=False,
                        )
                    raw_num_turns = getattr(sdk_msg, "num_turns", None)
                    if (
                        type(raw_num_turns) is int
                        and raw_num_turns >= 0
                    ):
                        self._last_num_turns = raw_num_turns
                    else:
                        terminal_fields_complete = False
                        result_error = TransportError(
                            http_status=None,
                            message=(
                                "Claude Agent SDK ResultMessage.num_turns "
                                "must be an exact non-negative integer"
                            ),
                            retriable=False,
                        )
                    raw_is_error = getattr(sdk_msg, "is_error", None)
                    if type(raw_is_error) is not bool:
                        terminal_fields_complete = False
                        result_error = TransportError(
                            http_status=None,
                            message=(
                                "Claude Agent SDK ResultMessage.is_error "
                                "must be bool"
                            ),
                            retriable=False,
                        )
                    reported_errors = getattr(sdk_msg, "errors", None)
                    if (
                        reported_errors is not None
                        and not isinstance(reported_errors, (list, tuple))
                    ):
                        terminal_fields_complete = False
                        result_error = TransportError(
                            http_status=None,
                            message=(
                                "Claude Agent SDK ResultMessage.errors must "
                                "be a sequence when present"
                            ),
                            retriable=False,
                        )
                    raw_model_usage = getattr(sdk_msg, "model_usage", None)
                    synthetic_api_error_model_usage_valid = (
                        _valid_synthetic_api_error_model_usage(raw_model_usage)
                    )
                    terminal_model_pricing = (
                        _terminal_model_usage_pricing_segment(raw_model_usage)
                    )
                    model_usage_error: IncompleteEvidence | None = None
                    terminal_served_model: str | None = None
                    if (
                        raw_model_usage is not None
                        and (
                            pending_api_error_category is None
                            or terminal_model_pricing is not None
                        )
                    ):
                        (
                            terminal_served_model,
                            model_usage_error,
                        ) = _result_model_usage_identity(
                            self.model,
                            raw_model_usage,
                            observed=(
                                observed_served_models
                                if pending_api_error_category is None
                                else set()
                            ),
                            exact=(
                                self._credential_source
                                == CLAUDE_SESSION_CREDENTIAL_SOURCE
                            ),
                        )
                        if (
                            pending_api_error_category is None
                            and terminal_served_model is not None
                        ):
                            # The terminal usage envelope is the authoritative
                            # raw spelling when it names exactly one model.
                            served_model = terminal_served_model
                    raw_usage_value = getattr(sdk_msg, "usage", None)
                    raw_usage = (
                        raw_usage_value
                        if isinstance(raw_usage_value, dict)
                        else {}
                    )
                    raw_in = raw_usage.get("input_tokens")
                    raw_out = raw_usage.get("output_tokens")
                    raw_cache_creation = raw_usage.get(
                        "cache_creation_input_tokens"
                    )
                    raw_cache_read = raw_usage.get(
                        "cache_read_input_tokens"
                    )
                    raw_usage_complete = (
                        _valid_token_counter(raw_in)
                        and _valid_token_counter(raw_out)
                        and _valid_token_counter(raw_cache_creation)
                        and _valid_token_counter(raw_cache_read)
                    )
                    ordinary_input_tokens = (
                        raw_in
                        if _valid_token_counter(raw_in)
                        else partial_in
                    )
                    out_tokens = (
                        raw_out
                        if _valid_token_counter(raw_out)
                        else partial_out
                    )
                    cache_creation_tokens = (
                        raw_cache_creation
                        if _valid_token_counter(raw_cache_creation)
                        else partial_cache_create
                    )
                    cache_read_tokens = (
                        raw_cache_read
                        if _valid_token_counter(raw_cache_read)
                        else partial_cache_read
                    )
                    inclusive_input_tokens = (
                        ordinary_input_tokens
                        + cache_creation_tokens
                        + cache_read_tokens
                    )
                    raw_cost = getattr(sdk_msg, "total_cost_usd", None)
                    raw_cost_complete = _valid_cost(raw_cost)
                    current_usage_pricing = (
                        _pricing_segment_from_usage(
                            raw_usage,
                            model=self.model,
                        )
                        if raw_usage_complete
                        else None
                    )
                    raw_terminal_pricing = (
                        UsagePricingSegment(
                            input_tokens=(
                                raw_in
                                + raw_cache_creation
                                + raw_cache_read
                            ),
                            output_tokens=raw_out,
                            cache_creation_tokens=raw_cache_creation,
                            cache_read_tokens=raw_cache_read,
                            cost_usd=float(raw_cost),
                        )
                        if (
                            _valid_token_counter(raw_in)
                            and _valid_token_counter(raw_out)
                            and _valid_token_counter(raw_cache_creation)
                            and _valid_token_counter(raw_cache_read)
                            and _valid_cost(raw_cost)
                        )
                        else None
                    )
                    real_api_error_accounting_reconciles = (
                        pending_api_error_category is not None
                        and not synthetic_api_error_model_usage_valid
                        and resume_session_id is None
                        and terminal_fields_complete
                        and not pending_api_error_conflict
                        and not pending_api_error_stream_conflict
                        and raw_is_error is True
                        and type(getattr(sdk_msg, "subtype", None)) is str
                        and getattr(sdk_msg, "subtype", None) == "success"
                        and pricing_is_flat
                        and admitted_pre_api_error_assistant_count == 1
                        and admitted_pre_api_error_usage_seen
                        and synthetic_api_error_marker_count == 1
                        and synthetic_api_error_marker_usage_zero
                        and assistant_pricing_stream_complete
                        and terminal_served_model is not None
                        and model_usage_error is None
                        and terminal_model_pricing is not None
                        and raw_terminal_pricing is not None
                        and current_usage_pricing is not None
                        and _pricing_segments_reconcile(
                            (raw_terminal_pricing,),
                            terminal_model_pricing,
                        )
                        and _pricing_segments_reconcile(
                            (current_usage_pricing,),
                            terminal_model_pricing,
                        )
                        and _pricing_segments_reconcile(
                            tuple(assistant_pricing_segments),
                            terminal_model_pricing,
                        )
                    )
                    pending_api_error_model_usage_valid = (
                        pending_api_error_category is None
                        or (
                            synthetic_api_error_model_usage_valid
                            and admitted_pre_api_error_assistant_count == 0
                        )
                        or real_api_error_accounting_reconciles
                    )
                    terminal_usage_authoritative = (
                        resume_session_id is None
                        and terminal_model_pricing is not None
                        and (
                            pending_api_error_category is None
                            or real_api_error_accounting_reconciles
                        )
                    )
                    # Claude Code restores modelUsage and total_cost_usd from
                    # session state on resume.  They are historical cumulative
                    # values, not this Invocation's delta.  Without a trusted
                    # pre-resume baseline, retain only the current outer-stream
                    # usage as an inclusive lower bound and never mark it exact.
                    usage_complete = (
                        terminal_usage_authoritative
                        or (
                            resume_session_id is None
                            and raw_usage_complete
                        )
                    )
                    cost_complete = (
                        resume_session_id is None and raw_cost_complete
                    )
                    terminal_pricing_reconciles = (
                        model_usage_error is None
                        and terminal_model_pricing is not None
                        and (
                            pending_api_error_category is None
                            or real_api_error_accounting_reconciles
                        )
                        and (
                            (
                                raw_terminal_pricing is not None
                                and _pricing_segments_reconcile(
                                    (raw_terminal_pricing,),
                                    terminal_model_pricing,
                                )
                                and current_usage_pricing is not None
                                and _pricing_segments_reconcile(
                                    (current_usage_pricing,),
                                    terminal_model_pricing,
                                )
                            )
                            or (
                                # On this exact terminal, Claude Code can
                                # scope ResultMessage.usage to its last
                                # response fragment while model_usage and
                                # total_cost_usd retain the fresh invocation
                                # aggregate. Admit the aggregate only when the
                                # fragment is contained, both cost views agree,
                                # and the local pricing table independently
                                # reproduces that aggregate cost.
                                pending_api_error_category is None
                                and getattr(sdk_msg, "subtype", None)
                                in _CLAUDE_MAX_BUDGET_RESULT_SUBTYPES
                                and raw_terminal_pricing is not None
                                and current_usage_pricing is not None
                                and _pricing_fragment_within_terminal(
                                    raw_terminal_pricing,
                                    terminal_model_pricing,
                                )
                                and raw_cost_complete
                                and math.isclose(
                                    float(raw_cost),
                                    terminal_model_pricing.cost_usd,
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                )
                                and math.isclose(
                                    compute_cost(
                                        self.model,
                                        input_tokens=(
                                            terminal_model_pricing.input_tokens
                                        ),
                                        output_tokens=(
                                            terminal_model_pricing.output_tokens
                                        ),
                                        cache_creation_tokens=(
                                            terminal_model_pricing
                                            .cache_creation_tokens
                                        ),
                                        cache_read_tokens=(
                                            terminal_model_pricing
                                            .cache_read_tokens
                                        ),
                                    ),
                                    terminal_model_pricing.cost_usd,
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                )
                            )
                        )
                    )
                    pricing_complete = (
                        resume_session_id is None
                        and pricing_is_flat
                        and terminal_pricing_reconciles
                        and usage_complete
                        and cost_complete
                        and terminal_fields_complete
                    )
                    # Claude Code may expose only partial or cumulative
                    # AssistantMessage usage while its fresh, one-model
                    # ResultMessage carries the exact invocation aggregate.
                    # On a complete path the independently reconciled terminal
                    # aggregate is authoritative. AssistantMessage segments
                    # remain useful only as incomplete partial attribution.
                    pricing_segments = (
                        (terminal_model_pricing,)
                        if pricing_complete
                        and terminal_model_pricing is not None
                        else tuple(assistant_pricing_segments)
                    )
                    pricing_evidence = UsagePricingEvidence(
                        segments=pricing_segments,
                        complete=pricing_complete,
                    )
                    if terminal_usage_authoritative:
                        assert terminal_model_pricing is not None
                        usage = Usage(
                            input_tokens=terminal_model_pricing.input_tokens,
                            output_tokens=terminal_model_pricing.output_tokens,
                            cost_usd=(
                                float(raw_cost)
                                if raw_cost_complete
                                else terminal_model_pricing.cost_usd
                            ),
                            cache_creation_tokens=(
                                terminal_model_pricing.cache_creation_tokens
                            ),
                            cache_read_tokens=(
                                terminal_model_pricing.cache_read_tokens
                            ),
                        )
                    else:
                        # ResultMessage.usage is scoped to the current outer
                        # stream even when the CLI's cost ledger is cumulative.
                        # Preserve its inclusive token subsets and price only
                        # that visible current usage on resume.
                        usage = Usage(
                            input_tokens=inclusive_input_tokens,
                            output_tokens=out_tokens,
                            cost_usd=(
                                float(raw_cost)
                                if raw_cost_complete
                                and resume_session_id is None
                                else (
                                    current_usage_pricing.cost_usd
                                    if current_usage_pricing is not None
                                    else partial_cost
                                )
                            ),
                            cache_creation_tokens=cache_creation_tokens,
                            cache_read_tokens=cache_read_tokens,
                        )
                    if (
                        raw_usage_value is not None
                        or self._partial_usage is not None
                        or raw_cost is not None
                    ):
                        _append_event(
                            events,
                            "usage",
                            name=(
                                "result_complete"
                                if usage_complete and cost_complete
                                else "result_partial"
                            ),
                        )
                    raw_stop_reason = getattr(sdk_msg, "stop_reason", None)
                    stop_reason = (
                        raw_stop_reason.strip()
                        if (
                            isinstance(raw_stop_reason, str)
                            and raw_stop_reason.strip()
                        )
                        else (
                            "error"
                            if raw_is_error is True
                            else "end_turn"
                        )
                    )
                    observed_uuid = getattr(sdk_msg, "uuid", None)
                    if isinstance(observed_uuid, str) and observed_uuid.strip():
                        provider_request_id = observed_uuid.strip()
                    permission_denials = getattr(
                        sdk_msg,
                        "permission_denials",
                        None,
                    )
                    terminal_subtype = getattr(sdk_msg, "subtype", None)
                    turn_limit_terminal = (
                        pending_api_error_category is None
                        and type(terminal_subtype) is str
                        and terminal_subtype
                        in _CLAUDE_MAX_TURNS_RESULT_SUBTYPES
                    )
                    if (
                        permission_denials
                        and pending_api_error_category is None
                    ):
                        stop_reason = "permission_denied"
                        result_error = TransportError(
                            http_status=None,
                            message=(
                                "Claude Agent SDK terminal reported "
                                f"{len(permission_denials)} permission "
                                "denial(s)"
                            ),
                            retriable=False,
                        )
                        _append_event(
                            events,
                            "limit",
                            name="permission_denial",
                        )
                    elif (
                        pending_api_error_category is None
                        and getattr(sdk_msg, "subtype", None)
                        in _CLAUDE_MAX_BUDGET_RESULT_SUBTYPES
                    ):
                        # Claude Code has emitted this exact subtype with both
                        # values of ``is_error`` across reviewed CLI versions.
                        # The provider-owned subtype, not that compatibility
                        # bit or free-form result prose, is authoritative.
                        stop_reason = "budget_exhausted"
                        budget_limit = (
                            effective_budget_usd
                            if effective_budget_usd is not None
                            else usage.cost_usd
                        )
                        raw_result = getattr(sdk_msg, "result", None)
                        result_detail = (
                            redact_sdk_text(raw_result.strip())[:500]
                            if (
                                isinstance(raw_result, str)
                                and raw_result.strip()
                            )
                            else "provider max_budget_usd terminal"
                        )
                        result_error = BudgetExhausted(
                            spent_usd=usage.cost_usd,
                            limit_usd=budget_limit,
                            message=(
                                "Claude Agent SDK stopped at the configured "
                                f"max_budget_usd boundary ${budget_limit:.6f}: "
                                f"{result_detail}"
                            ),
                            # The provider can stop before consuming the
                            # complete physical-query allowance (the frozen
                            # Breaker did so). A resumed query is admissible
                            # only when the terminal supplied exact cost and a
                            # strictly positive logical remainder; the outer
                            # retry loop forwards that smaller remainder.
                            retriable=(
                                effective_budget_usd is not None
                                and cost_complete
                                and terminal_fields_complete
                                and usage.cost_usd < effective_budget_usd
                            ),
                        )
                        _append_event(
                            events,
                            "limit",
                            name="budget_usd",
                        )
                    elif turn_limit_terminal:
                        # A used-up turn allowance is a limit of the work, not
                        # a provider failure.  The provider-owned subtype is
                        # authoritative over ``is_error``, the reported error
                        # prose and the last API stop reason.  Normalize it to
                        # the canonical stop reason; the max-turns policy
                        # after the stream owns the typed outcome.
                        stop_reason = "max_turns"
                    elif pending_api_error_category is not None:
                        stop_reason = "error"
                        status = getattr(
                            sdk_msg,
                            "api_error_status",
                            None,
                        )
                        result_subtype = getattr(sdk_msg, "subtype", None)
                        if pending_api_error_conflict:
                            terminal_fields_complete = False
                            usage_complete = False
                            cost_complete = False
                            usage = _usage_conservative_envelope(
                                usage,
                                self._partial_usage,
                            )
                            result_error = TransportError(
                                http_status=None,
                                message=(
                                    "Claude Agent SDK emitted conflicting "
                                    "synthetic API-error assistant categories"
                                ),
                                retriable=False,
                            )
                        elif raw_is_error is not True:
                            # The local synthetic marker is not terminal
                            # authority. A missing or contradictory canonical
                            # error result cannot become a completion, a model-
                            # drift claim, or zero-cost retry authority.
                            terminal_fields_complete = False
                            usage_complete = False
                            cost_complete = False
                            usage = _usage_conservative_envelope(
                                usage,
                                self._partial_usage,
                            )
                            result_error = TransportError(
                                http_status=None,
                                message=(
                                    "Claude Agent SDK synthetic API-error "
                                    "assistant marker was not confirmed by an "
                                    "authoritative error ResultMessage"
                                ),
                                retriable=False,
                            )
                        elif not (
                            type(result_subtype) is str
                            and result_subtype == "success"
                        ):
                            # In the pinned SDK, ``api_error_status`` belongs
                            # only to an error ResultMessage whose subtype is
                            # the exact nominal terminal spelling ``success``.
                            # A budget/other terminal cannot be reclassified
                            # as an HTTP transport failure merely because a
                            # preceding CLI-local marker is pending.
                            terminal_fields_complete = False
                            usage_complete = False
                            cost_complete = False
                            usage = _usage_conservative_envelope(
                                usage,
                                self._partial_usage,
                            )
                            result_error = TransportError(
                                http_status=None,
                                message=(
                                    "Claude Agent SDK synthetic API-error "
                                    "ResultMessage subtype must be the exact "
                                    "plain string 'success'"
                                ),
                                retriable=False,
                            )
                        elif pending_api_error_stream_conflict:
                            terminal_fields_complete = False
                            usage_complete = False
                            cost_complete = False
                            usage = _usage_conservative_envelope(
                                usage,
                                self._partial_usage,
                            )
                            result_error = TransportError(
                                http_status=None,
                                message=(
                                    "Claude Agent SDK emitted an additional "
                                    "AssistantMessage that contradicted the "
                                    "pending synthetic API-error marker"
                                ),
                                retriable=False,
                            )
                        elif not pending_api_error_model_usage_valid:
                            terminal_fields_complete = False
                            usage_complete = False
                            cost_complete = False
                            usage = _usage_conservative_envelope(
                                usage,
                                self._partial_usage,
                            )
                            result_error = TransportError(
                                http_status=None,
                                message=(
                                    "Claude Agent SDK terminal model_usage "
                                    "contradicted the pending synthetic "
                                    "API-error marker"
                                ),
                                retriable=False,
                            )
                        else:
                            admitted_status = (
                                status if _valid_http_status(status) else None
                            )
                            if admitted_status is None:
                                # The synthetic assistant marker is not enough
                                # to infer a provider status.  Preserve any
                                # exact terminal usage/cost, but keep the
                                # physical result incomplete unless the SDK
                                # supplies one exact integer HTTP status.
                                terminal_fields_complete = False
                            result_error = TransportError(
                                http_status=admitted_status,
                                message=(
                                    "Claude Agent SDK terminal confirmed "
                                    "provider API error "
                                    f"{pending_api_error_category!r}"
                                    if admitted_status is not None
                                    else (
                                        "Claude Agent SDK terminal API-error "
                                        "confirmation lacked an exact integer "
                                        "api_error_status"
                                    )
                                ),
                                retriable=(
                                    admitted_status in RETRIABLE_HTTP_STATUSES
                                    if admitted_status is not None
                                    else False
                                ),
                            )
                    elif reported_errors:
                        stop_reason = "error"
                        result_error = TransportError(
                            http_status=None,
                            message=(
                                "Claude Agent SDK terminal reported "
                                f"{len(reported_errors)} provider error(s)"
                            ),
                            retriable=False,
                        )
                    elif raw_is_error is True:
                        status = getattr(
                            sdk_msg,
                            "api_error_status",
                            None,
                        )
                        result_error = TransportError(
                            http_status=(
                                int(status) if status is not None else None
                            ),
                            message=(
                                "Claude Agent SDK terminal result reported "
                                f"is_error=true (subtype="
                                f"{getattr(sdk_msg, 'subtype', 'unknown')})"
                            ),
                            retriable=(
                                int(status) in RETRIABLE_HTTP_STATUSES
                                if status is not None
                                else False
                            ),
                        )
                    identity_error = (
                        model_usage_error
                        if pending_api_error_category is None
                        else None
                    )
                    identity_stop_reason = "model_mismatch"
                    # A turn-limit terminal carries no typed error at this
                    # point and may return its partial content, so it needs
                    # the same served-model proof whichever ``is_error`` value
                    # the provider reported.
                    if (
                        identity_error is None
                        and served_model is None
                        and (raw_is_error is False or turn_limit_terminal)
                        and result_error is None
                    ):
                        identity_error = _missing_served_model_identity()
                        identity_stop_reason = "missing_model_identity"
                    if identity_error is not None:
                        # Model identity is an admission prerequisite. Retain
                        # authoritative usage and an unambiguous raw identity,
                        # but never expose content from a terminal whose route
                        # is mismatched, malformed, ambiguous, or unproven.
                        _append_event(
                            events,
                            "limit",
                            name="served_model_mismatch",
                        )
                        return self._result_with_evidence(
                            final_message="",
                            transcript=(),
                            usage=usage,
                            stop_reason=identity_stop_reason,
                            error=identity_error,
                            token_usage_complete=usage_complete,
                            cost_complete=cost_complete,
                            terminal_complete=False,
                            events=events,
                            served_model=served_model,
                            provider_request_id=provider_request_id,
                            tool_authority_digest=tool_authority_digest,
                            pricing_evidence=pricing_evidence,
                            conversation=_conversation_outcome(
                                invocation,
                                opaque_id=provider_conversation_id,
                            ),
                        )
                    structured_output = (
                        getattr(sdk_msg, "structured_output", None)
                        if pending_api_error_category is None
                        else None
                    )
                    if structured_output is not None:
                        final_message = (
                            structured_output
                            if isinstance(structured_output, str)
                            else json.dumps(
                                structured_output,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )
                        transcript.append(
                            Message(
                                role="assistant",
                                content=final_message,
                            )
                        )
                    elif (
                        pending_api_error_category is None
                        # ``result`` on a turn-limit terminal is the
                        # provider's limit prose, never the model's reply:
                        # the last assistant text stays the partial result.
                        and not turn_limit_terminal
                        and isinstance(getattr(sdk_msg, "result", None), str)
                        and sdk_msg.result
                    ):
                        final_message = sdk_msg.result
                        transcript.append(
                            Message(role="assistant", content=final_message)
                        )
                    # ResultMessage is the SDK's terminal envelope. Claude Code
                    # intentionally exits non-zero after some error terminals
                    # (including max_budget_usd), and the pinned SDK exposes
                    # that later process exit as a second stream error. Do not
                    # let post-terminal transport teardown replace the exact
                    # usage and typed error already admitted above.
                    break
        except Exception as exc:  # noqa: BLE001 — caller classifies
            # Surface the bundled CLI's captured stderr in both the
            # ERROR log (so postmortem grep finds it without unpacking
            # the SDK debug JSON) and on the raised exception chain
            # (so structured callers can read it programmatically).
            # Without this, the SDK transport's "Command failed with
            # exit code 1 (Check stderr output for details)" reaches
            # us with no actual stderr — making transport-level
            # crashes opaque. The kept entries are published through
            # ``redact_sdk_lines``. An entry the window flags as its own
            # size note is no line of the CLI: it is published as redacted
            # on its own, so a kept note is never the withheld marker. The
            # search for a rejected resume reads the kept entries as the
            # CLI wrote them, never the redacted lines.
            if cli_stderr_window.entries:
                redacted_stderr = redact_sdk_lines(cli_stderr_window.entries)
                for index, noted in enumerate(
                    cli_stderr_window.notes[: len(redacted_stderr)]
                ):
                    if noted:
                        redacted_stderr[index] = redact_sdk_text(
                            cli_stderr_window.entries[index]
                        )
                tail = "\n".join(redacted_stderr[-50:])
                dropped = cli_stderr_window.received - len(redacted_stderr)
                logger.error(
                    "claude bundled CLI exited abnormally; %sstderr tail "
                    "(last %d of %d lines):\n%s",
                    (
                        f"earlier stderr lines dropped: {dropped}; "
                        if dropped
                        else ""
                    ),
                    min(len(redacted_stderr), 50),
                    len(redacted_stderr),
                    tail,
                )
                try:
                    exc.cli_stderr = "\n".join(redacted_stderr)  # type: ignore[attr-defined]
                except (AttributeError, TypeError):
                    pass
            if (
                external_resume
                and not session_id_holder
                and _claude_resume_was_rejected(
                    exc,
                    cli_stderr_window.entries,
                    cli_stderr_window.resume_marks(),
                )
            ):
                return self._result_with_evidence(
                    final_message=final_message,
                    transcript=tuple(transcript),
                    usage=self._partial_usage or Usage(),
                    stop_reason="stale_conversation",
                    error=StaleConversation(
                        message=(
                            "Claude provider rejected the requested "
                            "conversation handle"
                        )
                    ),
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=False,
                    events=events,
                    served_model=served_model,
                    provider_request_id=provider_request_id,
                    tool_authority_digest=tool_authority_digest,
                    pricing_evidence=self._partial_pricing_evidence,
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=None,
                        state="stale",
                    ),
                )
            # Decorate with usage we accumulated before the failure so
            # callers can attribute partial cost.
            err = self._classify_exception(exc, policy)
            if err is None:
                raise
            from crucible.tools._dispatch_budget import (  # noqa: PLC0415
                TerminalToolDispatchError,
                ToolDispatchBudgetExceeded,
                ToolMutationIndeterminate,
            )

            unwrapped = _unwrap_exception_group(exc)
            tool_dispatch_denied = isinstance(
                unwrapped,
                ToolDispatchBudgetExceeded,
            )
            tool_mutation_indeterminate = isinstance(
                unwrapped,
                ToolMutationIndeterminate,
            )
            terminal_tool_dispatch = isinstance(
                unwrapped,
                TerminalToolDispatchError,
            )
            partial = self._partial_usage or Usage()
            if (
                partial.input_tokens
                or partial.output_tokens
                or partial.cost_usd
            ) and not any(event.kind == "usage" for event in events):
                _append_event(events, "usage", name="partial")
            result_stop_reason = (
                "tool_mutation_indeterminate"
                if tool_mutation_indeterminate
                else (
                    "tool_dispatch_budget_exhausted"
                    if tool_dispatch_denied
                    else (
                        "cancelled"
                        if getattr(err, "kind", None) == "cancellation"
                        else (
                            "tool_dispatch_terminal"
                            if terminal_tool_dispatch
                            else "error"
                        )
                    )
                )
            )
            if tool_dispatch_denied:
                _append_event(
                    events,
                    "limit",
                    name="tool_dispatch_budget",
                )
            if getattr(err, "kind", None) == "cancellation":
                _append_event(events, "cancellation", name="external")
            return self._result_with_evidence(
                final_message=final_message,
                transcript=tuple(transcript),
                usage=partial,
                stop_reason=result_stop_reason,
                error=err,
                token_usage_complete=False,
                cost_complete=False,
                terminal_complete=False,
                events=events,
                served_model=served_model,
                provider_request_id=provider_request_id,
                tool_authority_digest=tool_authority_digest,
                pricing_evidence=self._partial_pricing_evidence,
                conversation=_conversation_outcome(
                    invocation,
                    opaque_id=provider_conversation_id,
                ),
            )
        finally:
            # PEP 533 was deferred: returning from an async-for body does not
            # close nested async generators. Close both layers explicitly so
            # every identity rejection and early terminal tears down the CLI.
            for stream in (message_stream, sdk_messages):
                close = getattr(stream, "aclose", None)
                if not callable(close):
                    continue
                try:
                    await close()
                except Exception:  # noqa: BLE001 - preserve primary result/error
                    logger.debug(
                        "Claude Agent SDK stream cleanup failed",
                        exc_info=True,
                    )

        if not result_seen:
            stop_reason = "incomplete_terminal"
            usage = self._partial_usage or Usage()
            pricing_evidence = self._partial_pricing_evidence
            result_error = TransportError(
                http_status=None,
                message=(
                    "Claude Agent SDK synthetic API-error assistant marker "
                    "was not confirmed by an authoritative ResultMessage"
                    if pending_api_error_category is not None
                    else (
                        "Claude Agent SDK stream ended without a "
                        "ResultMessage"
                    )
                ),
                retriable=pending_api_error_category is None,
            )

        def result_with_evidence(
            *,
            error: Any = result_error,
            result_stop_reason: str = stop_reason,
            result_final_message: str = final_message,
            result_transcript: tuple[Message, ...] = tuple(transcript),
        ) -> InvocationResult:
            return self._result_with_evidence(
                final_message=result_final_message,
                transcript=result_transcript,
                usage=usage,
                stop_reason=result_stop_reason,
                error=error,
                token_usage_complete=usage_complete,
                cost_complete=cost_complete,
                terminal_complete=(
                    result_seen and terminal_fields_complete
                ),
                events=events,
                served_model=served_model,
                provider_request_id=provider_request_id,
                tool_authority_digest=tool_authority_digest,
                pricing_evidence=pricing_evidence,
                conversation=_conversation_outcome(
                    invocation,
                    opaque_id=provider_conversation_id,
                ),
            )

        # Hard budget gate: if Invocation.budget_usd was set, return
        # BudgetExhausted instead of a normal result when cost > budget.
        if (
            effective_budget_usd is not None
            and usage.cost_usd > effective_budget_usd
        ):
            _append_event(events, "limit", name="budget_usd")
            return result_with_evidence(
                result_stop_reason="budget_exhausted",
                error=BudgetExhausted(
                    spent_usd=usage.cost_usd,
                    limit_usd=effective_budget_usd,
                    message=(
                        f"Invocation spent ${usage.cost_usd:.4f} "
                        "exceeding the remaining configured budget "
                        f"${effective_budget_usd:.4f}"
                    ),
                ),
            )

        # Max-turns handling per policy (B31).
        turns_used = self._last_num_turns or 0
        hit_max_turns = _classify_max_turns(
            stop_reason, turns_used, effective_max_turns,
        )
        if hit_max_turns:
            _append_event(events, "limit", name="max_turns")
            if policy.on_max_turns == "raise":
                return result_with_evidence(
                    result_stop_reason="max_turns",
                    error=MaxTurnsExceeded(
                        turns_used=turns_used,
                        turns_limit=effective_max_turns,
                        message=f"hit max_turns={effective_max_turns}",
                    ),
                )
            # "continue_with_warning" — return the partial result
            logger.warning(
                "ClaudeBackend hit max_turns=%d but policy=continue_with_warning",
                effective_max_turns,
            )

        # Stalled / empty completion guard. The bundled CLI can return a
        # clean-exit stream (no exception, ``stop_reason=end_turn``) that
        # nevertheless carries an empty ``final_message`` and zero output
        # tokens — observed as a ~10-min silent stall after the first
        # byte, then SDK shutdown. Returned as-is it looks like a
        # successful invocation, so the retry loop never re-tries it and
        # downstream JSON-parsing consumers (``assay.review`` /
        # ``assay.plan``) hit ``parse_failure`` on the empty text and fail
        # the gate hard. Convert it to a retriable TransportError so the
        # backend's own retry budget gets a fresh attempt instead. Only
        # fires for non-tool invocations (``invoke()`` / review / plan),
        # where an empty final message is unambiguously a failure; tool-
        # driven loops legitimately end with an empty text turn after a
        # tool_result and are handled in ``chat_with_tools``. A session
        # that ended on its turn limit is not a stall: under
        # ``continue_with_warning`` its partial result, text-free or not,
        # is the requested outcome.
        if (
            not final_message
            and usage.output_tokens == 0
            and not invocation.tools
            and result_error is None
            and not hit_max_turns
        ):
            logger.warning(
                "ClaudeBackend: empty completion (stop_reason=%s, "
                "0 output tokens, empty final_message); treating as a "
                "retriable transport stall",
                stop_reason,
            )
            return result_with_evidence(
                result_final_message="",
                error=TransportError(
                    http_status=None,
                    message=(
                        f"empty completion (stop_reason={stop_reason}, "
                        "0 output tokens) — bundled CLI returned a clean "
                        "but content-free stream"
                    ),
                    retriable=True,
                ),
            )

        return result_with_evidence()

    # ------------------------------------------------------------------
    # Exception → BackendError classification
    # ------------------------------------------------------------------

    def _classify_exception(
        self, exc: BaseException, policy: BackendPolicy,
    ) -> ClassifiedError | None:
        """Map a Python exception to an assay BackendError subtype.

        Returns ``None`` only when the caller explicitly opted out of
        classification — currently just the
        ``policy.on_cancellation == "raise"`` path through
        ``_is_external_cancellation``. For every other input the
        method returns a BackendError instance.

        Classification policy uses abort-list semantics:

        - External cancellation → ``ExternalCancellation`` (retriable=True)
          unless ``on_cancellation == "raise"``.
        - Content refusal → ``ContentRejected`` (retriable=False).
        - Terminal class names (``_TERMINAL_EXCEPTION_NAMES``) →
          ``TransportError(retriable=False)``.
        - Bundled-CLI ``Exception("Command failed ...")`` (prefix
          match — covers ``with exit code N``, ``during X``, and any
          future variants) → ``TransportError(retriable=False)``,
          non-retriable by design (cost amplification + idempotency).
        - Anthropic API status_code → retriable iff status ∈ {408,
          409, 429, 5xx}.
        - Everything else → ``TransportError(retriable=True)``, so
          unknown / future SDK shapes retry against the budget
          instead of leaking as raw Python exceptions.
        """
        inner = _unwrap_exception_group(exc)

        from crucible.tools._dispatch_budget import (  # noqa: PLC0415
            TerminalToolDispatchError,
            ToolDispatchBudgetExceeded,
            ToolMutationIndeterminate,
        )

        if isinstance(inner, ToolMutationIndeterminate):
            return TransportError(
                http_status=None,
                message=(
                    "Claude tool loop terminated after an indeterminate "
                    f"source mutation: {inner}"
                ),
                retriable=False,
            )
        if isinstance(inner, ToolDispatchBudgetExceeded):
            return MaxTurnsExceeded(
                turns_used=max(self._partial_turns, 0),
                turns_limit=max(self._partial_turns, 1),
                message=(
                    "Claude tool loop stopped at the first dispatch-budget "
                    f"denial: {inner}"
                ),
            )
        if isinstance(inner, TerminalToolDispatchError):
            return TransportError(
                http_status=None,
                message=(
                    "Claude tool loop terminated at a workflow-owned "
                    f"dispatch boundary: {inner}"
                ),
                retriable=False,
            )

        # External cancellation (B32).
        if _is_external_cancellation(inner):
            if policy.on_cancellation == "raise":
                # Caller wants us to re-raise — return None to signal.
                return None
            return ExternalCancellation(
                cause=type(inner).__name__,
                message=str(inner),
                retriable=True,
            )

        # Transport-class errors (network, SDK transport, etc).
        # We pattern-match on the exception class name since claude-agent-sdk
        # may not export a single base type. ``__name__`` (not
        # ``__qualname__``) is deliberate: it strips module qualifiers so a
        # match against bare "AuthenticationError" still hits when the
        # exception is ``anthropic._exceptions.AuthenticationError``. If
        # anyone switches to ``__qualname__`` here, the abort-list matches
        # below would silently miss.
        cls_name = type(inner).__name__
        msg = str(inner)

        # Content-rejection / refusal — the model refused to comply.
        if "refusal" in msg.lower() or cls_name in ("ContentFilterError", "RefusalError"):
            return ContentRejected(message=msg, retriable=False)

        # Abort-list — terminal class names that retry can't recover.
        # Surface a structured TransportError(retriable=False) so the
        # caller drops out of the retry loop instead of consuming the
        # budget against an immutable problem.
        if cls_name in TERMINAL_EXCEPTION_NAMES:
            return TransportError(
                http_status=None,
                message=f"{cls_name}: {msg}",
                retriable=False,
            )

        # ``claude_agent_sdk._internal.query.receive_messages`` raises
        # a bare ``Exception("Command failed ...")`` when the bundled
        # CLI exits non-zero mid-stream (observed: clean LSP / BigQuery
        # teardown sequence, then exit code 1 — a Node.js process-exit
        # corner case with an uncaught promise rejection during
        # shutdown). The message shape is not stable across SDK
        # versions ("with exit code N", "during X", "while Y"), so we
        # match on the ``Command failed`` prefix to cover all observed
        # variants. With the abort-list inversion in this file's
        # ``_classify_exception``, an exact-string match here would
        # silently let new SDK message variants fall through to
        # default-retry — re-introducing the cost-amplification class
        # that motivated the original carve-out.
        #
        # Classify as NON-retriable transport. Two reasons:
        #
        # 1. Cost amplification — each retry restarts the bundled CLI
        #    + replays the full prompt; on long Defender turns this
        #    doubles or triples spend before the retry budget runs
        #    out. The bundled-CLI bug is not transient enough that
        #    retrying succeeds reliably; one observed pass had 7 of
        #    20 instances trip this shape with the retry budget
        #    fully consumed.
        # 2. Idempotency mismatch — the workspace's working tree may
        #    already carry partial file edits from the failed call
        #    (tool calls are applied before the message stream
        #    completes). A blind retry replays the prompt against
        #    that pre-mutated state; the retried CLI then writes its
        #    own edits on top, producing contradictory diffs.
        #
        # Returning a structured TransportError lets ``Backend.invoke``
        # surface ``InvocationResult.error`` to the caller. Every
        # assay primitive that consumes a Backend already handles the
        # error field consistently:
        #
        #   - ``assay.primitives.implement._classify`` maps
        #     ``error_kind == "transport"`` to classification
        #     ``"hard_fail_transport"``; the Implementation record is
        #     persisted with that classification + an empty diff.
        #   - ``assay.primitives.review`` returns a Verdict with
        #     ``passed=False`` carrying a synthetic backend-error
        #     Finding so the recipe sees a failed verdict instead of
        #     silently approving on an empty findings list.
        #   - ``assay.primitives.plan``'s ``_planner_call`` returns
        #     ``(None, transcript_ref)`` when the result has an error,
        #     which the planner loop treats as a no-tier_dag round
        #     (retries up to ``max_critique_rounds``).
        #
        # Workflows that loop over these primitives (breach
        # Breaker/Defender/Judge, tier per-slice implementer, debate
        # workflows) consume the structured outputs without bespoke
        # try/except blocks. This is the consistent, primitive-level
        # failure handling — workflows stay focused on orchestration.
        if cls_name == "Exception" and msg.startswith("Command failed"):
            return TransportError(
                http_status=None,
                message=(
                    f"sdk_transport: "
                    f"{msg.splitlines()[0] if msg else 'no detail'}"
                ),
                retriable=False,
            )

        # Anthropic API errors carry an http_status we can extract.
        if hasattr(inner, "status_code"):
            status = int(getattr(inner, "status_code", 0) or 0)
            return TransportError(
                http_status=status,
                message=f"{cls_name}: {msg}",
                retriable=status in RETRIABLE_HTTP_STATUSES,
            )

        # Default: retry. Unknown SDK exception shapes get a retriable
        # TransportError instead of leaking as raw Python exceptions.
        # The intentional terminal cases (abort-list, the bundled-CLI
        # "Command failed with exit code" shape, non-retriable 4xx
        # status codes) are all caught above this fall-through, so
        # default-retry only applies to genuinely-unrecognized errors.
        return TransportError(
            http_status=None,
            message=f"{cls_name}: {msg}",
            retriable=True,
        )
