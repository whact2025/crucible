# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

# ruff: noqa: TRY003
"""Source-owned physical usage evidence for backend accounting.

The ordinary :mod:`assay` pricing evidence is intentionally provider-neutral:
it contains counters and prices, but cannot attest which transport event
supplied them. Adapters emit this prompt-free projection where they admit one
physical response's raw usage, preserving the source needed to inspect
accounting and cancellation outcomes.

A collector is active only inside ``invoke_with_physical_telemetry`` and is
carried through async Codex execution with a context variable. Its returned
immutable values have closed schemas and bounded response and attempt counts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Literal


PhysicalTelemetryBackend = Literal["codex", "kimi"]
PhysicalTelemetrySource = Literal[
    "codex_app_server.raw_response_completed.usage.v1",
    "codex_app_server.thread_token_usage_updated.last.v1",
    "kimi_worker.status_update.usage_checkpoint_delta.v1",
]
CacheWriteFieldState = Literal["present", "absent", "malformed"]
CacheWriteInterpretation = Literal[
    "explicit_positive",
    "raw_response_protocol_default_zero",
    "raw_response_explicit_zero",
    "explicit_zero_all_read",
    "arithmetic_all_read_zero",
    "explicit_zero_untrusted",
    "unavailable",
    "worker_checkpoint",
]

_MAX_TOKEN_COUNT = 1_000_000_000
_MAX_RESPONSES = 128
_MAX_ATTEMPTS = 64


def _exact_count(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_TOKEN_COUNT:
        raise ValueError(f"{label} must be an exact bounded non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class PhysicalUsageResponse:
    """One prompt-free raw response/checkpoint projection."""

    response_ordinal: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    reported_cache_write_tokens: int | None
    cache_write_field_state: CacheWriteFieldState
    cache_write_interpretation: CacheWriteInterpretation

    def __post_init__(self) -> None:  # noqa: PLR0912
        _exact_count(self.response_ordinal, "response_ordinal")
        _exact_count(self.input_tokens, "input_tokens")
        _exact_count(self.output_tokens, "output_tokens")
        _exact_count(self.cache_read_tokens, "cache_read_tokens")
        if self.reported_cache_write_tokens is not None:
            _exact_count(
                self.reported_cache_write_tokens,
                "reported_cache_write_tokens",
            )
        if self.cache_read_tokens > self.input_tokens:
            raise ValueError("cache-read tokens exceed response input tokens")
        if self.cache_write_field_state == "present":
            if self.reported_cache_write_tokens is None:
                raise ValueError("present cache-write field lacks a counter")
        elif self.cache_write_field_state in {"absent", "malformed"}:
            if self.reported_cache_write_tokens is not None:
                raise ValueError("untrusted cache-write field retained a counter")
        else:
            raise ValueError("unknown cache-write field state")

        reported = self.reported_cache_write_tokens
        interpretation = self.cache_write_interpretation
        if interpretation == "explicit_positive":
            valid = (
                self.cache_write_field_state == "present"
                and reported is not None
                and reported > 0
                and reported + self.cache_read_tokens <= self.input_tokens
            )
        elif interpretation == "raw_response_protocol_default_zero":
            valid = (
                self.cache_write_field_state == "absent"
                and reported is None
            )
        elif interpretation == "raw_response_explicit_zero":
            valid = (
                self.cache_write_field_state == "present"
                and reported == 0
            )
        elif interpretation == "explicit_zero_all_read":
            valid = (
                self.cache_write_field_state == "present"
                and reported == 0
                and self.input_tokens == self.cache_read_tokens
            )
        elif interpretation == "arithmetic_all_read_zero":
            valid = (
                self.cache_write_field_state == "absent"
                and reported is None
                and self.input_tokens == self.cache_read_tokens
            )
        elif interpretation == "explicit_zero_untrusted":
            valid = (
                self.cache_write_field_state == "present"
                and reported == 0
                and self.input_tokens != self.cache_read_tokens
            )
        elif interpretation == "unavailable":
            valid = (
                self.cache_write_field_state == "malformed"
                and reported is None
            ) or (
                self.cache_write_field_state == "absent"
                and reported is None
                and self.input_tokens != self.cache_read_tokens
            )
        elif interpretation == "worker_checkpoint":
            valid = (
                self.cache_write_field_state == "present"
                and reported is not None
                and reported + self.cache_read_tokens <= self.input_tokens
            )
        else:
            valid = False
        if not valid:
            raise ValueError("cache-write interpretation contradicts raw counters")


@dataclass(frozen=True, slots=True)
class PhysicalUsageAttempt:
    """Usage admitted for one physical adapter attempt."""

    backend: PhysicalTelemetryBackend
    source: PhysicalTelemetrySource
    attempt_ordinal: int
    complete: bool
    responses: tuple[PhysicalUsageResponse, ...]

    def __post_init__(self) -> None:
        _exact_count(self.attempt_ordinal, "attempt_ordinal")
        if type(self.complete) is not bool:
            raise ValueError("physical attempt completeness must be boolean")
        expected_sources = (
            {
                "codex_app_server.raw_response_completed.usage.v1",
                "codex_app_server.thread_token_usage_updated.last.v1",
            }
            if self.backend == "codex"
            else {"kimi_worker.status_update.usage_checkpoint_delta.v1"}
            if self.backend == "kimi"
            else set()
        )
        if self.source not in expected_sources:
            raise ValueError("physical telemetry source does not match backend")
        if (
            self.source
            == "codex_app_server.thread_token_usage_updated.last.v1"
            and self.complete
        ):
            raise ValueError(
                "cumulative Codex telemetry cannot be complete physical "
                "canary evidence"
            )
        if not isinstance(self.responses, tuple) or len(self.responses) > _MAX_RESPONSES:
            raise ValueError("physical response inventory exceeds its bound")
        if any(type(response) is not PhysicalUsageResponse for response in self.responses):
            raise TypeError("physical response inventory contains an unknown type")
        if (
            self.source == "codex_app_server.raw_response_completed.usage.v1"
            and self.complete
            and any(
                response.cache_write_interpretation
                not in {
                    "explicit_positive",
                    "raw_response_protocol_default_zero",
                    "raw_response_explicit_zero",
                }
                for response in self.responses
            )
        ):
            raise ValueError(
                "complete raw-response telemetry contains an unqualified "
                "cache-write interpretation"
            )
        if any(
            response.response_ordinal != ordinal
            for ordinal, response in enumerate(self.responses)
        ):
            raise ValueError("physical response ordinals are not contiguous")


@dataclass(frozen=True, slots=True)
class PhysicalUsageEvidence:
    """All source-owned physical attempts observed by one adapter invocation."""

    backend: PhysicalTelemetryBackend
    attempts: tuple[PhysicalUsageAttempt, ...]

    def __post_init__(self) -> None:
        if self.backend not in {"codex", "kimi"}:
            raise ValueError("unknown physical telemetry backend")
        if not isinstance(self.attempts, tuple) or len(self.attempts) > _MAX_ATTEMPTS:
            raise ValueError("physical attempt inventory exceeds its bound")
        if any(type(attempt) is not PhysicalUsageAttempt for attempt in self.attempts):
            raise TypeError("physical attempt inventory contains an unknown type")
        if any(
            attempt.backend != self.backend or attempt.attempt_ordinal != ordinal
            for ordinal, attempt in enumerate(self.attempts)
        ):
            raise ValueError("physical attempt identities are not contiguous")

    @property
    def complete(self) -> bool:
        return bool(self.attempts) and all(
            attempt.complete and bool(attempt.responses)
            for attempt in self.attempts
        )


class _PhysicalTelemetryCollector:
    def __init__(self, backend: PhysicalTelemetryBackend) -> None:
        self.backend = backend
        self._attempts: list[PhysicalUsageAttempt] = []
        self._closed = False

    def append(
        self,
        *,
        backend: PhysicalTelemetryBackend,
        source: PhysicalTelemetrySource,
        complete: bool,
        responses: tuple[PhysicalUsageResponse, ...],
    ) -> None:
        if self._closed:
            raise RuntimeError("physical telemetry collector is closed")
        if backend != self.backend:
            raise RuntimeError("physical telemetry crossed backend scope")
        self._attempts.append(
            PhysicalUsageAttempt(
                backend=backend,
                source=source,
                attempt_ordinal=len(self._attempts),
                complete=complete,
                responses=responses,
            )
        )

    def freeze(self) -> PhysicalUsageEvidence:
        self._closed = True
        return PhysicalUsageEvidence(
            backend=self.backend,
            attempts=tuple(self._attempts),
        )


_ACTIVE_COLLECTOR: ContextVar[_PhysicalTelemetryCollector | None] = ContextVar(
    "crucible_physical_telemetry_collector",
    default=None,
)


@contextmanager
def capture_physical_usage(
    backend: PhysicalTelemetryBackend,
) -> Iterator[_PhysicalTelemetryCollector]:
    """Create one non-nestable source telemetry scope."""

    if _ACTIVE_COLLECTOR.get() is not None:
        raise RuntimeError("physical telemetry capture scopes cannot nest")
    collector = _PhysicalTelemetryCollector(backend)
    token: Token[_PhysicalTelemetryCollector | None] = _ACTIVE_COLLECTOR.set(
        collector
    )
    try:
        yield collector
    finally:
        _ACTIVE_COLLECTOR.reset(token)


def emit_physical_usage_attempt(
    *,
    backend: PhysicalTelemetryBackend,
    source: PhysicalTelemetrySource,
    complete: bool,
    responses: tuple[PhysicalUsageResponse, ...],
) -> None:
    """Copy one adapter-qualified source projection into an active scope."""

    collector = _ACTIVE_COLLECTOR.get()
    if collector is not None:
        collector.append(
            backend=backend,
            source=source,
            complete=complete,
            responses=responses,
        )


__all__ = [
    "PhysicalUsageAttempt",
    "PhysicalUsageEvidence",
    "PhysicalUsageResponse",
]
