# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared exception-classification constants for backend adapters.

Both ``ClaudeBackend`` and ``CodexBackend`` use abort-list semantics
on their ``_classify_exception`` paths: unknown exception shapes default to
``TransportError(retriable=True)`` so they
retry against the per-invocation budget, while a small terminal set
of class names short-circuits to ``TransportError(retriable=False)``
because retry can't recover them.

Hosting the constants here (rather than per-backend) keeps the two
backends in sync — a new terminal class or HTTP-status policy change
only needs to land in one place, and the hybrid swebench surface that
routes through both backends sees consistent classification.
"""

from __future__ import annotations

from assay import ContentRejected, ExternalCancellation, TransportError


# Terminal exception class names — retry won't recover these:
#   AuthenticationError       — 401, credential won't change on retry
#   PermissionDeniedError     — 403, entitlement won't change on retry
#   NotFoundError             — 404, model / endpoint genuinely absent
#   ContextWindowExceededError — input bigger than the model's window
#   UnsupportedParamsError    — caller passed an SDK contract violation
#   ClaudeSessionIdentityError — pinned local auth principal/CLI drifted
#
# Matched by class ``__name__`` (not ``__qualname__``) so the bare
# class name still hits when the exception is namespaced like
# ``anthropic._exceptions.AuthenticationError`` or
# ``openai.codex._errors.AuthenticationError``. If anyone switches
# to ``__qualname__`` at a match site, abort-list matches would
# silently miss.
#
# ``KeyboardInterrupt`` is deliberately absent — it routes through
# the cancellation path (``_is_external_cancellation``) before any
# backend's classifier sees it.
TERMINAL_EXCEPTION_NAMES: frozenset[str] = frozenset({
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ContextWindowExceededError",
    "UnsupportedParamsError",
    "ClaudeSessionIdentityError",
})

# HTTP status codes that warrant a retry on the
# ``hasattr(inner, "status_code")`` branch of each backend's
# classifier. Everything else (mostly 4xx) maps to
# ``TransportError(retriable=False)`` — the request is malformed or
# the resource genuinely doesn't exist, so retry can't help.
#
#   408 Request Timeout         — connection-level transient
#   409 Conflict                — typically idempotency-window contention
#   429 Too Many Requests       — rate limited, backoff helps
#   500 Internal Server Error   — upstream hiccup
#   502 Bad Gateway             — proxy / load-balancer transient
#   503 Service Unavailable     — upstream warming / draining
#   504 Gateway Timeout         — upstream slow / overloaded
#   529 Overloaded              — Anthropic ``overloaded_error``; the
#                                 canonical "back off and retry"
#                                 transient. Omitting it made a single
#                                 529 a hard, non-retriable failure —
#                                 the root cause of the planner
#                                 silently producing no plan under API
#                                 load (see CRU-01 in
#                                 docs/audits/comprehensive_code_audit_2026-06-10.md).
RETRIABLE_HTTP_STATUSES: frozenset[int] = frozenset({
    408, 409, 429, 500, 502, 503, 504, 529,
})


# Type alias for the union ``_classify_exception`` returns. The
# ``| None`` case is narrow: it only fires when ``policy.on_cancellation
# == "raise"`` opts the caller out of classification (via
# ``_is_external_cancellation``). All other inputs return a non-None
# BackendError subtype. Defined here so both backends can share the
# annotation and a future contributor sees the "or None" case is
# explicitly cancellation-related rather than a generic
# unclassifiable-error sentinel.
ClassifiedError = TransportError | ExternalCancellation | ContentRejected
