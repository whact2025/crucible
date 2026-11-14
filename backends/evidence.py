# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Workflow-bound admission of backend result evidence.

Direct adapter construction is useful for synthetic and paid diagnostic
probes, so it remains diagnostic by default. Canonical workflow routing binds
an explicit mode before returning the adapter. Enforcing routes convert an
otherwise successful incomplete result into Assay's typed
``IncompleteEvidence`` outcome.

The programmatic ``semantic`` mode is intentionally narrower than the
configuration-level diagnostic escape: it still requires complete terminal
and trace evidence, while allowing a caller to persist content whose aggregate
token/cost evidence carries explicit accounting debt. Workflows using it must
stop at their next paid-work or promotion boundary.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from assay import (
    RESULT_EVIDENCE_ADMISSION_MODES,
    InvocationResult,
    ResultEvidenceAdmission,
    admit_result_evidence,
)


_MODE_ATTRIBUTE = "_crucible_result_evidence_admission"
_GUARDED_ATTRIBUTE = "_crucible_result_evidence_guarded"
_CHAT_GUARDED_ATTRIBUTE = "_crucible_chat_result_evidence_guarded"


def bind_result_evidence_admission(
    backend: Any,
    *,
    mode: ResultEvidenceAdmission,
) -> None:
    """Bind one routing decision to an adapter before its first invocation."""

    if mode not in RESULT_EVIDENCE_ADMISSION_MODES:
        raise ValueError(
            f"unknown result evidence admission mode {mode!r}; expected one "
            f"of {sorted(RESULT_EVIDENCE_ADMISSION_MODES)}"
        )
    setattr(backend, _MODE_ATTRIBUTE, mode)
    if getattr(backend, _GUARDED_ATTRIBUTE, False):
        return
    original_invoke = backend.invoke

    @wraps(original_invoke)
    def guarded_invoke(invocation: Any) -> InvocationResult:
        result = original_invoke(invocation)
        return admit_configured_result(backend, result)

    # Preserve the concrete adapter object (and therefore isinstance checks,
    # attach/close behavior, and provider-specific diagnostics) while placing
    # the evidence gate at its sole vendor-neutral Backend protocol boundary.
    backend.invoke = guarded_invoke
    setattr(backend, _GUARDED_ATTRIBUTE, True)

    original_chat = getattr(backend, "chat_with_tools", None)
    if callable(original_chat) and not getattr(
        backend,
        _CHAT_GUARDED_ATTRIBUTE,
        False,
    ):
        @wraps(original_chat)
        def guarded_chat(*args: Any, **kwargs: Any) -> InvocationResult:
            result = original_chat(*args, **kwargs)
            # Some adapters implement chat_with_tools by delegating to invoke.
            # Admission is idempotent for an existing typed failure, so the
            # peer protocol boundary remains guarded without special-casing
            # individual transports.
            return admit_configured_result(backend, result)

        backend.chat_with_tools = guarded_chat
        setattr(backend, _CHAT_GUARDED_ATTRIBUTE, True)


def admit_configured_result(
    backend: Any,
    result: InvocationResult,
) -> InvocationResult:
    """Apply the route-bound policy, defaulting direct probes to diagnostic."""

    mode = getattr(backend, _MODE_ATTRIBUTE, "diagnostic")
    return admit_result_evidence(result, mode=mode)


__all__ = [
    "admit_configured_result",
    "bind_result_evidence_admission",
]
