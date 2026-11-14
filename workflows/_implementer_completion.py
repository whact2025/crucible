# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared binding for Assay's implementer completion-report protocol.

Assay captures Git state and conversational evidence independently.  Crucible
must therefore bind the implementer's terminal ``BLOCKED`` disposition to
orchestration policy before a workflow can retain writes, dispatch a paid
recovery, or run a downstream gate. Other well-formed assistant completions
remain non-blocking; this boundary does not independently require ``DONE``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from typing import Final, Iterable, Literal

from assay import Implementation
from assay.corpus.store import Corpus

logger = logging.getLogger(__name__)

IMPLEMENTER_BLOCKED_CLASSIFICATION: Final = "hard_fail_implementer_blocked"
IMPLEMENTER_COMPLETION_CLEAR: Final[Literal["clear"]] = "clear"
IMPLEMENTER_COMPLETION_BLOCKED: Final[Literal["blocked"]] = "blocked"
IMPLEMENTER_COMPLETION_INCOMPLETE: Final[Literal["incomplete"]] = "incomplete"
IMPLEMENTER_TERMINAL_DISPOSITION_ARTIFACT: Final = (
    "implementer_terminal_disposition"
)

ImplementerCompletionDisposition = Literal[
    "clear",
    "blocked",
    "incomplete",
]

_IMPLEMENTER_BLOCKED_RE = re.compile(
    r"^\s*BLOCKED(?=$|\W)"
)
_TRANSCRIPT_ROLES = frozenset({"assistant", "system", "tool", "user"})
_RECLASSIFIABLE_GIT_OUTCOMES = frozenset({
    "soft_fail_no_changes",
    "hard_fail_empty_diff",
})


def _completion_disposition_from_payload(
    transcript: object,
) -> ImplementerCompletionDisposition:
    """Apply the negative-only protocol to one decoded transcript."""

    if not isinstance(transcript, list):
        return IMPLEMENTER_COMPLETION_INCOMPLETE
    if not transcript:
        return IMPLEMENTER_COMPLETION_CLEAR
    terminal = transcript[-1]
    if not isinstance(terminal, dict):
        return IMPLEMENTER_COMPLETION_INCOMPLETE
    role = terminal.get("role")
    if not isinstance(role, str) or role not in _TRANSCRIPT_ROLES:
        return IMPLEMENTER_COMPLETION_INCOMPLETE
    content = terminal.get("content")
    if not isinstance(content, str):
        return IMPLEMENTER_COMPLETION_INCOMPLETE
    if role != "assistant":
        return IMPLEMENTER_COMPLETION_CLEAR
    if _IMPLEMENTER_BLOCKED_RE.match(content) is not None:
        return IMPLEMENTER_COMPLETION_BLOCKED
    return IMPLEMENTER_COMPLETION_CLEAR


def implementer_completion_disposition_from_messages(
    messages: Iterable[object],
) -> ImplementerCompletionDisposition:
    """Classify a typed backend transcript before downstream execution.

    This is the same protocol boundary as
    :func:`implementer_completion_disposition`, applied before Assay performs
    Git capture. It lets a composing workflow honor ``BLOCKED`` before it
    dispatches a deterministic runner without interpreting blocker prose.
    """

    try:
        payload = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]
    except (AttributeError, TypeError):
        return IMPLEMENTER_COMPLETION_INCOMPLETE
    return _completion_disposition_from_payload(payload)


def implementer_completion_disposition(
    corpus: Corpus,
    impl: Implementation,
) -> ImplementerCompletionDisposition:
    """Classify a retained terminal implementer ``BLOCKED`` disposition.

    Only a canonical final assistant message controls the completion-report
    protocol. A valid terminal system/user/tool entry is non-authoritative at
    this negative-only boundary; the independently typed provider/Git outcome
    remains in force. Missing or malformed evidence fails closed instead of
    granting retry or mutation-retention authority. A well-formed assistant
    report that does not begin with the exact uppercase token ``BLOCKED`` is
    likewise clear; this boundary deliberately does not require ``DONE``.
    """

    try:
        transcript = json.loads(corpus.get_artifact(impl.transcript_ref))
    except (KeyError, OSError, TypeError, UnicodeDecodeError, ValueError):
        logger.warning(
            "could not inspect implementer transcript %s for a terminal "
            "completion report",
            impl.transcript_ref,
        )
        return IMPLEMENTER_COMPLETION_INCOMPLETE
    return _completion_disposition_from_payload(transcript)


def normalize_implementer_completion(
    corpus: Corpus,
    impl: Implementation,
    *,
    diagnostic_cost_policy: str | None = None,
) -> tuple[Implementation, ImplementerCompletionDisposition]:
    """Bind completion evidence to classification and a halt disposition.

    Git-success/no-change classifications are refined when the terminal
    evidence is blocked or incomplete. Stronger provider, accounting, scope,
    and workflow classifications remain primary; the content-addressed
    disposition artifact retains the independent completion attribution.
    """

    observed_classification = impl.classification
    # Accounting completeness is already present on the typed Assay record at
    # this boundary. It outranks a Git-only success/no-change/empty result,
    # while causally stronger provider and scope failures remain intact.
    from ._diagnostic_cost import diagnostic_call_covered

    covered = diagnostic_call_covered(impl, policy=diagnostic_cost_policy)
    if impl.accounting_debt and not covered and (
        impl.classification.startswith("success")
        or impl.classification in _RECLASSIFIABLE_GIT_OUTCOMES
    ):
        impl = replace(
            impl,
            classification="hard_fail_incomplete_evidence",
        )

    disposition = implementer_completion_disposition(corpus, impl)
    if disposition == IMPLEMENTER_COMPLETION_CLEAR:
        return impl, disposition
    if disposition == IMPLEMENTER_COMPLETION_INCOMPLETE:
        logger.warning(
            "implementer transcript %s is incomplete; suppressing paid "
            "recovery",
            impl.transcript_ref,
        )
        replacement_classification = "hard_fail_incomplete_evidence"
    else:
        logger.warning(
            "slice %s reported the implementer protocol's terminal BLOCKED "
            "disposition; suppressing paid recovery",
            impl.slice_id,
        )
        replacement_classification = IMPLEMENTER_BLOCKED_CLASSIFICATION

    if (not impl.accounting_debt or covered) and (
        impl.classification.startswith("success")
        or impl.classification in _RECLASSIFIABLE_GIT_OUTCOMES
    ):
        impl = replace(
            impl,
            classification=replacement_classification,
        )

    disposition_ref = corpus.put_artifact(
        json.dumps(
            {
                "schema_version": "crucible.implementer-terminal.v1",
                "slice_id": impl.slice_id,
                "disposition": disposition,
                "observed_classification": observed_classification,
                "retained_classification": impl.classification,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        kind="logs",
        name_hint=f"{impl.slice_id}_implementer_terminal.json",
    )
    return replace(
        impl,
        artifacts={
            **(impl.artifacts or {}),
            IMPLEMENTER_TERMINAL_DISPOSITION_ARTIFACT: disposition_ref,
        },
    ), disposition


def is_terminal_implementer_disposition(
    disposition: ImplementerCompletionDisposition,
) -> bool:
    """Return whether the disposition revokes continuation authority."""

    return disposition != IMPLEMENTER_COMPLETION_CLEAR


__all__ = [
    "IMPLEMENTER_BLOCKED_CLASSIFICATION",
    "IMPLEMENTER_COMPLETION_BLOCKED",
    "IMPLEMENTER_COMPLETION_CLEAR",
    "IMPLEMENTER_COMPLETION_INCOMPLETE",
    "IMPLEMENTER_TERMINAL_DISPOSITION_ARTIFACT",
    "ImplementerCompletionDisposition",
    "implementer_completion_disposition",
    "implementer_completion_disposition_from_messages",
    "is_terminal_implementer_disposition",
    "normalize_implementer_completion",
]
