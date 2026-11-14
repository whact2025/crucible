# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Workflow-facing adapters over Assay's transactional Corpus API."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Literal

from assay.canonical import hash_record
from assay.corpus.store import CommitRefs, ConcurrentSpecError, Corpus
from assay.types.artifacts import LifecycleEvent
from assay.types.identity import ContentHash
from assay.types.spec import Spec


Attachment = Literal["implementation", "measurement", "verdict"]
StructuralField = Literal["tier_dag"]


def _pending_refs(
    base: tuple[ContentHash, ...],
    staged: tuple[ContentHash, ...],
    *,
    field: str,
) -> tuple[ContentHash, ...]:
    """Return refs added by a draft while rejecting rewritten lineage."""

    if staged[: len(base)] != base:
        raise ValueError(
            f"staged Spec rewrote the existing {field} prefix"
        )
    return staged[len(base) :]


def commit_staged_spec(
    corpus: Corpus,
    *,
    base: Spec,
    staged: Spec,
    structural_fields: tuple[StructuralField, ...] = (),
    lifecycle_event_factory: Callable[[int], LifecycleEvent] | None = None,
    require_exact_base: bool = False,
    publication_guard: Callable[[], None] | None = None,
) -> Spec:
    """Publish an unpublished Spec draft through ``Corpus.commit``.

    Assay primitives may content-address records while deliberately leaving
    their refs on an unpublished next-version Spec.  ``record_refs`` makes
    those already-persisted records visible in one transaction, rebased onto
    the latest manifest.  Only explicitly named structural fields are copied
    from the draft; version, parent/root lineage, and ref collections remain
    owned by ``Corpus.commit``. ``lifecycle_event_factory`` creates an event
    against the exact version selected under the Corpus lock.
    ``require_exact_base`` rejects a concurrent Spec advance instead of
    rebasing derived state reviewed against older authority.
    ``publication_guard`` runs inside that locked transition, immediately
    before structural authority is copied; it must perform no paid calls.
    """

    if staged.schema_version != base.schema_version:
        raise ValueError("staged Spec changed schema_version")

    refs = CommitRefs(
        implementation_refs=_pending_refs(
            base.implementation_refs,
            staged.implementation_refs,
            field="implementation_refs",
        ),
        verdict_refs=_pending_refs(
            base.verdict_refs,
            staged.verdict_refs,
            field="verdict_refs",
        ),
        measurement_refs=_pending_refs(
            base.measurement_refs,
            staged.measurement_refs,
            field="measurement_refs",
        ),
        lifecycle_event_refs=_pending_refs(
            base.lifecycle_event_refs,
            staged.lifecycle_event_refs,
            field="lifecycle_event_refs",
        ),
    )
    changed_fields = tuple(
        field
        for field in dict.fromkeys(structural_fields)
        if getattr(staged, field) != getattr(base, field)
    )

    transform = None
    if changed_fields or require_exact_base or publication_guard is not None:
        values = {field: getattr(staged, field) for field in changed_fields}

        def _apply(candidate: Spec) -> Spec:
            if (
                require_exact_base
                and (
                    candidate.version != base.version + 1
                    or candidate.parent != hash_record(base)
                )
            ):
                raise ConcurrentSpecError(
                    "latest Spec advanced after the reviewed base; refusing "
                    "to rebase derived workflow state"
                )
            if publication_guard is not None:
                publication_guard()
            return replace(candidate, **values)

        transform = _apply

    return corpus.commit(
        record_refs=refs,
        transform=transform,
        lifecycle_event_factory=lifecycle_event_factory,
    ).spec


def persist_and_attach(
    corpus: Corpus,
    record: object,
    *,
    kind: str,
    attachment: Attachment,
) -> tuple[Spec, ContentHash]:
    """Persist ``record`` and atomically attach its ref to the latest Spec.

    Returns the committed (or already-current) Spec and the record ref.
    """
    expected_kind = {
        "implementation": "implementations",
        "measurement": "measurements",
        "verdict": "verdicts",
    }[attachment]
    if kind != expected_kind:
        raise ValueError(
            f"attachment {attachment!r} requires kind {expected_kind!r}, "
            f"got {kind!r}"
        )
    kwargs = {expected_kind: (record,)}
    result = corpus.commit(**kwargs)  # type: ignore[arg-type]
    refs = getattr(result.refs, f"{attachment}_refs")
    return result.spec, refs[0]


def persist_lifecycle_event(
    corpus: Corpus,
    factory: Callable[[int], LifecycleEvent],
) -> tuple[Spec, LifecycleEvent, ContentHash]:
    """Create and attach an event whose affected version is race-correct.

    ``factory`` receives the version that ``with_lifecycle_event`` will
    commit.  This prevents events from citing the stale pre-bump version.
    """
    result, event = corpus.commit_lifecycle_event(factory)
    return result.spec, event, result.refs.lifecycle_event_refs[0]


__all__ = [
    "commit_staged_spec",
    "persist_and_attach",
    "persist_lifecycle_event",
]
