# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Build + verify helper for workflows that wrap ``assay.implement``.

``assay.implement`` deliberately stops at one backend invocation + diff
capture (see ``assay.primitives.implement``). Build and verify are the
workflow layer's responsibility. This module is the canonical place
workflows look for them. Tier and breach compose these helpers instead of
reimplementing subprocess plumbing.

Runner protocol (mirrors ``assay.bench``):

- Spec carries ``project_context.runners`` mapping role → dotted module
  path. Recognized roles: ``build``, ``verify`` (or ``test`` —
  ``verify`` wins when both present).
- The helper starts one subprocess per role in the gated workspace and
  passes the BenchConfig-shaped JSON payload on stdin. ``crucible.*``
  runners start isolated through the trusted bootstrap. Any other runner
  starts as ``python -P -m <runner>``, and the empty and relative
  ``PYTHONPATH`` entries of its environment are removed, because Python
  resolves them against the working directory. Neither the launch nor such
  an entry adds the working directory to ``sys.path``, so the runner must be
  importable from the environment (installed, or under an absolute
  ``PYTHONPATH`` entry). Apart from the Git hook override, the rest of the
  environment reaches the runner unchanged.
- The workflow injects both ``parameters.role`` and
  ``parameters.runner_mode``.  The former selects the build/verify gate;
  the latter distinguishes ordinary tests from a requested benchmark
  without teaching the workflow about Cargo, pytest, or any other adapter.
- The runner emits ``{"values": {...}, "raw_output": "..."}`` on stdout
  and exits 0 on success, non-zero on hard failure.
- A runner that accepts ``runner_mode == "bench"`` must also echo
  ``{"runner_mode": "bench"}`` at the top level of its result.  Missing
  acknowledgement fails closed instead of letting a role-unaware adapter
  silently run tests and claim that the benchmark gate passed.

Classification matrix (returned by ``build_and_verify``):

- ``success``        — build returned 0 AND verify returned 0 (or ordinary
                       test verification is unconfigured). An explicitly
                       requested benchmark never takes the unconfigured
                       success path.
- ``build_failed``   — build returned non-zero. No verify attempted.
- ``verify_failed``  — build passed, verify returned non-zero.
- ``partial``        — verify returned 0 but the values map shows at
                       least one ``tests_failed > 0``. Pytest-style
                       runners can emit partial passes; the workflow
                       decides whether that's blocking.
- ``runner_missing`` — Spec did not declare a ``build`` runner.

The classifications above replace the coarser ``assay.implement``
classifications (``success`` / ``soft_fail_no_changes`` /
``hard_fail_<error>``) for the workflow layer: a workflow that wraps
``assay.implement`` then calls ``build_and_verify`` produces a
typed Implementation with the *richer* classification overwritten.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import select
import selectors
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import psutil

from crucible.common.runners.capabilities import (
    DECLARED_VERIFICATION_CHECKPOINT_V1,
    trusted_runner_has_capability,
)
from crucible.common.process_supervisor_contract import (
    TargetHeldDataDescriptor,
    encode_target_held_data_descriptors,
)
from crucible.common.runners.parameters import for_slice, validate_static
from crucible.common.runners.declared_verification import (
    ACCEPTED_VERIFICATION_REFERENCE_KEY,
    DECLARED_VERIFICATION_REFERENCE_OUTPUT_KEY,
    DECLARED_VERIFICATION_SURFACE_KEY,
    accepted_verification_reference_payload,
    parse_accepted_verification_reference,
    surface_from_directives,
)
from crucible.tools._buildlock import build_lock

from ._process_supervisor_contract import (
    acquire_trusted_supervisor_source,
    open_target_working_directory,
    sealed_target_environment_descriptor,
    supervisor_environment,
    supervisor_interpreter_path,
)
from ._run import write_atomic_bytes
from ._writer_epoch import (
    WriterEpochArmError,
    WriterEpochBoundary,
    WriterEpochLease,
    WriterEpochStorage,
    arm_writer_epoch_family,
    capture_writer_epoch_owner,
    prepare_writer_epoch_arm,
    process_supervisor_settlement_receipt,
    quiesce_writer_epoch_family,
    require_writer_epoch_quiescence,
    retry_writer_epoch_arm,
    synchronous_callback_settlement_receipt,
)

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")

_RUNTIME_METADATA_KEYS = frozenset({"deepswe_metadata", "swebench_metadata"})
ACCEPTED_VERIFICATION_CHECKPOINT_ARTIFACT = (
    "accepted_declared_verification_checkpoint"
)
_ACCEPTED_VERIFICATION_CHECKPOINT_SCHEMA = (
    "crucible.accepted-declared-verification-checkpoint.v1"
)
_ACCEPTED_VERIFICATION_CHECKPOINT_MAX_BYTES = 1_100_000

class _TrustedRunnerOverrides(dict):
    """Workflow-owned probe/control fields, never a public CLI envelope."""


def trusted_runner_overrides(parameters: dict | None) -> dict:
    """Mark internally derived runner controls after workflow validation."""

    return _TrustedRunnerOverrides(parameters or {})


def supports_declared_verification_checkpoints(spec) -> bool:
    """Return whether the configured dev-verifier owns checkpoint protocol v1."""

    runner = (getattr(spec.project_context, "runners", None) or {}).get(
        "dev_verify"
    )
    return trusted_runner_has_capability(
        runner,
        DECLARED_VERIFICATION_CHECKPOINT_V1,
    )


_SPEC_APPEND_ONLY_REF_FIELDS = (
    "implementation_refs",
    "verdict_refs",
    "measurement_refs",
    "lifecycle_event_refs",
)


def _validate_spec_manifest_identity(corpus, spec) -> tuple[bool, str]:
    """Bind a hydrated Spec to Assay's authenticated persisted identity.

    Legacy canonical manifests may omit fields that Assay hydrates with
    schema defaults.  Their authority is the digest of those exact persisted
    canonical bytes, not ``hash_record(hydrated_spec)``.  Delegate bounded,
    nonblocking, no-follow reading and filename/topology authentication to
    Assay's public atomic API, then compare the hydrated value separately.
    """

    version = getattr(spec, "version", None)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        return False, "Spec version is invalid"
    try:
        persisted, _digest = corpus.get_spec_with_digest(version)
    except Exception as exc:  # noqa: BLE001 - authority ambiguity fails closed
        return False, (
            "Spec canonical manifest is unavailable or invalid: "
            f"{type(exc).__name__}"
        )
    if persisted != spec:
        return False, "Spec canonical manifest hydrates to different content"
    return True, ""


def _validate_spec_lineage_step(corpus, spec) -> tuple[bool, str]:
    """Validate one canonical Spec edge and every append-only ref field."""

    from assay import ContentHash  # noqa: PLC0415

    version = getattr(spec, "version", None)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        return False, "Spec version is invalid"
    if version == 0:
        if getattr(spec, "parent", None) is not None:
            return False, "initial Spec has a parent"
        if getattr(spec, "root", None) is not None:
            return False, "initial Spec has a root"
        return True, ""
    try:
        previous, previous_hash = corpus.get_spec_with_digest(version - 1)
    except Exception as exc:  # noqa: BLE001 - lineage ambiguity fails closed
        return False, f"parent Spec is unavailable: {type(exc).__name__}"
    if getattr(previous, "version", None) != version - 1:
        return False, "parent Spec version is inconsistent"
    previous_hash = ContentHash(previous_hash)
    if getattr(spec, "parent", None) != previous_hash:
        return False, "Spec parent hash is inconsistent"
    expected_root = (
        previous.root if previous.root is not None else previous_hash
    )
    if getattr(spec, "root", None) != expected_root:
        return False, "Spec root hash is inconsistent"
    for field in _SPEC_APPEND_ONLY_REF_FIELDS:
        parent_refs = tuple(str(ref) for ref in getattr(previous, field, ()))
        child_refs = tuple(str(ref) for ref in getattr(spec, field, ()))
        if child_refs[:len(parent_refs)] != parent_refs:
            return False, f"Spec {field} is not append-only"
    return True, ""


def _validate_spec_lineage(
    corpus,
    spec,
    *,
    allow_staged_tip: bool = False,
) -> tuple[bool, str]:
    """Validate a persisted lineage, optionally including one staged tip.

    Architect's deterministic validation operates on an unpublished
    next-version draft and later publishes it through ``Corpus.commit``, which
    owns the final raw-digest parent/root lineage.  Read-only workspace
    reconstruction may recognize that draft only when it is exactly one
    version beyond the authenticated latest manifest, preserves every ref
    prefix, and carries a parent/root spelling produced from either the raw
    manifest digest or the hydrated pure-Spec helper.  Mutation and terminal
    callers retain the persisted-only default.
    """

    if allow_staged_tip:
        try:
            latest = corpus.latest_spec()
        except Exception as exc:  # noqa: BLE001 - ambiguity fails closed
            return False, (
                "staged Spec base is unavailable: "
                f"{type(exc).__name__}"
            )
        if latest is not None and spec.version == latest.version + 1:
            from assay import ContentHash  # noqa: PLC0415
            from assay.canonical import hash_record  # noqa: PLC0415

            try:
                persisted_latest, latest_digest = (
                    corpus.get_spec_with_digest(latest.version)
                )
            except Exception as exc:  # noqa: BLE001 - ambiguity fails closed
                return False, (
                    "staged Spec base manifest is unavailable: "
                    f"{type(exc).__name__}"
                )
            if persisted_latest != latest:
                return False, "staged Spec base differs from raw authority"
            if spec.schema_version != latest.schema_version:
                return False, "staged Spec changed schema_version"
            if not _spec_refs_are_prefix(latest, spec):
                return False, "staged Spec refs are not append-only"
            for field in (
                "implementation_refs",
                "measurement_refs",
                "lifecycle_event_refs",
            ):
                if getattr(spec, field) != getattr(latest, field):
                    return False, (
                        f"staged Spec changed workflow-owned {field}"
                    )
            raw_parent = ContentHash(str(latest_digest))
            hydrated_parent = ContentHash(hash_record(latest))
            if spec.parent not in {raw_parent, hydrated_parent}:
                return False, "staged Spec parent is unrelated to latest"
            expected_roots = (
                {latest.root}
                if latest.root is not None
                else {raw_parent, hydrated_parent}
            )
            if spec.root not in expected_roots:
                return False, "staged Spec root is unrelated to latest"
            return _validate_spec_lineage(corpus, latest)

    current = spec
    while True:
        valid, reason = _validate_spec_manifest_identity(corpus, current)
        if not valid:
            return False, f"Spec v{current.version} manifest is invalid: {reason}"
        valid, reason = _validate_spec_lineage_step(corpus, current)
        if not valid:
            return False, f"Spec v{current.version} lineage is invalid: {reason}"
        if current.version == 0:
            return True, ""
        try:
            current, _digest = corpus.get_spec_with_digest(current.version - 1)
        except Exception as exc:  # noqa: BLE001 - history ambiguity fails closed
            return False, (
                f"Spec v{current.version - 1} is unavailable: "
                f"{type(exc).__name__}"
            )


def _spec_refs_are_prefix(accepted_spec, current_spec) -> bool:
    return all(
        tuple(
            str(ref) for ref in getattr(current_spec, field, ())
        )[:len(getattr(accepted_spec, field, ()))]
        == tuple(str(ref) for ref in getattr(accepted_spec, field, ()))
        for field in _SPEC_APPEND_ONLY_REF_FIELDS
    )


def _tier_lifecycle_acceptance(
    corpus,
    spec,
    tier,
    landed: Mapping[str, object],
    *,
    allow_staged_spec: bool = False,
    artifact_reader=None,
) -> tuple[bool, str]:
    """Prove that ``landed`` is the exact tier state lifecycle-accepted.

    Slice resume records are provisional until the tier publishes its terminal
    ``landed`` event.  Bind that event to both the canonical execution
    structure and the exact Implementation records visible in the Spec version
    named by ``affected_versions``.  A later abandoned event or a newer slice
    record without a matching tier landing therefore cannot authorize a
    downstream tier.
    """

    from assay import LifecycleEvent  # noqa: PLC0415
    from assay.canonical import hash_record  # noqa: PLC0415

    valid_lineage, lineage_reason = _validate_spec_lineage(
        corpus,
        spec,
        allow_staged_tip=allow_staged_spec,
    )
    if not valid_lineage:
        return False, f"current Spec lineage is invalid: {lineage_reason}"

    latest: tuple[object, LifecycleEvent, str] | None = None
    for ref in getattr(spec, "lifecycle_event_refs", ()):
        try:
            event = corpus.get_record(ref, as_type=LifecycleEvent)
        except Exception as exc:  # noqa: BLE001 - corrupt evidence fails closed
            return False, (
                "attached lifecycle evidence is unreadable: "
                f"{type(exc).__name__}"
            )
        execution = getattr(event, "execution", None)
        if getattr(execution, "recipe", None) != "tier":
            continue
        from ._mutation_journal import (  # noqa: PLC0415
            MutationJournalError,
            mutation_lifecycle_reason_base,
        )

        try:
            reason = mutation_lifecycle_reason_base(event.reason)
        except MutationJournalError:
            continue
        if not isinstance(reason, str) or not reason.startswith("tier "):
            continue
        event_tier_id, separator, status = reason[5:].rpartition(": ")
        if not separator or event_tier_id != tier.tier_id:
            continue
        latest = (ref, event, status)

    if latest is None:
        return False, "no tier lifecycle event is attached"
    event_ref, event, status = latest
    if event.kind != "landed" or status != "complete":
        return False, (
            "latest tier lifecycle is not landed: "
            f"kind={event.kind!r}, status={status!r}"
        )
    if (
        len(event.affected_versions) != 1
        or isinstance(event.affected_versions[0], bool)
        or not isinstance(event.affected_versions[0], int)
        or event.affected_versions[0] < 0
        or event.affected_versions[0] > spec.version
    ):
        return False, "landed lifecycle has invalid affected_versions"

    accepted_version = event.affected_versions[0]
    try:
        accepted_spec = corpus.get_spec(accepted_version)
    except Exception as exc:  # noqa: BLE001 - corrupt evidence fails closed
        return False, f"landed lifecycle Spec is unavailable: {type(exc).__name__}"
    valid_lineage, lineage_reason = _validate_spec_lineage(
        corpus,
        accepted_spec,
    )
    if not valid_lineage:
        return False, f"landed lifecycle Spec lineage is invalid: {lineage_reason}"
    if not _spec_refs_are_prefix(accepted_spec, spec):
        return False, "landed lifecycle Spec refs are not a current prefix"
    accepted_lifecycle_refs = tuple(
        str(ref) for ref in accepted_spec.lifecycle_event_refs
    )
    rendered_event_ref = str(event_ref)
    if rendered_event_ref not in accepted_lifecycle_refs:
        return False, "landed lifecycle is not attached to its affected Spec"
    if accepted_version == 0:
        previous_lifecycle_refs: tuple[str, ...] = ()
    else:
        try:
            previous_spec = corpus.get_spec(accepted_version - 1)
        except Exception as exc:  # noqa: BLE001 - lineage ambiguity fails closed
            return False, (
                "landed lifecycle parent Spec is unavailable: "
                f"{type(exc).__name__}"
            )
        previous_lifecycle_refs = tuple(
            str(ref) for ref in previous_spec.lifecycle_event_refs
        )
    if rendered_event_ref not in accepted_lifecycle_refs[
        len(previous_lifecycle_refs):
    ]:
        return False, (
            "landed lifecycle was not first attached at its affected Spec"
        )

    from . import _partition  # noqa: PLC0415 - avoids workflow import cycle

    if (
        _partition.spec_execution_structure_hash(accepted_spec)
        != _partition.spec_execution_structure_hash(spec)
    ):
        return False, "landed lifecycle belongs to different execution authority"
    if artifact_reader is None:
        accepted_landed, accepted_pending = _partition.partition_for_resume(
            corpus,
            tier,
            spec=accepted_spec,
        )
    else:
        accepted_landed, accepted_pending = _partition.partition_for_resume(
            corpus,
            tier,
            spec=accepted_spec,
            artifact_reader=artifact_reader,
        )
    canonical_ids = {item.id for item in tier.slices}
    if accepted_pending or set(accepted_landed) != canonical_ids:
        return False, "landed lifecycle Spec does not contain the complete tier"
    accepted_hashes = {
        slice_id: hash_record(implementation)
        for slice_id, implementation in accepted_landed.items()
    }
    current_hashes = {
        slice_id: hash_record(implementation)
        for slice_id, implementation in landed.items()
    }
    if accepted_hashes != current_hashes:
        return False, "landed lifecycle does not bind the latest slice records"
    return True, ""


def tier_has_plan_bound_landed_lifecycle(corpus, spec, tier) -> bool:
    """Return whether Architect may safely skip this completed tier."""

    from . import _partition  # noqa: PLC0415 - avoids workflow import cycle

    landed, remaining = _partition.partition_for_resume(
        corpus,
        tier,
        spec=spec,
    )
    if remaining or set(landed) != {item.id for item in tier.slices}:
        return False
    accepted, _reason = _tier_lifecycle_acceptance(
        corpus,
        spec,
        tier,
        landed,
    )
    return accepted


def accepted_plan_verification_directives(
    corpus,
    spec,
    *,
    current_tier_id: str,
    ignore_landed_later: bool = False,
) -> tuple[object, ...]:
    """Return the canonical verification surface preceding ``current_tier``.

    Every fully landed canonical tier before the active tier is part of the
    surface.  This is intentionally broader than the formal dependency closure:
    benchmark tiers share one sequential workspace, so an independent earlier
    tier can still supply code which the active tier must not regress.  Formal
    transitive dependencies are stricter still -- a partially landed ancestor
    is invalid execution authority and fails setup rather than silently
    weakening the cumulative gate.  The active tier is always included from
    canonical Spec data; callers cannot provide directives.
    """

    dag = getattr(spec, "tier_dag", None)
    if dag is None:
        raise ValueError("declared verification requires a canonical TierDAG")
    ordered = tuple(dag.topo_order())
    tiers = {tier.tier_id: tier for tier in ordered}
    current = tiers.get(current_tier_id)
    if current is None:
        raise ValueError(
            f"declared verification tier {current_tier_id!r} is not canonical"
        )

    ancestors: set[str] = set()
    pending = list(current.depends_on)
    while pending:
        tier_id = pending.pop()
        if tier_id in ancestors:
            continue
        ancestor = tiers.get(tier_id)
        if ancestor is None:
            raise ValueError(
                f"declared verification dependency {tier_id!r} is missing"
            )
        ancestors.add(tier_id)
        pending.extend(ancestor.depends_on)

    from . import _partition  # noqa: PLC0415 - avoids workflow import cycle

    directives: list[object] = []
    reached_current = False
    for tier in ordered:
        if tier.tier_id == current_tier_id:
            directives.extend(item.directive for item in tier.slices)
            reached_current = True
            continue
        landed, remaining = _partition.partition_for_resume(
            corpus,
            tier,
            spec=spec,
        )
        if reached_current:
            if landed and not ignore_landed_later:
                raise ValueError(
                    "declared verification found landed slices in later "
                    f"canonical tier {tier.tier_id!r}; out-of-order shared "
                    "workspace state has no accepted predecessor checkpoint"
                )
            continue
        if tier.tier_id in ancestors and remaining:
            pending_ids = [item.id for item in remaining]
            raise ValueError(
                "declared verification dependency "
                f"{tier.tier_id!r} is only partially landed; "
                f"pending slices={pending_ids!r}"
            )
        canonical_slice_ids = {item.id for item in tier.slices}
        if not set(landed).issubset(canonical_slice_ids):
            raise ValueError(
                "declared verification partition for tier "
                f"{tier.tier_id!r} is inconsistent"
            )
        if remaining:
            # A wholly pending independent tier contributes no bytes.  Once
            # any slice lands, however, its shared-workspace state is only
            # provisional until the tier-level lifecycle acceptance exists.
            # Refuse to let a later tier inherit that state without the one
            # durable focused-outcome checkpoint published at tier landing.
            if landed:
                raise ValueError(
                    "declared verification independent predecessor "
                    f"{tier.tier_id!r} is only partially landed and has no "
                    "accepted tier checkpoint"
                )
        else:
            # Preserve canonical tier/slice order.  A fully landed tier's
            # partition must contain every slice, but the directives remain
            # canonical Spec objects rather than evidence-derived payloads.
            if set(landed) != canonical_slice_ids:
                raise ValueError(
                    "declared verification partition for fully landed tier "
                    f"{tier.tier_id!r} is inconsistent"
                )
            accepted, reason = _tier_lifecycle_acceptance(
                corpus,
                spec,
                tier,
                landed,
            )
            if not accepted:
                raise ValueError(
                    "declared verification predecessor "
                    f"{tier.tier_id!r} lacks plan-bound landed lifecycle: "
                    f"{reason}"
                )
            directives.extend(item.directive for item in tier.slices)
    return tuple(directives)


def accepted_predecessor_verification_directives(
    corpus,
    spec,
    *,
    current_tier_id: str,
    ignore_landed_later: bool = False,
) -> tuple[object, ...]:
    """Return only the fully accepted canonical surface before a tier."""

    directives = accepted_plan_verification_directives(
        corpus,
        spec,
        current_tier_id=current_tier_id,
        ignore_landed_later=ignore_landed_later,
    )
    dag = getattr(spec, "tier_dag", None)
    current = next(
        (
            tier
            for tier in dag.topo_order()
            if tier.tier_id == current_tier_id
        ),
        None,
    ) if dag is not None else None
    if current is None:
        raise ValueError(
            f"declared verification tier {current_tier_id!r} is not canonical"
        )
    current_count = len(current.slices)
    if current_count > len(directives):
        raise ValueError("declared verification current tier surface is invalid")
    return directives[:-current_count] if current_count else directives


def declared_verification_reference_from_outcome(
    outcome,
    *,
    verification_directives: tuple[object, ...],
) -> dict[str, object]:
    """Extract one complete trusted focused reference from a successful gate."""

    runner_result = getattr(outcome, "dev_verify", None)
    if (
        runner_result is None
        or not getattr(outcome, "classification", "").startswith("success")
        or runner_result.returncode != 0
    ):
        raise ValueError(
            "accepted verification checkpoint requires a successful "
            "development-verifier gate"
        )
    try:
        payload = json.loads(runner_result.raw_output.decode("utf-8"))
    except (
        AttributeError,
        RecursionError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "development verifier did not emit a JSON accepted reference"
        ) from exc
    raw_reference = (
        payload.get(DECLARED_VERIFICATION_REFERENCE_OUTPUT_KEY)
        if isinstance(payload, dict) else None
    )
    reference = parse_accepted_verification_reference(raw_reference)
    expected_surface = surface_from_directives(verification_directives)
    if reference.predecessor_surface != expected_surface:
        raise ValueError(
            "development verifier accepted reference does not match the "
            "canonical cumulative surface"
        )
    required = runner_result.values.get(
        "dev_verify_declared_test_files_required"
    )
    executed = runner_result.values.get(
        "dev_verify_declared_test_files_executed"
    )
    passed = runner_result.values.get(
        "dev_verify_declared_test_files_passed"
    )
    failed = runner_result.values.get(
        "dev_verify_declared_test_files_failed"
    )
    if (
        required is None
        or required != executed
        or required != passed
        or failed != 0.0
        or required != float(len(reference.outcomes))
    ):
        raise ValueError(
            "development verifier accepted reference is incomplete"
        )
    return accepted_verification_reference_payload(
        predecessor_surface=reference.predecessor_surface,
        language=reference.language,
        parser_kind=reference.parser_kind,
        outcomes=(
            (item.path, item.command_sha256, item.outcome)
            for item in reference.outcomes
        ),
    )


def encode_accepted_verification_checkpoint(
    reference: object,
    *,
    tier_id: str,
    spec_execution_authority_hash: str,
) -> bytes:
    """Bind a runner reference to one lifecycle-published tier authority."""

    parsed = parse_accepted_verification_reference(reference)
    if not isinstance(tier_id, str) or not tier_id or len(tier_id) > 256:
        raise ValueError("accepted verification checkpoint tier_id is invalid")
    if (
        not isinstance(spec_execution_authority_hash, str)
        or not spec_execution_authority_hash.startswith("sha256:")
        or len(spec_execution_authority_hash) != 71
    ):
        raise ValueError(
            "accepted verification checkpoint authority hash is invalid"
        )
    canonical_reference = accepted_verification_reference_payload(
        predecessor_surface=parsed.predecessor_surface,
        language=parsed.language,
        parser_kind=parsed.parser_kind,
        outcomes=(
            (item.path, item.command_sha256, item.outcome)
            for item in parsed.outcomes
        ),
    )
    encoded = json.dumps(
        {
            "schema_version": _ACCEPTED_VERIFICATION_CHECKPOINT_SCHEMA,
            "tier_id": tier_id,
            "spec_execution_authority_hash": spec_execution_authority_hash,
            "reference": canonical_reference,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _ACCEPTED_VERIFICATION_CHECKPOINT_MAX_BYTES:
        raise ValueError("accepted verification checkpoint is too large")
    return encoded


def parse_accepted_verification_checkpoint(
    encoded: bytes,
    *,
    expected_tier_id: str,
    expected_spec_execution_authority_hash: str,
    expected_verification_directives: tuple[object, ...],
) -> dict[str, object]:
    """Validate a durable checkpoint against exact plan/spec authority."""

    if (
        not isinstance(encoded, bytes)
        or len(encoded) > _ACCEPTED_VERIFICATION_CHECKPOINT_MAX_BYTES
    ):
        raise ValueError("accepted verification checkpoint is too large")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("accepted verification checkpoint is not JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "tier_id",
        "spec_execution_authority_hash",
        "reference",
    }:
        raise ValueError("accepted verification checkpoint has invalid fields")
    if payload.get("schema_version") != _ACCEPTED_VERIFICATION_CHECKPOINT_SCHEMA:
        raise ValueError("accepted verification checkpoint schema is unsupported")
    if payload.get("tier_id") != expected_tier_id:
        raise ValueError("accepted verification checkpoint tier mismatch")
    if (
        payload.get("spec_execution_authority_hash")
        != expected_spec_execution_authority_hash
    ):
        raise ValueError("accepted verification checkpoint authority mismatch")
    reference = parse_accepted_verification_reference(payload.get("reference"))
    expected_surface = surface_from_directives(
        expected_verification_directives
    )
    if reference.predecessor_surface != expected_surface:
        raise ValueError("accepted verification checkpoint surface mismatch")
    return accepted_verification_reference_payload(
        predecessor_surface=reference.predecessor_surface,
        language=reference.language,
        parser_kind=reference.parser_kind,
        outcomes=(
            (item.path, item.command_sha256, item.outcome)
            for item in reference.outcomes
        ),
    )


def accepted_predecessor_verification_reference(
    corpus,
    spec,
    *,
    current_tier_id: str,
    ignore_landed_later: bool = False,
) -> dict[str, object] | None:
    """Load the lifecycle-bound checkpoint for the exact predecessor surface."""

    predecessor_directives = accepted_predecessor_verification_directives(
        corpus,
        spec,
        current_tier_id=current_tier_id,
        ignore_landed_later=ignore_landed_later,
    )
    if not predecessor_directives:
        return None
    dag = getattr(spec, "tier_dag", None)
    if dag is None:
        raise ValueError("accepted verification requires a canonical TierDAG")
    from . import _partition  # noqa: PLC0415 - avoids workflow import cycle

    last_tier = None
    last_landed = None
    for tier in dag.topo_order():
        if tier.tier_id == current_tier_id:
            break
        landed, remaining = _partition.partition_for_resume(
            corpus,
            tier,
            spec=spec,
        )
        if not remaining:
            last_tier = tier
            last_landed = landed
    if last_tier is None or not last_landed:
        raise ValueError(
            "accepted verification predecessor checkpoint is missing"
        )
    artifact_refs = {
        implementation.artifacts.get(
            ACCEPTED_VERIFICATION_CHECKPOINT_ARTIFACT
        )
        for implementation in last_landed.values()
    }
    if None in artifact_refs or len(artifact_refs) != 1:
        raise ValueError(
            "accepted verification predecessor implementations do not share "
            "one lifecycle checkpoint"
        )
    artifact_ref = next(iter(artifact_refs))
    try:
        encoded = corpus.get_artifact_bounded(
            artifact_ref,
            max_bytes=_ACCEPTED_VERIFICATION_CHECKPOINT_MAX_BYTES,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            "accepted verification predecessor checkpoint is unavailable"
        ) from exc
    return parse_accepted_verification_checkpoint(
        encoded,
        expected_tier_id=last_tier.tier_id,
        expected_spec_execution_authority_hash=(
            _partition.spec_execution_structure_hash(spec)
        ),
        expected_verification_directives=predecessor_directives,
    )

_TRUSTED_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_TRUSTED_RUNNER_BOOTSTRAP = (
    "import runpy,sys;"
    "source_root=sys.argv.pop(1);module=sys.argv.pop(1);"
    "sys.path.insert(0,source_root);"
    "runpy.run_module(module,run_name='__main__',alter_sys=True)"
)

# A trusted runner may recursively launch a language adapter, Cargo, and a
# test binary. Give cooperative teardown a short window before forcing the
# host descendants observed at timeout down. Keep these waits independent and
# bounded: a deliberately detached process may retain a pipe indefinitely.
_RUNNER_TERMINATION_GRACE_S = 5.0
_RUNNER_KILL_GRACE_S = 2.0
_RUNNER_DRAIN_GRACE_S = 1.0
_RUNNER_REAP_GRACE_S = 1.0
_RUNNER_STDOUT_CAPTURE_BYTES = 8 * 1024 * 1024
_RUNNER_STDERR_CAPTURE_BYTES = 1024 * 1024
_RUNNER_STREAM_CHUNK_BYTES = 64 * 1024
_RUNNER_STREAM_POLL_S = 0.1
_RUNNER_COORDINATOR_ABORT_GRACE_S = 0.5
_RUNNER_COORDINATOR_IDLE_S = 30.0
_RUNNER_SUPERVISOR_CONTROL_MAX_BYTES = 4096
_RUNNER_SUPERVISOR_READY_SCHEMA = "crucible.process-supervisor.ready.v1"
_RUNNER_SUPERVISOR_COMMAND_SCHEMA = "crucible.process-supervisor.command.v1"
_RUNNER_SUPERVISOR_SCHEMA = "crucible.process-supervisor.terminal.v1"
_RUNNER_HANDSHAKE_GRACE_S = 10.0
_RUNNER_SUPERVISOR_PATH = Path(__file__).with_name("_process_supervisor.py")


class _RunnerSettlementError(RuntimeError):
    """The trusted subreaper did not prove complete runner settlement."""


class _BoundedRunnerStream:
    """Drain an arbitrary stream while retaining bounded edge evidence."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.edge = max(1, limit // 2)
        self.head = bytearray()
        self.tail = bytearray()
        self.total_bytes = 0
        self.digest = hashlib.sha256()

    @property
    def truncated(self) -> bool:
        return self.total_bytes > self.limit

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        self.digest.update(chunk)
        head_remaining = max(0, self.edge - len(self.head))
        if head_remaining:
            self.head.extend(chunk[:head_remaining])
        remainder = chunk[head_remaining:]
        if remainder:
            self.tail.extend(remainder)
            if len(self.tail) > self.edge:
                del self.tail[:-self.edge]

    def rendered(self) -> bytes:
        if not self.truncated:
            return bytes(self.head + self.tail)
        marker = (
            "\n[crucible runner stream truncated; "
            f"total_bytes={self.total_bytes}; "
            f"sha256={self.digest.hexdigest()}]\n"
        ).encode("ascii")
        # Keep the advertised limit inclusive of the evidence marker. The
        # capture retains half-limit edges while draining; trim only at render
        # time so streams that do not cross the limit remain byte-exact.
        evidence_bytes = max(0, self.limit - len(marker))
        head_bytes = evidence_bytes // 2
        tail_bytes = evidence_bytes - head_bytes
        return (
            bytes(self.head[:head_bytes])
            + marker[:self.limit]
            + (bytes(self.tail[-tail_bytes:]) if tail_bytes else b"")
        )[:self.limit]


def _bounded_runner_bytes(value: bytes, *, limit: int) -> bytes:
    capture = _BoundedRunnerStream(limit)
    capture.feed(value)
    return capture.rendered()


def _append_runner_diagnostic(
    value: bytes,
    diagnostic: bytes,
    *,
    capture_limit: int,
) -> bytes:
    """Append one bounded workflow diagnostic without hiding stream evidence."""

    bounded = (
        value
        if len(value) <= capture_limit
        else _bounded_runner_bytes(value, limit=capture_limit)
    )
    # ``diagnostic`` is a fixed Crucible-authored message, not runner output;
    # the maximum result is the capture limit plus this constant-size suffix.
    return bounded + diagnostic


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerResult:
    """One runner subprocess outcome.

    ``raw_output`` is the runner's stdout (typically JSON, but kept as
    bytes so the workflow can persist via ``corpus.put_artifact`` without
    reserializing). ``stderr`` is captured separately so build errors
    surface even when the runner itself produced no stdout.
    """

    role: str                    # "build" / "verify"
    runner: str                  # dotted module path actually invoked
    returncode: int
    values: dict[str, float]     # parsed from stdout JSON payload
    raw_output: bytes
    stderr: bytes
    duration_s: float
    # Exact digest of the trusted outer supervisor's terminal record.  It is
    # absent only for synthetic no-process outcomes; every launched runner,
    # including a translated timeout or background-descendant failure, must
    # carry it.
    process_settlement_digest: str | None = None

    def __post_init__(self) -> None:
        if (
            self.process_settlement_digest is not None
            and _SHA256_RE.fullmatch(self.process_settlement_digest) is None
        ):
            raise ValueError(
                "runner process settlement digest must be exact sha256"
            )

    @property
    def ok(self) -> bool:
        return self.returncode == 0


# ----------------------------------------------------------------------
# Single-runner subprocess
# ----------------------------------------------------------------------


def _validate_runner_path(runner: str) -> None:
    """Reject runner module paths that aren't dotted Python identifiers.

    Mirrors the discipline in ``assay.primitives.validate._default_resolver``:
    the runner string is fed straight to ``python -m``, so refuse anything
    that could let an attacker-controlled spec.yaml traverse the import
    namespace or inject shell metacharacters.
    """
    parts = runner.split(".")
    if not parts or not all(p.isidentifier() for p in parts):
        raise ValueError(
            f"runner path {runner!r} is not a dotted Python identifier; "
            "reject malformed project_context.runners entries"
        )


def _captured_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")


def _merge_captured(existing: bytes, candidate: bytes | str | None) -> bytes:
    """Merge cumulative/retried communicate output without duplicating it."""

    incoming = _captured_bytes(candidate)
    if not incoming:
        return existing
    if incoming.startswith(existing):
        return incoming
    if existing.startswith(incoming):
        return existing
    # CPython retries return cumulative output, while test doubles and future
    # adapters may return only the newly drained suffix. Preserve both rather
    # than attempting an expensive ambiguous overlap search on large logs.
    return existing + incoming


def _snapshot_runner_tree(
    process: subprocess.Popen[bytes],
) -> tuple[psutil.Process | None, tuple[psutil.Process, ...]]:
    """Snapshot the direct runner and its currently attached descendants."""

    try:
        root = psutil.Process(process.pid)
    except psutil.Error:
        return None, ()
    try:
        descendants = tuple(root.children(recursive=True))
    except psutil.Error:
        descendants = ()
    return root, descendants


def _process_is_executing(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        return True


def _signal_processes(
    processes: tuple[psutil.Process, ...],
    method: str,
) -> None:
    """Best-effort signal of a fixed psutil snapshot."""

    for target in processes:
        try:
            getattr(target, method)()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
        except psutil.Error as exc:
            logger.warning(
                "runner cleanup could not %s pid %d: %s",
                method,
                target.pid,
                exc,
            )


def _wait_processes(
    processes: tuple[psutil.Process, ...],
    timeout: float,
) -> tuple[psutil.Process, ...]:
    if not processes:
        return ()
    try:
        _gone, alive = psutil.wait_procs(
            list(processes),
            timeout=max(0.0, timeout),
        )
    except (psutil.Error, OSError):
        alive = list(processes)
    return tuple(target for target in alive if _process_is_executing(target))


def _bounded_communicate(
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
    stdout: bytes,
    stderr: bytes,
) -> tuple[bytes, bytes, bool]:
    """Drain for at most ``timeout`` while retaining every partial byte."""

    try:
        current_stdout, current_stderr = process.communicate(
            timeout=max(0.0, timeout)
        )
    except subprocess.TimeoutExpired as exc:
        return (
            _merge_captured(stdout, exc.stdout),
            _merge_captured(stderr, exc.stderr),
            False,
        )
    except Exception:  # cleanup diagnostics must not hide ordinary drain errors
        return stdout, stderr, False
    return (
        _merge_captured(stdout, current_stdout),
        _merge_captured(stderr, current_stderr),
        True,
    )


def _close_runner_pipes(process: subprocess.Popen[bytes]) -> None:
    """Release local pipe handles when an escaped holder prevents EOF."""

    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _bounded_reap_direct(
    process: subprocess.Popen[bytes], timeout: float
) -> bool:
    """Reap only Crucible's direct child, never a nonchild grandchild."""

    try:
        process.wait(timeout=max(0.0, timeout))
    except (subprocess.TimeoutExpired, OSError):
        return False
    return True




def _terminate_runner_tree(
    process: subprocess.Popen[bytes],
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    ) -> tuple[bytes, bytes]:
    """Bounded TERM→KILL cleanup for an observed host runner tree.

    ``psutil`` snapshots the direct runner and recursively attached host
    descendants at cleanup entry. Only ``process`` is our child and therefore
    reapable here; waits on grandchildren merely observe their termination.
    A process that deliberately detached/reparented before the snapshot, or a
    workload owned by a container daemon, is outside this lineage boundary.
    Breach-test authoring forbids such detachment. Bounded pipe draining and
    explicit close still ensure an escaped holder cannot stall the workflow.
    """

    root, descendants = _snapshot_runner_tree(process)
    # Signal deepest descendants first so the runner cannot abandon them while
    # reacting to TERM. Keep the runner in the workflow's inherited
    # process group and session; the outer supervisor retains its own kill
    # topology.
    term_targets = tuple(reversed(descendants)) + ((root,) if root else ())
    _signal_processes(term_targets, "terminate")
    if root is None and process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass

    term_deadline = time.monotonic() + _RUNNER_TERMINATION_GRACE_S
    stdout, stderr, _drained = _bounded_communicate(
        process,
        timeout=_RUNNER_TERMINATION_GRACE_S,
        stdout=stdout,
        stderr=stderr,
    )
    remaining = max(0.0, term_deadline - time.monotonic())
    alive_descendants = _wait_processes(descendants, remaining)

    # Kill stubborn descendants before the direct runner. A cooperative runner
    # gets a chance to wait/reap its own children during this bounded window.
    _signal_processes(alive_descendants, "kill")
    alive_descendants = _wait_processes(
        alive_descendants,
        _RUNNER_KILL_GRACE_S,
    )

    if process.poll() is None:
        if root is not None and _process_is_executing(root):
            _signal_processes((root,), "kill")
        else:
            try:
                process.kill()
            except OSError:
                pass

    stdout, stderr, drained = _bounded_communicate(
        process,
        timeout=_RUNNER_DRAIN_GRACE_S,
        stdout=stdout,
        stderr=stderr,
    )
    if not drained:
        _close_runner_pipes(process)
    direct_reaped = _bounded_reap_direct(process, _RUNNER_REAP_GRACE_S)
    if not direct_reaped:
        try:
            process.kill()
        except OSError:
            pass
        _close_runner_pipes(process)
        direct_reaped = _bounded_reap_direct(
            process,
            _RUNNER_REAP_GRACE_S,
        )

    if not direct_reaped:
        logger.warning(
            "runner cleanup could not reap direct child pid %d within the "
            "bounded kill window",
            process.pid,
        )

    alive_descendants = tuple(
        target for target in alive_descendants if _process_is_executing(target)
    )
    if alive_descendants:
        logger.warning(
            "runner cleanup did not observe termination of host descendant "
            "pids %s within the bounded kill window",
            [target.pid for target in alive_descendants],
        )
    return stdout, stderr


def _close_selector_stream(
    selector: selectors.BaseSelector,
    stream,
) -> None:
    try:
        selector.unregister(stream)
    except (KeyError, ValueError):
        pass
    try:
        stream.close()
    except (OSError, ValueError):
        pass


def _bounded_runner_communicate(
    process: subprocess.Popen[bytes],
    *,
    command: list[str],
    config_json: bytes,
    timeout: float | None,
) -> tuple[bytes, bytes, bool, bool] | None:
    """Concurrently drain real subprocess pipes into bounded edge captures.

    ``None`` preserves compatibility with lightweight test doubles that do not
    expose OS-backed file descriptors; production ``Popen`` pipes always take
    this path. The selector continuously drains both output streams, so a
    runner cannot deadlock by filling one pipe before writing the other.
    """

    streams = (process.stdin, process.stdout, process.stderr)
    if any(stream is None for stream in streams):
        return None
    try:
        for stream in streams:
            stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None

    stdin = process.stdin
    stdout = process.stdout
    stderr = process.stderr
    assert stdin is not None and stdout is not None and stderr is not None
    stdout_capture = _BoundedRunnerStream(_RUNNER_STDOUT_CAPTURE_BYTES)
    stderr_capture = _BoundedRunnerStream(_RUNNER_STDERR_CAPTURE_BYTES)
    selector = selectors.DefaultSelector()
    input_offset = 0
    started = time.monotonic()
    deadline = started + timeout if timeout is not None else None
    exited_at: float | None = None

    for stream in (stdin, stdout, stderr):
        os.set_blocking(stream.fileno(), False)
    if config_json:
        selector.register(stdin, selectors.EVENT_WRITE, "stdin")
    else:
        stdin.close()
    selector.register(stdout, selectors.EVENT_READ, "stdout")
    selector.register(stderr, selectors.EVENT_READ, "stderr")

    try:
        while True:
            now = time.monotonic()
            returncode = process.poll()
            if returncode is not None and exited_at is None:
                exited_at = now
                if stdin in (key.fileobj for key in selector.get_map().values()):
                    _close_selector_stream(selector, stdin)

            output_open = any(
                key.data in {"stdout", "stderr"}
                for key in selector.get_map().values()
            )
            if returncode is not None and not output_open:
                break
            if (
                returncode is not None
                and exited_at is not None
                and now - exited_at >= _RUNNER_DRAIN_GRACE_S
            ):
                # A successful runner may have left an unrelated descendant
                # holding the inherited pipe. Retain what was drained and
                # release our descriptors rather than waiting without bound.
                _close_selector_stream(selector, stdout)
                _close_selector_stream(selector, stderr)
                break
            if deadline is not None and now >= deadline:
                captured_stdout = stdout_capture.rendered()
                captured_stderr = stderr_capture.rendered()
                _close_selector_stream(selector, stdin)
                _close_selector_stream(selector, stdout)
                _close_selector_stream(selector, stderr)
                raise subprocess.TimeoutExpired(
                    command,
                    timeout,
                    output=captured_stdout,
                    stderr=captured_stderr,
                )

            wait = _RUNNER_STREAM_POLL_S
            if deadline is not None:
                wait = min(wait, max(0.0, deadline - now))
            if exited_at is not None:
                wait = min(
                    wait,
                    max(0.0, _RUNNER_DRAIN_GRACE_S - (now - exited_at)),
                )
            for key, _mask in selector.select(wait):
                stream = key.fileobj
                if key.data == "stdin":
                    try:
                        written = os.write(
                            stream.fileno(),
                            config_json[
                                input_offset:
                                input_offset + _RUNNER_STREAM_CHUNK_BYTES
                            ],
                        )
                    except BrokenPipeError:
                        _close_selector_stream(selector, stream)
                        continue
                    input_offset += written
                    if input_offset >= len(config_json):
                        _close_selector_stream(selector, stream)
                    continue
                try:
                    chunk = os.read(
                        stream.fileno(),
                        _RUNNER_STREAM_CHUNK_BYTES,
                    )
                except BlockingIOError:
                    continue
                if not chunk:
                    _close_selector_stream(selector, stream)
                    continue
                capture = (
                    stdout_capture if key.data == "stdout" else stderr_capture
                )
                capture.feed(chunk)
    finally:
        selector.close()

    process.wait(timeout=0)
    return (
        stdout_capture.rendered(),
        stderr_capture.rendered(),
        stdout_capture.truncated,
        stderr_capture.truncated,
    )


def _runner_epoch_binding(
    lease: WriterEpochLease | None,
) -> dict[str, object] | None:
    if lease is None:
        return None
    return {
        "epoch_id": lease.epoch_id,
        "generation": lease.generation,
        "family_digest": lease.family_digest,
    }


def _receive_runner_control_packet(
    descriptor: int,
    *,
    timeout: float | None,
) -> bytes:
    """Receive one bounded packet without changing the descriptor mode."""

    control: socket.socket | None = None
    try:
        control = socket.socket(fileno=descriptor)
        if control.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != (
            socket.SOCK_SEQPACKET
        ):
            raise _RunnerSettlementError(
                "runner supervisor control transport is not seqpacket"
            )
        if timeout is not None:
            readable, _writable, _exceptional = select.select(
                (control,), (), (), max(0.0, timeout)
            )
            if not readable:
                raise _RunnerSettlementError(
                    "runner supervisor control packet timed out"
                )
        try:
            payload = control.recv(_RUNNER_SUPERVISOR_CONTROL_MAX_BYTES + 1)
        except OSError as exc:
            raise _RunnerSettlementError(
                "runner supervisor control record is unavailable"
            ) from exc
        if not payload:
            raise _RunnerSettlementError(
                "runner supervisor control record is unavailable"
            )
        if len(payload) > _RUNNER_SUPERVISOR_CONTROL_MAX_BYTES:
            raise _RunnerSettlementError(
                "runner supervisor control record exceeded its bound"
            )
    finally:
        control.detach()
    return payload


def _require_runner_control_eof(descriptor: int) -> None:
    """Reject an additional packet and require the supervisor peer to close."""

    control = socket.socket(fileno=descriptor)
    try:
        deadline = time.monotonic() + _RUNNER_REAP_GRACE_S
        while True:
            remaining = deadline - time.monotonic()
            readable, _writable, _exceptional = select.select(
                (control,), (), (), max(0.0, remaining)
            )
            if not readable:
                raise _RunnerSettlementError(
                    "runner supervisor control channel remained open after terminal"
                )
            if control.recv(1):
                raise _RunnerSettlementError(
                    "runner supervisor emitted multiple terminal control packets"
                )
            break
    finally:
        control.detach()


def _decode_runner_control_packet(payload: bytes) -> dict[str, object]:
    """Decode strict JSON without duplicate fields or non-finite values."""

    def _reject_duplicates(pairs):
        record = {}
        for name, value in pairs:
            if name in record:
                raise ValueError("duplicate supervisor control field")
            record[name] = value
        return record

    try:
        record = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _RunnerSettlementError(
            "runner supervisor control record is unavailable or malformed"
        ) from exc
    if not isinstance(record, dict):
        raise _RunnerSettlementError(
            "runner supervisor control record is unavailable or malformed"
        )
    return record


def _read_runner_supervisor_ready(
    descriptor: int,
    *,
    launch_id: str,
    timeout: float | None = _RUNNER_HANDSHAKE_GRACE_S,
) -> None:
    """Require authenticated supervisor readiness before arming a writer."""

    record = _decode_runner_control_packet(
        _receive_runner_control_packet(descriptor, timeout=timeout)
    )
    if (
        set(record) != {"schema_version", "phase", "launch_id"}
        or record.get("schema_version") != _RUNNER_SUPERVISOR_READY_SCHEMA
        or record.get("phase") != "ready"
        or record.get("launch_id") != launch_id
    ):
        raise _RunnerSettlementError("runner supervisor did not prove launch readiness")


def _send_runner_supervisor_command(
    descriptor: int,
    *,
    launch_id: str,
    command: str,
    writer_epoch_lease: WriterEpochLease | None,
) -> None:
    """Send one exact START/CANCEL transition over the private channel."""

    if command not in {"start", "cancel"}:
        raise ValueError("runner supervisor command is unsupported")
    payload = json.dumps(
        {
            "schema_version": _RUNNER_SUPERVISOR_COMMAND_SCHEMA,
            "command": command,
            "launch_id": launch_id,
            "epoch": _runner_epoch_binding(writer_epoch_lease),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    control: socket.socket | None = None
    try:
        control = socket.socket(fileno=descriptor)
        if control.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != (
            socket.SOCK_SEQPACKET
        ):
            raise _RunnerSettlementError(
                "runner supervisor control transport is not seqpacket"
            )
        if control.send(payload) != len(payload):
            raise _RunnerSettlementError(
                "runner supervisor command packet was truncated"
            )
    except OSError as exc:
        raise _RunnerSettlementError(
            "runner supervisor command could not be delivered"
        ) from exc
    finally:
        if control is not None:
            control.detach()


def _read_runner_supervisor_control(
    descriptor: int,
    *,
    launch_id: str,
    writer_epoch_lease: WriterEpochLease | None,
) -> tuple[int, str | None, str, bytes]:
    """Parse one exact, launch- and epoch-bound terminal attestation."""

    payload = _receive_runner_control_packet(descriptor, timeout=None)
    record = _decode_runner_control_packet(payload)
    if (
        not isinstance(record, Mapping)
        or set(record)
        != {
            "schema_version",
            "phase",
            "launch_id",
            "epoch",
            "outcome",
            "target_started",
            "returncode",
            "settled",
            "error",
        }
        or record.get("schema_version") != _RUNNER_SUPERVISOR_SCHEMA
        or record.get("phase") != "terminal"
        or record.get("launch_id") != launch_id
        or record.get("epoch") != _runner_epoch_binding(writer_epoch_lease)
        or record.get("outcome")
        not in {
            "completed",
            "cancelled",
            "timed_out",
            "spawn_failed",
            "protocol_failed",
        }
        or type(record.get("target_started")) is not bool
        or record.get("settled") is not True
        or type(record.get("returncode")) is not int
        or record.get("error") not in {None, "background_descendants"}
        or (
            record.get("target_started") is False
            and record.get("outcome") not in {"cancelled", "timed_out", "spawn_failed"}
        )
    ):
        raise _RunnerSettlementError(
            "runner supervisor did not prove complete descendant settlement"
        )
    _require_runner_control_eof(descriptor)
    return (
        int(record["returncode"]),
        (
            "background_descendants"
            if record.get("error") == "background_descendants"
            else None
        ),
        str(record["outcome"]),
        json.dumps(
            record,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii"),
    )


def _observe_nonzero_runner_supervisor_exit(
    process: subprocess.Popen[bytes],
) -> int | None:
    """Return positive nonzero direct-child evidence after a terminal read fails.

    The retained settlement worker can observe control-channel EOF before any
    other thread has reaped the supervisor.  A bounded direct-child wait closes
    that diagnostic race.  A nonzero status never authorizes settlement; it is
    stronger evidence that the packet failure came from an untrusted producer
    exit and therefore takes precedence while retaining the packet error as its
    cause.
    """

    returncode = process.returncode
    if returncode is None:
        try:
            returncode = process.wait(timeout=_RUNNER_REAP_GRACE_S)
        except (subprocess.TimeoutExpired, OSError):
            # Popen.wait() may publish the direct-child status before an
            # injected/concurrent failure reaches this caller.  Re-read the
            # canonical field so exact nonzero evidence still wins that race.
            returncode = process.returncode
    if type(returncode) is int and returncode != 0:
        return returncode
    return None


def _consume_runner_supervisor_settlement(
    process: subprocess.Popen[bytes],
    control_read: int,
    *,
    launch_id: str,
    writer_epoch_lease: WriterEpochLease | None,
    settlement_owner_ids: tuple[str, ...],
    settlement_worker_ids: tuple[str, ...],
    settlement_callback: Callable[[str], None] | None,
) -> tuple[int, str | None, str]:
    """Compatibility wrapper around one retained settlement transaction."""

    transaction = _RunnerSettlementTransaction(
        process,
        control_read,
        launch_id=launch_id,
        writer_epoch_lease=writer_epoch_lease,
        settlement_owner_ids=settlement_owner_ids,
        settlement_worker_ids=settlement_worker_ids,
        settlement_callback=settlement_callback,
    )
    transaction._settle_from_terminal(command_already_sent=True)
    return transaction.join()


class _RunnerSettlementTransaction:
    """Retain one terminal receive through exact durable quiescence."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        control_read: int,
        *,
        launch_id: str,
        writer_epoch_lease: WriterEpochLease | None,
        settlement_owner_ids: tuple[str, ...],
        settlement_worker_ids: tuple[str, ...],
        settlement_callback: Callable[[str], None] | None,
    ) -> None:
        self._process = process
        self._control_read = control_read
        self._launch_id = launch_id
        self._writer_epoch_lease = writer_epoch_lease
        self._settlement_owner_ids = settlement_owner_ids
        self._settlement_worker_ids = settlement_worker_ids
        self._settlement_callback = settlement_callback
        self._result: list[tuple[int, str | None, str, str]] = []
        self._failure: list[BaseException] = []
        self._done = threading.Event()
        self._callback_lock = threading.Lock()
        self._callback_published = False
        self._interrupt_lock = threading.Lock()
        self._deferred_interrupt: KeyboardInterrupt | None = None
        self._command_lock = threading.Lock()
        self._initial_command: str | None = None
        self._initial_command_claimed = False
        self._command_available = threading.Event()
        self._initial_command_sent = threading.Event()
        self._cancel_requested = False
        self._cancel_sent = threading.Event()
        self._descriptor_closed = False

    def request_start(self) -> None:
        """Publish exact START and wait until the sole worker sends it."""

        pending_interrupt: KeyboardInterrupt | None = None
        with self._command_lock:
            if self._initial_command is not None:
                raise _RunnerSettlementError(
                    "runner supervisor received multiple initial commands"
            )
            self._initial_command = "start"
        while True:
            try:
                self._command_available.set()
                break
            except KeyboardInterrupt as exc:
                pending_interrupt = pending_interrupt or exc
                # No initial packet is deliverable until the wake succeeds.
                # Cancellation observed here replaces START authority with
                # initial CANCEL while it remains unclaimed.  Once the worker
                # has claimed START, publish the follow-up cancellation instead:
                # Event.set() may have completed its wake before surfacing the
                # interrupt, so rewriting the already-claimed command would
                # otherwise strand the live target without a CANCEL packet.
                with self._command_lock:
                    if self._initial_command == "start":
                        if self._initial_command_claimed:
                            self._cancel_requested = True
                        else:
                            self._initial_command = "cancel"
        while not self._initial_command_sent.is_set() and not self._done.is_set():
            try:
                self._initial_command_sent.wait(_RUNNER_STREAM_POLL_S)
            except KeyboardInterrupt as exc:
                pending_interrupt = pending_interrupt or exc
        if (
            not self._initial_command_sent.is_set()
            and not self._cancel_sent.is_set()
        ):
            raise _RunnerSettlementError(
                "runner settlement ended before initial command delivery"
            )
        if pending_interrupt is not None:
            raise pending_interrupt

    def request_cancel(self) -> None:
        """Request one CANCEL; repeated callers can only await that request."""

        pending_interrupt: KeyboardInterrupt | None = None
        self.publish_cancel()
        while not self._cancel_sent.is_set() and not self._done.is_set():
            try:
                self._cancel_sent.wait(_RUNNER_STREAM_POLL_S)
            except KeyboardInterrupt as exc:
                pending_interrupt = pending_interrupt or exc
        if pending_interrupt is not None:
            raise pending_interrupt

    def publish_cancel(self) -> None:
        """Idempotently publish cancellation without waiting for delivery."""

        pending_interrupt: KeyboardInterrupt | None = None
        while True:
            try:
                with self._command_lock:
                    if self._initial_command is None:
                        self._initial_command = "cancel"
                    elif self._initial_command == "start":
                        if self._initial_command_claimed:
                            self._cancel_requested = True
                        else:
                            self._initial_command = "cancel"
                break
            except KeyboardInterrupt as exc:
                pending_interrupt = pending_interrupt or exc
        while True:
            try:
                self._command_available.set()
                break
            except KeyboardInterrupt as exc:
                pending_interrupt = pending_interrupt or exc
        if pending_interrupt is not None:
            raise pending_interrupt

    def _settle_from_terminal(
        self,
        *,
        command_already_sent: bool = False,
    ) -> None:
        """Own irreversible receive through durable quiescence in one worker."""

        try:
            if not command_already_sent:
                initial_command: str | None = None
                while initial_command is None:
                    self._command_available.wait(_RUNNER_STREAM_POLL_S)
                    with self._command_lock:
                        initial_command = self._initial_command
                if initial_command not in {"start", "cancel"}:
                    raise _RunnerSettlementError(
                        "runner settlement command was not published"
                    )
                with self._command_lock:
                    initial_command = self._initial_command
                    if initial_command not in {"start", "cancel"}:
                        raise _RunnerSettlementError(
                            "runner settlement command changed before claim"
                        )
                    self._initial_command_claimed = True
                    _send_runner_supervisor_command(
                        self._control_read,
                        launch_id=self._launch_id,
                        command=initial_command,
                        writer_epoch_lease=self._writer_epoch_lease,
                    )
                self._initial_command_sent.set()
                if initial_command == "cancel":
                    self._cancel_sent.set()
                else:
                    # The same worker owns both command transitions and
                    # terminal receive. Polling the seqpacket peer lets a
                    # controller-side cancellation win without a second
                    # sender racing terminal consumption.
                    while True:
                        with self._command_lock:
                            cancel_requested = self._cancel_requested
                        if cancel_requested:
                            try:
                                _send_runner_supervisor_command(
                                    self._control_read,
                                    launch_id=self._launch_id,
                                    command="cancel",
                                    writer_epoch_lease=self._writer_epoch_lease,
                                )
                            except _RunnerSettlementError as exc:
                                cause = exc.__cause__
                                if not isinstance(
                                    cause, OSError
                                ) or cause.errno not in {
                                    errno.EPIPE,
                                    errno.ECONNRESET,
                                    errno.ENOTCONN,
                                }:
                                    raise
                            finally:
                                self._cancel_sent.set()
                            break
                        readable, _writable, _exceptional = select.select(
                            (self._control_read,),
                            (),
                            (),
                            _RUNNER_STREAM_POLL_S,
                        )
                        if readable:
                            break
            try:
                (
                    returncode,
                    settlement_error,
                    terminal_outcome,
                    settlement_record,
                ) = _read_runner_supervisor_control(
                    self._control_read,
                    launch_id=self._launch_id,
                    writer_epoch_lease=self._writer_epoch_lease,
                )
            except _RunnerSettlementError as terminal_error:
                supervisor_returncode = _observe_nonzero_runner_supervisor_exit(
                    self._process
                )
                if supervisor_returncode is not None:
                    raise _RunnerSettlementError(
                        "runner supervisor exited without trusted settlement "
                        f"(returncode={supervisor_returncode!r})"
                    ) from terminal_error
                raise
            # EOF is part of the exact packet proof; only then may supervisor
            # rc authenticate the producer. Poll/wait here because this worker
            # starts before START and normally receives while Popen is alive.
            if self._process.returncode is None:
                self._process.wait(timeout=_RUNNER_REAP_GRACE_S)
            if self._process.returncode != 0:
                raise _RunnerSettlementError(
                    "runner supervisor exited without trusted settlement "
                    f"(returncode={self._process.returncode!r})"
                )
            settlement_digest = (
                "sha256:" + hashlib.sha256(settlement_record).hexdigest()
            )
            if self._writer_epoch_lease is not None:
                retry_writer_epoch_arm(self._writer_epoch_lease)
                receipt = process_supervisor_settlement_receipt(
                    self._writer_epoch_lease,
                    settled_owner_ids=self._settlement_owner_ids,
                    joined_worker_ids=self._settlement_worker_ids,
                    evidence_digest=settlement_digest,
                    outcome=(
                        "succeeded" if terminal_outcome == "completed" else "failed"
                    ),
                )
                quiesce_writer_epoch_family(self._writer_epoch_lease, receipt)
                require_writer_epoch_quiescence(
                    self._writer_epoch_lease.members,
                    allow_never_armed=False,
                )
            self._result.append(
                (
                    returncode,
                    settlement_error,
                    terminal_outcome,
                    settlement_digest,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - transfer exact failure
            self._failure.append(exc)
        finally:
            self._descriptor_closed = True
            try:
                os.close(self._control_read)
            except OSError:
                pass
            self._done.set()

    def join(
        self,
        *,
        replay_interrupt: bool = True,
    ) -> tuple[int, str | None, str]:
        """Wait through caught SIGINT, publish callback once, then replay it."""

        pending_interrupt: KeyboardInterrupt | None = None
        while not self._done.is_set():
            try:
                self._done.wait(_RUNNER_STREAM_POLL_S)
            except KeyboardInterrupt as exc:
                pending_interrupt = pending_interrupt or exc
                self.defer_interrupt(exc)
        if self._failure:
            failure = self._failure[0]
            if pending_interrupt is not None:
                failure.add_note("SIGINT was deferred during runner settlement")
            raise failure
        if len(self._result) != 1:
            raise _RunnerSettlementError(
                "runner settlement worker produced ambiguous authority"
            )
        returncode, settlement_error, terminal_outcome, settlement_digest = (
            self._result[0]
        )
        with self._callback_lock:
            if (
                self._settlement_callback is not None
                and not self._callback_published
            ):
                # Mark the main-thread publication attempt before invoking
                # user cleanup. A callback that raises must not be repeated by
                # a second join and mutate rollback state twice.
                self._callback_published = True
                self._settlement_callback(settlement_digest)
        if replay_interrupt:
            deferred = self.take_deferred_interrupt()
            if deferred is not None:
                raise deferred
        return returncode, settlement_error, terminal_outcome

    def defer_interrupt(self, interrupt: KeyboardInterrupt) -> None:
        """Retain one main-thread interrupt until a safe replay boundary."""

        with self._interrupt_lock:
            self._deferred_interrupt = self._deferred_interrupt or interrupt

    def take_deferred_interrupt(self) -> KeyboardInterrupt | None:
        """Consume the interrupt retained by a replay-disabled join."""

        with self._interrupt_lock:
            interrupt = self._deferred_interrupt
            self._deferred_interrupt = None
        return interrupt

    @property
    def done(self) -> bool:
        """Whether the sole terminal transaction reached a retained result."""

        return self._done.is_set()

    def close_unclaimed(self) -> None:
        """Release the duplicate descriptor after coordinator abort wins."""

        if self._descriptor_closed:
            return
        # Consume integer-fd ownership before the kernel close. An async
        # exception after close returns must never retry a number that another
        # thread may already have reused for an unrelated descriptor.
        self._descriptor_closed = True
        try:
            os.close(self._control_read)
        except OSError:
            pass


class _RunnerSettlementCoordinator:
    """Prestart one per-call worker before any process or journal exists."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._abort = threading.Event()
        self._job_available = threading.Event()
        self._job_claimed = threading.Event()
        self._exited = threading.Event()
        self._lock = threading.Lock()
        self._transaction: _RunnerSettlementTransaction | None = None
        self._claimed_transaction: _RunnerSettlementTransaction | None = None
        self._worker = threading.Thread(
            target=self._run,
            name="crucible-runner-settlement",
            daemon=False,
        )

    def _run(self) -> None:
        try:
            self._ready.set()
            idle_deadline = time.monotonic() + _RUNNER_COORDINATOR_IDLE_S
            transaction: _RunnerSettlementTransaction | None = None
            while transaction is None:
                with self._lock:
                    if self._abort.is_set():
                        return
                    transaction = self._transaction
                    if transaction is not None:
                        self._claimed_transaction = transaction
                        self._job_claimed.set()
                        break
                if time.monotonic() >= idle_deadline:
                    return
                self._job_available.wait(_RUNNER_STREAM_POLL_S)
            transaction._settle_from_terminal()
        finally:
            self._exited.set()

    def start(self) -> None:
        """Start before external state; never retry an ambiguously started Thread."""

        pending_interrupt: KeyboardInterrupt | None = None
        try:
            self._worker.start()
        except KeyboardInterrupt as exc:
            pending_interrupt = exc
            self._abort.set()
            self._job_available.set()
        if pending_interrupt is None:
            while not self._ready.is_set():
                try:
                    self._ready.wait(_RUNNER_STREAM_POLL_S)
                except KeyboardInterrupt as exc:
                    pending_interrupt = pending_interrupt or exc
                    self._abort.set()
                    self._job_available.set()
        if pending_interrupt is not None:
            # No process, descriptor, or writer lease exists yet.  If a native
            # thread was launched but its bootstrap was delayed, it observes
            # abort and exits without ever receiving external authority.  If
            # no native thread was launched, an exit ack cannot exist.
            deadline = time.monotonic() + _RUNNER_COORDINATOR_ABORT_GRACE_S
            while not self._exited.is_set() and time.monotonic() < deadline:
                try:
                    self._exited.wait(_RUNNER_STREAM_POLL_S)
                except KeyboardInterrupt:
                    continue
            raise pending_interrupt

    def submit(
        self,
        transaction: _RunnerSettlementTransaction,
    ) -> KeyboardInterrupt | None:
        """Publish exactly one transaction and await positive worker claim."""

        pending_interrupt: KeyboardInterrupt | None = None
        with self._lock:
            if self._transaction is not None:
                raise _RunnerSettlementError(
                    "runner settlement coordinator received multiple jobs"
                )
            self._transaction = transaction
        try:
            self._job_available.set()
        except KeyboardInterrupt as exc:
            pending_interrupt = exc
            self._job_available.set()
        while not self._job_claimed.is_set() and not self._exited.is_set():
            try:
                self._job_claimed.wait(_RUNNER_STREAM_POLL_S)
            except KeyboardInterrupt as exc:
                pending_interrupt = pending_interrupt or exc
        if not self._job_claimed.is_set():
            raise _RunnerSettlementError(
                "runner settlement coordinator exited before claiming job"
            )
        return pending_interrupt

    def claimed_transaction(self) -> _RunnerSettlementTransaction | None:
        """Return only positively worker-owned terminal authority."""

        with self._lock:
            return self._claimed_transaction

    def abort_unclaimed_and_wait(self) -> KeyboardInterrupt | None:
        """Atomically abort or recover the transaction that already won."""

        pending_interrupt: KeyboardInterrupt | None = None
        while True:
            try:
                with self._lock:
                    claimed = self._claimed_transaction
                    if claimed is not None:
                        return pending_interrupt
                    self._abort.set()
                    transaction = self._transaction
                self._job_available.set()
                if transaction is not None:
                    transaction._command_available.set()
                if self._exited.is_set():
                    if transaction is not None:
                        transaction.close_unclaimed()
                    return pending_interrupt
                self._exited.wait(_RUNNER_STREAM_POLL_S)
            except KeyboardInterrupt as exc:
                pending_interrupt = pending_interrupt or exc

    def abort_and_wait(self) -> KeyboardInterrupt | None:
        """Abort an unclaimed worker and wait through repeated SIGINT."""

        pending_interrupt: KeyboardInterrupt | None = None
        unconfirmed_deadline = (
            time.monotonic() + _RUNNER_COORDINATOR_ABORT_GRACE_S
        )
        while True:
            try:
                if self._exited.is_set():
                    break
                with self._lock:
                    claimed = self._claimed_transaction
                    transaction = self._transaction
                if claimed is not None:
                    if not claimed.done:
                        claimed.publish_cancel()
                else:
                    self._abort.set()
                    self._job_available.set()
                if transaction is not None and claimed is None:
                    transaction._command_available.set()
                if (
                    not self._ready.is_set()
                    and time.monotonic() >= unconfirmed_deadline
                ):
                    # Thread.start may have raised before native launch. A
                    # delayed native bootstrap still sees the durable abort
                    # and has its own idle expiry, but absence of READY cannot
                    # be treated as positive join authority.
                    break
                self._exited.wait(_RUNNER_STREAM_POLL_S)
            except KeyboardInterrupt as exc:
                pending_interrupt = pending_interrupt or exc
        return pending_interrupt


def _cancel_runner_supervisor(
    process: subprocess.Popen[bytes],
    control_read: int,
    *,
    launch_id: str,
    ready_received: bool,
    writer_epoch_lease: WriterEpochLease | None,
    settlement_owner_ids: tuple[str, ...],
    settlement_worker_ids: tuple[str, ...],
    settlement_callback: Callable[[str], None] | None,
    settlement_grace: float,
    settlement_transaction: _RunnerSettlementTransaction | None = None,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> tuple[bytes, bytes]:
    """Request authenticated cancellation, then consume its terminal proof."""

    terminal_consume_started = False
    physical_teardown_attempted = False
    try:
        if settlement_transaction is not None and settlement_transaction.done:
            terminal_consume_started = True
            settlement_transaction.join(replay_interrupt=False)
            return stdout, stderr
        if settlement_transaction is not None and process.poll() is not None:
            # The worker may have consumed the packet and be finishing durable
            # quiescence while the main thread observes supervisor exit. Wait
            # for that sole retained transaction before attempting a command
            # on a descriptor it owns.
            terminal_consume_started = True
            settlement_transaction.join(replay_interrupt=False)
            return stdout, stderr
        if settlement_transaction is not None:
            settlement_transaction.request_cancel()
            # Once the coordinator owns a transaction, its worker is the sole
            # terminal consumer and the supervisor owns the bounded
            # TERM/KILL/reap window.  Await that authenticated settlement
            # before any outer pipe-drain timeout can trigger physical
            # teardown and strand the transaction's armed writer epoch.
            terminal_consume_started = True
            settlement_transaction.join(replay_interrupt=False)
            stdout, stderr, _drained = _bounded_communicate(
                process,
                timeout=settlement_grace + 0.25,
                stdout=stdout,
                stderr=stderr,
            )
            return stdout, stderr

        delivery_failure: _RunnerSettlementError | None = None
        try:
            _send_runner_supervisor_command(
                control_read,
                launch_id=launch_id,
                command="cancel",
                writer_epoch_lease=writer_epoch_lease,
            )
        except _RunnerSettlementError as exc:
            # The supervisor can win the race by publishing and closing
            # its exact terminal just before cancellation delivery.
            cause = exc.__cause__
            if not isinstance(cause, OSError) or cause.errno not in {
                errno.EPIPE,
                errno.ECONNRESET,
                errno.ENOTCONN,
            }:
                raise
            delivery_failure = exc
        stdout, stderr, drained = _bounded_communicate(
            process,
            timeout=settlement_grace + 0.25,
            stdout=stdout,
            stderr=stderr,
        )
        if not drained and process.poll() is None:
            # The private cancellation path did not reach a terminal packet in
            # its complete supervisor grace. Fall back to physical teardown,
            # but retain the epoch armed: a killed supervisor cannot attest.
            physical_teardown_attempted = True
            stdout, stderr = _terminate_runner_tree(
                process,
                stdout=stdout,
                stderr=stderr,
            )
        elif process.returncode is None:
            process.wait(timeout=_RUNNER_REAP_GRACE_S)
        if not ready_received:
            _read_runner_supervisor_ready(
                control_read,
                launch_id=launch_id,
                timeout=0.0,
            )
        terminal_consume_started = True
        try:
            if settlement_transaction is None:
                _consume_runner_supervisor_settlement(
                    process,
                    control_read,
                    launch_id=launch_id,
                    writer_epoch_lease=writer_epoch_lease,
                    settlement_owner_ids=settlement_owner_ids,
                    settlement_worker_ids=settlement_worker_ids,
                    settlement_callback=settlement_callback,
                )
            else:
                settlement_transaction.join(replay_interrupt=False)
        except BaseException as settlement_error:
            if delivery_failure is not None:
                settlement_error.add_note(
                    "runner CANCEL delivery also lost its terminal race: "
                    f"{delivery_failure}"
                )
            raise
        return stdout, stderr
    except KeyboardInterrupt as interrupt:
        # After a retained transaction exists, SIGINT is only a deferred
        # caller notification.  It must never enter physical teardown while
        # exact terminal authority can still quiesce the writer epoch.
        if settlement_transaction is not None:
            while True:
                try:
                    settlement_transaction.join(replay_interrupt=False)
                    break
                except KeyboardInterrupt:
                    continue
            raise interrupt
        if not terminal_consume_started:
            try:
                os.close(control_read)
            except OSError:
                pass
        if not physical_teardown_attempted and process.poll() is None:
            try:
                _terminate_runner_tree(
                    process,
                    stdout=stdout,
                    stderr=stderr,
                )
            except BaseException:
                pass
        raise
    except BaseException:
        if not terminal_consume_started and settlement_transaction is None:
            try:
                os.close(control_read)
            except OSError:
                pass
        if not physical_teardown_attempted and process.poll() is None:
            try:
                _terminate_runner_tree(
                    process,
                    stdout=stdout,
                    stderr=stderr,
                )
            except BaseException:
                pass
        raise


def _run_runner_subprocess(
    command: list[str],
    *,
    config_json: bytes,
    cwd: str,
    timeout: float | None,
    env: dict[str, str],
    settlement_callback: Callable[[str], None] | None = None,
    writer_epoch_members: tuple[WriterEpochStorage, ...] | None = None,
    target_pass_fds: tuple[TargetHeldDataDescriptor, ...] = (),
) -> subprocess.CompletedProcess[bytes]:
    """Run one trusted adapter under a Linux subreaper settlement boundary."""

    # Establish the sole settlement worker before acquiring descriptors,
    # launching a process, or preparing a writer lease.  Thread-start
    # ambiguity therefore cannot strand any external mutation authority.
    coordinator = _RunnerSettlementCoordinator()
    completed_normally = False
    result: subprocess.CompletedProcess[bytes] | None = None
    try:
        coordinator.start()
        result = _run_runner_subprocess_with_coordinator(
            command,
            config_json=config_json,
            cwd=cwd,
            timeout=timeout,
            env=env,
            settlement_callback=settlement_callback,
            writer_epoch_members=writer_epoch_members,
            target_pass_fds=target_pass_fds,
            coordinator=coordinator,
        )
        completed_normally = True
    finally:
        active_error = sys.exception()
        pending_interrupt = coordinator.abort_and_wait()
        settlement_failure: BaseException | None = None
        claimed = coordinator.claimed_transaction()
        if claimed is not None:
            while True:
                try:
                    claimed.join(replay_interrupt=False)
                    break
                except KeyboardInterrupt as exc:
                    pending_interrupt = pending_interrupt or exc
                except BaseException as exc:
                    settlement_failure = exc
                    break
            pending_interrupt = (
                pending_interrupt or claimed.take_deferred_interrupt()
            )
        if settlement_failure is not None:
            if active_error is None:
                raise settlement_failure
            active_error.add_note(
                "runner coordinator finalization could not authenticate "
                f"settlement: {type(settlement_failure).__name__}: "
                f"{settlement_failure}"
            )
            raise active_error from settlement_failure
        if pending_interrupt is not None:
            if active_error is None:
                raise pending_interrupt
            active_error.add_note(
                "SIGINT was deferred while retiring settlement coordinator"
            )
    if not completed_normally or result is None:
        raise _RunnerSettlementError(
            "runner coordinator returned without a completed result"
        )
    return result


def _run_runner_subprocess_with_coordinator(
    command: list[str],
    *,
    config_json: bytes,
    cwd: str,
    timeout: float | None,
    env: dict[str, str],
    settlement_callback: Callable[[str], None] | None,
    writer_epoch_members: tuple[WriterEpochStorage, ...] | None,
    target_pass_fds: tuple[TargetHeldDataDescriptor, ...],
    coordinator: _RunnerSettlementCoordinator,
) -> subprocess.CompletedProcess[bytes]:
    """Run after the per-call settlement coordinator is positively ready."""

    pending_claim_interrupt: KeyboardInterrupt | None = None

    settlement_grace = min(
        10.0,
        max(
            0.1,
            _RUNNER_TERMINATION_GRACE_S
            + _RUNNER_KILL_GRACE_S
            + _RUNNER_DRAIN_GRACE_S
            + _RUNNER_REAP_GRACE_S,
        ),
    )
    target_timeout_ns = (
        max(1, int(timeout * 1_000_000_000)) if timeout is not None else (1 << 63) - 1
    )
    held_data_spec = encode_target_held_data_descriptors(target_pass_fds)
    target_env_descriptor = sealed_target_environment_descriptor(env)
    try:
        target_cwd_descriptor = open_target_working_directory(cwd)
    except BaseException:
        os.close(target_env_descriptor)
        raise
    try:
        supervisor_source_descriptor, supervisor_source_path = (
            acquire_trusted_supervisor_source()
        )
    except BaseException:
        os.close(target_env_descriptor)
        os.close(target_cwd_descriptor)
        raise
    try:
        controller_control, supervisor_control = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC,
        )
        control_read = controller_control.detach()
        control_write = supervisor_control.detach()
    except BaseException:
        os.close(target_env_descriptor)
        os.close(target_cwd_descriptor)
        os.close(supervisor_source_descriptor)
        raise
    launch_id = f"runner-{secrets.token_hex(16)}"
    supervisor_command = [
        supervisor_interpreter_path(),
        "-I",
        "-S",
        supervisor_source_path,
        str(control_write),
        str(target_env_descriptor),
        str(supervisor_source_descriptor),
        str(target_cwd_descriptor),
        format(settlement_grace, ".17g"),
        str(os.getpid()),
        str(target_timeout_ns),
        "--controller-handshake-v1",
        launch_id,
        *(
            ("--held-data-fds-v1", held_data_spec)
            if target_pass_fds
            else ()
        ),
        "--",
        *command,
    ]
    writer_epoch_lease: WriterEpochLease | None = None
    interrupted_arm_lease: WriterEpochLease | None = None
    settlement_transaction: _RunnerSettlementTransaction | None = None
    settlement_owner_ids = (f"{launch_id}:supervisor",)
    settlement_worker_ids = (f"{launch_id}:target",)
    try:
        process = subprocess.Popen(
            supervisor_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            # The untrusted/project-selected target environment must never
            # influence the interpreter which emits settlement authority.
            # It travels separately through a sealed descriptor and is used
            # only by the supervisor's target Popen.
            env=supervisor_environment(),
            start_new_session=True,
            pass_fds=(
                control_write,
                target_env_descriptor,
                supervisor_source_descriptor,
                target_cwd_descriptor,
                *(item.descriptor for item in target_pass_fds),
            ),
        )
    except BaseException:
        # Popen occurs before READY and before any writer epoch is armed.
        # Launch failure therefore cannot strand mutation authority.
        os.close(control_read)
        os.close(control_write)
        os.close(target_env_descriptor)
        os.close(target_cwd_descriptor)
        os.close(supervisor_source_descriptor)
        raise
    stdout_truncated = False
    ready_received = False
    try:
        os.close(control_write)
        os.close(target_env_descriptor)
        os.close(target_cwd_descriptor)
        os.close(supervisor_source_descriptor)
        _read_runner_supervisor_ready(
            control_read,
            launch_id=launch_id,
        )
        ready_received = True
        if writer_epoch_members is not None:
            # Preparation validates the exact prior family but performs no
            # durable writes. Retain its complete retry authority before the
            # first root write, closing every async-exception assignment gap
            # without relying on a calling-thread-only signal mask.
            writer_epoch_lease = prepare_writer_epoch_arm(
                writer_epoch_members,
                owner=capture_writer_epoch_owner(
                    "process_supervisor",
                    launch_id,
                ),
                boundary=WriterEpochBoundary(
                    "process_supervisor",
                    launch_id,
                    "process_supervisor",
                    settlement_owner_ids,
                    settlement_worker_ids,
                ),
            )
        # READY is consumed on the controller descriptor.  The coordinator
        # then owns a duplicate for both START/CANCEL sends and the sole
        # terminal receive; main-thread cancellation only publishes intent.
        # Closing the controller descriptor therefore cannot race a command
        # or terminal read in the retained transaction.
        settlement_control_read = os.dup(control_read)
        try:
            settlement_transaction = _RunnerSettlementTransaction(
                process,
                settlement_control_read,
                launch_id=launch_id,
                writer_epoch_lease=writer_epoch_lease,
                settlement_owner_ids=settlement_owner_ids,
                settlement_worker_ids=settlement_worker_ids,
                settlement_callback=settlement_callback,
            )
        except BaseException:
            os.close(settlement_control_read)
            raise
        # Start with the immutable prepared lease before the first durable arm
        # write or START, eliminating both terminal receive and binding gaps.
        published_transaction = settlement_transaction
        try:
            pending_claim_interrupt = coordinator.submit(
                published_transaction
            )
        except BaseException as submit_error:
            pending_abort_interrupt = coordinator.abort_unclaimed_and_wait()
            claimed = coordinator.claimed_transaction()
            if claimed is None:
                published_transaction.close_unclaimed()
                settlement_transaction = None
                if pending_abort_interrupt is not None:
                    submit_error.add_note(
                        "SIGINT was deferred while aborting an unclaimed "
                        "settlement transaction"
                    )
                raise
            settlement_transaction = claimed
            if pending_abort_interrupt is not None:
                pending_claim_interrupt = pending_abort_interrupt
            raise
        if pending_claim_interrupt is not None:
            # Cancellation observed before dispatch binds the prepared lease
            # to initial CANCEL. The target is never spawned; the coordinator
            # alone may finish that exact lease and quiesce it from terminal
            # evidence before the interrupt is replayed.
            settlement_transaction.request_cancel()
            raise pending_claim_interrupt
        if writer_epoch_lease is not None:
            try:
                retry_writer_epoch_arm(writer_epoch_lease)
            except WriterEpochArmError as arm_error:
                writer_epoch_lease = arm_error.lease
                interrupted_arm_lease = arm_error.lease
                raise
        settlement_transaction.request_start()
        outer_timeout = (
            timeout + settlement_grace + 0.1 if timeout is not None else None
        )
        bounded = _bounded_runner_communicate(
            process,
            command=command,
            config_json=config_json,
            timeout=outer_timeout,
        )
        if bounded is None:
            stdout, stderr = process.communicate(
                config_json,
                timeout=outer_timeout,
            )
        else:
            stdout, stderr, stdout_truncated, _stderr_truncated = bounded
    except subprocess.TimeoutExpired as exc:
        settlement_failure: BaseException | None = None
        stdout = _captured_bytes(exc.stdout)
        stderr = _captured_bytes(exc.stderr)
        while True:
            try:
                stdout, stderr = _cancel_runner_supervisor(
                    process,
                    control_read,
                    launch_id=launch_id,
                    ready_received=ready_received,
                    writer_epoch_lease=writer_epoch_lease,
                    settlement_owner_ids=settlement_owner_ids,
                    settlement_worker_ids=settlement_worker_ids,
                    settlement_callback=settlement_callback,
                    settlement_grace=settlement_grace,
                    settlement_transaction=settlement_transaction,
                    stdout=stdout,
                    stderr=stderr,
                )
                break
            except KeyboardInterrupt:
                if settlement_transaction is None:
                    raise
                continue
            except BaseException as settlement_exc:
                settlement_failure = settlement_exc
                break
        timeout_error = subprocess.TimeoutExpired(
            exc.cmd,
            exc.timeout,
            output=stdout,
            stderr=stderr,
        )
        if settlement_failure is not None:
            timeout_error.add_note(
                "runner cancellation could not finalize authenticated "
                f"settlement: {type(settlement_failure).__name__}: "
                f"{settlement_failure}"
            )
            raise timeout_error from settlement_failure
        raise timeout_error from exc
    except BaseException as primary_error:
        claimed_transaction = coordinator.claimed_transaction()
        if claimed_transaction is None:
            pending_abort_interrupt = coordinator.abort_unclaimed_and_wait()
            if settlement_transaction is not None:
                settlement_transaction.close_unclaimed()
            settlement_transaction = None
            # Preparation has performed no writes and no worker accepted its
            # epoch binding. Discard it; pre-START CANCEL must bind epoch=None.
            writer_epoch_lease = None
            interrupted_arm_lease = None
            if pending_abort_interrupt is not None:
                primary_error.add_note(
                    "SIGINT was deferred while aborting unclaimed runner "
                    "settlement"
                )
        elif claimed_transaction is not None:
            settlement_transaction = claimed_transaction
        if writer_epoch_lease is None and isinstance(
            primary_error, WriterEpochArmError
        ):
            writer_epoch_lease = primary_error.lease
        elif interrupted_arm_lease is not None:
            writer_epoch_lease = interrupted_arm_lease
        settlement_failure = None
        while True:
            try:
                _cancel_runner_supervisor(
                    process,
                    control_read,
                    launch_id=launch_id,
                    ready_received=ready_received,
                    writer_epoch_lease=writer_epoch_lease,
                    settlement_owner_ids=settlement_owner_ids,
                    settlement_worker_ids=settlement_worker_ids,
                    settlement_callback=settlement_callback,
                    settlement_grace=settlement_grace,
                    settlement_transaction=settlement_transaction,
                )
                break
            except KeyboardInterrupt:
                if settlement_transaction is None:
                    raise
                continue
            except BaseException as settlement_exc:
                settlement_failure = settlement_exc
                break
        if settlement_failure is not None:
            primary_error.add_note(
                "runner cancellation could not finalize authenticated "
                f"settlement: {type(settlement_failure).__name__}: "
                f"{settlement_failure}"
            )
            raise primary_error from settlement_failure
        raise
    finally:
        # A retained transaction owns the duplicate command/receive descriptor.
        # Its terminal proof (or failure) survives closing the controller's
        # READY-only descriptor after the last pre-transaction fallback path.
        if settlement_transaction is not None:
            try:
                os.close(control_read)
            except OSError:
                pass
        if process.returncode is None:
            try:
                process.wait(timeout=0)
            except (subprocess.TimeoutExpired, OSError):
                pass
    pending_join_interrupt: KeyboardInterrupt | None = None
    while True:
        try:
            if settlement_transaction is not None:
                (
                    returncode,
                    settlement_error,
                    _terminal_outcome,
                ) = settlement_transaction.join(replay_interrupt=False)
            else:
                (
                    returncode,
                    settlement_error,
                    _terminal_outcome,
                ) = _consume_runner_supervisor_settlement(
                    process,
                    control_read,
                    launch_id=launch_id,
                    writer_epoch_lease=writer_epoch_lease,
                    settlement_owner_ids=settlement_owner_ids,
                    settlement_worker_ids=settlement_worker_ids,
                    settlement_callback=settlement_callback,
                )
            break
        except KeyboardInterrupt as exc:
            if settlement_transaction is None:
                raise
            pending_join_interrupt = pending_join_interrupt or exc
    if pending_join_interrupt is not None:
        raise pending_join_interrupt
    if settlement_transaction is not None:
        deferred_interrupt = settlement_transaction.take_deferred_interrupt()
        if deferred_interrupt is not None:
            raise deferred_interrupt
    if returncode == -124:
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout,
            stderr=stderr,
        )
    if settlement_error == "background_descendants":
        stderr = _append_runner_diagnostic(
            stderr,
            b"\n[runner settlement error: background descendants outlived "
            b"normal exit; all descendants were terminated]",
            capture_limit=_RUNNER_STDERR_CAPTURE_BYTES,
        )
        if returncode == 0:
            returncode = 2
    if stdout_truncated:
        protocol_error = (
            b"\n[runner protocol error: stdout exceeded the bounded capture limit]"
        )
        stderr = _append_runner_diagnostic(
            stderr,
            protocol_error,
            capture_limit=_RUNNER_STDERR_CAPTURE_BYTES,
        )
        if returncode == 0:
            returncode = 2
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout,
        stderr,
    )


def run_supervised_process(
    args,
    *,
    input=None,
    cwd=None,
    env=None,
    capture_output=False,
    timeout=None,
    check=False,
    text=False,
    encoding=None,
    errors=None,
    settlement_callback: Callable[[str], None] | None = None,
    writer_epoch_members: tuple[WriterEpochStorage, ...] | None = None,
):
    """A narrow ``subprocess.run`` adapter with positive descendant proof.

    Assay smoke/bench primitives accept this workflow-owned injection.  They
    always request captured output and never use a shell, so keep the surface
    intentionally smaller than the standard-library function.
    """

    if not capture_output:
        raise ValueError("supervised process execution requires captured output")
    if not isinstance(args, (list, tuple)) or not args or not all(
        isinstance(item, str) and item for item in args
    ):
        raise ValueError("supervised process argv must be non-empty strings")
    effective_encoding = "utf-8" if encoding is None else encoding
    effective_errors = "strict" if errors is None else errors
    if input is None:
        input_bytes = b""
    elif isinstance(input, bytes):
        input_bytes = input
    elif isinstance(input, str):
        input_bytes = input.encode(effective_encoding, effective_errors)
    else:
        raise TypeError("supervised process input must be bytes, str, or None")
    completed = _run_runner_subprocess(
        list(args),
        config_json=input_bytes,
        cwd=str(
            Path(os.getcwd() if cwd is None else cwd).resolve(strict=True)
        ),
        timeout=timeout,
        env=dict(os.environ if env is None else env),
        settlement_callback=settlement_callback,
        writer_epoch_members=writer_epoch_members,
    )
    rendered = subprocess.CompletedProcess(
        args,
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )
    if text or encoding is not None or errors is not None:
        rendered = subprocess.CompletedProcess(
            args,
            completed.returncode,
            completed.stdout.decode(effective_encoding, effective_errors),
            completed.stderr.decode(effective_encoding, effective_errors),
        )
    if check and rendered.returncode != 0:
        raise subprocess.CalledProcessError(
            rendered.returncode,
            args,
            output=rendered.stdout,
            stderr=rendered.stderr,
        )
    return rendered


def _run_runner(
    role: str,
    runner: str,
    payload: dict,
    *,
    runner_mode: str | None = None,
    cwd: str,
    timeout: float | None,
    log_dir: Path | None,
    writer_epoch_members: tuple[WriterEpochStorage, ...] | None = None,
) -> RunnerResult:
    _validate_runner_path(runner)
    # Inject the role into parameters so language runners can switch
    # commands ("build" → cargo build, "verify" → cargo test, etc.).
    # Without this, the polyglot rust runner can't distinguish a build
    # invocation from a verify invocation and unconditionally runs
    # `cargo test --workspace` in both phases — defeating
    # `verify_mode: build_only` slices and forcing every tier through
    # the full workspace test suite even when the slice declared no
    # behavioural changes.
    # Forward the phase timeout to the runner's inner subprocess as well as
    # the outer ``python -m <runner>`` process. Both boundaries honor the
    # workflow's explicit build or verification cap.
    inner_params = {**(payload.get("parameters") or {}), "role": role}
    if runner_mode is not None:
        inner_params["runner_mode"] = runner_mode
    if timeout is not None:
        inner_params["runner_timeout_s"] = timeout
    payload = {**payload, "parameters": inner_params}
    config_json = json.dumps(payload).encode("utf-8")
    log_path: Path | None = None
    latest_log_path: Path | None = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Every invocation gets immutable evidence. The stable role.log path
        # remains an atomic latest-view for operator tooling.
        log_path = log_dir / (
            f"{role}_{time.time_ns()}_{secrets.token_hex(4)}.log"
        )
        latest_log_path = log_dir / f"{role}.log"

    def _persist_log(payload: bytes) -> None:
        if log_path is None or latest_log_path is None:
            return
        write_atomic_bytes(log_path, payload)
        write_atomic_bytes(latest_log_path, payload)

    start = time.monotonic()
    settlement_digests: list[str] = []

    def _process_settlement_digest() -> str:
        if len(settlement_digests) != 1:
            raise _RunnerSettlementError(
                "launched runner did not publish exactly one process "
                "settlement digest"
            )
        return settlement_digests[0]

    try:
        # Bundled runners start isolated through the trusted bootstrap. Any
        # other runner starts as ``python -P -m <runner>``: the launch does
        # not add the working directory (the gated, agent-writable workspace)
        # to ``sys.path``, so the runner must be importable from the
        # environment. An absolute import root supplied there still applies.
        runner_command = (
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _TRUSTED_RUNNER_BOOTSTRAP,
                str(_TRUSTED_SOURCE_ROOT),
                runner,
            ]
            if runner.startswith("crucible.")
            else [sys.executable, "-P", "-m", runner]
        )
        runner_env = os.environ.copy()
        if not runner.startswith("crucible."):
            # ``-P`` does not cover ``PYTHONPATH``: Python resolves an empty
            # or relative entry there against the working directory and
            # places it ahead of the standard library. Only the absolute
            # import roots reach the runner and the processes it starts.
            import_roots = [
                entry
                for entry in runner_env.pop("PYTHONPATH", "").split(os.pathsep)
                if os.path.isabs(entry)
            ]
            if import_roots:
                runner_env["PYTHONPATH"] = os.pathsep.join(import_roots)
        runner_env.update({
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": "/dev/null",
        })
        completed = _run_runner_subprocess(
            runner_command,
            config_json=config_json,
            cwd=cwd,
            timeout=timeout,
            env=runner_env,
            settlement_callback=settlement_digests.append,
            writer_epoch_members=writer_epoch_members,
        )
    except subprocess.TimeoutExpired as exc:
        # Treat a timeout as a hard failure with a synthetic returncode.
        # The workflow sees ``runner_missing`` semantics if the runner
        # never started; we want to distinguish "ran and timed out."
        stderr = (exc.stderr or b"") + f"\n[timeout after {timeout}s]".encode()
        _persist_log(
            f"# role\n{role}\n# runner\n{runner}\n# cwd\n{cwd}\n"
            f"# returncode\nTIMEOUT\n# stderr\n".encode()
            + stderr
        )
        return RunnerResult(
            role=role,
            runner=runner,
            returncode=124,  # sh convention for timeout
            values={},
            raw_output=exc.stdout or b"",
            stderr=stderr,
            duration_s=time.monotonic() - start,
            process_settlement_digest=_process_settlement_digest(),
        )

    duration = time.monotonic() - start
    process_settlement_digest = _process_settlement_digest()
    # Parse the values map from stdout — but a non-zero exit doesn't
    # require parseable stdout (build failures often print to stderr
    # only). The workflow decides whether to act on missing values.
    values: dict[str, float] = {}
    reported_mode: str | None = None
    stdout_protocol_parsed = False
    payload_out: object = None
    if completed.stdout:
        try:
            payload_out = json.loads(completed.stdout.decode("utf-8"))
            stdout_protocol_parsed = True
            raw_values = payload_out.get("values") if isinstance(payload_out, dict) else None
            if isinstance(raw_values, dict):
                values = {str(k): float(v) for k, v in raw_values.items()
                          if isinstance(v, (int, float))}
            if isinstance(payload_out, dict) and isinstance(
                payload_out.get("runner_mode"), str
            ):
                reported_mode = payload_out["runner_mode"]
        except (RecursionError, json.JSONDecodeError, ValueError, TypeError):
            # Strategy 2: last-line JSON object (matches assay.bench).
            try:
                last_line = completed.stdout.decode("utf-8").strip().splitlines()[-1]
                payload_out = json.loads(last_line)
                stdout_protocol_parsed = True
                raw_values = payload_out.get("values") if isinstance(payload_out, dict) else None
                if isinstance(raw_values, dict):
                    values = {str(k): float(v) for k, v in raw_values.items()
                              if isinstance(v, (int, float))}
                if isinstance(payload_out, dict) and isinstance(
                    payload_out.get("runner_mode"), str
                ):
                    reported_mode = payload_out["runner_mode"]
            except Exception:  # noqa: BLE001
                logger.debug("%s runner stdout was not JSON-parseable", role)

    returncode = completed.returncode
    stderr = completed.stderr
    if returncode == 0 and completed.stdout and not stdout_protocol_parsed:
        returncode = 2
        stderr = _append_runner_diagnostic(
            stderr,
            b"\n[runner protocol error: successful runner stdout was not "
            b"valid bounded UTF-8 JSON]",
            capture_limit=_RUNNER_STDERR_CAPTURE_BYTES,
        )
    if (
        returncode == 0
        and runner_mode == "bench"
        and reported_mode != "bench"
    ):
        returncode = 2
        stderr += (
            b"\n[runner protocol error: bench mode was requested but the "
            b"adapter did not acknowledge runner_mode='bench']"
        )

    _persist_log(
        f"# role\n{role}\n# runner\n{runner}\n# cwd\n{cwd}\n"
        f"# runner_mode\n{runner_mode or ''}\n# returncode\n{returncode}\n"
        f"# duration_s\n{duration:.3f}\n# stdout\n".encode()
        + completed.stdout
        + (b"" if completed.stdout.endswith(b"\n") else b"\n")
        + b"# stderr\n"
        + stderr
    )

    return RunnerResult(
        role=role,
        runner=runner,
        returncode=returncode,
        values=values,
        raw_output=completed.stdout,
        stderr=stderr,
        duration_s=duration,
        process_settlement_digest=process_settlement_digest,
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def _resolve_languages(
    spec, slice_directive=None,
) -> tuple[str, ...]:
    """Resolve which languages to feed the polyglot runner.

    Per-slice override beats Spec-wide default:

    1. ``slice_directive.blast_radius.languages`` (per-slice) — when
       present and non-empty, wins. Lets a Spec carry mixed languages
       across slices.
    2. ``LanguagesConstraint`` on ``spec.constraints`` — Spec-wide
       default registered by ``crucible.constraints``.
    3. Empty tuple — caller's runner ignores the field (the project's
       runner may not need a language list).

    Returns the ordered tuple; the polyglot runner dispatches in this
    order and short-circuits on first non-zero exit.
    """
    if slice_directive is not None:
        br = getattr(slice_directive, "blast_radius", None) or {}
        slice_langs = br.get("languages") if isinstance(br, dict) else None
        if isinstance(slice_langs, (list, tuple)) and slice_langs:
            return tuple(str(s) for s in slice_langs)
    for c in (spec.constraints or ()):
        if getattr(c, "kind", None) == "languages":
            langs = getattr(c, "languages", None)
            if isinstance(langs, (list, tuple)) and langs:
                return tuple(str(s) for s in langs)
            break
    return ()


def _mutable_parameter_copy(value: object) -> object:
    """Copy frozen Spec parameter containers into ordinary JSON containers."""

    if isinstance(value, Mapping):
        return {
            str(key): _mutable_parameter_copy(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_mutable_parameter_copy(item) for item in value]
    return value


def _merge_parameter_maps(
    base: dict,
    overlay: dict,
    *,
    path: str = "runner_parameters",
) -> dict:
    """Add runtime metadata without overriding canonical Spec selection."""

    merged = dict(base)
    for key, value in overlay.items():
        prior = merged.get(key)
        if isinstance(prior, dict) and isinstance(value, dict):
            merged[key] = _merge_parameter_maps(
                prior,
                value,
                path=f"{path}.{key}",
            )
        elif key in merged:
            copied_value = _mutable_parameter_copy(value)
            if prior != copied_value:
                raise ValueError(
                    "runtime runner metadata may not override canonical Spec "
                    f"configuration at {path}.{key}"
                )
        else:
            merged[key] = _mutable_parameter_copy(value)
    return merged


def runner_parameters(
    spec,
    extra_parameters: dict | None = None,
    *,
    languages: tuple[str, ...] = (),
    slice_id: str | None = None,
) -> dict:
    """Resolve the generic, Spec-bound runner parameter envelope.

    Static project configuration lives in
    ``project_context.runner_parameters`` and therefore participates in the
    Spec content hash. ``extra_parameters`` is reserved for trusted workflow
    runtime metadata (for example a dataset container id or breach probe).
    The orchestration layer deliberately does not interpret language-specific
    keys; the selected runner owns that schema. Runtime data may add fields or
    repeat an identical value, but cannot replace a canonical Spec value.
    """

    configured = getattr(spec.project_context, "runner_parameters", {}) or {}
    validate_static(configured)
    copied = _mutable_parameter_copy(configured)
    if not isinstance(copied, dict):
        raise TypeError("project_context.runner_parameters must be a mapping")
    per_slice = copied.get("per_slice")
    tier_dag = getattr(spec, "tier_dag", None)
    if isinstance(per_slice, dict) and tier_dag is not None:
        declared_slice_ids = {
            slice_.id
            for tier in tier_dag.tiers
            for slice_ in tier.slices
        }
        unknown_slice_ids = sorted(set(per_slice) - declared_slice_ids)
        if unknown_slice_ids:
            raise ValueError(
                "runner parameter 'per_slice' contains unknown canonical "
                f"slice ids: {unknown_slice_ids!r}"
            )
    copied = for_slice(copied, slice_id)
    runtime = dict(extra_parameters or {})
    if not isinstance(extra_parameters, _TrustedRunnerOverrides):
        forbidden = sorted(set(runtime) - _RUNTIME_METADATA_KEYS)
        if forbidden:
            raise ValueError(
                "extra_parameters accepts runtime metadata namespaces only; "
                "static runner selection belongs in the canonical Spec: "
                f"forbidden keys={forbidden!r}"
            )
    merged = _merge_parameter_maps(copied, runtime)
    requested_languages = merged.get("languages")
    if languages:
        normalized = list(languages)
        if requested_languages is not None and requested_languages != normalized:
            raise ValueError(
                "runner parameter 'languages' is workflow-owned and must "
                f"match the resolved language constraint: {normalized!r}"
            )
        merged["languages"] = normalized
    elif requested_languages is not None:
        raise ValueError(
            "runner parameter 'languages' is workflow-owned; declare a "
            "LanguagesConstraint or slice blast_radius.languages"
        )
    return merged


def _inject_languages(
    spec,
    extra_parameters: dict | None,
    languages: tuple[str, ...],
    *,
    slice_id: str | None,
) -> dict:
    """Compatibility wrapper around the generic parameter resolver."""

    return runner_parameters(
        spec,
        extra_parameters,
        languages=languages,
        slice_id=slice_id,
    )


def build(
    spec,
    *,
    slice_id: str | None = None,
    cwd: str,
    log_dir: Path | None = None,
    timeout: float | None = None,
    extra_parameters: dict | None = None,
    slice_directive=None,
    writer_epoch_members: tuple[WriterEpochStorage, ...] | None = None,
) -> RunnerResult | None:
    """Run the project's build runner. Returns None if no build runner
    is configured (``project_context.runners`` lacks a ``build`` key)."""
    runner = (spec.project_context.runners or {}).get("build")
    if not runner:
        return None
    languages = _resolve_languages(spec, slice_directive)
    payload = {
        "bench_id": f"build_{slice_id or 'all'}",
        "parameters": _inject_languages(
            spec,
            extra_parameters,
            languages,
            slice_id=slice_id,
        ),
        "env": {},
    }
    return _run_runner(
        "build", runner, payload,
        cwd=cwd, timeout=timeout, log_dir=log_dir,
        writer_epoch_members=writer_epoch_members,
    )


def _requested_verify_mode(slice_directive) -> str:
    """Resolve the workflow-owned adapter mode for a verify invocation."""

    mode = (
        getattr(slice_directive, "verify_mode", None)
        if slice_directive is not None else None
    ) or "tests"
    if mode not in {"skip", "build_only", "tests", "bench"}:
        raise ValueError(f"unsupported verify_mode: {mode!r}")
    return mode


def _missing_bench_runner() -> RunnerResult:
    """Represent a requested benchmark with no adapter as a hard failure."""

    message = (
        b"bench verification was requested, but project_context.runners "
        b"declares neither a verify nor test runner"
    )
    return RunnerResult(
        role="verify",
        runner="<unconfigured>",
        returncode=2,
        values={"returncode": 2.0},
        raw_output=message,
        stderr=b"",
        duration_s=0.0,
    )


def verify(
    spec,
    *,
    slice_id: str | None = None,
    cwd: str,
    log_dir: Path | None = None,
    timeout: float | None = None,
    extra_parameters: dict | None = None,
    slice_directive=None,
    writer_epoch_members: tuple[WriterEpochStorage, ...] | None = None,
) -> RunnerResult | None:
    """Run the project's verify (or ``test``) runner. ``verify`` wins.

    ``tests`` and ``bench`` are passed as the generic workflow-owned
    ``runner_mode``. ``skip`` / ``build_only`` return None. A missing runner
    also returns None for ordinary tests,
    but produces a failing synthetic result for ``bench`` because silently
    omitting an explicitly requested benchmark would be a false success.
    """
    mode = _requested_verify_mode(slice_directive)
    if mode in ("skip", "build_only"):
        return None
    runners = spec.project_context.runners or {}
    runner = runners.get("verify") or runners.get("test")
    if not runner:
        return _missing_bench_runner() if mode == "bench" else None
    languages = _resolve_languages(spec, slice_directive)
    payload = {
        "bench_id": f"verify_{slice_id or 'all'}",
        "parameters": _inject_languages(
            spec,
            extra_parameters,
            languages,
            slice_id=slice_id,
        ),
        "env": {},
    }
    return _run_runner(
        "verify", runner, payload,
        runner_mode=mode,
        cwd=cwd, timeout=timeout, log_dir=log_dir,
        writer_epoch_members=writer_epoch_members,
    )


def dev_verify(
    spec,
    *,
    slice_id: str | None = None,
    cwd: str,
    log_dir: Path | None = None,
    timeout: float | None = None,
    extra_parameters: dict | None = None,
    slice_directive=None,
    verification_directives: tuple[object, ...] = (),
    accepted_verification_reference: dict[str, object] | None = None,
    writer_epoch_members: tuple[WriterEpochStorage, ...] | None = None,
) -> RunnerResult | None:
    """Run the project's ``dev_verify`` runner — the project-native test
    gate that runs INSIDE the per-task authoring container (DeepSWE) as
    opposed to ``verify``, which runs the dataset's hidden-test gate.

    Returns None if no ``dev_verify`` runner is declared. Unlike
    ``verify``, ``dev_verify`` does NOT honor ``verify_mode`` skip /
    build_only: a per-round breach gate is not a slice, so the slice's
    own verify_mode (a planner directive about the slice's behavioural
    surface) doesn't suppress it. The whole point of the per-round
    gate is to always run the Defender's adversarial tests + the
    project's own tests; a slice directive can't opt that out.
    """
    runner = (spec.project_context.runners or {}).get("dev_verify")
    if not runner:
        return None
    checkpoint_capable = supports_declared_verification_checkpoints(spec)
    languages = _resolve_languages(spec, slice_directive)
    parameters = _inject_languages(
        spec, extra_parameters, languages, slice_id=slice_id,
    )
    forbidden_controls = sorted(
        key
        for key in (
            DECLARED_VERIFICATION_SURFACE_KEY,
            ACCEPTED_VERIFICATION_REFERENCE_KEY,
        )
        if key in parameters
    )
    if forbidden_controls:
        raise ValueError(
            "declared verification controls are workflow-owned and must be "
            "derived from accepted slice directives: "
            f"{forbidden_controls!r}"
        )
    if (
        not checkpoint_capable
        and (verification_directives or accepted_verification_reference is not None)
    ):
        raise ValueError(
            "dev_verify runner does not have the trusted declared-verification "
            "checkpoint capability"
        )
    directives = (
        (
            verification_directives
            if verification_directives
            else ((slice_directive,) if slice_directive is not None else ())
        )
        if checkpoint_capable
        else ()
    )
    if directives:
        parameters[DECLARED_VERIFICATION_SURFACE_KEY] = (
            surface_from_directives(directives)
        )
    if accepted_verification_reference is not None:
        if not directives:
            raise ValueError(
                "accepted verification reference requires a declared surface"
            )
        parsed_reference = parse_accepted_verification_reference(
            accepted_verification_reference
        )
        cumulative_surface = surface_from_directives(directives)
        cumulative_slices = cumulative_surface["slices"]
        predecessor_slices = parsed_reference.predecessor_surface["slices"]
        if cumulative_slices[:len(predecessor_slices)] != predecessor_slices:
            raise ValueError(
                "accepted verification predecessor surface is not contained "
                "in the cumulative gate surface"
            )
        parameters[ACCEPTED_VERIFICATION_REFERENCE_KEY] = (
            accepted_verification_reference_payload(
                predecessor_surface=parsed_reference.predecessor_surface,
                language=parsed_reference.language,
                parser_kind=parsed_reference.parser_kind,
                outcomes=(
                    (item.path, item.command_sha256, item.outcome)
                    for item in parsed_reference.outcomes
                ),
            )
        )
    payload = {
        "bench_id": f"dev_verify_{slice_id or 'all'}",
        "parameters": parameters,
        "env": {},
    }
    return _run_runner(
        "dev_verify", runner, payload,
        cwd=cwd, timeout=timeout, log_dir=log_dir,
        writer_epoch_members=writer_epoch_members,
    )


@dataclass(frozen=True)
class BuildVerifyOutcome:
    classification: str
    build: RunnerResult | None
    verify: RunnerResult | None
    # The project-native (``dev_verify``) runner result, populated only
    # when ``build_and_verify`` is called with ``gate="dev_verify"``
    # (the per-round breach gate). None for the default ``verify`` gate.
    dev_verify: RunnerResult | None = None


_DEV_VERIFY_UNAVAILABLE_TRUE_FLAGS = frozenset({
    "dev_verify_baseline_unavailable",
    "dev_verify_infrastructure_unavailable",
    "dev_verify_source_unusable",
})


def _dev_verify_evidence_unavailable(result: RunnerResult) -> bool:
    """Return whether a development gate lacks actionable model evidence.

    DeepSWE's verifier deliberately uses return code 2 when it cannot make a
    structurally valid comparison. The more specific, runner-owned KPI flags
    are authoritative here: a generic third-party runner may use exit 2 for a
    normal compiler/test failure and receives the
    ``dev_verify_failed`` classification.
    """

    values = result.values
    if any(values.get(key) == 1.0 for key in _DEV_VERIFY_UNAVAILABLE_TRUE_FLAGS):
        return True
    return False


# Per-slice build and verification gates are bounded to one hour by default.
# Callers can override these limits through explicit workflow arguments.
DEFAULT_BUILD_TIMEOUT_S: float = 3600.0
DEFAULT_VERIFY_TIMEOUT_S: float = 3600.0


def _settle_no_process_writer_epoch(
    writer_epoch_members: tuple[WriterEpochStorage, ...] | None,
    *,
    phase: str,
    succeeded: bool,
) -> None:
    """Advance an exact family when a configured gate launches no runner.

    A long-lived workflow root can already carry a quiescent generation while
    its newly-created gate checkpoint is still generation zero. Returning a
    missing/optional-runner classification without a boundary would leave that
    family partial and block the caller's source capture. This synchronous
    no-process receipt proves only that runner dispatch was skipped; it never
    claims that an external process settled.
    """

    if writer_epoch_members is None:
        return
    boundary_id = f"build-verify-no-process-{secrets.token_hex(16)}"
    worker_ids = (f"{boundary_id}:{phase}",)
    lease = arm_writer_epoch_family(
        writer_epoch_members,
        owner=capture_writer_epoch_owner(
            "build_verify_no_process",
            boundary_id,
        ),
        boundary=WriterEpochBoundary(
            "build_verify_no_process",
            boundary_id,
            "synchronous_callback",
            (),
            worker_ids,
        ),
    )
    evidence = json.dumps(
        {
            "epoch_id": lease.epoch_id,
            "phase": phase,
            "runner_dispatched": False,
            "schema_version": "crucible.build-verify-no-process.v1",
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    receipt = synchronous_callback_settlement_receipt(
        lease,
        settled_owner_ids=(),
        joined_worker_ids=worker_ids,
        evidence_digest="sha256:" + hashlib.sha256(evidence).hexdigest(),
        outcome="succeeded" if succeeded else "failed",
    )
    quiesce_writer_epoch_family(lease, receipt)
    require_writer_epoch_quiescence(
        lease.members,
        allow_never_armed=False,
    )


def build_and_verify(
    spec,
    *,
    slice_id: str | None = None,
    cwd: str,
    log_dir: Path | None = None,
    build_timeout: float | None = DEFAULT_BUILD_TIMEOUT_S,
    verify_timeout: float | None = DEFAULT_VERIFY_TIMEOUT_S,
    slice_directive=None,
    verification_directives: tuple[object, ...] = (),
    accepted_verification_reference: dict[str, object] | None = None,
    extra_parameters: dict | None = None,
    gate: str = "verify",
    writer_epoch_members: tuple[WriterEpochStorage, ...] | None = None,
) -> BuildVerifyOutcome:
    """Run build then (if build passed) the test gate; return a classification.

    ``gate`` selects which test runner closes the loop:

      - ``"verify"`` (default) — the canonical gate. Runs the spec's
        ``verify`` (or ``test``) runner. For DeepSWE this is the
        dataset's hidden-test verdict; for everything else it's the
        project's configured test runner. Used at the END of the
        cascade (final verify) and by every non-DeepSWE workflow.
      - ``"dev_verify"`` — the per-round breach gate. Runs the spec's
        ``dev_verify`` runner (DeepSWE's project-native, in-container
        test runner). Used by the breach loop so each round gates on
        the Defender's adversarial tests + the project's own tests,
        WITHOUT touching the dataset's hidden suite. A spec with no
        ``dev_verify`` runner declared collapses to ``success`` when
        build passes (same contract as a missing ``verify`` runner).

    Classification strings:
      - ``success``           — build 0; gate runner 0 (or absent for an
                                ordinary test gate). A requested benchmark
                                requires a configured, acknowledging adapter.
      - ``build_failed``      — build returned non-zero.
      - ``verify_failed``     — build 0; the ``verify`` gate returned
                                non-zero.
      - ``dev_verify_failed`` — build 0; the ``dev_verify`` gate emitted
                                actionable test/build failure evidence.
                                Distinct from ``verify_failed``: the round's
                                project-native / adversarial tests caught the
                                impl; the dataset gate did not run.
      - ``dev_verify_unavailable`` — the ``dev_verify`` gate could not
                                produce comparable executable evidence (for
                                example its clean baseline/dependency was
                                unavailable or containment was unusable).
                                This is not model-retriable.
      - ``partial``           — gate returned 0 but values map shows
                                ``tests_failed > 0``.
      - ``runner_missing``    — spec did not declare a build runner.

    ``build_timeout`` / ``verify_timeout`` are passed through to the
    subprocess runner; ``None`` disables the per-call timeout (NOT
    recommended — see ``DEFAULT_BUILD_TIMEOUT_S`` for context).
    Defaults to 60 min each, which fits swebench / typical Python
    test runs. Large builds and comprehensive test suites may need
    higher caps (~2h build, ~4h verify); set via the calling
    workflow's ``--build-timeout-s`` / ``--verify-timeout-s`` CLI
    flags.

    ``slice_directive`` (optional ``ImplementationDirective``) lets a
    workflow pass per-slice ``blast_radius.languages`` to override the
    Spec-wide ``LanguagesConstraint`` default when feeding the
    polyglot runner. For ``dev_verify``, it also supplies the generic declared
    verification surface. ``verification_directives`` is the explicit
    multi-slice form used by final/post-breach gates; callers cannot inject a
    pre-rendered surface through static or runtime runner parameters.

    The workflow is expected to overwrite the
    ``Implementation.classification`` field with this value before
    appending the Implementation to the Spec.
    """
    if gate not in ("verify", "dev_verify"):
        raise ValueError(
            f"build_and_verify gate must be 'verify' or 'dev_verify', "
            f"got {gate!r}"
        )
    # Concurrent slice fan-out: build+verify must serialize against the
    # shared checkout, even though implementer phases run in parallel.
    # The fcntl gate is scoped to ``cwd`` so independent repos still
    # build in parallel.
    with build_lock(Path(cwd)):
        if gate == "dev_verify":
            # The per-round breach gate skips the host build phase
            # entirely: ``dev_verify`` runs the project's tests INSIDE
            # the dep-baked container, where there is no host build to
            # run, and DeepSWE's go/ts/js languages declare no build
            # runner at all (requiring one would wrongly return
            # ``runner_missing``). The in-container test run is the
            # whole gate.
            b = None
        else:
            b = build(
                spec, slice_id=slice_id, cwd=cwd, log_dir=log_dir,
                timeout=build_timeout, slice_directive=slice_directive,
                extra_parameters=extra_parameters,
                writer_epoch_members=writer_epoch_members,
            )
            if b is None:
                _settle_no_process_writer_epoch(
                    writer_epoch_members,
                    phase="build_runner_missing",
                    succeeded=False,
                )
                return BuildVerifyOutcome(
                    classification="runner_missing",
                    build=None,
                    verify=None,
                )
            if not b.ok:
                return BuildVerifyOutcome(
                    classification="build_failed",
                    build=b,
                    verify=None,
                )

        if gate == "dev_verify":
            v = dev_verify(
                spec, slice_id=slice_id, cwd=cwd, log_dir=log_dir,
                timeout=verify_timeout, slice_directive=slice_directive,
                verification_directives=verification_directives,
                accepted_verification_reference=(
                    accepted_verification_reference
                ),
                extra_parameters=extra_parameters,
                writer_epoch_members=writer_epoch_members,
            )
        else:
            v = verify(
                spec, slice_id=slice_id, cwd=cwd, log_dir=log_dir,
                timeout=verify_timeout, slice_directive=slice_directive,
                extra_parameters=extra_parameters,
                writer_epoch_members=writer_epoch_members,
            )
    # Keep development and ordinary verification results in separate fields
    # so the breach loop can distinguish their prior-feedback artifacts.
    dev_v = v if gate == "dev_verify" else None
    legacy_v = v if gate == "verify" else None
    failed_classification = (
        "dev_verify_failed" if gate == "dev_verify" else "verify_failed"
    )

    if v is None:
        # No gate runner declared: build alone is the contract.
        _settle_no_process_writer_epoch(
            writer_epoch_members,
            phase=f"{gate}_runner_absent",
            succeeded=True,
        )
        return BuildVerifyOutcome(
            classification="success",
            build=b,
            verify=legacy_v,
            dev_verify=dev_v,
        )
    if not v.ok:
        if gate == "dev_verify" and _dev_verify_evidence_unavailable(v):
            failed_classification = "dev_verify_unavailable"
        return BuildVerifyOutcome(
            classification=failed_classification,
            build=b,
            verify=legacy_v,
            dev_verify=dev_v,
        )
    # Partial: pytest-style runner emits ``tests_failed > 0`` even when
    # the runner-level returncode is 0 (e.g. when configured with
    # ``--maxfail=N`` and N wasn't reached).
    if v.values.get("tests_failed", 0.0) > 0.0:
        return BuildVerifyOutcome(
            classification="partial",
            build=b,
            verify=legacy_v,
            dev_verify=dev_v,
        )
    return BuildVerifyOutcome(
        classification="success",
        build=b,
        verify=legacy_v,
        dev_verify=dev_v,
    )


__all__ = [
    "RunnerResult",
    "BuildVerifyOutcome",
    "build",
    "verify",
    "dev_verify",
    "build_and_verify",
    "run_supervised_process",
]
