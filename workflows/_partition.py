# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Resumability partition for ``workflows.tier``.

Walks the latest Spec's attached ``Implementation`` records, identifies
which slices in a target tier have a plan-bound skip-worthy prior outcome, and
returns ``(landed_impls, pending_slices)`` so the tier loop can skip
already-completed work on a resume.

Skip-worthy set is intentionally narrow — mirrors
``workflows.swebench._is_skip_worthy_prior``:
- ``success``                  — clean implementer + cascade run
- ``soft_fail_no_changes``     — only for an intentionally no-write directive

Everything else (``hard_fail_*``, transport failures, build_failed,
verify_failed) re-runs by default. Those represent operator-paid
runs that never produced a terminal answer; a resume should attempt
them again rather than treat them as landed.

Operators bypass the partition entirely via the ``--restart`` opt-out
on the tier workflow (skill-internal, forwarded by ``/crucible-build``
when the user explicitly wants to redo a completed tier).

**Preconditions and invariants:**

- ``slice_id`` is globally unique within a Spec (assay's schema invariant),
  and each accepted record carries a resume identity binding it to the tier,
  directive, complete TierDAG, project execution authority (including runner
  modules and canonical runner parameters), project base, and implementation
  base.
- ``partition_for_resume`` decides lineage/plan authority; it does not inspect
  a checkout. Before implementing pending work, ``workflows.tier`` composes
  the selected current-tier and dependency diffs into ordinary workspaces,
  restoring accepted bytes after a reset and failing closed when a stored
  patch cannot apply. DeepSWE's explicitly persistent bind-mounted workspace
  is the exception: its later breach state can supersede an intermediate
  slice diff, so that mode retains the live workspace as its state authority
  instead of replaying stale intermediate patches.
- Crashed-rollback case: if the implementer process is killed
  *between* ``corpus.put_record(impl)`` and ``rollback_slice(impl)``,
  the Corpus has a hard_fail record but the working tree retains
  the dirt. On resume, this module correctly marks the slice as
  pending (hard_fail isn't skip-worthy), but ``assay.implement``
  runs against the dirty tree and the captured diff_ref ends up a
  union of prior-crash dirt + new attempt. A pre-implement reaper
  (REC-8 in the parity assessment) is the right place to detect
  + clean this; the partition alone does not address it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from assay.canonical import hash_record
from assay.types.artifacts import Implementation
from crucible.backends.cost_bounds import DIAGNOSTIC_COST_POLICY

if TYPE_CHECKING:
    from assay.corpus.store import Corpus
    from assay.types.dag import Slice, Tier
    from assay.types.spec import Spec

logger = logging.getLogger(__name__)


_RESUME_IDENTITY_KIND = "tier_resume_identity.v3"
_RESUME_IDENTITY_MAX_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UNSPECIFIED = object()


class _BoundedArtifactReader(Protocol):
    def __call__(
        self,
        ref: object,
        *,
        max_bytes: int,
        label: str,
    ) -> bytes: ...


@dataclasses.dataclass(frozen=True, slots=True)
class VerificationResumePartition:
    """Three-way resume authority for a required deterministic gate.

    ``verified_retained`` and ``verification_pending`` both contain exact,
    plan-bound candidates whose implementation work is complete.  The latter
    must be replayed through the deterministic gate before it may authorize a
    required-pass resume or landing.  ``implementation_pending`` is the
    declaration-ordered set that still needs an Implementer attempt.

    Handoff emits candidate-bound per-slice evidence. Tier publishes one
    shared manifest that atomically binds every current slice, its dependency
    closure, and all no-surface or Handoff-replay obligations.
    """

    verified_retained: dict[str, Implementation]
    verification_pending: dict[str, Implementation]
    implementation_pending: tuple["Slice", ...]
    tier_manifest_obligations: tuple[str, ...] = ()
    tier_manifest_ref: str | None = None
    tier_manifest_verified: bool = False

    @property
    def retained(self) -> dict[str, Implementation]:
        """Return all exact candidates without conflating their gate state."""

        return {
            **self.verified_retained,
            **self.verification_pending,
        }


def _validated_tool_authority_digest(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            "tool_authority_digest must be 'sha256:' followed by 64 "
            "lowercase hexadecimal digits or None"
        )
    return value


def _project_execution_authority(project_context) -> dict:
    return {
        "project": getattr(project_context, "project", None),
        "repo_url": getattr(project_context, "repo_url", None),
        "base_commit": project_context.base_commit,
        "branch": getattr(project_context, "branch", None),
        "submodule_path": getattr(project_context, "submodule_path", None),
        "runners": dict(getattr(project_context, "runners", {}) or {}),
        "runner_parameters": dict(
            getattr(project_context, "runner_parameters", {}) or {}
        ),
    }


def spec_execution_structure_hash(spec: "Spec") -> str:
    """Hash authored execution authority while excluding evidence lineage."""

    return hash_record({
        "schema_version": getattr(spec, "schema_version", None),
        "project_context": _project_execution_authority(spec.project_context),
        "intent": getattr(spec, "intent", None),
        "constraints": getattr(spec, "constraints", ()),
        "done_criteria": getattr(spec, "done_criteria", ()),
        "depends_on": getattr(spec, "depends_on", ()),
        "plan_policy": getattr(spec, "plan_policy", None),
        "tier_dag": spec.tier_dag,
    })


def _is_skip_worthy(
    classification: str,
    slice_: "Slice",
    implementation: Implementation | None = None,
) -> bool:
    """Return whether an accepted attempt may satisfy this exact slice.

    ``soft_fail_no_changes`` is terminal for a directive with no write surface,
    or when Handoff recorded the no-change result under explicit operator
    authority (``--allow-empty``). Autonomous scoped slices that produce no
    changes remain incomplete and must be retried; treating those as landed
    was the source of partial-tier promotion on resume.
    """

    bounds = getattr(implementation, "cost_bounds", None)
    if bounds is not None and any(
        segment.policy == DIAGNOSTIC_COST_POLICY for segment in bounds.segments
    ):
        # This fresh-run pilot is never cross-run or cross-tier authority,
        # including when a particular diagnostic call happened to be exact.
        return False
    if classification == "success":
        return True
    directive = slice_.directive
    if classification != "soft_fail_no_changes":
        return False
    if not directive.files_in_scope and not directive.creates_paths:
        return True
    return implementation is not None and _is_handoff_candidate(implementation)


def _resume_identity_payload(
    spec: "Spec",
    tier: "Tier",
    slice_: "Slice",
    impl,
    *,
    tool_authority_digest: str | None = None,
) -> dict:
    if spec.tier_dag is None:
        raise ValueError("cannot create tier resume identity without a tier_dag")
    return {
        "schema_version": _RESUME_IDENTITY_KIND,
        "tier_id": tier.tier_id,
        "slice_id": slice_.id,
        "tier_dag_hash": hash_record(spec.tier_dag),
        "directive_hash": hash_record(slice_.directive),
        "project_base_commit": spec.project_context.base_commit,
        "spec_execution_authority_hash": spec_execution_structure_hash(spec),
        "implementation_base_commit": impl.base_commit,
        "tool_authority_digest": _validated_tool_authority_digest(
            tool_authority_digest
        ),
    }


def attach_resume_identity(
    corpus: "Corpus",
    spec: "Spec",
    tier: "Tier",
    slice_: "Slice",
    impl: Implementation,
    *,
    tool_authority_digest: str | None = None,
) -> Implementation:
    """Bind an Implementation to the exact executable plan it satisfied."""

    # Handoff is operator authority, not an invocation of Tier's routed
    # Implementer.  Tier may derive verification and final-manifest bindings
    # from a Handoff record, but those bindings must retain the absent tool
    # authority that _identity_matches() requires for every Handoff candidate.
    if _is_handoff_candidate(impl):
        tool_authority_digest = None
    payload = _resume_identity_payload(
        spec,
        tier,
        slice_,
        impl,
        tool_authority_digest=tool_authority_digest,
    )
    ref = corpus.put_artifact(
        json.dumps(payload, sort_keys=True).encode("utf-8"),
        kind="resume_identities",
        name_hint=f"{tier.tier_id}_{slice_.id}.json",
    )
    return dataclasses.replace(
        impl,
        artifacts={**(impl.artifacts or {}), "resume_identity": ref},
    )


def _identity_matches(
    corpus: "Corpus",
    spec: "Spec",
    tier: "Tier",
    slice_: "Slice",
    impl: Implementation,
    *,
    tool_authority_digest: str | None | object = _UNSPECIFIED,
    artifact_reader: _BoundedArtifactReader | None = None,
) -> bool:
    ref = (impl.artifacts or {}).get("resume_identity")
    if ref is None:
        return False
    try:
        encoded = (
            corpus.get_artifact_bounded(
                ref,
                max_bytes=_RESUME_IDENTITY_MAX_BYTES,
            )
            if artifact_reader is None
            else artifact_reader(
                ref,
                max_bytes=_RESUME_IDENTITY_MAX_BYTES,
                label="resume identity",
            )
        )
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    current_tool_authority: Any
    if _is_handoff_candidate(impl):
        # Handoff is explicit operator authority, not an invocation of the
        # currently routed Implementer ToolSet.  Its sealed identity must
        # therefore retain the ``None`` authority written by Handoff instead
        # of being invalidated by whichever autonomous route is active when
        # Tier later resumes it.  Required verification remains enforced by
        # ``partition_for_required_verification`` below.
        current_tool_authority = None
    elif tool_authority_digest is _UNSPECIFIED:
        # Backward-compatible library callers that have no active ToolSet can
        # still validate the rest of the sealed payload. Production Tier
        # execution supplies the current per-slice digest explicitly.
        current_tool_authority = payload.get("tool_authority_digest")
    else:
        current_tool_authority = tool_authority_digest
    try:
        expected = _resume_identity_payload(
            spec,
            tier,
            slice_,
            impl,
            tool_authority_digest=current_tool_authority,
        )
    except ValueError:
        return False
    return payload == expected


def _latest_impl_per_slice(
    corpus: "Corpus", slice_ids: set[str], *, refs=None,
) -> dict[str, Implementation]:
    """Return the latest *lineage-attached* Implementation per slice_id.

    Spec ref order is authoritative. Scanning the entire content-addressed
    artifact store admits orphan records left by a crash between record write
    and Spec attachment. When ``refs`` is explicit, the caller must already
    have proved that it is trusted lineage order; artifact-store enumeration,
    timestamps, mtimes, and sorted hashes are never chronology.
    """
    latest: dict[str, Implementation] = {}
    if refs is None:
        spec = corpus.latest_spec()
        refs = spec.implementation_refs if spec is not None else ()
    for ref in refs:
        try:
            impl = corpus.get_record(ref, as_type=Implementation)
        except Exception:  # noqa: BLE001 — forward-compat tolerance
            continue
        if impl.slice_id not in slice_ids:
            continue
        latest[impl.slice_id] = impl
    return latest


def partition_for_resume(
    corpus: "Corpus",
    tier: "Tier",
    *,
    spec: "Spec | None" = None,
    tool_authority_digests: Mapping[str, str | None] | None = None,
    artifact_reader: _BoundedArtifactReader | None = None,
) -> tuple[dict[str, Implementation], tuple["Slice", ...]]:
    """Return ``(landed_impls, pending_slices)`` for a tier's resume.

    - ``landed_impls`` maps slice_id → latest skip-worthy Implementation
      (so the caller can splice the diff_ref + classification into the
      tier's Phase 2 union diff without re-running the implementer).
    - ``pending_slices`` is the subset of ``tier.slices`` that the
      caller must still implement, in original declaration order
      (the tier_dag's ordering is load-bearing for cascade halt
      semantics, so preserve it).
    """
    spec = spec or corpus.latest_spec()
    if spec is None or spec.tier_dag is None:
        return {}, tuple(tier.slices)
    slice_ids = {s.id for s in tier.slices}
    latest = _latest_impl_per_slice(
        corpus, slice_ids, refs=spec.implementation_refs,
    )
    landed: dict[str, Implementation] = {}
    pending: list = []
    for s in tier.slices:
        prior = latest.get(s.id)
        expected_tool_authority: str | None | object = _UNSPECIFIED
        if tool_authority_digests is not None:
            expected_tool_authority = tool_authority_digests.get(
                s.id,
                _UNSPECIFIED,
            )
            # A partially populated current-authority mapping must never turn
            # a missing slice into legacy permissive behavior.
            if expected_tool_authority is _UNSPECIFIED:
                pending.append(s)
                continue
        if (
            prior is not None
            and _is_skip_worthy(prior.classification, s, prior)
            and _identity_matches(
                corpus,
                spec,
                tier,
                s,
                prior,
                tool_authority_digest=expected_tool_authority,
                artifact_reader=artifact_reader,
            )
        ):
            landed[s.id] = prior
        else:
            pending.append(s)
    return landed, tuple(pending)


def _manifest_dependency_implementations(
    corpus: "Corpus",
    spec: "Spec",
    tier: "Tier",
) -> tuple[tuple[str, Implementation], ...] | None:
    """Return the producer's canonical dependency closure, or fail closed."""

    tier_dag = spec.tier_dag
    if tier_dag is None:
        return None
    tiers_by_id = {
        candidate.tier_id: candidate for candidate in tier_dag.tiers
    }
    if tier.tier_id not in tiers_by_id:
        return None
    dependency_ids: list[str] = []
    seen: set[str] = set()

    def visit(tier_id: str) -> bool:
        candidate = tiers_by_id.get(tier_id)
        if candidate is None:
            return False
        for dependency_id in candidate.depends_on:
            if dependency_id in seen:
                continue
            seen.add(dependency_id)
            if not visit(dependency_id):
                return False
            dependency_ids.append(dependency_id)
        return True

    if not visit(tier.tier_id):
        return None

    bindings: list[tuple[str, Implementation]] = []
    for dependency_id in dependency_ids:
        dependency = tiers_by_id[dependency_id]
        partition = partition_for_required_verification(
            corpus,
            dependency,
            spec=spec,
        )
        retained = partition.verified_retained
        if (
            partition.implementation_pending
            or partition.verification_pending
            or set(retained) != {
                slice_.id for slice_ in dependency.slices
            }
        ):
            return None
        bindings.extend(
            (dependency_id, retained[slice_.id])
            for slice_ in dependency.slices
        )
    return tuple(bindings)


def _is_handoff_candidate(implementation: Implementation) -> bool:
    """Return whether ``implementation`` is governed by Handoff evidence.

    Assay records loaded from older stores may expose ``execution`` as either
    the typed dataclass or its mapping projection.  The durable artifact key is
    also authoritative so a malformed/missing recipe cannot silently bypass a
    gate that Handoff attached.
    """

    from ._gate_evidence import (
        CONDITIONAL_ABSENCE_ARTIFACT,
        HANDOFF_AUTHORIZING_FINALIZATION_ARTIFACT,
        HANDOFF_VERIFICATION_ARTIFACT,
    )

    execution = getattr(implementation, "execution", None)
    recipe = (
        execution.get("recipe")
        if isinstance(execution, Mapping)
        else getattr(execution, "recipe", None)
    )
    artifacts = getattr(implementation, "artifacts", None)
    has_handoff_artifact = (
        isinstance(artifacts, Mapping)
        and (
            HANDOFF_VERIFICATION_ARTIFACT in artifacts
            or HANDOFF_AUTHORIZING_FINALIZATION_ARTIFACT in artifacts
            or CONDITIONAL_ABSENCE_ARTIFACT in artifacts
        )
    )
    return recipe == "handoff" or has_handoff_artifact


def _handoff_verification_obligation(
    spec: "Spec",
    slice_id: str,
) -> tuple[str, tuple[str, ...] | None] | None:
    """Return the exact gate and materialized output obligation for one slice.

    ``None`` is a fail-closed malformed authority. Ordinary Handoff records
    retain the historical required ``verify`` gate. A typed authoring
    constraint selects its exact gate and complete materialized output set.
    """

    from crucible.constraints import AuthoringFinalizationConstraint

    configured: dict[str, AuthoringFinalizationConstraint] = {}
    for constraint in (getattr(spec, "constraints", ()) or ()):
        if not isinstance(constraint, AuthoringFinalizationConstraint):
            continue
        if constraint.slice_id in configured:
            return None
        configured[constraint.slice_id] = constraint
    selected = configured.get(slice_id)
    if selected is None:
        return "verify", None

    return selected.gate, tuple(selected.materializes_paths)


def _verification_matches_handoff_authority(
    corpus: "Corpus",
    implementation: Implementation,
    *,
    spec: "Spec",
    tier_id: str,
    slice_id: str,
    spec_execution_authority_hash: str,
    require_gate: bool,
) -> bool:
    """Match Handoff evidence against its current Spec-bound obligation."""

    obligation = _handoff_verification_obligation(spec, slice_id)
    if obligation is None:
        return False
    expected_gate, expected_output_paths = obligation
    from ._gate_evidence import (
        handoff_authoring_finalization_matches,
        verification_matches_implementation,
    )

    if expected_output_paths is None:
        matches = verification_matches_implementation(
            corpus,
            implementation,
            tier_id=tier_id,
            slice_id=slice_id,
            spec_execution_authority_hash=spec_execution_authority_hash,
            require_gate=require_gate,
        )
    else:
        matches = verification_matches_implementation(
            corpus,
            implementation,
            tier_id=tier_id,
            slice_id=slice_id,
            spec_execution_authority_hash=spec_execution_authority_hash,
            require_gate=require_gate,
            expected_gate=expected_gate,
        )
    if not matches:
        return False
    if expected_output_paths is None:
        return True
    return handoff_authoring_finalization_matches(
        corpus,
        implementation,
        tier_id=tier_id,
        slice_id=slice_id,
        spec_execution_authority_hash=spec_execution_authority_hash,
        expected_gate=expected_gate,
        expected_output_paths=expected_output_paths,
    )


def partition_for_required_verification(
    corpus: "Corpus",
    tier: "Tier",
    *,
    spec: "Spec | None" = None,
    tool_authority_digests: Mapping[str, str | None] | None = None,
) -> VerificationResumePartition:
    """Partition resume work without turning a waived gate into a pass.

    This is the required-pass counterpart to :func:`partition_for_resume`.
    Existing callers retain the original two-way structural API.  New callers
    that must prove deterministic verification receive three disjoint sets:

    * exact candidates with sufficient verification authority;
    * exact Handoff candidates needing deterministic replay; and
    * slices whose implementation is missing, failed, or stale.

    Missing, corrupt, waived, unavailable, or candidate-mismatched Handoff or
    Tier evidence lands in ``verification_pending``. It never falls through to
    ``implementation_pending``, because replaying a deterministic gate must not
    acquire or charge an Implementer backend.
    """

    current_spec = spec or corpus.latest_spec()
    retained, implementation_pending = partition_for_resume(
        corpus,
        tier,
        spec=current_spec,
        tool_authority_digests=tool_authority_digests,
    )
    if current_spec is None:
        return VerificationResumePartition(
            verified_retained={},
            verification_pending={},
            implementation_pending=implementation_pending,
        )

    from ._gate_evidence import (
        CONDITIONAL_ABSENCE_ARTIFACT,
        TIER_VERIFICATION_ARTIFACT,
        TIER_VERIFICATION_MANIFEST_ARTIFACT,
        conditional_absence_matches_implementation,
        load_conditional_absence_evidence,
        tier_verification_manifest_matches,
    )

    execution_hash = spec_execution_structure_hash(current_spec)
    all_tiers = {item.tier_id: item for item in current_spec.tier_dag.tiers}
    slice_locations = {
        slice_.id: (owner, slice_)
        for owner in current_spec.tier_dag.tiers
        for slice_ in owner.slices
    }
    all_latest = _latest_impl_per_slice(
        corpus,
        set(slice_locations),
        refs=current_spec.implementation_refs,
    )

    def dependency_tier_ids(tier_id: str) -> set[str]:
        found: set[str] = set()
        pending = list(all_tiers[tier_id].depends_on)
        while pending:
            dependency_id = pending.pop()
            if dependency_id in found or dependency_id not in all_tiers:
                continue
            found.add(dependency_id)
            pending.extend(all_tiers[dependency_id].depends_on)
        return found

    def conditional_match(slice_, implementation) -> bool:
        ref = (implementation.artifacts or {}).get(CONDITIONAL_ABSENCE_ARTIFACT)
        if ref is None or slice_.directive.files_in_scope:
            return False
        try:
            evidence = load_conditional_absence_evidence(corpus, str(ref))
        except (KeyError, OSError, TypeError, ValueError):
            return False
        decision_location = slice_locations.get(evidence.decision_slice_id)
        selected_location = slice_locations.get(evidence.selected_slice_id)
        if decision_location is None or selected_location is None:
            return False
        decision_tier, decision_slice = decision_location
        selected_tier, _selected_slice = selected_location
        if (
            decision_tier.tier_id not in dependency_tier_ids(tier.tier_id)
            or selected_tier.tier_id != tier.tier_id
            or evidence.selector_path not in {
                *decision_slice.directive.files_in_scope,
                *decision_slice.directive.creates_paths,
            }
        ):
            return False
        decision_impl = all_latest.get(evidence.decision_slice_id)
        selected_impl = retained.get(evidence.selected_slice_id)
        if (
            decision_impl is None
            or not _is_skip_worthy(
                decision_impl.classification, decision_slice, decision_impl
            )
            or not _identity_matches(
                corpus,
                current_spec,
                decision_tier,
                decision_slice,
                decision_impl,
            )
            or (
                _is_handoff_candidate(decision_impl)
                and not _verification_matches_handoff_authority(
                    corpus,
                    decision_impl,
                    spec=current_spec,
                    tier_id=decision_tier.tier_id,
                    slice_id=decision_slice.id,
                    spec_execution_authority_hash=execution_hash,
                    require_gate=True,
                )
            )
            or selected_impl is None
            or (
                _is_handoff_candidate(selected_impl)
                and not _verification_matches_handoff_authority(
                    corpus,
                    selected_impl,
                    spec=current_spec,
                    tier_id=selected_tier.tier_id,
                    slice_id=evidence.selected_slice_id,
                    spec_execution_authority_hash=execution_hash,
                    require_gate=True,
                )
            )
        ):
            return False
        return conditional_absence_matches_implementation(
            corpus,
            implementation,
            tier_id=tier.tier_id,
            slice_id=slice_.id,
            spec_execution_authority_hash=execution_hash,
            creates_paths=tuple(slice_.directive.creates_paths),
            decision_slice_id=evidence.decision_slice_id,
            decision_implementation=decision_impl,
            selected_slice_id=evidence.selected_slice_id,
            selected_implementation=selected_impl,
        )

    verified_retained: dict[str, Implementation] = {}
    verification_pending: dict[str, Implementation] = {}
    for slice_ in tier.slices:
        implementation = retained.get(slice_.id)
        if implementation is None:
            continue
        if not _is_handoff_candidate(implementation):
            verified_retained[slice_.id] = implementation
            continue
        if _verification_matches_handoff_authority(
            corpus,
            implementation,
            spec=current_spec,
            tier_id=tier.tier_id,
            slice_id=slice_.id,
            spec_execution_authority_hash=execution_hash,
            require_gate=True,
        ) or conditional_match(slice_, implementation):
            verified_retained[slice_.id] = implementation
        else:
            verification_pending[slice_.id] = implementation

    # Tier evidence is a whole-candidate authority. A per-slice proof cannot
    # survive a sibling replacement, so validate one identical manifest over
    # the complete current tier and its canonical transitive dependencies.
    manifest_refs = tuple(
        str(ref)
        for slice_ in tier.slices
        if (implementation := retained.get(slice_.id)) is not None
        if (
            ref := (implementation.artifacts or {}).get(
                TIER_VERIFICATION_MANIFEST_ARTIFACT
            )
        ) is not None
    )
    tier_evidence_present = any(
        TIER_VERIFICATION_ARTIFACT in (implementation.artifacts or {})
        for implementation in retained.values()
    )
    retained_no_surface_present = any(
        slice_.id in retained
        and not slice_.directive.files_in_scope
        and not slice_.directive.creates_paths
        for slice_ in tier.slices
    )
    tier_manifest_present = bool(
        manifest_refs
        or tier_evidence_present
        or retained_no_surface_present
    )
    tier_manifest_obligations: tuple[str, ...] = ()
    tier_manifest_ref: str | None = None
    tier_manifest_verified = False
    if tier_manifest_present:
        tier_manifest_obligations = tuple(
            slice_.id
            for slice_ in tier.slices
            if (
                not slice_.directive.files_in_scope
                and not slice_.directive.creates_paths
            )
            or (
                (implementation := retained.get(slice_.id)) is not None
                and TIER_VERIFICATION_ARTIFACT
                in (implementation.artifacts or {})
            )
            or slice_.id in verification_pending
        )
        current_implementations = tuple(
            retained[slice_.id]
            for slice_ in tier.slices
            if slice_.id in retained
        )
        unique_manifest_refs = set(manifest_refs)
        dependency_implementations = _manifest_dependency_implementations(
            corpus,
            current_spec,
            tier,
        )
        if (
            tier_manifest_obligations
            and len(current_implementations) == len(tier.slices)
            and len(manifest_refs) == len(current_implementations)
            and len(unique_manifest_refs) == 1
            and dependency_implementations is not None
        ):
            tier_manifest_ref = next(iter(unique_manifest_refs))
            tier_manifest_verified = tier_verification_manifest_matches(
                corpus,
                manifest_ref=tier_manifest_ref,
                tier_id=tier.tier_id,
                spec_execution_authority_hash=execution_hash,
                expected_project_base_commit=(
                    current_spec.project_context.base_commit
                ),
                current_implementations=current_implementations,
                dependency_implementations=dependency_implementations,
                expected_obligation_slice_ids=(
                    tier_manifest_obligations
                ),
                require_gate=True,
            )
        for slice_id in tier_manifest_obligations:
            implementation = retained.get(slice_id)
            if implementation is None:
                continue
            if tier_manifest_verified:
                verification_pending.pop(slice_id, None)
                verified_retained[slice_id] = implementation
            else:
                verified_retained.pop(slice_id, None)
                verification_pending[slice_id] = implementation

    return VerificationResumePartition(
        verified_retained=verified_retained,
        verification_pending=verification_pending,
        implementation_pending=implementation_pending,
        tier_manifest_obligations=tier_manifest_obligations,
        tier_manifest_ref=tier_manifest_ref,
        tier_manifest_verified=tier_manifest_verified,
    )


__all__ = [
    "VerificationResumePartition",
    "attach_resume_identity",
    "partition_for_required_verification",
    "partition_for_resume",
]
