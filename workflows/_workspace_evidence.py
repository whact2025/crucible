# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Dependency-composed workspace evidence for safe Architect resume.

Per-slice and per-tier diffs are snapshots relative to the project base.  A
later tier may legitimately edit a path created or edited by an earlier tier,
so comparing every old snapshot directly with the final checkout produces
false drift.  This module reconstructs the selected, current-plan prefix and
compares the dependency-ordered final postimage instead.

The proof deliberately reuses Reconcile's Git-safe primitives.  It grants an
existing ``creates_paths`` exemption only when the creator is selected by the
current plan's resume identity, the path was absent at the project base, the
creator actually touched it, and the final retained writer's exact Git
type/mode/blob matches the live checkout.

Selection covers the retained prefix.  Tier keeps the writes of a slice that
succeeded before a sibling failed and skips that slice on the next run, so for
Architect's resume the retained slices of the first incomplete tier join the
selection when, and only when, the whole extended state is proven.  A failed,
pending, or unbound sibling is never selected, and Tier admission always
receives the retained prefix alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from assay.canonical import hash_record
from assay.types.artifacts import Implementation

from crucible.rules.cited_path_exists import CitedPathExistsContext

from . import _build_verify, _partition
from .reconcile import (
    _PlanImplementationEvidence,
    _canonical_repo_path,
    _directive_scope,
    _implementation_postimages,
    _path_absent_at_commit,
    _path_allowed_by_scope,
    _recursive_scope_children,
    _resolve_checkout_commit,
    _workspace_matches_postimage,
)


@dataclass(frozen=True, slots=True)
class PlanWorkspaceEvidence:
    """Exact retained-state proof derived from one Spec snapshot."""

    context: CitedPathExistsContext
    retained_tier_ids: tuple[str, ...]
    accepted_tier_ids: tuple[str, ...]
    all_slices_complete: bool
    workspace_evidence_passed: bool
    required_create_paths: frozenset[tuple[str, str, str]]
    selected_implementation_hashes: tuple[tuple[str, str, str], ...]
    rows: tuple[dict[str, object], ...]
    errors: tuple[str, ...]


_ADMISSION_BOUNDARY_REASON = (
    "not attempted: Tier admission receives the retained prefix only"
)


def reconstruct_plan_workspace_evidence(
    *,
    corpus,
    spec,
    source_root: Path,
    admission_tier=None,
) -> PlanWorkspaceEvidence:
    """Reconstruct the maximal current-plan retained tier prefix.

    ``retained_tier_ids`` is the contiguous topo prefix whose slices all have
    skip-worthy, resume-identity-bound Implementations.  It is the ownership
    surface used for collision-aware validation. ``accepted_tier_ids`` is the
    narrower contiguous prefix whose exact records also have required-pass
    Handoff verification evidence and a terminal plan-bound ``landed``
    lifecycle; only it may suppress tier execution.

    Any malformed evidence returns a context carrying deterministic blocking
    errors.  No record-store scan is used: the required-verification resume
    partition selects only lineage-attached records from this exact Spec
    snapshot.  When ``admission_tier`` is provided, its complete directive
    surface is added to the reconstructed workspace expectation even when its
    implementation is still pending. Dependency final writers remain
    authoritative on overlaps; every other path must match project-base
    content or absence. This prevents persistent workspaces from admitting
    unrecorded current-tier dirt before a provider runs.

    The boundary tier is the first tier that is not fully retained.  Every
    dependency of it is in the retained prefix, so the prefix plus its
    retained slices is the dependency-composed state a failed tier leaves
    behind.  Without an ``admission_tier`` those slices obtain the same
    authority as the prefix, but only together: a second proof selects the
    prefix and all of them, and it replaces the prefix proof only when it has
    no error, every expected surface path matches the checkout, and every
    required create is verified.  Otherwise the prefix proof is returned
    unchanged, so an unproven sibling grants nothing and adds no finding.
    A failed, pending, or unbound sibling is never retained and is therefore
    never selected, and no tier after the boundary is eligible.

    ``retained_tier_ids``, ``accepted_tier_ids``, ``all_slices_complete``,
    and ``selected_implementation_hashes`` always describe the prefix alone.
    The boundary tier's row names its retained slices, whether they were
    proven, the ``[slice_id, implementation_hash]`` pairs of the proven
    records, and otherwise why the extended proof was not adopted.  Tier
    admission never receives boundary authority: with an ``admission_tier``
    the result is the prefix proof, because Tier compares only the prefix
    bindings and could not see a record outside its dependency closure.
    """

    prefix = _reconstruct_selection(
        corpus=corpus,
        spec=spec,
        source_root=source_root,
        admission_tier=admission_tier,
        include_boundary=False,
    )
    boundary_index = next(
        (
            index
            for index, row in enumerate(prefix.rows)
            if row["boundary_retained_slices"]
        ),
        None,
    )
    if boundary_index is None:
        return prefix
    if admission_tier is not None:
        return replace(
            prefix,
            rows=_with_boundary_row(
                prefix.rows,
                boundary_index,
                boundary_retained_unproven=[_ADMISSION_BOUNDARY_REASON],
            ),
        )

    extended = _reconstruct_selection(
        corpus=corpus,
        spec=spec,
        source_root=source_root,
        admission_tier=None,
        include_boundary=True,
    )
    if not extended.workspace_evidence_passed:
        # The retained siblings obtain authority only as part of one complete
        # proof.  An incomplete extended proof grants nothing and adds no
        # finding; the row keeps why, because the operator otherwise sees
        # only the collision the unproven sibling leaves behind.
        return replace(
            prefix,
            rows=_with_boundary_row(
                prefix.rows,
                boundary_index,
                boundary_retained_unproven=_unproven_boundary_reasons(
                    extended,
                ),
            ),
        )
    return PlanWorkspaceEvidence(
        context=extended.context,
        retained_tier_ids=prefix.retained_tier_ids,
        accepted_tier_ids=prefix.accepted_tier_ids,
        all_slices_complete=prefix.all_slices_complete,
        workspace_evidence_passed=True,
        required_create_paths=extended.required_create_paths,
        selected_implementation_hashes=prefix.selected_implementation_hashes,
        rows=_with_boundary_row(
            prefix.rows,
            boundary_index,
            boundary_retained_proven=True,
            boundary_retained_implementation_hashes=extended.rows[
                boundary_index
            ]["boundary_retained_implementation_hashes"],
        ),
        errors=extended.errors,
    )


def _with_boundary_row(
    rows: tuple[dict[str, object], ...],
    boundary_index: int,
    **changes: object,
) -> tuple[dict[str, object], ...]:
    """Return ``rows`` with the boundary tier's row updated."""

    return tuple(
        {**row, **changes} if index == boundary_index else row
        for index, row in enumerate(rows)
    )


def _unproven_boundary_reasons(extended: PlanWorkspaceEvidence) -> list[str]:
    """Return why the extended proof is incomplete, in a stable order."""

    context = extended.context
    return [
        *extended.errors,
        *(
            f"{path}: workspace does not match the composed postimage"
            for path in sorted(
                context.expected_plan_surface_paths
                - context.verified_plan_surface_paths
            )
        ),
        *(
            f"{tier_id}/{slice_id}: declared create '{path}' is not verified"
            for tier_id, slice_id, path in sorted(
                extended.required_create_paths
                - context.verified_create_paths
            )
        ),
    ]


def _reconstruct_selection(
    *,
    corpus,
    spec,
    source_root: Path,
    admission_tier,
    include_boundary: bool,
) -> PlanWorkspaceEvidence:
    """Prove the retained prefix, optionally with the boundary tier's slices.

    ``include_boundary`` only adds the retained slices of the boundary tier
    to the proven selection.  The prefix fields and every row value except
    the boundary bindings are the same in both modes.
    """

    repo = Path(source_root).resolve()
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    retained_tier_ids: list[str] = []
    accepted_tier_ids: list[str] = []
    selected: list[tuple[object, object, Implementation]] = []
    selected_hashes: list[tuple[str, str, str]] = []

    tier_dag = getattr(spec, "tier_dag", None)
    if tier_dag is None:
        context = CitedPathExistsContext(
            workspace_root=repo,
            submodule_path=None,
        )
        return PlanWorkspaceEvidence(
            context=context,
            retained_tier_ids=(),
            accepted_tier_ids=(),
            all_slices_complete=False,
            workspace_evidence_passed=True,
            required_create_paths=frozenset(),
            selected_implementation_hashes=(),
            rows=(),
            errors=(),
        )

    ordered_tiers = tuple(tier_dag.topo_order())
    retained_open = True
    accepted_open = True
    tier_ancestors: dict[str, frozenset[str]] = {}

    for tier in ordered_tiers:
        ancestors = set(tier.depends_on)
        for dependency in tier.depends_on:
            ancestors.update(tier_ancestors.get(dependency, ()))
        tier_ancestors[tier.tier_id] = frozenset(ancestors)

        partition = _partition.partition_for_required_verification(
            corpus,
            tier,
            spec=spec,
        )
        landed = partition.retained
        pending = partition.implementation_pending
        verification_pending = partition.verification_pending
        canonical_ids = {item.id for item in tier.slices}
        complete = not pending and set(landed) == canonical_ids
        verification_complete = (
            complete
            and not verification_pending
            and set(partition.verified_retained) == canonical_ids
        )
        lifecycle_accepted = False
        lifecycle_reason = "not evaluated"
        if complete:
            lifecycle_accepted, lifecycle_reason = (
                _build_verify._tier_lifecycle_acceptance(
                    corpus,
                    spec,
                    tier,
                    landed,
                    allow_staged_spec=True,
                )
            )

        retained_here = retained_open and complete
        # Tier commits the journal of a slice that succeeded before a sibling
        # failed, and the next run skips it.  Only the first incomplete tier
        # can hold such records as part of the state that produced this tree:
        # every dependency of it is retained.
        boundary_here = retained_open and not complete and bool(landed)
        accepted_here = (
            accepted_open
            and retained_here
            and verification_complete
            and lifecycle_accepted
        )
        if retained_here:
            retained_tier_ids.append(tier.tier_id)
            for slice_ in tier.slices:
                implementation = landed[slice_.id]
                selected.append((tier, slice_, implementation))
                selected_hashes.append((
                    tier.tier_id,
                    slice_.id,
                    hash_record(implementation),
                ))
        else:
            retained_open = False

        if accepted_here:
            accepted_tier_ids.append(tier.tier_id)
        else:
            accepted_open = False

        # The boundary records follow every prefix record in ``selected`` and
        # stay out of the prefix bindings that Tier admission compares.
        boundary_hashes: list[list[str]] = []
        if boundary_here and include_boundary:
            for slice_ in tier.slices:
                if slice_.id not in landed:
                    continue
                implementation = landed[slice_.id]
                selected.append((tier, slice_, implementation))
                boundary_hashes.append([
                    slice_.id,
                    hash_record(implementation),
                ])

        # A completed record set after a gap is not a prefix and therefore
        # cannot be interpreted safely as the state that produced this tree.
        if complete and not retained_here:
            errors.append(
                f"{tier.tier_id}: plan-bound implementations occur after "
                "the retained-prefix boundary"
            )
        # A later tier may still carry a sound historical landing while an
        # earlier retained implementation has just been superseded by a
        # handoff or review correction.  It cannot join the accepted prefix
        # until that predecessor is re-landed, but it is not corrupt evidence:
        # Architect must be allowed to dispatch the predecessor and then
        # re-evaluate later skips against the newly attached lifecycle.

        rows.append({
            "tier_id": tier.tier_id,
            "landed_slices": sorted(landed),
            "pending_slices": [item.id for item in pending],
            "verified_slices": sorted(partition.verified_retained),
            "verification_pending_slices": sorted(verification_pending),
            "tier_manifest_obligations": list(
                partition.tier_manifest_obligations
            ),
            "tier_manifest_ref": partition.tier_manifest_ref,
            "tier_manifest_verified": partition.tier_manifest_verified,
            "retained_prefix": retained_here,
            "verification_complete": verification_complete,
            "lifecycle_accepted": lifecycle_accepted,
            "lifecycle_reason": lifecycle_reason,
            "boundary_retained_slices": (
                sorted(landed) if boundary_here else []
            ),
            "boundary_retained_proven": False,
            "boundary_retained_implementation_hashes": boundary_hashes,
            "boundary_retained_unproven": [],
        })

    successful_plan_bound_slices: set[tuple[str, str]] = set()
    required_create_paths: set[tuple[str, str, str]] = set()
    candidate_create_paths: set[tuple[str, str, str]] = set()
    verified_create_paths: set[tuple[str, str, str]] = set()
    successful_records: list[_PlanImplementationEvidence] = []

    current_head = _resolve_checkout_commit(repo, "HEAD")
    project_base = _resolve_checkout_commit(
        repo,
        getattr(spec.project_context, "base_commit", ""),
    )
    if project_base is None:
        errors.append("declared project base does not resolve in the checkout")
    if current_head is None:
        errors.append("current checkout HEAD does not resolve")
    elif project_base is not None and current_head != project_base:
        errors.append("current checkout HEAD differs from the project base")

    resolved_implementation_bases: dict[str, str | None] = {}
    base_path_absence: dict[str, bool] = {}
    selected_slice_map: dict[tuple[str, str], object] = {}

    for tier, slice_, implementation in selected:
        identity = (tier.tier_id, slice_.id)
        selected_slice_map[identity] = slice_
        # A required-pass conditional absence is retained and accepted, but
        # its soft-fail record is authority that declared outputs must remain
        # absent. It must not waive create collisions or turn those paths into
        # required products. Reconcile applies the same success-only rule.
        if implementation.classification == "success":
            successful_plan_bound_slices.add(identity)
            required_create_paths.update(
                (tier.tier_id, slice_.id, path)
                for path in slice_.directive.creates_paths
            )
        if implementation.base_commit not in resolved_implementation_bases:
            resolved_implementation_bases[implementation.base_commit] = (
                _resolve_checkout_commit(repo, implementation.base_commit)
            )
        implementation_base = resolved_implementation_bases[
            implementation.base_commit
        ]
        base_compatible = bool(
            current_head is not None
            and project_base is not None
            and implementation_base is not None
            and current_head == project_base
            and implementation_base == project_base
        )
        successful_records.append(_PlanImplementationEvidence(
            tier_id=tier.tier_id,
            slice_id=slice_.id,
            implementation=implementation,
            scope=frozenset((
                *slice_.directive.files_in_scope,
                *slice_.directive.creates_paths,
            )),
            files_in_scope=frozenset(slice_.directive.files_in_scope),
            base_compatible=base_compatible,
        ))
        if not base_compatible or project_base is None:
            continue
        modified = set(implementation.files_modified)
        for path in slice_.directive.creates_paths:
            if path not in modified:
                continue
            if path not in base_path_absence:
                base_path_absence[path] = _path_absent_at_commit(
                    repo,
                    project_base,
                    path,
                )
            if base_path_absence[path]:
                candidate_create_paths.add((tier.tier_id, slice_.id, path))

    evidence_states: dict[tuple[str, str], dict] = {}
    evidence_touched_paths: dict[tuple[str, str], frozenset[str]] = {}
    record_scope_policies: dict[tuple[str, str], object] = {}
    record_files_in_scope_policies: dict[tuple[str, str], object] = {}
    evidence_valid: set[tuple[str, str]] = set()

    for record in successful_records:
        identity = (record.tier_id, record.slice_id)
        implementation = record.implementation
        if project_base is None or not record.base_compatible:
            errors.append(
                f"{record.tier_id}/{record.slice_id}: implementation, "
                "project, and checkout bases do not agree"
            )
            continue
        scope_policy = _directive_scope(repo, project_base, record.scope)
        if scope_policy is None:
            errors.append(
                f"{record.tier_id}/{record.slice_id}: directive surface "
                "is unsafe or cannot be resolved"
            )
            continue
        record_scope_policies[identity] = scope_policy
        files_in_scope_policy = _directive_scope(
            repo,
            project_base,
            record.files_in_scope,
        )
        if files_in_scope_policy is None:
            errors.append(
                f"{record.tier_id}/{record.slice_id}: files_in_scope surface "
                "is unsafe or cannot be resolved"
            )
            continue
        record_files_in_scope_policies[identity] = files_in_scope_policy
        files_modified = tuple(implementation.files_modified)
        modified = set(files_modified)
        if implementation.scope_escapes:
            errors.append(
                f"{record.tier_id}/{record.slice_id}: successful "
                f"Implementation carries scope escapes "
                f"{sorted(implementation.scope_escapes)!r}"
            )
            continue
        if len(modified) != len(files_modified):
            errors.append(
                f"{record.tier_id}/{record.slice_id}: files_modified "
                "contains duplicate paths"
            )
            continue
        if (
            any(not _canonical_repo_path(repo, path) for path in modified)
            or any(
                not _path_allowed_by_scope(path, scope_policy)
                for path in modified
            )
        ):
            errors.append(
                f"{record.tier_id}/{record.slice_id}: implementation paths "
                "are non-canonical or outside the directive surface"
            )
            continue
        try:
            diff_bytes = corpus.get_artifact(implementation.diff_ref)
        except (OSError, TypeError, ValueError, KeyError):
            errors.append(
                f"{record.tier_id}/{record.slice_id}: diff_ref is unavailable"
            )
            continue
        reconstructed = _implementation_postimages(
            repo,
            project_base,
            diff_bytes,
            modified,
        )
        if reconstructed is None:
            errors.append(
                f"{record.tier_id}/{record.slice_id}: cumulative diff "
                "cannot be reconstructed against the agreed base"
            )
            continue
        postimages, touched, rename_pairs = reconstructed
        rename_sources = {source for source, _ in rename_pairs}
        metadata_touched = touched - rename_sources
        if (
            any(
                not _path_allowed_by_scope(path, scope_policy)
                for path in touched
            )
            or metadata_touched != frozenset(modified)
        ):
            errors.append(
                f"{record.tier_id}/{record.slice_id}: diff touched paths "
                f"{sorted(touched)!r} (renames {sorted(rename_pairs)!r}) "
                "do not match the directive surface/files_modified "
                f"{sorted(modified)!r}"
            )
            continue
        evidence_states[identity] = postimages
        evidence_touched_paths[identity] = touched
        evidence_valid.add(identity)

    planned_surface: set[str] = set()
    surface_valid = True
    recursive_children_cache: dict[str, set[str] | None] = {}
    for record in successful_records:
        identity = (record.tier_id, record.slice_id)
        scope_policy = record_scope_policies.get(identity)
        if scope_policy is None:
            surface_valid = False
            continue
        planned_surface.update(scope_policy.exact_paths)
        for directory in scope_policy.recursive_directories:
            if directory not in recursive_children_cache:
                recursive_children_cache[directory] = (
                    _recursive_scope_children(repo, project_base, directory)
                    if project_base is not None
                    else None
                )
            children = recursive_children_cache[directory]
            if children is None:
                surface_valid = False
                errors.append(
                    f"recursive scope '{directory}' cannot be expanded"
                )
            else:
                planned_surface.update(children)
        planned_surface.update(evidence_touched_paths.get(identity, ()))

    if admission_tier is not None and project_base is not None:
        for slice_ in admission_tier.slices:
            admission_scope = frozenset((
                *slice_.directive.files_in_scope,
                *slice_.directive.creates_paths,
            ))
            scope_policy = _directive_scope(
                repo,
                project_base,
                admission_scope,
            )
            if scope_policy is None:
                surface_valid = False
                errors.append(
                    f"{admission_tier.tier_id}/{slice_.id}: admission "
                    "directive surface is unsafe or cannot be resolved"
                )
                continue
            planned_surface.update(scope_policy.exact_paths)
            for directory in scope_policy.recursive_directories:
                if directory not in recursive_children_cache:
                    recursive_children_cache[directory] = (
                        _recursive_scope_children(
                            repo,
                            project_base,
                            directory,
                        )
                    )
                children = recursive_children_cache[directory]
                if children is None:
                    surface_valid = False
                    errors.append(
                        f"admission recursive scope '{directory}' cannot "
                        "be expanded"
                    )
                else:
                    planned_surface.update(children)

    expected_states: dict[str, object] = {}
    chain_valid_paths: set[str] = set()
    planned_deletion_paths: set[str] = set()
    untouched_paths: set[str] = set()
    if project_base is not None and surface_valid:
        for path in sorted(planned_surface):
            writers = [
                record
                for record in successful_records
                if path in evidence_touched_paths.get(
                    (record.tier_id, record.slice_id),
                    (),
                )
            ]
            if not writers:
                untouched_paths.add(path)
                continue
            chain_valid = True
            for prior, current in zip(writers, writers[1:], strict=False):
                if prior.tier_id not in tier_ancestors.get(
                    current.tier_id,
                    (),
                ):
                    errors.append(
                        f"{path}: incomparable writers "
                        f"{prior.tier_id}/{prior.slice_id} and "
                        f"{current.tier_id}/{current.slice_id}"
                    )
                    chain_valid = False
                    break
            final_writer = writers[-1]
            final_identity = (final_writer.tier_id, final_writer.slice_id)
            postimage = evidence_states.get(final_identity, {}).get(path)
            if not chain_valid or postimage is None:
                continue
            expected_states[path] = postimage
            chain_valid_paths.add(path)
            if postimage.mode is None:
                planned_deletion_paths.add(path)

        if untouched_paths:
            base_reconstruction = _implementation_postimages(
                repo,
                project_base,
                None,
                untouched_paths,
            )
            if base_reconstruction is None:
                errors.append(
                    "untouched retained surface cannot be reconstructed "
                    "from the agreed base"
                )
            else:
                base_states, base_touched, base_renames = base_reconstruction
                if base_touched or base_renames:
                    errors.append(
                        "base reconstruction unexpectedly reported retained "
                        "surface changes"
                    )
                else:
                    expected_states.update(base_states)
                    chain_valid_paths.update(base_states)

    expected_plan_surface_paths = frozenset(planned_surface)
    verified_plan_surface_paths = frozenset(
        path
        for path, postimage in expected_states.items()
        if _workspace_matches_postimage(repo, path, postimage)
    )

    for candidate in candidate_create_paths:
        creator_identity = (candidate[0], candidate[1])
        path = candidate[2]
        postimage = expected_states.get(path)
        creator_postimage = evidence_states.get(creator_identity, {}).get(path)
        if (
            creator_identity in evidence_valid
            and creator_postimage is not None
            and creator_postimage.mode is not None
            and path in chain_valid_paths
            and path in verified_plan_surface_paths
            and postimage is not None
            and postimage.mode is not None
        ):
            verified_create_paths.add(candidate)

    verified_recursive_scope_paths = frozenset(
        (*identity, directory)
        for identity in (evidence_valid & successful_plan_bound_slices)
        for directory in record_files_in_scope_policies[
            identity
        ].recursive_directories
    )

    context = CitedPathExistsContext(
        workspace_root=repo,
        submodule_path=None,
        successful_plan_bound_slices=frozenset(
            successful_plan_bound_slices,
        ),
        verified_create_paths=frozenset(verified_create_paths),
        expected_plan_surface_paths=expected_plan_surface_paths,
        verified_plan_surface_paths=verified_plan_surface_paths,
        verified_recursive_scope_paths=verified_recursive_scope_paths,
        planned_deletion_paths=frozenset(planned_deletion_paths),
        postimage_evidence_errors=tuple(errors),
    )
    workspace_evidence_passed = (
        not errors
        and expected_plan_surface_paths == verified_plan_surface_paths
        and required_create_paths.issubset(verified_create_paths)
    )

    return PlanWorkspaceEvidence(
        context=context,
        retained_tier_ids=tuple(retained_tier_ids),
        accepted_tier_ids=tuple(accepted_tier_ids),
        all_slices_complete=(
            len(retained_tier_ids) == len(ordered_tiers)
        ),
        workspace_evidence_passed=workspace_evidence_passed,
        required_create_paths=frozenset(required_create_paths),
        selected_implementation_hashes=tuple(selected_hashes),
        rows=tuple(rows),
        errors=tuple(errors),
    )


__all__ = [
    "PlanWorkspaceEvidence",
    "reconstruct_plan_workspace_evidence",
]
