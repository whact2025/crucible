# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``cited_path_exists`` — context-aware: cited slice paths must be filesystem-grounded.

Filesystem checks per slice are anchored at
``ctx.workspace_root / ctx.submodule_path``:

1. **files_in_scope** entries MUST exist on disk.
   Severity: **Critical**. The slice can't edit a file that isn't
   there; the implementer will produce an empty diff (or silently
   write to a different file if it's smart enough to re-resolve).
   This is the canonical "phantom path" signal — typically caused
   by the planner inferring a header path from a namespace structure
   without verifying against the filesystem.

   During post-implementation reconciliation only, a missing path is exempt
   when exact Git postimage evidence proves its planned deletion. Recursive
   directory scopes have no Git directory postimage, so their missing root is
   exempt only when the successful plan-bound slice was base-classified as
   recursive, it has a nonempty expected descendant set, and every expected
   descendant is both deletion-recorded and postimage-verified.

2. **creates_paths** entries MUST NOT exist on disk before implementation.
   Severity: **High**. A creates_path that already exists is either
   stale state (previous run's output not cleaned up) or an
   accidental overwrite of an existing file. Either way the slice's
   intent is suspect — the planner should split the slice into a
   delete + recreate, or change the bucket from creates_paths to
   files_in_scope (edit-in-place semantics). During post-implementation
   reconciliation only, the exact path is exempt when it appears in the
   exact declaring slice's successful, current-plan Implementation, its
   implementation/project/checkout bases agree, and the live Git
   mode/type/blob matches the dependency-ordered final postimage. A plain
   ``RuleContext`` carries no such authority, so planning-time collisions
   remain blocking. The authority is per exact triple. A retained slice of
   an incomplete tier holds it once Crucible proves its postimage; a failed,
   pending, or unrecorded sibling never inherits it.

3. **creates_paths** parent directory existence.
   Severity: **Low** (informational, non-blocking). A creates_path
   whose parent dir is missing means the implementer ``mkdir -p``s it
   to land the file — usually fine, surfaced so the planner can confirm
   the implicit mkdir is intended, but not a stamp-blocking defect.
   Single-component paths (e.g. ``Makefile``) resolve their parent to
   ``.`` which always exists, so they pass trivially.

   ``creates_paths`` is a file-output contract (matching Assay's planner and
   implementer prompts), not a recursive scope. Entries with a trailing slash
   are rejected as malformed instead of reaching reconciliation as an
   impossible exact directory postimage. Use ``files_in_scope`` for a
   recursive directory surface and list each new file in ``creates_paths``.
   Multiple missing parents in one slice are grouped into one finding with
   sorted per-path evidence, so advisory volume reflects affected work units
   rather than raw file count.

4. During post-implementation reconciliation, every ``creates_paths`` entry
   in a successful current-plan slice MUST exist. Severity: **High**. This
   prevents a stale success record from landing after its declared output was
   removed.

``Slice.id`` is the canonical lifecycle identity. A mismatch with
``ImplementationDirective.slice_id`` is a **Critical** plan-structure error;
aliases cannot grant or consume implementation authority.

Note on the **files_in_scope** check: a path an actual dependency tier
declares it *creates* is exempt — a file created upstream and edited
downstream is legitimately absent from the pre-execution tree, not a phantom.
Same-tier, downstream, and unrelated creation declarations grant no exemption.

Context-aware rule (arity 2). When ``ctx is None`` or
``ctx.workspace_root is None`` the rule no-ops (returns ``()``) —
``assay.validate``'s dispatcher already skips arity-2 rules when ctx
isn't provided, so the inner guard is belt-and-suspenders for callers
that thread a partial context.

Verdicts produced by this rule reflect the local working tree at
validation time. Two machines validating the same Spec can disagree
if their checkouts diverge — that disagreement is the diagnostic:
the spec's ``project_context.base_commit`` declares the intended tree.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from assay import Evidence, Finding, RuleContext, Spec


@dataclass(frozen=True)
class CitedPathExistsContext(RuleContext):
    """Opt-in authority for post-implementation create-path validation.

    ``successful_plan_bound_slices`` entries are ``(tier_id, slice_id)`` pairs
    whose latest lineage-attached Implementation has been verified as
    successful and bound to the current TierDAG. ``verified_create_paths``
    grants collision authority only to exact ``(tier_id, slice_id, path)``
    triples whose diff-derived postimage, dependency ordering, declared
    scope, base identity, and live workspace Git state were all verified.
    The plan-surface fields carry the same proof for edit/delete paths.
    ``verified_recursive_scope_paths`` grants descendant-derived deletion
    authority only to exact ``(tier_id, slice_id, normalized_directory)``
    triples whose directive was classified as recursive against the agreed
    base and whose successful Implementation evidence was validated. Callers
    must derive these sets from Crucible's resume partition rather than from
    unbound records in the artifact store.

    The create authority is per exact triple. A retained slice of an
    incomplete tier holds it once Crucible proves its postimage; a failed,
    pending, or unrecorded sibling never inherits it. ``scope_covers_intent``
    reads ``verified_create_paths`` with ``successful_plan_bound_slices`` for
    the same proof.

    Planning and pre-implementation validation deliberately use the base
    :class:`RuleContext`, which cannot suppress an existing-path collision.
    """

    successful_plan_bound_slices: frozenset[tuple[str, str]] = field(
        default_factory=frozenset,
    )
    verified_create_paths: frozenset[tuple[str, str, str]] = field(
        default_factory=frozenset,
    )
    expected_plan_surface_paths: frozenset[str] = field(
        default_factory=frozenset,
    )
    verified_plan_surface_paths: frozenset[str] = field(
        default_factory=frozenset,
    )
    verified_recursive_scope_paths: frozenset[tuple[str, str, str]] = field(
        default_factory=frozenset,
    )
    planned_deletion_paths: frozenset[str] = field(default_factory=frozenset)
    postimage_evidence_errors: tuple[str, ...] = ()


def _slice_identity_mismatch_finding(
    *, tier_id: str, slice_id: str, directive_slice_id: str,
) -> Finding:
    return Finding(
        id=f"slice_identity_mismatch__{tier_id}__{slice_id}",
        severity="Critical",
        category="slice_identity_mismatch",
        evidence=(
            Evidence(
                kind="plan_structure",
                location=f"{tier_id}.{slice_id}.directive.slice_id",
                excerpt=(
                    f"Slice.id '{slice_id}' does not match directive.slice_id "
                    f"'{directive_slice_id}'"
                ),
            ),
        ),
        suggested_action=(
            "Use one canonical slice identifier: set Slice.id and "
            "ImplementationDirective.slice_id to the same value before "
            "planning or implementation."
        ),
        source="deterministic",
    )


def _missing_files_in_scope_finding(
    *, slice_id: str, tier_id: str, path: str, full: Path
) -> Finding:
    return Finding(
        id=f"missing_files_in_scope__{slice_id}__{path}",
        severity="Critical",
        category="phantom_reference",
        evidence=(
            Evidence(
                kind="filesystem_check",
                location=f"{tier_id}.{slice_id}.files_in_scope",
                excerpt=f"path '{path}' does not exist at {full}",
            ),
        ),
        suggested_action=(
            f"Verify '{path}' against the actual repository layout. If the "
            f"planner inferred this path from a namespace / module / package "
            f"name, use ``find_definition`` to resolve the symbol's real "
            f"declaration site, then correct files_in_scope accordingly. "
            f"This finding blocks plan stamping because the implementer "
            f"would otherwise produce an empty diff."
        ),
        source="deterministic",
    )


def _path_confinement_error(base: Path, path: str) -> str | None:
    """Return why a planner path is not confined beneath ``base``."""

    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        return "absolute paths are forbidden"
    if ".." in Path(normalized).parts:
        return "parent traversal segments are forbidden"
    try:
        resolved_base = base.resolve()
        resolved = (base / normalized).resolve(strict=False)
        resolved.relative_to(resolved_base)
    except (OSError, RuntimeError, ValueError):
        return "the resolved path escapes the project root"
    return None


def _is_verified_planned_deletion(
    path: str,
    *,
    recursive_scope_is_verified: bool,
    expected_plan_surface_paths: frozenset[str],
    verified_plan_surface_paths: frozenset[str],
    planned_deletion_paths: frozenset[str],
) -> bool:
    """Return whether postimage evidence proves ``path`` was deleted.

    Exact file and symlink deletions have their own Git postimage. Git does
    not represent directories, so a recursive scope root can only inherit
    deletion authority from a nonempty, completely verified descendant set.
    An empty or partially deleted/verified set fails closed.
    """

    if path in planned_deletion_paths:
        return path in verified_plan_surface_paths
    if not recursive_scope_is_verified:
        return False

    prefix = f"{path.rstrip('/')}/"
    expected_descendants = frozenset(
        candidate
        for candidate in expected_plan_surface_paths
        if candidate.startswith(prefix)
    )
    return bool(expected_descendants) and (
        expected_descendants <= planned_deletion_paths
        and expected_descendants <= verified_plan_surface_paths
    )


def _scope_path_escape_finding(
    *,
    slice_id: str,
    tier_id: str,
    field_name: str,
    path: str,
    reason: str,
) -> Finding:
    return Finding(
        id=f"scope_path_escape__{slice_id}__{field_name}__{path}",
        severity="Critical",
        category="scope_escape",
        evidence=(
            Evidence(
                kind="path_containment",
                location=f"{tier_id}.{slice_id}.{field_name}",
                excerpt=f"path {path!r} is not project-root-confined: {reason}",
            ),
        ),
        suggested_action=(
            "Use the exact repository-root-relative path without an absolute "
            "prefix, parent traversal, or a symlinked ancestor that resolves "
            "outside the project root."
        ),
        source="deterministic",
    )


def _creates_path_already_exists_finding(
    *, slice_id: str, tier_id: str, path: str, full: Path
) -> Finding:
    return Finding(
        id=f"creates_path_already_exists__{slice_id}__{path}",
        severity="High",
        category="overwrite_collision",
        evidence=(
            Evidence(
                kind="filesystem_check",
                location=f"{tier_id}.{slice_id}.creates_paths",
                excerpt=f"path '{path}' already exists at {full}",
            ),
        ),
        suggested_action=(
            f"The slice declares it CREATES '{path}', but the file already "
            f"exists. Either move the path from creates_paths to "
            f"files_in_scope (edit-in-place semantics), or split this into "
            f"a delete + recreate sequence across two slices, or clean up "
            f"stale state from a prior run before re-validating."
        ),
        source="deterministic",
    )


def _missing_creates_anchor_finding(
    *,
    slice_id: str,
    tier_id: str,
    anchors: tuple[tuple[str, Path], ...],
) -> Finding:
    rendered_anchors = ", ".join(
        f"{path!r} -> {parent}" for path, parent in anchors
    )
    return Finding(
        id=f"creates_path_missing_anchor__{tier_id}__{slice_id}",
        # Low (informational, non-blocking): the implementer always
        # mkdir(parents=True)s a creates_path's parent, so a missing parent
        # is "usually fine, but worth surfacing" (per this rule's contract) —
        # NOT a stamp-blocking defect. Blocking it (the former High) made
        # greenfield specs that stand up a new crate/module fail the gate on
        # every new directory. Kept as a surfaced nudge so the planner can
        # still confirm the implicit mkdir is intended.
        severity="Low",
        category="phantom_anchor",
        evidence=tuple(
            Evidence(
                kind="filesystem_check",
                location=f"{tier_id}.{slice_id}.creates_paths",
                excerpt=(
                    f"creates_paths entry '{path}' has missing parent "
                    f"directory {parent}"
                ),
            )
            for path, parent in anchors
        ),
        suggested_action=(
            "The following declared paths have missing parent directories: "
            f"{rendered_anchors}. The implementer would have to implicitly "
            "mkdir(parents=True) to land them. If that is intended, confirm "
            "it once in the slice intent; otherwise split parent-directory "
            "creation into a predecessor slice or correct the paths."
        ),
        source="deterministic",
    )


def _implemented_create_path_missing_finding(
    *, slice_id: str, tier_id: str, path: str, full: Path,
) -> Finding:
    return Finding(
        id=f"implemented_create_path_missing__{slice_id}__{path}",
        severity="High",
        category="implementation_output_missing",
        evidence=(
            Evidence(
                kind="filesystem_check",
                location=f"{tier_id}.{slice_id}.creates_paths",
                excerpt=(
                    f"successful current-plan slice declares '{path}', but "
                    f"the implemented output does not exist at {full}"
                ),
            ),
        ),
        suggested_action=(
            f"Restore or re-run slice '{slice_id}' so its declared created "
            f"path '{path}' exists before reconciliation."
        ),
        source="deterministic",
    )


def _directory_creates_path_finding(
    *, slice_id: str, tier_id: str, path: str,
) -> Finding:
    return Finding(
        id=f"directory_creates_path__{slice_id}__{path}",
        severity="High",
        category="creates_path_not_file",
        evidence=(
            Evidence(
                kind="plan_structure",
                location=f"{tier_id}.{slice_id}.creates_paths",
                excerpt=(
                    f"creates_paths entry '{path}' ends with '/', but "
                    "creates_paths declares exact new files"
                ),
            ),
        ),
        suggested_action=(
            "Declare the directory in files_in_scope when recursive write "
            "scope is intended, and list each new file explicitly in "
            "creates_paths."
        ),
        source="deterministic",
    )


def _postimage_evidence_error_finding(*, index: int, error: str) -> Finding:
    return Finding(
        id=f"postimage_evidence_error__{index}",
        severity="Critical",
        category="implementation_evidence_invalid",
        evidence=(
            Evidence(
                kind="implementation_evidence",
                location="reconcile.postimage",
                excerpt=error,
            ),
        ),
        suggested_action=(
            "Re-run the affected tier so it attaches a successful, "
            "current-plan cumulative Implementation whose diff, scope, and "
            "base identity agree."
        ),
        source="deterministic",
    )


def _plan_surface_postimage_mismatch_finding(
    *, path: str, full: Path,
) -> Finding:
    return Finding(
        id=f"plan_surface_postimage_mismatch__{path}",
        severity="High",
        category="implementation_postimage_mismatch",
        evidence=(
            Evidence(
                kind="filesystem_check",
                location="reconcile.plan_surface",
                excerpt=(
                    f"current workspace path '{path}' at {full} does not "
                    "match the final plan-bound Git mode/type/blob postimage"
                ),
            ),
        ),
        suggested_action=(
            "Restore the recorded final postimage or re-run the final "
            "dependent slice so the current workspace state is captured by "
            "new plan-bound Implementation evidence."
        ),
        source="deterministic",
    )


def cited_path_exists(spec: Spec, ctx: RuleContext) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    if ctx is None or ctx.workspace_root is None:
        # Belt-and-suspenders: assay.validate already skips arity-2
        # rules when ctx is None, but a partial RuleContext with
        # workspace_root unset would otherwise crash the Path math.
        return ()

    base = Path(ctx.workspace_root)
    if ctx.submodule_path:
        base = base / ctx.submodule_path

    # Only the explicit post-implementation context can authorize an
    # existing creates_path. In particular, do not infer completion merely
    # from Spec.implementation_refs: those refs must first be resolved and
    # checked for latest-attached, successful, current-plan resume identity.
    successful_plan_bound_slices = (
        ctx.successful_plan_bound_slices
        if isinstance(ctx, CitedPathExistsContext)
        else frozenset()
    )
    verified_create_paths = (
        ctx.verified_create_paths
        if isinstance(ctx, CitedPathExistsContext)
        else frozenset()
    )
    expected_plan_surface_paths = (
        ctx.expected_plan_surface_paths
        if isinstance(ctx, CitedPathExistsContext)
        else frozenset()
    )
    verified_plan_surface_paths = (
        ctx.verified_plan_surface_paths
        if isinstance(ctx, CitedPathExistsContext)
        else frozenset()
    )
    verified_recursive_scope_paths = (
        ctx.verified_recursive_scope_paths
        if isinstance(ctx, CitedPathExistsContext)
        else frozenset()
    )
    postimage_evidence_errors = (
        ctx.postimage_evidence_errors
        if isinstance(ctx, CitedPathExistsContext)
        else ()
    )
    planned_deletion_paths = (
        ctx.planned_deletion_paths
        if isinstance(ctx, CitedPathExistsContext)
        else frozenset()
    )

    # A file created by a dependency tier and edited downstream is not a
    # phantom. Do not grant that exemption merely because the same, a later,
    # or an unrelated slice also lists the path in creates_paths: without an
    # explicit dependency the file does not exist at this slice's authority
    # boundary.
    tiers_by_id = {
        candidate_tier.tier_id: candidate_tier
        for candidate_tier in spec.tier_dag.tiers
    }
    created_paths_by_tier = {
        candidate_tier.tier_id: {
            path
            for candidate_slice in candidate_tier.slices
            for path in (candidate_slice.directive.creates_paths or ())
            if _path_confinement_error(base, path) is None
        }
        for candidate_tier in spec.tier_dag.tiers
    }

    def _ancestor_tier_ids(tier_id: str) -> set[str]:
        ancestors: set[str] = set()
        pending = list(tiers_by_id[tier_id].depends_on)
        while pending:
            dependency_id = pending.pop()
            if dependency_id in ancestors:
                continue
            dependency = tiers_by_id.get(dependency_id)
            if dependency is None:
                # The DAG-structure rule owns the unknown dependency finding.
                continue
            ancestors.add(dependency_id)
            pending.extend(dependency.depends_on)
        return ancestors

    upstream_created_paths = {
        tier_id: {
            path
            for ancestor_id in _ancestor_tier_ids(tier_id)
            for path in created_paths_by_tier[ancestor_id]
        }
        for tier_id in tiers_by_id
    }

    findings: list[Finding] = []
    findings.extend(
        _postimage_evidence_error_finding(index=index, error=error)
        for index, error in enumerate(postimage_evidence_errors)
    )
    findings.extend(
        _plan_surface_postimage_mismatch_finding(
            path=path,
            full=base / path,
        )
        for path in sorted(
            expected_plan_surface_paths - verified_plan_surface_paths
        )
    )
    for tier in spec.tier_dag.tiers:
        for slice_ in tier.slices:
            tier_id = tier.tier_id
            # Slice.id is the canonical identity used by Implementation and
            # resume binding. A divergent directive id is malformed rather
            # than an alternate alias and must fail both plan and reconcile
            # validation.
            slice_id = slice_.id
            missing_create_anchors: set[tuple[str, Path]] = set()
            directive_slice_id = slice_.directive.slice_id
            if directive_slice_id != slice_id:
                findings.append(
                    _slice_identity_mismatch_finding(
                        tier_id=tier_id,
                        slice_id=slice_id,
                        directive_slice_id=directive_slice_id,
                    )
                )

            for p in (slice_.directive.files_in_scope or ()):
                confinement_error = _path_confinement_error(base, p)
                if confinement_error is not None:
                    findings.append(
                        _scope_path_escape_finding(
                            slice_id=slice_id,
                            tier_id=tier_id,
                            field_name="files_in_scope",
                            path=p,
                            reason=confinement_error,
                        )
                    )
                    continue
                # A file the dag creates upstream is legitimately editable
                # downstream even though it isn't on disk at validation time.
                if p in upstream_created_paths[tier_id]:
                    continue
                full = base / p
                exists = os.path.lexists(full)
                verified_planned_deletion = _is_verified_planned_deletion(
                    p,
                    recursive_scope_is_verified=(
                        tier_id, slice_id, p.rstrip("/")
                    ) in verified_recursive_scope_paths,
                    expected_plan_surface_paths=expected_plan_surface_paths,
                    verified_plan_surface_paths=verified_plan_surface_paths,
                    planned_deletion_paths=planned_deletion_paths,
                )
                if not exists and not verified_planned_deletion:
                    findings.append(
                        _missing_files_in_scope_finding(
                            slice_id=slice_id,
                            tier_id=tier_id,
                            path=p,
                            full=full,
                        )
                    )

            for p in (slice_.directive.creates_paths or ()):
                confinement_error = _path_confinement_error(base, p)
                if confinement_error is not None:
                    findings.append(
                        _scope_path_escape_finding(
                            slice_id=slice_id,
                            tier_id=tier_id,
                            field_name="creates_paths",
                            path=p,
                            reason=confinement_error,
                        )
                    )
                    continue
                if p.endswith("/"):
                    findings.append(
                        _directory_creates_path_finding(
                            slice_id=slice_id,
                            tier_id=tier_id,
                            path=p,
                        )
                    )
                    continue
                full = base / p
                slice_key = (tier_id, slice_id)
                path_key = (tier_id, slice_id, p)
                path_is_authorized = (
                    slice_key in successful_plan_bound_slices
                    and path_key in verified_create_paths
                )
                surface_postimage_mismatch = (
                    p in expected_plan_surface_paths
                    and p not in verified_plan_surface_paths
                )
                # lexists treats a symlink itself as present even when its
                # target is absent. Postimage authority separately verifies
                # the exact symlink mode + target bytes.
                exists = os.path.lexists(full)
                if (
                    not exists
                    and slice_key in successful_plan_bound_slices
                    and not surface_postimage_mismatch
                ):
                    findings.append(
                        _implemented_create_path_missing_finding(
                            slice_id=slice_id,
                            tier_id=tier_id,
                            path=p,
                            full=full,
                        )
                    )
                elif (
                    exists
                    and not path_is_authorized
                    and not surface_postimage_mismatch
                ):
                    findings.append(
                        _creates_path_already_exists_finding(
                            slice_id=slice_id,
                            tier_id=tier_id,
                            path=p,
                            full=full,
                        )
                    )
                parent = full.parent
                # Single-component paths give parent == base; base always
                # exists, so no false positive at the root. A missing parent
                # is surfaced (the implementer mkdir(parents=True)s it) but at
                # Low severity — see _missing_creates_anchor_finding — so a
                # greenfield new-crate scaffold doesn't BLOCK plan stamping.
                if not parent.exists():
                    missing_create_anchors.add((p, parent))

            if missing_create_anchors:
                findings.append(_missing_creates_anchor_finding(
                    slice_id=slice_id,
                    tier_id=tier_id,
                    anchors=tuple(sorted(
                        missing_create_anchors,
                        key=lambda item: (item[0], str(item[1])),
                    )),
                ))

    return tuple(findings)
