# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.rules.cited_path_exists`` — filesystem-grounded.

Planning checks cover files_in_scope existence, creates_paths collisions, and
creates_paths parent anchors. Post-implementation context additionally proves
exact modified create paths and requires successful outputs to remain present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assay import (
    BudgetConstraint,
    ImplementationDirective,
    ParityCriterion,
    ProjectContext,
    RuleContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
)
from crucible.rules.cited_path_exists import (
    CitedPathExistsContext,
    cited_path_exists,
)


def _spec(*, files_in_scope=(), creates_paths=()) -> Spec:
    directive = ImplementationDirective(
        slice_id="s",
        intent="t",
        files_in_scope=tuple(files_in_scope),
        creates_paths=tuple(creates_paths),
        verify_mode="build_only",
        blast_radius={"max_files": 5},
    )
    slice_ = Slice(id="s", directive=directive)
    tier = Tier(tier_id="t0", depends_on=(), slices=(slice_,))
    return Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="kit", repo_url="ssh://example/kit.git",
            base_commit="deadbeef", branch="main",
            submodule_path=None, runners={},
        ),
        execution=None,
        intent="t",
        constraints=(BudgetConstraint(usd_limit=1.0),),
        done_criteria=(ParityCriterion(reference_commit="a", test_name="t"),),
        plan_policy=None,
        tier_dag=TierDAG(tiers=(tier,)),
    )


# ----------------------------------------------------------------------
# No-op cases
# ----------------------------------------------------------------------


def test_no_tier_dag_returns_empty(tmp_path):
    spec = _spec()
    # Replace tier_dag with None.
    from dataclasses import replace
    spec = replace(spec, tier_dag=None)
    assert cited_path_exists(spec, RuleContext(workspace_root=tmp_path)) == ()


def test_ctx_none_returns_empty(tmp_path):
    spec = _spec(files_in_scope=["nonexistent.txt"])
    # Calling with ctx=None should no-op (belt-and-suspenders;
    # validate.py also skips arity-2 rules when ctx is None).
    assert cited_path_exists(spec, None) == ()


def test_workspace_root_none_returns_empty(tmp_path):
    spec = _spec(files_in_scope=["nonexistent.txt"])
    assert cited_path_exists(spec, RuleContext(workspace_root=None)) == ()


# ----------------------------------------------------------------------
# files_in_scope check
# ----------------------------------------------------------------------


def test_files_in_scope_existing_no_findings(tmp_path):
    (tmp_path / "real.txt").write_text("hi")
    spec = _spec(files_in_scope=["real.txt"])
    findings = cited_path_exists(spec, RuleContext(workspace_root=tmp_path))
    assert findings == ()


def test_files_in_scope_missing_critical(tmp_path):
    spec = _spec(files_in_scope=["phantom.txt"])
    findings = cited_path_exists(spec, RuleContext(workspace_root=tmp_path))
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "Critical"
    assert f.category == "phantom_reference"
    assert "phantom.txt" in f.id
    assert "missing_files_in_scope" in f.id


@pytest.mark.parametrize("scope_path", ["obsolete", "obsolete/"])
def test_verified_recursive_scope_deletion_allows_missing_root(
    tmp_path,
    scope_path,
):
    descendants = frozenset({
        "obsolete/first.txt",
        "obsolete/nested/second.txt",
    })
    spec = _spec(files_in_scope=[scope_path])
    ctx = CitedPathExistsContext(
        workspace_root=tmp_path,
        expected_plan_surface_paths=descendants,
        verified_plan_surface_paths=descendants,
        verified_recursive_scope_paths=frozenset({("t0", "s", "obsolete")}),
        planned_deletion_paths=descendants,
    )

    assert cited_path_exists(spec, ctx) == ()


@pytest.mark.parametrize(
    ("expected", "verified", "deleted"),
    [
        (frozenset(), frozenset(), frozenset()),
        (
            frozenset({"obsolete/first.txt", "obsolete/second.txt"}),
            frozenset({"obsolete/first.txt", "obsolete/second.txt"}),
            frozenset({"obsolete/first.txt"}),
        ),
        (
            frozenset({"obsolete/first.txt", "obsolete/second.txt"}),
            frozenset({"obsolete/first.txt"}),
            frozenset({"obsolete/first.txt", "obsolete/second.txt"}),
        ),
        (
            frozenset({"obsolete/first.txt", "obsolete/second.txt"}),
            frozenset({"obsolete/first.txt", "obsolete/second.txt"}),
            frozenset({"unrelated/deleted.txt"}),
        ),
    ],
    ids=(
        "empty-descendants",
        "partial-deletion",
        "partial-verification",
        "unrelated-deletion",
    ),
)
def test_recursive_scope_deletion_authority_fails_closed(
    tmp_path,
    expected,
    verified,
    deleted,
):
    spec = _spec(files_in_scope=["obsolete"])
    ctx = CitedPathExistsContext(
        workspace_root=tmp_path,
        expected_plan_surface_paths=expected,
        verified_plan_surface_paths=verified,
        verified_recursive_scope_paths=frozenset({("t0", "s", "obsolete")}),
        planned_deletion_paths=deleted,
    )

    findings = cited_path_exists(spec, ctx)

    assert any(
        finding.id == "missing_files_in_scope__s__obsolete"
        and finding.severity == "Critical"
        and finding.category == "phantom_reference"
        for finding in findings
    )


def test_deleted_descendants_do_not_authorize_exact_absent_parent(tmp_path):
    descendants = frozenset({"generated/child.txt"})
    spec = _spec(files_in_scope=["generated"])
    ctx = CitedPathExistsContext(
        workspace_root=tmp_path,
        expected_plan_surface_paths=descendants,
        verified_plan_surface_paths=descendants,
        planned_deletion_paths=descendants,
    )

    findings = cited_path_exists(spec, ctx)

    assert any(
        finding.id == "missing_files_in_scope__s__generated"
        and finding.category == "phantom_reference"
        for finding in findings
    )


def test_verified_exact_file_deletion_remains_allowed(tmp_path):
    path = "obsolete.txt"
    spec = _spec(files_in_scope=[path])
    ctx = CitedPathExistsContext(
        workspace_root=tmp_path,
        expected_plan_surface_paths=frozenset({path}),
        verified_plan_surface_paths=frozenset({path}),
        planned_deletion_paths=frozenset({path}),
    )

    assert cited_path_exists(spec, ctx) == ()


@pytest.mark.parametrize(
    "field_name,path",
    [
        ("files_in_scope", "../outside.txt"),
        ("files_in_scope", "/etc/passwd"),
        ("creates_paths", "../new.txt"),
        ("creates_paths", "C:/Windows/new.txt"),
    ],
)
def test_scope_paths_must_be_project_root_relative(
    tmp_path,
    field_name,
    path,
):
    findings = cited_path_exists(
        _spec(**{field_name: [path]}),
        RuleContext(workspace_root=tmp_path),
    )

    assert len(findings) == 1
    assert findings[0].severity == "Critical"
    assert findings[0].category == "scope_escape"
    assert "project-root-confined" in findings[0].evidence[0].excerpt


@pytest.mark.parametrize("field_name", ["files_in_scope", "creates_paths"])
def test_scope_path_rejects_symlinked_ancestor_escape(tmp_path, field_name):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "alias").symlink_to(outside, target_is_directory=True)
    if field_name == "files_in_scope":
        (outside / "escaped.txt").write_text("outside")

    findings = cited_path_exists(
        _spec(**{field_name: ["alias/escaped.txt"]}),
        RuleContext(workspace_root=tmp_path),
    )

    assert len(findings) == 1
    assert findings[0].severity == "Critical"
    assert findings[0].category == "scope_escape"


# ----------------------------------------------------------------------
# creates_paths checks
# ----------------------------------------------------------------------


def test_creates_paths_not_existing_no_findings(tmp_path):
    # Create the parent dir but NOT the file → anchor present, no collision.
    (tmp_path / "src").mkdir()
    spec = _spec(creates_paths=["src/newfile.cpp"])
    findings = cited_path_exists(spec, RuleContext(workspace_root=tmp_path))
    assert findings == ()


def test_creates_paths_already_exists_high(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "existing.cpp").write_text("// already here")
    spec = _spec(creates_paths=["src/existing.cpp"])
    findings = cited_path_exists(spec, RuleContext(workspace_root=tmp_path))
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "High"
    assert f.category == "overwrite_collision"
    assert "already_exists" in f.id


def test_directory_valued_creates_path_is_rejected_as_non_file(tmp_path):
    spec = _spec(creates_paths=["generated/"])

    findings = cited_path_exists(
        spec,
        RuleContext(workspace_root=tmp_path),
    )

    assert len(findings) == 1
    assert findings[0].severity == "High"
    assert findings[0].category == "creates_path_not_file"


def test_successful_current_slice_allows_post_implementation_create(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "created.cpp").write_text("// implemented")
    spec = _spec(creates_paths=["src/created.cpp"])
    ctx = CitedPathExistsContext(
        workspace_root=tmp_path,
        successful_plan_bound_slices=frozenset({("t0", "s")}),
        verified_create_paths=frozenset({("t0", "s", "src/created.cpp")}),
    )

    assert cited_path_exists(spec, ctx) == ()


@pytest.mark.parametrize(
    "authorized_slice",
    [
        ("other-tier", "s"),
        ("t0", "other-slice"),
    ],
)
def test_post_implementation_exemption_requires_exact_slice(
    tmp_path, authorized_slice,
):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "created.cpp").write_text("// implemented")
    spec = _spec(creates_paths=["src/created.cpp"])
    ctx = CitedPathExistsContext(
        workspace_root=tmp_path,
        successful_plan_bound_slices=frozenset({("t0", "s")}),
        verified_create_paths=frozenset({
            (*authorized_slice, "src/created.cpp"),
        }),
    )

    findings = cited_path_exists(spec, ctx)

    assert len(findings) == 1
    assert findings[0].category == "overwrite_collision"


def test_modified_path_without_successful_slice_cannot_authorize(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "created.cpp").write_text("// implemented")
    spec = _spec(creates_paths=["src/created.cpp"])
    ctx = CitedPathExistsContext(
        workspace_root=tmp_path,
        verified_create_paths=frozenset({("t0", "s", "src/created.cpp")}),
    )

    findings = cited_path_exists(spec, ctx)

    assert len(findings) == 1
    assert findings[0].category == "overwrite_collision"


def test_successful_current_slice_requires_created_output_to_exist(tmp_path):
    (tmp_path / "src").mkdir()
    spec = _spec(creates_paths=["src/created.cpp"])
    ctx = CitedPathExistsContext(
        workspace_root=tmp_path,
        successful_plan_bound_slices=frozenset({("t0", "s")}),
        verified_create_paths=frozenset({("t0", "s", "src/created.cpp")}),
    )

    findings = cited_path_exists(spec, ctx)

    assert len(findings) == 1
    assert findings[0].severity == "High"
    assert findings[0].category == "implementation_output_missing"


def test_slice_and_directive_ids_must_match(tmp_path):
    from dataclasses import replace

    spec = _spec()
    assert spec.tier_dag is not None
    tier = spec.tier_dag.tiers[0]
    slice_ = tier.slices[0]
    malformed = replace(
        spec,
        tier_dag=TierDAG(tiers=(replace(
            tier,
            slices=(replace(
                slice_,
                directive=replace(slice_.directive, slice_id="alias"),
            ),),
        ),)),
    )

    findings = cited_path_exists(
        malformed,
        RuleContext(workspace_root=tmp_path),
    )

    assert len(findings) == 1
    assert findings[0].severity == "Critical"
    assert findings[0].category == "slice_identity_mismatch"


def test_creates_paths_missing_anchor_low(tmp_path):
    # Parent dir 'src/nested' doesn't exist → anchor surfaced, but at LOW
    # severity (informational, non-blocking): the implementer mkdir -p's it.
    # Formerly High/blocking, which broke greenfield new-crate specs.
    spec = _spec(creates_paths=["src/nested/newfile.cpp"])
    findings = cited_path_exists(spec, RuleContext(workspace_root=tmp_path))
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "Low"
    assert f.category == "phantom_anchor"
    assert "missing_anchor" in f.id


def test_missing_create_anchors_are_grouped_and_sorted_per_slice(tmp_path):
    spec = _spec(creates_paths=[
        "zeta/nested/two.cpp",
        "alpha/one.cpp",
        "zeta/nested/one.cpp",
    ])

    findings = cited_path_exists(spec, RuleContext(workspace_root=tmp_path))

    assert len(findings) == 1
    finding = findings[0]
    assert finding.id == "creates_path_missing_anchor__t0__s"
    assert finding.severity == "Low"
    assert finding.category == "phantom_anchor"
    assert [evidence.excerpt.split("'")[1] for evidence in finding.evidence] == [
        "alpha/one.cpp",
        "zeta/nested/one.cpp",
        "zeta/nested/two.cpp",
    ]
    assert all(
        str(tmp_path) in evidence.excerpt for evidence in finding.evidence
    )
    assert finding.suggested_action.count("implicitly mkdir") == 1


def test_anchor_grouping_does_not_group_high_or_critical_findings(tmp_path):
    (tmp_path / "existing").mkdir()
    (tmp_path / "existing" / "one.cpp").write_text("one")
    (tmp_path / "existing" / "two.cpp").write_text("two")
    spec = _spec(
        files_in_scope=["missing/one.cpp", "missing/two.cpp"],
        creates_paths=["existing/one.cpp", "existing/two.cpp"],
    )

    findings = cited_path_exists(spec, RuleContext(workspace_root=tmp_path))

    assert [finding.severity for finding in findings] == [
        "Critical",
        "Critical",
        "High",
        "High",
    ]
    assert len({finding.id for finding in findings}) == 4


def test_files_in_scope_same_slice_create_does_not_hide_phantom(tmp_path):
    spec = _spec(
        files_in_scope=["crates/newcrate/src/lib.rs"],
        creates_paths=["crates/newcrate/src/lib.rs"],
    )
    findings = cited_path_exists(spec, RuleContext(workspace_root=tmp_path))

    assert any(f.category == "phantom_reference" for f in findings)


def test_files_in_scope_created_by_dependency_tier_is_exempt(tmp_path):
    from dataclasses import replace

    path = "crates/newcrate/src/lib.rs"
    seed = _spec(creates_paths=[path])
    assert seed.tier_dag is not None
    create_tier = seed.tier_dag.tiers[0]
    edit_slice = replace(
        create_tier.slices[0],
        id="edit",
        directive=replace(
            create_tier.slices[0].directive,
            slice_id="edit",
            files_in_scope=(path,),
            creates_paths=(),
        ),
    )
    edit_tier = Tier(
        tier_id="t1",
        depends_on=("t0",),
        slices=(edit_slice,),
    )
    spec = replace(
        seed,
        tier_dag=TierDAG(tiers=(create_tier, edit_tier)),
    )

    findings = cited_path_exists(
        spec,
        RuleContext(workspace_root=tmp_path),
    )

    assert not any(
        finding.category == "phantom_reference"
        and finding.evidence[0].location.startswith("t1.")
        for finding in findings
    )


def test_files_in_scope_unrelated_create_does_not_hide_phantom(tmp_path):
    from dataclasses import replace

    path = "crates/newcrate/src/lib.rs"
    seed = _spec(creates_paths=[path])
    assert seed.tier_dag is not None
    create_tier = seed.tier_dag.tiers[0]
    edit_slice = replace(
        create_tier.slices[0],
        id="edit",
        directive=replace(
            create_tier.slices[0].directive,
            slice_id="edit",
            files_in_scope=(path,),
            creates_paths=(),
        ),
    )
    unrelated_tier = Tier(
        tier_id="t1",
        depends_on=(),
        slices=(edit_slice,),
    )
    spec = replace(
        seed,
        tier_dag=TierDAG(tiers=(create_tier, unrelated_tier)),
    )

    findings = cited_path_exists(
        spec,
        RuleContext(workspace_root=tmp_path),
    )

    assert any(
        finding.category == "phantom_reference"
        and finding.evidence[0].location.startswith("t1.")
        for finding in findings
    )


def test_creates_paths_root_anchor_always_exists(tmp_path):
    # Single-component path → parent is the workspace_root itself.
    # workspace_root exists by construction so no anchor finding.
    spec = _spec(creates_paths=["Makefile"])
    findings = cited_path_exists(spec, RuleContext(workspace_root=tmp_path))
    assert findings == ()


# ----------------------------------------------------------------------
# Submodule path threading
# ----------------------------------------------------------------------


def test_submodule_path_used_for_resolution(tmp_path):
    """workspace_root / submodule_path / cited_path."""
    (tmp_path / "kit").mkdir()
    (tmp_path / "kit" / "real.h").write_text("// hi")
    spec = _spec(files_in_scope=["real.h"])
    ctx = RuleContext(workspace_root=tmp_path, submodule_path="kit")
    findings = cited_path_exists(spec, ctx)
    assert findings == ()


def test_submodule_path_makes_path_missing(tmp_path):
    """Same path resolves DIFFERENTLY through submodule_path — and is missing."""
    (tmp_path / "real.h").write_text("// at top")  # NOT under kit/
    spec = _spec(files_in_scope=["real.h"])
    ctx = RuleContext(workspace_root=tmp_path, submodule_path="kit")
    findings = cited_path_exists(spec, ctx)
    assert len(findings) == 1
    assert findings[0].severity == "Critical"


# ----------------------------------------------------------------------
# Mixed: combine all three signals at once
# ----------------------------------------------------------------------


def test_mixed_findings_all_three_signals(tmp_path):
    (tmp_path / "exists.txt").write_text("hi")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "collision.cpp").write_text("// collides")
    spec = _spec(
        files_in_scope=["exists.txt", "phantom.txt"],
        creates_paths=["src/collision.cpp", "missing/dir/file.cpp"],
    )
    findings = cited_path_exists(spec, RuleContext(workspace_root=tmp_path))
    # Expected: 3 findings — 1 Critical (phantom files_in_scope),
    # 1 High (overwrite collision), 1 Low (missing anchor — informational).
    by_sev = {f.severity for f in findings}
    by_cat = {f.category for f in findings}
    assert len(findings) == 3
    assert by_sev == {"Critical", "High", "Low"}
    assert by_cat == {"phantom_reference", "overwrite_collision", "phantom_anchor"}
