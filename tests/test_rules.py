# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the 10 deterministic rules in ``crucible.rules``.

Each rule has at least one passing case and one failing case.
Integration with ``assay.validate`` is exercised at the end via the
default importlib resolver.
"""

from __future__ import annotations

import importlib

import pytest

from assay import (
    AssertionCriterion,
    BlastRadiusConstraint,
    ImplementationDirective,
    ProjectContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
    ValidationRuleset,
)
from assay.primitives.validate import validate

from crucible.rules import (
    blast_radius,
    creates_declaration,
    creates_paths,
    cycle,
    deletion_violates_goal,
    implementer_verify,
    parallel_tier_conflicts,
    phantom_reference,
    runner_path_prefix,
    ungrounded_citation,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _ctx(*, runners: dict[str, str] | None = None) -> ProjectContext:
    return ProjectContext(
        project="demo",
        repo_url="https://example.invalid/demo.git",
        base_commit="0" * 40,
        branch=None,
        submodule_path=None,
        runners=runners or {},
    )


def _slice(
    sid: str,
    *,
    intent: str = "do thing",
    files_in_scope: tuple[str, ...] = (),
    creates_paths: tuple[str, ...] = (),
    verify_mode: str = "tests",
    blast_radius: dict | None = None,
) -> Slice:
    return Slice(
        id=sid,
        directive=ImplementationDirective(
            slice_id=sid,
            intent=intent,
            files_in_scope=files_in_scope,
            creates_paths=creates_paths,
            verify_mode=verify_mode,
            blast_radius=blast_radius or {},
        ),
    )


def _spec(
    *,
    tier_dag: TierDAG | None = None,
    constraints: tuple = (),
    done_criteria: tuple = (),
    ctx: ProjectContext | None = None,
    intent: str = "test",
) -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ctx or _ctx(),
        execution=None,
        intent=intent,
        constraints=constraints,
        done_criteria=done_criteria,
        plan_policy=None,
        depends_on=(),
        tier_dag=tier_dag,
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


# ----------------------------------------------------------------------
# cycle
# ----------------------------------------------------------------------


def test_cycle_passes_on_dag():
    dag = TierDAG(
        tiers=(
            Tier(tier_id="t0", slices=(_slice("a"),), depends_on=()),
            Tier(tier_id="t1", slices=(_slice("b"),), depends_on=("t0",)),
        )
    )
    assert cycle(_spec(tier_dag=dag)) == ()


def test_cycle_fails_on_cycle():
    dag = TierDAG(
        tiers=(
            Tier(tier_id="t0", slices=(_slice("a"),), depends_on=("t1",)),
            Tier(tier_id="t1", slices=(_slice("b"),), depends_on=("t0",)),
        )
    )
    findings = cycle(_spec(tier_dag=dag))
    assert len(findings) == 1
    assert findings[0].severity == "Critical"
    assert findings[0].id == "cycle"


def test_cycle_noop_for_empty_dag():
    assert cycle(_spec(tier_dag=None)) == ()


# ----------------------------------------------------------------------
# runner_path_prefix
# ----------------------------------------------------------------------


def test_runner_path_prefix_passes_on_relative():
    s = _spec(ctx=_ctx(runners={"build": "tools/build.sh", "test": "scripts/test.py"}))
    assert runner_path_prefix(s) == ()


def test_runner_path_prefix_flags_absolute():
    s = _spec(ctx=_ctx(runners={"build": "/usr/bin/build"}))
    findings = runner_path_prefix(s)
    assert any("absolute" in f.id for f in findings)


def test_runner_path_prefix_flags_traversal():
    s = _spec(ctx=_ctx(runners={"build": "../other-repo/build.sh"}))
    findings = runner_path_prefix(s)
    assert any("traversal" in f.id for f in findings)


def test_runner_path_prefix_flags_empty():
    s = _spec(ctx=_ctx(runners={"build": ""}))
    findings = runner_path_prefix(s)
    assert any("empty" in f.id for f in findings)


def test_runner_path_prefix_flags_windows_drive():
    s = _spec(ctx=_ctx(runners={"build": "C:\\tools\\build.bat"}))
    findings = runner_path_prefix(s)
    assert any("windows_drive" in f.id for f in findings)


# ----------------------------------------------------------------------
# implementer_verify
# ----------------------------------------------------------------------


def test_implementer_verify_accepts_known():
    for mode in ("skip", "build_only", "tests", "bench"):
        dag = TierDAG(
            tiers=(
                Tier(tier_id="t0", slices=(_slice("a", verify_mode=mode),), depends_on=()),
            )
        )
        assert implementer_verify(_spec(tier_dag=dag)) == ()


def test_implementer_verify_rejects_unknown():
    dag = TierDAG(
        tiers=(
            Tier(tier_id="t0", slices=(_slice("a", verify_mode="run_em_all"),), depends_on=()),
        )
    )
    findings = implementer_verify(_spec(tier_dag=dag))
    assert len(findings) == 1
    assert findings[0].severity == "High"


def test_implementer_verify_rejects_empty_string():
    dag = TierDAG(
        tiers=(
            Tier(tier_id="t0", slices=(_slice("a", verify_mode=""),), depends_on=()),
        )
    )
    findings = implementer_verify(_spec(tier_dag=dag))
    assert len(findings) == 1


# ----------------------------------------------------------------------
# creates_declaration
# ----------------------------------------------------------------------


def test_creates_declaration_passes_when_declared():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="create new file `foo/bar.py` with helpers",
                        creates_paths=("foo/bar.py",),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    assert creates_declaration(_spec(tier_dag=dag)) == ()


def test_creates_declaration_flags_undeclared():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="create `foo/bar.py` with helpers",
                        creates_paths=(),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    findings = creates_declaration(_spec(tier_dag=dag))
    assert len(findings) == 1
    assert "foo/bar.py" in findings[0].id


def test_creates_declaration_ignores_intent_without_create_verb():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="modify `foo/bar.py` to handle UTF-8",
                        files_in_scope=("foo/bar.py",),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    assert creates_declaration(_spec(tier_dag=dag)) == ()


def test_creates_declaration_matches_full_cpp_extension():
    # Regression: ordered alternation used to capture `.c` from `.cpp`
    # (matching `c` before `cpp`), so a correctly-declared .cpp create
    # was flagged as undeclared. The path token must extract whole.
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent=(
                            "Then add the intern pool "
                            "src/plugins/DebugNameInternPool.cpp here"
                        ),
                        creates_paths=("src/plugins/DebugNameInternPool.cpp",),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    assert creates_declaration(_spec(tier_dag=dag)) == ()


def test_creates_declaration_ignores_rewrite_of_existing_file():
    # Regression: substring verb test fired on "write" inside "Rewrite",
    # mis-flagging a modification of an existing file as a create.
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="Rewrite src/UtilsVulkan.cpp to accept DebugName",
                        files_in_scope=("src/UtilsVulkan.cpp",),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    assert creates_declaration(_spec(tier_dag=dag)) == ()


def test_creates_declaration_word_boundary_excludes_address():
    # "address" must not match the "add" creation verb.
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="Document the ip address in `docs/net.md`",
                        files_in_scope=("docs/net.md",),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    assert creates_declaration(_spec(tier_dag=dag)) == ()


def test_creates_declaration_matches_declared_by_basename():
    # Intent names a bare filename; creates_paths holds the full path.
    # Basename matching keeps a correctly-declared create from firing.
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="add DebugNameInternPool.h to the plugin",
                        creates_paths=(
                            "rendering/plugins/carb/DebugNameInternPool.h",
                        ),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    assert creates_declaration(_spec(tier_dag=dag)) == ()


def test_creates_declaration_inflected_verb_still_flags():
    # "creates"/"adds" (inflected) must still trigger on a truly
    # undeclared create — the word-boundary fix keeps the positive case.
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="This step creates `foo/new_helper.py` from scratch",
                        creates_paths=(),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    findings = creates_declaration(_spec(tier_dag=dag))
    assert len(findings) == 1
    assert "foo/new_helper.py" in findings[0].id


# ----------------------------------------------------------------------
# blast_radius
# ----------------------------------------------------------------------


def test_blast_radius_passes_when_within_prefix():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", files_in_scope=("src/foo.py",)),),
                depends_on=(),
            ),
        )
    )
    s = _spec(
        tier_dag=dag,
        constraints=(BlastRadiusConstraint(no_edits_outside=("src/",)),),
    )
    assert blast_radius(s) == ()


def test_blast_radius_flags_out_of_prefix():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", files_in_scope=("docs/note.md",)),),
                depends_on=(),
            ),
        )
    )
    s = _spec(
        tier_dag=dag,
        constraints=(BlastRadiusConstraint(no_edits_outside=("src/",)),),
    )
    findings = blast_radius(s)
    assert any("out_of_blast_radius" in f.id for f in findings)


def test_blast_radius_tolerates_workspace_relative_path_vs_submodule_constraint():
    # The planner sometimes emits workspace-relative paths prefixed with the
    # project/submodule name (`demo/src/foo.py`) while the constraint is
    # project-root-relative (`src/`). The rule normalizes the leading
    # project-name segment so the two conventions compare equal — no false
    # out_of_blast_radius finding.
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", files_in_scope=("demo/src/foo.py",)),),
                depends_on=(),
            ),
        )
    )
    s = _spec(
        tier_dag=dag,
        constraints=(BlastRadiusConstraint(no_edits_outside=("src/",)),),
    )
    assert blast_radius(s) == ()


def test_blast_radius_tolerates_submodule_path_vs_workspace_constraint():
    # Mirror image: project-root-relative path, project-prefixed constraint.
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", files_in_scope=("src/foo.py",)),),
                depends_on=(),
            ),
        )
    )
    s = _spec(
        tier_dag=dag,
        constraints=(BlastRadiusConstraint(no_edits_outside=("demo/src/",)),),
    )
    assert blast_radius(s) == ()


def test_blast_radius_whole_submodule_constraint_allows_everything():
    # A constraint that is exactly the project prefix ("demo/" under
    # project="demo") means "anywhere under the submodule root" — it must allow
    # every path, not normalize-to-empty and flag everything. Regression guard
    # for the constraint-emptying bug in the prefix normalization.
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        files_in_scope=("src/foo.py",),
                        creates_paths=("demo/docs/new.md",),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    s = _spec(
        tier_dag=dag,
        constraints=(BlastRadiusConstraint(no_edits_outside=("demo/",)),),
    )
    assert blast_radius(s) == ()


def test_blast_radius_still_flags_genuine_escape_after_normalization():
    # Normalization must not mask a real out-of-scope path.
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", files_in_scope=("demo/docs/note.md",)),),
                depends_on=(),
            ),
        )
    )
    s = _spec(
        tier_dag=dag,
        constraints=(BlastRadiusConstraint(no_edits_outside=("src/",)),),
    )
    findings = blast_radius(s)
    assert any("out_of_blast_radius" in f.id for f in findings)


def test_blast_radius_flags_max_files_exceeded():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        files_in_scope=("a.py", "b.py", "c.py", "d.py"),
                        blast_radius={"max_files": 2},
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    findings = blast_radius(_spec(tier_dag=dag))
    assert any("max_files_exceeded" in f.id for f in findings)


def test_blast_radius_noop_without_constraint():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", files_in_scope=("anywhere.py",)),),
                depends_on=(),
            ),
        )
    )
    assert blast_radius(_spec(tier_dag=dag)) == ()


# ----------------------------------------------------------------------
# parallel_tier_conflicts
# ----------------------------------------------------------------------


def test_parallel_tier_conflicts_passes_on_disjoint_siblings():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice("a", files_in_scope=("a.py",)),
                    _slice("b", files_in_scope=("b.py",)),
                ),
                depends_on=(),
            ),
        )
    )
    assert parallel_tier_conflicts(_spec(tier_dag=dag)) == ()


def test_parallel_tier_conflicts_flags_shared_file():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice("a", files_in_scope=("shared.py",)),
                    _slice("b", files_in_scope=("shared.py",)),
                ),
                depends_on=(),
            ),
        )
    )
    findings = parallel_tier_conflicts(_spec(tier_dag=dag))
    assert len(findings) == 1
    assert findings[0].severity == "High"


def test_parallel_tier_conflicts_ignores_single_slice_tier():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", files_in_scope=("a.py",)),),
                depends_on=(),
            ),
        )
    )
    assert parallel_tier_conflicts(_spec(tier_dag=dag)) == ()


# ----------------------------------------------------------------------
# creates_paths
# ----------------------------------------------------------------------


def test_creates_paths_passes_on_unique_creates():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", creates_paths=("a.py",)),),
                depends_on=(),
            ),
            Tier(
                tier_id="t1",
                slices=(_slice("b", creates_paths=("b.py",)),),
                depends_on=("t0",),
            ),
        )
    )
    assert creates_paths(_spec(tier_dag=dag)) == ()


def test_creates_paths_flags_duplicate_create_across_tiers():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", creates_paths=("shared.py",)),),
                depends_on=(),
            ),
            Tier(
                tier_id="t1",
                slices=(_slice("b", creates_paths=("shared.py",)),),
                depends_on=("t0",),
            ),
        )
    )
    findings = creates_paths(_spec(tier_dag=dag))
    assert len(findings) == 1
    assert findings[0].severity == "High"


# ----------------------------------------------------------------------
# deletion_violates_goal
# ----------------------------------------------------------------------


def test_deletion_violates_goal_flags_outside_prefix():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="remove `docs/legacy.md` after migration",
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    s = _spec(
        tier_dag=dag,
        constraints=(BlastRadiusConstraint(no_edits_outside=("src/",)),),
    )
    findings = deletion_violates_goal(s)
    assert any("delete_outside_radius" in f.id for f in findings)
    assert findings[0].severity == "Critical"


def test_deletion_violates_goal_passes_when_within_prefix():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice("a", intent="remove `src/legacy.py` after migration"),
                ),
                depends_on=(),
            ),
        )
    )
    s = _spec(
        tier_dag=dag,
        constraints=(BlastRadiusConstraint(no_edits_outside=("src/",)),),
    )
    assert deletion_violates_goal(s) == ()


def _deletion_findings(
    intent: str,
    *,
    allowed: tuple[str, ...] = ("src/",),
    ctx: ProjectContext | None = None,
):
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", intent=intent),),
                depends_on=(),
            ),
        )
    )
    return deletion_violates_goal(
        _spec(
            tier_dag=dag,
            constraints=(BlastRadiusConstraint(no_edits_outside=allowed),),
            ctx=ctx,
        )
    )


@pytest.mark.parametrize(
    "intent",
    (
        (
            "Treat the deleted historical violation at "
            "ovsim-bridge/examples/combined-conveyor/main.cpp only as read-only "
            "negative evidence from the protected historical-donor tag and external "
            "object database—Git mode 100644, blob "
            "2d5304e097ab493209d66161c9b016b5506f68a3, SHA-256 "
            "10d1498b1671d7a25c7c7643bf18d3842f957063a73dd8d1b8d2e23916f38b9e, "
            "size 26682 bytes—and record and verify it without recreating, formatting, "
            "remediating, or including that deleted path in current mutation or format "
            "scope."
        ),
        "Do not delete `docs/a.cpp`.",
        "Never remove `docs/a.cpp`.",
        "Do not delete or remove `docs/a.cpp`.",
        "Avoid deleting or removing `docs/a.cpp`.",
        "Avoid accidentally deleting `docs/a.cpp`.",
        "Avoid ever deleting `docs/a.cpp`.",
        "Inspect without deleting `docs/a.cpp`.",
        "Not only must we not delete `docs/a.cpp`, but preserve it.",
        "Without hesitation, do not delete `docs/a.cpp`.",
        "Not surprisingly, do not delete `docs/a.cpp`.",
        "Do not delete and preserve `docs/a.cpp`.",
        "`docs/a.cpp` is preserved and not deleted.",
        "`docs/a.cpp` is preserved and must not be deleted.",
        "`docs/a.cpp` is preserved but isn't deleted.",
        "`docs/a.cpp` is preserved but is not deleted.",
        "`docs/a.cpp` is preserved but is already deleted.",
        "`docs/a.cpp` is not deleted but preserved.",
        "`docs/a.cpp` must not be deleted and must be preserved.",
        "`docs/a.cpp` shouldn't be deleted.",
        "`docs/a.cpp` isn't to be deleted.",
        "`docs/a.cpp` must not ever be deleted.",
        "`docs/a.cpp` must never again be deleted.",
        "Do not accidentally delete `docs/a.cpp`.",
        "We shouldn't accidentally delete `docs/a.cpp`.",
        "`docs/a.cpp` must not accidentally be deleted.",
        "`docs/a.cpp` shouldn't accidentally be deleted.",
        "`docs/a.cpp` cannot be removed.",
        "`docs/a.cpp` should never be deleted.",
        "`docs/a.cpp` is not being deleted.",
        "`docs/a.cpp` isn't being deleted.",
        "`docs/a.cpp` is never being deleted.",
        "`docs/a.cpp` was not being deleted.",
        "`docs/a.cpp` wasn't being deleted.",
        "`docs/a.cpp` should avoid being deleted.",
        "`docs/a.cpp` must avoid being deleted.",
        "Verify without deleting `docs/a.cpp`.",
        "Do not remove `src/allowed.cpp`; preserve `docs/a.cpp`.",
        "`docs/a.cpp` was deleted previously.",
        "Previously, this slice deleted `docs/a.cpp`.",
        "Previously, this slice ripped out `docs/a.cpp`.",
        "Previously, `docs/a.cpp` was ripped out.",
        "Previously, this slice tore out `docs/a.cpp`.",
        "Previously, this slice was deleting `docs/a.cpp`.",
        "Previously, this slice was removing `docs/a.cpp`.",
        "Previously, this slice was deleting and removing `docs/a.cpp`.",
        "Previously, this slice was deleting, removing, and purging `docs/a.cpp`.",
        "Previously, this slice was preserving and deleting `docs/a.cpp`.",
        "Previously, this slice was inspecting, documenting, and removing `docs/a.cpp`.",
        "Previously, `docs/a.cpp` was deleted.",
        "Previously, `docs/a.cpp` was being deleted.",
    ),
)
def test_deletion_violates_goal_ignores_tightly_non_authorizing_language(intent):
    assert _deletion_findings(intent) == ()


@pytest.mark.parametrize(
    "intent",
    (
        "Delete `docs/a.cpp`.",
        "Deleting `docs/a.cpp` is required.",
        "`docs/a.cpp` must be deleted.",
        "Do not inspect and delete `docs/a.cpp`.",
        "Do not retain but delete `docs/a.cpp`.",
        "Do not archive and delete `docs/a.cpp`.",
        "Do not frobnicate and delete `docs/a.cpp`.",
        "Do not fail to delete `docs/a.cpp`.",
        "Do not forget to remove `docs/a.cpp`.",
        "Never fail to delete `docs/a.cpp`.",
        "Without hesitation delete `docs/a.cpp`.",
        "Do not avoid deleting `docs/a.cpp`.",
        "Avoid not deleting `docs/a.cpp`.",
        "Avoid never removing `docs/a.cpp`.",
        "Avoid not accidentally deleting `docs/a.cpp`.",
        "Avoid accidentally not deleting `docs/a.cpp`.",
        "Avoid not accidentally being deleted `docs/a.cpp`.",
        "Do not not delete `docs/a.cpp`.",
        "Never not delete `docs/a.cpp`.",
        "We cannot not remove `docs/a.cpp`.",
        "`docs/a.cpp` must not not be deleted.",
        "`docs/a.cpp` cannot not be removed.",
        "`docs/a.cpp` should never not be deleted.",
        "`docs/a.cpp` is not not being deleted.",
        "`docs/a.cpp` cannot avoid being deleted.",
        "`docs/a.cpp` must not avoid being deleted.",
        "`docs/a.cpp` should never avoid being deleted.",
        "Do not continue without deleting `docs/a.cpp`.",
        "We cannot continue without deleting `docs/a.cpp`.",
        "Do not inspect without deleting `docs/a.cpp`.",
        "Never preserve without removing `docs/a.cpp`.",
        "Do not delete without preserving `docs/a.cpp`.",
        "Never remove without documenting `docs/a.cpp`.",
        "Avoid only deleting `docs/a.cpp`; also update callers.",
        "Avoid merely removing `docs/a.cpp`; migrate it too.",
        "Do not delete `docs/a.cpp` unless compatibility requires it.",
        "Never delete `docs/a.cpp` except when compatibility requires it.",
        "Unless compatibility requires it, do not delete `docs/a.cpp`.",
        "Except when compatibility requires it, never remove `docs/a.cpp`.",
        "`docs/a.cpp` must not be deleted unless compatibility requires it.",
        "Unless compatibility requires it, `docs/a.cpp` must not be deleted.",
        "`docs/a.cpp` should never be removed except when required.",
        "`docs/a.cpp` shouldn't be deleted unless required.",
        "`docs/a.cpp` isn't being deleted unless required.",
        "Unless required, `docs/a.cpp` shouldn't be deleted.",
        "`docs/a.cpp` isn't to be deleted unless required.",
        "`docs/a.cpp` must not ever be deleted unless required.",
        "`docs/a.cpp` must never again be deleted unless required.",
        "Not only delete `docs/a.cpp`, but also update `src/callers.cpp`.",
        "We must not only delete `docs/a.cpp` but also update `src/callers.cpp`.",
        "We must not only immediately delete `docs/a.cpp` but also document it.",
        "We must not merely immediately remove `docs/a.cpp` but also document it.",
        "`docs/a.cpp` isn't only deleted but also replaced.",
        "We shouldn't only delete `docs/a.cpp`; update callers too.",
        "`docs/a.cpp` must not only be deleted but also replaced.",
        "`docs/a.cpp` shouldn't only be deleted but also replaced.",
        "`docs/a.cpp` is not only to be deleted but also replaced.",
        "`docs/a.cpp` isn't only to be deleted but also replaced.",
        "`docs/a.cpp` is not only being deleted but also replaced.",
        "`docs/a.cpp` isn't only being deleted but also replaced.",
        "`docs/a.cpp` must not only immediately be deleted but also documented.",
        "`docs/a.cpp` must not only be immediately deleted but also documented.",
        "`docs/a.cpp` must not just be deleted but also migrated.",
        "`docs/a.cpp` shouldn't just be deleted but also migrated.",
        "`docs/a.cpp` must be deleted as legacy.",
        "`docs/a.cpp` must be removed as evidence.",
        "`docs/a.cpp` should be destroyed because it is legacy.",
        "`docs/a.cpp` must definitely be deleted.",
        "`docs/a.cpp` must already be deleted.",
        "Do not delete `src/allowed.cpp`; remove `docs/a.cpp`.",
        "This slice removed `docs/a.cpp`.",
    ),
)
def test_deletion_violates_goal_keeps_affirmative_and_ambiguous_language(intent):
    findings = _deletion_findings(intent)
    assert len(findings) == 1
    assert findings[0].id.endswith("docs/a.cpp")


@pytest.mark.parametrize("path", ("tools/remove.py", "tools/inspect.py"))
def test_deletion_violates_goal_ignores_action_words_inside_paths(path):
    findings = _deletion_findings(f"`{path}` must be deleted.")
    assert len(findings) == 1
    assert findings[0].id.endswith(path)


@pytest.mark.parametrize(
    "intent",
    (
        "Delete and document `docs/a.cpp`.",
        "Delete or preserve `docs/a.cpp` based on compatibility.",
        "Delete and then document `docs/a.cpp`.",
        "Delete, then document `docs/a.cpp`.",
        "Delete but preserve `docs/a.cpp`.",
        "Delete and carefully document `docs/a.cpp`.",
        "Delete and also document `docs/a.cpp`.",
        "Delete and only document `docs/a.cpp`.",
        "Delete and merely document `docs/a.cpp`.",
        "Delete and just document `docs/a.cpp`.",
        "Delete and do not preserve `docs/a.cpp`.",
        "Delete and do not ever preserve `docs/a.cpp`.",
        "Delete and never again remove `docs/a.cpp`.",
        "Delete and avoid not preserving `docs/a.cpp`.",
        "Delete but don't preserve `docs/a.cpp`.",
        "Delete but not preserve `docs/a.cpp`.",
        "Delete rather than preserve `docs/a.cpp`.",
        "Delete without preserving `docs/a.cpp`.",
        "Delete, preserve, and document `docs/a.cpp`.",
        "Delete, inspect, and document `docs/a.cpp`.",
        "Delete, then inspect `docs/a.cpp`.",
        "`docs/a.cpp` is to be deleted.",
        "`docs/a.cpp` needs to be deleted.",
        "`docs/a.cpp` has to be removed.",
        "`docs/a.cpp` is scheduled to be deleted.",
        "Previously, old behavior existed, and `docs/a.cpp` is to be deleted.",
        "Previously, old behavior existed, and `docs/a.cpp` is being deleted.",
        "Previously, old behavior existed, and `docs/a.cpp` is currently being deleted.",
        "Previously, old behavior existed, and `docs/a.cpp` is definitely to be deleted.",
        "Previously, old behavior existed, and `docs/a.cpp` is not only to be deleted but also migrated.",
        "Previously, old behavior existed, and `docs/a.cpp` shouldn't only be deleted but also migrated.",
        "`docs/a.cpp` is preserved and deleted.",
        "`docs/a.cpp` should be inspected, then deleted.",
        "`docs/a.cpp` should be preserved but must be deleted.",
        "`docs/a.cpp` is preserved but is deleted.",
        "`docs/a.cpp` is preserved but only deleted.",
        "`docs/a.cpp` is preserved but merely deleted.",
        "`docs/a.cpp` is preserved but just deleted.",
        "`docs/a.cpp` is preserved but must definitely be deleted.",
        "`docs/a.cpp` is preserved but is not deleted unless required.",
        "`docs/a.cpp` is preserved but isn't deleted unless required.",
        "`docs/a.cpp` is preserved but should never ever be deleted unless required.",
        "`docs/a.cpp` is not preserved but deleted.",
        "`docs/a.cpp` must not be preserved but deleted.",
        "`docs/a.cpp` should not be retained and must be removed.",
        "`docs/a.cpp` is not deleted but removed.",
        "Previously, this slice was deleting but removes `docs/a.cpp`.",
        "Previously, this slice was deleting and is removing `docs/a.cpp`.",
    ),
)
def test_deletion_violates_goal_keeps_shared_objects_and_common_passives(intent):
    findings = _deletion_findings(intent)
    assert len(findings) == 1
    assert findings[0].id.endswith("docs/a.cpp")


@pytest.mark.parametrize("quote", ('"', "'"))
def test_deletion_violates_goal_recognizes_ordinary_quoted_paths(quote):
    findings = _deletion_findings(f"Delete {quote}docs/a.cpp{quote}.")
    assert len(findings) == 1
    assert findings[0].id.endswith("docs/a.cpp")


def test_deletion_violates_goal_binds_coordinated_passive_subjects():
    findings = _deletion_findings(
        "`docs/a.cpp`, `src/allowed.cpp`, and `docs/b.cpp` must be deleted."
    )
    assert [finding.id.rsplit("_", 1)[-1] for finding in findings] == [
        "docs/a.cpp",
        "docs/b.cpp",
    ]


def test_deletion_violates_goal_does_not_cross_known_action_predicate():
    assert _deletion_findings("Delete stale behavior and inspect `docs/a.cpp`.") == ()


@pytest.mark.parametrize(
    "intent",
    (
        "Do not delete but remove `docs/a.cpp`.",
        "Never delete but purge `docs/a.cpp`.",
        "Avoid deleting but remove `docs/a.cpp`.",
    ),
)
def test_deletion_violates_goal_does_not_inherit_negation_across_contrast(intent):
    findings = _deletion_findings(intent)
    assert len(findings) == 1
    assert findings[0].id.endswith("docs/a.cpp")


@pytest.mark.parametrize("extension", ("html", "js", "rst"))
def test_deletion_violates_goal_shields_predicates_in_unsupported_quoted_paths(
    extension,
):
    assert (
        _deletion_findings(
            f"Preserve `docs/delete/file.{extension}` and `docs/a.cpp`."
        )
        == ()
    )


def test_deletion_violates_goal_indexes_coordinated_predicates_once(monkeypatch):
    module = importlib.import_module("crucible.rules.deletion_violates_goal")
    original = module._predicate_bridge_shares_object
    calls = 0

    def counting_bridge(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_predicate_bridge_shares_object", counting_bridge)
    predicate_count = 512
    intent = " and ".join(["delete"] * predicate_count) + " `docs/a.cpp`."

    findings = _deletion_findings(intent)

    assert len(findings) == 1
    assert findings[0].id.endswith("docs/a.cpp")
    assert calls == predicate_count - 1


def test_deletion_violates_goal_processes_shared_paths_once(monkeypatch):
    module = importlib.import_module("crucible.rules.deletion_violates_goal")
    original = module._verb_is_historical_for_path
    calls = 0

    def counting_history(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_verb_is_historical_for_path", counting_history)
    item_count = 128
    predicates = " and ".join(["delete"] * item_count)
    paths = ", ".join(f"docs/p{index}.cpp" for index in range(item_count))

    findings = _deletion_findings(f"{predicates} {paths}.")

    assert len(findings) == item_count
    assert calls <= item_count


def test_deletion_violates_goal_indexes_clause_exceptions_once(monkeypatch):
    module = importlib.import_module("crucible.rules.deletion_violates_goal")
    original = module._CONDITIONAL_EXCEPTION_RE

    class CountingPattern:
        calls = 0

        def search(self, *args, **kwargs):
            self.calls += 1
            return original.search(*args, **kwargs)

    counting = CountingPattern()
    monkeypatch.setattr(module, "_CONDITIONAL_EXCEPTION_RE", counting)
    item_count = 256
    segments = [
        f"delete docs/p{index}.cpp and inspect marker{index}"
        for index in range(item_count)
    ]

    findings = _deletion_findings(", before ".join(segments) + ".")

    assert len(findings) == item_count
    assert counting.calls == 1


def test_deletion_violates_goal_indexes_long_historical_predicate_run():
    predicate_count = 4096
    intent = "Previously, " + " and ".join(
        ["deleted"] * predicate_count
    ) + " `docs/a.cpp`."

    assert _deletion_findings(intent) == ()


def test_deletion_violates_goal_bounds_past_progressive_history_scans(monkeypatch):
    module = importlib.import_module("crucible.rules.deletion_violates_goal")
    original = module._PAST_PROGRESSIVE_AUXILIARY_RE

    class CountingPattern:
        calls = 0
        examined = 0

        def search(self, string, pos=0, endpos=None):
            self.calls += 1
            bounded_end = len(string) if endpos is None else endpos
            self.examined += bounded_end - pos
            return original.search(string, pos, bounded_end)

    counting = CountingPattern()
    monkeypatch.setattr(module, "_PAST_PROGRESSIVE_AUXILIARY_RE", counting)
    predicate_count = 512
    intent = (
        "Previously, "
        + " and ".join(["was deleting"] * predicate_count)
        + " `docs/a.cpp`."
    )

    assert _deletion_findings(intent) == ()
    assert counting.calls == predicate_count
    assert counting.examined <= len(intent) * 2


@pytest.mark.parametrize("coordinator", ("but", "while"))
def test_deletion_violates_goal_does_not_apply_leading_history_to_later_action(
    coordinator,
):
    findings = _deletion_findings(
        "Previously, the implementation preserved `src/allowed.cpp`, "
        f"{coordinator} this slice removed `docs/a.cpp`."
    )
    assert len(findings) == 1
    assert findings[0].id.endswith("docs/a.cpp")


@pytest.mark.parametrize(
    "earlier_event",
    (
        "the implementation archived `src/allowed.cpp`",
        "`src/allowed.cpp` existed",
        "no cleanup ran",
    ),
)
def test_deletion_violates_goal_bounds_history_without_known_earlier_action(
    earlier_event,
):
    findings = _deletion_findings(
        f"Previously, {earlier_event}, but this slice removed `docs/a.cpp`."
    )
    assert len(findings) == 1
    assert findings[0].id.endswith("docs/a.cpp")


def test_deletion_violates_goal_preserves_full_extension():
    findings = _deletion_findings("remove `docs/legacy.cpp` after migration")
    assert len(findings) == 1
    assert findings[0].id.endswith("docs/legacy.cpp")
    assert "docs/legacy.c'" not in findings[0].suggested_action


def test_deletion_violates_goal_rejects_partial_backup_extension_match():
    assert _deletion_findings("delete `docs/legacy.cpp.bak`") == ()


@pytest.mark.parametrize(
    "path",
    (
        "bridge/client.cc",
        "bridge/client.cxx",
        "assets/stage.usd",
        "assets/stage.usda",
        "package/conanfile.toml",
        "proto/world.proto",
        "cmake/options.cmake",
        "cmake/options.cmake.in",
    ),
)
def test_deletion_violates_goal_recognizes_bridge_extensions(path):
    findings = _deletion_findings(f"delete `{path}`")
    assert len(findings) == 1
    assert findings[0].id.endswith(path)


@pytest.mark.parametrize(
    "path",
    ("../src/allowed.cpp", "/src/allowed.cpp", "C:/src/allowed.cpp"),
)
def test_deletion_violates_goal_never_allows_unsafe_path_by_prefix(path):
    findings = _deletion_findings(f"delete `{path}`")
    assert len(findings) == 1


def test_deletion_violates_goal_normalizes_benign_relative_segments():
    assert _deletion_findings("delete `./src/allowed.cpp`") == ()


@pytest.mark.parametrize(
    ("path", "allowed"),
    (("demo/src/a.py", ("src/",)), ("src/a.py", ("demo/src/",))),
)
def test_deletion_violates_goal_matches_project_prefix_semantics(path, allowed):
    assert _deletion_findings(f"delete `{path}`", allowed=allowed) == ()


def test_deletion_violates_goal_noop_without_constraint():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(_slice("a", intent="remove `anything.py`"),),
                depends_on=(),
            ),
        )
    )
    assert deletion_violates_goal(_spec(tier_dag=dag)) == ()


# ----------------------------------------------------------------------
# ungrounded_citation
# ----------------------------------------------------------------------


def test_ungrounded_citation_passes_when_cited_in_scope():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="modify `foo/bar.py` per the rubric",
                        files_in_scope=("foo/bar.py",),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    assert ungrounded_citation(_spec(tier_dag=dag)) == ()


def test_ungrounded_citation_flags_uncited_path():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="modify `foo/bar.py` per the rubric",
                        files_in_scope=(),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    findings = ungrounded_citation(_spec(tier_dag=dag))
    assert len(findings) == 1
    assert "foo/bar.py" in findings[0].id


# ----------------------------------------------------------------------
# phantom_reference
# ----------------------------------------------------------------------


def test_phantom_reference_low_severity_when_unresolved():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="call `MysteryFunction` somewhere",
                        files_in_scope=(),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    findings = phantom_reference(_spec(tier_dag=dag))
    assert len(findings) == 1
    assert findings[0].severity == "Low"
    # category was renamed from "phantom_reference" → "phantom_symbol"
    # to disambiguate from cited_path_exists's filesystem-grounded
    # phantom_reference findings (Critical severity).
    assert findings[0].category == "phantom_symbol"


def test_phantom_reference_passes_when_file_stem_matches():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice(
                        "a",
                        intent="adapt `RequestContext` for new layout",
                        files_in_scope=("src/RequestContext.cpp",),
                    ),
                ),
                depends_on=(),
            ),
        )
    )
    assert phantom_reference(_spec(tier_dag=dag)) == ()


def test_phantom_reference_allowlists_builtins():
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice("a", intent="use a `dict` for caching `None` keys"),
                ),
                depends_on=(),
            ),
        )
    )
    assert phantom_reference(_spec(tier_dag=dag)) == ()


@pytest.mark.parametrize("symbol", ["StorageClient", "JobQueue", "ApiRouter"])
def test_project_local_symbols_remain_diagnosed(symbol):
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                slices=(
                    _slice("a", intent=f"update project-local `{symbol}`"),
                ),
                depends_on=(),
            ),
        )
    )

    findings = phantom_reference(_spec(tier_dag=dag, intent="Generic UI"))

    assert len(findings) == 1
    assert findings[0].evidence[0].excerpt == f"`{symbol}`"


# ----------------------------------------------------------------------
# Integration: rules are resolvable via crucible.rules.<rule_id>
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule_id",
    [
        "blast_radius",
        "creates_declaration",
        "creates_paths",
        "cycle",
        "deletion_violates_goal",
        "implementer_verify",
        "parallel_tier_conflicts",
        "phantom_reference",
        "runner_path_prefix",
        "ungrounded_citation",
    ],
)
def test_assay_validate_resolves_rule(rule_id):
    """Each ported rule is importable via assay.validate's default resolver."""
    spec = _spec()  # no tier_dag — every rule should no-op cleanly
    verdict = validate(
        spec,
        ruleset=ValidationRuleset(name="t", rule_ids=(rule_id,)),
    )
    assert verdict.passed is True
    assert verdict.findings == ()


def test_assay_validate_runs_all_ten_rules_in_one_call():
    """Full ruleset against a clean Spec should pass."""
    spec = _spec()
    ruleset = ValidationRuleset(
        name="all",
        rule_ids=(
            "blast_radius",
            "creates_declaration",
            "creates_paths",
            "cycle",
            "deletion_violates_goal",
            "implementer_verify",
            "parallel_tier_conflicts",
            "phantom_reference",
            "runner_path_prefix",
            "ungrounded_citation",
        ),
    )
    verdict = validate(spec, ruleset=ruleset)
    assert verdict.passed is True
