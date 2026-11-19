# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.rules.scope_covers_intent``.

Catches explicit mis-slices where the intent prose names a real source
file outside the slice's files_in_scope / creates_paths.
"""

from __future__ import annotations

from dataclasses import replace
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
from crucible.rules.cited_path_exists import CitedPathExistsContext
from crucible.rules.scope_covers_intent import (
    _extract_referenced_paths,
    scope_covers_intent,
)


def _spec(*, intent="t", files_in_scope=(), creates_paths=()) -> Spec:
    directive = ImplementationDirective(
        slice_id="s",
        intent=intent,
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
    spec = replace(_spec(), tier_dag=None)
    assert scope_covers_intent(spec, RuleContext(workspace_root=tmp_path)) == ()


def test_ctx_none_returns_empty(tmp_path):
    spec = _spec(intent="see rendering/source/foo.cpp")
    assert scope_covers_intent(spec, None) == ()


def test_workspace_root_none_returns_empty(tmp_path):
    spec = _spec(intent="see rendering/source/foo.cpp")
    assert scope_covers_intent(spec, RuleContext(workspace_root=None)) == ()


# ----------------------------------------------------------------------
# Regex extraction
# ----------------------------------------------------------------------


def test_extract_multi_component_path():
    paths = _extract_referenced_paths("In rendering/source/sharedlibs/foo.cpp, do X.")
    assert paths == {"rendering/source/sharedlibs/foo.cpp"}


def test_extract_stops_at_trailing_punctuation():
    # Sentence punctuation after the extension is outside the \b boundary,
    # so it never enters the captured group — no normalization needed.
    paths = _extract_referenced_paths("modify a/b/foo.cpp.")
    assert "a/b/foo.cpp" in paths
    paths = _extract_referenced_paths("see a/b/foo.h, then a/b/bar.h;")
    assert paths == {"a/b/foo.h", "a/b/bar.h"}


def test_extract_ignores_bare_filename():
    # No slash → not a path token, just a filename mention.
    paths = _extract_referenced_paths("see the foo.h header")
    assert paths == set()


def test_extract_ignores_unknown_extension():
    # Markdown / yaml / toml are informational citations, not mutation sites.
    paths = _extract_referenced_paths(
        "see docs/notes.md and config/build.yaml and pyproject.toml"
    )
    assert paths == set()


def test_extract_handles_dot_in_directory_name():
    # rtx.hydra is a real directory style in kit.
    paths = _extract_referenced_paths(
        "edit rendering/source/sharedlibs/rtx.hydra/RtxHydraEngine.h"
    )
    assert paths == {"rendering/source/sharedlibs/rtx.hydra/RtxHydraEngine.h"}


def test_extract_handles_multiple_in_sentence():
    paths = _extract_referenced_paths(
        "wire src/foo/bar.cpp to include/foo/bar.h"
    )
    assert paths == {"src/foo/bar.cpp", "include/foo/bar.h"}


# ----------------------------------------------------------------------
# Rule behaviour
# ----------------------------------------------------------------------


def test_in_scope_path_no_finding(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.cpp").write_text("")
    spec = _spec(
        intent="In src/foo.cpp, add a method.",
        files_in_scope=["src/foo.cpp"],
    )
    assert scope_covers_intent(spec, RuleContext(workspace_root=tmp_path)) == ()


def test_creates_paths_branch_when_file_already_exists(tmp_path):
    # If a path is in creates_paths AND already exists on disk
    # (cited_path_exists is the rule that flags that as a separate
    # overwrite_collision; this rule just needs the creates_paths
    # membership to take precedence over the scope check).
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new.cpp").write_text("")
    spec = _spec(
        intent="creates src/new.cpp",
        creates_paths=["src/new.cpp"],
    )
    assert scope_covers_intent(spec, RuleContext(workspace_root=tmp_path)) == ()


def test_exists_short_circuit_when_path_absent_on_disk(tmp_path):
    # Even without creates_paths membership, a path named in intent
    # that doesn't exist on disk shouldn't fire here — phantom_reference
    # / cited_path_exists own the "hallucinated path" signal.
    spec = _spec(
        intent="planner intends to touch src/new.cpp",
        files_in_scope=[],
    )
    assert scope_covers_intent(spec, RuleContext(workspace_root=tmp_path)) == ()


def test_out_of_scope_real_path_fires_critical(tmp_path):
    """The canonical mis-slice case: intent names a real file not in scope."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.cpp").write_text("")
    (tmp_path / "src" / "bar.h").write_text("")
    spec = _spec(
        intent="In src/foo.cpp, declare X; in src/bar.h, add Y.",
        files_in_scope=["src/foo.cpp"],
    )
    findings = scope_covers_intent(spec, RuleContext(workspace_root=tmp_path))
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "Critical"
    assert f.category == "scope_omits_referenced_path"
    assert "src/bar.h" in f.id
    assert f.source == "deterministic"


def test_context_only_label_requires_removing_explicit_path_token(tmp_path):
    """Read-only prose cannot suppress an explicit out-of-scope path."""
    context_path = "src/adaptix/_internal/name_style.py"
    (tmp_path / "src" / "adaptix" / "_internal").mkdir(parents=True)
    (tmp_path / context_path).write_text("")
    spec = _spec(
        intent=(
            "Context only, NOT to be edited by this slice: the `NameStyle` "
            "enum is already declared at "
            "`src/adaptix/_internal/name_style.py:8` and must be imported, "
            "never modified."
        ),
        files_in_scope=[
            "src/adaptix/_internal/morphing/name_layout/name_mapping.py",
        ],
    )

    findings = scope_covers_intent(spec, RuleContext(workspace_root=tmp_path))

    assert len(findings) == 1
    action = findings[0].suggested_action
    assert f"add '{context_path}' to files_in_scope" in action
    assert f"remove the explicit '{context_path}' path token" in action
    assert "Merely labeling a named path 'context only'" in action

    corrected = replace(
        spec.tier_dag.tiers[0].slices[0].directive,
        intent=(
            "Import the existing `NameStyle` enum as read-only context; "
            "do not modify its defining module."
        ),
    )
    corrected_slice = Slice(id="s", directive=corrected)
    corrected_tier = replace(
        spec.tier_dag.tiers[0],
        slices=(corrected_slice,),
    )
    corrected_spec = replace(
        spec,
        tier_dag=TierDAG(tiers=(corrected_tier,)),
    )
    assert (
        scope_covers_intent(
            corrected_spec,
            RuleContext(workspace_root=tmp_path),
        )
        == ()
    )


def test_out_of_scope_phantom_path_no_finding(tmp_path):
    """A typo'd / hallucinated path is owned by phantom_reference, not this rule."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.cpp").write_text("")
    spec = _spec(
        intent="In src/foo.cpp, declare X; in src/does_not_exist.h, add Y.",
        files_in_scope=["src/foo.cpp"],
    )
    assert scope_covers_intent(spec, RuleContext(workspace_root=tmp_path)) == ()


def test_submodule_path_is_honored(tmp_path):
    """Paths in intent resolve under workspace_root / submodule_path."""
    sub = tmp_path / "kit"
    (sub / "rendering" / "src").mkdir(parents=True)
    (sub / "rendering" / "src" / "foo.cpp").write_text("")
    (sub / "rendering" / "src" / "bar.h").write_text("")
    spec = _spec(
        intent="In rendering/src/foo.cpp do X; in rendering/src/bar.h do Y.",
        files_in_scope=["rendering/src/foo.cpp"],
    )
    findings = scope_covers_intent(
        spec, RuleContext(workspace_root=tmp_path, submodule_path="kit"),
    )
    assert len(findings) == 1
    assert "rendering/src/bar.h" in findings[0].id


def test_multiple_slices_independently_flagged(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.cpp").write_text("")
    (tmp_path / "src" / "b.cpp").write_text("")
    (tmp_path / "src" / "shared.h").write_text("")

    d1 = ImplementationDirective(
        slice_id="s1",
        intent="In src/a.cpp, also touch src/shared.h.",
        files_in_scope=("src/a.cpp",),
        creates_paths=(),
        verify_mode="build_only",
        blast_radius={"max_files": 5},
    )
    d2 = ImplementationDirective(
        slice_id="s2",
        intent="In src/b.cpp, fully self-contained.",
        files_in_scope=("src/b.cpp",),
        creates_paths=(),
        verify_mode="build_only",
        blast_radius={"max_files": 5},
    )
    tier = Tier(
        tier_id="t0", depends_on=(),
        slices=(Slice(id="s1", directive=d1), Slice(id="s2", directive=d2)),
    )
    spec_template = _spec()
    spec = replace(spec_template, tier_dag=TierDAG(tiers=(tier,)))

    findings = scope_covers_intent(spec, RuleContext(workspace_root=tmp_path))
    assert len(findings) == 1
    f = findings[0]
    assert f.id == "scope_omits_referenced_path__s1__src/shared.h"


# ----------------------------------------------------------------------
# Post-implementation: a proven create of this plan is not pre-existing
# ----------------------------------------------------------------------


def _spec_naming_upstream_create(tmp_path) -> Spec:
    """``up/creator`` creates ``pkg/new.py``; ``t0/s`` names it in prose."""

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "seed.py").write_text("")
    (tmp_path / "pkg" / "new.py").write_text("")
    spec = _spec(
        intent="Then import from pkg/new.py in the seed module.",
        files_in_scope=["pkg/seed.py"],
    )
    creator = ImplementationDirective(
        slice_id="creator",
        intent="Create the new module.",
        files_in_scope=(),
        creates_paths=("pkg/new.py",),
        verify_mode="build_only",
        blast_radius={"max_files": 1},
    )
    upstream = Tier(
        tier_id="up",
        depends_on=(),
        slices=(Slice(id="creator", directive=creator),),
    )
    downstream = replace(spec.tier_dag.tiers[0], depends_on=("up",))
    return replace(spec, tier_dag=TierDAG(tiers=(upstream, downstream)))


def test_verified_plan_create_is_not_a_preexisting_reference(tmp_path):
    spec = _spec_naming_upstream_create(tmp_path)
    ctx = CitedPathExistsContext(
        workspace_root=tmp_path,
        successful_plan_bound_slices=frozenset({("up", "creator")}),
        verified_create_paths=frozenset({("up", "creator", "pkg/new.py")}),
    )

    # The file did not exist when the plan was accepted, so the reference
    # keeps the verdict it had then.
    assert scope_covers_intent(spec, ctx) == ()


@pytest.mark.parametrize(
    "ctx_fields",
    [
        # Planning carries no post-implementation proof at all.
        None,
        # The create is recorded but its slice is not successful/plan-bound.
        {
            "verified_create_paths": frozenset({
                ("up", "creator", "pkg/new.py"),
            }),
        },
        # The slice is successful but this path's postimage was not verified.
        {"successful_plan_bound_slices": frozenset({("up", "creator")})},
        # Another path of the same slice was verified.
        {
            "successful_plan_bound_slices": frozenset({("up", "creator")}),
            "verified_create_paths": frozenset({
                ("up", "creator", "pkg/other.py"),
            }),
        },
        # The proof names a slice that does not declare the path.
        {
            "successful_plan_bound_slices": frozenset({("t0", "s")}),
            "verified_create_paths": frozenset({("t0", "s", "pkg/new.py")}),
        },
        {
            "successful_plan_bound_slices": frozenset({("gone", "creator")}),
            "verified_create_paths": frozenset({
                ("gone", "creator", "pkg/new.py"),
            }),
        },
    ],
)
def test_existing_reference_without_proven_create_still_blocks(
    tmp_path, ctx_fields,
):
    spec = _spec_naming_upstream_create(tmp_path)
    ctx = (
        RuleContext(workspace_root=tmp_path)
        if ctx_fields is None
        else CitedPathExistsContext(workspace_root=tmp_path, **ctx_fields)
    )

    findings = scope_covers_intent(spec, ctx)

    assert [finding.id for finding in findings] == [
        "scope_omits_referenced_path__s__pkg/new.py",
    ]
    assert findings[0].severity == "Critical"
    assert findings[0].category == "scope_omits_referenced_path"
