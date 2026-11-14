# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for the shared pre-review executable-plan candidate gate."""

from __future__ import annotations

from pathlib import Path

from assay import (
    ImplementationDirective,
    PlanPolicy,
    ProjectContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
)

from crucible.workflows._plan_validation import (
    build_pre_review_candidate_validator,
)


def _spec(dag: TierDAG) -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="demo",
            repo_url="https://example.invalid/demo.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="update the implementation",
        constraints=(),
        done_criteria=(),
        plan_policy=PlanPolicy(),
        depends_on=(),
        tier_dag=dag,
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _dag(
    *,
    intent: str,
    files_in_scope: tuple[str, ...],
    creates_paths: tuple[str, ...] = (),
) -> TierDAG:
    return TierDAG(tiers=(
        Tier(
            tier_id="t0",
            depends_on=(),
            slices=(
                Slice(
                    id="s0",
                    directive=ImplementationDirective(
                        slice_id="s0",
                        intent=intent,
                        requirement_refs=("update the implementation",),
                        files_in_scope=files_in_scope,
                        creates_paths=creates_paths,
                        verify_mode="tests",
                        blast_radius={"max_files": 1},
                    ),
                ),
            ),
        ),
    ))


def test_pre_review_candidate_gate_blocks_phantom_path(tmp_path: Path):
    dag = _dag(
        intent="update src/phantom.py",
        files_in_scope=("src/phantom.py",),
    )
    validator = build_pre_review_candidate_validator(
        spec=_spec(dag),
        project_root=tmp_path,
    )

    findings = validator(dag)

    assert [finding.id for finding in findings] == [
        "missing_files_in_scope__s0__src/phantom.py",
    ]
    assert all(
        finding.severity in ("Critical", "High")
        for finding in findings
    )


def test_pre_review_candidate_gate_allows_grounded_path(tmp_path: Path):
    source = tmp_path / "src" / "thing.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    dag = _dag(
        intent="update src/thing.py",
        files_in_scope=("src/thing.py",),
    )
    validator = build_pre_review_candidate_validator(
        spec=_spec(dag),
        project_root=tmp_path,
    )

    assert validator(dag) == ()


def test_pre_review_candidate_gate_blocks_project_root_escape(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    dag = _dag(
        intent="update the outside file",
        files_in_scope=("../" + outside.name,),
    )
    validator = build_pre_review_candidate_validator(
        spec=_spec(dag),
        project_root=tmp_path,
    )

    findings = validator(dag)

    assert any(
        finding.category == "scope_escape"
        and finding.severity == "Critical"
        for finding in findings
    )


def test_pre_review_candidate_gate_does_not_spend_repair_on_medium_nudge(
    tmp_path: Path,
):
    source = tmp_path / "src" / "thing.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    dag = _dag(
        intent="create src/thing.py",
        files_in_scope=("src/thing.py",),
    )
    validator = build_pre_review_candidate_validator(
        spec=_spec(dag),
        project_root=tmp_path,
    )

    # creates_declaration reports a Medium advisory finding, but only
    # Critical/High defects consume the bounded candidate correction.
    assert validator(dag) == ()
