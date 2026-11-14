# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json

import pytest

from assay import (
    ImplementationDirective,
    ReadOnlySourceReference,
    RuleContext,
    Slice,
    Tier,
    TierDAG,
    hash_record,
)
from assay.corpus.store import Corpus
from crucible.constraints import (
    AuthoringInstructionConstraint,
    LanguagesConstraint,
)
from crucible.products.swebench.dataset import SwebenchInstance
from crucible.products.swebench.plan_normalization import (
    AuditedSwebenchPlanTransform,
    TestOnlyPlanError,
    normalize_swebench_plan,
)
from crucible.products.swebench.spec_synthesis import (
    SWEBENCH_AUTHORING_POLICY_OWNER,
    SWEBENCH_PLANNER_CONSTRAINTS,
    is_swebench_managed_authoring_constraint,
    reconcile_swebench_authoring_constraints,
    synthesize_starter_spec,
)
from crucible.rules.scope_covers_intent import scope_covers_intent
from crucible.rules.ungrounded_citation import ungrounded_citation


def _slice(
    id_: str,
    *paths: str,
    intent: str | None = None,
    requirement_refs: tuple[str, ...] = (),
) -> Slice:
    return Slice(
        id=id_,
        directive=ImplementationDirective(
            slice_id=id_,
            requirement_refs=requirement_refs,
            intent=intent or f"implement {id_}",
            files_in_scope=tuple(paths),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": len(paths)},
        ),
    )


def test_normalizer_keeps_source_and_drops_test_sibling() -> None:
    dag = TierDAG(tiers=(Tier(
        tier_id="t0",
        depends_on=(),
        slices=(
            _slice("source", "astropy/modeling/separable.py"),
            _slice("tests", "astropy/modeling/tests/test_separable.py"),
        ),
    ),))

    result = normalize_swebench_plan(dag)

    assert [item.id for item in result.dag.tiers[0].slices] == ["source"]
    assert result.dropped_slices == ("tests",)
    assert result.stripped_paths == (
        "astropy/modeling/tests/test_separable.py",
    )
    assert result.changed is True


def test_normalizer_strips_test_path_from_mixed_slice() -> None:
    dag = TierDAG(tiers=(Tier(
        tier_id="t0",
        depends_on=(),
        slices=(_slice(
            "mixed",
            "astropy/timeseries/core.py",
            "astropy/timeseries/tests/test_sampled.py",
        ),),
    ),))

    result = normalize_swebench_plan(dag)

    directive = result.dag.tiers[0].slices[0].directive
    assert directive.files_in_scope == ("astropy/timeseries/core.py",)
    assert result.dropped_slices == ()


def test_normalizer_rewires_dependencies_around_test_only_tier() -> None:
    dag = TierDAG(tiers=(
        Tier("source", (_slice("source", "pkg/core.py"),), ()),
        Tier(
            "test-only",
            (_slice("tests", "tests/test_core.py"),),
            ("source",),
        ),
        Tier("docs", (_slice("docs", "CHANGES.rst"),), ("test-only",)),
    ))

    result = normalize_swebench_plan(dag)

    assert [tier.tier_id for tier in result.dag.tiers] == ["source", "docs"]
    assert result.dag.tiers[1].depends_on == ("source",)
    assert result.dropped_tiers == ("test-only",)
    assert result.rewired_dependencies == (
        ("docs", ("test-only",), ("source",)),
    )


def test_normalizer_fails_closed_for_test_only_plan() -> None:
    dag = TierDAG(tiers=(Tier(
        "tests",
        (_slice("tests", "tests/test_only.py"),),
        (),
    ),))

    with pytest.raises(TestOnlyPlanError, match="no source-bearing slice"):
        normalize_swebench_plan(dag)


def test_normalizer_does_not_misclassify_contest_source() -> None:
    dag = TierDAG(tiers=(Tier(
        "source",
        (_slice("source", "pkg/contest.py"),),
        (),
    ),))

    result = normalize_swebench_plan(dag)

    assert result.dag == dag
    assert result.changed is False


def test_standalone_test_filename_is_rewritten_as_one_path_token() -> None:
    intent = (
        "Fix behavior in `pkg/contest.py`; use `test_widget.py` only as "
        "read-only verification evidence and leave contest.py unchanged."
    )
    dag = TierDAG(tiers=(Tier(
        "source",
        (_slice("source", "pkg/contest.py", intent=intent),),
        (),
    ),))

    result = normalize_swebench_plan(dag)
    normalized_intent = result.dag.tiers[0].slices[0].directive.intent

    assert "test_widget" not in normalized_intent
    assert "`.py`" not in normalized_intent
    assert "`pkg/contest.py`" in normalized_intent
    assert "contest.py unchanged" in normalized_intent
    assert [
        rewrite.original_token for rewrite in result.intent_citation_rewrites
    ] == ["`test_widget.py`"]
    assert result.intent_citation_rewrites[0].kind == "test_path"


def test_audited_transform_persists_raw_normalized_and_manifest(
    tmp_path,
) -> None:
    requirement = (
        "When tests/test_core.py exercises the public API, preserve the first "
        "value."
    )
    dag = TierDAG(tiers=(Tier(
        "t0",
        (
            _slice(
                "source",
                "pkg/core.py",
                requirement_refs=(requirement,),
            ),
            _slice("tests", "tests/test_core.py"),
        ),
        (),
    ),))
    corpus = Corpus(tmp_path / "corpus")
    transform = AuditedSwebenchPlanTransform(corpus)

    normalized = transform(dag)

    assert [item.id for item in normalized.tiers[0].slices] == ["source"]
    assert len(transform.artifact_refs) == 1
    manifest = json.loads(corpus.get_artifact(transform.artifact_refs[0]))
    assert manifest["schema_version"] == (
        "crucible.swebench-plan-normalization.v2"
    )
    assert manifest["dropped_slices"] == ["tests"]
    raw = json.loads(corpus.get_artifact(manifest["raw_candidate_ref"]))
    normalized_payload = json.loads(
        corpus.get_artifact(manifest["normalized_candidate_ref"])
    )
    raw_refs = raw["tiers"][0]["slices"][0]["directive"][
        "requirement_refs"
    ]
    normalized_refs = normalized_payload["tiers"][0]["slices"][0][
        "directive"
    ]["requirement_refs"]
    assert raw_refs == [requirement]
    assert normalized_refs == [requirement]
    assert normalized.tiers[0].slices[0].directive.requirement_refs == (
        requirement,
    )


@pytest.mark.parametrize("test_only", [False, True])
def test_audited_transform_retains_read_only_references(tmp_path, test_only):
    reference = ReadOnlySourceReference(
        "existing-test", "tests/reference.py", "sha256:" + "a" * 64, 2, 4,
    )
    slice_ = _slice("mixed", "pkg/core.py", "tests/test_core.py")
    slice_ = replace(slice_, directive=replace(
        slice_.directive,
        files_in_scope=("tests/test_core.py",) if test_only else (
            "pkg/core.py", "tests/test_core.py",
        ),
        read_only_references=(reference,),
    ))
    dag = TierDAG(tiers=(Tier("t0", (slice_,), ()),))
    corpus = Corpus(tmp_path / "corpus")
    transform = AuditedSwebenchPlanTransform(corpus)
    if test_only:
        with pytest.raises(TestOnlyPlanError):
            transform(dag)
    else:
        normalized = transform(dag)
        directive = normalized.tiers[0].slices[0].directive
        assert directive.read_only_references == (reference,)
        assert directive.files_in_scope == ("pkg/core.py",)
        assert directive.creates_paths == ()
    manifest = json.loads(corpus.get_artifact(transform.artifact_refs[0]))
    artifact_keys = ["raw_candidate_ref"]
    if not test_only:
        artifact_keys.append("normalized_candidate_ref")
    for key in artifact_keys:
        payload = json.loads(corpus.get_artifact(manifest[key]))
        directive_payload = payload["tiers"][0]["slices"][0]["directive"]
        assert directive_payload["read_only_references"] == [asdict(reference)]
        assert reference.path not in directive_payload["files_in_scope"]
        assert reference.path not in directive_payload["creates_paths"]


def test_empty_reference_audit_preserves_historical_artifact_digests(tmp_path):
    dag = TierDAG(tiers=(Tier(
        "t0", (_slice("mixed", "pkg/core.py", "tests/test_core.py"),), (),
    ),))
    corpus = Corpus(tmp_path / "corpus")
    transform = AuditedSwebenchPlanTransform(corpus)
    transform(dag)
    manifest = json.loads(corpus.get_artifact(transform.artifact_refs[0]))
    # These are the exact pre-reference audit JSON bytes, including indentation,
    # empty requirement_refs and the unchanged historical encoding choices.
    for key, expected in {
        "raw_candidate_ref": "c045820aff3ae105ad30184f030a928cfac9918b286998bd80d9b5febec9073a",
        "normalized_candidate_ref": "39715d69c63891697b5cf6d47e9c4a742a447f7e82eb793d40e064b16d2d7757",
    }.items():
        raw = corpus.get_artifact(manifest[key])
        assert hashlib.sha256(raw).hexdigest() == expected
        assert b"read_only_references" not in raw


def test_live_read_only_test_citations_are_neutralized_and_audited(
    tmp_path,
) -> None:
    live_intent = (
        "Fix `_cstack` in `astropy/modeling/separable.py` so that when the "
        "`right` operand is an already-computed coord matrix, the bottom-right "
        "block of `cright` is filled with the actual `right` array rather than "
        "with `1`. Change the `= 1` assignment to `= right` so nested compound "
        "models on the right side of `&` preserve their computed separability "
        "structure. The existing tests in "
        "`astropy/modeling/tests/test_separable.py` (e.g. `test_cstack` and "
        "the parametrized `test_separable`) are referenced only as read-only "
        "verification evidence to be run via verify_mode=tests; no edits to "
        "that test file are made by this slice."
    )
    dag = TierDAG(tiers=(Tier(
        "t0",
        (_slice(
            "fix_cstack_right_matrix",
            "astropy/modeling/separable.py",
            intent=live_intent,
        ),),
        (),
    ),))

    corpus = Corpus(tmp_path / "citation-audit")
    transform = AuditedSwebenchPlanTransform(corpus)
    normalized = transform(dag)
    directive = normalized.tiers[0].slices[0].directive

    assert directive.verify_mode == "tests"
    assert "Fix `_cstack` in `astropy/modeling/separable.py`" in directive.intent
    assert "Change the `= 1` assignment to `= right`" in directive.intent
    assert "astropy/modeling/tests/test_separable.py" not in directive.intent
    assert "test_cstack" not in directive.intent
    assert "test_separable" not in directive.intent
    assert "`existing repository test evidence`" not in directive.intent
    assert "existing repository test evidence" in directive.intent

    assert len(transform.artifact_refs) == 1
    manifest = json.loads(corpus.get_artifact(transform.artifact_refs[0]))
    rewrites = manifest["rewritten_intent_citations"]
    assert [item["original_token"] for item in rewrites] == [
        "`astropy/modeling/tests/test_separable.py`",
        "`test_cstack`",
        "`test_separable`",
    ]
    assert {item["kind"] for item in rewrites} == {
        "test_path",
        "test_symbol",
    }
    assert all(
        item["replacement"] == "existing repository test evidence"
        for item in rewrites
    )

    # The unchanged global rules must see the source path as covered and no
    # longer see either the real test path or its backtick citation.
    test_file = tmp_path / "astropy/modeling/tests/test_separable.py"
    source_file = tmp_path / "astropy/modeling/separable.py"
    test_file.parent.mkdir(parents=True)
    source_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("")
    source_file.write_text("")
    inst = SwebenchInstance(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        base_commit="0" * 40,
        patch="",
        test_patch="",
        problem_statement="Fix nested separability.",
        fail_to_pass=(),
        pass_to_pass=(),
    )
    spec = replace(synthesize_starter_spec(inst), tier_dag=normalized)
    assert scope_covers_intent(
        spec, RuleContext(workspace_root=tmp_path),
    ) == ()
    assert ungrounded_citation(spec) == ()


def test_audited_transform_records_test_only_rejection(tmp_path) -> None:
    dag = TierDAG(tiers=(Tier(
        "tests",
        (_slice("tests", "tests/test_only.py"),),
        (),
    ),))
    corpus = Corpus(tmp_path / "corpus")
    transform = AuditedSwebenchPlanTransform(corpus)

    with pytest.raises(TestOnlyPlanError):
        transform(dag)

    manifest = json.loads(corpus.get_artifact(transform.artifact_refs[0]))
    assert manifest["terminal_reason"] == "test_only_plan"
    assert corpus.get_artifact(manifest["raw_candidate_ref"])


def test_unowned_authoring_constraint_keeps_legacy_canonical_hash() -> None:
    constraint = AuthoringInstructionConstraint("Keep operator guidance.")

    assert hash_record(constraint) == hash_record({
        "instruction": "Keep operator guidance.",
        "kind": "authoring_instruction",
    })


def test_managed_policy_reconcile_preserves_unknown_constraints() -> None:
    operator = AuthoringInstructionConstraint(
        "Keep the operator's local release constraint.",
    )
    language = LanguagesConstraint(languages=("python",))
    legacy_read_only = replace(
        SWEBENCH_PLANNER_CONSTRAINTS[0],
        policy_owner=None,
        policy_rule_id=None,
        policy_revision=0,
    )
    legacy_minimal = replace(
        SWEBENCH_PLANNER_CONSTRAINTS[1],
        policy_owner=None,
        policy_rule_id=None,
        policy_revision=0,
    )
    same_text_but_operator_owned = replace(
        SWEBENCH_PLANNER_CONSTRAINTS[2],
        policy_owner="operator.local",
        policy_rule_id="cleanup",
        policy_revision=1,
    )
    existing = (
        operator,
        legacy_read_only,
        language,
        legacy_minimal,
        same_text_but_operator_owned,
    )

    reconciled = reconcile_swebench_authoring_constraints(existing)

    assert reconciled == (
        operator,
        *SWEBENCH_PLANNER_CONSTRAINTS,
        language,
        same_text_but_operator_owned,
    )
    assert not is_swebench_managed_authoring_constraint(operator)
    assert not is_swebench_managed_authoring_constraint(
        same_text_but_operator_owned,
    )
    assert all(
        is_swebench_managed_authoring_constraint(constraint)
        for constraint in SWEBENCH_PLANNER_CONSTRAINTS
    )


def test_managed_policy_recognizes_exact_old_read_only_wording() -> None:
    oldest_read_only_rule = AuthoringInstructionConstraint(
        "Repository test files are read-only authoring evidence: inspect and "
        "run existing tests through verify_mode=tests, but never include a "
        "test path in files_in_scope or creates_paths and never create a "
        "test-writing implementation slice."
    )
    near_match = AuthoringInstructionConstraint(
        oldest_read_only_rule.instruction + " Operator exception."
    )

    assert is_swebench_managed_authoring_constraint(oldest_read_only_rule)
    assert not is_swebench_managed_authoring_constraint(near_match)


def test_starter_spec_exposes_read_only_test_policy_to_planner(
    tmp_path,
) -> None:
    inst = SwebenchInstance(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="0" * 40,
        patch="",
        test_patch="",
        problem_statement="Fix the source behavior.",
        fail_to_pass=(),
        pass_to_pass=(),
    )

    spec = synthesize_starter_spec(inst)

    assert spec.constraints
    assert all(
        isinstance(item, AuthoringInstructionConstraint)
        for item in spec.constraints
    )
    assert all(
        item.policy_owner == SWEBENCH_AUTHORING_POLICY_OWNER
        for item in spec.constraints
    )
    rendered = "\n".join(str(item) for item in spec.constraints)
    assert "test files are read-only" in rendered.lower()
    assert "never include a test path" in rendered.lower()
    assert "do not name or quote test paths" in rendered.lower()
    assert "individual test symbols" in rendered.lower()
    assert "cleanup obligations evidence-bound" in rendered.lower()
    assert "public re-export" in rendered.lower()
    assert "omit speculative cleanup" in rendered.lower()
    corpus = Corpus(tmp_path / "roundtrip")
    corpus.put_spec(spec)
    assert corpus.latest_spec() == spec
