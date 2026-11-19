# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for every bundled Crucible rubric."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from assay import (
    ContentHash,
    CriterionResult,
    Evidence,
    Finding,
    Proposal,
    ReviewCoverage,
    Rubric,
    Verdict,
)
from crucible.rubrics import (
    ArtifactFindingIdClosure,
    _contract_from_mapping,
    active_prior_frontier,
    available,
    canonicalize_verdict_contract,
    contract,
    contracts,
    lint_contracts,
    load,
    plan_review_contract_semantics,
    render_inventory_markdown,
    validate_verdict_contract,
)
from crucible.workflows._review import _context_prior_findings

RUBRICS_DIR = Path(__file__).parents[1] / "rubrics"


def test_inventory_is_complete_and_all_bundled_rubrics_are_active():
    assert len(available()) == 18
    assert {item.name for item in contracts()} == set(available())
    assert available(include_dormant=False) == available()
    assert all(item.status == "active" for item in contracts())


def test_manifest_linter_is_clean():
    assert lint_contracts() == ()


def test_manifest_parses_optional_contract_policies():
    expected = {
        "plan_judge": ("original_intent", "original intent"),
        "swebench_plan_critic": ("issue", "issue"),
        "deepswe_plan_critic": (
            "deepswe_directive", "DeepSWE directive",
        ),
    }

    for name, (scope, authority_label) in expected.items():
        rule = contract(name).reject_proof
        assert rule is not None
        assert (rule.scope, rule.authority_label) == (scope, authority_label)

    assert contract(
        "deepswe_plan_critic"
    ).finding_required_source_sets == (("artifact", "spec_intent"),)
    assert contract("plan_critic").criterion_results == "required"
    assert contract("plan_critic").requires_criterion_results
    assert {
        requirement.criterion_id
        for requirement in contract("plan_critic").criterion_source_requirements
    } == {"intent_coverage", "done_criteria_coverage"}
    assert contract("spec_clarity").reject_proof is None
    assert contract("spec_clarity").finding_required_source_sets == ()
    assert contract("impl_breaker").inconclusive_requires_incomplete_coverage
    assert contract(
        "swebench_impl_breaker"
    ).inconclusive_requires_incomplete_coverage
    assert not contract(
        "breach_attack_authority"
    ).inconclusive_requires_incomplete_coverage
    assert contract(
        "breach_attack_entailment"
    ).semantic_inconclusive_requires_response_repair
    assert not contract(
        "breach_attack_authority"
    ).semantic_inconclusive_requires_response_repair
    assert contract("breach_judge").severity_floor == "High"
    assert contract("impl_breaker").severity_floor == "High"
    assert contract("swebench_impl_breaker").severity_floor == "High"
    assert (
        contract("breach_judge").target_closure_severity_floor
        == "Medium"
    )
    assert "target_closure_severity_floor" not in (
        plan_review_contract_semantics(contract("plan_judge"))
    )
    projected_lower_floor = plan_review_contract_semantics(
        dataclasses.replace(
            contract("plan_judge"),
            target_closure_severity_floor="Medium",
        )
    )
    assert projected_lower_floor["target_closure_severity_floor"] == "Medium"


@pytest.mark.parametrize(
    ("rubric_name", "schema_version", "collection_path"),
    (
        (
            "breach_attack_authority",
            "crucible.breach-authority-admission.v1",
            ("generated_attacks",),
        ),
        (
            "breach_attack_entailment",
            "crucible.breach-attack-entailment.v1",
            ("attacks",),
        ),
    ),
)
def test_breach_rubrics_declare_artifact_finding_id_closure(
    rubric_name,
    schema_version,
    collection_path,
):
    closure = contract(rubric_name).artifact_finding_id_closure

    assert isinstance(closure, ArtifactFindingIdClosure)
    assert closure == ArtifactFindingIdClosure(
        schema_version=schema_version,
        collection_path=collection_path,
        id_field="attack_id",
    )

    prompt = load(rubric_name).system_prompt
    finding_id_path = "$." + ".".join(collection_path) + "[*].attack_id"
    assert finding_id_path in prompt
    assert "exactly one model-authored finding" in prompt
    assert "no other findings" in prompt


@pytest.mark.parametrize(
    "rubric_name",
    ("breach_attack_authority", "breach_attack_entailment"),
)
def test_breach_rubrics_declare_harness_attested_coverage(
    rubric_name,
):
    assert contract(rubric_name).coverage_completeness == "artifact_attested"

    prompt = load(rubric_name).system_prompt
    assert "coverage.complete=true" in prompt
    assert "coverage.omissions=[]" in prompt
    assert "Genuine semantic ambiguity" in prompt


def test_unconfigured_rubric_defaults_to_model_reported_coverage():
    assert contract("spec_clarity").coverage_completeness == "model_reported"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        (
            "reject_proof",
            {"scope": "issue"},
            "reject_proof must contain",
        ),
        (
            "finding_required_source_sets",
            "artifact",
            "finding_required_source_sets must be a list",
        ),
        (
            "inconclusive_requires_incomplete_coverage",
            "true",
            "inconclusive_requires_incomplete_coverage must be a boolean",
        ),
        (
            "semantic_inconclusive_requires_response_repair",
            "true",
            "semantic_inconclusive_requires_response_repair must be a boolean",
        ),
        (
            "target_closure_severity_floor",
            1,
            "target_closure_severity_floor must be one of",
        ),
        (
            "target_closure_severity_floor",
            "Critical",
            "target_closure_severity_floor cannot be stricter",
        ),
    ),
)
def test_manifest_parser_rejects_malformed_optional_contract_fields(
    field,
    value,
    error,
):
    raw = yaml.safe_load(
        (RUBRICS_DIR / "_manifest.yaml").read_text(encoding="utf-8")
    )["rubrics"]["plan_judge"]
    malformed = {**raw, field: value}

    with pytest.raises(ValueError, match=error):
        _contract_from_mapping("plan_judge", malformed)


def test_target_closure_floor_is_only_configurable_for_material_closure():
    raw = yaml.safe_load(
        (RUBRICS_DIR / "_manifest.yaml").read_text(encoding="utf-8")
    )["rubrics"]["spec_clarity"]

    with pytest.raises(
        ValueError,
        match=(
            "target_closure_severity_floor requires "
            "target_closure=all_material"
        ),
    ):
        _contract_from_mapping(
            "spec_clarity",
            {**raw, "target_closure_severity_floor": "Low"},
        )


@pytest.mark.parametrize(
    "closure",
    (
        {
            "collection_path": ["generated_attacks"],
            "id_field": "attack_id",
        },
        {
            "schema_version": "crucible.breach-authority-admission.v1",
            "collection_path": ["generated_attacks"],
            "id_field": "attack_id",
            "extra": "not allowed",
        },
        {
            "schema_version": "",
            "collection_path": ["generated_attacks"],
            "id_field": "attack_id",
        },
        {
            "schema_version": "crucible.breach-authority-admission.v1",
            "collection_path": "generated_attacks",
            "id_field": "attack_id",
        },
        {
            "schema_version": "crucible.breach-authority-admission.v1",
            "collection_path": [],
            "id_field": "attack_id",
        },
        {
            "schema_version": "crucible.breach-authority-admission.v1",
            "collection_path": ["generated_attacks", ""],
            "id_field": "attack_id",
        },
        {
            "schema_version": "crucible.breach-authority-admission.v1",
            "collection_path": ["generated_attacks"],
            "id_field": "",
        },
    ),
)
def test_manifest_parser_rejects_malformed_artifact_finding_id_closure(
    closure,
):
    raw = yaml.safe_load(
        (RUBRICS_DIR / "_manifest.yaml").read_text(encoding="utf-8")
    )["rubrics"]["breach_attack_authority"]
    malformed = {**raw, "artifact_finding_id_closure": closure}

    with pytest.raises(ValueError, match="artifact_finding_id_closure"):
        _contract_from_mapping("breach_attack_authority", malformed)


@pytest.mark.parametrize("value", ("", "complete", True, []))
def test_manifest_parser_rejects_invalid_coverage_completeness(
    value,
):
    raw = yaml.safe_load(
        (RUBRICS_DIR / "_manifest.yaml").read_text(encoding="utf-8")
    )["rubrics"]["breach_attack_authority"]
    malformed = {**raw, "coverage_completeness": value}

    with pytest.raises(ValueError, match="coverage_completeness"):
        _contract_from_mapping("breach_attack_authority", malformed)


_ARTIFACT_FINDING_ID_CLOSURE_CASES = (
    (
        "breach_attack_authority",
        "crucible.breach-authority-admission.v1",
        "generated_attacks",
        "attack_authority",
        "compatible",
    ),
    (
        "breach_attack_entailment",
        "crucible.breach-attack-entailment.v1",
        "attacks",
        "attack_entailment",
        "entailed",
    ),
)


def _artifact_closure_verdict(
    finding_ids,
    *,
    criterion_id,
    category,
    decision="pass",
    coverage_complete=True,
    coverage_omissions=(),
):
    findings = tuple(
        SimpleNamespace(
            id=finding_id,
            criterion_id=criterion_id,
            category=category,
            severity="Low",
            disposition="addressed",
            targets=(),
            evidence=(SimpleNamespace(
                source_id="artifact",
                kind="artifact_record",
                location=f"attack:{finding_id}",
                excerpt=finding_id,
            ),),
            proposal=None,
            attack_spec=None,
            source="llm",
        )
        for finding_id in finding_ids
    )
    return SimpleNamespace(
        decision=decision,
        coverage=SimpleNamespace(
            complete=coverage_complete,
            sources_reviewed=("artifact",),
            omissions=coverage_omissions,
        ),
        findings=findings,
        criterion_results=(),
    )


def _artifact_closure_payload(schema_version, collection, rows):
    return json.dumps({
        "schema_version": schema_version,
        collection: rows,
    })


@pytest.mark.parametrize(
    (
        "rubric_name",
        "schema_version",
        "collection",
        "criterion_id",
        "category",
    ),
    _ARTIFACT_FINDING_ID_CLOSURE_CASES,
)
def test_attested_artifact_rejects_false_incomplete_model_coverage(
    rubric_name,
    schema_version,
    collection,
    criterion_id,
    category,
):
    del category
    finding_ids = ("attack-a", "attack-b")
    verdict = _artifact_closure_verdict(
        finding_ids,
        criterion_id=criterion_id,
        category="inconclusive",
        decision="inconclusive",
        coverage_complete=False,
        coverage_omissions=("selected root clauses unavailable",),
    )
    for finding in verdict.findings:
        finding.disposition = "suspected"
    artifact = _artifact_closure_payload(
        schema_version,
        collection,
        tuple({"attack_id": finding_id} for finding_id in finding_ids),
    )

    errors = validate_verdict_contract(
        rubric_name,
        verdict,
        current_artifact=artifact,
    )

    rendered = "\n".join(errors)
    assert "harness-attested complete artifact requires" in rendered
    assert "complete=False" in rendered
    assert "selected root clauses unavailable" in rendered


@pytest.mark.parametrize(
    (
        "rubric_name",
        "schema_version",
        "collection",
        "criterion_id",
        "conclusive_category",
    ),
    _ARTIFACT_FINDING_ID_CLOSURE_CASES,
)
def test_attested_artifact_preserves_mixed_semantic_inconclusive_result(
    rubric_name,
    schema_version,
    collection,
    criterion_id,
    conclusive_category,
):
    finding_ids = ("attack-a", "attack-b")
    verdict = _artifact_closure_verdict(
        finding_ids,
        criterion_id=criterion_id,
        category=conclusive_category,
        decision="inconclusive",
    )
    verdict.findings[1].category = "inconclusive"
    verdict.findings[1].disposition = "suspected"
    artifact = _artifact_closure_payload(
        schema_version,
        collection,
        tuple({"attack_id": finding_id} for finding_id in finding_ids),
    )

    errors = validate_verdict_contract(
        rubric_name,
        verdict,
        current_artifact=artifact,
    )
    if rubric_name == "breach_attack_entailment":
        assert errors == (
            "breach_attack_entailment: semantic inconclusive findings "
            "require response repair; ids=['attack-b']",
        )
    else:
        assert errors == ()


def test_entailment_semantic_inconclusive_cannot_evade_repair_with_pass():
    verdict = _artifact_closure_verdict(
        ("attack-a",),
        criterion_id="attack_entailment",
        category="inconclusive",
        decision="pass",
    )
    verdict.findings[0].disposition = "suspected"
    artifact = _artifact_closure_payload(
        "crucible.breach-attack-entailment.v1",
        "attacks",
        ({"attack_id": "attack-a"},),
    )

    errors = validate_verdict_contract(
        "breach_attack_entailment",
        verdict,
        current_artifact=artifact,
    )

    assert any(
        "semantic inconclusive findings require response repair" in error
        for error in errors
    )


@pytest.mark.parametrize(
    (
        "rubric_name",
        "schema_version",
        "collection",
        "criterion_id",
        "category",
    ),
    _ARTIFACT_FINDING_ID_CLOSURE_CASES,
)
def test_artifact_finding_id_closure_accepts_every_expected_id_exactly_once(
    rubric_name,
    schema_version,
    collection,
    criterion_id,
    category,
):
    verdict = _artifact_closure_verdict(
        ("attack-a", "attack-b"),
        criterion_id=criterion_id,
        category=category,
    )
    artifact = _artifact_closure_payload(
        schema_version,
        collection,
        ({"attack_id": "attack-a"}, {"attack_id": "attack-b"}),
    )

    assert validate_verdict_contract(
        rubric_name,
        verdict,
        current_artifact=artifact,
    ) == ()


@pytest.mark.parametrize(
    (
        "rubric_name",
        "schema_version",
        "collection",
        "criterion_id",
        "category",
    ),
    _ARTIFACT_FINDING_ID_CLOSURE_CASES,
)
def test_artifact_finding_id_closure_rejects_missing_response_id(
    rubric_name,
    schema_version,
    collection,
    criterion_id,
    category,
):
    verdict = _artifact_closure_verdict(
        ("attack-a",),
        criterion_id=criterion_id,
        category=category,
    )
    artifact = _artifact_closure_payload(
        schema_version,
        collection,
        ({"attack_id": "attack-a"}, {"attack_id": "attack-b"}),
    )

    errors = validate_verdict_contract(
        rubric_name,
        verdict,
        current_artifact=artifact,
    )
    rendered = "\n".join(errors).lower()
    assert "missing" in rendered
    assert "attack-b" in rendered


@pytest.mark.parametrize(
    (
        "rubric_name",
        "schema_version",
        "collection",
        "criterion_id",
        "category",
    ),
    _ARTIFACT_FINDING_ID_CLOSURE_CASES,
)
def test_artifact_finding_id_closure_rejects_unexpected_response_id(
    rubric_name,
    schema_version,
    collection,
    criterion_id,
    category,
):
    verdict = _artifact_closure_verdict(
        ("attack-a", "attack-c"),
        criterion_id=criterion_id,
        category=category,
    )
    artifact = _artifact_closure_payload(
        schema_version,
        collection,
        ({"attack_id": "attack-a"},),
    )

    errors = validate_verdict_contract(
        rubric_name,
        verdict,
        current_artifact=artifact,
    )
    rendered = "\n".join(errors).lower()
    assert "unexpected" in rendered
    assert "attack-c" in rendered


@pytest.mark.parametrize(
    (
        "rubric_name",
        "schema_version",
        "collection",
        "criterion_id",
        "category",
    ),
    _ARTIFACT_FINDING_ID_CLOSURE_CASES,
)
def test_artifact_finding_id_closure_rejects_duplicate_response_id(
    rubric_name,
    schema_version,
    collection,
    criterion_id,
    category,
):
    verdict = _artifact_closure_verdict(
        ("attack-a", "attack-a"),
        criterion_id=criterion_id,
        category=category,
    )
    artifact = _artifact_closure_payload(
        schema_version,
        collection,
        ({"attack_id": "attack-a"},),
    )

    errors = validate_verdict_contract(
        rubric_name,
        verdict,
        current_artifact=artifact,
    )
    rendered = "\n".join(errors).lower()
    assert "duplicate" in rendered
    assert "attack-a" in rendered


@pytest.mark.parametrize(
    (
        "rubric_name",
        "schema_version",
        "collection",
        "criterion_id",
        "category",
    ),
    _ARTIFACT_FINDING_ID_CLOSURE_CASES,
)
def test_artifact_finding_id_closure_accepts_empty_collection_and_no_findings(
    rubric_name,
    schema_version,
    collection,
    criterion_id,
    category,
):
    verdict = _artifact_closure_verdict(
        (),
        criterion_id=criterion_id,
        category=category,
    )
    artifact = _artifact_closure_payload(schema_version, collection, ())

    assert validate_verdict_contract(
        rubric_name,
        verdict,
        current_artifact=artifact,
    ) == ()


@pytest.mark.parametrize(
    ("artifact_factory", "expected"),
    (
        (
            lambda schema, collection: _artifact_closure_payload(
                "crucible.wrong-schema.v1",
                collection,
                (),
            ),
            "schema_version",
        ),
        (
            lambda schema, collection: json.dumps({
                "schema_version": schema,
            }),
            "generated_attacks",
        ),
        (
            lambda schema, collection: _artifact_closure_payload(
                schema,
                collection,
                {"attack_id": "attack-a"},
            ),
            "generated_attacks",
        ),
        (
            lambda schema, collection: _artifact_closure_payload(
                schema,
                collection,
                ("not-a-row",),
            ),
            "object",
        ),
        (
            lambda schema, collection: _artifact_closure_payload(
                schema,
                collection,
                ({"not_attack_id": "attack-a"},),
            ),
            "invalid id",
        ),
        (
            lambda schema, collection: _artifact_closure_payload(
                schema,
                collection,
                ({"attack_id": ""},),
            ),
            "invalid id",
        ),
        (
            lambda schema, collection: _artifact_closure_payload(
                schema,
                collection,
                ({"attack_id": 7},),
            ),
            "invalid id",
        ),
        (
            lambda schema, collection: _artifact_closure_payload(
                schema,
                collection,
                ({"attack_id": "attack-a"}, {"attack_id": "attack-a"}),
            ),
            "duplicate",
        ),
    ),
)
def test_artifact_finding_id_closure_rejects_malformed_authority_artifact(
    artifact_factory,
    expected,
):
    rubric_name, schema_version, collection, criterion_id, category = (
        _ARTIFACT_FINDING_ID_CLOSURE_CASES[0]
    )
    verdict = _artifact_closure_verdict(
        (),
        criterion_id=criterion_id,
        category=category,
    )

    errors = validate_verdict_contract(
        rubric_name,
        verdict,
        current_artifact=artifact_factory(schema_version, collection),
    )

    assert expected in "\n".join(errors).lower()


def test_generic_rubric_and_plan_review_code_is_product_neutral():
    repository_root = Path(__file__).parents[1]
    generic_sources = (
        repository_root / "rubrics" / "__init__.py",
        repository_root / "workflows" / "architect.py",
        repository_root / "workflows" / "_plan_provenance.py",
        repository_root / "workflows" / "_review.py",
        repository_root / "workflows" / "plan.py",
    )

    for path in generic_sources:
        source = path.read_text(encoding="utf-8").lower()
        assert "deepswe" not in source, path
        assert "swebench" not in source, path
        assert "swe-bench" not in source, path


@pytest.mark.parametrize("name", available())
def test_every_bundled_rubric_parses_as_review_v2(name: str):
    rubric = load(name)
    meta = contract(name)

    assert isinstance(rubric, Rubric)
    assert rubric.name == name
    assert rubric.response_format == "review_v2"
    assert rubric.allowed_decisions == meta.allowed_decisions
    assert rubric.requires_complete_artifact is True
    assert "{artifact}" in rubric.user_template
    for context_key in meta.required_context:
        assert "{" + context_key + "}" in rubric.user_template


@pytest.mark.parametrize("name", available())
def test_every_loaded_prompt_has_one_shared_v2_contract(name: str):
    rubric = load(name)
    meta = contract(name)
    prompt = rubric.system_prompt

    assert prompt.count("=== MANDATORY ASSAY.REVIEW.V2 CONTRACT ===") == 1
    assert "schema_version`` must be ``assay.review.v2" in prompt
    assert "TRUST BOUNDARY" in prompt
    assert "untrusted DATA" in prompt
    assert '"passed"' not in prompt
    assert "suggested_action" not in prompt
    assert "kind=test_code" not in prompt
    for decision in meta.allowed_decisions:
        assert decision in prompt
    for criterion in meta.criteria:
        assert criterion in prompt
    for category in meta.categories:
        assert category in prompt


@pytest.mark.parametrize("name", available())
def test_shared_contract_requires_character_exact_evidence(name: str):
    prompt = " ".join(load(name).system_prompt.split())

    assert "one contiguous, character-for-character substring" in prompt
    assert "Do not reserialize or compact JSON" in prompt
    assert '``["a", "b"]``, ``["a","b"]`` is not verbatim' in prompt
    assert "after response JSON decoding" in prompt
    assert "shortest sufficient exact excerpt" in prompt
    assert "Never join, unwrap, or reflow" in prompt


@pytest.mark.parametrize(
    "name",
    [item.name for item in contracts() if item.output.proposal != "forbidden"],
)
def test_shared_contract_requires_complete_proposal_shape(name: str):
    prompt = " ".join(load(name).system_prompt.split())

    assert "must contain exactly the keys ``action`` and ``replacement``" in prompt
    assert "Never omit ``replacement``" in prompt
    assert '``"replacement": null`` (JSON null)' in prompt


@pytest.mark.parametrize("name", available())
def test_every_template_delimits_untrusted_data_without_code_fences(name: str):
    rubric = load(name)
    meta = contract(name)
    assert rubric.user_template.count("BEGIN_UNTRUSTED_ARTIFACT") == 1
    assert rubric.user_template.count("END_UNTRUSTED_ARTIFACT") == 1
    assert "```" not in rubric.user_template
    for key in meta.required_context:
        label = key.upper()
        assert rubric.user_template.count(
            f"BEGIN_UNTRUSTED_CONTEXT_{label}"
        ) == 1
        assert rubric.user_template.count(
            f"END_UNTRUSTED_CONTEXT_{label}"
        ) == 1


@pytest.mark.parametrize("name", available())
def test_yaml_has_no_legacy_or_duplicated_response_schema(name: str):
    raw = (RUBRICS_DIR / f"{name}.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    assert {
        "response_format",
        "requires_complete_artifact",
        "allowed_decisions",
        "severity_floor",
    }.isdisjoint(data)
    assert '"passed"' not in raw
    assert "suggested_action" not in raw
    assert "kind=test_code" not in raw
    assert "Schema:" not in raw


@pytest.mark.parametrize(
    "name", [item.name for item in contracts() if item.generated_output == "proposal"]
)
def test_proposal_rubrics_separate_generated_content_from_evidence(name: str):
    prompt = load(name).system_prompt
    assert 'non-null "proposal"' in prompt
    assert "A proposal is never evidence" in prompt


@pytest.mark.parametrize(
    "name", [item.name for item in contracts() if item.generated_output == "attack_spec"]
)
def test_attack_rubrics_emit_declarative_specs_not_code(name: str):
    prompt = load(name).system_prompt.lower()
    assert 'non-null "attack_spec"' in prompt
    assert "setup" in prompt and "stimulus" in prompt and "oracle" in prompt
    assert "never emit executable test source" in prompt
    assert "test_code" not in prompt


def test_breaker_prompt_separates_behavior_from_candidate_diagnosis():
    breaker = " ".join(load("impl_breaker").system_prompt.split()).lower()

    assert "expected externally observable result" in breaker
    assert "never put candidate diagnosis in the oracle" in breaker


def test_control_flow_uses_typed_decisions_not_magic_finding_ids():
    breach = load("breach_judge").system_prompt
    goal = load("goal_judge").system_prompt
    assert "typed decision" in breach.lower()
    assert "magic finding id" in breach.lower()
    assert "magic commands" in goal.lower()
    assert "judge_reject" not in breach + goal


def test_breach_judge_surviving_tests_are_nonblocking_findings():
    prompt = " ".join(load("breach_judge").system_prompt.split())

    assert "finding disposition MUST be ``addressed``" in prompt
    assert "never use ``confirmed`` or ``unfixable``" in prompt
    assert "pair that unresolved disposition with a non-pass decision" in prompt


def test_planning_contract_has_original_intent_and_complete_history():
    critic = load("plan_critic")
    judge = load("plan_judge")

    required_spec_context = {
        "{spec_intent}",
        "{spec_constraints}",
        "{spec_done_criteria}",
    }
    assert all(key in critic.user_template for key in required_spec_context)
    assert all(key in judge.user_template for key in required_spec_context)
    assert "{prior_critic_findings}" in judge.user_template
    assert "{prior_judge_findings}" in judge.user_template
    assert "do NOT prove semantic" in critic.system_prompt
    assert "Do not promise tier-level parallel execution" in critic.system_prompt


def test_plan_critic_does_not_promote_preserved_ambiguity_to_a_gap():
    prompt = " ".join(load("plan_critic").system_prompt.split())

    assert "preserves ambiguity about behavior that the root" in prompt
    assert "Do not demand a default, fallback, validation rule" in prompt
    assert "test expectation that the root does not require" in prompt
    assert "required in-domain behavior lacks an executable instruction" in prompt
    assert "cite the exact root obligation" in prompt


def test_swebench_plan_critic_reserves_reject_for_unbounded_failure():
    prompt = load("swebench_plan_critic").system_prompt
    flattened = " ".join(prompt.split())

    assert "Test-write and over-decomposition defects" in prompt
    assert "MUST use ``revise``" in prompt
    assert "Reserve ``reject``" in prompt
    assert "UNFIXABLE_PROOF:" in flattened
    assert '"bounded_plan_possible":false' in flattened


def test_plan_judge_reject_contract_requires_unfixable_blocker():
    prompt = load("plan_judge").system_prompt

    assert "decision=reject requires at least one High or Critical" in prompt
    assert "disposition=unfixable" in prompt
    assert "repairable confirmed or suspected finding requires decision=revise" in prompt


def test_swebench_plan_critic_reject_runtime_requires_unfixable_source_plan():
    finding = SimpleNamespace(
        id="test-edits",
        criterion_id="path_grounding",
        category="edits_test_files",
        severity="Critical",
        disposition="confirmed",
        targets=(),
        evidence=(SimpleNamespace(source_id="artifact", kind="current_state"),),
        proposal=SimpleNamespace(
            action="Remove the repairable test slice.", replacement=None,
        ),
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="reject",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", "context:spec_intent"),
        ),
        findings=(finding,),
    )

    errors = validate_verdict_contract("swebench_plan_critic", verdict)
    assert any("bounded repair requires decision='revise'" in error for error in errors)
    assert any("no bounded source plan" in error for error in errors)

    finding.disposition = "unfixable"
    finding.proposal.action = "Abandon this strategy."
    errors = validate_verdict_contract("swebench_plan_critic", verdict)
    assert any("no bounded source plan" in error for error in errors)

    finding.proposal.action = (
        "UNFIXABLE_PROOF: "
        '{"scope":"issue","bounded_plan_possible":false,'
        '"reason":"immutable API constraints conflict with required behavior"}'
    )
    assert validate_verdict_contract("swebench_plan_critic", verdict) == ()

    finding.proposal.action = (
        "UNFIXABLE_PROOF: "
        '{"scope":"issue","bounded_plan_possible":false,'
        '"reason":"it is false; a bounded source plan is possible"}'
    )
    assert any(
        "no bounded source plan" in error
        for error in validate_verdict_contract("swebench_plan_critic", verdict)
    )


def test_deepswe_plan_critic_reject_runtime_requires_unfixable_source_plan():
    finding = SimpleNamespace(
        id="broad-slice",
        criterion_id="responsibility_size",
        category="under_decomposed",
        severity="High",
        disposition="confirmed",
        targets=(),
        evidence=(
            SimpleNamespace(source_id="artifact", kind="current_state"),
            SimpleNamespace(
                source_id="context:spec_intent", kind="directive",
            ),
        ),
        proposal=SimpleNamespace(
            action="Split the independently landable responsibilities.",
            replacement=None,
        ),
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="reject",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", "context:spec_intent"),
        ),
        findings=(finding,),
    )

    errors = validate_verdict_contract("deepswe_plan_critic", verdict)
    assert any("bounded repair requires decision='revise'" in error for error in errors)
    assert any("no bounded source plan" in error for error in errors)

    finding.disposition = "unfixable"
    finding.proposal.action = (
        "UNFIXABLE_PROOF: "
        '{"scope":"deepswe_directive","bounded_plan_possible":false,'
        '"reason":"immutable source API constraints conflict with the directive"}'
    )
    assert validate_verdict_contract("deepswe_plan_critic", verdict) == ()

    finding.evidence = (
        SimpleNamespace(source_id="artifact", kind="current_state"),
    )
    assert any(
        "paired evidence" in error
        for error in validate_verdict_contract("deepswe_plan_critic", verdict)
    )


def test_plan_judge_active_intent_gap_requires_plan_and_root_evidence():
    spec_evidence = SimpleNamespace(
        source_id="context:spec_intent",
        kind="root_requirement",
    )
    artifact_evidence = SimpleNamespace(
        source_id="artifact",
        kind="tier_dag",
    )
    finding = SimpleNamespace(
        id="claimed-coverage-gap",
        criterion_id="intent_coverage",
        category="intent_gap",
        severity="High",
        disposition="confirmed",
        targets=(),
        evidence=(spec_evidence,),
        proposal=SimpleNamespace(
            action="Cover the missing behavior.",
            replacement=None,
        ),
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="revise",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=(
                "artifact",
                "context:spec_intent",
                "context:spec_constraints",
                "context:spec_done_criteria",
                "context:prior_critic_findings",
                "context:prior_judge_findings",
            ),
        ),
        findings=(finding,),
        criterion_results=(),
    )
    context = {
        "spec_intent": "Required behavior.",
        "spec_constraints": "[]",
        "spec_done_criteria": "Behavior is covered.",
        "prior_critic_findings": "[]",
        "prior_judge_findings": "[]",
    }
    validation_kwargs = {
        "available_source_ids": {
            "artifact",
            *(f"context:{key}" for key in context),
        },
        "supplied_context": context,
        "current_artifact": '{"tiers": []}',
    }

    errors = validate_verdict_contract(
        "plan_judge",
        verdict,
        **validation_kwargs,
    )
    assert any(
        "requires paired evidence from artifact + context:spec_intent"
        in error
        for error in errors
    )

    finding.evidence = (spec_evidence, artifact_evidence)
    errors = validate_verdict_contract(
        "plan_judge",
        verdict,
        **validation_kwargs,
    )
    assert not any("requires paired evidence" in error for error in errors)


def test_prior_sources_are_allowed_only_for_typed_target_closing_roles():
    closing = {item.name for item in contracts() if item.target_closure == "all_material"}
    assert closing
    for item in contracts():
        if item.name in closing:
            assert (
                "prior_output" in item.evidence_sources
                or {
                    "prior_critic_findings",
                    "prior_judge_findings",
                } <= set(item.evidence_sources)
            )
            if "prior_output" in item.evidence_sources:
                assert (
                    "prior:<role>:<round>:sha256-<verdict-digest>:<finding-id>"
                    in load(item.name).system_prompt
                )
        else:
            assert "prior_output" not in item.evidence_sources

    finding = SimpleNamespace(
        id="f1",
        criterion_id="intent_clarity",
        category="ambiguity",
        severity="High",
        disposition="addressed",
        evidence=(SimpleNamespace(source_id="prior:red:red-1"),),
        proposal=SimpleNamespace(action="Clarify", replacement=None),
        attack_spec=None,
        targets=(),
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", "prior:red:red-1"),
        ),
        findings=(finding,),
    )
    errors = validate_verdict_contract(
        "spec_clarity",
        verdict,
        available_source_ids={"artifact", "prior:red:red-1"},
    )
    assert any("undeclared source_id" in error for error in errors)


def test_goal_blue_proposes_one_complete_machine_applicable_candidate():
    prompt = load("goal_blue").system_prompt
    assert "Exactly ONE finding must have a non-null" in prompt
    assert 'Exactly one finding must use a non-null "proposal"' in prompt
    assert "COMPLETE canonical goal payload" in prompt
    assert '"intent"' in prompt
    assert '"constraints"' in prompt
    assert '"done_criteria"' in prompt


def test_decision_floors_keep_documented_revision_outcomes_reachable():
    assert load("deepswe_plan_critic").severity_floor == "Medium"
    assert load("diff_scope").severity_floor == "Medium"
    assert load("plan_judge").severity_floor == "High"
    assert load("tier_completeness").severity_floor == "Medium"


def test_tier_completeness_reviews_allocated_increment_not_broad_root_clause():
    prompt = " ".join(load("tier_completeness").system_prompt.split())

    assert "may be broader" in prompt
    assert "current-slice allocation" in prompt
    assert "do not demand the downstream behavior" in prompt
    assert "Full root coverage across later tiers" in prompt
    assert "verification-only" in prompt
    assert "crucible.tier-verification-evidence.v1" in prompt
    assert "gate_status`` is ``passed" in prompt
    assert "never substitutes for required edits" in prompt


def test_specialized_planners_keep_required_failure_taxonomies():
    expected = {
        "swebench_plan_critic": {
            "over_decomposed",
            "under_decomposed",
            "over_tiered",
            "hallucinated_path",
            "edits_test_files",
            "ungrounded_intent",
            "parallel_conflict",
        },
        "deepswe_plan_critic": {
            "under_decomposed",
            "scope_too_broad",
            "missing_sequencing",
            "incomplete_increment",
            "parallel_conflict",
            "ignored_path",
        },
        "swebench_impl_breaker": {
            "unstated_boundary",
            "parameter_ambiguity",
            "broken_meaning",
            "invariant_drift",
            "test_contract_drift",
            "incomplete_scope",
        },
    }
    for name, required in expected.items():
        assert required <= set(contract(name).categories)
        prompt = load(name).system_prompt
        for category in required:
            assert category in prompt


@pytest.mark.parametrize(
    "name", ("swebench_plan_critic", "deepswe_plan_critic")
)
def test_test_file_edit_rule_matches_manifest_criterion(name: str):
    meta = contract(name)
    mapped_criteria = {
        criterion
        for criterion, categories in meta.criterion_categories
        if "edits_test_files" in categories
    }

    assert mapped_criteria == {"path_grounding"}
    assert "`edits_test_files` finding with `criterion_id=path_grounding`" in (
        " ".join(load(name).system_prompt.split())
    )


def test_deepswe_plan_critic_treats_test_named_repo_paths_as_canonical():
    prompt = load("deepswe_plan_critic").system_prompt
    compact = " ".join(prompt.split())

    assert "grades the complete repository patch" in prompt
    assert "bandit/core/test_set.py" in prompt
    assert "mobly/base_test.py" in prompt
    assert "lib/runners/browser_test_runner.js" in prompt
    assert "NEVER reject, remove" in prompt
    assert "hidden benchmark mount (`/tests` or `/solution`)" in prompt
    assert "Do not infer pipeline ownership from a path pattern" in compact


def test_deepswe_plan_critic_uses_independent_coherent_increment_granularity():
    prompt = " ".join(load("deepswe_plan_critic").system_prompt.split())

    assert "independently implementable and independently verifiable" in prompt
    assert "leaves the repository coherent" in prompt
    assert "not an automatic split rule" in prompt
    assert "tightly coupled validation/lookup wiring may be one increment" in prompt
    assert "UNFIXABLE_PROOF:" in prompt
    assert '"scope":"deepswe_directive"' in prompt


def _deepswe_critic_rule_body() -> str:
    """Return the authored DeepSWE critic rules, whitespace-normalized.

    The generated response contract lists every manifest category, so a rule
    pin on the complete prompt could pass without the rule that explains it.
    """

    body, marker, _ = load("deepswe_plan_critic").system_prompt.partition(
        "=== MANDATORY ASSAY.REVIEW.V2 CONTRACT ==="
    )
    assert marker
    return " ".join(body.split())


def _deepswe_critic_revise_verdict(
    *, criterion_id: str, category: str
) -> SimpleNamespace:
    finding = SimpleNamespace(
        id="new-file-at-ignored-path",
        criterion_id=criterion_id,
        category=category,
        severity="High",
        disposition="confirmed",
        targets=(),
        evidence=(
            SimpleNamespace(source_id="artifact", kind="current_state"),
            SimpleNamespace(
                source_id="context:spec_intent", kind="directive",
            ),
        ),
        proposal=SimpleNamespace(
            action="Create the file at a path the repository tracks.",
            replacement=None,
        ),
        attack_spec=None,
    )
    return SimpleNamespace(
        decision="revise",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", "context:spec_intent"),
        ),
        findings=(finding,),
    )


def test_deepswe_plan_critic_keeps_one_contract_in_one_slice():
    body = _deepswe_critic_rule_body()

    assert "Contract atomicity" in body
    for shape in (
        "Mirrored or forwarded signatures",
        "An interface and its implementers",
        "A public API and its registry",
        "A signature and its call sites",
        "A required field and its construction sites",
    ):
        assert shape in body
    assert "implements and gates one slice at a time" in body
    assert "sibling slices of one tier OR slices in different tiers" in body
    assert (
        "High `incomplete_increment` finding with "
        "`criterion_id=responsibility_size`"
    ) in body
    assert "MERGE" in body
    assert "NEVER to split further" in body
    assert "do not file `under_decomposed` or `scope_too_broad`" in body
    assert "OPTIONAL field with an inline default" in body
    # A mirrored signature is one contract even when the parameter is
    # optional: the project's signature-consistency test still needs both.
    assert (
        "a new optional parameter on a signature that nothing mirrors, "
        "re-exposes, or forwards"
    ) in body
    assert (
        "shape (a) applies to a mirrored signature even when the new "
        "parameter is optional"
    ) in body
    assert "does not license bundling" in body
    # A Critic without repository access cannot assume that the other side of
    # a contract exists, and an unanswerable finding would reject the plan.
    assert (
        "file that finding only when the root directive, the TierDAG, or the "
        "captured read-only source evidence shows that the other side exists"
    ) in body
    assert (
        "A slice whose intent names the paired sites inside its write scope, "
        "or states that none exist outside it, satisfies this rule"
    ) in body
    # Every slice is gated on its own; no gate waits for a tier's siblings.
    assert "combined change" not in body
    assert "once every slice of a tier" not in body
    assert "incomplete_increment" in dict(
        contract("deepswe_plan_critic").criterion_categories
    )["responsibility_size"]


def test_deepswe_plan_critic_does_not_tier_the_parts_of_one_contract():
    body = _deepswe_critic_rule_body()
    template = " ".join(load("deepswe_plan_critic").user_template.split())

    assert (
        "a tier boundary is not a way to separate the parts of one contract"
    ) in body
    assert (
        "A new type or interface with no existing implementer or caller is "
        "such a prerequisite, not a contract part"
    ) in body
    assert "do not file `missing_sequencing`" in body
    assert "every contract pairing in rule 8" in template


def test_deepswe_ignored_path_rule_matches_manifest_criterion():
    mapped_criteria = {
        criterion
        for criterion, categories in contract(
            "deepswe_plan_critic"
        ).criterion_categories
        if "ignored_path" in categories
    }

    assert mapped_criteria == {"path_grounding"}
    assert "`ignored_path` finding with `criterion_id=path_grounding`" in (
        _deepswe_critic_rule_body()
    )


def test_deepswe_plan_critic_flags_new_files_at_ignored_paths():
    body = _deepswe_critic_rule_body()
    template = " ".join(load("deepswe_plan_critic").user_template.split())
    _, heading, tail = body.partition("New files at ignored paths")
    assert heading
    rule, scale_heading, tail = tail.partition("Severity scale")
    assert scale_heading
    scale, closing, _ = tail.partition("Be evidence-grounded")
    assert closing

    for phrase in (
        "`git add -A`",
        "`.gitignore`",
        "`creates_paths`",
        "never captured",
        "High `ignored_path` finding",
    ):
        assert phrase in rule

    # A supplied pattern proves a match. Only the complete root file shows
    # that nothing matches; otherwise the path heuristics still apply.
    supplied, unsupplied_label, rest = rule.partition(
        "Ignore rules not supplied"
    )
    assert unsupplied_label
    assert (
        "captured read-only source evidence contains the repository's ignore "
        "rules"
    ) in supplied
    assert "judge every `creates_paths` entry against them" in supplied
    assert (
        "A supplied pattern that matches an entry proves the defect"
    ) in supplied
    assert "including `build/` or `dist/`" in supplied
    assert (
        "show that NO pattern matches only when they include the complete "
        "root `.gitignore`: its reference starts at line 1 and the citing "
        "slice's intent states that the reference is the whole file"
    ) in supplied
    assert (
        "File nothing for an entry that such complete rules do not match"
    ) in supplied
    assert (
        "judge an entry that no supplied pattern matches by the second case"
    ) in supplied
    assert "File nothing when the supplied rules show no match" not in body
    assert (
        "Unless the captured read-only source evidence includes them, you are "
        "not shown the repository's ignore rules"
    ) in rest
    assert "a partial excerpt shows only some of them" in rest

    listed, exemption, exempt = rest.partition("Do NOT flag")
    assert exemption
    for pattern in (
        "`__pycache__/`",
        "`node_modules/`",
        "`.venv/`",
        "`*.pyc`",
        "`.env`",
    ):
        assert pattern in listed
    # Generated files are routinely tracked; only "ignored" is a trigger.
    assert "the slice itself describes the location as ignored." in listed
    assert "ignored or generated" not in body
    assert (
        "cite the complete root `.gitignore` and any `.gitignore` in a "
        "parent directory of the path as read-only references"
    ) in listed
    assert "governing `.gitignore` lines" not in body
    for directory in ("`build/`", "`dist/`", "`target/`"):
        assert directory in exempt
    assert "the root directive itself names that exact path" in exempt
    assert "hold existing repository files" in exempt
    assert "Never apply this rule to `files_in_scope`" in exempt

    assert "Medium `ignored_path`" not in body
    assert "never reach verification" in scale
    assert "rule 10" in template


def test_deepswe_plan_critic_ignored_path_finding_satisfies_runtime_contract():
    verdict = _deepswe_critic_revise_verdict(
        criterion_id="path_grounding", category="ignored_path",
    )

    assert validate_verdict_contract("deepswe_plan_critic", verdict) == ()


def test_deepswe_plan_critic_rejects_ignored_path_outside_path_grounding():
    verdict = _deepswe_critic_revise_verdict(
        criterion_id="sequencing", category="ignored_path",
    )

    errors = validate_verdict_contract("deepswe_plan_critic", verdict)

    assert any("not allowed for criterion" in error for error in errors)
    assert not any("undeclared category" in error for error in errors)


def test_deepswe_plan_review_prompts_are_compact_and_delta_focused():
    critic = " ".join(load("deepswe_plan_critic").system_prompt.split())
    judge = " ".join(load("plan_judge").system_prompt.split())

    assert "compact and delta-focused" in critic
    assert "every required check on the complete current TierDAG" in critic
    assert "one canonical finding per distinct current defect" in critic
    assert "smallest concrete field or slice repair" in critic
    assert "Do not add narrative outside the typed verdict" in critic

    assert "active frontier as the review delta" in judge
    assert "do not summarize or replay closed history" in judge
    assert "exactly one disposition for each active target" in judge
    assert "Delta focus never waives the complete independent audit" in judge
    assert "Do not add narrative outside the typed verdict" in judge


def test_plan_review_rubrics_enforce_bidirectional_semantic_fidelity():
    for name in ("deepswe_plan_critic", "plan_critic", "plan_judge"):
        prompt = " ".join(load(name).system_prompt.split())

        assert "Root-to-plan" in prompt
        assert "Plan-to-root" in prompt
        assert "every externally observable" in prompt
        assert "entailed by" in prompt
        assert "cannot add, resolve, strengthen, weaken, or override" in prompt
        assert "requirement_refs" in prompt
        assert "does not by itself prove entailment" in prompt
        assert "widened" in prompt
        assert "narrowed" in prompt
        assert "composition" in prompt
        assert "precedence" in prompt
        assert "ordering" in prompt
        assert "exclusions" in prompt
        assert "preserve that ambiguity verbatim" in prompt
        assert "decision=revise" in prompt

    plan_judge_prompt = " ".join(load("plan_judge").system_prompt.split())
    assert "paired evidence from both the candidate" in plan_judge_prompt
    assert (
        "proves the obligation, not its alleged omission"
        in plan_judge_prompt
    )


def _plan_critic_contract_verdict(
    *,
    criterion_id: str = "intent_coverage",
    category: str = "intent_gap",
    finding_sources: tuple[str, ...],
) -> SimpleNamespace:
    def evidence(source_id: str) -> SimpleNamespace:
        return SimpleNamespace(source_id=source_id, kind="source_quote")

    results = []
    for index, declared in enumerate(contract("plan_critic").criteria):
        if declared == "intent_coverage":
            sources = ("artifact", "context:spec_constraints")
            outcome = "violated" if criterion_id == declared else "satisfied"
        elif declared == "done_criteria_coverage":
            sources = ("artifact", "context:spec_done_criteria")
            outcome = "violated" if criterion_id == declared else "satisfied"
        else:
            sources = ("artifact",)
            outcome = "satisfied"
        results.append(SimpleNamespace(
            criterion_id=declared,
            outcome=outcome,
            reasoning="Evidence-backed criterion result.",
            evidence=tuple(evidence(source) for source in sources),
            headline=index == 0,
        ))
    return SimpleNamespace(
        decision="revise",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=(
                "artifact",
                "context:spec_intent",
                "context:spec_constraints",
                "context:spec_done_criteria",
            ),
        ),
        findings=(SimpleNamespace(
            id="semantic-gap",
            criterion_id=criterion_id,
            category=category,
            severity="High",
            disposition="confirmed",
            targets=(),
            evidence=tuple(evidence(source) for source in finding_sources),
            proposal=SimpleNamespace(action="Repair the semantic gap."),
            attack_spec=None,
        ),),
        criterion_results=tuple(results),
    )


def test_plan_critic_requires_complete_structured_criterion_results():
    rubric = load("plan_critic")
    assert rubric.required_criteria == contract("plan_critic").criteria
    verdict = _plan_critic_contract_verdict(
        finding_sources=("artifact", "context:spec_intent"),
    )
    verdict.criterion_results = ()

    errors = validate_verdict_contract("plan_critic", verdict)

    assert any("must cover every criterion exactly once" in error for error in errors)
    assert any("exactly one headline" in error for error in errors)


def test_advisory_contract_rejects_forbidden_criterion_results():
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact",),
        ),
        findings=(),
        criterion_results=(SimpleNamespace(
            criterion_id="intent_clarity",
            outcome="satisfied",
            reasoning="Unexpected direct-consumer result.",
            evidence=(SimpleNamespace(source_id="artifact"),),
            headline=True,
        ),),
    )

    assert validate_verdict_contract("spec_clarity", verdict) == (
        "spec_clarity: criterion_results are forbidden",
    )


def test_plan_critic_semantic_findings_require_candidate_and_root_evidence():
    root_only = _plan_critic_contract_verdict(
        finding_sources=("context:spec_intent",),
    )
    errors = validate_verdict_contract("plan_critic", root_only)
    assert any("finding[0] requires evidence from artifact" in error for error in errors)

    artifact_only = _plan_critic_contract_verdict(
        finding_sources=("artifact",),
    )
    errors = validate_verdict_contract("plan_critic", artifact_only)
    assert any("requires evidence from at least one of" in error for error in errors)

    paired = _plan_critic_contract_verdict(
        finding_sources=("artifact", "context:spec_constraints"),
    )
    errors = validate_verdict_contract("plan_critic", paired)
    assert not any("finding[0] requires evidence" in error for error in errors)


def test_plan_critic_done_criteria_gap_requires_done_criteria_evidence():
    wrong_root = _plan_critic_contract_verdict(
        criterion_id="done_criteria_coverage",
        category="done_criteria_gap",
        finding_sources=("artifact", "context:spec_intent"),
    )
    errors = validate_verdict_contract("plan_critic", wrong_root)
    assert any(
        "finding[0] requires evidence from artifact + context:spec_done_criteria"
        in error
        for error in errors
    )


def test_plan_review_semantic_fidelity_rules_are_product_neutral():
    prompts = "\n".join(
        load(name).system_prompt
        for name in ("deepswe_plan_critic", "plan_critic", "plan_judge")
    ).lower()

    assert "adaptix" not in prompts
    assert "name_mapping" not in prompts
    assert "alias_resolution" not in prompts
    assert "frt-01a" not in prompts


def test_inventory_markdown_is_deterministic_and_complete():
    rendered = render_inventory_markdown()
    assert rendered.startswith(
        "| Rubric | Family | Role | Status | Artifact | Floor |"
    )
    assert rendered.count("\n| `") == 18
    for name in available():
        assert f"| `{name}` |" in rendered


def test_runtime_contract_validator_accepts_declared_vocabulary_and_sources():
    finding = SimpleNamespace(
        criterion_id="intent_clarity",
        category="ambiguity",
        disposition="addressed",
        evidence=(SimpleNamespace(source_id="artifact"),),
        proposal=SimpleNamespace(action="Clarify the boundary", replacement=None),
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(complete=True, sources_reviewed=("artifact",)),
        findings=(finding,),
    )
    assert validate_verdict_contract("spec_clarity", verdict) == ()


def test_runtime_contract_validator_allows_active_finding_below_severity_floor():
    finding = SimpleNamespace(
        criterion_id="intent_clarity",
        category="ambiguity",
        severity="Medium",
        disposition="confirmed",
        evidence=(SimpleNamespace(source_id="artifact"),),
        proposal=SimpleNamespace(action="Clarify when convenient", replacement=None),
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(complete=True, sources_reviewed=("artifact",)),
        findings=(finding,),
    )
    # spec_clarity has a High floor: a retained Medium concern is explicit
    # non-blocking information, matching Assay's universal decision rule.
    assert validate_verdict_contract("spec_clarity", verdict) == ()


def test_runtime_contract_validator_ignores_harness_owned_diagnostics():
    model_finding = SimpleNamespace(
        source="llm",
        criterion_id="intent_clarity",
        category="ambiguity",
        severity="High",
        disposition="confirmed",
        evidence=(SimpleNamespace(source_id="artifact"),),
        proposal=SimpleNamespace(action="Clarify", replacement=None),
        attack_spec=None,
    )
    harness_finding = SimpleNamespace(
        source="deterministic",
        criterion_id="coverage.complete",
        category="review_coverage",
        severity="Critical",
        disposition="confirmed",
        evidence=(SimpleNamespace(source_id="harness:coverage"),),
        proposal=None,
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="inconclusive",
        coverage=SimpleNamespace(
            complete=False,
            sources_reviewed=("artifact",),
            omissions=("artifact elided",),
        ),
        findings=(model_finding, harness_finding),
    )
    assert validate_verdict_contract("spec_clarity", verdict) == ()


def test_runtime_contract_validator_rejects_vocabulary_source_and_output_drift():
    finding = SimpleNamespace(
        criterion_id="invented_criterion",
        category="invented_category",
        disposition="confirmed",
        evidence=(SimpleNamespace(source_id="tool:unavailable"),),
        proposal=None,
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(complete=False),
        findings=(finding,),
    )
    errors = validate_verdict_contract("spec_clarity", verdict)
    assert any("pass requires complete coverage" in error for error in errors)
    assert any("undeclared criterion_id" in error for error in errors)
    assert any("undeclared category" in error for error in errors)
    assert any("only addressed or dismissed" in error for error in errors)
    assert any("undeclared source_id" in error for error in errors)
    assert any("missing its required proposal" in error for error in errors)


def test_runtime_contract_validator_requires_declarative_attack_spec():
    finding = SimpleNamespace(
        criterion_id="correctness",
        category="correctness",
        disposition="suspected",
        evidence=(SimpleNamespace(source_id="artifact"),),
        proposal=None,
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="revise",
        coverage=SimpleNamespace(complete=True),
        findings=(finding,),
    )
    assert any(
        "missing its declarative attack_spec" in error
        for error in validate_verdict_contract("impl_breaker", verdict)
    )


@pytest.mark.parametrize(
    ("rubric_name", "criterion_id", "category"),
    (
        ("impl_breaker", "correctness", "correctness"),
        ("swebench_impl_breaker", "meaning", "broken_meaning"),
    ),
)
@pytest.mark.parametrize("field_name", ("setup", "stimulus", "oracle"))
@pytest.mark.parametrize(
    "placeholder",
    ("...", "…", "---", "TODO", "TBD", "N/A", "[omitted]", "same as above"),
)
def test_breaker_contract_rejects_placeholder_attack_spec_fields(
    rubric_name,
    criterion_id,
    category,
    field_name,
    placeholder,
):
    attack_fields = {
        "setup": "Create a runner with one queued test.",
        "stimulus": "Abort the runner before the queued test executes.",
        "oracle": "The queued test does not execute after abort.",
    }
    attack_fields[field_name] = placeholder
    finding = SimpleNamespace(
        id="abort-stops-queued-test",
        source="llm",
        criterion_id=criterion_id,
        category=category,
        severity="High",
        disposition="confirmed",
        evidence=(
            SimpleNamespace(source_id="artifact"),
            SimpleNamespace(source_id="context:spec_intent"),
        ),
        proposal=None,
        attack_spec=SimpleNamespace(**attack_fields),
    )
    verdict = SimpleNamespace(
        decision="revise",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", "context:spec_intent"),
            omissions=(),
        ),
        findings=(finding,),
    )

    errors = validate_verdict_contract(rubric_name, verdict)

    assert any(
        f".attack_spec.{field_name}" in error
        and "non-placeholder" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("rubric_name", "criterion_id", "category"),
    (
        ("impl_breaker", "correctness", "correctness"),
        ("swebench_impl_breaker", "meaning", "broken_meaning"),
    ),
)
def test_breaker_contract_accepts_concrete_attack_spec_fields(
    rubric_name,
    criterion_id,
    category,
):
    finding = SimpleNamespace(
        id="abort-stops-queued-test",
        source="llm",
        criterion_id=criterion_id,
        category=category,
        severity="High",
        disposition="confirmed",
        evidence=(
            SimpleNamespace(source_id="artifact"),
            SimpleNamespace(source_id="context:spec_intent"),
        ),
        proposal=None,
        attack_spec=SimpleNamespace(
            setup="Create a runner with one queued test.",
            stimulus="Abort the runner before the queued test executes.",
            oracle="The queued test does not execute after abort.",
        ),
    )
    verdict = SimpleNamespace(
        decision="revise",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", "context:spec_intent"),
            omissions=(),
        ),
        findings=(finding,),
    )

    errors = validate_verdict_contract(rubric_name, verdict)

    assert not any("non-placeholder" in error for error in errors)


@pytest.mark.parametrize(
    ("oracle", "valid"),
    (
        (
            "Across two calls, broadcastAbort and each runner abort are "
            "observed exactly once.",
            True,
        ),
        (
            "Across two calls, broadcastAbort is observed exactly once. "
            "The current diff has no guard, so the second call repeats it.",
            False,
        ),
        (
            "The required output is present; this diff's implementation "
            "writes only the legacy summary.",
            False,
        ),
        (
            "The operation is idempotent because the implementation stores "
            "a completion marker.",
            False,
        ),
    ),
)
def test_impl_breaker_oracle_excludes_candidate_diagnosis(oracle, valid):
    finding = SimpleNamespace(
        source="llm",
        criterion_id="correctness",
        category="correctness",
        severity="High",
        disposition="confirmed",
        evidence=(SimpleNamespace(source_id="artifact"),),
        proposal=None,
        attack_spec=SimpleNamespace(
            setup="Create a runner.",
            stimulus="Call abort twice.",
            oracle=oracle,
        ),
    )
    verdict = SimpleNamespace(
        decision="revise",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", "context:spec_intent"),
            omissions=(),
        ),
        findings=(finding,),
    )

    errors = validate_verdict_contract("impl_breaker", verdict)
    diagnosis_errors = tuple(
        error for error in errors if "expected observable behavior" in error
    )
    assert bool(diagnosis_errors) is not valid


def test_breaker_inconclusive_cannot_claim_complete_coverage():
    verdict = SimpleNamespace(
        decision="inconclusive",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", "context:spec_intent"),
            omissions=(),
        ),
        findings=(),
    )

    assert validate_verdict_contract("impl_breaker", verdict) == (
        "impl_breaker: inconclusive requires coverage.complete=false",
    )
    non_breaker_coverage = SimpleNamespace(
        complete=True,
        sources_reviewed=("artifact",),
        omissions=(),
    )
    assert validate_verdict_contract(
        "spec_clarity",
        SimpleNamespace(
            decision="inconclusive",
            coverage=non_breaker_coverage,
            findings=(),
        ),
    ) == ()


def test_runtime_contract_validator_enforces_single_goal_replacement():
    finding = SimpleNamespace(
        criterion_id="red_coverage",
        category="canonical_revision",
        disposition="addressed",
        evidence=(SimpleNamespace(source_id="artifact"),),
        proposal=SimpleNamespace(
            action="Apply the complete candidate",
            replacement='{"intent":"clear","constraints":[],"done_criteria":[]}',
        ),
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(complete=True, sources_reviewed=("artifact",)),
        findings=(finding,),
    )
    assert validate_verdict_contract("goal_blue", verdict) == ()

    duplicate = SimpleNamespace(**{**finding.__dict__})
    verdict.findings = (finding, duplicate)
    assert any(
        "expected exactly one canonical replacement proposal" in error
        for error in validate_verdict_contract("goal_blue", verdict)
    )


@pytest.mark.parametrize(
    ("category", "replacement", "expected"),
    [
        (
            "add_constraint",
            '{"intent":"clear","constraints":[],"done_criteria":[]}',
            "forbids proposal.replacement",
        ),
        ("canonical_revision", "not json", "must be valid JSON"),
        (
            "canonical_revision",
            '{"intent":"clear","constraints":[]}',
            "must contain exactly",
        ),
        (
            "canonical_revision",
            '{"intent":"","constraints":{},"done_criteria":[]}',
            "invalid types",
        ),
    ],
)
def test_runtime_contract_validator_rejects_invalid_goal_replacement(
    category: str, replacement: str, expected: str,
):
    finding = SimpleNamespace(
        criterion_id="red_coverage",
        category=category,
        disposition="addressed",
        evidence=(SimpleNamespace(source_id="artifact"),),
        proposal=SimpleNamespace(
            action="Apply the candidate",
            replacement=replacement,
        ),
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(complete=True, sources_reviewed=("artifact",)),
        findings=(finding,),
    )
    errors = validate_verdict_contract("goal_blue", verdict)
    assert any(expected in error for error in errors)


def test_runtime_contract_validator_enforces_criterion_category_pairing():
    finding = SimpleNamespace(
        id="bad_pair",
        criterion_id="intent_clarity",
        category="contradiction",
        severity="Low",
        disposition="addressed",
        targets=(),
        evidence=(SimpleNamespace(source_id="artifact"),),
        proposal=SimpleNamespace(action="clarify", replacement=None),
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(complete=True, sources_reviewed=("artifact",)),
        findings=(finding,),
    )
    assert any(
        "not allowed for criterion" in error
        for error in validate_verdict_contract("spec_clarity", verdict)
    )


def test_runtime_contract_validator_requires_material_target_closure():
    prior_ref = (
        "prior:red:0:sha256-" + "a" * 64 + ":red-1"
    )
    finding = SimpleNamespace(
        id="blue_disposition",
        criterion_id="attack_execution",
        category="dismissed_attack",
        severity="Low",
        disposition="addressed",
        targets=(),
        evidence=(SimpleNamespace(source_id="artifact"),),
        proposal=SimpleNamespace(action="retain result", replacement=None),
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", prior_ref),
        ),
        findings=(finding,),
        criterion_results=tuple(
            SimpleNamespace(
                criterion_id=criterion,
                outcome="satisfied",
                reasoning="Current attack evidence establishes this result.",
                evidence=(SimpleNamespace(source_id="artifact"),),
                headline=index == 0,
            )
            for index, criterion in enumerate(contract("breach_judge").criteria)
        ),
    )
    prior = ({
        "id": "red-1",
        "ref": prior_ref,
        "source_id": prior_ref,
        "severity": "High",
        "disposition": "confirmed",
    },)
    errors = validate_verdict_contract(
        "breach_judge",
        verdict,
        prior_findings=prior,
        available_source_ids={"artifact", prior_ref},
    )
    assert any("requires exactly one adjudication" in error for error in errors)
    finding.targets = (prior_ref,)
    # The exact target is the prior-record join. Independent artifact evidence
    # is required, but duplicating the prior packet as Evidence is not.
    finding.evidence = (SimpleNamespace(source_id="artifact"),)
    assert validate_verdict_contract(
        "breach_judge",
        verdict,
        prior_findings=prior,
        available_source_ids={"artifact", prior_ref},
    ) == ()

    verdict.coverage.sources_reviewed = ("artifact",)
    errors = validate_verdict_contract(
        "breach_judge",
        verdict,
        prior_findings=prior,
        available_source_ids={"artifact", prior_ref},
    )
    assert any("linked prior source" in error for error in errors)
    verdict.coverage.sources_reviewed = ("artifact", prior_ref)

    unknown_ref = "prior:red:9:sha256-" + "b" * 64 + ":unknown"
    finding.targets = (unknown_ref,)
    errors = validate_verdict_contract(
        "breach_judge",
        verdict,
        prior_findings=prior,
        available_source_ids={"artifact", prior_ref},
    )
    assert any("does not name a supplied prior finding" in error for error in errors)
    assert any("unknown qualified prior targets" in error for error in errors)
    finding.targets = (prior_ref,)

    deterministic_prior = ({**prior[0], "source": "deterministic"},)
    errors = validate_verdict_contract(
        "breach_judge",
        verdict,
        prior_findings=deterministic_prior,
        available_source_ids={"artifact", prior_ref},
    )
    assert any("deterministic prior failures cannot be closed" in error
               for error in errors)


def test_breach_judge_closes_medium_prior_without_lowering_blocker_floor():
    prior_ref = "prior:breaker:0:sha256-" + "a" * 64 + ":race-a"
    evidence = (SimpleNamespace(source_id="artifact"),)
    finding = SimpleNamespace(
        id="race-a-survived",
        criterion_id="attack_execution",
        category="surviving_test",
        severity="Medium",
        disposition="addressed",
        targets=(prior_ref,),
        evidence=evidence,
        proposal=SimpleNamespace(
            action="Retain the passing attack as a regression test.",
            replacement=None,
        ),
        attack_spec=None,
    )
    criterion_results = tuple(
        SimpleNamespace(
            criterion_id=criterion_id,
            outcome="satisfied",
            reasoning="The current attack ledger supplies the result.",
            evidence=evidence,
            headline=index == 0,
        )
        for index, criterion_id in enumerate(
            contract("breach_judge").criteria
        )
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", prior_ref),
        ),
        findings=(finding,),
        criterion_results=criterion_results,
    )
    prior = ({
        "id": "race-a",
        "ref": prior_ref,
        "source_id": prior_ref,
        "severity": "Medium",
        "disposition": "suspected",
        "category": "correctness",
        "targets": (),
        "source": "llm",
    },)
    validation_kwargs = {
        "prior_findings": prior,
        "available_source_ids": {"artifact", prior_ref},
    }

    assert validate_verdict_contract(
        "breach_judge",
        verdict,
        **validation_kwargs,
    ) == ()

    finding.targets = ()
    errors = validate_verdict_contract(
        "breach_judge",
        verdict,
        **validation_kwargs,
    )
    assert any("requires exactly one adjudication" in error for error in errors)

    finding.targets = (prior_ref,)
    finding.category = "confirmed_attack"
    finding.severity = "High"
    finding.disposition = "confirmed"
    errors = validate_verdict_contract(
        "breach_judge",
        verdict,
        **validation_kwargs,
    )
    assert any("at/above the High floor" in error for error in errors)


def test_plan_judge_under_decomposition_closure_requires_real_repartition():
    prior_ref = (
        "prior:critic:0:sha256-" + "a" * 64 + ":under-decomposed"
    )
    plan = {
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [
                {
                    "id": "config-load",
                    "directive": {
                        "slice_id": "config-load",
                        "intent": "Load configuration values",
                    },
                },
                {
                    "id": "alias-validation",
                    "directive": {
                        "slice_id": "alias-validation",
                        "intent": "Validate alias mappings",
                    },
                },
                {
                    "id": "release-notes",
                    "directive": {
                        "slice_id": "release-notes",
                        "intent": "Publish release documentation",
                    },
                },
            ],
        }],
    }
    plan_text = json.dumps(plan, sort_keys=True)
    evidence = SimpleNamespace(
        source_id="artifact",
        kind="tier_dag",
        excerpt=plan_text,
    )
    finding = SimpleNamespace(
        id="critic-disposition",
        criterion_id="critic_disposition",
        category="unresolved_critic",
        severity="High",
        disposition="addressed",
        targets=(prior_ref,),
        evidence=(evidence,),
        proposal=SimpleNamespace(
            action="No further plan change is required.",
            replacement=None,
        ),
        attack_spec=None,
    )
    criteria = tuple(
        SimpleNamespace(
            criterion_id=criterion,
            outcome="satisfied",
            evidence=(evidence,),
            headline=index == 0,
        )
        for index, criterion in enumerate((
            "intent_coverage",
            "executability",
            "critic_disposition",
            "contradiction",
        ))
    )
    required_sources = (
        "artifact",
        "context:spec_intent",
        "context:spec_constraints",
        "context:spec_done_criteria",
        "context:prior_critic_findings",
        "context:prior_judge_findings",
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=required_sources,
        ),
        findings=(finding,),
        criterion_results=criteria,
    )
    prior = ({
        "id": "under-decomposed",
        "ref": prior_ref,
        "source_id": "context:prior_critic_findings",
        "severity": "High",
        "disposition": "confirmed",
        "category": "under_decomposed",
        "targets": (),
        "payload": {
            "proposal": {
                "action": (
                    "Split configuration loading and alias validation into "
                    "independently landable responsibilities."
                ),
            },
            "evidence": [],
        },
    },)

    errors = validate_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=prior,
        current_artifact=plan_text,
    )
    assert any("requires REPARTITION_PROOF" in error for error in errors)

    finding.proposal.action = (
        "REPARTITION_PROOF: "
        + json.dumps({
            "config-load": "Load configuration values",
            "alias-validation": "Validate alias mappings",
        }, separators=(",", ":"))
    )
    assert validate_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=prior,
        current_artifact=plan_text,
    ) == ()

    finding.proposal.action = (
        "REPARTITION_PROOF: "
        + json.dumps({
            "config-load": "Load configuration values",
            "release-notes": "Publish release documentation",
        }, separators=(",", ":"))
    )
    errors = validate_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=prior,
        current_artifact=plan_text,
    )
    assert any("semantically related actual current slice" in error
               for error in errors)

    finding.disposition = "dismissed"
    finding.proposal.action = (
        "Dismissed: required field and default seeding are one atomic increment."
    )
    errors = validate_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=prior,
        current_artifact=plan_text,
    )
    assert any("requires ATOMICITY_PROOF" in error for error in errors)

    atomic_plan = {
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "schema-overlay",
                "directive": {
                    "slice_id": "schema-overlay",
                    "intent": "Add required field and seed default configuration",
                },
            }],
        }],
    }
    atomic_text = json.dumps(atomic_plan, sort_keys=True)
    evidence.excerpt = atomic_text
    prior[0]["payload"]["proposal"]["action"] = (
        "Split the required field from default configuration seeding."
    )
    finding.proposal.action = (
        "ATOMICITY_PROOF: "
        + json.dumps({
            "slice_id": "schema-overlay",
            "cannot_land_independently": True,
            "cannot_verify_independently": True,
            "reason": (
                "The required field and default configuration seeding share "
                "one constructor invariant"
            ),
        }, separators=(",", ":"))
    )
    assert validate_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=prior,
        current_artifact=atomic_text,
    ) == ()

    finding.proposal.action = (
        "ATOMICITY_PROOF: "
        + json.dumps({
            "slice_id": "schema-overlay",
            "cannot_land_independently": True,
            "cannot_verify_independently": True,
            "reason": (
                "It is false: the field can land independently and the "
                "configuration can be verified independently"
            ),
        }, separators=(",", ":"))
    )
    assert any(
        "requires ATOMICITY_PROOF" in error
        for error in validate_verdict_contract(
            "plan_judge",
            verdict,
            prior_findings=prior,
            current_artifact=atomic_text,
        )
    )


def _compact_proof_fixture(
    *,
    plan_text: str,
    disposition: str,
    proposal_action: str,
    prior_action: str,
    evidence_slice_ids: tuple[str, ...],
):
    prior_ref = (
        "prior:critic:0:sha256-" + "a" * 64 + ":under-decomposed"
    )
    evidence = tuple(
        SimpleNamespace(
            source_id="artifact",
            kind="tier_dag",
            excerpt=json.dumps(slice_id),
        )
        for slice_id in evidence_slice_ids
    )
    finding = SimpleNamespace(
        id="compact-proof-closure",
        criterion_id="critic_disposition",
        category="unresolved_critic",
        severity="High",
        disposition=disposition,
        targets=(prior_ref,),
        evidence=evidence,
        proposal=SimpleNamespace(
            action=proposal_action,
            replacement=None,
        ),
        attack_spec=None,
    )
    criteria = tuple(
        SimpleNamespace(
            criterion_id=criterion,
            outcome="satisfied",
            evidence=evidence,
            headline=index == 0,
        )
        for index, criterion in enumerate((
            "intent_coverage",
            "executability",
            "critic_disposition",
            "contradiction",
        ))
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=(
                "artifact",
                "context:spec_intent",
                "context:spec_constraints",
                "context:spec_done_criteria",
                "context:prior_critic_findings",
                "context:prior_judge_findings",
            ),
        ),
        findings=(finding,),
        criterion_results=criteria,
    )
    prior = ({
        "id": "under-decomposed",
        "ref": prior_ref,
        "source_id": "context:prior_critic_findings",
        "severity": "High",
        "disposition": "confirmed",
        "category": "under_decomposed",
        "targets": (),
        "payload": {
            "proposal": {
                "action": prior_action,
                "replacement": None,
            },
            "evidence": [],
        },
    },)
    return verdict, finding, prior


def _intent_sha256(intent: str) -> str:
    return "sha256:" + hashlib.sha256(intent.encode("utf-8")).hexdigest()


def test_plan_judge_compact_repartition_proof_binds_long_intents():
    config_intent = (
        "Load configuration values into an independently landable foundation "
        "for alias validation. "
        + "Preserve the configuration loading invariant. " * 20
    )
    validation_intent = (
        "Validate alias mappings as an independently verifiable increment "
        "after configuration loading. "
        + "Preserve the alias validation invariant. " * 20
    )
    assert len(config_intent) > 512
    assert len(validation_intent) > 512
    plan_text = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [
                {
                    "id": "config-load",
                    "directive": {
                        "slice_id": "config-load",
                        "intent": config_intent,
                    },
                },
                {
                    "id": "alias-validation",
                    "directive": {
                        "slice_id": "alias-validation",
                        "intent": validation_intent,
                    },
                },
            ],
        }],
    }, sort_keys=True)
    proof = {
        "config-load": {"intent_sha256": _intent_sha256(config_intent)},
        "alias-validation": {
            "intent_sha256": _intent_sha256(validation_intent),
        },
    }
    verdict, finding, prior = _compact_proof_fixture(
        plan_text=plan_text,
        disposition="addressed",
        proposal_action="REPARTITION_PROOF: " + json.dumps(proof),
        prior_action=(
            "Split configuration loading and alias validation into "
            "independently landable responsibilities."
        ),
        evidence_slice_ids=("config-load", "alias-validation"),
    )

    def validate() -> tuple[str, ...]:
        return validate_verdict_contract(
            "plan_judge",
            verdict,
            prior_findings=prior,
            current_artifact=plan_text,
        )

    assert validate() == ()

    wrong_hash = json.loads(json.dumps(proof))
    wrong_hash["config-load"]["intent_sha256"] = "sha256:" + "0" * 64
    finding.proposal.action = (
        "REPARTITION_PROOF: " + json.dumps(wrong_hash)
    )
    assert any("requires REPARTITION_PROOF" in error for error in validate())

    unknown_slice = {
        "config-load": proof["config-load"],
        "unknown-slice": proof["alias-validation"],
    }
    finding.proposal.action = (
        "REPARTITION_PROOF: " + json.dumps(unknown_slice)
    )
    assert any("requires REPARTITION_PROOF" in error for error in validate())

    finding.proposal.action = "REPARTITION_PROOF: " + json.dumps(proof)
    finding.evidence = finding.evidence[:1]
    assert any("requires REPARTITION_PROOF" in error for error in validate())


def test_plan_judge_compact_atomicity_proof_binds_long_intent():
    atomic_intent = (
        "Add the required field and seed default configuration under one "
        "shared constructor invariant. "
        + "Keep required field and default configuration seeding atomic. " * 20
    )
    assert len(atomic_intent) > 512
    plan_text = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "schema-overlay",
                "directive": {
                    "slice_id": "schema-overlay",
                    "intent": atomic_intent,
                },
            }],
        }],
    }, sort_keys=True)
    proof = {
        "slice_id": "schema-overlay",
        "intent_sha256": _intent_sha256(atomic_intent),
        "cannot_land_independently": True,
        "cannot_verify_independently": True,
        "reason": (
            "The required field and default configuration seeding share one "
            "constructor invariant"
        ),
    }
    verdict, finding, prior = _compact_proof_fixture(
        plan_text=plan_text,
        disposition="dismissed",
        proposal_action="ATOMICITY_PROOF: " + json.dumps(proof),
        prior_action=(
            "Split the required field from default configuration seeding."
        ),
        evidence_slice_ids=("schema-overlay",),
    )

    def validate() -> tuple[str, ...]:
        return validate_verdict_contract(
            "plan_judge",
            verdict,
            prior_findings=prior,
            current_artifact=plan_text,
        )

    assert validate() == ()

    wrong_hash = {**proof, "intent_sha256": "sha256:" + "0" * 64}
    finding.proposal.action = "ATOMICITY_PROOF: " + json.dumps(wrong_hash)
    assert any("requires ATOMICITY_PROOF" in error for error in validate())

    unknown_slice = {**proof, "slice_id": "unknown-slice"}
    finding.proposal.action = "ATOMICITY_PROOF: " + json.dumps(unknown_slice)
    assert any("requires ATOMICITY_PROOF" in error for error in validate())

    finding.proposal.action = "ATOMICITY_PROOF: " + json.dumps(proof)
    finding.evidence = (
        SimpleNamespace(
            source_id="artifact",
            kind="tier_dag",
            excerpt=json.dumps("different-slice"),
        ),
    )
    assert any("requires ATOMICITY_PROOF" in error for error in validate())


def test_contract_canonicalization_retains_unproved_prior_fail_closed():
    prior_ref = (
        "prior:critic:0:sha256-" + "a" * 64 + ":under-decomposed"
    )
    plan_text = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "combined",
                "directive": {
                    "slice_id": "combined",
                    "intent": "Load configuration and validate aliases",
                },
            }],
        }],
    }, sort_keys=True)
    evidence = Evidence(
        source_id="artifact",
        kind="directive",
        location="artifact",
        excerpt=plan_text,
    )
    finding = Finding(
        id="close-prior",
        severity="High",
        category="other",
        evidence=(evidence,),
        suggested_action="ATOMICITY_PROOF: {",
        source="llm",
        criterion_id="critic_disposition",
        confidence=0.9,
        disposition="dismissed",
        targets=(prior_ref,),
        proposal=Proposal(action="ATOMICITY_PROOF: {"),
    )
    criteria = tuple(
        CriterionResult(
            criterion_id=criterion,
            outcome="satisfied",
            reasoning="model claimed the criterion was satisfied",
            evidence=(evidence,),
            headline=index == 0,
        )
        for index, criterion in enumerate((
            "intent_coverage",
            "executability",
            "critic_disposition",
            "contradiction",
        ))
    )
    required_sources = (
        "artifact",
        "context:spec_intent",
        "context:spec_constraints",
        "context:spec_done_criteria",
        "context:prior_critic_findings",
        "context:prior_judge_findings",
    )
    verdict = Verdict(
        rubric="plan_judge",
        decision="pass",
        findings=(finding,),
        artifact_ref=ContentHash("sha256:" + "b" * 64),
        artifact_commit=None,
        reasoning="the model claimed the prior was closed",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=required_sources,
            omissions=(),
        ),
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
        criterion_results=criteria,
    )
    prior_action = "Split loading from validation into independent slices."
    prior = ({
        "id": "under-decomposed",
        "ref": prior_ref,
        "source_id": "context:prior_critic_findings",
        "severity": "High",
        "disposition": "confirmed",
        "category": "under_decomposed",
        "targets": (),
        "payload": {
            "proposal": {"action": prior_action, "replacement": None},
            "evidence": [],
        },
    },)

    normalized, actions = canonicalize_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=prior,
        current_artifact=plan_text,
    )

    assert normalized.decision == "revise"
    assert normalized.findings[0].category == "unresolved_critic"
    assert normalized.findings[0].disposition == "confirmed"
    assert normalized.findings[0].proposal.action == prior_action
    assert next(
        result for result in normalized.criterion_results
        if result.criterion_id == "critic_disposition"
    ).outcome == "violated"
    actions_by_path = {action["path"]: action for action in actions}
    assert {
        "findings[0].category",
        "findings[0].disposition",
        "findings[0].proposal",
        "findings[0].suggested_action",
        "criterion_results[2].outcome",
        "criterion_results[2].reasoning",
        "decision",
        "reasoning",
    } <= set(actions_by_path)
    assert actions_by_path["findings[0].category"]["before"] == "other"
    assert (
        actions_by_path["findings[0].category"]["after"]
        == "unresolved_critic"
    )
    assert (
        actions_by_path["findings[0].disposition"]["before"]
        == "dismissed"
    )
    assert (
        actions_by_path["findings[0].disposition"]["after"]
        == "confirmed"
    )
    assert (
        actions_by_path["criterion_results[2].outcome"]["before"]
        == "satisfied"
    )
    assert (
        actions_by_path["criterion_results[2].outcome"]["after"]
        == "violated"
    )
    assert actions_by_path["decision"]["before"] == "pass"
    assert actions_by_path["decision"]["after"] == "revise"
    for path in (
        "findings[0].proposal",
        "findings[0].suggested_action",
        "criterion_results[2].reasoning",
        "reasoning",
    ):
        hashes = [
            value
            for key, value in actions_by_path[path].items()
            if key.endswith("_sha256")
        ]
        assert len(hashes) == 2
        assert all(
            isinstance(value, str) and value.startswith("sha256:")
            for value in hashes
        )
        assert hashes[0] != hashes[1]
    assert validate_verdict_contract(
        "plan_judge",
        normalized,
        prior_findings=prior,
        current_artifact=plan_text,
    ) == ()
    again, second_actions = canonicalize_verdict_contract(
        "plan_judge",
        normalized,
        prior_findings=prior,
        current_artifact=plan_text,
    )
    assert again == normalized
    assert second_actions == ()


def test_contract_canonicalization_never_chooses_multi_category_mapping():
    finding = Finding(
        id="ambiguous-category",
        severity="High",
        category="not-declared",
        evidence=(),
        suggested_action="repair",
        source="llm",
        criterion_id="intent_coverage",
        confidence=0.9,
        disposition="confirmed",
        proposal=Proposal(action="repair"),
    )
    verdict = SimpleNamespace(
        decision="revise",
        findings=(finding,),
        criterion_results=(),
        reasoning="repair",
    )
    normalized, actions = canonicalize_verdict_contract(
        "plan_judge",
        verdict,
    )
    assert normalized is verdict
    assert actions == ()


def test_contract_canonicalization_never_rewrites_deterministic_category():
    finding = Finding(
        id="harness-diagnostic",
        severity="High",
        category="harness_category",
        evidence=(),
        suggested_action="inspect the harness diagnostic",
        source="deterministic",
        criterion_id="critic_disposition",
        disposition="confirmed",
    )
    verdict = SimpleNamespace(
        decision="revise",
        findings=(finding,),
        criterion_results=(),
        reasoning="the harness emitted a deterministic diagnostic",
    )

    normalized, actions = canonicalize_verdict_contract(
        "plan_judge",
        verdict,
    )

    assert normalized is verdict
    assert normalized.findings[0].category == "harness_category"
    assert actions == ()


def _deepswe_critic_mixed_contract_verdict() -> tuple[
    Verdict, Finding, Finding, str, str,
]:
    directive = "Add alias support while preserving compatibility."
    artifact = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "alias-support",
                "directive": {
                    "slice_id": "alias-support",
                    "intent": (
                        "Implement alias parsing and compatibility wiring in "
                        "one increment."
                    ),
                },
            }],
        }],
    }, separators=(",", ":"))
    paired_evidence = (
        Evidence(
            source_id="artifact",
            kind="tier_dag",
            location="artifact",
            excerpt=artifact,
        ),
        Evidence(
            source_id="context:spec_intent",
            kind="directive",
            location="spec_intent",
            excerpt=directive,
        ),
    )
    valid = Finding(
        id="independent-alias-responsibilities",
        severity="High",
        category="under_decomposed",
        evidence=paired_evidence,
        suggested_action=(
            "Split alias parsing from independently landable compatibility "
            "wiring."
        ),
        source="llm",
        criterion_id="responsibility_size",
        confidence=0.95,
        disposition="confirmed",
        proposal=Proposal(action=(
            "Split alias parsing from independently landable compatibility "
            "wiring."
        )),
    )
    invalid = Finding(
        id="sequence-alias-wiring",
        severity="Medium",
        category="missing_sequencing",
        evidence=(paired_evidence[0],),
        suggested_action="Make compatibility wiring depend on alias parsing.",
        source="llm",
        criterion_id="sequencing",
        confidence=0.8,
        disposition="confirmed",
        proposal=Proposal(
            action="Make compatibility wiring depend on alias parsing."
        ),
    )
    verdict = Verdict(
        rubric="deepswe_plan_critic",
        decision="revise",
        findings=(valid, invalid),
        artifact_ref=ContentHash("sha256:" + "d" * 64),
        artifact_commit=None,
        reasoning="The plan contains independently actionable problems.",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact", "context:spec_intent"),
            omissions=(),
        ),
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
        criterion_results=(),
    )
    return verdict, valid, invalid, artifact, directive


def _canonicalization_test_hash(value: object) -> str:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_contract_canonicalization_salvages_grounded_critic_blocker():
    verdict, valid, invalid, artifact, directive = (
        _deepswe_critic_mixed_contract_verdict()
    )
    source_ids = {"artifact", "context:spec_intent"}
    context = {"spec_intent": directive}

    errors = validate_verdict_contract(
        "deepswe_plan_critic",
        verdict,
        available_source_ids=source_ids,
        supplied_context=context,
        current_artifact=artifact,
    )
    assert len(errors) == 1
    assert "finding[1] requires paired evidence" in errors[0]

    normalized, actions = canonicalize_verdict_contract(
        "deepswe_plan_critic",
        verdict,
        available_source_ids=source_ids,
        supplied_context=context,
        current_artifact=artifact,
    )

    assert normalized.decision == "revise"
    assert normalized.findings == (valid,)
    assert normalized.findings[0] is valid
    assert validate_verdict_contract(
        "deepswe_plan_critic",
        normalized,
        available_source_ids=source_ids,
        supplied_context=context,
        current_artifact=artifact,
    ) == ()
    isolated_errors = validate_verdict_contract(
        "deepswe_plan_critic",
        dataclasses.replace(verdict, findings=(invalid,)),
        available_source_ids=source_ids,
        supplied_context=context,
        current_artifact=artifact,
    )
    assert len(actions) == 1
    (action,) = actions
    assert action == {
        "operation": "drop_contract_invalid_critic_finding",
        "path": "findings[1]",
        "finding_id": invalid.id,
        "before_sha256": _canonicalization_test_hash(invalid),
        "validation_errors": list(isolated_errors),
        "validation_errors_sha256": _canonicalization_test_hash(
            isolated_errors
        ),
    }

    again, second_actions = canonicalize_verdict_contract(
        "deepswe_plan_critic",
        normalized,
        available_source_ids=source_ids,
        supplied_context=context,
        current_artifact=artifact,
    )
    assert again is normalized
    assert second_actions == ()


@pytest.mark.parametrize("decision", ("pass", "reject", "inconclusive"))
def test_contract_canonicalization_does_not_prune_non_revise_critic(
    decision,
):
    verdict, _, _, artifact, directive = _deepswe_critic_mixed_contract_verdict()
    verdict = dataclasses.replace(verdict, decision=decision)

    normalized, actions = canonicalize_verdict_contract(
        "deepswe_plan_critic",
        verdict,
        available_source_ids={"artifact", "context:spec_intent"},
        supplied_context={"spec_intent": directive},
        current_artifact=artifact,
    )

    assert normalized is verdict
    assert normalized.findings == verdict.findings
    assert actions == ()


def test_contract_canonicalization_does_not_prune_every_critic_finding():
    verdict, valid, invalid, artifact, directive = (
        _deepswe_critic_mixed_contract_verdict()
    )
    invalid_high = dataclasses.replace(
        valid,
        evidence=(valid.evidence[0],),
    )
    verdict = dataclasses.replace(verdict, findings=(invalid_high, invalid))

    normalized, actions = canonicalize_verdict_contract(
        "deepswe_plan_critic",
        verdict,
        available_source_ids={"artifact", "context:spec_intent"},
        supplied_context={"spec_intent": directive},
        current_artifact=artifact,
    )

    assert normalized is verdict
    assert normalized.findings == (invalid_high, invalid)
    assert actions == ()


@pytest.mark.parametrize("rubric_name", ("plan_judge",))
def test_contract_canonicalization_does_not_prune_join_constrained_roles(
    rubric_name,
):
    verdict, _, _, artifact, directive = _deepswe_critic_mixed_contract_verdict()

    normalized, actions = canonicalize_verdict_contract(
        rubric_name,
        verdict,
        available_source_ids={"artifact", "context:spec_intent"},
        supplied_context={"spec_intent": directive},
        current_artifact=artifact,
    )

    assert normalized is verdict
    assert normalized.findings == verdict.findings
    assert actions == ()


def test_contract_canonicalization_never_demotes_original_reject():
    prior_ref = "prior:critic:0:sha256-" + "a" * 64 + ":split-work"
    plan_text = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "combined",
                "directive": {
                    "slice_id": "combined",
                    "intent": "Change two coupled behaviors",
                },
            }],
        }],
    }, sort_keys=True)
    evidence = Evidence(
        source_id="artifact",
        kind="directive",
        location="artifact",
        excerpt=plan_text,
    )
    invalid_closure = Finding(
        id="invalid-closure",
        severity="High",
        category="other",
        evidence=(evidence,),
        suggested_action="ATOMICITY_PROOF: {",
        source="llm",
        criterion_id="critic_disposition",
        confidence=0.9,
        disposition="dismissed",
        targets=(prior_ref,),
        proposal=Proposal(action="ATOMICITY_PROOF: {"),
    )
    unfixable = Finding(
        id="unfixable-conflict",
        severity="Critical",
        category="intent_gap",
        evidence=(evidence,),
        suggested_action="the root requirements conflict",
        source="llm",
        criterion_id="intent_coverage",
        confidence=0.9,
        disposition="unfixable",
        proposal=Proposal(action=(
            "UNFIXABLE_PROOF: "
            '{"scope":"original_intent","bounded_plan_possible":false,'
            '"reason":"immutable root requirements fundamentally conflict"}'
        )),
    )
    criteria = tuple(
        CriterionResult(
            criterion_id=criterion,
            outcome=("violated" if criterion == "intent_coverage"
                     else "satisfied"),
            reasoning="the current candidate was checked",
            evidence=(evidence,),
            headline=criterion == "intent_coverage",
        )
        for criterion in (
            "intent_coverage",
            "executability",
            "critic_disposition",
            "contradiction",
        )
    )
    verdict = Verdict(
        rubric="plan_judge",
        decision="reject",
        findings=(invalid_closure, unfixable),
        artifact_ref=ContentHash("sha256:" + "b" * 64),
        artifact_commit=None,
        reasoning="the root requirements cannot be reconciled",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=(
                "artifact",
                "context:spec_intent",
                "context:spec_constraints",
                "context:spec_done_criteria",
                "context:prior_critic_findings",
                "context:prior_judge_findings",
            ),
            omissions=(),
        ),
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
        criterion_results=criteria,
    )
    prior = ({
        "id": "split-work",
        "ref": prior_ref,
        "source_id": "context:prior_critic_findings",
        "severity": "High",
        "disposition": "confirmed",
        "category": "under_decomposed",
        "targets": (),
        "payload": {
            "proposal": {
                "action": "Split the coupled behaviors into independent work.",
                "replacement": None,
            },
            "evidence": [],
        },
    },)

    normalized, actions = canonicalize_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=prior,
        current_artifact=plan_text,
    )

    assert normalized.decision == "reject"
    assert not any(action["path"] == "decision" for action in actions)


def test_contract_canonicalization_preserves_valid_atomicity_closure():
    prior_ref = "prior:critic:0:sha256-" + "a" * 64 + ":split-work"
    intent = "Load configuration and validate values in one constructor"
    plan_text = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "combined",
                "directive": {
                    "slice_id": "combined",
                    "intent": intent,
                },
            }],
        }],
    }, sort_keys=True)
    proof = (
        "ATOMICITY_PROOF: "
        + json.dumps({
            "slice_id": "combined",
            "cannot_land_independently": True,
            "cannot_verify_independently": True,
            "reason": (
                "Configuration loading and value validation share one "
                "constructor invariant"
            ),
        }, separators=(",", ":"))
    )
    finding = Finding(
        id="valid-closure",
        severity="High",
        category="unresolved_critic",
        evidence=(Evidence(
            source_id="artifact",
            kind="directive",
            location="artifact",
            excerpt=plan_text,
        ),),
        suggested_action=proof,
        source="llm",
        criterion_id="critic_disposition",
        confidence=0.9,
        disposition="dismissed",
        targets=(prior_ref,),
        proposal=Proposal(action=proof),
    )
    criteria = tuple(
        CriterionResult(
            criterion_id=criterion,
            outcome="satisfied",
            reasoning="the current candidate was checked",
            evidence=finding.evidence,
            headline=criterion == "critic_disposition",
        )
        for criterion in (
            "intent_coverage",
            "executability",
            "critic_disposition",
            "contradiction",
        )
    )
    verdict = Verdict(
        rubric="plan_judge",
        decision="pass",
        findings=(finding,),
        artifact_ref=ContentHash("sha256:" + "b" * 64),
        artifact_commit=None,
        reasoning="the prior is disproved by an atomic increment",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=(
                "artifact",
                "context:spec_intent",
                "context:spec_constraints",
                "context:spec_done_criteria",
                "context:prior_critic_findings",
                "context:prior_judge_findings",
            ),
            omissions=(),
        ),
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
        criterion_results=criteria,
    )
    prior = ({
        "id": "split-work",
        "ref": prior_ref,
        "source_id": "context:prior_critic_findings",
        "severity": "High",
        "disposition": "confirmed",
        "category": "under_decomposed",
        "targets": (),
        "payload": {
            "proposal": {
                "action": "Split configuration loading from value validation.",
                "replacement": None,
            },
            "evidence": [],
        },
    },)

    assert validate_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=prior,
        current_artifact=plan_text,
    ) == ()

    normalized, actions = canonicalize_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=prior,
        current_artifact=plan_text,
    )

    assert normalized is verdict
    assert normalized.findings[0] == finding
    assert actions == ()


@pytest.mark.parametrize(
    ("disposition", "proof_action"),
    (
        ("addressed", "REPARTITION_PROOF: {"),
        ("dismissed", "ATOMICITY_PROOF: {"),
    ),
)
def test_contract_canonicalization_retains_multi_target_closure_fail_closed(
    disposition,
    proof_action,
):
    under_ref = (
        "prior:critic:0:sha256-" + "a" * 64 + ":under-decomposed"
    )
    other_ref = (
        "prior:judge:0:sha256-" + "b" * 64 + ":other-material-prior"
    )
    plan_text = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "combined",
                "directive": {
                    "slice_id": "combined",
                    "intent": "Change two coupled behaviors",
                },
            }],
        }],
    }, sort_keys=True)
    evidence = (Evidence(
        source_id="artifact",
        kind="directive",
        location="artifact",
        excerpt=plan_text,
    ),)
    closure = Finding(
        id="grouped-closure",
        severity="High",
        category="unresolved_critic",
        evidence=evidence,
        suggested_action=proof_action,
        source="llm",
        criterion_id="critic_disposition",
        confidence=0.9,
        disposition=disposition,
        targets=(under_ref, other_ref),
        proposal=Proposal(action=proof_action),
    )
    blocker = Finding(
        id="independent-blocker",
        severity="High",
        category="parallel_conflict",
        evidence=evidence,
        suggested_action="remove the conflicting write",
        source="llm",
        criterion_id="executability",
        confidence=0.9,
        disposition="confirmed",
        proposal=Proposal(action="remove the conflicting write"),
    )
    criteria = tuple(
        CriterionResult(
            criterion_id=criterion,
            outcome=("violated" if criterion == "executability"
                     else "not_applicable" if criterion == "contradiction"
                     else "satisfied"),
            reasoning="the current candidate was checked",
            evidence=evidence,
            headline=criterion == "executability",
        )
        for criterion in (
            "intent_coverage",
            "executability",
            "critic_disposition",
            "contradiction",
        )
    )
    verdict = Verdict(
        rubric="plan_judge",
        decision="revise",
        findings=(closure, blocker),
        artifact_ref=ContentHash("sha256:" + "c" * 64),
        artifact_commit=None,
        reasoning="one structural blocker remains",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=(
                "artifact",
                "context:spec_intent",
                "context:spec_constraints",
                "context:spec_done_criteria",
                "context:prior_critic_findings",
                "context:prior_judge_findings",
            ),
            omissions=(),
        ),
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
        criterion_results=criteria,
    )
    priors = (
        {
            "id": "under-decomposed",
            "ref": under_ref,
            "source_id": "context:prior_critic_findings",
            "severity": "High",
            "disposition": "confirmed",
            "category": "under_decomposed",
            "targets": (),
            "payload": {
                "proposal": {
                    "action": "Split the coupled work into independent slices.",
                    "replacement": None,
                },
                "evidence": [],
            },
        },
        {
            "id": "other-material-prior",
            "ref": other_ref,
            "source_id": "context:prior_judge_findings",
            "severity": "High",
            "disposition": "confirmed",
            "category": "unresolved_critic",
            "targets": (),
            "payload": {
                "proposal": {
                    "action": "Retain the independent verification boundary.",
                    "replacement": None,
                },
                "evidence": [],
            },
        },
    )

    normalized, actions = canonicalize_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=priors,
        current_artifact=plan_text,
    )

    normalized_closure = normalized.findings[0]
    assert normalized.decision == verdict.decision
    assert normalized_closure.disposition == "confirmed"
    assert normalized_closure.targets == closure.targets
    assert normalized_closure.evidence == closure.evidence
    assert normalized_closure.proposal == closure.proposal
    assert normalized_closure.suggested_action == closure.suggested_action
    assert normalized.findings[1] == blocker
    normalized_criterion = next(
        result for result in normalized.criterion_results
        if result.criterion_id == "critic_disposition"
    )
    original_criterion = next(
        result for result in verdict.criterion_results
        if result.criterion_id == "critic_disposition"
    )
    assert normalized_criterion.outcome == "violated"
    assert normalized_criterion.reasoning != original_criterion.reasoning
    assert normalized.reasoning != verdict.reasoning
    assert {action["path"] for action in actions} == {
        "findings[0].disposition",
        "criterion_results[2].outcome",
        "criterion_results[2].reasoning",
        "reasoning",
    }
    assert validate_verdict_contract(
        "plan_judge",
        normalized,
        prior_findings=priors,
        current_artifact=plan_text,
    ) == ()
    again, second_actions = canonicalize_verdict_contract(
        "plan_judge",
        normalized,
        prior_findings=priors,
        current_artifact=plan_text,
    )
    assert again == normalized
    assert second_actions == ()


@pytest.mark.parametrize("target_kind", ("unknown", "duplicate"))
def test_contract_canonicalization_refuses_unsafe_multi_target_join(
    target_kind,
):
    under_ref = (
        "prior:critic:0:sha256-" + "a" * 64 + ":under-decomposed"
    )
    unsafe_target = (
        "prior:critic:1:sha256-" + "b" * 64 + ":unknown"
        if target_kind == "unknown"
        else under_ref
    )
    plan_text = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "combined",
                "directive": {
                    "slice_id": "combined",
                    "intent": "Change two coupled behaviors",
                },
            }],
        }],
    }, sort_keys=True)
    closure = Finding(
        id="unsafe-grouped-closure",
        severity="High",
        category="unresolved_critic",
        evidence=(Evidence(
            source_id="artifact",
            kind="directive",
            location="artifact",
            excerpt=plan_text,
        ),),
        suggested_action="REPARTITION_PROOF: {",
        source="llm",
        criterion_id="critic_disposition",
        confidence=0.9,
        disposition="addressed",
        targets=(under_ref, unsafe_target),
        proposal=Proposal(action="REPARTITION_PROOF: {"),
    )
    verdict = SimpleNamespace(
        decision="revise",
        findings=(closure,),
        criterion_results=(),
        reasoning="the grouped closure is unsafe",
    )
    priors = ({
        "id": "under-decomposed",
        "ref": under_ref,
        "source_id": "context:prior_critic_findings",
        "severity": "High",
        "disposition": "confirmed",
        "category": "under_decomposed",
        "targets": (),
        "payload": {
            "proposal": {
                "action": "Split the coupled work into independent slices.",
                "replacement": None,
            },
            "evidence": [],
        },
    },)

    normalized, actions = canonicalize_verdict_contract(
        "plan_judge",
        verdict,
        prior_findings=priors,
        current_artifact=plan_text,
    )

    assert normalized is verdict
    assert normalized.findings[0] == closure
    assert actions == ()


def test_qualified_refs_disambiguate_same_finding_id_across_rounds():
    refs = (
        "prior:red:0:sha256-" + "a" * 64 + ":duplicate",
        "prior:red:1:sha256-" + "b" * 64 + ":duplicate",
    )
    findings = tuple(
        SimpleNamespace(
            id=f"adjudicate-{index}",
            criterion_id="attack_execution",
            category="dismissed_attack",
            severity="Low",
            disposition="addressed",
            targets=(prior_ref,),
            evidence=(SimpleNamespace(source_id=prior_ref, kind="prior"),),
            proposal=SimpleNamespace(action="retain", replacement=None),
            attack_spec=None,
        )
        for index, prior_ref in enumerate(refs)
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(
            complete=True, sources_reviewed=("artifact", *refs),
        ),
        findings=findings,
        criterion_results=tuple(
            SimpleNamespace(
                criterion_id=criterion,
                outcome="satisfied",
                reasoning="Current attack evidence establishes this result.",
                evidence=(SimpleNamespace(source_id="artifact"),),
                headline=index == 0,
            )
            for index, criterion in enumerate(contract("breach_judge").criteria)
        ),
    )
    priors = tuple(
        {
            "id": "duplicate",
            "ref": prior_ref,
            "source_id": prior_ref,
            "severity": "High",
            "disposition": "confirmed",
        }
        for prior_ref in refs
    )

    errors = validate_verdict_contract(
        "breach_judge",
        verdict,
        prior_findings=priors,
        available_source_ids={"artifact", *refs},
    )
    assert len([
        error for error in errors if "prior-only evidence is insufficient" in error
    ]) == 2

    for finding in findings:
        finding.evidence = (
            *finding.evidence,
            SimpleNamespace(source_id="artifact", kind="current_state"),
        )
    assert validate_verdict_contract(
        "breach_judge",
        verdict,
        prior_findings=priors,
        available_source_ids={"artifact", *refs},
    ) == ()

    findings[1].targets = (refs[0],)
    errors = validate_verdict_contract(
        "breach_judge",
        verdict,
        prior_findings=priors,
        available_source_ids={"artifact", *refs},
    )
    assert any("got 2" in error for error in errors)
    assert any("got 0" in error for error in errors)


def test_runtime_contract_validator_rejects_unretained_tool_source():
    finding = SimpleNamespace(
        id="cause",
        criterion_id="causal_chain",
        category="root_cause",
        severity="Low",
        disposition="addressed",
        targets=(),
        evidence=(SimpleNamespace(source_id="tool:source-0001"),),
        proposal=None,
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", "tool:source-0001"),
        ),
        findings=(finding,),
    )
    errors = validate_verdict_contract(
        "spec_clarity", verdict, available_source_ids={"artifact"},
    )
    assert any("undeclared source_id" in error for error in errors)


def test_runtime_contract_validator_requires_declared_context_coverage():
    verdict = SimpleNamespace(
        decision="inconclusive",
        coverage=SimpleNamespace(
            complete=False,
            sources_reviewed=("artifact",),
        ),
        findings=(),
    )
    errors = validate_verdict_contract(
        "api_contract_fidelity", verdict, supplied_context={},
    )
    assert any("required context is missing" in error for error in errors)
    assert any("context:spec_intent" in error for error in errors)


def test_api_contract_replacement_and_paired_evidence_are_consistent():
    meta = contract("api_contract_fidelity")
    assert meta.output.replacement_default == "required"
    assert "proposal.replacement is required" in load(
        "api_contract_fidelity"
    ).system_prompt
    finding = SimpleNamespace(
        id="wrong-shape",
        criterion_id="api_shape",
        category="api_deviation",
        severity="High",
        disposition="confirmed",
        targets=(),
        evidence=(
            SimpleNamespace(source_id="artifact", kind="implemented_shape"),
            SimpleNamespace(source_id="context:spec_intent", kind="required_shape"),
        ),
        proposal=SimpleNamespace(
            action="match the required object shape",
            replacement="return {duration: elapsed}",
        ),
        attack_spec=None,
    )
    verdict = SimpleNamespace(
        decision="revise",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", "context:spec_intent"),
        ),
        findings=(finding,),
    )

    assert validate_verdict_contract(
        "api_contract_fidelity",
        verdict,
        supplied_context={"spec_intent": "return {duration: number}"},
    ) == ()


def test_planning_history_context_produces_typed_target_closure_inputs():
    history = [{
        "round": 0,
        "findings": [{
            "id": "critic-1",
            "severity": "High",
            "disposition": "confirmed",
        }],
    }]
    (finding,) = _context_prior_findings({
        "prior_critic_findings": json.dumps(history),
    })
    assert finding["id"] == "critic-1"
    assert finding["severity"] == "High"
    assert finding["source_id"] == "context:prior_critic_findings"
    assert finding["ref"].startswith("prior:prior_critic_findings:0:sha256-")


def test_planning_history_refs_stay_stable_when_later_role_records_append():
    critic = [{
        "round": 0,
        "findings": [{
            "id": "critic-1",
            "severity": "High",
            "disposition": "confirmed",
            "category": "under_decomposed",
            "targets": [],
        }],
    }]
    judge = [{
        "round": 0,
        "findings": [{
            "id": "judge-1",
            "severity": "High",
            "disposition": "confirmed",
            "category": "contradiction",
            "targets": [],
        }],
    }]
    initial = _context_prior_findings({
        "prior_critic_findings": json.dumps(critic),
        "prior_judge_findings": json.dumps(judge),
    })
    judge_ref = next(item["ref"] for item in initial if item["id"] == "judge-1")
    critic.append({
        "round": 1,
        "findings": [{
            "id": "critic-2",
            "severity": "High",
            "disposition": "confirmed",
            "category": "under_decomposed",
            "targets": [],
        }],
    })

    appended = _context_prior_findings({
        "prior_critic_findings": json.dumps(critic),
        "prior_judge_findings": json.dumps(judge),
    })

    assert next(
        item["ref"] for item in appended if item["id"] == "judge-1"
    ) == judge_ref


def test_active_prior_frontier_orders_lagging_roles_by_authored_round():
    critic = [
        {
            "round": 0,
            "findings": [{
                "id": "earlier-nonblocking",
                "severity": "Low",
                "disposition": "addressed",
                "targets": [],
            }],
        },
        {
            "round": 2,
            "findings": [{
                "id": "current-blocker",
                "severity": "High",
                "disposition": "confirmed",
                "targets": [],
            }],
        },
    ]
    critic_only = _context_prior_findings({
        "prior_critic_findings": json.dumps(critic),
        "prior_judge_findings": "[]",
    })
    blocker_ref = next(
        item["ref"] for item in critic_only
        if item["id"] == "current-blocker"
    )
    judge = [{
        "round": 2,
        "findings": [{
            "id": "current-adjudication",
            "severity": "High",
            "disposition": "addressed",
            "targets": [blocker_ref],
        }],
    }]
    history = _context_prior_findings({
        "prior_critic_findings": json.dumps(critic),
        "prior_judge_findings": json.dumps(judge),
    })

    assert [
        (item["history_round"], item["role"], item["id"])
        for item in history
    ] == [
        (0, "prior_critic_findings", "earlier-nonblocking"),
        (2, "prior_critic_findings", "current-blocker"),
        (2, "prior_judge_findings", "current-adjudication"),
    ]
    assert active_prior_frontier(history, severity_floor="High") == ()


def test_active_prior_frontier_compacts_terminal_lineage_acknowledgements():
    critic_ref = "prior:critic:0:sha256-" + "a" * 64 + ":under-decomposed"
    judge_ref = "prior:judge:0:sha256-" + "b" * 64 + ":adjudication"
    next_ref = "prior:critic:1:sha256-" + "c" * 64 + ":current-blocker"
    history = (
        {
            "ref": critic_ref,
            "role": "critic",
            "round_index": 0,
            "severity": "High",
            "disposition": "confirmed",
            "targets": (),
        },
        {
            "ref": judge_ref,
            "role": "judge",
            "round_index": 0,
            "severity": "High",
            "disposition": "addressed",
            "targets": (critic_ref,),
        },
        {
            "ref": next_ref,
            "role": "critic",
            "round_index": 1,
            "severity": "High",
            "disposition": "confirmed",
            "targets": (),
        },
    )

    frontier = active_prior_frontier(history, severity_floor="High")

    assert tuple(item["ref"] for item in frontier) == (next_ref,)


def test_active_prior_frontier_carries_unresolved_descendant_not_ancestor():
    ancestor = "prior:critic:0:sha256-" + "a" * 64 + ":old"
    descendant = "prior:judge:0:sha256-" + "b" * 64 + ":still-active"
    frontier = active_prior_frontier((
        {
            "ref": ancestor,
            "role": "critic",
            "round_index": 0,
            "severity": "High",
            "disposition": "confirmed",
            "targets": (),
        },
        {
            "ref": descendant,
            "role": "judge",
            "round_index": 0,
            "severity": "High",
            "disposition": "confirmed",
            "targets": (ancestor,),
        },
    ), severity_floor="High")

    assert tuple(item["ref"] for item in frontier) == (descendant,)


def test_active_prior_frontier_does_not_downgrade_material_ancestor():
    ancestor = "prior:critic:0:sha256-" + "a" * 64 + ":high"
    low_descendant = "prior:judge:0:sha256-" + "b" * 64 + ":low"
    frontier = active_prior_frontier((
        {
            "ref": ancestor,
            "role": "critic",
            "round_index": 0,
            "severity": "High",
            "disposition": "confirmed",
            "targets": (),
        },
        {
            "ref": low_descendant,
            "role": "judge",
            "round_index": 0,
            "severity": "Low",
            "disposition": "confirmed",
            "targets": (ancestor,),
        },
    ), severity_floor="High")

    assert tuple(item["ref"] for item in frontier) == (ancestor,)


def test_lower_closure_floor_does_not_downgrade_blocking_ancestor():
    ancestor = "prior:breaker:0:sha256-" + "a" * 64 + ":high"
    descendant = "prior:judge:1:sha256-" + "b" * 64 + ":medium"
    frontier = active_prior_frontier((
        {
            "ref": ancestor,
            "role": "breaker",
            "round_index": 0,
            "severity": "High",
            "disposition": "confirmed",
            "targets": (),
        },
        {
            "ref": descendant,
            "role": "judge",
            "round_index": 1,
            "severity": "Medium",
            "disposition": "confirmed",
            "targets": (ancestor,),
        },
    ), severity_floor="High", closure_severity_floor="Medium")

    assert tuple(item["ref"] for item in frontier) == (
        ancestor,
        descendant,
    )


def test_lower_closure_floor_compacts_nonblocking_lineage():
    ancestor = "prior:breaker:0:sha256-" + "a" * 64 + ":medium-old"
    descendant = "prior:judge:1:sha256-" + "b" * 64 + ":medium-current"
    frontier = active_prior_frontier((
        {
            "ref": ancestor,
            "role": "breaker",
            "round_index": 0,
            "severity": "Medium",
            "disposition": "confirmed",
            "targets": (),
        },
        {
            "ref": descendant,
            "role": "judge",
            "round_index": 1,
            "severity": "Medium",
            "disposition": "confirmed",
            "targets": (ancestor,),
        },
    ), severity_floor="High", closure_severity_floor="Medium")

    assert tuple(item["ref"] for item in frontier) == (descendant,)


def test_lower_closure_floor_addressed_descendant_closes_blocker():
    ancestor = "prior:breaker:0:sha256-" + "a" * 64 + ":high"
    descendant = "prior:judge:1:sha256-" + "b" * 64 + ":addressed"
    frontier = active_prior_frontier((
        {
            "ref": ancestor,
            "role": "breaker",
            "round_index": 0,
            "severity": "High",
            "disposition": "confirmed",
            "targets": (),
        },
        {
            "ref": descendant,
            "role": "judge",
            "round_index": 1,
            "severity": "Medium",
            "disposition": "addressed",
            "targets": (ancestor,),
        },
    ), severity_floor="High", closure_severity_floor="Medium")

    assert frontier == ()


def test_dual_floor_frontier_preserves_malformed_and_duplicate_records_once():
    duplicate_ref = "prior:breaker:0:sha256-" + "a" * 64 + ":duplicate"
    malformed = {
        "role": "breaker",
        "round_index": 0,
        "severity": "Medium",
        "disposition": "confirmed",
        "targets": (),
    }
    duplicate_a = {
        **malformed,
        "ref": duplicate_ref,
    }
    duplicate_b = dict(duplicate_a)

    frontier = active_prior_frontier(
        (malformed, duplicate_a, duplicate_b),
        severity_floor="High",
        closure_severity_floor="Medium",
    )

    assert len(frontier) == 3
    assert frontier[0] is malformed
    assert frontier[1] is duplicate_a
    assert frontier[2] is duplicate_b


def test_dual_floor_frontier_keeps_lower_floor_deterministic_failure():
    deterministic_ref = (
        "prior:breaker:0:sha256-" + "a" * 64 + ":deterministic"
    )
    deterministic = {
        "ref": deterministic_ref,
        "role": "breaker",
        "round_index": 0,
        "severity": "Medium",
        "disposition": "confirmed",
        "targets": (),
        "source": "deterministic",
    }
    addressed = {
        "ref": "prior:judge:1:sha256-" + "b" * 64 + ":addressed",
        "role": "judge",
        "round_index": 1,
        "severity": "Medium",
        "disposition": "addressed",
        "targets": (deterministic_ref,),
        "source": "llm",
    }

    assert active_prior_frontier(
        (deterministic, addressed),
        severity_floor="High",
        closure_severity_floor="Medium",
    ) == (deterministic,)
    assert active_prior_frontier(
        (deterministic, addressed),
        severity_floor="High",
        closure_severity_floor="Medium",
        verified_deterministic_closures=frozenset({deterministic_ref}),
    ) == ()


def test_active_prior_frontier_preserves_duplicate_refs_for_diagnostics():
    duplicate_ref = "prior:red:0:sha256-" + "a" * 64 + ":duplicate"
    duplicate = {
        "id": "duplicate",
        "ref": duplicate_ref,
        "role": "red",
        "round_index": 0,
        "severity": "High",
        "disposition": "confirmed",
        "source_id": duplicate_ref,
        "targets": (),
    }
    history = (duplicate, dict(duplicate))

    assert active_prior_frontier(
        history, severity_floor="High",
    ) == history

    verdict = SimpleNamespace(
        decision="pass",
        coverage=SimpleNamespace(
            complete=True,
            sources_reviewed=("artifact", duplicate_ref),
        ),
        findings=(),
    )
    errors = validate_verdict_contract(
        "breach_judge",
        verdict,
        prior_findings=history,
        available_source_ids={"artifact", duplicate_ref},
    )
    assert any("duplicate qualified prior ref" in error for error in errors)


def test_load_unknown_rubric_raises():
    with pytest.raises(FileNotFoundError):
        load("does_not_exist")
