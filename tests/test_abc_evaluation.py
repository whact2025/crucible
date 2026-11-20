# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from crucible.backends.config import lint_routing_config, routing_fingerprint
from crucible.workflows import deepswe, swebench
from crucible.workflows._abc_evaluation import (
    E12ArmSet,
    E12CommonBinding,
    E12Lane,
    E12PreflightError,
    E12Qualification,
    E12Result,
    E12Task,
    E12ValidationError,
    ATTEMPT_MANIFEST_SCHEMA,
    COHORT_MANIFEST_SCHEMA,
    RESULT_SCHEMA,
    build_e12_report,
    counterbalanced_assignments,
    freeze_e12_cohort,
    ingest_e12_results,
    load_e12_arm_set,
    preflight_e12,
    validate_e12_execution_boundary,
    validate_e12_result,
)


_DIGEST = "sha256:" + "1" * 64
_SOURCE_REVISIONS = {"crucible": "abc123", "assay": "def456"}
_ENDPOINTS = {
    "A": "codex-cli://workspace",
    "B": "claude-cli://organization-session",
    "C": "https://api.moonshot.ai/v1",
}


def _qualifications(
    arm_set: E12ArmSet,
    *,
    endpoints: dict[str, str] | None = None,
) -> dict[str, E12Qualification]:
    endpoints = endpoints or _ENDPOINTS
    return {
        arm: E12Qualification(
            available=True,
            promotable=True,
            requested_endpoint=endpoints[arm],
            served_endpoint=endpoints[arm],
            served_model=arm_set.lanes[arm].model,
            requested_effort=arm_set.lanes[arm].effort,
            deployment_revision=f"deployment-{arm}",
            immutable_deployment_identity=True,
            adapter_version=f"adapter-{arm}-1",
            runtime_version=f"runtime-{arm}-1",
            pricing_digest=_DIGEST,
            feature_profile_digest=arm_set.lanes[arm].feature_fingerprint,
            routing_fingerprint=arm_set.lanes[arm].routing_fingerprint,
            conformance_digest=_DIGEST,
            capability_probe_digest=_DIGEST,
            prompt_profile_digest=_DIGEST,
            tool_profile_digest=_DIGEST,
            policy_profile_digest=_DIGEST,
            source_revisions=_SOURCE_REVISIONS,
            cost_comparable=arm != "C",
        )
        for arm in ("A", "B", "C")
    }


def _build_qualified_arm_set(
    root: Path,
    *,
    product: str = "swebench",
) -> E12ArmSet:
    """Build a canonical, hermetic qualified fixture from copied routes."""

    source = (
        Path(__file__).resolve().parents[1]
        / "products"
        / product
    )
    destination = root / "qualified-e12"
    destination.mkdir(parents=True)
    for filename in (
        "backends_codex_only.yaml",
        "backends_claude_opus5_org.yaml",
        "backends_kimi_sdk.yaml",
    ):
        shutil.copy2(source / filename, destination / filename)
    manifest = (source / "e12_arms.yaml").read_text(encoding="utf-8")
    assert manifest.count("promotion_gate: blocked") == 2
    manifest_path = destination / "e12_arms.yaml"
    manifest_path.write_text(
        manifest.replace(
            "promotion_gate: blocked",
            "promotion_gate: qualified",
        ),
        encoding="utf-8",
    )
    return load_e12_arm_set(manifest_path)


@pytest.fixture(autouse=True)
def exact_moonshot_endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KIMI_BASE_URL", _ENDPOINTS["C"])


@pytest.fixture
def qualified_arm_set(tmp_path: Path) -> E12ArmSet:
    return _build_qualified_arm_set(tmp_path)


def _common():
    return E12CommonBinding(
        task_manifest_digest=_DIGEST,
        prompt_abi_digest=_DIGEST,
        semantic_prompt_digest=_DIGEST,
        schema_digest=_DIGEST,
        rubric_digest=_DIGEST,
        tool_schema_digest=_DIGEST,
        execution_policy_digest=_DIGEST,
        hidden_evaluation_policy_digest=_DIGEST,
        source_revisions=_SOURCE_REVISIONS,
        pricing_policy_revision="pricing-v1",
    )


def _tasks(count: int = 4, *, replicates: int = 3):
    return [
        E12Task(
            task_id=f"task-{index}",
            task_digest=_DIGEST,
            image_digest=_DIGEST,
            base_commit=f"commit-{index}",
            workspace_fingerprint=_DIGEST,
            stratum=f"repo-{index % 2}",
            replicates=replicates,
        )
        for index in range(count)
    ]


def _record(attempt, *, resolved: bool, **overrides):
    qualification = attempt.qualification
    value = {
        "schema_version": RESULT_SCHEMA,
        "task_id": attempt.task.task_id,
        "replicate": attempt.replicate,
        "arm": attempt.arm,
        "assignment": attempt.assignment,
        "attempt_manifest_digest": attempt.digest,
        "execution_identity_digest": attempt.execution_identity_digest,
        "terminal_status": "resolved" if resolved else "unresolved",
        "terminal_event_count": 1,
        "resolved": resolved,
        "usage_complete": True,
        "quota_complete": True,
        "trace_complete": True,
        "fallback_used": False,
        "hard_safety_counts": {
            "hidden_data_access": 0,
            "capability_escalation": 0,
            "shared_checkout_concurrent_writer": 0,
            "scope_escape_surviving_rollback": 0,
            "unresolved_mutation_journal": 0,
            "candidate_verifier_hash_mismatch": 0,
            "unaccounted_cost_or_missing_terminal_trace": 0,
            "evidence_source_target_closure": 0,
            "cleanup_or_recovery_failure": 0,
        },
        "served_model": qualification.served_model,
        "served_endpoint": qualification.served_endpoint,
        "deployment_revision": qualification.deployment_revision,
        "wall_time_seconds": 10.0 + attempt.order_index,
        "input_tokens": 100,
        "output_tokens": 20,
        "cost_usd": 1.0 if qualification.cost_comparable else 0.0,
        "cost_comparable": qualification.cost_comparable,
        "diagnostic_only": False,
    }
    value.update(overrides)
    return value


def test_checked_in_arm_sets_are_exact_and_kimi_is_preflight_blocked():
    swe = swebench.load_e12_arm_set_for_swebench()
    deep = deepswe.load_e12_arm_set_for_deepswe()

    assert [swe.lanes[arm].backend for arm in ("A", "B", "C")] == [
        "codex",
        "claude",
        "kimi",
    ]
    assert swe.lanes["A"].role_efforts == {}
    assert deep.lanes["A"].role_efforts == {
        "architect_planner": "high",
        "architect_critic": "high",
        "architect_judge": "high",
    }
    assert swe.lanes["B"].role_efforts == {}
    assert deep.lanes["B"].role_efforts == {
        "architect_planner": "medium",
        "architect_critic": "medium",
        "architect_judge": "medium",
    }
    deep_codex_route = lint_routing_config(deep.lanes["A"].routing_path)
    assert deep.lanes["A"].routing_fingerprint == routing_fingerprint(
        deep_codex_route
    )
    for role in deep.lanes["A"].role_efforts:
        config = deep_codex_route.for_role(role)
        assert config.backend == "codex"
        assert config.model == "gpt-5.6-sol"
        assert config.effort == "high"
    for role in ("implementer", "breach_defender", "impl_breaker"):
        assert deep_codex_route.for_role(role).effort == "max"
    deep_claude_route = lint_routing_config(deep.lanes["B"].routing_path)
    assert deep.lanes["B"].routing_fingerprint == routing_fingerprint(
        deep_claude_route
    )
    for role in deep.lanes["B"].role_efforts:
        config = deep_claude_route.for_role(role)
        assert config.backend == "claude"
        assert config.model == "claude-opus-5"
        assert config.effort == "medium"
    assert deep_claude_route.for_role("implementer").effort == "high"
    for arm_set in (swe, deep):
        route = arm_set.lanes["A"].routing_path.read_text(encoding="utf-8")
        assert "credential_source: chatgpt_session" in route
        assert "INFER_" not in route
        assert "base_url:" not in route
        assert "api_key:" not in route
        assert arm_set.lanes["B"].model == "claude-opus-5"
        assert arm_set.lanes["B"].endpoint_template == _ENDPOINTS["B"]
        route = arm_set.lanes["B"].routing_path.read_text(encoding="utf-8")
        assert "credential_source: claude_session" in route
        assert (
            "expected_org_id: 76f2dd92-7d56-4550-afaf-abd363e3235c"
            in route
        )
        assert (
            "expected_org_name: NVIDIA Corporation - Power Users"
            in route
        )
        assert "INFER_" not in route
        assert "base_url:" not in route
        assert "api_key:" not in route
    for arm_set in (swe, deep):
        assert arm_set.lanes["C"].model == "kimi-k3.1-pro"
        assert arm_set.lanes["C"].effort == "max"
        assert arm_set.lanes["C"].quality_setting == "max"
        route_text = arm_set.lanes["C"].routing_path.read_text(encoding="utf-8")
        assert "base_url: ${KIMI_BASE_URL}" in route_text
        assert "api_key: ${MOONSHOT_API_KEY}" in route_text
        assert "worker_source: project_locked" in route_text
        assert "INFER_" not in route_text
        kimi_route = lint_routing_config(arm_set.lanes["C"].routing_path)
        assert kimi_route.for_role("architect_planner").extra == {
            "worker_source": "project_locked",
            "max_response_tokens": 16_384,
        }
        expected_implementer_extra = {
            "worker_source": "project_locked",
        }
        if arm_set.workflow == "deepswe":
            expected_implementer_extra["timeout_seconds"] = 1_200
        assert (
            kimi_route.for_role("implementer").extra
            == expected_implementer_extra
        )
    for arm_set in (swe, deep):
        assert arm_set.lanes["A"].promotion_gate == "blocked"
        assert arm_set.lanes["A"].gate_ticket == "CDX-01-MAX"
        assert arm_set.lanes["C"].promotion_gate == "blocked"
    with pytest.raises(E12PreflightError, match="CDX-01-MAX"):
        preflight_e12(swe, _qualifications(swe))


@pytest.mark.parametrize("product", ["swebench", "deepswe"])
def test_checked_in_arm_c_is_refused_by_e12_preflight_whatever_is_qualified(
    product: str,
):
    # Arm C routes the strict Kimi profile. Inside E12 its gate refuses the
    # lane before a cohort can be frozen, even when a complete, promotable
    # qualification is supplied, so no E12 attempt reaches that route.
    arm_set = (
        swebench.load_e12_arm_set_for_swebench()
        if product == "swebench"
        else deepswe.load_e12_arm_set_for_deepswe()
    )
    lane = arm_set.lanes["C"]
    assert (lane.backend, lane.model, lane.effort) == ("kimi", "kimi-k3.1-pro", "max")
    assert (lane.promotion_gate, lane.gate_ticket) == ("blocked", "KIM-03")
    with pytest.raises(E12PreflightError) as refused:
        preflight_e12(arm_set, _qualifications(arm_set))
    assert "KIM-03 has not promoted this lane" in refused.value.blockers["C"]
    with pytest.raises(E12PreflightError) as unqualified:
        preflight_e12(arm_set, {})
    assert "KIM-03 has not promoted this lane" in unqualified.value.blockers["C"]


def test_deepswe_claude_role_effort_overlay_is_exact_and_drift_closed(
    tmp_path: Path,
):
    source = (
        Path(__file__).resolve().parents[1]
        / "products"
        / "deepswe"
    )
    destination = tmp_path / "deepswe-role-efforts"
    destination.mkdir()
    filenames = (
        "e12_arms.yaml",
        "backends_codex_only.yaml",
        "backends_claude_opus5_org.yaml",
        "backends_kimi_sdk.yaml",
    )
    for filename in filenames:
        shutil.copy2(source / filename, destination / filename)

    manifest_path = destination / "e12_arms.yaml"
    route_path = destination / "backends_claude_opus5_org.yaml"
    baseline_route = route_path.read_text(encoding="utf-8")
    route_path.write_text(
        baseline_route.replace(
            "architect_critic:\n    effort: medium",
            "architect_critic:\n    effort: high",
        ),
        encoding="utf-8",
    )
    with pytest.raises(E12ValidationError, match="architect_critic.*drifts"):
        load_e12_arm_set(manifest_path)

    route_path.write_text(baseline_route, encoding="utf-8")
    baseline_manifest = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        baseline_manifest.replace(
            "architect_critic: medium",
            "architect_critic: low",
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        E12ValidationError,
        match="role_efforts must be exact frozen overrides",
    ):
        load_e12_arm_set(manifest_path)


def test_deepswe_codex_role_effort_overlay_is_exact_and_drift_closed(
    tmp_path: Path,
):
    source = (
        Path(__file__).resolve().parents[1]
        / "products"
        / "deepswe"
    )
    destination = tmp_path / "deepswe-codex-role-efforts"
    destination.mkdir()
    filenames = (
        "e12_arms.yaml",
        "backends_codex_only.yaml",
        "backends_claude_opus5_org.yaml",
        "backends_kimi_sdk.yaml",
    )
    for filename in filenames:
        shutil.copy2(source / filename, destination / filename)

    manifest_path = destination / "e12_arms.yaml"
    route_path = destination / "backends_codex_only.yaml"
    baseline_route = route_path.read_text(encoding="utf-8")
    route_path.write_text(
        baseline_route.replace(
            "architect_critic:\n"
            "    backend: codex\n"
            "    model: gpt-5.6-sol\n"
            "    effort: high",
            "architect_critic:\n"
            "    backend: codex\n"
            "    model: gpt-5.6-sol\n"
            "    effort: ultra",
        ),
        encoding="utf-8",
    )
    with pytest.raises(E12ValidationError, match="architect_critic.*drifts"):
        load_e12_arm_set(manifest_path)

    route_path.write_text(baseline_route, encoding="utf-8")
    baseline_manifest = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        baseline_manifest.replace(
            "architect_critic: high",
            "architect_critic: low",
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        E12ValidationError,
        match="role_efforts must be exact frozen overrides",
    ):
        load_e12_arm_set(manifest_path)


def test_e12_kimi_planner_response_limit_is_exact_and_role_scoped(
    tmp_path: Path,
):
    arm_set = _build_qualified_arm_set(tmp_path)
    route_path = arm_set.lanes["C"].routing_path
    baseline = route_path.read_text(encoding="utf-8")

    route = yaml.safe_load(baseline)
    route["roles"]["architect_planner"]["extra"]["max_response_tokens"] = 8192
    route_path.write_text(yaml.safe_dump(route, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        E12ValidationError,
        match="architect_planner.*exact Kimi routing extra",
    ):
        load_e12_arm_set(arm_set.source_path)

    route = yaml.safe_load(baseline)
    route["roles"]["implementer"]["extra"]["max_response_tokens"] = 16384
    route_path.write_text(yaml.safe_dump(route, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        E12ValidationError,
        match="implementer.*exact Kimi routing extra",
    ):
        load_e12_arm_set(arm_set.source_path)


def test_deepswe_e12_kimi_implementer_timeout_is_exact_and_role_scoped(
    tmp_path: Path,
):
    arm_set = _build_qualified_arm_set(tmp_path, product="deepswe")
    route_path = arm_set.lanes["C"].routing_path
    baseline = route_path.read_text(encoding="utf-8")

    route_path.write_text(
        baseline.replace("timeout_seconds: 1200", "timeout_seconds: 600", 1),
        encoding="utf-8",
    )
    with pytest.raises(
        E12ValidationError,
        match="implementer.*exact Kimi routing extra",
    ):
        load_e12_arm_set(arm_set.source_path)


def test_in_memory_gate_replacement_is_rejected_by_preflight_and_freeze(
    tmp_path: Path,
):
    arm_set = swebench.load_e12_arm_set_for_swebench()
    forged = dataclasses.replace(
        arm_set,
        lanes={
            **arm_set.lanes,
            "C": dataclasses.replace(
                arm_set.lanes["C"],
                promotion_gate="qualified",
            ),
        },
    )

    with pytest.raises(
        E12ValidationError,
        match="arm C promotion_gate",
    ):
        preflight_e12(forged, _qualifications(forged))
    with pytest.raises(
        E12ValidationError,
        match="arm C promotion_gate",
    ):
        freeze_e12_cohort(
            root=tmp_path / "forged-cohort",
            arm_set=forged,
            qualifications=_qualifications(forged),
            common=_common(),
            tasks=_tasks(1),
            cohort="forged-gate",
            phase="synthetic",
            assignment_seed="seed",
            materialize_state_roots=False,
        )


def test_arm_set_and_lanes_require_exact_runtime_types(
    qualified_arm_set: E12ArmSet,
):
    class ArmSetSubclass(E12ArmSet):
        pass

    class LaneSubclass(E12Lane):
        pass

    arm_subclass = ArmSetSubclass(
        **{
            field.name: getattr(qualified_arm_set, field.name)
            for field in dataclasses.fields(E12ArmSet)
        }
    )
    with pytest.raises(E12ValidationError, match="exact E12ArmSet"):
        preflight_e12(arm_subclass, _qualifications(qualified_arm_set))

    lane = qualified_arm_set.lanes["C"]
    lane_subclass = LaneSubclass(
        **{
            field.name: getattr(lane, field.name)
            for field in dataclasses.fields(E12Lane)
        }
    )
    forged = dataclasses.replace(
        qualified_arm_set,
        lanes={**qualified_arm_set.lanes, "C": lane_subclass},
    )
    with pytest.raises(E12ValidationError, match="exact E12Lane"):
        preflight_e12(forged, _qualifications(qualified_arm_set))


def test_arm_set_rejects_forged_source_path_and_digest(
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    forged_digest = dataclasses.replace(
        qualified_arm_set,
        source_digest=_DIGEST,
    )
    with pytest.raises(E12ValidationError, match="source bytes/digest"):
        preflight_e12(forged_digest, _qualifications(qualified_arm_set))

    alternate = tmp_path / "alternate-source"
    shutil.copytree(qualified_arm_set.source_path.parent, alternate)
    forged_source = dataclasses.replace(
        qualified_arm_set,
        source_path=alternate / "e12_arms.yaml",
    )
    with pytest.raises(E12ValidationError, match="routing_path"):
        preflight_e12(forged_source, _qualifications(qualified_arm_set))


def test_route_file_drift_after_load_is_rejected(
    qualified_arm_set: E12ArmSet,
):
    route_path = qualified_arm_set.lanes["B"].routing_path
    route = route_path.read_text(encoding="utf-8")
    route_path.write_text(
        route + "\n# post-load routing byte drift\n",
        encoding="utf-8",
    )

    with pytest.raises(
        E12ValidationError,
        match="routing_(digest|fingerprint)",
    ):
        preflight_e12(
            qualified_arm_set,
            _qualifications(qualified_arm_set),
        )


def test_preflight_rejects_qualification_from_a_different_route(
    qualified_arm_set: E12ArmSet,
):
    qualifications = _qualifications(qualified_arm_set)
    qualifications["A"] = dataclasses.replace(
        qualifications["A"],
        routing_fingerprint=_DIGEST,
    )

    with pytest.raises(
        E12PreflightError,
        match="frozen routing fingerprint",
    ):
        preflight_e12(qualified_arm_set, qualifications)


def test_preflight_rejects_retained_ultra_qualification_for_max_lane(
    qualified_arm_set: E12ArmSet,
):
    qualifications = _qualifications(qualified_arm_set)
    qualifications["A"] = dataclasses.replace(
        qualifications["A"],
        requested_effort="ultra",
    )

    with pytest.raises(E12PreflightError, match="qualified effort 'ultra'"):
        preflight_e12(qualified_arm_set, qualifications)


def test_e12_quality_setting_is_exact_and_cannot_relabel_route_effort(
    tmp_path: Path,
):
    arm_set = _build_qualified_arm_set(tmp_path / "quality-setting")
    manifest_path = arm_set.source_path
    manifest = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest.replace("quality_setting: max", "quality_setting: ultra", 1),
        encoding="utf-8",
    )

    with pytest.raises(E12ValidationError, match="quality_setting.*max"):
        load_e12_arm_set(manifest_path)


def test_preflight_accepts_exact_first_party_claude_session_identity(
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    arm_set = qualified_arm_set
    qualifications = _qualifications(arm_set)

    preflight_e12(arm_set, qualifications)
    cohort = freeze_e12_cohort(
        root=tmp_path / "cohort",
        arm_set=arm_set,
        qualifications=qualifications,
        common=_common(),
        tasks=_tasks(1),
        cohort="claude-session",
        phase="synthetic",
        assignment_seed="seed",
    )
    payload = json.loads(cohort.manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == COHORT_MANIFEST_SCHEMA

    assert payload["arms"]["B"]["lane"]["model"] == "claude-opus-5"
    assert (
        payload["arms"]["B"]["lane"]["endpoint_template"]
        == "claude-cli://organization-session"
    )
    assert (
        payload["arms"]["B"]["qualification"]["served_model"]
        == "claude-opus-5"
    )
    assert (
        payload["arms"]["B"]["qualification"]["served_endpoint"]
        == "claude-cli://organization-session"
    )


def test_arm_a_route_rejects_non_chatgpt_credential_source(tmp_path: Path):
    arm_set = _build_qualified_arm_set(tmp_path / "arm-a-route")
    route_path = arm_set.lanes["A"].routing_path
    route = route_path.read_text(encoding="utf-8")
    assert route.count("credential_source: chatgpt_session") == 1
    route_path.write_text(
        route.replace(
            "credential_source: chatgpt_session",
            "credential_source: environment",
        ),
        encoding="utf-8",
    )

    with pytest.raises(E12ValidationError, match="exact Codex routing extra"):
        load_e12_arm_set(arm_set.source_path)


@pytest.mark.parametrize(
    "needle,replacement,match",
    [
        (
            "credential_source: claude_session",
            "credential_source: inference_gateway",
            "credential_source must be one of",
        ),
        (
            "  effort: high",
            "  effort: high\n  base_url: https://gateway.example.test/v1",
            "cannot be combined with base_url or api_key",
        ),
        (
            "  effort: high",
            "  effort: high\n  api_key: forbidden-test-token",
            "cannot be combined with base_url or api_key",
        ),
    ],
)
def test_arm_b_route_rejects_non_session_credentials_and_endpoints(
    needle: str,
    replacement: str,
    match: str,
    tmp_path: Path,
):
    arm_set = _build_qualified_arm_set(tmp_path / "arm-b-route")
    route_path = arm_set.lanes["B"].routing_path
    route = route_path.read_text(encoding="utf-8")
    assert route.count(needle) == 1
    route_path.write_text(
        route.replace(needle, replacement),
        encoding="utf-8",
    )

    with pytest.raises(E12ValidationError, match=match):
        load_e12_arm_set(arm_set.source_path)


@pytest.mark.parametrize(
    "needle,replacement,match",
    [
        (
            "base_url: ${KIMI_BASE_URL}",
            "base_url: ${INFER_BASE_URL}",
            "exact direct Moonshot endpoint template",
        ),
        (
            "api_key: ${MOONSHOT_API_KEY}",
            "api_key: ${INFER_AUTH_TOKEN}",
            "exact direct Moonshot credential template",
        ),
        (
            "worker_source: project_locked",
            "worker_source: environment",
            "exact Kimi routing extra",
        ),
    ],
)
def test_arm_c_route_rejects_inference_gateway_templates(
    needle: str,
    replacement: str,
    match: str,
    tmp_path: Path,
):
    arm_set = _build_qualified_arm_set(tmp_path / "arm-c-route")
    route_path = arm_set.lanes["C"].routing_path
    route = route_path.read_text(encoding="utf-8")
    assert needle in route
    route_path.write_text(
        route.replace(needle, replacement),
        encoding="utf-8",
    )

    with pytest.raises(E12ValidationError, match=match):
        load_e12_arm_set(arm_set.source_path)


@pytest.mark.parametrize(
    "served_model",
    [
        "aws/anthropic/bedrock-claude-opus-5",
        "claude-sonnet-4-6",
        "attacker/claude-opus-5",
    ],
)
def test_preflight_rejects_claude_model_drift_and_adversarial_prefix(
    served_model,
    qualified_arm_set: E12ArmSet,
):
    arm_set = qualified_arm_set
    qualifications = _qualifications(arm_set)
    qualifications["B"] = dataclasses.replace(
        qualifications["B"],
        served_model=served_model,
    )

    with pytest.raises(E12PreflightError, match="served model"):
        preflight_e12(arm_set, qualifications)


def test_preflight_rejects_highspeed_kimi_as_arm_c_fallback(
    qualified_arm_set: E12ArmSet,
):
    qualifications = _qualifications(qualified_arm_set)
    qualifications["C"] = dataclasses.replace(
        qualifications["C"],
        served_model="kimi-k3.1-pro-highspeed",
    )

    with pytest.raises(E12PreflightError, match="served model"):
        preflight_e12(qualified_arm_set, qualifications)


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("available", 1),
        ("promotable", "yes"),
        ("immutable_deployment_identity", 1),
        ("cost_comparable", "true"),
    ],
)
def test_qualification_booleans_require_exact_bool(
    field_name,
    value,
    qualified_arm_set: E12ArmSet,
):
    arm_set = qualified_arm_set
    qualification = _qualifications(arm_set)["A"]

    with pytest.raises(E12ValidationError, match="exact boolean"):
        dataclasses.replace(qualification, **{field_name: value})


@pytest.mark.parametrize("value", [1, " max "])
def test_qualification_requested_effort_requires_exact_canonical_string(
    value,
    qualified_arm_set: E12ArmSet,
):
    qualification = _qualifications(qualified_arm_set)["A"]

    with pytest.raises(E12ValidationError, match="requested_effort"):
        dataclasses.replace(qualification, requested_effort=value)


@pytest.mark.parametrize("kind", ["structural", "subclass"])
def test_preflight_rejects_nonexact_qualification_values(
    kind: str,
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    qualifications: dict[str, Any] = _qualifications(qualified_arm_set)
    baseline = qualifications["A"]
    values = {
        field.name: getattr(baseline, field.name)
        for field in dataclasses.fields(E12Qualification)
    }
    if kind == "structural":
        qualifications["A"] = SimpleNamespace(**values)
    else:
        class QualificationSubclass(E12Qualification):
            pass

        qualifications["A"] = QualificationSubclass(**values)

    with pytest.raises(E12ValidationError, match="exact E12Qualification"):
        preflight_e12(qualified_arm_set, qualifications)
    with pytest.raises(E12ValidationError, match="exact E12Qualification"):
        freeze_e12_cohort(
            root=tmp_path / f"qualification-{kind}",
            arm_set=qualified_arm_set,
            qualifications=qualifications,
            common=_common(),
            tasks=_tasks(1),
            cohort=f"qualification-{kind}",
            phase="synthetic",
            assignment_seed="seed",
            materialize_state_roots=False,
        )


def test_preflight_revalidates_qualification_at_use_time(
    qualified_arm_set: E12ArmSet,
):
    qualifications = _qualifications(qualified_arm_set)
    object.__setattr__(qualifications["A"], "promotable", 1)

    with pytest.raises(E12ValidationError, match="exact boolean"):
        preflight_e12(qualified_arm_set, qualifications)


def test_freeze_requires_exact_common_and_task_types(
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    class CommonSubclass(E12CommonBinding):
        pass

    class TaskSubclass(E12Task):
        pass

    common = _common()
    common_subclass = CommonSubclass(
        **{
            field.name: getattr(common, field.name)
            for field in dataclasses.fields(E12CommonBinding)
        }
    )
    with pytest.raises(E12ValidationError, match="exact E12CommonBinding"):
        freeze_e12_cohort(
            root=tmp_path / "common-subclass",
            arm_set=qualified_arm_set,
            qualifications=_qualifications(qualified_arm_set),
            common=common_subclass,
            tasks=_tasks(1),
            cohort="common-subclass",
            phase="synthetic",
            assignment_seed="seed",
        )

    task = _tasks(1)[0]
    task_subclass = TaskSubclass(
        **{
            field.name: getattr(task, field.name)
            for field in dataclasses.fields(E12Task)
        }
    )
    with pytest.raises(E12ValidationError, match=r"tasks\[0\].*exact E12Task"):
        freeze_e12_cohort(
            root=tmp_path / "task-subclass",
            arm_set=qualified_arm_set,
            qualifications=_qualifications(qualified_arm_set),
            common=common,
            tasks=[task_subclass],
            cohort="task-subclass",
            phase="synthetic",
            assignment_seed="seed",
        )


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_resolved_endpoint_must_match_qualification_request(
    arm: str,
    qualified_arm_set: E12ArmSet,
):
    qualifications = _qualifications(qualified_arm_set)
    qualifications[arm] = dataclasses.replace(
        qualifications[arm],
        requested_endpoint="https://other.example.test/v1",
        served_endpoint="https://other.example.test/v1",
    )

    with pytest.raises(
        E12PreflightError,
        match="resolved lane endpoint",
    ):
        preflight_e12(qualified_arm_set, qualifications)


def test_endpoint_env_is_resolved_without_reading_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    arm_set = _build_qualified_arm_set(tmp_path / "env-routes")
    qualifications = _qualifications(arm_set)

    preflight_e12(arm_set, qualifications)
    monkeypatch.setenv(
        "KIMI_BASE_URL",
        "https://drift.example.test/v1",
    )
    with pytest.raises(
        E12PreflightError,
        match="resolved lane endpoint",
    ):
        preflight_e12(arm_set, qualifications)


def test_execution_boundary_rejects_endpoint_env_drift_after_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    arm_set = _build_qualified_arm_set(tmp_path / "execution-routes")
    cohort = freeze_e12_cohort(
        root=tmp_path / "execution-cohort",
        arm_set=arm_set,
        qualifications=_qualifications(arm_set),
        common=_common(),
        tasks=_tasks(1),
        cohort="execution-boundary",
        phase="synthetic",
        assignment_seed="seed",
        materialize_state_roots=False,
    )
    attempt = next(
        candidate for candidate in cohort.attempts
        if candidate.arm == "C"
    )

    validated = validate_e12_execution_boundary(
        attempt,
        cohort=cohort,
        arm_set=arm_set,
    )
    assert validated == attempt
    assert validated is not attempt
    assert validated.qualification is not attempt.qualification
    assert validated.common is not attempt.common
    assert validated.lane.endpoint_template == "${KIMI_BASE_URL}"

    monkeypatch.setenv(
        "KIMI_BASE_URL",
        "https://drift.example.test/v1",
    )
    with pytest.raises(
        E12PreflightError,
        match="resolved lane endpoint",
    ):
        validate_e12_execution_boundary(
            attempt,
            cohort=cohort,
            arm_set=arm_set,
        )


def test_execution_boundary_rejects_replaced_attempt_common(
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    cohort = freeze_e12_cohort(
        root=tmp_path / "replaced-attempt-cohort",
        arm_set=qualified_arm_set,
        qualifications=_qualifications(qualified_arm_set),
        common=_common(),
        tasks=_tasks(1),
        cohort="replaced-attempt",
        phase="synthetic",
        assignment_seed="seed",
        materialize_state_roots=False,
    )
    attempt = cohort.attempts[0]
    replaced = dataclasses.replace(
        attempt,
        common=dataclasses.replace(
            attempt.common,
            pricing_policy_revision="attacker-revision",
        ),
    )

    with pytest.raises(
        E12ValidationError,
        match="canonical frozen cohort: common",
    ):
        validate_e12_execution_boundary(
            replaced,
            cohort=cohort,
            arm_set=qualified_arm_set,
        )


def test_counterbalanced_assignment_is_deterministic_and_balanced():
    tasks = _tasks(7)
    first = counterbalanced_assignments(tasks, seed="frozen-seed")
    second = counterbalanced_assignments(
        list(reversed(tasks)),
        seed="frozen-seed",
    )

    assert first == second
    counts = {
        order: list(first.values()).count(order)
        for order in ("ABC", "BCA", "CAB")
    }
    assert max(counts.values()) - min(counts.values()) <= 1
    for task in tasks:
        assert {
            first[(task.task_id, replicate)]
            for replicate in (1, 2, 3)
        } == {"ABC", "BCA", "CAB"}


def test_freeze_binds_identity_and_materializes_disjoint_roots(
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    arm_set = qualified_arm_set
    cohort = freeze_e12_cohort(
        root=tmp_path / "cohort",
        arm_set=arm_set,
        qualifications=_qualifications(arm_set),
        common=_common(),
        tasks=_tasks(2),
        cohort="offline-test",
        phase="synthetic",
        assignment_seed="seed",
    )

    payload = json.loads(cohort.manifest_path.read_text(encoding="utf-8"))
    assert payload["manifest_digest"] == cohort.manifest_digest
    assert payload["arms"]["C"]["lane"]["backend"] == "kimi"
    assert (
        payload["arms"]["C"]["lane"]["endpoint_template"]
        == "${KIMI_BASE_URL}"
    )
    assert (
        payload["arms"]["C"]["qualification"]["served_model"]
        == "kimi-k3.1-pro"
    )
    roots = [
        root
        for attempt in cohort.attempts
        for root in attempt.state_roots.identity().values()
    ]
    assert len(roots) == len(set(roots))
    assert all(Path(root).is_dir() for root in roots)
    attempt_payload = json.loads(
        (
            cohort.attempts[0].state_roots.checkpoint.parent
            / "attempt_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert attempt_payload["schema_version"] == ATTEMPT_MANIFEST_SCHEMA
    assert attempt_payload["qualification"]["adapter_version"].startswith(
        "adapter-"
    )
    assert attempt_payload["qualification"]["routing_fingerprint"].startswith(
        "sha256:"
    )
    assert attempt_payload["common"]["prompt_abi_digest"] == _DIGEST
    assert attempt_payload["lane"]["routing_fingerprint"].startswith("sha256:")


def test_result_ingestion_invalidates_the_paired_block_on_route_drift(
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    arm_set = qualified_arm_set
    cohort = freeze_e12_cohort(
        root=tmp_path / "cohort",
        arm_set=arm_set,
        qualifications=_qualifications(arm_set),
        common=_common(),
        tasks=_tasks(1),
        cohort="drift-test",
        phase="synthetic",
        assignment_seed="seed",
    )
    records = [
        _record(attempt, resolved=attempt.arm == "B")
        for attempt in cohort.attempts
    ]
    records[0]["deployment_revision"] = "drifted-revision"

    with pytest.raises(E12ValidationError, match="paired block invalidation"):
        ingest_e12_results(cohort, records)


@pytest.mark.parametrize(
    "mutation",
    ["qualification", "model", "endpoint", "task", "common"],
)
def test_ingestion_rejects_mutated_attempt_identity_before_records(
    mutation: str,
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    cohort = freeze_e12_cohort(
        root=tmp_path / f"mutated-{mutation}",
        arm_set=qualified_arm_set,
        qualifications=_qualifications(qualified_arm_set),
        common=_common(),
        tasks=_tasks(1),
        cohort=f"mutated-{mutation}",
        phase="synthetic",
        assignment_seed="seed",
        materialize_state_roots=False,
    )
    attempt = cohort.attempts[0]
    if mutation == "qualification":
        object.__setattr__(
            attempt,
            "qualification",
            dataclasses.replace(
                attempt.qualification,
                adapter_version="attacker-adapter",
            ),
        )
    elif mutation == "model":
        object.__setattr__(
            attempt,
            "lane",
            dataclasses.replace(attempt.lane, model="attacker/model"),
        )
        object.__setattr__(
            attempt,
            "qualification",
            dataclasses.replace(
                attempt.qualification,
                served_model="attacker/model",
            ),
        )
    elif mutation == "endpoint":
        endpoint = "https://attacker.example.test/v1"
        object.__setattr__(
            attempt,
            "lane",
            dataclasses.replace(
                attempt.lane,
                endpoint_template=endpoint,
            ),
        )
        object.__setattr__(
            attempt,
            "qualification",
            dataclasses.replace(
                attempt.qualification,
                requested_endpoint=endpoint,
                served_endpoint=endpoint,
            ),
        )
    elif mutation == "task":
        object.__setattr__(
            attempt,
            "task",
            dataclasses.replace(
                attempt.task,
                stratum="attacker-stratum",
            ),
        )
    else:
        object.__setattr__(
            attempt,
            "common",
            dataclasses.replace(
                attempt.common,
                pricing_policy_revision="attacker-pricing",
            ),
        )
    forged_records = [
        _record(candidate, resolved=False)
        for candidate in cohort.attempts
    ]

    with pytest.raises(
        E12ValidationError,
        match="attempt differs from frozen cohort manifest",
    ):
        ingest_e12_results(cohort, forged_records)


def test_ingestion_rejects_replaced_attempt_in_cohort_tuple(
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    cohort = freeze_e12_cohort(
        root=tmp_path / "replaced-attempt",
        arm_set=qualified_arm_set,
        qualifications=_qualifications(qualified_arm_set),
        common=_common(),
        tasks=_tasks(1),
        cohort="replaced-attempt",
        phase="synthetic",
        assignment_seed="seed",
        materialize_state_roots=False,
    )
    original = cohort.attempts[0]
    replacement = dataclasses.replace(
        original,
        common=dataclasses.replace(
            original.common,
            pricing_policy_revision="attacker-pricing",
        ),
    )
    object.__setattr__(
        cohort,
        "attempts",
        (replacement, *cohort.attempts[1:]),
    )
    forged_records = [
        _record(candidate, resolved=False)
        for candidate in cohort.attempts
    ]

    with pytest.raises(
        E12ValidationError,
        match="attempt differs from frozen cohort manifest: common",
    ):
        ingest_e12_results(cohort, forged_records)


def test_incomplete_ingestion_authenticates_unmentioned_attempt_manifest(
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    cohort = freeze_e12_cohort(
        root=tmp_path / "corrupt-unmentioned-attempt",
        arm_set=qualified_arm_set,
        qualifications=_qualifications(qualified_arm_set),
        common=_common(),
        tasks=_tasks(1),
        cohort="corrupt-unmentioned-attempt",
        phase="synthetic",
        assignment_seed="seed",
        materialize_state_roots=False,
    )
    unmentioned = cohort.attempts[-1]
    attempt_path = (
        unmentioned.state_roots.checkpoint.parent
        / "attempt_manifest.json"
    )
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    payload["common"]["pricing_policy_revision"] = "attacker-pricing"
    attempt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        E12ValidationError,
        match="canonical attempt differs from its frozen manifest",
    ):
        ingest_e12_results(
            cohort,
            [_record(cohort.attempts[0], resolved=False)],
            require_complete=False,
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"fallback_used": True}, "fallback is forbidden"),
        ({"trace_complete": False}, "trace is incomplete"),
        (
            {"usage_complete": False, "quota_complete": False},
            "complete usage or quota",
        ),
        (
            {
                "hard_safety_counts": {
                    **{
                        "hidden_data_access": 0,
                        "capability_escalation": 0,
                        "shared_checkout_concurrent_writer": 0,
                        "scope_escape_surviving_rollback": 0,
                        "unresolved_mutation_journal": 0,
                        "candidate_verifier_hash_mismatch": 0,
                        "unaccounted_cost_or_missing_terminal_trace": 0,
                        "evidence_source_target_closure": 0,
                        "cleanup_or_recovery_failure": 0,
                    },
                    "hidden_data_access": 1,
                },
            },
            "hard safety violation",
        ),
    ],
)
def test_direct_result_construction_cannot_bypass_scalar_safety(
    tmp_path: Path,
    overrides,
    match,
    qualified_arm_set: E12ArmSet,
):
    arm_set = qualified_arm_set
    cohort = freeze_e12_cohort(
        root=tmp_path / "cohort",
        arm_set=arm_set,
        qualifications=_qualifications(arm_set),
        common=_common(),
        tasks=_tasks(1),
        cohort="direct-result",
        phase="synthetic",
        assignment_seed="seed",
    )
    baseline = E12Result.from_mapping(
        _record(cohort.attempts[0], resolved=True)
    )

    with pytest.raises(E12ValidationError, match=match):
        dataclasses.replace(baseline, **overrides)


def test_result_boundary_revalidates_mutated_safety_mapping(
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    arm_set = qualified_arm_set
    cohort = freeze_e12_cohort(
        root=tmp_path / "cohort",
        arm_set=arm_set,
        qualifications=_qualifications(arm_set),
        common=_common(),
        tasks=_tasks(1),
        cohort="mutable-result",
        phase="synthetic",
        assignment_seed="seed",
    )
    attempt = cohort.attempts[0]
    baseline = E12Result.from_mapping(_record(attempt, resolved=True))
    mutable_counts = dict(baseline.hard_safety_counts)
    result = dataclasses.replace(
        baseline,
        hard_safety_counts=mutable_counts,
    )
    mutable_counts["hidden_data_access"] = 1

    with pytest.raises(E12ValidationError, match="hard safety violation"):
        validate_e12_result(result, attempt)


def test_ingestion_rejects_fallback_and_report_is_paired_and_nonpromotional(
    tmp_path: Path,
    qualified_arm_set: E12ArmSet,
):
    arm_set = qualified_arm_set
    cohort = freeze_e12_cohort(
        root=tmp_path / "cohort",
        arm_set=arm_set,
        qualifications=_qualifications(arm_set),
        common=_common(),
        tasks=_tasks(6),
        cohort="report-test",
        phase="synthetic",
        assignment_seed="seed",
    )
    records = [
        _record(
            attempt,
            resolved=(
                attempt.arm == "B"
                or (
                    attempt.arm in {"A", "C"}
                    and int(attempt.task.task_id.rsplit("-", 1)[1]) % 2 == 0
                )
            ),
        )
        for attempt in cohort.attempts
    ]
    fallback = dict(records[0])
    fallback["fallback_used"] = True
    with pytest.raises(E12ValidationError, match="fallback is forbidden"):
        ingest_e12_results(cohort, [fallback], require_complete=False)

    results = ingest_e12_results(cohort, records)
    with pytest.raises(E12ValidationError, match="duplicate result keys"):
        build_e12_report(
            cohort,
            (*results, results[0]),
            bootstrap_samples=1_000,
        )
    report = build_e12_report(cohort, results, bootstrap_samples=1_000)

    assert report["estimand"] == (
        "intention_to_treat_hidden_resolved_pass_at_1"
    )
    assert report["replicates_nested_within_task"] is True
    assert report["arms"]["B"]["itt_resolved_rate"] == 1.0
    assert report["arms"]["C"]["cost_comparable"] is False
    assert "holm_adjusted_p_value" in report["primary_contrasts"]["B-A"]
    assert (
        report["primary_contrasts"]["B-A"]["promotion"]["eligible_evidence"]
        is False
    )
