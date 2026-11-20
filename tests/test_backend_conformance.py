# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared offline contract tests for concrete backend feature profiles."""

from __future__ import annotations

import pytest

from assay import BackendFeatureSet, CONSERVATIVE_BACKEND_FEATURES
from crucible.backends.features import (
    FeatureAdmissionError,
    FeatureRequirement,
    admit_feature_requirements,
    evaluate_feature_requirements,
    execution_feature_requirements,
    feature_profile_digest,
    feature_profile_for_route,
)
from crucible.backends.roles import role_feature_requirements


@pytest.mark.parametrize(
    ("backend", "model", "structured_output", "tool_loop"),
    [
        ("codex", "gpt-5.6-sol", "native", "sdk_owned"),
        ("claude", "claude-opus-5", "native", "sdk_owned"),
        (
            "openai_compat",
            "generic-model",
            "unsupported",
            "adapter_owned",
        ),
        ("kimi", "kimi-k3.1-pro", "prompt_only", "sdk_owned"),
    ],
)
def test_declared_features_match_observed_behavior(
    backend,
    model,
    structured_output,
    tool_loop,
):
    profile = feature_profile_for_route(
        backend,
        model=model,
        base_url=(
            "https://api.moonshot.ai/v1"
            if backend == "kimi"
            else None
        ),
    )
    assert profile.structured_output == structured_output
    assert profile.tool_loop == tool_loop
    assert profile.native_delegation == "none"
    if backend == "kimi":
        assert profile.event_detail == "tool_calls"
        assert profile.turn_limit == "hard"
        assert profile.response_token_limit == "provider_stop"
        assert profile.deadline == "hard_invocation"
        assert profile.cancellation == "boundary_only"


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_capable_agent_adapters_declare_workflow_continuity(backend):
    profile = feature_profile_for_route(backend)

    assert profile.session_continuity == "workflow_bound"


@pytest.mark.parametrize("backend", ["kimi", "openai_compat"])
def test_unqualified_adapters_declare_no_provider_continuity(backend):
    profile = feature_profile_for_route(backend)

    assert profile.session_continuity == "none"


def test_unknown_backend_resolves_to_conservative_profile():
    assert (
        feature_profile_for_route("third_party", model="unknown")
        == CONSERVATIVE_BACKEND_FEATURES
    )


def test_retired_kimi_model_does_not_inherit_qualification():
    assert (
        feature_profile_for_route(
            "kimi",
            model="kimi-k3",
            base_url="https://api.moonshot.ai/v1",
        )
        == CONSERVATIVE_BACKEND_FEATURES
    )


def test_required_unsupported_feature_rejected_before_transport():
    profile = feature_profile_for_route(
        "kimi",
        model="kimi-k3.1-pro",
        base_url="https://api.moonshot.ai/v1",
    )
    decision = evaluate_feature_requirements(
        profile,
        (
            FeatureRequirement(
                "structured_output",
                frozenset({"native"}),
            ),
            FeatureRequirement(
                "native_delegation",
                frozenset({"read_only", "writer"}),
            ),
        ),
    )
    assert decision.allowed is False
    assert decision.reasons == (
        "structured_output='prompt_only' is not one of ['native']",
        "native_delegation='none' is not one of ['read_only', 'writer']",
    )


def test_supported_feature_policy_allows_in_shadow_mode():
    profile = feature_profile_for_route("codex", model="gpt-5.6-sol")
    decision = evaluate_feature_requirements(
        profile,
        (
            FeatureRequirement(
                "structured_output",
                frozenset({"native", "prompt_only"}),
            ),
            FeatureRequirement(
                "project_instructions",
                frozenset({"suppressible", "not_applicable"}),
            ),
        ),
    )
    assert decision.allowed is True
    assert decision.reasons == ()


def test_feature_requirement_rejects_values_outside_closed_vocabulary():
    with pytest.raises(ValueError, match="unknown values"):
        FeatureRequirement(
            "structured_output",
            frozenset({"native", "best_effort"}),
        )


@pytest.mark.parametrize(
    ("field", "accepted"),
    [
        (None, frozenset({"native"})),
        ("structured_output", {"native"}),
        ("structured_output", frozenset({1})),
    ],
)
def test_feature_requirement_shape_is_closed(field, accepted):
    with pytest.raises(TypeError):
        FeatureRequirement(field, accepted)


def test_execution_requirements_derive_tool_and_confinement_contracts():
    requirements = execution_feature_requirements(
        capability="read_only",
        native_shell=False,
        additional=(
            FeatureRequirement(
                "structured_output",
                frozenset({"native", "prompt_only"}),
            ),
        ),
    )
    assert requirements == (
        FeatureRequirement(
            "tool_loop",
            frozenset({"sdk_owned", "adapter_owned"}),
        ),
        FeatureRequirement(
            "project_instructions",
            frozenset({"suppressible", "not_applicable"}),
        ),
        FeatureRequirement(
            "structured_output",
            frozenset({"native", "prompt_only"}),
        ),
    )


@pytest.mark.parametrize(
    ("role", "requires_structure"),
    [
        ("plan_planner", True),
        ("plan_critic", True),
        ("implementer", False),
        ("tier", False),
    ],
)
def test_workflow_roles_declare_structured_output_requirements(
    role,
    requires_structure,
):
    requirements = role_feature_requirements(role)

    assert bool(requirements) is requires_structure
    if requires_structure:
        assert requirements == (
            FeatureRequirement(
                "structured_output",
                frozenset({"native", "prompt_only"}),
            ),
        )


def test_enforcing_admission_rejects_unsupported_route():
    profile = CONSERVATIVE_BACKEND_FEATURES
    requirements = execution_feature_requirements(
        capability="read_only",
        native_shell=False,
    )

    with pytest.raises(FeatureAdmissionError, match="before"):
        admit_feature_requirements(
            profile,
            requirements,
            mode="enforce",
            route="unknown/model before transport",
        )


def test_diagnostic_admission_preserves_rejected_decision():
    decision = admit_feature_requirements(
        CONSERVATIVE_BACKEND_FEATURES,
        (
            FeatureRequirement(
                "structured_output",
                frozenset({"native"}),
            ),
        ),
        mode="diagnostic",
        route="diagnostic/model",
    )

    assert decision.allowed is False
    assert decision.reasons == (
        "structured_output='unsupported' is not one of ['native']",
    )


def test_feature_profile_digest_is_stable_and_version_bound():
    profile = BackendFeatureSet(structured_output="prompt_only")
    assert feature_profile_digest(profile) == feature_profile_digest(profile)
    assert feature_profile_digest(profile).startswith("sha256:")
    assert feature_profile_digest(profile) != feature_profile_digest(
        BackendFeatureSet(structured_output="native")
    )


@pytest.mark.parametrize(
    "model",
    ["nvidia/zai-org/glm-5.3", "nvidia/moonshotai/kimi-k3.1-pro"],
)
@pytest.mark.parametrize(
    "base_url",
    [
        "https://inference-api.nvidia.com/v1",
        "https://inference-api-dev.nvidia.com/v1",
    ],
)
def test_inference_hub_successor_routes_carry_the_generic_profile(model, base_url):
    # Structured output has not been observed on either successor route, so
    # neither has an exact-route profile. A row is added only together with
    # that observation, and this test then changes with it.
    generic = feature_profile_for_route(
        "openai_compat",
        model=model,
        base_url="https://example.invalid/v1",
    )
    hub = feature_profile_for_route(
        "openai_compat",
        model=model,
        base_url=base_url,
    )
    local = feature_profile_for_route(
        "local",
        model=model,
        base_url="http://127.0.0.1:8000/v1",
    )

    assert hub == generic == local
    assert hub.structured_output == "unsupported"
    decision = evaluate_feature_requirements(
        hub,
        role_feature_requirements("architect_planner"),
    )
    assert decision.allowed is False
    assert decision.reasons == (
        "structured_output='unsupported' is not one of "
        "['native', 'prompt_only']",
    )


@pytest.mark.parametrize(
    "model",
    ["nvidia/zai-org/glm-5.2", "nvidia/moonshotai/kimi-k2.6"],
)
def test_retired_inference_hub_ids_do_not_keep_a_qualification(model):
    assert feature_profile_for_route(
        "openai_compat",
        model=model,
        base_url="https://inference-api.nvidia.com/v1",
    ) == feature_profile_for_route(
        "openai_compat",
        model=model,
        base_url="https://example.invalid/v1",
    )
