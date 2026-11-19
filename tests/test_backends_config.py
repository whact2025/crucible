# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for per-role backend routing config discovery + resolution.

Discovery precedence (first hit wins):
  1. explicit ``config_path`` argument
  2. CRUCIBLE_BACKEND_CONFIG env var
  3. ``<spec_dir>/.crucible/backends.yaml``
  4. ``<workspace>/.crucible/backends.yaml`` (resolved from cwd or
     ``CRUCIBLE_WORKSPACE_ROOT``)
  5. built-in DEFAULT_ROUTING fallthrough

These tests pin the precedence so a reorder regression trips a
failure rather than silently producing a different routing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from crucible.backends import BACKEND_NAMES, resolve
from crucible.backends import config as config_module
from crucible.backends.claude import (
    CLAUDE_NVIDIA_ORG_ID,
    CLAUDE_NVIDIA_ORG_NAME,
)
from crucible.backends.config import (
    BackendRouting,
    DEFAULT_ROUTING,
    RoleConfig,
    bind_routing_snapshot,
    lint_routing_config,
    load_routing,
    resolve_for_role,
    routing_feature_fingerprint,
    routing_fingerprint,
    routing_source_fingerprint,
)
from crucible.backends.features import (
    FeatureAdmissionError,
    FeatureRequirement,
    feature_profile_for_route,
)
from crucible.backends.roles import registry


# ----------------------------------------------------------------------
# Built-in defaults
# ----------------------------------------------------------------------


def test_default_routing_default_backend_is_codex_astra_max():
    assert DEFAULT_ROUTING.default.backend == "codex"
    assert DEFAULT_ROUTING.default.model == "gpt-6-astra"
    assert DEFAULT_ROUTING.default.effort == "max"
    assert DEFAULT_ROUTING.default.feature_admission == "enforce"


@pytest.mark.parametrize(
    "alias",
    [
        "claude_sdk",
        "agent_sdk",
        "anthropic",
        "codex_sdk",
        "openai",
        "gpt",
        "vllm",
        "local_openai",
        "inference_hub",
        "nvidia",
    ],
)
def test_backend_factory_rejects_noncanonical_aliases(alias):
    with pytest.raises(ValueError, match="unknown backend"):
        resolve(alias)


def test_backend_factory_registers_kimi_as_dedicated_adapter(monkeypatch):
    import crucible.backends as backends

    captured = {}

    class StubKimiBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(backends, "KimiBackend", StubKimiBackend)

    backend = resolve(
        "kimi",
        model="kimi-k3.1-pro",
        cwd="/workspace",
    )

    assert "kimi" in BACKEND_NAMES
    assert isinstance(backend, StubKimiBackend)
    assert captured == {
        "cwd": "/workspace",
        "log_dir": None,
        "model": "kimi-k3.1-pro",
    }


def test_low_level_codex_factory_rejects_unqualified_delegating_effort():
    with pytest.raises(ValueError, match="automatic task delegation"):
        resolve("codex", effort="ultra")


def test_every_registered_role_inherits_codex_astra_max():
    assert DEFAULT_ROUTING.roles == {}
    for role in registry():
        assert DEFAULT_ROUTING.for_role(role) == DEFAULT_ROUTING.default


def test_default_route_constructs_codex_backend():
    from crucible.backends.codex import CodexBackend

    backend = resolve_for_role("implementer", routing=DEFAULT_ROUTING)

    assert isinstance(backend, CodexBackend)
    assert backend.model == "gpt-6-astra"
    assert backend._effort == "max"


def test_default_routing_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="unknown backend role"):
        DEFAULT_ROUTING.for_role("never_named_role_xyz")


def test_default_routing_swebench_is_aliased_to_implementer():
    """SWE-bench uses the implementer role (workflows/swebench.py routes
    that way). The 'swebench' string itself is unknown to the table —
    operators wanting a different model for SWE-bench set
    ``implementer:`` in their config, not ``swebench:``."""
    assert "swebench" not in DEFAULT_ROUTING.roles
    assert DEFAULT_ROUTING.for_role("implementer") == DEFAULT_ROUTING.default


# ----------------------------------------------------------------------
# Role-level override semantics
# ----------------------------------------------------------------------


def test_role_level_overrides_default_field_by_field():
    """A role config with only ``model`` set inherits backend / effort
    from the default — partial overrides don't wipe sibling fields."""
    routing = BackendRouting(
        default=RoleConfig(
            backend="claude",
            model="aws/anthropic/bedrock-claude-opus-4-8",
            effort="high",
        ),
        roles={
            "plan_critic": RoleConfig(model="aws/anthropic/bedrock-claude-sonnet-4-6")
        },
    )
    cfg = routing.for_role("plan_critic")
    assert cfg.backend == "claude"  # inherited
    assert cfg.model == "aws/anthropic/bedrock-claude-sonnet-4-6"  # overridden
    assert cfg.effort == "high"  # inherited


def test_role_level_extra_kwargs_merge_with_default():
    """``extra`` dicts are merged (role wins on conflict)."""
    routing = BackendRouting(
        default=RoleConfig(
            backend="openai_compat",
            model="m",
            extra={"reasoning_effort": "medium", "max_tokens": 2048},
        ),
        roles={"plan_critic": RoleConfig(extra={"max_tokens": 4096})},
    )
    cfg = routing.for_role("plan_critic")
    assert cfg.extra == {"reasoning_effort": "medium", "max_tokens": 4096}


def test_role_backend_switch_does_not_inherit_codex_fields():
    routing = BackendRouting(
        default=RoleConfig(
            backend="codex",
            model="gpt-5.6-sol",
            effort="ultra",
            extra={"output_schema": {"type": "object"}},
        ),
        roles={
            "plan_critic": RoleConfig(
                backend="claude",
                model="claude-sonnet-4-6",
            ),
        },
    )

    cfg = routing.for_role("plan_critic")
    assert cfg.backend == "claude"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.effort is None
    assert cfg.extra == {}


# ----------------------------------------------------------------------
# Discovery precedence
# ----------------------------------------------------------------------


def _write_yaml(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": "crucible.backend-routing.v1", **doc}
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


# ----------------------------------------------------------------------
# ${VAR} env-ref expansion in base_url / api_key (secret handling)
# ----------------------------------------------------------------------


def test_api_key_env_ref_expands_from_environ(tmp_path, monkeypatch):
    """``api_key: ${VAR}`` resolves from os.environ at load time so the
    secret never lives in the YAML."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-hub-key-xyz")
    cfg = tmp_path / "c.yaml"
    _write_yaml(
        cfg,
        {
            "roles": {
                "implementer": {
                    "backend": "openai_compat",
                    "base_url": "https://inference-api.nvidia.com",
                    "api_key": "${NVIDIA_API_KEY}",
                    "model": "nvidia/minimaxai/minimax-m2.7",
                },
            },
        },
    )
    b = resolve_for_role("implementer", config_path=cfg)
    assert b._api_key == "secret-hub-key-xyz"
    assert b._base_url == "https://inference-api.nvidia.com"


def test_expanded_api_key_is_excluded_from_routing_repr_and_equality():
    secret = "inference-token-super-secret"
    left = RoleConfig(
        backend="openai_compat",
        model="model",
        base_url="https://example.invalid/v1",
        api_key=secret,
    )
    right = RoleConfig(
        backend="openai_compat",
        model="model",
        base_url="https://example.invalid/v1",
        api_key="rotated-token",
    )

    assert secret not in repr(left)
    assert secret not in repr(BackendRouting(default=left))
    assert left == right


def test_unset_env_ref_raises_rather_than_sending_literal(tmp_path, monkeypatch):
    """An unset ${VAR} hard-fails at load — better than a confusing 401
    when the literal ``${VAR}`` ships as a bearer token."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("MISSING_KEY_VAR", raising=False)
    cfg = tmp_path / "c.yaml"
    _write_yaml(
        cfg,
        {
            "roles": {
                "implementer": {
                    "backend": "openai_compat",
                    "api_key": "${MISSING_KEY_VAR}",
                    "model": "m",
                },
            },
        },
    )
    with pytest.raises(ValueError, match="MISSING_KEY_VAR"):
        load_routing(config_path=cfg)


@pytest.mark.parametrize("value", ["", " \t "])
def test_empty_env_ref_raises_before_backend_fallback(
    tmp_path,
    monkeypatch,
    value,
):
    """An explicit hosted credential cannot degrade to ``not-needed``."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("EMPTY_KEY_VAR", value)
    cfg = tmp_path / "c.yaml"
    _write_yaml(
        cfg,
        {
            "roles": {
                "implementer": {
                    "backend": "openai_compat",
                    "api_key": "${EMPTY_KEY_VAR}",
                    "model": "m",
                },
            },
        },
    )

    with pytest.raises(ValueError, match="empty or whitespace"):
        load_routing(config_path=cfg)


def test_discovery_explicit_path_wins_over_ambient_env(tmp_path, monkeypatch):
    """An explicit arm cannot be redirected by CRUCIBLE_BACKEND_CONFIG."""
    env_yaml = tmp_path / "from_env.yaml"
    arg_yaml = tmp_path / "from_arg.yaml"
    spec_yaml = tmp_path / "spec/.crucible/backends.yaml"
    _write_yaml(env_yaml, {"default": {"model": "from-env-model"}})
    _write_yaml(arg_yaml, {"default": {"model": "from-arg-model"}})
    _write_yaml(spec_yaml, {"default": {"model": "from-spec-model"}})

    monkeypatch.setenv("CRUCIBLE_BACKEND_CONFIG", str(env_yaml))
    r = load_routing(config_path=arg_yaml, spec_dir=tmp_path / "spec")
    assert r.default.model == "from-arg-model"


def test_discovery_env_var_wins_when_no_explicit_path(tmp_path, monkeypatch):
    env_yaml = tmp_path / "from_env.yaml"
    spec_yaml = tmp_path / "spec/.crucible/backends.yaml"
    _write_yaml(env_yaml, {"default": {"model": "from-env-model"}})
    _write_yaml(spec_yaml, {"default": {"model": "from-spec-model"}})

    monkeypatch.setenv("CRUCIBLE_BACKEND_CONFIG", str(env_yaml))
    r = load_routing(config_path=None, spec_dir=tmp_path / "spec")
    assert r.default.model == "from-env-model"


def test_missing_explicit_config_fails_instead_of_falling_back(
    tmp_path,
    monkeypatch,
):
    fallback = tmp_path / "fallback.yaml"
    _write_yaml(fallback, {"default": {"model": "ambient-fallback"}})
    monkeypatch.setenv("CRUCIBLE_BACKEND_CONFIG", str(fallback))

    with pytest.raises(
        ValueError,
        match="explicit backend routing config is not a file",
    ):
        load_routing(config_path=tmp_path / "missing-arm.yaml")


def test_discovery_config_path_arg_wins_over_spec_local(tmp_path, monkeypatch):
    """Explicit --backend-config beats <spec>/.crucible/backends.yaml."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    arg_yaml = tmp_path / "from_arg.yaml"
    spec_yaml = tmp_path / "spec/.crucible/backends.yaml"
    _write_yaml(arg_yaml, {"default": {"model": "from-arg-model"}})
    _write_yaml(spec_yaml, {"default": {"model": "from-spec-model"}})

    r = load_routing(config_path=arg_yaml, spec_dir=tmp_path / "spec")
    assert r.default.model == "from-arg-model"
    assert r.source_sha256 is not None and len(r.source_sha256) == 64


def test_discovery_spec_local_wins_when_no_explicit_path(
    tmp_path,
    monkeypatch,
):
    """<spec>/.crucible/backends.yaml beats workspace-level + defaults."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    spec_yaml = tmp_path / "spec/.crucible/backends.yaml"
    _write_yaml(spec_yaml, {"default": {"model": "from-spec-model"}})

    r = load_routing(config_path=None, spec_dir=tmp_path / "spec")
    assert r.default.model == "from-spec-model"


def test_bound_routing_snapshot_is_immune_to_child_context_and_file_edits(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    root_spec = tmp_path / "root-spec"
    config = root_spec / ".crucible" / "backends.yaml"
    _write_yaml(config, {"default": {"model": "snapshot-model"}})
    snapshot = load_routing(spec_dir=root_spec)
    identity = routing_fingerprint(snapshot)

    _write_yaml(config, {"default": {"model": "edited-model"}})
    child_spec = tmp_path / "root-spec" / "instances" / "task" / "corpus"
    _write_yaml(
        child_spec / ".crucible" / "backends.yaml",
        {"default": {"model": "child-model"}},
    )

    with bind_routing_snapshot(snapshot):
        nested = load_routing(spec_dir=child_spec)
        assert nested is snapshot
        assert nested.default.model == "snapshot-model"
        assert routing_fingerprint(nested) == identity

    assert load_routing(spec_dir=root_spec).default.model == "edited-model"


def test_routing_source_fingerprint_binds_exact_selected_bytes(tmp_path):
    left_path = tmp_path / "left.yaml"
    right_path = tmp_path / "right.yaml"
    left_path.write_text(
        "schema_version: crucible.backend-routing.v1\n"
        "default: {backend: codex, model: gpt-6-astra}\n",
        encoding="utf-8",
    )
    right_path.write_text(
        "# operationally distinct source bytes\n" + left_path.read_text(),
        encoding="utf-8",
    )

    left = load_routing(config_path=left_path)
    right = load_routing(config_path=right_path)
    left_source = "sha256:" + hashlib.sha256(left_path.read_bytes()).hexdigest()
    right_source = "sha256:" + hashlib.sha256(right_path.read_bytes()).hexdigest()

    assert routing_fingerprint(left) == routing_fingerprint(right)
    assert routing_source_fingerprint(left) == left_source
    assert routing_source_fingerprint(right) == right_source
    assert routing_source_fingerprint(lint_routing_config(left_path)) == left_source
    assert routing_source_fingerprint(
        lint_routing_config(left_path.read_bytes())
    ) == left_source
    assert routing_source_fingerprint(left) != routing_source_fingerprint(right)
    assert routing_source_fingerprint(DEFAULT_ROUTING) is None


def test_routing_source_fingerprint_rejects_malformed_retained_digest():
    routing = BackendRouting(
        default=RoleConfig(backend="codex", model="gpt-6-astra"),
        source_sha256="not-a-digest",
    )

    with pytest.raises(ValueError, match="source_sha256"):
        routing_source_fingerprint(routing)


def test_routing_source_fingerprint_ignores_expanded_credential_rotation(
    tmp_path, monkeypatch,
):
    path = tmp_path / "route.yaml"
    path.write_text(
        "schema_version: crucible.backend-routing.v1\n"
        "default:\n"
        "  backend: openai_compat\n"
        "  model: routed-model\n"
        "  base_url: https://provider.invalid/v1\n"
        "  api_key: ${ROUTING_TEST_TOKEN}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ROUTING_TEST_TOKEN", "first-secret")
    left = load_routing(config_path=path)
    monkeypatch.setenv("ROUTING_TEST_TOKEN", "rotated-secret")
    right = load_routing(config_path=path)

    assert left.default.api_key != right.default.api_key
    assert routing_fingerprint(left) == routing_fingerprint(right)
    assert routing_source_fingerprint(left) == routing_source_fingerprint(right)


def test_routing_fingerprint_redacts_credential_rotation():
    left = BackendRouting(
        default=RoleConfig(
            backend="openai_compat",
            model="m",
            api_key="secret-one",
        ),
    )
    right = BackendRouting(
        default=RoleConfig(
            backend="openai_compat",
            model="m",
            api_key="secret-two",
        ),
    )
    assert routing_fingerprint(left) == routing_fingerprint(right)
    assert routing_feature_fingerprint(left) == routing_feature_fingerprint(right)


def test_routing_fingerprint_binds_backend_feature_profile():
    # Same adapter and model on two hosts: the Kimi feature row is qualified
    # by the exact Moonshot endpoint, so only the host tells the two apart.
    moonshot = BackendRouting(
        default=RoleConfig(
            backend="kimi",
            model="kimi-k3.1-pro",
            base_url="https://api.moonshot.ai/v1",
            api_key="secret",
        ),
    )
    unqualified = BackendRouting(
        default=RoleConfig(
            backend="kimi",
            model="kimi-k3.1-pro",
            base_url="https://example.invalid/v1",
            api_key="secret",
        ),
    )
    assert routing_feature_fingerprint(moonshot) != routing_feature_fingerprint(
        unqualified
    )
    assert routing_fingerprint(moonshot) != routing_fingerprint(unqualified)


def test_kimi_route_fingerprint_binds_dedicated_feature_profile():
    dedicated = BackendRouting(
        default=RoleConfig(
            backend="kimi",
            model="kimi-k3.1-pro",
            base_url="https://api.moonshot.ai/v1",
            api_key="secret",
        ),
    )
    control = BackendRouting(
        default=RoleConfig(
            backend="openai_compat",
            model="kimi-k3.1-pro",
            base_url="https://api.moonshot.ai/v1",
            api_key="secret",
        ),
    )

    assert routing_feature_fingerprint(dedicated) != (
        routing_feature_fingerprint(control)
    )
    assert routing_fingerprint(dedicated) != routing_fingerprint(control)


def test_routing_fingerprints_bind_feature_admission_policy():
    enforced = BackendRouting(
        default=RoleConfig(
            backend="codex",
            model="gpt-5.6-sol",
            feature_admission="enforce",
        ),
    )
    diagnostic = BackendRouting(
        default=RoleConfig(
            backend="codex",
            model="gpt-5.6-sol",
            feature_admission="diagnostic",
        ),
    )

    assert routing_fingerprint(enforced) != routing_fingerprint(diagnostic)
    assert routing_feature_fingerprint(enforced) != (
        routing_feature_fingerprint(diagnostic)
    )


def test_discovery_uses_configured_workspace_not_package_location(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    workspace = tmp_path / "external-workspace"
    (workspace / "crucible").mkdir(parents=True)
    workspace_yaml = workspace / ".crucible/backends.yaml"
    _write_yaml(workspace_yaml, {"default": {"model": "workspace-model"}})
    monkeypatch.setenv("CRUCIBLE_WORKSPACE_ROOT", str(workspace))

    routing = load_routing(spec_dir=tmp_path / "empty-spec")

    assert routing.default.model == "workspace-model"


def test_discovery_falls_through_to_defaults_when_no_config_present(
    tmp_path,
    monkeypatch,
):
    """When no env / arg / spec-local config exists, the built-in
    DEFAULT_ROUTING is returned (workspace-level may or may not exist
    in the test environment; built-in is the floor)."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    # Direct a spec_dir at an empty tmp tree so workspace-level
    # discovery is the only remaining candidate.
    r = load_routing(config_path=None, spec_dir=tmp_path / "empty")
    # Either built-in defaults, or workspace-level if one exists. In
    # either case the implementer role MUST resolve to a real model.
    cfg = r.for_role("implementer")
    assert cfg.backend is not None
    assert cfg.model is not None


def test_discovery_partial_override_preserves_default_for_other_roles(
    tmp_path,
    monkeypatch,
):
    """A one-role override leaves every other role on Codex Sol."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    spec_yaml = tmp_path / "spec/.crucible/backends.yaml"
    _write_yaml(
        spec_yaml,
        {
            "roles": {
                "plan_critic": {"backend": "codex", "model": "gpt-5.5"},
            },
        },
    )

    r = load_routing(config_path=None, spec_dir=tmp_path / "spec")
    # Overridden role hits the new backend.
    plan_critic = r.for_role("plan_critic")
    assert plan_critic.backend == "codex"
    assert plan_critic.model == "gpt-5.5"
    # Every other role continues to inherit the built-in Codex default.
    goal_blue = r.for_role("goal_blue")
    assert goal_blue.backend == "codex"
    assert goal_blue.model == "gpt-6-astra"
    assert goal_blue.effort == "max"


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_load_routing_raises_on_invalid_yaml(tmp_path, monkeypatch):
    """Malformed YAML raises ``ValueError`` — silent fallthrough would
    mask operator typos."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    bad = tmp_path / "bad.yaml"
    bad.write_text("default:\n  model: [unterminated", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_routing(config_path=bad)


def test_load_routing_raises_on_non_mapping_root(tmp_path, monkeypatch):
    """A list-typed root is rejected — the schema requires a mapping."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    bad = tmp_path / "list.yaml"
    bad.write_text("- foo\n- bar\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_routing(config_path=bad)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("default", [], "default must be a mapping"),
        ("roles", [], "`roles` must be a mapping"),
    ],
)
def test_falsey_nonmapping_sections_never_fall_back_to_default_route(
    tmp_path,
    monkeypatch,
    field,
    value,
    message,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    bad = tmp_path / "malformed-section.yaml"
    document = {
        "schema_version": "crucible.backend-routing.v1",
        "default": {
            "backend": "claude",
            "model": "aws/anthropic/bedrock-claude-opus-5",
        },
        "roles": {},
    }
    document[field] = value
    _write_yaml(bad, document)

    with pytest.raises(ValueError, match=message):
        load_routing(config_path=bad)


def test_load_routing_requires_versioned_schema(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    bad = tmp_path / "unversioned.yaml"
    bad.write_text("default:\n  backend: claude\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_routing(config_path=bad)


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ({"roles": {"typo_reviwer": {"backend": "claude"}}}, "unknown backend role"),
        ({"roles": {"plan_critic": {"backned": "claude"}}}, "unknown fields"),
        ({"surprise": True}, "unknown root fields"),
    ],
)
def test_load_routing_rejects_closed_schema_violations(
    tmp_path,
    monkeypatch,
    doc,
    message,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "bad.yaml"
    _write_yaml(path, doc)
    with pytest.raises(ValueError, match=message):
        load_routing(config_path=path)


def test_adapter_options_are_closed_inside_explicit_extra_mapping(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "invalid.yaml"
    _write_yaml(
        path,
        {
            "roles": {
                "plan_critic": {
                    "backend": "claude",
                    "model": "claude-sonnet-4-6",
                    "extra": {"custom_transport_option": 7},
                },
            },
        },
    )
    with pytest.raises(ValueError, match="unsupported.*extra fields"):
        load_routing(config_path=path)


def test_adapter_options_validate_backend_specific_types(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "invalid.yaml"
    _write_yaml(
        path,
        {
            "roles": {
                "implementer": {
                    "backend": "openai_compat",
                    "model": "model",
                    "extra": {"max_tokens": True},
                },
            },
        },
    )
    with pytest.raises(ValueError, match="positive integer"):
        load_routing(config_path=path)


@pytest.mark.parametrize("value", ["", " model", "model ", True, 7])
def test_openai_compatible_expected_served_model_requires_exact_string(
    tmp_path,
    monkeypatch,
    value,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "invalid-expected-served-model.yaml"
    _write_yaml(
        path,
        {
            "roles": {
                "implementer": {
                    "backend": "openai_compat",
                    "model": "model",
                    "extra": {"expected_served_model": value},
                },
            },
        },
    )
    with pytest.raises(ValueError, match="expected_served_model"):
        load_routing(config_path=path)


@pytest.mark.parametrize("backend", ["local", "openai_compat"])
def test_openai_compatible_routes_accept_expected_served_model(
    tmp_path,
    monkeypatch,
    backend,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / f"{backend}-expected-served-model.yaml"
    _write_yaml(
        path,
        {
            "default": {
                "backend": backend,
                "model": "model",
                "extra": {"expected_served_model": "served-model"},
            },
        },
    )

    routing = load_routing(config_path=path)

    assert routing.default.extra["expected_served_model"] == "served-model"


@pytest.mark.parametrize(
    "credential_source",
    [None, "", "session", "claude-session", True],
)
def test_claude_credential_source_is_closed(
    tmp_path,
    monkeypatch,
    credential_source,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "invalid-claude-credential-source.yaml"
    _write_yaml(
        path,
        {
            "default": {
                "backend": "claude",
                "model": "claude-opus-5",
                "extra": {"credential_source": credential_source},
            },
        },
    )

    with pytest.raises(ValueError, match="credential_source must be one of"):
        load_routing(config_path=path)


@pytest.mark.parametrize(
    "conflict",
    [
        {"base_url": "https://gateway.example/v1"},
        {"api_key": "test-only-token"},
    ],
)
def test_claude_session_route_rejects_endpoint_and_key_conflicts(
    tmp_path,
    monkeypatch,
    conflict,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "conflicting-claude-session.yaml"
    _write_yaml(
        path,
        {
            "default": {
                "backend": "claude",
                "model": "claude-opus-5",
                "extra": {
                    "credential_source": "claude_session",
                    "expected_org_id": CLAUDE_NVIDIA_ORG_ID,
                },
                **conflict,
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="cannot be combined with base_url or api_key",
    ):
        load_routing(config_path=path)


def test_claude_session_route_rejects_gateway_model_spelling(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "aliased-claude-session.yaml"
    _write_yaml(
        path,
        {
            "default": {
                "backend": "claude",
                "model": "aws/anthropic/bedrock-claude-fable-5-1",
                "extra": {
                    "credential_source": "claude_session",
                    "expected_org_id": CLAUDE_NVIDIA_ORG_ID,
                },
            },
        },
    )

    with pytest.raises(ValueError, match="requires exact model from reviewed set"):
        load_routing(config_path=path)


def test_claude_session_route_accepts_exact_fable_5_1_model(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "fable-claude-session.yaml"
    _write_yaml(
        path,
        {
            "default": {
                "backend": "claude",
                "model": "claude-fable-5-1",
                "extra": {
                    "credential_source": "claude_session",
                    "expected_org_id": CLAUDE_NVIDIA_ORG_ID,
                },
            },
        },
    )

    routing = load_routing(config_path=path)

    assert routing.default.model == "claude-fable-5-1"


def test_claude_session_route_requires_expected_org_id(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "unbound-claude-session.yaml"
    _write_yaml(
        path,
        {
            "default": {
                "backend": "claude",
                "model": "claude-opus-5",
                "extra": {"credential_source": "claude_session"},
            },
        },
    )

    with pytest.raises(ValueError, match="requires.*expected_org_id"):
        load_routing(config_path=path)


def test_claude_environment_route_rejects_expected_org_fields(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "inert-claude-org.yaml"
    _write_yaml(
        path,
        {
            "default": {
                "backend": "claude",
                "model": "claude-opus-5",
                "extra": {"expected_org_id": CLAUDE_NVIDIA_ORG_ID},
            },
        },
    )

    with pytest.raises(ValueError, match="require.*claude_session"):
        load_routing(config_path=path)


@pytest.mark.parametrize("backend", ["kimi", "local"])
def test_unsupported_effort_setting_rejects_during_routing_preflight(
    tmp_path,
    monkeypatch,
    backend,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "invalid.yaml"
    _write_yaml(
        path,
        {
            "default": {
                "backend": backend,
                "model": "model",
                "effort": "high",
            },
        },
    )

    with pytest.raises(ValueError, match="does not support that setting"):
        load_routing(config_path=path)


def test_unknown_feature_admission_mode_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "invalid.yaml"
    _write_yaml(
        path,
        {
            "default": {
                "feature_admission": "best_effort",
            },
        },
    )

    with pytest.raises(ValueError, match="feature_admission"):
        load_routing(config_path=path)


def test_every_bundled_product_routing_file_lints_without_secrets():
    products = Path(__file__).parents[1] / "products"
    paths = sorted(products.glob("*/backends*.yaml"))
    assert paths
    for path in paths:
        lint_routing_config(path)


def test_routing_lint_accepts_already_measured_bytes_without_path_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "route.yaml"
    _write_yaml(path, {})
    payload = path.read_bytes()

    monkeypatch.setattr(
        config_module,
        "_routing_payload",
        lambda *_args, **_kwargs: pytest.fail("measured route was reopened"),
    )
    routing = lint_routing_config(payload)

    assert routing.default == config_module.DEFAULT_ROUTING.default


@pytest.mark.parametrize("product", ["swebench", "deepswe"])
def test_inference_hub_profile_routes_every_role_to_glm_5_3(product):
    path = Path(__file__).parents[1] / f"products/{product}/backends_inference_hub.yaml"
    routing = lint_routing_config(path)

    assert routing.default.backend == "openai_compat"
    assert routing.default.model == "nvidia/zai-org/glm-5.3"
    assert routing.default.base_url == "${INFER_BASE_URL}"
    assert routing.default.feature_admission == "diagnostic"
    assert routing.default.effort is None
    assert routing.default.extra == {
        "expected_served_model": "nvidia/zai-org/glm-5.3",
    }
    expected_roles = {
        "architect_planner",
        "architect_critic",
        "architect_judge",
        "implementer",
        "breach_defender",
        "breach_judge",
        "diff_scope",
        "tier_completeness",
        "review_conflict_authority",
    }
    expected_roles.add(
        "swebench_impl_breaker" if product == "swebench" else "impl_breaker"
    )
    if product == "deepswe":
        expected_roles.add("api_contract_fidelity")
    assert set(routing.roles) == expected_roles
    for role in expected_roles:
        assert routing.for_role(role) == routing.default


def test_deepswe_inference_profile_uses_clean_shell_auth_token(monkeypatch):
    """The documented DeepSWE env file exports INFER_AUTH_TOKEN only."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("INFER_AUTH_TOKEN", "clean-shell-token")
    monkeypatch.setenv(
        "INFER_BASE_URL",
        "https://inference-api.nvidia.com",
    )
    path = Path(__file__).parents[1] / "products/deepswe/backends_inference_hub.yaml"

    routing = load_routing(config_path=path)

    assert routing.default.api_key == "clean-shell-token"
    assert routing.default.base_url == "https://inference-api.nvidia.com"


@pytest.mark.parametrize("product", ["swebench", "deepswe"])
def test_kimi_inference_profile_is_diagnostic_openai_control(product):
    path = (
        Path(__file__).parents[1] / f"products/{product}/backends_kimi_inference.yaml"
    )
    routing = lint_routing_config(path)

    assert routing.default.backend == "openai_compat"
    assert routing.default.model == "nvidia/moonshotai/kimi-k3.1-pro"
    assert routing.default.effort == "high"
    assert routing.default.base_url == "${INFER_BASE_URL}"
    assert routing.default.api_key == "${INFER_AUTH_TOKEN}"
    assert routing.default.feature_admission == "diagnostic"
    assert routing.default.extra == {
        "expected_served_model": "nvidia/moonshotai/kimi-k3.1-pro",
    }


@pytest.mark.parametrize("product", ["swebench", "deepswe"])
def test_kimi_sdk_profile_is_dedicated_arm_not_openai_alias(product):
    path = Path(__file__).parents[1] / f"products/{product}/backends_kimi_sdk.yaml"
    routing = lint_routing_config(path)

    assert routing.default.backend == "kimi"
    assert routing.default.model == "kimi-k3.1-pro"
    assert routing.default.effort == "max"
    assert routing.default.base_url == "${KIMI_BASE_URL}"
    assert routing.default.api_key == "${MOONSHOT_API_KEY}"
    assert routing.default.extra == {"worker_source": "project_locked"}
    for role in (
        "plan_planner",
        "architect_planner",
    ):
        assert routing.for_role(role).extra == {
            "worker_source": "project_locked",
            "max_response_tokens": 16_384,
        }
    expected_implementer_extra = {
        "worker_source": "project_locked",
    }
    if product == "deepswe":
        expected_implementer_extra["timeout_seconds"] = 1_200
    assert routing.for_role("implementer").extra == expected_implementer_extra


@pytest.mark.parametrize(
    "value",
    [True, 0, -1, 1_048_577, "16384"],
)
def test_kimi_max_response_tokens_routing_extra_is_closed(value):
    routing = BackendRouting(
        default=RoleConfig(
            backend="kimi",
            model="kimi-k3.1-pro",
            extra={"max_response_tokens": value},
        ),
    )

    with pytest.raises(ValueError, match="max_response_tokens"):
        routing_fingerprint(routing)


@pytest.mark.parametrize(
    "value",
    [True, 0, -1, float("inf"), float("nan"), "1200"],
)
def test_kimi_timeout_seconds_routing_extra_is_closed(value):
    routing = BackendRouting(
        default=RoleConfig(
            backend="kimi",
            model="kimi-k3.1-pro",
            extra={"timeout_seconds": value},
        ),
    )

    with pytest.raises(ValueError, match="timeout_seconds"):
        routing_fingerprint(routing)


@pytest.mark.parametrize(
    "value",
    [True, 0, -1, float("inf"), float("nan"), "600"],
)
def test_codex_timeout_seconds_routing_extra_is_closed(value):
    routing = BackendRouting(
        default=RoleConfig(
            backend="codex",
            model="gpt-5.6-sol",
            extra={"timeout_seconds": value},
        ),
    )

    with pytest.raises(ValueError, match="timeout_seconds"):
        routing_fingerprint(routing)


def test_kimi_expected_worker_runtime_fingerprint_reaches_adapter(
    monkeypatch,
):
    import crucible.backends.config as config_module

    captured = {}

    def fake_resolve(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    digest = "sha256:" + "a" * 64
    routing = BackendRouting(
        default=RoleConfig(
            backend="kimi",
            model="kimi-k3.1-pro",
            base_url="https://api.moonshot.ai/v1",
            api_key="runtime-test-token",
            extra={
                "worker_source": "project_locked",
                "expected_worker_runtime_fingerprint": digest,
                "max_response_tokens": 16_384,
                "timeout_seconds": 1_200,
            },
        ),
    )

    resolve_for_role("implementer", routing=routing, capability="text")

    assert captured["name"] == "kimi"
    assert captured["kwargs"]["worker_source"] == "project_locked"
    assert captured["kwargs"]["expected_worker_runtime_fingerprint"] == digest
    assert captured["kwargs"]["max_response_tokens"] == 16_384
    assert captured["kwargs"]["timeout_seconds"] == 1_200


@pytest.mark.parametrize(
    "extra,match",
    [
        (
            {
                "worker_source": "project_locked",
                "expected_worker_runtime_fingerprint": "sha256:NOT-A-DIGEST",
            },
            "must be sha256",
        ),
        (
            {
                "worker_source": "environment",
                "expected_worker_runtime_fingerprint": "sha256:" + "a" * 64,
            },
            "requires worker_source='project_locked'",
        ),
    ],
)
def test_kimi_expected_worker_runtime_fingerprint_is_closed(
    monkeypatch,
    extra,
    match,
):
    import crucible.backends.config as config_module

    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid runtime identity must reject before construction"
        ),
    )
    routing = BackendRouting(
        default=RoleConfig(
            backend="kimi",
            model="kimi-k3.1-pro",
            base_url="https://api.moonshot.ai/v1",
            api_key="runtime-test-token",
            extra=extra,
        ),
    )

    with pytest.raises(ValueError, match=match):
        resolve_for_role("implementer", routing=routing, capability="text")


def _kimi_diagnostic_writer_routing(**overrides):
    values = {
        "backend": "kimi",
        "model": "kimi-k3.1-pro",
        "base_url": "https://api.moonshot.ai/v1",
        "api_key": "runtime-test-token",
        "feature_admission": "diagnostic",
        "extra": {"diagnostic_workspace_write": True},
    }
    values.update(overrides)
    return BackendRouting(default=RoleConfig(**values))


@pytest.mark.parametrize("product", ["swebench", "deepswe"])
def test_kimi_diagnostic_writer_profile_is_closed_and_outside_e12(
    product,
    monkeypatch,
):
    root = Path(__file__).parents[1]
    path = root / f"products/{product}/backends_kimi_sdk_diagnostic.yaml"

    linted = lint_routing_config(path)

    assert linted.default.backend == "kimi"
    assert linted.default.model == "kimi-k3.1-pro"
    assert linted.default.effort == "max"
    assert linted.default.base_url == "${KIMI_BASE_URL}"
    assert linted.default.api_key == "${MOONSHOT_API_KEY}"
    assert linted.default.feature_admission == "diagnostic"
    assert linted.default.extra == {
        "diagnostic_workspace_write": True,
        "diagnostic_cost_upper_bound": True,
        "worker_source": "project_locked",
    }
    for role in (
        "plan_planner",
        "architect_planner",
    ):
        assert linted.for_role(role).extra == {
            "diagnostic_workspace_write": True,
            "diagnostic_cost_upper_bound": True,
            "worker_source": "project_locked",
            "max_response_tokens": 16_384,
        }
    expected_implementer_extra = {
        "diagnostic_workspace_write": True,
        "diagnostic_cost_upper_bound": True,
        "worker_source": "project_locked",
    }
    if product == "deepswe":
        expected_implementer_extra["timeout_seconds"] = 1_200
    assert linted.for_role("implementer").extra == expected_implementer_extra

    e12_path = root / f"products/{product}/e12_arms.yaml"
    e12 = yaml.safe_load(e12_path.read_text(encoding="utf-8"))
    assert e12["arms"]["C"]["routing"] == "backends_kimi_sdk.yaml"
    assert e12["arms"]["C"]["promotion_gate"] == "blocked"
    assert "backends_kimi_sdk_diagnostic.yaml" not in e12_path.read_text(
        encoding="utf-8",
    )

    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    monkeypatch.setenv("MOONSHOT_API_KEY", "runtime-test-token")

    loaded = load_routing(config_path=path)

    assert loaded.default.base_url == "https://api.moonshot.ai/v1"
    assert loaded.default.api_key == "runtime-test-token"
    assert loaded.default.extra == {
        "diagnostic_workspace_write": True,
        "diagnostic_cost_upper_bound": True,
        "worker_source": "project_locked",
    }


@pytest.mark.parametrize(
    "control",
    [None, 0, 1, "true", [], {}],
    ids=["none", "zero", "one", "string", "list", "mapping"],
)
def test_kimi_diagnostic_writer_control_requires_an_exact_bool(
    monkeypatch,
    control,
):
    import crucible.backends.config as config_module

    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid control must reject before adapter construction"
        ),
    )
    routing = _kimi_diagnostic_writer_routing(
        extra={"diagnostic_workspace_write": control},
    )

    with pytest.raises(
        ValueError,
        match=r"extra\.diagnostic_workspace_write must be a bool",
    ):
        resolve_for_role(
            "implementer",
            routing=routing,
            cwd="/workspace",
            capability="workspace_write",
            native_shell=False,
            allow_trusted_project_instructions=False,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"feature_admission": "enforce"},
            "requires feature_admission='diagnostic'",
        ),
        (
            {"model": "kimi-k3"},
            "requires an exact admitted model",
        ),
        (
            {"model": "kimi-k2.7-code-highspeed"},
            "requires an exact admitted model",
        ),
        (
            {"model": "kimi-k3.1-pro-highspeed"},
            "requires an exact admitted model",
        ),
        (
            {"model": "nvidia/moonshotai/kimi-k3.1-pro"},
            "requires an exact admitted model",
        ),
        (
            {"base_url": "https://api.moonshot.ai/v1/"},
            "requires the exact Moonshot public endpoint",
        ),
        (
            {"api_key": None},
            "requires a configured api_key",
        ),
        (
            {"api_key": "   "},
            "requires a configured api_key",
        ),
    ],
    ids=[
        "enforcing",
        "retired-k3",
        "retired-k27-highspeed",
        "unadmitted-k31-pro-highspeed",
        "gateway-model-id",
        "wrong-endpoint",
        "missing-api-key",
        "blank-api-key",
    ],
)
def test_kimi_diagnostic_writer_static_contract_fails_closed(
    monkeypatch,
    overrides,
    message,
):
    import crucible.backends.config as config_module

    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid route must reject before adapter construction"
        ),
    )

    with pytest.raises(ValueError, match=message):
        resolve_for_role(
            "implementer",
            routing=_kimi_diagnostic_writer_routing(**overrides),
            cwd="/workspace",
            capability="workspace_write",
            native_shell=False,
            allow_trusted_project_instructions=False,
        )


@pytest.mark.parametrize(
    ("call_overrides", "message"),
    [
        ({"native_shell": None}, "native_shell must be disabled"),
        ({"native_shell": True}, "native_shell must be disabled"),
        ({"native_shell": 0}, "native_shell must be disabled"),
        (
            {"allow_trusted_project_instructions": True},
            "trusted project instructions must be disabled",
        ),
        ({"cwd": None}, "explicit non-empty confined cwd is required"),
        ({"cwd": ""}, "explicit non-empty confined cwd is required"),
        ({"cwd": "relative/workspace"}, "confined cwd must be absolute"),
    ],
    ids=[
        "unspecified-shell",
        "native-shell",
        "non-bool-shell",
        "trusted-instructions",
        "missing-cwd",
        "empty-cwd",
        "relative-cwd",
    ],
)
def test_kimi_diagnostic_writer_runtime_boundary_fails_closed(
    monkeypatch,
    call_overrides,
    message,
):
    import crucible.backends.config as config_module

    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid boundary must reject before adapter construction"
        ),
    )
    call = {
        "cwd": "/workspace",
        "native_shell": False,
        "allow_trusted_project_instructions": False,
    }
    call.update(call_overrides)

    with pytest.raises(FeatureAdmissionError, match=message):
        resolve_for_role(
            "implementer",
            routing=_kimi_diagnostic_writer_routing(),
            capability="workspace_write",
            **call,
        )


def test_deepswe_pool_resolves_confined_kimi_diagnostic_writer_exactly(
    tmp_path,
    monkeypatch,
):
    import crucible.backends.config as config_module
    from crucible.tools import ToolContext
    from crucible.tools.profiles import build_autonomous_implementer_profile
    from crucible.workflows._backends import _BackendPool

    captured = {}

    class Backend:
        name = "kimi"

        def attach_toolset(self, toolset, *, context_budget=None):
            captured["toolset"] = toolset
            captured["context_budget"] = context_budget

        def close(self):
            captured["closed"] = True

    def fake_resolve(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return Backend()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    monkeypatch.setenv("MOONSHOT_API_KEY", "never-forward-this-secret")
    root = Path(__file__).parents[1]
    routing = load_routing(
        config_path=(root / "products/deepswe/backends_kimi_sdk_diagnostic.yaml"),
    )
    profile = build_autonomous_implementer_profile(
        ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
            write_scope=("target.py",),
        ),
    )
    log_dir = tmp_path / "logs"
    pool = _BackendPool(routing, log_dir=log_dir, cwd=str(tmp_path))

    backend = pool.get(
        "implementer",
        toolset=profile,
        context_budget=16_384,
        native_shell=False,
    )
    pool.close()

    assert backend.name == "kimi"
    assert captured["name"] == "kimi"
    assert captured["kwargs"] == {
        "log_dir": log_dir,
        "cwd": str(tmp_path),
        "capability": "workspace_write",
        "model": "kimi-k3.1-pro",
        "effort": "max",
        "expected_base_url": "https://api.moonshot.ai/v1",
        "worker_source": "project_locked",
        "diagnostic_cost_upper_bound": True,
        "timeout_seconds": 1_200,
    }
    assert captured["toolset"] is profile
    assert captured["context_budget"] == 16_384
    assert captured["closed"] is True


@pytest.mark.parametrize("admission", ["enforce", "diagnostic"])
@pytest.mark.parametrize(
    "extra",
    [{}, {"diagnostic_workspace_write": False}],
    ids=["control-absent", "control-false"],
)
def test_kimi_workspace_write_is_promotion_blocked_before_construction(
    monkeypatch,
    admission,
    extra,
):
    import crucible.backends.config as config_module

    constructed = False

    def unexpected_resolve(*_args, **_kwargs):
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", unexpected_resolve)
    routing = BackendRouting(
        default=RoleConfig(
            backend="kimi",
            model="kimi-k3.1-pro",
            base_url="https://api.moonshot.ai/v1",
            api_key="must-not-reach-construction",
            feature_admission=admission,
            extra=extra,
        ),
    )

    with pytest.raises(
        FeatureAdmissionError,
        match="workspace_write routing is promotion-blocked",
    ):
        resolve_for_role(
            "implementer",
            routing=routing,
            capability="workspace_write",
            native_shell=False,
        )

    assert constructed is False


def test_resolve_for_role_does_not_forward_kimi_credentials(
    monkeypatch,
):
    import crucible.backends.config as config_module

    captured = {}

    def fake_resolve(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    routing = BackendRouting(
        default=RoleConfig(
            backend="kimi",
            model="kimi-k3.1-pro",
            base_url="https://api.moonshot.ai/v1",
            api_key="never-forward-this-secret",
        ),
    )

    resolve_for_role("implementer", routing=routing)

    assert captured["name"] == "kimi"
    assert captured["kwargs"]["model"] == "kimi-k3.1-pro"
    assert "base_url" not in captured["kwargs"]
    assert "api_key" not in captured["kwargs"]
    assert captured["kwargs"]["expected_base_url"] == "https://api.moonshot.ai/v1"


def test_active_organization_claude_profiles_are_opus_only_and_credential_closed():
    root = Path(__file__).parents[1]
    governed = {
        root / "products/swebench/backends.yaml": (
            "claude-opus-5",
            "claude_session",
        ),
        root / "products/swebench/backends_hybrid_codex_review.yaml": (
            "claude-opus-5",
            "claude_session",
        ),
        root / "products/swebench/backends_claude_opus5_org.yaml": (
            "claude-opus-5",
            "claude_session",
        ),
        root / "products/deepswe/backends_claude_opus5_org.yaml": (
            "claude-opus-5",
            "claude_session",
        ),
    }
    for path, (expected_model, credential_source) in governed.items():
        text = path.read_text(encoding="utf-8").lower()
        assert "sonnet" not in text, path
        routing = lint_routing_config(path)
        claude_configs = [
            routing.default,
            *(
                routing.for_role(role)
                for role in routing.roles
                if routing.for_role(role).backend == "claude"
            ),
        ]
        assert all(config.backend == "claude" for config in claude_configs)
        assert all(config.model == expected_model for config in claude_configs)
        if credential_source is not None:
            assert routing.default.base_url is None
            assert routing.default.api_key is None
            assert all(
                config.extra.get("credential_source") == credential_source
                for config in claude_configs
            )


def test_deepswe_claude_review_overlay_uses_supported_effort_and_truthful_limits():
    from crucible.workflows import deepswe as deepswe_workflow
    from crucible.workflows._development_budget import DevelopmentBudgetLedger

    root = Path(__file__).parents[1]
    path = root / "products/deepswe/backends_claude_opus5_org.yaml"
    routing = lint_routing_config(path)
    review_roles = {
        "architect_planner": (9, 147_456),
        "architect_critic": (21, 688_128),
        "architect_judge": (21, 688_128),
    }

    for role in review_roles:
        config = routing.for_role(role)
        assert config.backend == "claude"
        assert config.model == "claude-opus-5"
        assert config.effort == "medium"
        assert config.base_url is None
        assert config.api_key is None
        assert config.extra == {
            "credential_source": "claude_session",
            "expected_org_id": CLAUDE_NVIDIA_ORG_ID,
            "expected_org_name": CLAUDE_NVIDIA_ORG_NAME,
        }
        assert "max_tokens" not in config.extra
        assert "max_response_tokens" not in config.extra
        assert not any("fallback" in key.lower() for key in config.extra)

    assert routing.for_role("implementer").effort == "high"
    assert (
        feature_profile_for_route(
            "claude",
            model="claude-opus-5",
        ).response_token_limit
        == "unsupported"
    )

    policy = deepswe_workflow._deepswe_development_budget_policy(10.0, planner_count=1)
    allocations = {item.role: item for item in policy.roles}
    for role, (calls, output_tokens) in review_roles.items():
        assert allocations[role].max_calls == calls
        assert allocations[role].max_output_tokens == output_tokens
        assert allocations[role].group == "planning_review"
    for role in ("architect_critic", "architect_judge"):
        assert allocations[role].max_output_tokens == (
            allocations[role].max_calls * 32_768
        )
    assert policy.schema_version == "crucible.development-budget.v3"
    assert policy.planning_review_cap_usd == 4.0
    assert policy.release_unused_planning_authority is True
    assert policy.implementation_verifier_cap_usd == 6.0
    assert {
        role: allocation.protected_tail_usd
        for role, allocation in allocations.items()
        if allocation.protected_tail_usd > 0.0
    } == {
        "impl_breaker": 0.5,
        "breach_defender": 0.3,
        "diff_scope": 0.1,
        "tier_completeness": 0.1,
    }
    assert allocations["diff_scope"].max_cost_usd == 6.0
    assert allocations["tier_completeness"].max_cost_usd == 6.0
    assert {
        role: allocations[role].max_cost_usd
        for role in (
            "implementer",
            "impl_breaker",
            "breach_defender",
            "breach_judge",
            "diff_scope",
            "tier_completeness",
            "api_contract_fidelity",
            "review_conflict_authority",
        )
    } == {
        "implementer": 10.0,
        "impl_breaker": 6.0,
        "breach_defender": 6.0,
        "breach_judge": 6.0,
        "diff_scope": 6.0,
        "tier_completeness": 6.0,
        "api_contract_fidelity": 6.0,
        "review_conflict_authority": 6.0,
    }
    assert (
        DevelopmentBudgetLedger(policy).to_summary()["enforcement"]["output_tokens"]
        == "post_hoc_lockout"
    )


def test_deepswe_codex_planning_overlay_keeps_mutating_roles_max():
    root = Path(__file__).parents[1]
    path = root / "products/deepswe/backends_codex_only.yaml"
    routing = lint_routing_config(path)

    assert routing.default.backend == "codex"
    assert routing.default.model == "gpt-5.6-sol"
    assert routing.default.effort == "max"
    assert routing.default.extra == {
        "credential_source": "chatgpt_session",
    }
    for role in (
        "architect_planner",
        "architect_critic",
        "architect_judge",
    ):
        config = routing.for_role(role)
        assert config.backend == "codex"
        assert config.model == "gpt-5.6-sol"
        assert config.effort == "high"
        assert config.extra == {
            "credential_source": "chatgpt_session",
        }

    for role in (
        "implementer",
        "breach_defender",
        "impl_breaker",
        "breach_judge",
    ):
        assert routing.for_role(role).effort == "max"


def test_role_capability_registry_blocks_escalation():
    with pytest.raises(ValueError, match="permits at most"):
        resolve_for_role(
            "plan_critic",
            routing=DEFAULT_ROUTING,
            capability="workspace_write",
        )


def test_enforcing_feature_admission_rejects_before_backend_construction(
    monkeypatch,
):
    import crucible.backends.config as config_module

    constructed = False

    def fake_resolve(*_args, **_kwargs):
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    requirement = FeatureRequirement(
        "native_delegation",
        frozenset({"writer"}),
    )

    with pytest.raises(FeatureAdmissionError, match="native_delegation"):
        resolve_for_role(
            "implementer",
            routing=DEFAULT_ROUTING,
            feature_requirements=(requirement,),
        )

    assert constructed is False


@pytest.mark.parametrize("capability", ("text", "read_only", "workspace_write"))
@pytest.mark.parametrize("feature_admission", ("enforce", "diagnostic"))
def test_codex_ultra_rejects_unqualified_delegation_before_construction(
    monkeypatch,
    capability,
    feature_admission,
):
    import crucible.backends.config as config_module

    constructed = False

    def fake_resolve(*_args, **_kwargs):
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    routing = BackendRouting(
        default=RoleConfig(
            backend="codex",
            model="gpt-5.6-sol",
            effort="ultra",
            feature_admission=feature_admission,
        ),
    )

    with pytest.raises(
        FeatureAdmissionError,
        match="automatic task delegation",
    ):
        resolve_for_role(
            (
                "implementer"
                if capability == "workspace_write"
                else (
                    "architect_planner"
                    if capability == "read_only"
                    else "diff_scope"
                )
            ),
            routing=routing,
            capability=capability,
        )

    assert constructed is False


def test_reviewer_role_rejects_unqualified_structured_output_route(
    monkeypatch,
):
    import crucible.backends.config as config_module

    constructed = False

    def fake_resolve(*_args, **_kwargs):
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    routing = BackendRouting(
        default=RoleConfig(
            backend="openai_compat",
            model="unqualified-model",
            base_url="https://example.invalid/v1",
        ),
    )

    with pytest.raises(FeatureAdmissionError, match="structured_output"):
        resolve_for_role("plan_critic", routing=routing)

    assert constructed is False


def test_explicit_diagnostic_admission_retains_rejected_route(
    monkeypatch,
    caplog,
):
    import crucible.backends.config as config_module

    sentinel = object()
    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: sentinel,
    )
    routing = BackendRouting(
        default=RoleConfig(
            backend="codex",
            model="gpt-5.6-sol",
            feature_admission="diagnostic",
        ),
    )

    result = resolve_for_role(
        "implementer",
        routing=routing,
        feature_requirements=(
            FeatureRequirement(
                "native_delegation",
                frozenset({"writer"}),
            ),
        ),
    )

    assert result is sentinel
    assert "diagnostic feature admission retained rejected route" in caplog.text


# ----------------------------------------------------------------------
# resolve_for_role: backend-portability hygiene
# ----------------------------------------------------------------------


def test_resolve_for_role_passes_effort_to_codex(tmp_path, monkeypatch):
    """Codex effort is forwarded to app-server TurnOverrides."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    cfg = tmp_path / "c.yaml"
    _write_yaml(
        cfg,
        {
            "roles": {
                "plan_critic": {
                    "backend": "codex",
                    "model": "gpt-5.6-sol",
                    "effort": "max",
                },
            },
        },
    )

    b = resolve_for_role("plan_critic", config_path=cfg)
    assert b.model == "gpt-5.6-sol"
    assert b._effort == "max"


def test_resolve_for_role_can_disable_codex_native_shell():
    b = resolve_for_role(
        "implementer",
        routing=DEFAULT_ROUTING,
        capability="workspace_write",
        native_shell=False,
    )

    assert b.capability == "workspace_write"
    assert b._native_shell is False


def test_resolve_for_role_can_disable_claude_native_shell():
    routing = BackendRouting(
        default=RoleConfig(
            backend="claude",
            model="aws/anthropic/bedrock-claude-opus-4-8",
            effort="high",
        ),
    )

    b = resolve_for_role(
        "implementer",
        routing=routing,
        capability="workspace_write",
        native_shell=False,
    )

    assert b.capability == "workspace_write"
    assert b._native_shell is False


def test_resolve_for_role_translates_effort_to_reasoning_effort_on_openai_compat(
    tmp_path,
    monkeypatch,
):
    """``effort: high`` on an openai_compat role is NOT dropped — it
    translates to the hub's top-level ``reasoning_effort`` param. Both
    published SWE-bench numbers for the hosted reasoning models assume
    high effort, so stripping it (the old behavior) silently ran them
    at the hub default."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    cfg = tmp_path / "c.yaml"
    _write_yaml(
        cfg,
        {
            "roles": {
                "plan_critic": {
                    "backend": "openai_compat",
                    "base_url": "https://inference-api.nvidia.com",
                    "model": "nvidia/deepseek-ai/deepseek-v4-pro",
                    "effort": "high",
                    # This fixture exercises setting translation, not route
                    # qualification; retain the unqualified model diagnostically.
                    "feature_admission": "diagnostic",
                },
            },
        },
    )

    b = resolve_for_role("plan_critic", config_path=cfg)
    assert b.name == "openai_compat"
    assert b._reasoning_effort == "high"


def test_resolve_for_role_binds_claude_endpoint_without_forwarding_secret(
    tmp_path,
    monkeypatch,
):
    import crucible.backends.config as config_module

    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    captured = {}

    def fake_resolve(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    cfg = tmp_path / "c.yaml"
    _write_yaml(
        cfg,
        {
            "roles": {
                "plan_critic": {
                    "backend": "claude",
                    "model": "aws/anthropic/bedrock-claude-opus-4-8",
                    "base_url": "http://localhost:8000/v1",
                    "api_key": "sk-not-applicable",
                },
            },
        },
    )

    resolve_for_role("plan_critic", config_path=cfg)

    assert captured["name"] == "claude"
    assert captured["kwargs"]["expected_base_url"] == "http://localhost:8000/v1"
    assert captured["kwargs"]["prefer_infer_credentials"] is True
    assert "base_url" not in captured["kwargs"]
    assert "api_key" not in captured["kwargs"]


@pytest.mark.parametrize(
    "model",
    ["claude-opus-5", "claude-fable-5-1", "claude-mythos-5-1"],
)
def test_resolve_for_role_binds_direct_claude_session_without_gateway_fields(
    tmp_path,
    monkeypatch,
    model,
):
    import crucible.backends.config as config_module

    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    captured = {}

    def fake_resolve(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    cfg = tmp_path / "direct-claude-session.yaml"
    _write_yaml(
        cfg,
        {
            "default": {
                "backend": "claude",
                "model": model,
                "effort": "high",
                "extra": {
                    "credential_source": "claude_session",
                    "expected_org_id": CLAUDE_NVIDIA_ORG_ID,
                    "expected_org_name": CLAUDE_NVIDIA_ORG_NAME,
                },
            },
        },
    )

    resolve_for_role("plan_critic", config_path=cfg)

    assert captured["name"] == "claude"
    assert captured["kwargs"]["model"] == model
    assert captured["kwargs"]["effort"] == "high"
    assert captured["kwargs"]["credential_source"] == "claude_session"
    assert captured["kwargs"]["expected_org_id"] == CLAUDE_NVIDIA_ORG_ID
    assert captured["kwargs"]["expected_org_name"] == CLAUDE_NVIDIA_ORG_NAME
    assert "expected_base_url" not in captured["kwargs"]
    assert "prefer_infer_credentials" not in captured["kwargs"]
    assert "base_url" not in captured["kwargs"]
    assert "api_key" not in captured["kwargs"]


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-astra"])
def test_resolve_for_role_binds_codex_chatgpt_session(
    tmp_path,
    monkeypatch,
    model,
):
    import crucible.backends.config as config_module

    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    captured = {}

    def fake_resolve(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    cfg = tmp_path / "codex-chatgpt-session.yaml"
    _write_yaml(
        cfg,
        {
            "default": {
                "backend": "codex",
                "model": model,
                "effort": "max",
                "extra": {
                    "credential_source": "chatgpt_session",
                    "timeout_seconds": 600,
                },
            },
        },
    )

    resolve_for_role("architect_planner", config_path=cfg)

    assert captured["name"] == "codex"
    assert captured["kwargs"]["model"] == model
    assert captured["kwargs"]["effort"] == "max"
    assert captured["kwargs"]["credential_source"] == "chatgpt_session"
    assert captured["kwargs"]["timeout_seconds"] == 600


@pytest.mark.parametrize(
    "extra,model,match",
    [
        (
            {"credential_source": "chatgpt_session"},
            "gpt-5.5",
            "requires exact model",
        ),
        (
            {
                "credential_source": "chatgpt_session",
                "transport": "official_sdk",
            },
            "gpt-5.6-sol",
            "requires transport='legacy_app_server'",
        ),
        (
            {"credential_source": "environment", "cli_path": "/bin/true"},
            "gpt-5.6-sol",
            "cli_path requires",
        ),
    ],
)
def test_codex_chatgpt_session_route_rejects_identity_drift(
    tmp_path,
    monkeypatch,
    extra,
    model,
    match,
):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    cfg = tmp_path / "invalid-codex-session.yaml"
    _write_yaml(
        cfg,
        {
            "default": {
                "backend": "codex",
                "model": model,
                "effort": "ultra",
                "extra": extra,
            },
        },
    )

    with pytest.raises(ValueError, match=match):
        load_routing(config_path=cfg)


def test_session_model_authority_sets_remain_aligned():
    from crucible.backends import config as config_module
    from crucible.backends import codex as codex_module
    from crucible.backends._claude_session_transport import (
        CLAUDE_SESSION_FRONTIER_MODELS,
    )
    from crucible.backends.claude import CLAUDE_SESSION_MODELS

    assert config_module._CODEX_CHATGPT_MODELS == (  # noqa: SLF001
        codex_module._CODEX_CHATGPT_SESSION_MODELS  # noqa: SLF001
    )
    assert config_module._CLAUDE_SESSION_MODELS == (  # noqa: SLF001
        CLAUDE_SESSION_MODELS
    )
    assert CLAUDE_SESSION_MODELS == CLAUDE_SESSION_FRONTIER_MODELS


# ----------------------------------------------------------------------
# Switching the default backend switches everything (routing footgun fix)
# ----------------------------------------------------------------------


def test_default_codex_only_switches_all_roles(tmp_path, monkeypatch):
    """A Codex default with no roles routes the whole cascade to Codex."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    cfg = tmp_path / "codex.yaml"
    _write_yaml(cfg, {"default": {"backend": "codex", "model": "gpt-5.5"}})
    routing = load_routing(config_path=cfg)
    for role in (
        "architect_planner",
        "plan_critic",
        "breach_defender",
        "impl_breaker",
        "implementer",
        "diff_scope",
    ):
        rc = routing.for_role(role)
        assert rc.backend == "codex", f"{role} → {rc.backend}, expected codex"
        assert rc.model == "gpt-5.5"


def test_explicit_claude_default_switches_all_roles(tmp_path, monkeypatch):
    """An explicit Claude profile overrides the built-in Codex default."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    cfg = tmp_path / "claude.yaml"
    _write_yaml(
        cfg,
        {
            "default": {
                "backend": "claude",
                "model": "claude-opus-4-8",
                "effort": "high",
            }
        },
    )
    routing = load_routing(config_path=cfg)
    for role in ("plan_critic", "breach_defender", "implementer"):
        selected = routing.for_role(role)
        assert selected.backend == "claude"
        assert selected.model == "claude-opus-4-8"
        assert selected.effort == "high"


def test_backend_switch_requires_explicit_model(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    cfg = tmp_path / "incomplete-claude.yaml"
    _write_yaml(cfg, {"default": {"backend": "claude"}})

    with pytest.raises(ValueError, match="must declare a model"):
        load_routing(config_path=cfg)


def test_explicit_role_override_wins_over_switched_default(tmp_path, monkeypatch):
    """Mixing still works: a codex default with one role pinned back to
    claude keeps that role on claude and everything else on codex."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    cfg = tmp_path / "mixed.yaml"
    _write_yaml(
        cfg,
        {
            "default": {"backend": "codex", "model": "gpt-5.5"},
            "roles": {
                "plan_critic": {
                    "backend": "claude",
                    "model": "aws/anthropic/bedrock-claude-sonnet-4-6",
                },
            },
        },
    )
    routing = load_routing(config_path=cfg)
    critic = routing.for_role("plan_critic")
    assert critic.backend == "claude"
    assert critic.effort is None  # Codex Max must not leak across backends.
    assert routing.for_role("breach_defender").backend == "codex"
    assert routing.for_role("implementer").backend == "codex"
