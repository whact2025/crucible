# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""K3 config admission is explicit and never broadens writer authority."""

from __future__ import annotations

import pytest

from assay import CONSERVATIVE_BACKEND_FEATURES
from crucible.backends import config as config_module
from crucible.backends.config import BackendRouting, RoleConfig, resolve_for_role
from crucible.backends.features import FeatureAdmissionError, feature_profile_for_route


def _routing(**overrides):
    values = {
        "backend": "kimi",
        "model": "kimi-k3",
        "effort": "max",
        "base_url": "https://api.moonshot.ai/v1",
        "api_key": "test-credential-not-for-constructor",
        "feature_admission": "diagnostic",
        "extra": {
            "diagnostic_workspace_write": True,
            "worker_source": "project_locked",
            "max_response_tokens": 32_768,
        },
    }
    values.update(overrides)
    return BackendRouting(default=RoleConfig(**values))


def _resolve(routing, **overrides):
    kwargs = {
        "cwd": "/confined/workspace",
        "capability": "workspace_write",
        "native_shell": False,
        "allow_trusted_project_instructions": False,
    }
    kwargs.update(overrides)
    return resolve_for_role("implementer", routing=routing, **kwargs)


def test_k3_confined_writer_passes_exact_model_effort_without_credentials(monkeypatch):
    captured = {}

    def fake_resolve(name, **kwargs):
        captured.update(name=name, **kwargs)
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    _resolve(_routing())
    assert captured == {
        "name": "kimi",
        "log_dir": None,
        "cwd": "/confined/workspace",
        "capability": "workspace_write",
        "model": "kimi-k3",
        "effort": "max",
        "expected_base_url": "https://api.moonshot.ai/v1",
        "worker_source": "project_locked",
        "max_response_tokens": 32_768,
    }


def test_conservative_cost_option_is_explicit_and_forwarded(monkeypatch):
    monkeypatch.setattr(config_module, "_resolve_backend", lambda *args, **kwargs: kwargs)
    config = _routing()
    from dataclasses import replace
    routing = BackendRouting(default=replace(config.default, extra={
        **config.default.extra, "diagnostic_cost_upper_bound": True,
    }))
    assert _resolve(routing)["diagnostic_cost_upper_bound"] is True
    assert "diagnostic_cost_upper_bound" not in _resolve(config)


@pytest.mark.parametrize("changes", [
    {"model": "kimi-k2.7-code-highspeed", "effort": None},
    {"effort": None}, {"feature_admission": "enforce"},
    {"base_url": "https://unqualified.invalid/v1"},
])
def test_conservative_cost_option_does_not_expand_routes(changes):
    from dataclasses import replace
    config = _routing(**changes)
    with pytest.raises(ValueError, match="diagnostic_cost_upper_bound"):
        config_module._validate_routing(BackendRouting(default=replace(config.default, extra={
            "worker_source": "project_locked", "diagnostic_cost_upper_bound": True,
        })))


@pytest.mark.parametrize("value", [1, "true", None])
def test_conservative_cost_option_rejects_coerced_booleans(value):
    from dataclasses import replace
    config = _routing()
    with pytest.raises(ValueError, match="diagnostic_cost_upper_bound must be a bool"):
        config_module._validate_routing(BackendRouting(default=replace(config.default, extra={
            **config.default.extra, "diagnostic_cost_upper_bound": value,
        })))


@pytest.mark.parametrize("effort", ["low", "high", "ultra", "", "MAX"])
def test_k3_unqualified_effort_rejects_before_constructor(monkeypatch, effort):
    monkeypatch.setattr(
        config_module, "_resolve_backend", lambda *a, **kw: pytest.fail("constructed"),
    )
    with pytest.raises(ValueError, match="supported only as 'max'"):
        _resolve(_routing(effort=effort))


@pytest.mark.parametrize("tokens", [True, 0, -1, 1_048_577, 1.5, "32768"])
def test_k3_response_limit_rejects_invalid_or_out_of_context_values(monkeypatch, tokens):
    monkeypatch.setattr(
        config_module, "_resolve_backend", lambda *a, **kw: pytest.fail("constructed"),
    )
    with pytest.raises(ValueError, match="from 1 through 1048576"):
        _resolve(_routing(extra={"max_response_tokens": tokens}))


@pytest.mark.parametrize("tokens", [1, 32_768, 262_145, 1_048_576])
def test_k3_response_limit_uses_model_context(monkeypatch, tokens):
    monkeypatch.setattr(config_module, "_resolve_backend", lambda *a, **kw: kw)
    result = _resolve(
        _routing(extra={"max_response_tokens": tokens}), capability="text",
    )
    assert result["max_response_tokens"] == tokens


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"native_shell": True}, "native_shell must be disabled"),
        ({"allow_trusted_project_instructions": True}, "instructions must be disabled"),
        ({"cwd": "relative"}, "cwd must be absolute"),
    ],
)
def test_k3_writer_keeps_confined_boundary(monkeypatch, overrides, message):
    monkeypatch.setattr(
        config_module, "_resolve_backend", lambda *a, **kw: pytest.fail("constructed"),
    )
    with pytest.raises(FeatureAdmissionError, match=message):
        _resolve(_routing(), **overrides)


@pytest.mark.parametrize("extra", [{}, {"diagnostic_workspace_write": False}])
def test_k3_writer_without_explicit_opt_in_remains_blocked(monkeypatch, extra):
    monkeypatch.setattr(
        config_module, "_resolve_backend", lambda *a, **kw: pytest.fail("constructed"),
    )
    with pytest.raises(FeatureAdmissionError, match="promotion-blocked"):
        _resolve(_routing(extra=extra))


@pytest.mark.parametrize(
    ("model", "endpoint"),
    [
        ("kimi-k3", "https://unqualified.example/v1"),
        ("kimi-k3-latest", "https://api.moonshot.ai/v1"),
        ("kimi-k3", "https://api.moonshot.ai/v1?route=other"),
        ("kimi-k3", "http://api.moonshot.ai/v1"),
    ],
)
def test_k3_features_do_not_admit_aliases_or_other_endpoints(model, endpoint):
    assert feature_profile_for_route(
        "kimi", model=model, base_url=endpoint,
    ) == CONSERVATIVE_BACKEND_FEATURES
