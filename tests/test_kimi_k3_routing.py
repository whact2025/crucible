# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""K3-series config admission is explicit and never broadens writer authority."""

from __future__ import annotations

import pytest

from assay import CONSERVATIVE_BACKEND_FEATURES
from crucible.backends import config as config_module
from crucible.backends import kimi as parent
from crucible.backends.config import BackendRouting, RoleConfig, resolve_for_role
from crucible.backends.features import FeatureAdmissionError, feature_profile_for_route
from crucible.backends.pricing import DEFAULT_PRICING
from crucible.tests import test_backends_kimi as kimi_fixtures

worker = kimi_fixtures.worker_mod
PRO = "kimi-k3.1-pro"
# A current Moonshot id that the dedicated route does not admit.
HIGHSPEED = "kimi-k3.1-pro-highspeed"
_MOONSHOT = "https://api.moonshot.ai/v1"


def _routing(**overrides):
    values = {
        "backend": "kimi",
        "model": PRO,
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
        "model": PRO,
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


def test_conservative_cost_option_stays_with_the_policy_model(monkeypatch):
    # The admitted set and the policy's model are separate tables: a second
    # admitted id would be routable and would still not carry the bound.
    from dataclasses import replace
    other = "kimi-second-admitted-id"
    monkeypatch.setitem(config_module._KIMI_MAX_CONTEXT_SIZES, other, 1_048_576)
    config = _routing(model=other)
    config_module._validate_routing(BackendRouting(default=replace(
        config.default, extra={"worker_source": "project_locked"},
    )))
    with pytest.raises(ValueError, match="diagnostic_cost_upper_bound requires exact"):
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
        (PRO, "https://unqualified.example/v1"),
        (HIGHSPEED, _MOONSHOT),
        ("kimi-k3.1-pro-latest", _MOONSHOT),
        ("kimi-k3", _MOONSHOT),
        (PRO, "https://api.moonshot.ai/v1?route=other"),
        (PRO, "http://api.moonshot.ai/v1"),
    ],
)
def test_k3_features_do_not_admit_aliases_or_other_endpoints(model, endpoint):
    assert feature_profile_for_route(
        "kimi", model=model, base_url=endpoint,
    ) == CONSERVATIVE_BACKEND_FEATURES


@pytest.mark.parametrize(
    "unadmitted",
    ["kimi-k2.7-code-highspeed", "kimi-k2.7-code", "kimi-k3", HIGHSPEED],
)
def test_unadmitted_kimi_ids_are_refused_by_every_table(unadmitted):
    # The retired K2 series and kimi-k3 still answer at the provider.
    # kimi-k3.1-pro-highspeed is a current id that had strict accounting
    # only: Moonshot omits cached_tokens when nothing was read from cache
    # (observed 2026-11-18), so each of its replies in that check was billed
    # and then rejected. Only the closed tables can show that none of these
    # ids is selectable and that no alias remains.
    with pytest.raises(ValueError, match="supports only the qualified model"):
        parent.KimiBackend(model=unadmitted)
    payload = kimi_fixtures._worker_request_payload(
        model=unadmitted, reasoning_effort="max",
    )
    with pytest.raises(worker.WorkerQualificationError, match="unqualified model/effort"):
        worker._Request.from_wire(payload)
    with pytest.raises(ValueError, match="exact admitted Kimi model"):
        config_module._validate_routing(_routing(
            model=unadmitted, effort=None, extra={"worker_source": "project_locked"},
        ))
    with pytest.raises(ValueError, match="does not support that setting"):
        config_module._validate_routing(_routing(
            model=unadmitted, extra={"worker_source": "project_locked"},
        ))
    assert feature_profile_for_route(
        "kimi", model=unadmitted, base_url=_MOONSHOT,
    ) == CONSERVATIVE_BACKEND_FEATURES
    assert unadmitted not in DEFAULT_PRICING


def test_admitted_kimi_model_tables_agree():
    admitted = {PRO}
    assert set(parent._MODEL_CONTEXT_SIZES) == admitted
    assert set(worker._MODEL_CONTEXT_SIZES) == admitted
    assert set(config_module._KIMI_MAX_CONTEXT_SIZES) == admitted
    assert config_module._KIMI_DIAGNOSTIC_MODELS == admitted
    assert (parent._MODEL, worker._MODEL) == (PRO, PRO)
    for model in sorted(admitted):
        assert parent._MODEL_CONTEXT_SIZES[model] == 1_048_576
        assert worker._MODEL_CONTEXT_SIZES[model] == 1_048_576
        assert config_module._KIMI_MAX_CONTEXT_SIZES[model] == 1_048_576
        assert model in DEFAULT_PRICING
        profile = feature_profile_for_route("kimi", model=model, base_url=_MOONSHOT)
        assert profile != CONSERVATIVE_BACKEND_FEATURES
        assert profile.response_token_limit == "provider_stop"
