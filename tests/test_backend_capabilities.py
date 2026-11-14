# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from crucible.backends.capabilities import validate_toolset_capability
from crucible.backends.config import BackendRouting, RoleConfig
from crucible.backends.features import FeatureAdmissionError, FeatureRequirement
from crucible.workflows._backends import _BackendPool


class _Toolset:
    def __init__(self, writes=()):
        self.write_names = set(writes)


def test_text_profile_rejects_every_toolset():
    with pytest.raises(ValueError, match="text capability"):
        validate_toolset_capability("text", _Toolset())


def test_read_only_profile_rejects_mutating_toolset():
    with pytest.raises(ValueError, match="mutating tools"):
        validate_toolset_capability("read_only", _Toolset({"apply_patch"}))


def test_read_only_profile_accepts_classified_read_tools():
    validate_toolset_capability("read_only", _Toolset())


def test_unclassified_toolset_fails_closed():
    with pytest.raises(ValueError, match="write_names metadata"):
        validate_toolset_capability("read_only", object())


def test_backend_pool_forwards_native_shell_policy(monkeypatch):
    captured = []

    class Backend:
        name = "stub"

        def attach_toolset(self, toolset, *, context_budget=None):
            self.toolset = toolset

        def close(self):
            pass

    def resolve(role, **kwargs):
        captured.append((role, kwargs))
        return Backend()

    monkeypatch.setattr("crucible.workflows._backends.resolve_for_role", resolve)
    pool = _BackendPool(object(), log_dir=None, cwd="/workspace")

    backend = pool.get(
        "implementer", toolset=_Toolset({"apply_patch"}), native_shell=False,
    )

    assert backend.toolset.write_names == {"apply_patch"}
    assert captured == [(
        "implementer",
        {
            "routing": pool.routing,
            "log_dir": None,
            "cwd": "/workspace",
            "capability": "workspace_write",
            "native_shell": False,
            "allow_trusted_project_instructions": False,
        },
    )]


def test_backend_pool_does_not_reuse_native_shell_variant(monkeypatch):
    created = []

    class Backend:
        name = "stub"

        def close(self):
            pass

    def resolve(_role, **_kwargs):
        backend = Backend()
        created.append(backend)
        return backend

    monkeypatch.setattr("crucible.workflows._backends.resolve_for_role", resolve)
    pool = _BackendPool(object(), log_dir=None, cwd=None)

    host = pool.get(
        "implementer", capability="workspace_write", native_shell=None,
    )
    confined = pool.get(
        "implementer", capability="workspace_write", native_shell=False,
    )

    assert host is not confined
    assert created == [host, confined]


def test_backend_pool_forwards_and_keys_workflow_feature_requirements(
    monkeypatch,
):
    captured = []

    class Backend:
        name = "stub"

    def resolve(_role, **kwargs):
        captured.append(kwargs)
        return Backend()

    monkeypatch.setattr("crucible.workflows._backends.resolve_for_role", resolve)
    pool = _BackendPool(object(), log_dir=None, cwd=None)
    requirements = (
        FeatureRequirement(
            "structured_output",
            frozenset({"native", "prompt_only"}),
        ),
    )

    qualified = pool.get(
        "implementer",
        feature_requirements=requirements,
    )
    baseline = pool.get("implementer")

    assert qualified is not baseline
    assert captured[0]["feature_requirements"] == requirements
    assert "feature_requirements" not in captured[1]


def test_backend_pool_admits_conversation_mode_before_construction(
    monkeypatch,
):
    captured = []

    class Backend:
        name = "stub"

    def resolve(_role, **kwargs):
        captured.append(kwargs)
        return Backend()

    monkeypatch.setattr("crucible.workflows._backends.resolve_for_role", resolve)
    pool = _BackendPool(object(), log_dir=None, cwd=None)

    workflow = pool.get(
        "implementer",
        conversation_mode="workflow_bound",
    )
    retry = pool.get(
        "implementer",
        conversation_mode="retry_only",
    )
    fresh = pool.get("implementer", conversation_mode="none")

    assert workflow is not retry
    assert retry is not fresh
    workflow_requirement = captured[0]["feature_requirements"][0]
    assert workflow_requirement.field == "session_continuity"
    assert workflow_requirement.accepted == frozenset({"workflow_bound"})
    retry_requirement = captured[1]["feature_requirements"][0]
    assert retry_requirement.accepted == frozenset({
        "retry_only",
        "workflow_bound",
    })
    assert "feature_requirements" not in captured[2]


def test_backend_pool_rejects_unknown_conversation_mode_before_resolve(
    monkeypatch,
):
    def unexpected_resolve(*_args, **_kwargs):
        pytest.fail("unsupported mode must reject before backend construction")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        unexpected_resolve,
    )
    pool = _BackendPool(object(), log_dir=None, cwd=None)

    with pytest.raises(ValueError, match="unsupported conversation mode"):
        pool.get("implementer", conversation_mode="durable")  # type: ignore[arg-type]


def test_exclusive_writer_pool_cannot_bypass_kimi_promotion_gate(
    tmp_path,
    monkeypatch,
):
    import crucible.backends.config as config_module

    constructed = False

    def unexpected_resolve(*_args, **_kwargs):
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", unexpected_resolve)
    pool = _BackendPool(
        BackendRouting(
            default=RoleConfig(
                backend="kimi",
                model="kimi-k2.7-code-highspeed",
                base_url="https://api.moonshot.ai/v1",
                api_key="must-not-reach-construction",
                feature_admission="diagnostic",
            ),
        ),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )

    with pytest.raises(
        FeatureAdmissionError,
        match="workspace_write routing is promotion-blocked",
    ):
        pool.new_exclusive_writer(
            "implementer",
            toolset=_Toolset({"apply_patch"}),
            log_dir=tmp_path / "writer",
            cwd=str(tmp_path),
        )

    assert constructed is False
