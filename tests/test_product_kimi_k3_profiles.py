# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""All-K3 product profiles are explicit, confined, and separate from old cohorts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from crucible.backends import config as config_module
from crucible.backends.config import (
    DEFAULT_ROUTING,
    lint_routing_config,
    load_routing,
    resolve_for_role,
    routing_fingerprint,
)
from crucible.backends.features import FeatureAdmissionError
from crucible.products.deepswe.backend_convergence import (
    load_deepswe_backend_profile_catalog,
)
from crucible.products.swebench.backend_convergence import (
    load_swebench_backend_profile_catalog,
)


ROOT = Path(__file__).parents[1]
PROFILE = "kimi_k3_sdk_diagnostic"
WRITERS = {"implementer", "breach_defender"}
LOADERS = {
    "swebench": load_swebench_backend_profile_catalog,
    "deepswe": load_deepswe_backend_profile_catalog,
}


def _path(product: str) -> Path:
    return ROOT / "products" / product / "backends_kimi_k3_sdk_diagnostic.yaml"


@pytest.fixture(autouse=True)
def _provider_free_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    monkeypatch.setenv("MOONSHOT_API_KEY", "provider-free-test-token")


@pytest.mark.parametrize("product", LOADERS)
def test_k3_profiles_pin_every_role_and_limit_writer_opt_in(product: str) -> None:
    raw = yaml.safe_load(_path(product).read_text(encoding="utf-8"))
    routing = lint_routing_config(_path(product))
    catalog = LOADERS[product]()
    assert set(raw["roles"]) == {
        *catalog.contract.roles,
        "plan_planner",
    }

    for role, route in {"default": raw["default"], **raw["roles"]}.items():
        assert route["backend"] == "kimi"
        assert route["model"] == "kimi-k3"
        assert route["effort"] == "max"
        assert route["feature_admission"] == "diagnostic"
        assert route["base_url"] == "${KIMI_BASE_URL}"
        assert route["api_key"] == "${MOONSHOT_API_KEY}"
        expected_extra: dict[str, Any] = {
            "worker_source": "project_locked",
            "max_response_tokens": 32_768,
        }
        if role in WRITERS:
            expected_extra["diagnostic_workspace_write"] = True
        if product == "deepswe" and role == "implementer":
            expected_extra["timeout_seconds"] = 1200
        assert route["extra"] == expected_extra
        effective = routing.default if role == "default" else routing.for_role(role)
        assert effective.extra == expected_extra


@pytest.mark.parametrize("product", LOADERS)
def test_k3_catalog_lane_is_new_non_promotional_identity(product: str) -> None:
    catalog = LOADERS[product]()
    lane = catalog.lane(PROFILE)
    old_lane = catalog.lane("kimi_sdk_diagnostic")
    assert lane.routing_path == _path(product).resolve()
    assert lane.evidence_status == "diagnostic"
    assert lane.promotion_eligible is False
    assert lane.promotion_authority_digest is None
    for route in (lane.default_route, *lane.role_routes):
        assert route.model == "kimi-k3"
        assert route.effort == "max"
    assert lane.routing_digest != old_lane.routing_digest
    assert lane.routing_fingerprint != old_lane.routing_fingerprint
    assert lane.routing_fingerprint == routing_fingerprint(
        lint_routing_config(_path(product))
    )
    assert old_lane.default_route.model == "kimi-k2.7-code-highspeed"
    assert DEFAULT_ROUTING.default.backend == "codex"


@pytest.mark.parametrize("product", LOADERS)
def test_k3_product_route_resolves_exact_bounded_kwargs_without_credentials(
    product: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_resolve(name: str, **kwargs: Any) -> object:
        calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    routing = load_routing(config_path=_path(product))
    for role in routing.roles:
        capability = (
            "workspace_write"
            if role in WRITERS
            else "read_only"
            if role.endswith("planner")
            else "text"
        )
        resolve_for_role(
            role,
            routing=routing,
            capability=capability,
            native_shell=False,
            allow_trusted_project_instructions=False,
            cwd=str(tmp_path.resolve()),
        )
        name, kwargs = calls[-1]
        assert name == "kimi"
        assert kwargs["model"] == "kimi-k3"
        assert kwargs["effort"] == "max"
        assert kwargs["max_response_tokens"] == 32_768
        assert kwargs["worker_source"] == "project_locked"
        assert kwargs["expected_base_url"] == "https://api.moonshot.ai/v1"
        assert kwargs["capability"] == capability
        assert kwargs["cwd"] == str(tmp_path.resolve())
        assert "api_key" not in kwargs
        assert "diagnostic_workspace_write" not in kwargs
        if product == "deepswe" and role == "implementer":
            assert kwargs["timeout_seconds"] == 1200
        else:
            assert "timeout_seconds" not in kwargs
    assert len(calls) == len(routing.roles)


@pytest.mark.parametrize("product", LOADERS)
@pytest.mark.parametrize("writer", sorted(WRITERS))
@pytest.mark.parametrize("mutation", ["remove_opt_in", "inherit_default"])
def test_k3_writer_cannot_inherit_authority_from_default(
    product: str,
    writer: str,
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = yaml.safe_load(_path(product).read_text(encoding="utf-8"))
    if mutation == "remove_opt_in":
        raw["roles"][writer]["extra"].pop("diagnostic_workspace_write")
    else:
        raw["roles"].pop(writer)
    routing = lint_routing_config(yaml.safe_dump(raw).encode("utf-8"))

    def forbidden_resolve(*_args: Any, **_kwargs: Any) -> object:
        pytest.fail("an unapproved writer must not construct a provider")

    monkeypatch.setattr(config_module, "_resolve_backend", forbidden_resolve)
    with pytest.raises(FeatureAdmissionError, match="promotion-blocked"):
        resolve_for_role(
            writer,
            routing=routing,
            capability="workspace_write",
            native_shell=False,
            allow_trusted_project_instructions=False,
            cwd=str(tmp_path.resolve()),
        )


@pytest.mark.parametrize(
    ("path", "digest"),
    [
        (
            "swebench/backends_kimi_sdk.yaml",
            "14a820c0ca07db86c5bdbbc5d475bfb790e5d1df7e91e46d1590004773465e5b",
        ),
        (
            "swebench/backends_kimi_sdk_diagnostic.yaml",
            "2dbe69473697fb4925db607bad605d307bb289728ea8174bfc2b4bcaa2c5efef",
        ),
        (
            "swebench/e12_arms.yaml",
            "80adc4a4fb1e108b801e7c1ab40684a22ea3fbf666a1baecd9911f41ec609bc2",
        ),
        (
            "swebench/backend_convergence_matrix.yaml",
            "7d9dd1815876569111e5b02ae48603392634e96005ae7a747cb2880da8e6bc55",
        ),
        (
            "swebench/backend_convergence_matrix_astra_mythos51.yaml",
            "f17e597da7b034a232cd9c29cd3cb9c982227f791aeccb52212fb9899947ee8e",
        ),
        (
            "deepswe/backends_kimi_sdk.yaml",
            "91e7e0a040180e1def994a730bbe89fa69021cc2b890d38956541cb15536c7a4",
        ),
        (
            "deepswe/backends_kimi_sdk_diagnostic.yaml",
            "0e772e66620c898592eb6ea48cbaf976a6de561363cfca8f370db01bf81a2c4b",
        ),
        (
            "deepswe/e12_arms.yaml",
            "5808b716c412db89501b08f285072b3daf66925c5703278bd359fddfbaeb58f6",
        ),
        (
            "deepswe/backend_convergence_matrix.yaml",
            "26323f98a6510e0a24626adf3b0aea08ea481e9484252712aea92880faf0082d",
        ),
    ],
)
def test_profiles_and_matrices_bind_exact_reviewed_bytes(
    path: str,
    digest: str,
) -> None:
    assert hashlib.sha256((ROOT / "products" / path).read_bytes()).hexdigest() == digest
