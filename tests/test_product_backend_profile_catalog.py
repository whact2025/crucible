# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Provider-free coverage for product backend convergence profiles."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from assay import Invocation, InvocationResult
from crucible.backends import BACKEND_NAMES
from crucible.backends import config as config_module
from crucible.backends import openai_compat as openai_compat_module
from crucible.backends.config import (
    lint_routing_config,
    load_routing,
    resolve_for_role,
)
from crucible.backends.features import FeatureAdmissionError
from crucible.backends.pricing import DEFAULT_PRICING, ModelPricing
from crucible.tools import ToolContext
from crucible.tools.profiles import (
    build_autonomous_implementer_profile,
    build_breach_defender_profile,
)
from crucible.tools.source import build_source_tools
from crucible.workflows import _review as review_module
from crucible.workflows._backends import _BackendPool


ROOT = Path(__file__).parents[1]
CATALOG_SCHEMA = "crucible.backend-profile-catalog.v1"

PRODUCT_ROLES = {
    "swebench": (
        "architect_planner",
        "architect_critic",
        "architect_judge",
        "implementer",
        "breach_defender",
        "swebench_impl_breaker",
        "breach_judge",
        "diff_scope",
        "tier_completeness",
        "review_conflict_authority",
    ),
    "deepswe": (
        "architect_planner",
        "architect_critic",
        "architect_judge",
        "implementer",
        "breach_defender",
        "impl_breaker",
        "breach_judge",
        "diff_scope",
        "tier_completeness",
        "api_contract_fidelity",
        "review_conflict_authority",
    ),
}

EXPECTED_PROFILES = {
    "codex_astra": {
        "routing": "backends_codex_astra.yaml",
        "backend": "codex",
        "model": "gpt-6-astra",
        "transport": "legacy_app_server",
        "feature_admission": "enforce",
        "availability": "runtime_required",
        "runtime_binding": "chatgpt_session",
    },
    "codex_official_sdk_diagnostic": {
        "routing": "backends_codex_official_sdk_diagnostic.yaml",
        "backend": "codex",
        "model": "gpt-5.6-sol",
        "transport": "official_sdk",
        "feature_admission": "diagnostic",
        "availability": "diagnostic",
        "runtime_binding": "project_locked",
    },
    "claude_fable51": {
        "routing": "backends_claude_fable51_org.yaml",
        "backend": "claude",
        "model": "claude-fable-5-1",
        "transport": "claude_agent_sdk",
        "feature_admission": "enforce",
        "availability": "runtime_required",
        "runtime_binding": "claude_session",
    },
    "claude_mythos51": {
        "routing": "backends_claude_mythos51_org.yaml",
        "backend": "claude",
        "model": "claude-mythos-5-1",
        "transport": "claude_agent_sdk",
        "feature_admission": "enforce",
        "availability": "runtime_required",
        "runtime_binding": "claude_session",
    },
    "deepseek_responses_diagnostic": {
        "routing": "backends_deepseek_diagnostic.yaml",
        "backend": "deepseek",
        "model": "deepseek-flash",
        "transport": "native_responses",
        "feature_admission": "diagnostic",
        "availability": "diagnostic",
        "runtime_binding": "run_local_required",
    },
    "kimi_sdk": {
        "routing": "backends_kimi_sdk.yaml",
        "backend": "kimi",
        "model": "kimi-k3.1-pro",
        "transport": "kimi_agent_sdk",
        "feature_admission": "enforce",
        "availability": "blocked",
        "runtime_binding": "project_locked",
    },
    "kimi_sdk_diagnostic": {
        "routing": "backends_kimi_sdk_diagnostic.yaml",
        "backend": "kimi",
        "model": "kimi-k3.1-pro",
        "transport": "kimi_agent_sdk",
        "feature_admission": "diagnostic",
        "availability": "diagnostic",
        "runtime_binding": "project_locked",
    },
    "kimi_k31_pro_sdk_diagnostic": {
        "routing": "backends_kimi_k31_pro_sdk_diagnostic.yaml",
        "backend": "kimi",
        "model": "kimi-k3.1-pro",
        "transport": "kimi_agent_sdk",
        "feature_admission": "diagnostic",
        "availability": "diagnostic",
        "runtime_binding": "project_locked",
    },
    "openai_glm53_diagnostic": {
        "routing": "backends_inference_hub.yaml",
        "backend": "openai_compat",
        "model": "nvidia/zai-org/glm-5.3",
        "transport": "openai_chat_completions",
        "feature_admission": "diagnostic",
        "availability": "diagnostic",
        "runtime_binding": "configured_endpoint",
    },
    "openai_kimi_k31_pro_diagnostic": {
        "routing": "backends_kimi_inference.yaml",
        "backend": "openai_compat",
        "model": "nvidia/moonshotai/kimi-k3.1-pro",
        "transport": "openai_chat_completions",
        "feature_admission": "diagnostic",
        "availability": "diagnostic",
        "runtime_binding": "configured_endpoint",
    },
    "local_glm53_diagnostic": {
        "routing": "backends_local_glm53_diagnostic.yaml",
        "backend": "local",
        "model": "nvidia/zai-org/glm-5.3",
        "transport": "openai_chat_completions",
        "feature_admission": "diagnostic",
        "availability": "diagnostic",
        "runtime_binding": "configured_endpoint",
    },
}

PROFILE_NAMES = set(EXPECTED_PROFILES)

PROFILE_FIELDS = {
    "routing",
    "backend",
    "model",
    "transport",
    "feature_admission",
    "availability",
    "runtime_binding",
    "promotion_evidence",
}

NON_EXECUTABLE_AVAILABILITY = {
    "blocked",
    "construction_only",
    "diagnostic",
    "runtime_required",
}

USABLE_WORKFLOW_PROFILES = (
    ("claude_fable51", "claude", "claude_agent_sdk"),
    ("claude_mythos51", "claude", "claude_agent_sdk"),
    ("codex_astra", "codex", "legacy_app_server"),
    ("codex_official_sdk_diagnostic", "codex", "official_sdk"),
    ("deepseek_responses_diagnostic", "deepseek", "native_responses"),
    ("kimi_sdk_diagnostic", "kimi", "kimi_agent_sdk"),
    ("kimi_k31_pro_sdk_diagnostic", "kimi", "kimi_agent_sdk"),
    ("local_glm53_diagnostic", "local", "openai_chat_completions"),
    ("openai_glm53_diagnostic", "openai_compat", "openai_chat_completions"),
)


def _catalog(product: str) -> dict[str, Any]:
    path = ROOT / "products" / product / "backend_profile_catalog.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _capability(role: str) -> str:
    if role == "architect_planner":
        return "read_only"
    if role in {"implementer", "breach_defender"}:
        return "workspace_write"
    return "text"


@pytest.mark.parametrize("product", tuple(PRODUCT_ROLES))
def test_product_catalog_is_closed_complete_and_non_promotional(product: str) -> None:
    document = _catalog(product)

    assert set(document) == {"schema_version", "product", "roles", "profiles"}
    assert document["schema_version"] == CATALOG_SCHEMA
    assert document["product"] == product
    assert tuple(document["roles"]) == PRODUCT_ROLES[product]
    assert set(document["profiles"]) == PROFILE_NAMES
    assert {entry["backend"] for entry in document["profiles"].values()} == set(
        BACKEND_NAMES
    )
    for backend in BACKEND_NAMES:
        assert any(
            entry["backend"] == backend and isinstance(entry["routing"], str)
            for entry in document["profiles"].values()
        )

    for name, entry in document["profiles"].items():
        assert set(entry) == PROFILE_FIELDS, name
        assert entry == {**EXPECTED_PROFILES[name], "promotion_evidence": False}
        assert entry["backend"] in BACKEND_NAMES
        assert entry["feature_admission"] in {"enforce", "diagnostic"}
        assert entry["availability"] in NON_EXECUTABLE_AVAILABILITY
        assert entry["promotion_evidence"] is False


@pytest.mark.parametrize("product", tuple(PRODUCT_ROLES))
def test_every_catalogued_model_has_an_explicit_default_price(product: str) -> None:
    # Pricing admission fails closed on an unpriced routed model. A profile
    # moved to a new model id must therefore bring its own price row.
    for name, entry in _catalog(product)["profiles"].items():
        assert entry["model"] in DEFAULT_PRICING, (product, name, entry["model"])


def test_product_catalogs_have_symmetric_adapter_and_transport_inventory() -> None:
    swebench = _catalog("swebench")["profiles"]
    deepswe = _catalog("deepswe")["profiles"]

    assert set(swebench) == set(deepswe) == PROFILE_NAMES
    for name in PROFILE_NAMES:
        assert swebench[name] == deepswe[name]


@pytest.mark.parametrize("product", tuple(PRODUCT_ROLES))
def test_catalogued_routes_pin_every_product_role_without_secrets(product: str) -> None:
    product_dir = ROOT / "products" / product
    catalog = _catalog(product)
    allowed_api_key_refs = {"${INFER_AUTH_TOKEN}", "${MOONSHOT_API_KEY}"}
    allowed_base_url_refs = {
        "${CRUCIBLE_LOCAL_URL}",
        "${INFER_BASE_URL}",
        "${KIMI_BASE_URL}",
    }

    for name, entry in catalog["profiles"].items():
        routing_name = entry["routing"]
        assert Path(routing_name).name == routing_name
        path = product_dir / routing_name
        assert path.is_file(), name
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        routing = lint_routing_config(path)

        assert raw["schema_version"] == "crucible.backend-routing.v1"
        assert set(PRODUCT_ROLES[product]).issubset(raw["roles"]), name
        assert routing.default.backend == entry["backend"]
        assert routing.default.model == entry["model"]
        assert routing.default.feature_admission == entry["feature_admission"]

        for role in PRODUCT_ROLES[product]:
            explicit = raw["roles"][role]
            effective = routing.for_role(role)
            assert explicit["backend"] == entry["backend"], (name, role)
            assert explicit["model"] == entry["model"], (name, role)
            assert explicit["feature_admission"] == entry["feature_admission"], (
                name,
                role,
            )
            assert effective.backend == entry["backend"], (name, role)
            assert effective.model == entry["model"], (name, role)
            assert effective.feature_admission == entry["feature_admission"], (
                name,
                role,
            )
            for key in ("effort", "base_url", "api_key"):
                if key in raw["default"]:
                    assert explicit[key] == raw["default"][key], (name, role, key)
            default_extra = set(raw["default"].get("extra", {}))
            assert (default_extra - {"expected_served_model"}).issubset(
                explicit.get("extra", {})
            ), (name, role, "extra")

        raw_routes = (raw["default"], *raw["roles"].values())
        for raw_route in raw_routes:
            if "base_url" in raw_route:
                assert raw_route["base_url"] in allowed_base_url_refs
            if "api_key" in raw_route:
                assert raw_route["api_key"] in allowed_api_key_refs

        if entry["transport"] == "official_sdk":
            assert routing.default.extra["transport"] == "official_sdk"
            assert routing.default.extra["credential_source"] == "environment"
        if entry["runtime_binding"] in {"chatgpt_session", "claude_session"}:
            assert routing.default.base_url is None
            assert routing.default.api_key is None
        if entry["backend"] in {"local", "openai_compat"}:
            assert routing.default.extra["expected_served_model"] == entry["model"]
            for role in PRODUCT_ROLES[product]:
                assert (
                    routing.for_role(role).extra["expected_served_model"]
                    == entry["model"]
                )


@pytest.mark.parametrize("product", tuple(PRODUCT_ROLES))
def test_catalogued_runnable_routes_resolve_without_provider_calls(
    product: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("INFER_BASE_URL", "https://inference-api.nvidia.com/v1")
    monkeypatch.setenv("INFER_AUTH_TOKEN", "provider-free-test-token")
    monkeypatch.setenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    monkeypatch.setenv("MOONSHOT_API_KEY", "provider-free-test-token")
    monkeypatch.setenv("CRUCIBLE_LOCAL_URL", "http://127.0.0.1:8000/v1")
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_resolve(name: str, **kwargs: Any) -> object:
        calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    catalog = _catalog(product)
    excluded = {"deepseek_responses_diagnostic"}

    for name, entry in catalog["profiles"].items():
        if name in excluded:
            continue
        path = ROOT / "products" / product / entry["routing"]
        routing = load_routing(config_path=path)
        for role in PRODUCT_ROLES[product]:
            before = len(calls)
            kwargs = {
                "routing": routing,
                "capability": _capability(role),
                "native_shell": False,
                "allow_trusted_project_instructions": False,
                "cwd": str(tmp_path.resolve()),
            }
            if name == "kimi_sdk" and role in {"implementer", "breach_defender"}:
                with pytest.raises(FeatureAdmissionError, match="promotion-blocked"):
                    resolve_for_role(role, **kwargs)
                assert len(calls) == before
                continue

            resolved = resolve_for_role(role, **kwargs)
            assert type(resolved) is object
            assert calls[-1][0] == entry["backend"]
            assert calls[-1][1]["model"] == entry["model"]


@pytest.mark.parametrize("product", tuple(PRODUCT_ROLES))
def test_every_registered_adapter_has_one_all_role_checked_in_route(
    product: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_for_backend = {
        "claude": "claude_fable51",
        "codex": "codex_official_sdk_diagnostic",
        "deepseek": "deepseek_responses_diagnostic",
        "kimi": "kimi_sdk_diagnostic",
        "local": "local_glm53_diagnostic",
        "openai_compat": "openai_glm53_diagnostic",
    }
    assert set(profile_for_backend) == set(BACKEND_NAMES)
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("INFER_BASE_URL", "https://inference-api.nvidia.com/v1")
    monkeypatch.setenv("INFER_AUTH_TOKEN", "provider-free-test-token")
    monkeypatch.setenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    monkeypatch.setenv("MOONSHOT_API_KEY", "provider-free-test-token")
    monkeypatch.setenv("CRUCIBLE_LOCAL_URL", "http://127.0.0.1:8000/v1")
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_resolve(name: str, **kwargs: Any) -> object:
        calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    catalog = _catalog(product)["profiles"]

    for backend in BACKEND_NAMES:
        entry = catalog[profile_for_backend[backend]]
        path = ROOT / "products" / product / entry["routing"]
        if backend == "deepseek":
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            fingerprint = "sha256:" + "b" * 64
            for route in (raw["default"], *raw["roles"].values()):
                route.setdefault("extra", {})["expected_worker_runtime_fingerprint"] = (
                    fingerprint
                )
            routing = lint_routing_config(yaml.safe_dump(raw).encode("utf-8"))
        else:
            routing = load_routing(config_path=path)

        for role in PRODUCT_ROLES[product]:
            before = len(calls)
            resolve_for_role(
                role,
                routing=routing,
                capability=_capability(role),
                native_shell=False,
                allow_trusted_project_instructions=False,
                cwd=str(tmp_path.resolve()),
            )
            assert len(calls) == before + 1
            assert calls[-1][0] == backend
            assert calls[-1][1]["model"] == entry["model"]


@pytest.mark.parametrize("product", tuple(PRODUCT_ROLES))
def test_all_adapter_transports_attach_product_workflow_toolsets(
    product: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the exact pool call shapes used by both product cascades."""

    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("INFER_BASE_URL", "https://inference-api.nvidia.com/v1")
    monkeypatch.setenv("INFER_AUTH_TOKEN", "provider-free-test-token")
    monkeypatch.setenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    monkeypatch.setenv("MOONSHOT_API_KEY", "provider-free-test-token")
    monkeypatch.setenv("CRUCIBLE_LOCAL_URL", "http://127.0.0.1:8000/v1")

    class FakeBackend:
        def __init__(self, name: str, model: str) -> None:
            self.name = "openai_compat" if name == "local" else name
            self.model = model
            self.attachments: list[tuple[Any, int | None]] = []
            self.closed = False

        def invoke(self, _invocation: Any) -> None:
            raise AssertionError("provider transport must not run")

        def attach_toolset(
            self,
            toolset: Any,
            *,
            context_budget: int | None = None,
        ) -> None:
            self.attachments.append((toolset, context_budget))

        def close(self) -> None:
            self.closed = True

    constructed: list[tuple[str, dict[str, Any], FakeBackend]] = []

    def fake_resolve(name: str, **kwargs: Any) -> FakeBackend:
        backend = FakeBackend(name, kwargs["model"])
        constructed.append((name, kwargs, backend))
        return backend

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    product_dir = ROOT / "products" / product
    catalog = _catalog(product)["profiles"]
    project_root = tmp_path.resolve()
    context = ToolContext(
        workspace_root=project_root,
        repo_root=project_root,
        write_scope=("target.py",),
    )
    source_tools = build_source_tools(context)
    implementer_tools = build_autonomous_implementer_profile(context)
    defender_tools = build_breach_defender_profile(context)

    observed_profiles: list[tuple[str, str, str]] = []
    for profile_name, expected_backend, expected_transport in USABLE_WORKFLOW_PROFILES:
        entry = catalog[profile_name]
        assert entry["backend"] == expected_backend
        assert entry["transport"] == expected_transport
        route_path = product_dir / entry["routing"]
        if profile_name == "deepseek_responses_diagnostic":
            raw = deepcopy(yaml.safe_load(route_path.read_text(encoding="utf-8")))
            fingerprint = "sha256:" + "d" * 64
            for route in (raw["default"], *raw["roles"].values()):
                route.setdefault("extra", {})["expected_worker_runtime_fingerprint"] = (
                    fingerprint
                )
            routing = lint_routing_config(yaml.safe_dump(raw).encode("utf-8"))
        else:
            routing = load_routing(config_path=route_path)

        pool = _BackendPool(routing, log_dir=None, cwd=str(project_root))
        acquired: list[FakeBackend] = []
        for role in PRODUCT_ROLES[product]:
            before = len(constructed)
            if role == "architect_planner":
                backend = pool.get(
                    role,
                    toolset=source_tools,
                    context_budget=4096,
                    capability="read_only",
                )
                assert backend.attachments == [(source_tools, 4096)]
            elif role == "implementer":
                backend = pool.get(
                    role,
                    toolset=implementer_tools,
                    native_shell=False,
                )
                assert backend.attachments == [(implementer_tools, None)]
            elif role == "breach_defender":
                backend = pool.get(
                    role,
                    toolset=defender_tools,
                    native_shell=False,
                )
                assert backend.attachments == [(defender_tools, None)]
            else:
                backend = pool.get(role, capability="text")
                assert backend.attachments == []

            assert len(constructed) == before + 1
            adapter_name, kwargs, constructed_backend = constructed[-1]
            assert backend is constructed_backend
            assert adapter_name == expected_backend
            assert backend.name == (
                "openai_compat" if expected_backend == "local" else expected_backend
            )
            assert kwargs["model"] == entry["model"]
            assert kwargs["capability"] == _capability(role)
            assert kwargs["cwd"] == str(project_root)
            if expected_backend == "codex":
                assert (
                    kwargs.get("transport", "legacy_app_server") == expected_transport
                )
            acquired.append(backend)

        pool.close()
        assert all(backend.closed for backend in acquired)
        observed_profiles.append((profile_name, expected_backend, expected_transport))

    assert tuple(observed_profiles) == USABLE_WORKFLOW_PROFILES


@pytest.mark.parametrize("product", tuple(PRODUCT_ROLES))
def test_deepseek_diagnostic_profile_requires_run_local_identity_before_use(
    product: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _catalog(product)["profiles"]["deepseek_responses_diagnostic"]
    path = ROOT / "products" / product / entry["routing"]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    routing = lint_routing_config(path)

    assert "expected_worker_runtime_fingerprint" not in routing.default.extra
    assert "base_url" not in raw["default"]
    assert "api_key" not in raw["default"]
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_resolve(name: str, **kwargs: Any) -> object:
        calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    for role in PRODUCT_ROLES[product]:
        with pytest.raises(
            FeatureAdmissionError,
            match="expected_worker_runtime_fingerprint",
        ):
            resolve_for_role(
                role,
                routing=routing,
                capability=_capability(role),
                native_shell=False,
                allow_trusted_project_instructions=False,
                cwd=str(tmp_path.resolve()),
            )
    assert calls == []

    for role in PRODUCT_ROLES[product]:
        role_extra = raw["roles"][role].get("extra", {})
        if role in {"implementer", "breach_defender"}:
            assert role_extra["diagnostic_workspace_write"] is True
        else:
            assert "diagnostic_workspace_write" not in role_extra

    run_local = deepcopy(raw)
    fingerprint = "sha256:" + "a" * 64
    for route in (run_local["default"], *run_local["roles"].values()):
        route.setdefault("extra", {})["expected_worker_runtime_fingerprint"] = (
            fingerprint
        )
    bound = lint_routing_config(yaml.safe_dump(run_local).encode("utf-8"))
    assert bound.default.extra["expected_worker_runtime_fingerprint"] == fingerprint
    for role in PRODUCT_ROLES[product]:
        assert (
            bound.for_role(role).extra["expected_worker_runtime_fingerprint"]
            == fingerprint
        )
        resolve_for_role(
            role,
            routing=bound,
            capability=_capability(role),
            native_shell=False,
            allow_trusted_project_instructions=False,
            cwd=str(tmp_path.resolve()),
        )

    assert [name for name, _kwargs in calls] == [
        "deepseek" for _role in PRODUCT_ROLES[product]
    ]
    assert all(
        kwargs["model"] == "deepseek-flash"
        and kwargs["expected_worker_runtime_fingerprint"] == fingerprint
        for _name, kwargs in calls
    )
    for role, (_name, kwargs) in zip(PRODUCT_ROLES[product], calls, strict=True):
        if role in {"implementer", "breach_defender"}:
            assert kwargs["diagnostic_workspace_write"] is True
        else:
            assert "diagnostic_workspace_write" not in kwargs


@pytest.mark.parametrize("product", tuple(PRODUCT_ROLES))
def test_checked_in_local_route_resolves_every_product_role(
    product: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _catalog(product)["profiles"]["local_glm53_diagnostic"] == {
        "routing": "backends_local_glm53_diagnostic.yaml",
        "backend": "local",
        "model": "nvidia/zai-org/glm-5.3",
        "transport": "openai_chat_completions",
        "feature_admission": "diagnostic",
        "availability": "diagnostic",
        "runtime_binding": "configured_endpoint",
        "promotion_evidence": False,
    }
    path = ROOT / "products" / product / "backends_local_glm53_diagnostic.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["default"]["base_url"] == "${CRUCIBLE_LOCAL_URL}"
    assert "api_key" not in raw["default"]
    assert raw["default"]["extra"] == {
        "expected_served_model": "nvidia/zai-org/glm-5.3",
        "max_tokens": 16384,
    }
    pricing = DEFAULT_PRICING["nvidia/zai-org/glm-5.3"]
    assert pricing.input_per_mtok == 0.0
    assert pricing.output_per_mtok == 0.0
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("CRUCIBLE_LOCAL_URL", "http://127.0.0.1:8000/v1")
    routing = load_routing(config_path=path)
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_resolve(name: str, **kwargs: Any) -> object:
        calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", fake_resolve)
    for role in PRODUCT_ROLES[product]:
        resolve_for_role(
            role,
            routing=routing,
            capability=_capability(role),
            native_shell=False,
            allow_trusted_project_instructions=False,
            cwd=str(tmp_path.resolve()),
        )

    assert [name for name, _kwargs in calls] == [
        "local" for _role in PRODUCT_ROLES[product]
    ]
    assert all(
        kwargs["model"] == "nvidia/zai-org/glm-5.3"
        and kwargs["base_url"] == "http://127.0.0.1:8000/v1"
        and kwargs["expected_served_model"] == "nvidia/zai-org/glm-5.3"
        and kwargs["max_tokens"] == 16384
        and "api_key" not in kwargs
        for _name, kwargs in calls
    )


GATEWAY_PROFILES = (
    # catalog key, exact route id, predecessor id
    (
        "openai_glm53_diagnostic",
        "nvidia/zai-org/glm-5.3",
        "nvidia/zai-org/glm-5.2",
    ),
    (
        "openai_kimi_k31_pro_diagnostic",
        "nvidia/moonshotai/kimi-k3.1-pro",
        "nvidia/moonshotai/kimi-k2.6",
    ),
    (
        "local_glm53_diagnostic",
        "nvidia/zai-org/glm-5.3",
        "nvidia/zai-org/glm-5.2",
    ),
)


def _chat_completion(model: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        id="request-1",
        model=model,
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="ok",
                tool_calls=None,
                reasoning_content=None,
                provider_specific_fields=None,
            ),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


@pytest.mark.parametrize("product", tuple(PRODUCT_ROLES))
@pytest.mark.parametrize(("profile", "model", "predecessor"), GATEWAY_PROFILES)
def test_checked_in_gateway_profiles_enforce_exact_served_model(
    product: str,
    profile: str,
    model: str,
    predecessor: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    # The built-in price row is what ``invoke`` must find before transport.
    monkeypatch.delenv("CRUCIBLE_PRICING_CONFIG", raising=False)
    monkeypatch.setenv("INFER_BASE_URL", "https://inference-api.nvidia.com/v1")
    monkeypatch.setenv("INFER_AUTH_TOKEN", "provider-free-test-token")
    monkeypatch.setenv("CRUCIBLE_LOCAL_URL", "http://127.0.0.1:8000/v1")
    entry = _catalog(product)["profiles"][profile]
    assert entry["model"] == model
    routing = load_routing(
        config_path=ROOT / "products" / product / entry["routing"],
    )

    def invoke(served_model: str | None) -> InvocationResult:
        backend = resolve_for_role(
            "breach_judge",
            routing=routing,
            capability="text",
            native_shell=False,
            allow_trusted_project_instructions=False,
            cwd=str(tmp_path.resolve()),
        )
        assert type(backend) is openai_compat_module.OpenAIBackend
        backend._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: _chat_completion(served_model),
                ),
            ),
        )
        return backend.invoke(Invocation(
            system="s",
            user="u",
            tools=None,
            max_turns=1,
            budget_usd=None,
        ))

    admitted = invoke(model)
    assert admitted.error is None
    assert admitted.stop_reason == "end_turn"
    assert admitted.final_message == "ok"
    assert admitted.execution_trace is not None
    assert admitted.execution_trace.requested_model == model
    assert admitted.execution_trace.served_model == model
    assert admitted.usage.cost_usd == 0.0

    # A reply that names the predecessor is a different model, not an alias.
    mismatch = invoke(predecessor)
    assert mismatch.stop_reason == "model_mismatch"
    assert mismatch.final_message == ""
    assert mismatch.error is not None
    assert mismatch.error.kind == "incomplete_evidence"
    assert mismatch.execution_trace is not None
    assert mismatch.execution_trace.served_model == predecessor

    missing = invoke(None)
    assert missing.stop_reason == "missing_model_identity"
    assert missing.final_message == ""
    assert missing.error is not None
    assert missing.error.kind == "incomplete_evidence"


def test_gateway_identity_tables_agree_on_current_ids() -> None:
    glm = "nvidia/zai-org/glm-5.3"
    kimi = "nvidia/moonshotai/kimi-k3.1-pro"

    assert review_module._NVIDIA_GLM_MODEL == glm
    for product in PRODUCT_ROLES:
        profiles = _catalog(product)["profiles"]
        assert profiles["openai_glm53_diagnostic"]["model"] == glm
        assert profiles["local_glm53_diagnostic"]["model"] == glm
        assert profiles["openai_kimi_k31_pro_diagnostic"]["model"] == kimi
        assert not (
            ROOT / "products" / product / "backends_local_glm52_diagnostic.yaml"
        ).exists()
    zero_rate = ModelPricing(input_per_mtok=0.0, output_per_mtok=0.0)
    assert DEFAULT_PRICING[glm] == zero_rate
    assert DEFAULT_PRICING[kimi] == zero_rate
    # The predecessors still answer on the gateway, so only the closed tables
    # can show that the move was complete.
    for retired in ("nvidia/zai-org/glm-5.2", "nvidia/moonshotai/kimi-k2.6"):
        assert retired not in DEFAULT_PRICING
