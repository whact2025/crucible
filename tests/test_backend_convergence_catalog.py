# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest
import yaml  # type: ignore[import-untyped]
from crucible.backends.config import lint_routing_config, routing_fingerprint
from crucible.products.backend_convergence_catalog import (
    REGISTERED_ADAPTER_IDENTITIES,
    REGISTERED_ADAPTER_TRANSPORTS,
    BackendLaneCatalog,
    BackendLaneCatalogError,
    BackendLaneDefinition,
    generate_backend_lane_pairs,
    generate_reference_arm_pairs,
)
from crucible.products.deepswe.backend_convergence import (
    DEEPSWE_BACKEND_CONVERGENCE_ROLES,
    validate_deepswe_backend_catalog,
)
from crucible.products.swebench.backend_convergence import (
    SWEBENCH_BACKEND_CONVERGENCE_ROLES,
    validate_swebench_backend_catalog,
)

_ADAPTER_LANES = (
    ("claude", "claude", "claude", "claude_agent_sdk"),
    ("codex-legacy", "codex", "codex", "legacy_app_server"),
    ("codex-official", "codex", "codex", "official_sdk"),
    ("deepseek", "deepseek", "deepseek", "deepseek_responses_worker"),
    ("kimi", "kimi", "kimi", "kimi_agent_sdk_worker"),
    ("local", "local", "openai_compat", "openai_compatible_http"),
    (
        "openai-compat",
        "openai_compat",
        "openai_compat",
        "openai_compatible_http",
    ),
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _route_document(
    backend: str,
    transport: str,
    roles: tuple[str, ...],
    *,
    feature_admission: str = "diagnostic",
) -> dict[str, object]:
    model = {
        "claude": "claude-fable-5-1",
        "codex": "gpt-6-astra",
        "deepseek": "deepseek-v4-flash",
        "kimi": "kimi-k2.7-code-highspeed",
        "local": "local-test-model",
        "openai_compat": "openai-compatible-test-model",
    }[backend]
    default: dict[str, object] = {
        "backend": backend,
        "model": model,
        "feature_admission": feature_admission,
    }
    if backend in {"claude", "codex", "openai_compat"}:
        default["effort"] = "high" if backend != "codex" else "max"
    if backend == "codex":
        default["extra"] = {"transport": transport}
    elif backend == "kimi":
        default.update(
            {
                "base_url": "${KIMI_BASE_URL}",
                "api_key": "${MOONSHOT_API_KEY}",
                "extra": {"worker_source": "project_locked"},
            }
        )
    elif backend == "local":
        default["base_url"] = "http://127.0.0.1:8000/v1"
    elif backend == "openai_compat":
        default.update(
            {
                "base_url": "${INFER_BASE_URL}",
                "api_key": "${INFER_AUTH_TOKEN}",
            }
        )
    role_routes: dict[str, dict[str, object]] = {role: {} for role in roles}
    if backend == "kimi" and feature_admission == "diagnostic":
        role_routes["implementer"] = {"extra": {"diagnostic_workspace_write": True}}
    return {
        "schema_version": "crucible.backend-routing.v1",
        "default": default,
        "roles": role_routes,
    }


def _catalog_definitions(
    root: Path,
    roles: tuple[str, ...],
    *,
    feature_admission: str = "diagnostic",
) -> tuple[BackendLaneDefinition, ...]:
    definitions: list[BackendLaneDefinition] = []
    for key, backend, runtime_backend, transport in _ADAPTER_LANES:
        route_bytes = yaml.safe_dump(
            _route_document(
                backend,
                transport,
                roles,
                feature_admission=feature_admission,
            ),
            sort_keys=False,
        ).encode("utf-8")
        route_path = root / f"{key}.yaml"
        route_path.write_bytes(route_bytes)
        definitions.append(
            BackendLaneDefinition(
                key=key,
                lane_id=f"{key}-diagnostic",
                backend=backend,
                runtime_backend=runtime_backend,
                transport=transport,
                routing_path=route_path.name,
                routing_digest=_digest(route_bytes),
                routing_fingerprint=routing_fingerprint(
                    lint_routing_config(route_bytes)
                ),
                evidence_status="diagnostic",
            )
        )
    return tuple(definitions)


def test_product_catalog_role_contracts_are_exact() -> None:
    assert SWEBENCH_BACKEND_CONVERGENCE_ROLES == (
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
    )
    assert DEEPSWE_BACKEND_CONVERGENCE_ROLES == (
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
    )


@pytest.mark.parametrize(
    ("product", "roles", "validator"),
    (
        (
            "swebench",
            SWEBENCH_BACKEND_CONVERGENCE_ROLES,
            validate_swebench_backend_catalog,
        ),
        (
            "deepswe",
            DEEPSWE_BACKEND_CONVERGENCE_ROLES,
            validate_deepswe_backend_catalog,
        ),
    ),
)
def test_full_registered_adapter_catalog_is_valid_for_each_product(
    tmp_path: Path,
    product: str,
    roles: tuple[str, ...],
    validator: Callable[..., BackendLaneCatalog],
) -> None:
    definitions = _catalog_definitions(tmp_path, roles)

    catalog = validator(
        route_root=tmp_path,
        lanes=definitions,
        require_full_registry=True,
    )

    assert catalog.contract.product == product
    assert catalog.contract.roles == roles
    assert len(catalog.contract.roles) == (10 if product == "swebench" else 11)
    assert catalog.adapter_transports == REGISTERED_ADAPTER_TRANSPORTS
    assert catalog.adapter_identities == REGISTERED_ADAPTER_IDENTITIES
    assert len(catalog.lanes) == len(_ADAPTER_LANES) == 7
    assert all(not lane.promotion_eligible for lane in catalog.lanes)
    assert all(lane.evidence_status == "diagnostic" for lane in catalog.lanes)
    assert all(lane.promotion_authority_digest is None for lane in catalog.lanes)
    for lane in catalog.lanes:
        assert tuple(route.role for route in lane.role_routes) == roles
        assert {
            (route.backend, route.runtime_backend, route.transport)
            for route in (lane.default_route, *lane.role_routes)
        } == {(lane.backend, lane.runtime_backend, lane.transport)}
    local = catalog.lane("local")
    compat = catalog.lane("openai-compat")
    assert local.backend == "local"
    assert local.runtime_backend == compat.runtime_backend == "openai_compat"
    assert local.transport == compat.transport == "openai_compatible_http"
    assert local.routing_fingerprint != compat.routing_fingerprint
    assert local.default_route.model != compat.default_route.model
    identity = catalog.identity()
    encoded = json.dumps(identity, sort_keys=True)
    assert identity["product"] == product
    assert "MOONSHOT_API_KEY" not in encoded
    assert "INFER_AUTH_TOKEN" not in encoded


def test_catalog_generates_all_pairs_and_reference_arm_pairs(tmp_path: Path) -> None:
    catalog = validate_swebench_backend_catalog(
        route_root=tmp_path,
        lanes=_catalog_definitions(
            tmp_path,
            SWEBENCH_BACKEND_CONVERGENCE_ROLES,
        ),
        require_full_registry=True,
    )

    pairs = generate_backend_lane_pairs(catalog)
    assert len(pairs) == 21
    assert [(pair.arm_a.key, pair.arm_b.key) for pair in pairs[:3]] == [
        ("claude", "codex-legacy"),
        ("claude", "codex-official"),
        ("claude", "deepseek"),
    ]
    assert all(pair.reference_arm is None for pair in pairs)
    assert all(pair.evidence_status == "diagnostic" for pair in pairs)
    assert all(not pair.promotion_eligible for pair in pairs)

    reference_pairs = generate_reference_arm_pairs(
        catalog,
        reference_arm="codex-legacy",
    )
    assert len(reference_pairs) == 6
    assert all(pair.arm_a.key == "codex-legacy" for pair in reference_pairs)
    assert [pair.arm_b.key for pair in reference_pairs] == [
        "claude",
        "codex-official",
        "deepseek",
        "kimi",
        "local",
        "openai-compat",
    ]
    assert all(pair.reference_arm == "codex-legacy" for pair in reference_pairs)
    with pytest.raises(BackendLaneCatalogError, match="unknown catalog lane"):
        generate_reference_arm_pairs(catalog, reference_arm="missing")


def test_promotable_status_requires_authority_and_enforcing_routes(
    tmp_path: Path,
) -> None:
    definitions = _catalog_definitions(
        tmp_path,
        SWEBENCH_BACKEND_CONVERGENCE_ROLES,
        feature_admission="enforce",
    )
    authority = "sha256:" + "a" * 64
    first = dataclasses.replace(
        definitions[0],
        evidence_status="promotable",
        promotion_authority_digest=authority,
    )
    second = dataclasses.replace(
        definitions[1],
        evidence_status="promotable",
        promotion_authority_digest=authority,
    )
    validated_promotions: list[str] = []
    catalog = validate_swebench_backend_catalog(
        route_root=tmp_path,
        lanes=(first, second, *definitions[2:]),
        promotion_authority_validator=lambda lane: validated_promotions.append(
            lane.key
        ),
    )

    assert validated_promotions == [first.key, second.key]
    assert catalog.lanes[0].promotion_eligible
    assert generate_backend_lane_pairs(catalog)[0].promotion_eligible
    assert not generate_backend_lane_pairs(catalog)[1].promotion_eligible

    with pytest.raises(BackendLaneCatalogError, match="tagged SHA-256"):
        validate_swebench_backend_catalog(
            route_root=tmp_path,
            lanes=(
                dataclasses.replace(
                    definitions[0],
                    evidence_status="promotable",
                ),
                definitions[1],
            ),
        )
    with pytest.raises(BackendLaneCatalogError, match="product-owned"):
        validate_swebench_backend_catalog(
            route_root=tmp_path,
            lanes=(first, definitions[1]),
        )
    with pytest.raises(BackendLaneCatalogError, match="cannot claim"):
        validate_swebench_backend_catalog(
            route_root=tmp_path,
            lanes=(
                dataclasses.replace(
                    definitions[0],
                    promotion_authority_digest=authority,
                ),
                definitions[1],
            ),
        )


def test_promotable_status_rejects_diagnostic_feature_admission(
    tmp_path: Path,
) -> None:
    definitions = _catalog_definitions(
        tmp_path,
        DEEPSWE_BACKEND_CONVERGENCE_ROLES,
    )
    promoted = dataclasses.replace(
        definitions[0],
        evidence_status="promotable",
        promotion_authority_digest="sha256:" + "b" * 64,
    )

    with pytest.raises(BackendLaneCatalogError, match="must enforce"):
        validate_deepswe_backend_catalog(
            route_root=tmp_path,
            lanes=(promoted, definitions[1]),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("backend_transport", "runtime-backend/transport"),
        ("runtime_backend", "runtime-backend/transport"),
        ("route_transport", "runtime-backend/transport"),
        ("route_digest", "routing bytes"),
        ("route_fingerprint", "routing fingerprint drifted"),
        ("duplicate_key", "keys must be unique"),
        ("duplicate_id", "ids must be unique"),
        ("missing_registry_lane", "does not cover registered"),
        ("unexpected_role", "outside the swebench contract"),
        ("route_escape", "escapes the route root"),
    ),
)
def test_catalog_validation_fails_closed(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    definitions = list(
        _catalog_definitions(tmp_path, SWEBENCH_BACKEND_CONVERGENCE_ROLES)
    )
    if mutation == "backend_transport":
        definitions[0] = dataclasses.replace(
            definitions[0],
            backend="codex",
            transport="legacy_app_server",
        )
    elif mutation == "runtime_backend":
        definitions[5] = dataclasses.replace(
            definitions[5],
            runtime_backend="local",
        )
    elif mutation == "route_transport":
        definitions[1] = dataclasses.replace(
            definitions[1],
            transport="official_sdk",
        )
    elif mutation == "route_digest":
        route_path = tmp_path / str(definitions[0].routing_path)
        route_path.write_bytes(route_path.read_bytes() + b"\n# drift\n")
    elif mutation == "route_fingerprint":
        route_path = tmp_path / str(definitions[0].routing_path)
        route = yaml.safe_load(route_path.read_text(encoding="utf-8"))
        route["default"]["model"] = "claude-opus-5"
        route_bytes = yaml.safe_dump(route, sort_keys=False).encode("utf-8")
        route_path.write_bytes(route_bytes)
        definitions[0] = dataclasses.replace(
            definitions[0],
            routing_digest=_digest(route_bytes),
        )
    elif mutation == "duplicate_key":
        definitions[1] = dataclasses.replace(
            definitions[1],
            key=definitions[0].key,
        )
    elif mutation == "duplicate_id":
        definitions[1] = dataclasses.replace(
            definitions[1],
            lane_id=definitions[0].lane_id,
        )
    elif mutation == "missing_registry_lane":
        definitions.pop()
    elif mutation == "unexpected_role":
        route_path = tmp_path / str(definitions[0].routing_path)
        route = yaml.safe_load(route_path.read_text(encoding="utf-8"))
        route["roles"]["goal_judge"] = {}
        route_bytes = yaml.safe_dump(route, sort_keys=False).encode("utf-8")
        route_path.write_bytes(route_bytes)
        definitions[0] = dataclasses.replace(
            definitions[0],
            routing_digest=_digest(route_bytes),
            routing_fingerprint=routing_fingerprint(lint_routing_config(route_bytes)),
        )
    else:
        definitions[0] = dataclasses.replace(
            definitions[0],
            routing_path=Path("..") / "escaped.yaml",
        )

    with pytest.raises(BackendLaneCatalogError, match=match):
        validate_swebench_backend_catalog(
            route_root=tmp_path,
            lanes=tuple(definitions),
            require_full_registry=True,
        )
