# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Provider-free binding of checked-in product profiles to N-lane matrices."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml  # type: ignore[import-untyped]

from crucible.backends.config import lint_routing_config, routing_fingerprint
from crucible.products.backend_convergence_catalog import (
    REGISTERED_ADAPTER_IDENTITIES,
    REGISTERED_PROFILE_TRANSPORTS,
    BackendLaneCatalog,
    BackendLaneCatalogError,
    generate_backend_lane_pairs,
    generate_reference_arm_pairs,
)
from crucible.products.deepswe.backend_convergence import (
    DEEPSWE_BACKEND_CONVERGENCE_ROLES,
    load_deepswe_backend_profile_catalog,
)
from crucible.products.swebench.backend_convergence import (
    SWEBENCH_BACKEND_CONVERGENCE_ROLES,
    load_swebench_backend_profile_catalog,
)


ROOT = Path(__file__).parents[1]
PROFILE_KEYS = (
    "codex_astra",
    "codex_official_sdk_diagnostic",
    "claude_fable51",
    "claude_mythos51",
    "deepseek_responses_diagnostic",
    "kimi_sdk",
    "kimi_sdk_diagnostic",
    "kimi_k31_pro_sdk_diagnostic",
    "openai_glm53_diagnostic",
    "openai_kimi_k31_pro_diagnostic",
    "local_glm53_diagnostic",
)
PRODUCTS: tuple[
    tuple[str, tuple[str, ...], Callable[..., BackendLaneCatalog]], ...
] = (
    (
        "swebench",
        SWEBENCH_BACKEND_CONVERGENCE_ROLES,
        load_swebench_backend_profile_catalog,
    ),
    (
        "deepswe",
        DEEPSWE_BACKEND_CONVERGENCE_ROLES,
        load_deepswe_backend_profile_catalog,
    ),
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _copied_product(tmp_path: Path, product: str) -> Path:
    source = ROOT / "products" / product
    target = tmp_path / product
    shutil.copytree(source, target)
    return target / "backend_profile_catalog.yaml"


def _rewrite_yaml(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert type(document) is dict
    mutate(document)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize(("product", "roles", "loader"), PRODUCTS)
def test_checked_in_profiles_bind_to_complete_ordered_n_lane_catalogs(
    product: str,
    roles: tuple[str, ...],
    loader: Callable[..., BackendLaneCatalog],
) -> None:
    catalog_path = ROOT / "products" / product / "backend_profile_catalog.yaml"
    catalog_bytes = catalog_path.read_bytes()
    profile_document = yaml.safe_load(catalog_bytes)

    catalog = loader(expected_catalog_digest=_digest(catalog_bytes))

    assert catalog.contract.product == product
    assert catalog.contract.roles == roles
    assert catalog.profile_catalog_path == catalog_path.resolve()
    assert catalog.profile_catalog_digest == _digest(catalog_bytes)
    assert tuple(lane.key for lane in catalog.lanes) == PROFILE_KEYS
    assert tuple(lane.lane_id for lane in catalog.lanes) == PROFILE_KEYS
    assert catalog.adapter_identities == REGISTERED_ADAPTER_IDENTITIES
    assert {
        (entry["backend"], entry["transport"])
        for entry in profile_document["profiles"].values()
    } == REGISTERED_PROFILE_TRANSPORTS
    assert all(lane.evidence_status == "diagnostic" for lane in catalog.lanes)
    assert all(not lane.promotion_eligible for lane in catalog.lanes)
    assert all(lane.promotion_authority_digest is None for lane in catalog.lanes)

    for lane in catalog.lanes:
        route_bytes = lane.routing_path.read_bytes()
        assert lane.routing_digest == _digest(route_bytes)
        assert lane.routing_fingerprint == routing_fingerprint(
            lint_routing_config(route_bytes)
        )
        assert tuple(route.role for route in lane.role_routes) == roles
        assert {
            (route.backend, route.runtime_backend, route.transport)
            for route in (lane.default_route, *lane.role_routes)
        } == {(lane.backend, lane.runtime_backend, lane.transport)}

    # The checked Kimi recipes also route two shared non-product planner roles.
    # They are authenticated by the file fingerprint but cannot leak into the
    # product's closed comparison identity.
    assert tuple(route.role for route in catalog.lane("kimi_sdk").role_routes) == roles

    local = catalog.lane("local_glm53_diagnostic")
    hosted = catalog.lane("openai_glm53_diagnostic")
    assert local.backend == "local"
    assert local.runtime_backend == hosted.runtime_backend == "openai_compat"
    assert local.transport == hosted.transport == "openai_compatible_http"
    assert local.routing_digest != hosted.routing_digest
    assert local.routing_fingerprint != hosted.routing_fingerprint

    pairs = generate_backend_lane_pairs(catalog)
    assert len(pairs) == 55
    assert all(not pair.promotion_eligible for pair in pairs)
    reference_pairs = generate_reference_arm_pairs(
        catalog,
        reference_arm="codex_astra",
    )
    assert len(reference_pairs) == 10
    assert all(pair.arm_a.key == "codex_astra" for pair in reference_pairs)
    assert tuple(pair.arm_b.key for pair in reference_pairs) == PROFILE_KEYS[1:]
    assert all(not pair.promotion_eligible for pair in reference_pairs)

    identity = catalog.identity()
    assert identity["profile_catalog"] == {
        "path": str(catalog_path.resolve()),
        "digest": _digest(catalog_bytes),
    }


@pytest.mark.parametrize(("product", "_roles", "loader"), PRODUCTS)
def test_profile_catalog_exact_byte_digest_rejects_drift(
    tmp_path: Path,
    product: str,
    _roles: tuple[str, ...],
    loader: Callable[..., BackendLaneCatalog],
) -> None:
    catalog_path = _copied_product(tmp_path, product)
    expected = _digest(catalog_path.read_bytes())
    catalog_path.write_bytes(catalog_path.read_bytes() + b"\n# byte drift\n")

    with pytest.raises(BackendLaneCatalogError, match="catalog bytes"):
        loader(catalog_path=catalog_path, expected_catalog_digest=expected)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("catalog_role", "ordered product contract"),
        ("missing_route_role", "does not explicitly route every swebench role"),
        ("model", "does not match catalog model"),
        ("transport_alias", "no exact registered binding"),
        ("wrong_backend_transport", "no exact registered binding"),
        ("promotion", "must be false"),
        ("missing_transport_identity", "does not cover registered"),
        ("local_served_model", "must pin expected_served_model"),
    ),
)
def test_checked_in_profile_binding_fails_closed_on_catalog_or_route_drift(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    catalog_path = _copied_product(tmp_path, "swebench")

    if mutation == "catalog_role":
        _rewrite_yaml(
            catalog_path,
            lambda document: document["roles"].pop(),  # type: ignore[index,union-attr]
        )
    elif mutation == "missing_route_role":
        route_path = catalog_path.parent / "backends_codex_astra.yaml"
        _rewrite_yaml(
            route_path,
            lambda document: document["roles"].pop("architect_judge"),  # type: ignore[index,union-attr]
        )
    elif mutation == "model":
        _rewrite_yaml(
            catalog_path,
            lambda document: document["profiles"]["codex_astra"].__setitem__(  # type: ignore[index,union-attr]
                "model", "gpt-drift"
            ),
        )
    elif mutation == "transport_alias":
        _rewrite_yaml(
            catalog_path,
            lambda document: document["profiles"][  # type: ignore[index]
                "deepseek_responses_diagnostic"
            ].__setitem__("transport", "deepseek_responses_worker"),  # type: ignore[union-attr]
        )
    elif mutation == "wrong_backend_transport":
        _rewrite_yaml(
            catalog_path,
            lambda document: document["profiles"]["claude_fable51"].__setitem__(  # type: ignore[index,union-attr]
                "transport", "official_sdk"
            ),
        )
    elif mutation == "promotion":
        _rewrite_yaml(
            catalog_path,
            lambda document: document["profiles"]["claude_fable51"].__setitem__(  # type: ignore[index,union-attr]
                "promotion_evidence", True
            ),
        )
    elif mutation == "missing_transport_identity":
        _rewrite_yaml(
            catalog_path,
            lambda document: document["profiles"].pop(  # type: ignore[index,union-attr]
                "codex_official_sdk_diagnostic"
            ),
        )
    else:
        route_path = catalog_path.parent / "backends_local_glm53_diagnostic.yaml"

        def drift_local(document: dict[str, Any]) -> None:
            document["default"]["extra"]["expected_served_model"] = "wrong"  # type: ignore[index]

        _rewrite_yaml(route_path, drift_local)

    with pytest.raises(BackendLaneCatalogError, match=match):
        load_swebench_backend_profile_catalog(catalog_path=catalog_path)


def test_profile_transport_vocabulary_is_closed_and_backend_qualified() -> None:
    assert REGISTERED_PROFILE_TRANSPORTS == {
        ("claude", "claude_agent_sdk"),
        ("codex", "legacy_app_server"),
        ("codex", "official_sdk"),
        ("deepseek", "native_responses"),
        ("kimi", "kimi_agent_sdk"),
        ("local", "openai_chat_completions"),
        ("openai_compat", "openai_chat_completions"),
    }


def test_profile_catalog_must_be_one_exact_regular_file(tmp_path: Path) -> None:
    source = ROOT / "products" / "swebench" / "backend_profile_catalog.yaml"
    link = tmp_path / "backend_profile_catalog.yaml"
    link.symlink_to(source)

    with pytest.raises(BackendLaneCatalogError, match="not a regular file"):
        load_swebench_backend_profile_catalog(catalog_path=link)
