# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Provider-free DeepSWE adapter-catalog validation."""

from __future__ import annotations

from pathlib import Path

from crucible.products.backend_convergence_catalog import (
    BackendCatalogContract,
    BackendLaneCatalog,
    BackendLaneDefinition,
    PromotionAuthorityValidator,
    load_backend_profile_catalog,
    validate_backend_lane_catalog,
)

DEEPSWE_BACKEND_CONVERGENCE_ROLES = (
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

DEEPSWE_BACKEND_CATALOG_CONTRACT = BackendCatalogContract(
    product="deepswe",
    roles=DEEPSWE_BACKEND_CONVERGENCE_ROLES,
)


def validate_deepswe_backend_catalog(
    *,
    route_root: Path | str,
    lanes: tuple[BackendLaneDefinition, ...],
    require_full_registry: bool = False,
    promotion_authority_validator: PromotionAuthorityValidator | None = None,
) -> BackendLaneCatalog:
    """Validate lanes against DeepSWE's complete ordered role set."""

    return validate_backend_lane_catalog(
        DEEPSWE_BACKEND_CATALOG_CONTRACT,
        route_root=route_root,
        lanes=lanes,
        require_full_registry=require_full_registry,
        promotion_authority_validator=promotion_authority_validator,
    )


def load_deepswe_backend_profile_catalog(
    *,
    catalog_path: Path | str | None = None,
    expected_catalog_digest: str | None = None,
    require_full_registry: bool = True,
) -> BackendLaneCatalog:
    """Bind every checked-in DeepSWE profile as a diagnostic N-lane arm."""

    source = (
        Path(__file__).resolve().with_name("backend_profile_catalog.yaml")
        if catalog_path is None
        else catalog_path
    )
    return load_backend_profile_catalog(
        DEEPSWE_BACKEND_CATALOG_CONTRACT,
        catalog_path=source,
        expected_catalog_digest=expected_catalog_digest,
        require_full_registry=require_full_registry,
    )


__all__ = [
    "DEEPSWE_BACKEND_CATALOG_CONTRACT",
    "DEEPSWE_BACKEND_CONVERGENCE_ROLES",
    "load_deepswe_backend_profile_catalog",
    "validate_deepswe_backend_catalog",
]
