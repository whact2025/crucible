# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Provider-free validation for product backend-lane catalogs.

This module authenticates routing recipes and projects deterministic lane and
pair identities.  It does not construct a backend, resolve credential values,
dispatch a provider request, or qualify an adapter.  Product modules supply
their own exact role contract; this layer only applies that closed contract to
every lane.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import yaml  # type: ignore[import-untyped]

from crucible.backends import BACKEND_NAMES
from crucible.backends.config import (
    RoleConfig,
    lint_routing_config,
    routing_fingerprint,
)
from crucible.common.authority_io import AuthorityIOError, read_exact_regular

EvidenceStatus = Literal["diagnostic", "promotable"]
PromotionAuthorityValidator = Callable[["BackendLane"], None]

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
_MAX_ROUTE_BYTES = 4 * 1024 * 1024
PROFILE_CATALOG_SCHEMA = "crucible.backend-profile-catalog.v1"
_BACKEND_IMPLEMENTATIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "claude": ("claude", ("claude_agent_sdk",)),
    "codex": ("codex", ("legacy_app_server", "official_sdk")),
    "deepseek": ("deepseek", ("deepseek_responses_worker",)),
    "kimi": ("kimi", ("kimi_agent_sdk_worker",)),
    # ``local`` is a routing alias for OpenAIBackend, whose result and trace
    # backend identity is ``openai_compat``.  The authenticated route digest,
    # fingerprint, and model retain the distinct local profile identity.
    "local": ("openai_compat", ("openai_compatible_http",)),
    "openai_compat": ("openai_compat", ("openai_compatible_http",)),
}

# Product profile catalogs use transport names intended for operators while
# the N-lane identity records the concrete adapter implementation boundary.
# Keep this translation closed and backend-qualified: accepting a transport
# name under the wrong backend (or inventing a near-match alias) is drift.
_PROFILE_TRANSPORT_BINDINGS: dict[
    tuple[str, str], tuple[str, str]
] = {
    ("claude", "claude_agent_sdk"): ("claude", "claude_agent_sdk"),
    ("codex", "legacy_app_server"): ("codex", "legacy_app_server"),
    ("codex", "official_sdk"): ("codex", "official_sdk"),
    ("deepseek", "native_responses"): (
        "deepseek",
        "deepseek_responses_worker",
    ),
    ("kimi", "kimi_agent_sdk"): ("kimi", "kimi_agent_sdk_worker"),
    ("local", "openai_chat_completions"): (
        "openai_compat",
        "openai_compatible_http",
    ),
    ("openai_compat", "openai_chat_completions"): (
        "openai_compat",
        "openai_compatible_http",
    ),
}

REGISTERED_PROFILE_TRANSPORTS = frozenset(_PROFILE_TRANSPORT_BINDINGS)
_PROFILE_FIELDS = frozenset(
    {
        "routing",
        "backend",
        "model",
        "transport",
        "feature_admission",
        "availability",
        "runtime_binding",
        "promotion_evidence",
    }
)
_PROFILE_AVAILABILITY = frozenset(
    {"blocked", "construction_only", "diagnostic", "runtime_required"}
)
_PROFILE_RUNTIME_BINDINGS = frozenset(
    {
        "chatgpt_session",
        "claude_session",
        "configured_endpoint",
        "project_locked",
        "run_local_required",
    }
)

REGISTERED_ADAPTER_TRANSPORTS = frozenset(
    (backend, transport)
    for backend, (_runtime_backend, transports) in _BACKEND_IMPLEMENTATIONS.items()
    for transport in transports
)
REGISTERED_ADAPTER_IDENTITIES = frozenset(
    (backend, runtime_backend, transport)
    for backend, (runtime_backend, transports) in _BACKEND_IMPLEMENTATIONS.items()
    for transport in transports
)


class BackendLaneCatalogError(ValueError):
    """Raised when a catalog or one of its measured routes drifts."""


@dataclasses.dataclass(frozen=True)
class BackendCatalogContract:
    """One product's ordered, complete workflow-role contract."""

    product: str
    roles: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BackendLaneDefinition:
    """Caller-declared identity for one measured adapter route."""

    key: str
    lane_id: str
    backend: str
    runtime_backend: str
    transport: str
    routing_path: Path | str
    routing_digest: str
    routing_fingerprint: str
    evidence_status: EvidenceStatus = "diagnostic"
    promotion_authority_digest: str | None = None


@dataclasses.dataclass(frozen=True)
class EffectiveRouteIdentity:
    """Secret-free effective adapter identity for one workflow role."""

    role: str
    backend: str
    runtime_backend: str
    transport: str
    model: str
    effort: str | None
    feature_admission: str

    def identity(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class BackendLane:
    """Validated, provider-free lane with an exact effective role closure."""

    key: str
    lane_id: str
    backend: str
    runtime_backend: str
    transport: str
    routing_path: Path
    routing_digest: str
    routing_fingerprint: str
    evidence_status: EvidenceStatus
    promotion_authority_digest: str | None
    default_route: EffectiveRouteIdentity
    role_routes: tuple[EffectiveRouteIdentity, ...]

    @property
    def promotion_eligible(self) -> bool:
        return self.evidence_status == "promotable"

    def identity(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "id": self.lane_id,
            "backend": self.backend,
            "runtime_backend": self.runtime_backend,
            "transport": self.transport,
            "routing_path": str(self.routing_path),
            "routing_digest": self.routing_digest,
            "routing_fingerprint": self.routing_fingerprint,
            "evidence_status": self.evidence_status,
            "promotion_eligible": self.promotion_eligible,
            "promotion_authority_digest": self.promotion_authority_digest,
            "default_route": self.default_route.identity(),
            "role_routes": [route.identity() for route in self.role_routes],
        }


@dataclasses.dataclass(frozen=True)
class BackendLaneCatalog:
    """Validated ordered catalog for one product."""

    contract: BackendCatalogContract
    lanes: tuple[BackendLane, ...]
    profile_catalog_path: Path | None = None
    profile_catalog_digest: str | None = None

    @property
    def adapter_transports(self) -> frozenset[tuple[str, str]]:
        return frozenset((lane.backend, lane.transport) for lane in self.lanes)

    @property
    def adapter_identities(self) -> frozenset[tuple[str, str, str]]:
        return frozenset(
            (lane.backend, lane.runtime_backend, lane.transport) for lane in self.lanes
        )

    def lane(self, key: str) -> BackendLane:
        matches = tuple(lane for lane in self.lanes if lane.key == key)
        if len(matches) != 1:
            raise BackendLaneCatalogError(f"unknown catalog lane {key!r}")
        return matches[0]

    def identity(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "crucible.backend-lane-catalog.v1",
            "product": self.contract.product,
            "roles": list(self.contract.roles),
            "lanes": [lane.identity() for lane in self.lanes],
        }
        if self.profile_catalog_path is not None:
            result["profile_catalog"] = {
                "path": str(self.profile_catalog_path),
                "digest": self.profile_catalog_digest,
            }
        return result


@dataclasses.dataclass(frozen=True)
class BackendLanePair:
    """One deterministic two-lane comparison projection."""

    arm_a: BackendLane
    arm_b: BackendLane
    reference_arm: str | None
    evidence_status: EvidenceStatus

    @property
    def promotion_eligible(self) -> bool:
        return self.evidence_status == "promotable"

    def identity(self) -> dict[str, Any]:
        return {
            "arm_a": self.arm_a.identity(),
            "arm_b": self.arm_b.identity(),
            "reference_arm": self.reference_arm,
            "evidence_status": self.evidence_status,
            "promotion_eligible": self.promotion_eligible,
        }


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _identifier(value: Any, *, field: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise BackendLaneCatalogError(f"{field} must be an exact identifier")
    return value


def _tagged_digest(value: Any, *, field: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise BackendLaneCatalogError(f"{field} must be tagged SHA-256")
    return value


def _validate_contract(contract: BackendCatalogContract) -> None:
    if type(contract) is not BackendCatalogContract:
        raise BackendLaneCatalogError("contract must be BackendCatalogContract")
    _identifier(contract.product, field="contract.product")
    if (
        type(contract.roles) is not tuple
        or not contract.roles
        or any(
            type(role) is not str or _IDENTIFIER_RE.fullmatch(role) is None
            for role in contract.roles
        )
        or len(contract.roles) != len(set(contract.roles))
    ):
        raise BackendLaneCatalogError(
            "contract.roles must be an ordered tuple of unique identifiers"
        )


def _validate_transport_registry() -> None:
    registered = set(BACKEND_NAMES)
    described = set(_BACKEND_IMPLEMENTATIONS)
    if registered != described:
        raise BackendLaneCatalogError(
            "backend transport registry drifted; "
            f"missing={sorted(registered - described)}, "
            f"extra={sorted(described - registered)}"
        )
    profile_bindings = {
        (backend, runtime_backend, runtime_transport)
        for (backend, _profile_transport), (
            runtime_backend,
            runtime_transport,
        ) in _PROFILE_TRANSPORT_BINDINGS.items()
    }
    if profile_bindings != REGISTERED_ADAPTER_IDENTITIES:
        raise BackendLaneCatalogError(
            "profile transport bindings drifted from adapter implementations; "
            f"missing={sorted(REGISTERED_ADAPTER_IDENTITIES - profile_bindings)}, "
            f"extra={sorted(profile_bindings - REGISTERED_ADAPTER_IDENTITIES)}"
        )


def _runtime_identity_for_config(config: RoleConfig) -> tuple[str, str]:
    backend = config.backend
    if type(backend) is not str or backend not in _BACKEND_IMPLEMENTATIONS:
        raise BackendLaneCatalogError(
            f"effective route has unsupported backend {backend!r}"
        )
    runtime_backend, transports = _BACKEND_IMPLEMENTATIONS[backend]
    if backend == "codex":
        transport = config.extra.get("transport", "legacy_app_server")
        if type(transport) is not str:
            raise BackendLaneCatalogError(
                "effective Codex transport must be exact text"
            )
        return runtime_backend, transport
    return runtime_backend, transports[0]


def _effective_route(role: str, config: RoleConfig) -> EffectiveRouteIdentity:
    backend = config.backend
    model = config.model
    feature_admission = config.feature_admission
    if type(backend) is not str or type(model) is not str or not model:
        raise BackendLaneCatalogError(
            f"effective route {role!r} lacks explicit backend/model identity"
        )
    if (
        type(feature_admission) is not str
        or feature_admission not in {"diagnostic", "enforce"}
    ):
        raise BackendLaneCatalogError(
            f"effective route {role!r} lacks feature-admission identity"
        )
    runtime_backend, transport = _runtime_identity_for_config(config)
    return EffectiveRouteIdentity(
        role=role,
        backend=backend,
        runtime_backend=runtime_backend,
        transport=transport,
        model=model,
        effort=config.effort,
        feature_admission=feature_admission,
    )


def _confined_route(route_root: Path, value: Path | str, *, field: str) -> Path:
    if not isinstance(value, (Path, str)):
        raise BackendLaneCatalogError(f"{field} must be a path")
    raw = Path(value)
    root = Path(os.path.abspath(route_root))
    candidate = Path(os.path.abspath(raw if raw.is_absolute() else root / raw))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BackendLaneCatalogError(f"{field} escapes the route root") from exc
    return candidate


def _validate_lane(
    definition: BackendLaneDefinition,
    *,
    contract: BackendCatalogContract,
    route_root: Path,
    promotion_authority_validator: PromotionAuthorityValidator | None,
    allow_additional_roles: bool,
) -> BackendLane:
    if type(definition) is not BackendLaneDefinition:
        raise BackendLaneCatalogError(
            "catalog lanes must be BackendLaneDefinition values"
        )
    key = _identifier(definition.key, field="lane.key")
    lane_id = _identifier(definition.lane_id, field=f"lanes.{key}.id")
    backend = _identifier(definition.backend, field=f"lanes.{key}.backend")
    runtime_backend = _identifier(
        definition.runtime_backend,
        field=f"lanes.{key}.runtime_backend",
    )
    transport = _identifier(
        definition.transport,
        field=f"lanes.{key}.transport",
    )
    if (
        backend,
        runtime_backend,
        transport,
    ) not in REGISTERED_ADAPTER_IDENTITIES:
        raise BackendLaneCatalogError(
            f"lane {key!r} has unsupported routing-backend/runtime-backend/"
            f"transport identity {backend!r}/{runtime_backend!r}/{transport!r}"
        )
    if definition.evidence_status not in {"diagnostic", "promotable"}:
        raise BackendLaneCatalogError(
            f"lanes.{key}.evidence_status must be diagnostic or promotable"
        )
    promotion_digest = definition.promotion_authority_digest
    if definition.evidence_status == "diagnostic":
        if promotion_digest is not None:
            raise BackendLaneCatalogError(
                f"diagnostic lane {key!r} cannot claim promotion authority"
            )
    else:
        promotion_digest = _tagged_digest(
            promotion_digest,
            field=f"lanes.{key}.promotion_authority_digest",
        )

    routing_digest = _tagged_digest(
        definition.routing_digest,
        field=f"lanes.{key}.routing_digest",
    )
    expected_fingerprint = _tagged_digest(
        definition.routing_fingerprint,
        field=f"lanes.{key}.routing_fingerprint",
    )
    route_path = _confined_route(
        route_root,
        definition.routing_path,
        field=f"lanes.{key}.routing_path",
    )
    try:
        route_bytes, _metadata = read_exact_regular(
            route_path,
            maximum=_MAX_ROUTE_BYTES,
        )
        routing = lint_routing_config(route_bytes)
    except (AuthorityIOError, OSError, ValueError) as exc:
        raise BackendLaneCatalogError(
            f"lane {key!r} routing file is invalid: {exc}"
        ) from exc
    if _digest_bytes(route_bytes) != routing_digest:
        raise BackendLaneCatalogError(
            f"lane {key!r} routing bytes do not match its digest"
        )
    observed_fingerprint = routing_fingerprint(routing)
    if observed_fingerprint != expected_fingerprint:
        raise BackendLaneCatalogError(
            f"lane {key!r} effective routing fingerprint drifted"
        )
    missing_roles = sorted(set(contract.roles) - set(routing.roles))
    if missing_roles:
        raise BackendLaneCatalogError(
            f"lane {key!r} does not explicitly route every {contract.product} "
            f"role: {missing_roles}"
        )
    unexpected_roles = sorted(set(routing.roles) - set(contract.roles))
    if unexpected_roles and not allow_additional_roles:
        raise BackendLaneCatalogError(
            f"lane {key!r} routes roles outside the {contract.product} "
            f"contract: {unexpected_roles}"
        )

    default_route = _effective_route("default", routing.default)
    role_routes = tuple(
        _effective_route(role, routing.for_role(role)) for role in contract.roles
    )
    for effective in (default_route, *role_routes):
        if (
            effective.backend,
            effective.runtime_backend,
            effective.transport,
        ) != (backend, runtime_backend, transport):
            raise BackendLaneCatalogError(
                f"lane {key!r} role {effective.role!r} has routing-backend/"
                f"runtime-backend/transport {effective.backend!r}/"
                f"{effective.runtime_backend!r}/{effective.transport!r}; expected "
                f"{backend!r}/{runtime_backend!r}/{transport!r}"
            )
        if (
            definition.evidence_status == "promotable"
            and effective.feature_admission != "enforce"
        ):
            raise BackendLaneCatalogError(
                f"promotable lane {key!r} role {effective.role!r} must enforce "
                "feature admission"
            )
    lane = BackendLane(
        key=key,
        lane_id=lane_id,
        backend=backend,
        runtime_backend=runtime_backend,
        transport=transport,
        routing_path=route_path,
        routing_digest=routing_digest,
        routing_fingerprint=observed_fingerprint,
        evidence_status=definition.evidence_status,
        promotion_authority_digest=promotion_digest,
        default_route=default_route,
        role_routes=role_routes,
    )
    if lane.promotion_eligible:
        if promotion_authority_validator is None:
            raise BackendLaneCatalogError(
                f"promotable lane {key!r} requires a product-owned promotion "
                "authority validator"
            )
        try:
            promotion_authority_validator(lane)
        except BackendLaneCatalogError:
            raise
        except Exception as exc:
            raise BackendLaneCatalogError(
                f"promotable lane {key!r} promotion authority was rejected: {exc}"
            ) from exc
    return lane


def validate_backend_lane_catalog(
    contract: BackendCatalogContract,
    *,
    route_root: Path | str,
    lanes: tuple[BackendLaneDefinition, ...],
    require_full_registry: bool = False,
    promotion_authority_validator: PromotionAuthorityValidator | None = None,
    allow_additional_roles: bool = False,
) -> BackendLaneCatalog:
    """Authenticate an ordered N-lane catalog without provider authority.

    ``require_full_registry`` additionally requires coverage of every public
    backend name and every selectable implementation transport.  It does not
    make any lane promotable; evidence status remains an explicit lane input.
    A promotable declaration additionally requires a product-owned authority
    validator.  This generic layer never treats a self-declared digest as
    qualification evidence.  ``allow_additional_roles`` is reserved for
    shared checked-in route files: additional routes remain fingerprinted but
    are excluded from the product contract's projected role closure.
    """

    _validate_transport_registry()
    _validate_contract(contract)
    if type(require_full_registry) is not bool:
        raise BackendLaneCatalogError("require_full_registry must be an exact bool")
    if type(allow_additional_roles) is not bool:
        raise BackendLaneCatalogError("allow_additional_roles must be an exact bool")
    if promotion_authority_validator is not None and not callable(
        promotion_authority_validator
    ):
        raise BackendLaneCatalogError(
            "promotion_authority_validator must be callable or None"
        )
    if type(lanes) is not tuple or len(lanes) < 2:
        raise BackendLaneCatalogError(
            "catalog requires an ordered tuple of at least two lanes"
        )
    root = Path(route_root).resolve()
    validated = tuple(
        _validate_lane(
            lane,
            contract=contract,
            route_root=root,
            promotion_authority_validator=promotion_authority_validator,
            allow_additional_roles=allow_additional_roles,
        )
        for lane in lanes
    )
    keys = tuple(lane.key for lane in validated)
    lane_ids = tuple(lane.lane_id for lane in validated)
    if len(keys) != len(set(keys)):
        raise BackendLaneCatalogError("catalog lane keys must be unique")
    if len(lane_ids) != len(set(lane_ids)):
        raise BackendLaneCatalogError("catalog lane ids must be unique")
    catalog = BackendLaneCatalog(contract=contract, lanes=validated)
    if require_full_registry:
        missing = REGISTERED_ADAPTER_IDENTITIES - catalog.adapter_identities
        if missing:
            rendered = [
                f"{backend}->{runtime_backend}/{transport}"
                for backend, runtime_backend, transport in sorted(missing)
            ]
            raise BackendLaneCatalogError(
                f"catalog does not cover registered adapter transports: {rendered}"
            )
    return catalog


def _profile_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise BackendLaneCatalogError(f"{field} must be an exact mapping")
    return value


def _profile_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    field: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise BackendLaneCatalogError(
            f"{field} fields drifted; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _profile_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise BackendLaneCatalogError(f"{field} must be exact non-empty text")
    return value


def _profile_transport_binding(
    backend: str,
    profile_transport: Any,
    *,
    field: str,
) -> tuple[str, str]:
    transport = _profile_text(profile_transport, field=field)
    binding = _PROFILE_TRANSPORT_BINDINGS.get((backend, transport))
    if binding is None:
        raise BackendLaneCatalogError(
            f"{field} has no exact registered binding for backend {backend!r}"
        )
    return binding


def _profile_lane_definition(
    *,
    key: str,
    raw_profile: Any,
    contract: BackendCatalogContract,
    route_root: Path,
) -> BackendLaneDefinition:
    profile = _profile_mapping(raw_profile, field=f"profiles.{key}")
    _profile_fields(profile, expected=_PROFILE_FIELDS, field=f"profiles.{key}")
    backend = _identifier(profile["backend"], field=f"profiles.{key}.backend")
    if backend not in BACKEND_NAMES:
        raise BackendLaneCatalogError(
            f"profiles.{key}.backend is not a registered adapter"
        )
    model = _profile_text(profile["model"], field=f"profiles.{key}.model")
    feature_admission = profile["feature_admission"]
    if feature_admission not in {"diagnostic", "enforce"}:
        raise BackendLaneCatalogError(
            f"profiles.{key}.feature_admission must be diagnostic or enforce"
        )
    availability = profile["availability"]
    if type(availability) is not str or availability not in _PROFILE_AVAILABILITY:
        raise BackendLaneCatalogError(
            f"profiles.{key}.availability is not a closed catalog status"
        )
    runtime_binding = profile["runtime_binding"]
    if (
        type(runtime_binding) is not str
        or runtime_binding not in _PROFILE_RUNTIME_BINDINGS
    ):
        raise BackendLaneCatalogError(
            f"profiles.{key}.runtime_binding is not a closed runtime binding"
        )
    if profile["promotion_evidence"] is not False:
        raise BackendLaneCatalogError(
            f"profiles.{key}.promotion_evidence must be false; profile metadata "
            "is not promotion authority"
        )
    runtime_backend, runtime_transport = _profile_transport_binding(
        backend,
        profile["transport"],
        field=f"profiles.{key}.transport",
    )

    routing_name = _profile_text(
        profile["routing"],
        field=f"profiles.{key}.routing",
    )
    routing_relative = Path(routing_name)
    if (
        routing_relative.is_absolute()
        or routing_relative.name != routing_name
        or routing_relative.suffix != ".yaml"
    ):
        raise BackendLaneCatalogError(
            f"profiles.{key}.routing must be one confined YAML filename"
        )
    route_path = _confined_route(
        route_root,
        routing_relative,
        field=f"profiles.{key}.routing",
    )
    try:
        route_bytes, _metadata = read_exact_regular(
            route_path,
            maximum=_MAX_ROUTE_BYTES,
        )
        routing = lint_routing_config(route_bytes)
    except (AuthorityIOError, OSError, ValueError) as exc:
        raise BackendLaneCatalogError(
            f"profile {key!r} routing file is invalid: {exc}"
        ) from exc

    missing_roles = sorted(set(contract.roles) - set(routing.roles))
    if missing_roles:
        raise BackendLaneCatalogError(
            f"profile {key!r} does not explicitly route every {contract.product} "
            f"role: {missing_roles}"
        )
    for role, config in (
        ("default", routing.default),
        *((role, routing.for_role(role)) for role in contract.roles),
    ):
        observed_runtime_backend, observed_transport = _runtime_identity_for_config(
            config
        )
        if (
            config.backend,
            observed_runtime_backend,
            observed_transport,
        ) != (backend, runtime_backend, runtime_transport):
            raise BackendLaneCatalogError(
                f"profile {key!r} role {role!r} does not match its exact "
                "backend/transport binding"
            )
        if config.model != model:
            raise BackendLaneCatalogError(
                f"profile {key!r} role {role!r} does not match catalog model "
                f"{model!r}"
            )
        if config.feature_admission != feature_admission:
            raise BackendLaneCatalogError(
                f"profile {key!r} role {role!r} does not match catalog "
                f"feature admission {feature_admission!r}"
            )
        if backend in {"local", "openai_compat"} and config.extra.get(
            "expected_served_model"
        ) != model:
            raise BackendLaneCatalogError(
                f"profile {key!r} role {role!r} must pin expected_served_model "
                "to its catalog model"
            )

    return BackendLaneDefinition(
        key=key,
        lane_id=key,
        backend=backend,
        runtime_backend=runtime_backend,
        transport=runtime_transport,
        routing_path=route_path,
        routing_digest=_digest_bytes(route_bytes),
        routing_fingerprint=routing_fingerprint(routing),
        evidence_status="diagnostic",
    )


def load_backend_profile_catalog(
    contract: BackendCatalogContract,
    *,
    catalog_path: Path | str,
    expected_catalog_digest: str | None = None,
    require_full_registry: bool = True,
) -> BackendLaneCatalog:
    """Bind one checked-in product profile catalog to N-lane identities.

    Loading is provider-free.  The catalog and every referenced route are read
    as bounded exact regular files; route fingerprints are derived from those
    measured bytes.  Profile metadata cannot declare promotion authority, so
    every projected lane remains diagnostic even when its route enforces the
    backend feature contract.
    """

    _validate_contract(contract)
    if type(require_full_registry) is not bool:
        raise BackendLaneCatalogError("require_full_registry must be an exact bool")
    if not isinstance(catalog_path, (Path, str)):
        raise BackendLaneCatalogError("catalog_path must be a path")
    source = Path(os.path.abspath(Path(catalog_path)))
    try:
        catalog_bytes, _metadata = read_exact_regular(
            source,
            maximum=_MAX_ROUTE_BYTES,
        )
        document = yaml.safe_load(catalog_bytes.decode("utf-8"))
    except (AuthorityIOError, OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BackendLaneCatalogError(
            f"profile catalog {source} is invalid: {exc}"
        ) from exc
    catalog_digest = _digest_bytes(catalog_bytes)
    if expected_catalog_digest is not None:
        expected = _tagged_digest(
            expected_catalog_digest,
            field="expected_catalog_digest",
        )
        if catalog_digest != expected:
            raise BackendLaneCatalogError(
                "profile catalog bytes do not match the expected digest"
            )

    root = _profile_mapping(document, field="profile catalog")
    _profile_fields(
        root,
        expected=frozenset({"schema_version", "product", "roles", "profiles"}),
        field="profile catalog",
    )
    if root["schema_version"] != PROFILE_CATALOG_SCHEMA:
        raise BackendLaneCatalogError(
            f"profile catalog schema_version must be {PROFILE_CATALOG_SCHEMA!r}"
        )
    if root["product"] != contract.product:
        raise BackendLaneCatalogError(
            f"profile catalog product must be {contract.product!r}"
        )
    raw_roles = root["roles"]
    if type(raw_roles) is not list or tuple(raw_roles) != contract.roles:
        raise BackendLaneCatalogError(
            "profile catalog roles do not match the ordered product contract"
        )
    raw_profiles = _profile_mapping(root["profiles"], field="profiles")
    if len(raw_profiles) < 2:
        raise BackendLaneCatalogError(
            "profile catalog requires at least two ordered profiles"
        )
    definitions = tuple(
        _profile_lane_definition(
            key=_identifier(key, field="profile key"),
            raw_profile=raw_profile,
            contract=contract,
            route_root=source.parent,
        )
        for key, raw_profile in raw_profiles.items()
    )
    if len({definition.routing_path for definition in definitions}) != len(
        definitions
    ):
        raise BackendLaneCatalogError(
            "profile catalog routing filenames must be unique"
        )
    catalog = validate_backend_lane_catalog(
        contract,
        route_root=source.parent,
        lanes=definitions,
        require_full_registry=require_full_registry,
        allow_additional_roles=True,
    )
    return dataclasses.replace(
        catalog,
        profile_catalog_path=source,
        profile_catalog_digest=catalog_digest,
    )


def _pair_status(arm_a: BackendLane, arm_b: BackendLane) -> EvidenceStatus:
    if arm_a.promotion_eligible and arm_b.promotion_eligible:
        return "promotable"
    return "diagnostic"


def generate_backend_lane_pairs(
    catalog: BackendLaneCatalog,
) -> tuple[BackendLanePair, ...]:
    """Return every unordered two-lane comparison in catalog order."""

    if type(catalog) is not BackendLaneCatalog:
        raise BackendLaneCatalogError("catalog must be BackendLaneCatalog")
    return tuple(
        BackendLanePair(
            arm_a=arm_a,
            arm_b=arm_b,
            reference_arm=None,
            evidence_status=_pair_status(arm_a, arm_b),
        )
        for arm_a, arm_b in combinations(catalog.lanes, 2)
    )


def generate_reference_arm_pairs(
    catalog: BackendLaneCatalog,
    *,
    reference_arm: str,
) -> tuple[BackendLanePair, ...]:
    """Compare every non-reference lane to one explicit first arm."""

    if type(catalog) is not BackendLaneCatalog:
        raise BackendLaneCatalogError("catalog must be BackendLaneCatalog")
    reference = catalog.lane(reference_arm)
    return tuple(
        BackendLanePair(
            arm_a=reference,
            arm_b=candidate,
            reference_arm=reference.key,
            evidence_status=_pair_status(reference, candidate),
        )
        for candidate in catalog.lanes
        if candidate.key != reference.key
    )


__all__ = [
    "BackendCatalogContract",
    "BackendLane",
    "BackendLaneCatalog",
    "BackendLaneCatalogError",
    "BackendLaneDefinition",
    "BackendLanePair",
    "EffectiveRouteIdentity",
    "EvidenceStatus",
    "PROFILE_CATALOG_SCHEMA",
    "PromotionAuthorityValidator",
    "REGISTERED_ADAPTER_IDENTITIES",
    "REGISTERED_ADAPTER_TRANSPORTS",
    "REGISTERED_PROFILE_TRANSPORTS",
    "generate_backend_lane_pairs",
    "generate_reference_arm_pairs",
    "load_backend_profile_catalog",
    "validate_backend_lane_catalog",
]
