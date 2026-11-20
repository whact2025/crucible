# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""K3.1 Pro product profiles are explicit, confined, and bound to the cost policy."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from crucible.backends import config as config_module
from crucible.backends import cost_bounds
from crucible.backends.config import (
    DEFAULT_ROUTING,
    lint_routing_config,
    load_routing,
    resolve_for_role,
    routing_fingerprint,
)
from crucible.backends.features import FeatureAdmissionError
from crucible.backends.kimi import KimiBackend
from crucible.backends.limits import (
    INVOCATION_LIMIT_LEDGER_LEAF,
    InvocationLimitContractError,
    InvocationLimitLedger,
    InvocationLimitPolicy,
)
from crucible.products.deepswe.backend_convergence import (
    load_deepswe_backend_profile_catalog,
)
from crucible.products.swebench.backend_convergence import (
    load_swebench_backend_profile_catalog,
)


ROOT = Path(__file__).parents[1]
PROFILE = "kimi_k31_pro_sdk_diagnostic"
MODEL = "kimi-k3.1-pro"
POLICY = "kimi-k3.1-pro-cache-read-upper-bound-v1"
WRITERS = {"implementer", "breach_defender"}
# The two profiles with diagnostic admission; the E12 arm profile enforces.
DIAGNOSTIC_PROFILES = (
    "backends_kimi_sdk_diagnostic.yaml",
    "backends_kimi_k31_pro_sdk_diagnostic.yaml",
)
LOADERS = {
    "swebench": load_swebench_backend_profile_catalog,
    "deepswe": load_deepswe_backend_profile_catalog,
}


def _path(product: str) -> Path:
    return ROOT / "products" / product / "backends_kimi_k31_pro_sdk_diagnostic.yaml"


def _effective_routes(routing: Any) -> list[tuple[str, Any]]:
    return [
        ("default", routing.default),
        *((role, routing.for_role(role)) for role in routing.roles),
    ]


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
        assert route["model"] == MODEL
        assert route["effort"] == "max"
        assert route["feature_admission"] == "diagnostic"
        assert route["base_url"] == "${KIMI_BASE_URL}"
        assert route["api_key"] == "${MOONSHOT_API_KEY}"
        expected_extra: dict[str, Any] = {
            "worker_source": "project_locked",
            "max_response_tokens": 32_768,
            "diagnostic_cost_upper_bound": True,
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
    writer_lane = catalog.lane("kimi_sdk_diagnostic")
    assert lane.routing_path == _path(product).resolve()
    assert lane.evidence_status == "diagnostic"
    assert lane.promotion_eligible is False
    assert lane.promotion_authority_digest is None
    for route in (lane.default_route, *lane.role_routes):
        assert route.model == MODEL
        assert route.effort == "max"
    # Same model, separate route identities: the all-role writer profile
    # differs in writer scope and response caps.
    assert lane.routing_digest != writer_lane.routing_digest
    assert lane.routing_fingerprint != writer_lane.routing_fingerprint
    assert lane.routing_fingerprint == routing_fingerprint(
        lint_routing_config(_path(product))
    )
    for route in (writer_lane.default_route, *writer_lane.role_routes):
        assert route.model == MODEL
        assert route.effort == "max"
    assert writer_lane.promotion_eligible is False
    assert not (
        ROOT / "products" / product / "backends_kimi_k3_sdk_diagnostic.yaml"
    ).exists()
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
        assert kwargs["model"] == MODEL
        assert kwargs["effort"] == "max"
        assert kwargs["max_response_tokens"] == 32_768
        assert kwargs["worker_source"] == "project_locked"
        assert kwargs["diagnostic_cost_upper_bound"] is True
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


@pytest.mark.parametrize("product", LOADERS)
@pytest.mark.parametrize("filename", DIAGNOSTIC_PROFILES)
def test_diagnostic_profiles_carry_what_the_cost_bound_policy_requires(
    product: str,
    filename: str,
) -> None:
    # Routing validation admits extra.diagnostic_cost_upper_bound only with
    # the policy's exact model, effort max, diagnostic admission, the public
    # Moonshot endpoint and the project-locked worker. Every effective route
    # of both diagnostic profiles carries exactly that.
    path = ROOT / "products" / product / filename
    linted = lint_routing_config(path)
    loaded = load_routing(config_path=path)
    assert MODEL == cost_bounds.DIAGNOSTIC_COST_MODEL
    assert POLICY == cost_bounds.DIAGNOSTIC_COST_POLICY
    for routing, endpoint in (
        (linted, "${KIMI_BASE_URL}"),
        (loaded, "https://api.moonshot.ai/v1"),
    ):
        for role, route in _effective_routes(routing):
            assert route.backend == "kimi", role
            assert route.model == MODEL, role
            assert route.effort == "max", role
            assert route.feature_admission == "diagnostic", role
            assert route.base_url == endpoint, role
            assert route.extra["worker_source"] == "project_locked", role
            assert route.extra["diagnostic_cost_upper_bound"] is True, role


@pytest.mark.parametrize("product", LOADERS)
def test_enforcing_arm_profile_cannot_carry_the_cost_bound_opt_in(product: str) -> None:
    # E12 Arm C must enforce feature admission, and routing validation admits
    # the cost-bound opt-in only under diagnostic admission. The arm profile
    # stays strict; neither check is relaxed for it.
    path = ROOT / "products" / product / "backends_kimi_sdk.yaml"
    for role, route in _effective_routes(lint_routing_config(path)):
        assert route.model == MODEL, role
        assert route.effort == "max", role
        assert route.feature_admission == "enforce", role
        assert "diagnostic_cost_upper_bound" not in route.extra, role
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["default"]["extra"]["diagnostic_cost_upper_bound"] = True
    with pytest.raises(ValueError, match="diagnostic_cost_upper_bound requires exact"):
        lint_routing_config(yaml.safe_dump(raw).encode("utf-8"))


@pytest.mark.parametrize("product", LOADERS)
@pytest.mark.parametrize("filename", DIAGNOSTIC_PROFILES)
def test_diagnostic_profile_without_the_cost_policy_is_refused_before_transport(
    product: str,
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The profile's adapter flag and the run's --diagnostic-cost-policy must
    # agree. A run with diagnostic limits but without the policy is refused
    # when the backend is bound: no worker starts and no request is paid.
    dispatched: list[Any] = []
    constructed: list[KimiBackend] = []

    def real_adapter(name: str, **kwargs: Any) -> KimiBackend:
        assert name == "kimi"
        assert kwargs["worker_source"] == "project_locked"
        backend = KimiBackend(
            model=kwargs["model"],
            effort=kwargs["effort"],
            capability=kwargs["capability"],
            expected_base_url=kwargs["expected_base_url"],
            diagnostic_cost_upper_bound=kwargs["diagnostic_cost_upper_bound"],
            _worker_runner=lambda *args, **_kwargs: dispatched.append(args),
        )
        constructed.append(backend)
        return backend

    monkeypatch.setattr(config_module, "_resolve_backend", real_adapter)
    routing = load_routing(config_path=ROOT / "products" / product / filename)

    def resolve(policy: InvocationLimitPolicy, run: str) -> Any:
        run_dir = tmp_path / run
        run_dir.mkdir()
        return resolve_for_role(
            "architect_planner",
            routing=routing,
            capability="read_only",
            native_shell=False,
            allow_trusted_project_instructions=False,
            cwd=str(tmp_path.resolve()),
            limit_policy=policy,
            limit_ledger=InvocationLimitLedger(run_dir / INVOCATION_LIMIT_LEDGER_LEAF),
        )

    try:
        with pytest.raises(InvocationLimitContractError, match="explicit opt-in"):
            resolve(InvocationLimitPolicy("diagnostic", 600, True), "without-policy")
        backend = resolve(
            InvocationLimitPolicy("diagnostic", 600, True, POLICY), "with-policy",
        )
        assert backend.diagnostic_cost_upper_bound is True
        assert (backend.name, backend.model, backend.effort) == ("kimi", MODEL, "max")
    finally:
        for item in constructed:
            item.close()
    assert len(constructed) == 2
    assert dispatched == []


@pytest.mark.parametrize(
    ("path", "digest"),
    [
        (
            "swebench/backends_kimi_sdk.yaml",
            "3a866437e9976748dbc8b02a7fcf3f1d56ac9003369fdf19d5c54768ae811c79",
        ),
        (
            "swebench/backends_kimi_sdk_diagnostic.yaml",
            "3dcedeca5c2bf88798ce67006427b1e94260f29c5313f5473fc7ef7bc14c0342",
        ),
        (
            "swebench/e12_arms.yaml",
            "f6279cb9a4950852f295139b43ca684afba19278f72ddc3786804fbb6319cd27",
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
            "283fcc6df73525667d9b50b405acb4ed7a6faca96866b7fea20aa202374c6c53",
        ),
        (
            "deepswe/backends_kimi_sdk_diagnostic.yaml",
            "847a868cffcfcef12aacff8b0a709ef990c61cd2fe8175b271dc1a104c4a6c65",
        ),
        (
            "deepswe/e12_arms.yaml",
            "4a12e6d8f1ec27e3650434daaf3df897ed822f4460866aedb81769096dc3a1f0",
        ),
        (
            "deepswe/backend_convergence_matrix.yaml",
            "cc66a06a628013da307f5e703a207f4c3e9da0df7465c4ec9a94ae4372dfa9bd",
        ),
    ],
)
def test_profiles_and_matrices_bind_exact_reviewed_bytes(
    path: str,
    digest: str,
) -> None:
    assert hashlib.sha256((ROOT / "products" / path).read_bytes()).hexdigest() == digest
