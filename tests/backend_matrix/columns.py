# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Columns: the shipped catalog profiles, named by driver kind and role.

This is the only module of the matrix that holds route constants and profile
names. Model ids are read from the catalogs at run time and never typed. Cell
ids, ledger keys and the declaration snapshot use kind and role, so a renamed
profile or model edits the role table below and nothing else.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml  # type: ignore[import-untyped]

from assay import BackendFeatureSet
from crucible.backends.config import BackendRouting
from crucible.backends.pricing import DEFAULT_PRICING, ModelPricing
from crucible.products.deepswe.backend_convergence import (
    load_deepswe_backend_profile_catalog,
)
from crucible.products.swebench.backend_convergence import (
    load_swebench_backend_profile_catalog,
)

from .drivers.base import MatrixSetupError

TOY_KIND = "toy"
TOY_MODEL = "matrix-toy-model"
TOY_ADAPTER = "toy"

# The behaviour columns come from the first product; the second adds no column
# and must list the same profiles (static cell P04).
PRODUCTS: Mapping[str, Any] = {
    "deepswe": load_deepswe_backend_profile_catalog,
    "swebench": load_swebench_backend_profile_catalog,
}
COLUMN_PRODUCT = "deepswe"
# The profile catalog of a product, beside the routing files it names.
CATALOG_FILE = "backend_profile_catalog.yaml"

HOSTED_GATEWAY_KIND = "openai_compat:hosted"
LOCAL_GATEWAY_KIND = "openai_compat:local"
CLAUDE_KIND = "claude"
DEEPSEEK_KIND = "deepseek"
KIMI_KIND = "kimi"
CODEX_LEGACY_KIND = "codex:legacy"
CODEX_OFFICIAL_KIND = "codex:official"

# (routing backend, adapter transport) of a catalog lane -> driver kind.
KIND_TABLE: Mapping[tuple[str, str], str] = {
    ("openai_compat", "openai_compatible_http"): HOSTED_GATEWAY_KIND,
    ("local", "openai_compatible_http"): LOCAL_GATEWAY_KIND,
    ("deepseek", "deepseek_responses_worker"): DEEPSEEK_KIND,
    ("kimi", "kimi_agent_sdk_worker"): "kimi",
    ("claude", "claude_agent_sdk"): CLAUDE_KIND,
    ("codex", "legacy_app_server"): "codex:legacy",
    ("codex", "official_sdk"): "codex:official",
}
REAL_KINDS: tuple[str, ...] = tuple(sorted(set(KIND_TABLE.values())))
REFERENCE_KINDS: frozenset[str] = frozenset({CLAUDE_KIND, "codex:legacy", "codex:official"})
OPEN_WEIGHT_KINDS: frozenset[str] = frozenset(REAL_KINDS) - REFERENCE_KINDS
# Kinds whose wire names no served model: no reply of their dialect can name
# one, so the contract clause about identity has nothing to hold the trace's
# served model against. The declared difference of that wire asserts instead
# that the trace records none, and it is owed on every result cell of these
# kinds (static cell H03 holds the ledger's list to it).
UNNAMED_SERVED_MODEL_KINDS: frozenset[str] = frozenset({CODEX_LEGACY_KIND, CODEX_OFFICIAL_KIND})

# Catalog profile -> role within its kind. The primary of a kind is written
# down here; it does not follow the order of the catalog file.
ROLE_TABLE: Mapping[str, str] = {
    "codex_astra": "primary",
    "codex_official_sdk_diagnostic": "primary",
    "claude_fable51": "primary",
    "claude_mythos51": "secondary",
    "deepseek_responses_diagnostic": "primary",
    "kimi_sdk": "blocked",
    "kimi_sdk_diagnostic": "primary",
    "kimi_k31_pro_sdk_diagnostic": "secondary",
    "openai_glm53_diagnostic": "primary",
    "openai_kimi_k31_pro_diagnostic": "secondary",
    "local_glm53_diagnostic": "primary",
}
_ROLE_RE = re.compile(r"(primary|secondary|blocked)(-[2-9])?\Z")

# The credential every scripted route carries. It is a fake; C9 looks for it in
# results, traces, logs and debug dumps.
CANARY_CREDENTIAL = "matrix-canary-credential-do-not-log-51c7e0"

# What each ``${VAR}`` of the shipped routing files expands to in scripted
# mode: the production endpoint (so the route-qualified profile applies) or,
# for the local kind, a loopback address. A driver replaces the loopback
# placeholder by its provider; nothing else is rewritten.
SCRIPTED_ENV: Mapping[str, str] = {
    "INFER_BASE_URL": "https://inference-api.nvidia.com/v1",
    "INFER_AUTH_TOKEN": CANARY_CREDENTIAL,
    "KIMI_BASE_URL": "https://api.moonshot.ai/v1",
    "MOONSHOT_API_KEY": CANARY_CREDENTIAL,
    "CRUCIBLE_LOCAL_URL": "http://127.0.0.1:9/v1",
}

# The ``${VAR}`` a kind's shipped route takes its endpoint from, for the kinds
# whose driver needs to know it: the hosted gateway answers that origin on
# loopback, the local kind points it at the scripted provider.
ENDPOINT_ENV: Mapping[str, str] = {
    HOSTED_GATEWAY_KIND: "INFER_BASE_URL",
    LOCAL_GATEWAY_KIND: "CRUCIBLE_LOCAL_URL",
    # The locked worker of the kind addresses this origin; the worker harness
    # answers it on loopback.
    KIMI_KIND: "KIMI_BASE_URL",
}

# A worker adapter names no endpoint in its route: its locked worker holds the
# production origin. The origin a worker driver answers on loopback, by kind.
WORKER_ORIGIN: Mapping[str, str] = {
    DEEPSEEK_KIND: "https://api.deepseek.com",
}

# The credential variable an adapter reads from the environment itself, where
# its routing file has no ``${VAR}`` for it. The driver of the kind gives it
# the canary credential for as long as it holds a backend open.
ADAPTER_CREDENTIAL_ENV: Mapping[str, str] = {
    DEEPSEEK_KIND: "DEEPSEEK_API_KEY",
}

# Credential and configuration variables a developer machine may hold. No cell
# may see one: the per-test environment deletes each. They are kept apart from
# ``SCRIPTED_ENV``, which must equal the ``${VAR}`` references of the routing
# files. A driver that needs one of these gives it a fake value when it lands.
CLEARED_ENV: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "DEEPSEEK_API_KEY",
    "CRUCIBLE_LOCAL_API_KEY",
    "CRUCIBLE_LOCAL_MODEL",
    "CRUCIBLE_BACKEND_CONFIG",
    "CRUCIBLE_CODEX_WORKER_PYTHON",
    "CRUCIBLE_DEEPSEEK_WORKER_PYTHON",
    "CRUCIBLE_KIMI_WORKER_PYTHON",
    # The pinned Agent SDK reads this one in the parent process, and the
    # session route refuses to run while it is set.
    "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT",
    "CODEX_HOME",
    "KIMI_SHARE_DIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    # An HTTP client that trusts its environment would route a cell through a
    # developer machine's proxy.
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    # The Kimi adapter hands the TLS trust paths of its environment on to its
    # worker, whose client reads them when it is built: one that names no
    # file ends the worker before its first request.
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)

# Worker sub-project (``backends/<name>_worker``) -> the kind it serves. Static
# cell H01 has one cell per worker.
WORKER_KINDS: Mapping[str, str] = {
    "codex": "codex:official",
    "deepseek": DEEPSEEK_KIND,
    "kimi": "kimi",
}

# The closed routing overlay: the only routing fields a driver may add, by
# kind. A ``base_url`` is never among them.
ALLOWED_OVERLAY: Mapping[str, tuple[str, ...]] = {
    CLAUDE_KIND: ("extra.cli_path",),
    "codex:legacy": ("extra.cli_path",),
    DEEPSEEK_KIND: ("extra.expected_worker_runtime_fingerprint",),
}

# Routing role by the capability a cell asks for.
ROLE_BY_CAPABILITY: Mapping[str, str] = {
    "text": "architect_critic",
    "read_only": "architect_planner",
    "workspace_write": "implementer",
}

# One uniform matrix rate for behaviour cells; shipped prices are a static
# cell's business (P05).
MATRIX_RATE = ModelPricing(input_per_mtok=2.0, output_per_mtok=8.0)

# The closed routing extra by which a route opts into a diagnostic cost upper
# bound. Its routing file states what the route then asks of a caller: it
# returns a result only for an invocation with a finite positive budget, and
# its limit guard binds only with the cost policy the file names and a durable
# ledger. A cell on such a route is called as the route asks: it carries this
# budget, which no scripted exchange reaches, unless its scenario sets a
# budget of its own, and its guarded lens binds that policy.
COST_BOUND_EXTRA = "diagnostic_cost_upper_bound"
ROUTE_BUDGET_USD = 1.0


@dataclass(frozen=True)
class Column:
    """One catalog profile (or one toy column) as the matrix names it."""

    kind: str
    role: str
    profile: str
    backend: str
    adapter: str
    model: str
    routing: tuple[tuple[str, Path], ...] = ()
    features: BackendFeatureSet | None = None

    @property
    def id(self) -> str:
        return f"{self.kind}/{self.role}"

    def routing_path(self, product: str) -> Path:
        for name, path in self.routing:
            if name == product:
                return path
        raise MatrixSetupError(f"column {self.id} has no {product} routing file")

    def routed_by(self, product: str) -> Column:
        """This column as ``product`` routes it.

        A driver builds from the routing file of the product the behaviour
        columns come from. The returned column names the file of ``product``
        there, for a cell that runs on the routing files of every product.
        """

        return dataclasses.replace(
            self, routing=((COLUMN_PRODUCT, self.routing_path(product)),)
        )


def build_columns(
    lanes: Sequence[Any],
    *,
    product: str,
    kind_table: Mapping[tuple[str, str], str] = KIND_TABLE,
    role_table: Mapping[str, str] = ROLE_TABLE,
) -> tuple[Column, ...]:
    """Name every catalog lane by kind and role, or say why it cannot be."""

    columns: list[Column] = []
    seen: dict[tuple[str, str], str] = {}
    for lane in lanes:
        pair = (lane.backend, lane.transport)
        kind = kind_table.get(pair)
        if kind is None:
            raise MatrixSetupError(
                f"catalog profile {lane.key!r} has (backend, transport) {pair}, "
                "which maps to no driver kind; add it to the kind table"
            )
        role = role_table.get(lane.key)
        if role is None:
            raise MatrixSetupError(
                f"catalog profile {lane.key!r} has no role; add it to the role table"
            )
        if _ROLE_RE.fullmatch(role) is None:
            raise MatrixSetupError(f"role {role!r} of profile {lane.key!r} is malformed")
        if (kind, role) in seen:
            raise MatrixSetupError(
                f"profiles {seen[(kind, role)]!r} and {lane.key!r} share the "
                f"column {kind}/{role}; number the second (for example "
                f"{role.split('-')[0]}-2)"
            )
        seen[(kind, role)] = lane.key
        columns.append(
            Column(
                kind=kind,
                role=role,
                profile=lane.key,
                backend=lane.backend,
                adapter=lane.runtime_backend,
                model=lane.default_route.model,
                routing=((product, Path(lane.routing_path)),),
            )
        )
    present = {lane.key for lane in lanes}
    for profile in role_table:
        if profile not in present:
            raise MatrixSetupError(
                f"the role table names profile {profile!r}, which the {product} "
                "catalog does not list"
            )
    for kind in sorted(set(kind_table.values())):
        if not any(column.kind == kind for column in columns):
            raise MatrixSetupError(f"driver kind {kind!r} has no catalog column")
    return tuple(columns)


@functools.cache
def catalog_lanes(product: str) -> tuple[Any, ...]:
    """The lanes of one product's shipped catalog, through the product loader."""

    return tuple(PRODUCTS[product]().lanes)


@functools.cache
def real_columns() -> tuple[Column, ...]:
    """The behaviour columns, each with every product's routing file it has."""

    others = {
        product: {lane.key: Path(lane.routing_path) for lane in catalog_lanes(product)}
        for product in PRODUCTS
        if product != COLUMN_PRODUCT
    }
    merged: list[Column] = []
    for column in build_columns(catalog_lanes(COLUMN_PRODUCT), product=COLUMN_PRODUCT):
        routing = list(column.routing)
        for product, by_profile in others.items():
            if column.profile in by_profile:
                routing.append((product, by_profile[column.profile]))
        merged.append(
            Column(
                kind=column.kind,
                role=column.role,
                profile=column.profile,
                backend=column.backend,
                adapter=column.adapter,
                model=column.model,
                routing=tuple(routing),
            )
        )
    return tuple(merged)


def toy_columns(
    rows: Mapping[tuple[str, str], BackendFeatureSet],
    governing: Sequence[str],
) -> tuple[Column, ...]:
    """One toy column per distinct combination of governing declared values.

    Computed from the declaration snapshot, never typed: each toy column takes
    the whole declared profile of the first snapshot row (in key order) that
    shows its combination, and is named after the combination.
    """

    columns: dict[tuple[str, ...], Column] = {}
    for key in sorted(rows):
        profile = rows[key]
        values = tuple(str(getattr(profile, field)) for field in governing)
        if values in columns:
            continue
        columns[values] = Column(
            kind=TOY_KIND,
            role="+".join(values),
            profile="toy:" + "+".join(values),
            backend=TOY_KIND,
            adapter=TOY_ADAPTER,
            model=TOY_MODEL,
            features=profile,
        )
    return tuple(columns[values] for values in sorted(columns))


def matrix_rate(model: str) -> ModelPricing:
    """The one matrix rate, in the shape pricing accepts for ``model``."""

    built_in = DEFAULT_PRICING.get(model)
    if built_in is None or built_in.long_context_threshold_tokens is None:
        return MATRIX_RATE
    # Pricing refuses to override a model that has a long-context tier unless
    # the whole tier is repeated; the matrix repeats it at its one rate.
    return dataclasses.replace(
        MATRIX_RATE,
        long_context_threshold_tokens=built_in.long_context_threshold_tokens,
        long_context_input_per_mtok=MATRIX_RATE.input_per_mtok,
        long_context_output_per_mtok=MATRIX_RATE.output_per_mtok,
        long_context_cache_write_per_mtok=MATRIX_RATE.input_per_mtok,
        long_context_cache_read_per_mtok=MATRIX_RATE.input_per_mtok,
    )


def rate_card() -> dict[str, ModelPricing]:
    """The matrix rate for the toy model and every model a shipped route names.

    A lane's roles are separate routes and may name another model than its
    default route; a cell is priced at the model of the route of its role.
    """

    models = {TOY_MODEL}
    for product in PRODUCTS:
        for lane in catalog_lanes(product):
            models.add(lane.default_route.model)
            models.update(route.model for route in lane.role_routes)
    return {model: matrix_rate(model) for model in sorted(models)}


def write_rate_card(path: Path) -> Path:
    """Write the rate card as a pricing file that pricing discovery can load."""

    document = {
        model: {
            name: value
            for name, value in dataclasses.asdict(rate).items()
            if value is not None
        }
        for model, rate in rate_card().items()
    }
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    return path


@contextlib.contextmanager
def scripted_environment(**endpoints: str) -> Iterator[None]:
    """Expand the routing files' ``${VAR}`` references to their scripted values.

    ``endpoints`` replaces the value of a scripted variable for the block: the
    address of the scripted provider where a route's endpoint is loopback.
    """

    unknown = sorted(set(endpoints) - set(SCRIPTED_ENV))
    if unknown:
        raise MatrixSetupError(f"{unknown} are not scripted route variables")
    previous = {name: os.environ.get(name) for name in SCRIPTED_ENV}
    os.environ.update({**SCRIPTED_ENV, **endpoints})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def adapter_credential(kind: str) -> Iterator[None]:
    """Give the adapter of ``kind`` the canary credential for the block.

    For an adapter that reads its credential from the environment at each
    dispatch. The variable is restored afterwards: no test outside the block
    sees it, and a real one on a developer machine is never used.
    """

    try:
        name = ADAPTER_CREDENTIAL_ENV[kind]
    except KeyError as exc:
        raise MatrixSetupError(
            f"kind {kind} reads no credential from the environment itself"
        ) from exc
    previous = os.environ.get(name)
    os.environ[name] = CANARY_CREDENTIAL
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def overlaid_routing(
    routing: BackendRouting, fields: Mapping[str, Any], *, kind: str, backend: str
) -> BackendRouting:
    """``routing`` with a driver's overlay on every route of ``backend``.

    Only the fields the closed list allows for ``kind``, all of them route
    extras. Everything else of the shipped route stays: the overlay adds no
    endpoint, no credential and no model.
    """

    allowed = ALLOWED_OVERLAY.get(kind, ())
    refused = sorted(set(fields) - set(allowed))
    if refused:
        raise MatrixSetupError(
            f"kind {kind} may overlay {list(allowed)} on its shipped route, not {refused}"
        )
    extras = {name.removeprefix("extra."): value for name, value in fields.items()}

    def overlaid(config: Any, effective_backend: str | None) -> Any:
        if effective_backend != backend:
            return config  # a route of another backend family is left alone
        return dataclasses.replace(config, extra={**config.extra, **extras})

    return dataclasses.replace(
        routing,
        default=overlaid(routing.default, routing.default.backend),
        roles={
            name: overlaid(config, routing.for_role(name).backend)
            for name, config in routing.roles.items()
        },
    )


def catalog_availability(column: Column, product: str) -> str:
    """The word the profile catalog of ``product`` has for a column.

    Read from the catalog file itself, which lies beside the routing files
    it names: the product loader validates the word and does not keep it.
    """

    catalog = column.routing_path(product).with_name(CATALOG_FILE)
    profiles = yaml.safe_load(catalog.read_text(encoding="utf-8"))["profiles"]
    return str(profiles[column.profile]["availability"])


def catalog_literals() -> frozenset[str]:
    """Every profile name and model id the catalogs hold (for the hygiene grep)."""

    literals: set[str] = set()
    for product in PRODUCTS:
        for lane in catalog_lanes(product):
            literals.add(lane.key)
            literals.add(lane.default_route.model)
            literals.update(route.model for route in lane.role_routes)
    return frozenset(literals)
