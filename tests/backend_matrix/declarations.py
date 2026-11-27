# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Declarations: they select a scenario's contract, so they are pinned.

``routed`` is the feature profile routing itself computes for a column, with
the three arguments routing passes; ``routed_model`` and ``routed_endpoint``
are the model and the endpoint the route of a cell's role names. A scenario's
difference table maps each value of its governing key to ``BASE``,
``FALLBACK`` or ``REFUSED``; the variant is selected from ``routed``, never
from what a backend reports about itself. The checked-in snapshot pins every
column's declared profile (static cell P04), so a declaration cannot be
weakened to turn a cell green without an edit to a test file.

A governing key is a declared field, with one exception: a closed route
extra that keys a table (:data:`KEYING_EXTRAS`). Its value is no declared
value and no part of ``routed``. It is the state of the extra as routing
states it for the route of the cell's role, one of three
(:data:`EXTRA_STATES`), computed through routing's public validation of the
column's routing document (:func:`extra_state`) and never typed for a
column that has a routing file.

Checking declarations against observed behaviour arrives with pack B.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence
from urllib.parse import urlsplit

import yaml  # type: ignore[import-untyped]

from assay import CONSERVATIVE_BACKEND_FEATURES, BackendFeatureSet
from crucible.backends.config import BackendRouting, lint_routing_config, load_routing
from crucible.backends.features import feature_profile_for_route
from crucible.backends.limits import (
    InvocationLimitAdmissionError,
    InvocationLimitRequest,
    resolve_invocation_limits,
)

from . import columns as matrix_columns
from .columns import COLUMN_PRODUCT, Column
from .contract import Deviation
from .drivers.base import MatrixSetupError

FEATURE_FIELDS: tuple[str, ...] = tuple(
    item.name
    for item in dataclasses.fields(BackendFeatureSet)
    if item.name != "schema_version"
)
SNAPSHOT_PATH = Path(__file__).with_name("declared_profiles.yaml")
SNAPSHOT_SCHEMA = "crucible.backend-matrix.declared-profiles.v1"
ROUTED_ROLES: tuple[str, ...] = tuple(matrix_columns.ROLE_BY_CAPABILITY.values())


@dataclass(frozen=True)
class Variant:
    """What a declared value means for a scenario: one of exactly three kinds."""

    kind: Literal["BASE", "FALLBACK", "REFUSED"]
    name: str = ""
    citation: str = ""
    # The declared difference of the ledger this fallback is, if it is one.
    difference: str = ""


BASE = Variant("BASE")


def FALLBACK(  # noqa: N802 - table vocabulary
    name: str, citation: str, *, difference: str = ""
) -> Variant:
    """A declared alternative, asserted positively; cites where it is documented.

    ``difference`` names the declared difference of the ledger whose
    alternative contract this fallback asserts.
    """

    if not name or not citation:
        raise MatrixSetupError("a FALLBACK needs a name and a product citation")
    return Variant("FALLBACK", name, citation, difference)


def REFUSED(error_type: str) -> Variant:  # noqa: N802 - table vocabulary
    """The route refuses before transport with this exception type."""

    return Variant("REFUSED", error_type)


Differences = Mapping[str, Mapping[str, Variant]]


def routed(column: Column, product: str = COLUMN_PRODUCT) -> BackendFeatureSet:
    """The profile routing computes for a column: backend, model and base_url.

    ``base_url`` is whatever the loaded route holds after the scripted
    ``${VAR}`` expansion, ``None`` when the route has none. The matrix never
    passes an endpoint of its own to this call.
    """

    if column.features is not None:
        return column.features
    routing = _routing(column, product)
    profiles = {}
    for role in ROUTED_ROLES:
        config = routing.for_role(role)
        profiles[role] = feature_profile_for_route(
            str(config.backend), model=config.model, base_url=config.base_url
        )
    if len(set(profiles.values())) != 1:
        raise MatrixSetupError(
            f"column {column.id}: the roles {ROUTED_ROLES} of its {product} route "
            "resolve to different feature profiles"
        )
    return profiles[ROUTED_ROLES[0]]


def routed_model(column: Column, role: str, product: str = COLUMN_PRODUCT) -> str:
    """The model the route of ``role`` names: what the wire must carry.

    The roles of one routing file are separate entries and may name different
    models, so the reference of a cell is the route of its own role. A column
    without a routing file (a toy column) has one model for every role.
    """

    if not column.routing:
        return column.model
    return str(_routing(column, product).for_role(role).model)


def endpoint_identity(url: str) -> str:
    """``url`` as a run record may name it: scheme, host, port and path.

    The user-info part and the query of an endpoint may hold a credential, so
    a trace carries neither (clause C9), and neither does the reference.
    """

    parts = urlsplit(url)
    if not parts.scheme or not parts.hostname:
        raise MatrixSetupError(f"route endpoint {url!r} is not an absolute URL")
    return f"{parts.scheme}://{parts.netloc.rpartition('@')[2]}{parts.path}"


def routed_endpoint(column: Column, role: str, product: str = COLUMN_PRODUCT) -> str | None:
    """The endpoint the route of ``role`` names: what the trace must record.

    Read from the column's routing file as routing loads it, with the
    ``${VAR}`` values in force at the moment of the call. A driver points a
    loopback route at its scripted provider for the time the backend is open,
    so the runner asks while it is. ``None`` when the route names no endpoint.
    """

    base_url = load_routing(config_path=column.routing_path(product)).for_role(role).base_url
    return endpoint_identity(str(base_url)) if base_url else None


def _routing(column: Column, product: str) -> BackendRouting:
    """The column's routing file of ``product``, as routing itself loads it."""

    with matrix_columns.scripted_environment():
        return load_routing(config_path=column.routing_path(product))


def route_extra(
    column: Column, role: str, name: str, product: str = COLUMN_PRODUCT
) -> Any:
    """A closed extra of the route of ``role``, as its routing file states it.

    ``None`` where the route has no such extra, and for a column without a
    routing file (a toy column).
    """

    if not column.routing:
        return None
    return _routing(column, product).for_role(role).extra.get(name)


# How a routing file names a variable in the place of a value.
_VARIABLE_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def credential_variable(path: Path, role: str) -> str | None:
    """The variable the route of ``role`` in the routing file ``path`` takes its key from.

    Read through routing's own validation of the file, which resolves no
    variable reference: the key of the route is then the reference as the
    file writes it. ``None`` where the route names no key. A key that is
    anything but one whole reference (a literal, a reference inside other
    text) is an error: no variable could be taken away from such a route.
    """

    return _key_variable(lint_routing_config(path).for_role(role).api_key, role, path)


def credential_variables(path: Path) -> dict[str, str | None]:
    """The variable every route of the routing file ``path`` takes its key from.

    The default route and the route of every role the file names, each as
    :func:`credential_variable` reads it.
    """

    routing = lint_routing_config(path)
    return {
        "default": _key_variable(routing.default.api_key, "default", path),
        **{
            role: _key_variable(routing.for_role(role).api_key, role, path)
            for role in sorted(routing.roles)
        },
    }


def _key_variable(key: Any, route: str, path: Path) -> str | None:
    if key is None:
        return None
    reference = _VARIABLE_REFERENCE.fullmatch(str(key))
    if reference is None:
        raise MatrixSetupError(
            f"the route {route!r} of {path.name} names its key as a value, not as "
            "one variable reference"
        )
    return reference.group(1)


def cost_bounded(column: Column, role: str, product: str = COLUMN_PRODUCT) -> bool:
    """Whether the route of ``role`` opts into a diagnostic cost upper bound.

    Read from the routing file, never from the adapter. A result of such a
    route is a cost interval, which clause C6 then holds to what the
    provider billed.
    """

    return route_extra(column, role, matrix_columns.COST_BOUND_EXTRA, product) is True


# How a routing file that opts into the cost upper bound names the policy a
# caller of the route has to pass: the option of the run, with its value.
_COST_POLICY_OPTION = re.compile(r"--diagnostic-cost-policy[ =](\S+)")


def cost_policy(column: Column, role: str, product: str = COLUMN_PRODUCT) -> str | None:
    """The cost policy a route that opts into the cost upper bound names.

    Its routing file states, where it states the opt-in, which policy a run
    on the route has to pass to get a result at all. Read from that file,
    never from the adapter or the limit guard: it is what clause C5 holds the
    terminal of a guarded result to. ``None`` for a route that does not opt
    in and for a column without a routing file; a file that opts in and
    names no policy, or more than one, is an error.
    """

    if not column.routing or not cost_bounded(column, role, product):
        return None
    path = column.routing_path(product)
    named = set(_COST_POLICY_OPTION.findall(path.read_text(encoding="utf-8")))
    if len(named) != 1:
        raise MatrixSetupError(
            f"column {column.id}: the {product} routing file opts into the cost "
            f"upper bound and names {sorted(named) or 'no'} cost policy for its "
            "callers; exactly one is expected"
        )
    return named.pop()


def route_budget(column: Column, role: str, product: str = COLUMN_PRODUCT) -> float | None:
    """The budget a cell passes because the route of ``role`` asks for one.

    A route that opts into a diagnostic cost upper bound returns a result
    only for an invocation with a finite positive budget. Its cells carry the
    matrix's route budget, read from the routing file and never from the
    adapter; every other route gets none.
    """

    if cost_bounded(column, role, product):
        return matrix_columns.ROUTE_BUDGET_USD
    return None


# The closed route extras that key a difference table, and the three states
# routing can state of such an extra for a route. ``carried``: the route
# carries the extra, set to the route's own model. ``admitted_not_carried``:
# routing admits the extra on the route and the route leaves it out.
# ``not_admitted``: routing refuses the extra on the route, so the route can
# state no value of it and cannot leave one out.
SERVED_MODEL_EXTRA = "expected_served_model"
KEYING_EXTRAS: tuple[str, ...] = (SERVED_MODEL_EXTRA,)
CARRIED, ADMITTED_NOT_CARRIED, NOT_ADMITTED = "carried", "admitted_not_carried", "not_admitted"
EXTRA_STATES: tuple[str, ...] = (CARRIED, ADMITTED_NOT_CARRIED, NOT_ADMITTED)
# How routing words its refusal of a route for extras its backend does not
# take: the route, the backend and the refused fields.
_UNSUPPORTED_EXTRA = re.compile(
    r"^(?P<route>\S+) config has unsupported '[^']+' extra fields (?P<fields>\[[^\]]*\]); "
)


def _linted(document: bytes, what: str) -> BackendRouting:
    try:
        return lint_routing_config(document)
    except ValueError as exc:
        raise MatrixSetupError(f"routing refuses {what}: {exc}") from exc


def extra_state(document: bytes, role: str, name: str) -> str:
    """The state of the closed extra ``name`` on the route of ``role``, as routing states it.

    ``document`` is a routing document held in memory. It must pass routing's
    own validation unchanged. ``carried`` is read from the validated route;
    a route that carries another value than its own model is an error of the
    matrix, because the scenario this keys plays a reply under another model
    than the route's. Otherwise the same validation is asked with the extra
    added to that route and set to the route's own model: accepted is
    ``admitted_not_carried``, and refused with routing's sentence about that
    route whose refused fields are exactly the one extra is
    ``not_admitted``. Any other answer is an error of the matrix, never a
    state. No variable reference is resolved, no file is written and no
    backend is built; no kind or backend name and nothing an adapter does is
    read, and no private name of routing.
    """

    route = _linted(document, "the routing document as it stands").for_role(role)
    if name in route.extra:
        if route.extra[name] != route.model:
            raise MatrixSetupError(
                f"the route of role {role!r} carries {name}={route.extra[name]!r} and "
                f"its own model is {route.model!r}: a route pinned to another id is "
                "not played"
            )
        return CARRIED
    raw = yaml.safe_load(document.decode("utf-8")) or {}
    roles = raw.get("roles") or {}
    # The route of the role: its own entry where the document has one, the
    # default route where the role inherits all of it.
    where = role if role in roles else "default"
    entry = roles[role] if role in roles else raw.setdefault("default", {})
    entry["extra"] = {**(entry.get("extra") or {}), name: route.model}
    try:
        lint_routing_config(yaml.safe_dump(raw, sort_keys=False).encode("utf-8"))
    except ValueError as exc:
        refusal = _UNSUPPORTED_EXTRA.match(str(exc))
        if (
            refusal is None
            or refusal.group("route") != where
            or refusal.group("fields") != repr([name])
        ):
            raise MatrixSetupError(
                f"routing answers the route of role {role!r} with {name} added "
                f"neither with an acceptance nor with its refusal of exactly that "
                f"extra on that route: {exc}"
            ) from exc
        return NOT_ADMITTED
    return ADMITTED_NOT_CARRIED


def route_state(column: Column, role: str, name: str, product: str = COLUMN_PRODUCT) -> str:
    """The state of the keying extra ``name`` on the route of ``role`` of ``column``.

    Computed from the column's routing file of ``product``. A column without
    a routing file (a toy column) states its state itself, and no other
    column may: a typed state on a column that has a routing file is an
    error of the matrix.
    """

    if name not in KEYING_EXTRAS:
        raise MatrixSetupError(f"{name} is no route extra that keys a difference table")
    stated = dict(column.route_states)
    if column.routing:
        if stated:
            raise MatrixSetupError(
                f"column {column.id} has a routing file and states {stated} itself: "
                "the state of a route extra is read from the file, never typed"
            )
        state = extra_state(column.routing_path(product).read_bytes(), role, name)
        if state == ADMITTED_NOT_CARRIED:
            # The fallback of a table keyed by this state is asserted on the
            # toy only, until a ruling names a real column in it.
            raise MatrixSetupError(
                f"column {column.id}: routing admits {name} on the route of role "
                f"{role!r} and the route leaves it out. No ruling names a real "
                "column in that state, so the column is not played"
            )
        return state
    if stated.get(name) not in EXTRA_STATES:
        raise MatrixSetupError(
            f"column {column.id} has no routing file and states no state of {name} "
            f"(one of {list(EXTRA_STATES)})"
        )
    return stated[name]


def route_states(
    column: Column, role: str, keys: Sequence[str], product: str = COLUMN_PRODUCT
) -> dict[str, str]:
    """The state of every keying extra among ``keys``, for the route of ``role``.

    Nothing is read for a scenario that no route extra keys.
    """

    return {
        name: route_state(column, role, name, product)
        for name in keys
        if name in KEYING_EXTRAS
    }


def governing_values(
    profile: BackendFeatureSet,
    fields: Sequence[str],
    states: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The value of every governing key: a declared value, or the state of a keying extra.

    ``states`` holds the states of the keying extras (:func:`route_states`);
    a key of that kind without its state is an error of the matrix.
    """

    values: dict[str, str] = {}
    for name in fields:
        if name not in KEYING_EXTRAS:
            values[name] = str(getattr(profile, name))
        elif states is None or name not in states:
            raise MatrixSetupError(
                f"{name} keys a difference table and its state on the route was not read"
            )
        else:
            values[name] = states[name]
    return values


def totality_gaps(differences: Differences) -> list[str]:
    """Vocabulary values a difference table does not answer, and values it invents.

    The vocabulary of a declared field is the protocol's; that of a keying
    route extra is its three states.
    """

    gaps: list[str] = []
    for name, table in differences.items():
        allowed = (
            frozenset(EXTRA_STATES)
            if name in KEYING_EXTRAS
            else BackendFeatureSet.allowed_values(name)
        )
        gaps.extend(f"{name}={value} has no variant" for value in sorted(allowed - set(table)))
        gaps.extend(f"{name}={value} is not a value" for value in sorted(set(table) - allowed))
    return gaps


def variants(differences: Differences, values: Mapping[str, str]) -> dict[str, Variant]:
    """The variant of every governing key, from the governing values of a cell."""

    selected: dict[str, Variant] = {}
    for name, table in differences.items():
        value = values[name]
        if value not in table:
            raise MatrixSetupError(
                f"the difference table for {name} has no variant for {value!r}"
            )
        selected[name] = table[value]
    return selected


def select(
    differences: Differences,
    profile: BackendFeatureSet,
    states: Mapping[str, str] | None = None,
) -> dict[str, Variant]:
    """Select the variant of every governing key.

    From the routed profile for a declared field, and for a keying route
    extra from its state on the route of the cell's role (``states``).
    """

    return variants(differences, governing_values(profile, tuple(differences), states))


# -- the declared-profile snapshot (P04) -----------------------------------


@dataclass(frozen=True)
class SnapshotRow:
    adapter: str
    features: BackendFeatureSet


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[tuple[str, str], SnapshotRow]:
    """Load the checked-in snapshot: (kind, role) -> adapter and declared values."""

    return parse_snapshot(yaml.safe_load(path.read_text(encoding="utf-8")))


def parse_snapshot(document: Any) -> dict[tuple[str, str], SnapshotRow]:
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "columns"}
        or document["schema_version"] != SNAPSHOT_SCHEMA
        or not isinstance(document["columns"], dict)
    ):
        raise MatrixSetupError("declared_profiles.yaml has an unknown shape")
    rows: dict[tuple[str, str], SnapshotRow] = {}
    for kind, roles in document["columns"].items():
        if not isinstance(roles, dict) or not roles:
            raise MatrixSetupError(f"snapshot kind {kind!r} has no roles")
        for role, row in roles.items():
            if (
                not isinstance(row, dict)
                or set(row) != {"adapter", "features"}
                or not isinstance(row["adapter"], str)
                or not isinstance(row["features"], dict)
                or set(row["features"]) != set(FEATURE_FIELDS)
            ):
                raise MatrixSetupError(
                    f"snapshot row {kind}/{role} must hold an adapter name and "
                    f"exactly the {len(FEATURE_FIELDS)} declared values"
                )
            try:
                features = BackendFeatureSet(**row["features"])
            except ValueError as exc:
                raise MatrixSetupError(f"snapshot row {kind}/{role}: {exc}") from exc
            rows[(str(kind), str(role))] = SnapshotRow(row["adapter"], features)
    return rows


def snapshot_profiles(
    snapshot: Mapping[tuple[str, str], SnapshotRow],
) -> dict[tuple[str, str], BackendFeatureSet]:
    return {key: row.features for key, row in snapshot.items()}


def snapshot_problems(
    columns: Sequence[Column],
    snapshot: Mapping[tuple[str, str], SnapshotRow],
    *,
    lanes_by_product: Mapping[str, Sequence[Any]],
    routed_fn: Callable[[Column, str], BackendFeatureSet] = routed,
) -> list[Deviation]:
    """P04: the snapshot equals what routing computes, in both directions."""

    out: list[Deviation] = []
    base = {
        lane.key: (lane.backend, lane.transport)
        for lane in lanes_by_product[COLUMN_PRODUCT]
    }
    for product, lanes in lanes_by_product.items():
        other = {lane.key: (lane.backend, lane.transport) for lane in lanes}
        if other != base:
            differing = sorted(
                key for key in set(base) | set(other) if base.get(key) != other.get(key)
            )
            out.append(
                Deviation(
                    "static:catalog_mismatch",
                    f"the {product} and {COLUMN_PRODUCT} catalogs differ on "
                    f"{len(differing)} profile(s)",
                )
            )
    seen: set[tuple[str, str]] = set()
    for column in columns:
        key = (column.kind, column.role)
        seen.add(key)
        row = snapshot.get(key)
        if row is None:
            out.append(
                Deviation("static:snapshot_mismatch", f"column {column.id} has no row")
            )
            continue
        if row.features == CONSERVATIVE_BACKEND_FEATURES:
            out.append(
                Deviation(
                    "static:conservative_profile",
                    f"snapshot row {column.id} is the conservative fallback",
                )
            )
        if column.adapter != row.adapter:
            out.append(
                Deviation(
                    "static:snapshot_mismatch",
                    f"column {column.id}: adapter {column.adapter!r}, snapshot "
                    f"{row.adapter!r}",
                )
            )
        for product, _path in column.routing:
            profile = routed_fn(column, product)
            if profile == CONSERVATIVE_BACKEND_FEATURES:
                out.append(
                    Deviation(
                        "static:conservative_profile",
                        f"column {column.id} ({product}) resolves to the "
                        "conservative fallback: its route lost its profile",
                    )
                )
                continue
            for name in FEATURE_FIELDS:
                got, pinned = getattr(profile, name), getattr(row.features, name)
                if got != pinned:
                    out.append(
                        Deviation(
                            "static:snapshot_mismatch",
                            f"column {column.id} ({product}): {name} is {got!r}, "
                            f"the snapshot pins {pinned!r}",
                        )
                    )
    for kind, role in sorted(set(snapshot) - seen):
        out.append(
            Deviation("static:snapshot_mismatch", f"row {kind}/{role} has no column")
        )
    return out


# -- profile binding (P02) -------------------------------------------------


REFUSED_LIMITS = "refused"
RESOLVED_LIMITS = "resolved"
UNBOUND_LIMITS = "unbound"


def implied_limits(
    profile: BackendFeatureSet, request: InvocationLimitRequest, *, route: str
) -> tuple[str, Any]:
    """What ``profile`` implies for ``request``: a resolution, or the refusal."""

    try:
        resolved = resolve_invocation_limits(profile, request, route=route)
    except InvocationLimitAdmissionError:
        return (REFUSED_LIMITS, InvocationLimitAdmissionError.__name__)
    return (RESOLVED_LIMITS, resolved)


def profile_binding_problems(
    backend: Any,
    routed_profile: BackendFeatureSet,
    *,
    bound_limits: tuple[str, Any],
    request: InvocationLimitRequest,
) -> list[Deviation]:
    """P02: the backend is bound to the profile routing computes for its column.

    ``bound_limits`` is what one guarded call with ``request`` left on the
    backend: ``(RESOLVED_LIMITS, <the resolution the limit guard retained>)``,
    ``(REFUSED_LIMITS, <admission error type>)`` or ``(UNBOUND_LIMITS, None)``.
    The trace's profile digest is clause C7's business on every cell.
    """

    instance = getattr(backend, "features", None)
    if instance != routed_profile:
        differing = [
            name
            for name in FEATURE_FIELDS
            if getattr(instance, name, None) != getattr(routed_profile, name)
        ]
        return [
            Deviation(
                "static:profile_binding",
                f"backend.features differs from the routed profile on {differing}",
            )
        ]
    status, bound = bound_limits
    if status == UNBOUND_LIMITS:
        return [
            Deviation(
                "static:profile_binding",
                "a guarded call left no limit resolution on the backend: no "
                "limit guard is bound to it",
            )
        ]
    route = bound.route if status == RESOLVED_LIMITS else "matrix"
    implied = implied_limits(routed_profile, request, route=route)
    if (status, bound) == implied:
        return []
    if status == implied[0] == RESOLVED_LIMITS:
        differing = sorted(
            item.name
            for item in dataclasses.fields(bound)
            if getattr(bound, item.name) != getattr(implied[1], item.name)
        )
        detail = f"differs on {differing}"
    else:
        detail = f"is {status}, the profile implies {implied[0]}"
    return [
        Deviation(
            "static:profile_binding",
            "the limit resolution bound to the backend is not what the routed "
            f"profile implies: it {detail}",
        )
    ]
