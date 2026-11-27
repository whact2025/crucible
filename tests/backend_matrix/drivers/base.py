# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The driver interface, the driver registry and the matrix's setup error.

A driver replaces the network or the vendor executable of one backend kind,
never the adapter. It declares what it bypasses and every private product name
it patches. Everything else (ToolSet, invocation, observation, clauses, ledger)
belongs to the shared runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ContextManager,
    Literal,
    Mapping,
    Protocol,
    runtime_checkable,
)

from assay import BackendFeatureSet
from crucible.backends.evidence import bind_result_evidence_admission
from crucible.backends.limits import InvocationLimitPolicy, bind_invocation_limit_policy

if TYPE_CHECKING:
    from ..columns import Column
    from ..provider import LoopbackProvider
    from ..script import ProviderScript

Lens = Literal["raw", "guarded"]
LENSES: tuple[str, ...] = ("raw", "guarded")
# The limit policy of the guarded lens: what routing binds on every
# open-weight route, with a deadline no scripted cell reaches.
GUARD_POLICY = InvocationLimitPolicy(mode="diagnostic", deadline_seconds=60.0)


class MatrixSetupError(RuntimeError):
    """The matrix itself is mis-built or cannot run here. Never a skip."""


@runtime_checkable
class Bound(Protocol):
    """One built backend with the provider side it talks to."""

    backend: Any
    provider: LoopbackProvider
    tool_path: Literal["function_call", "mcp_bridge", "control_channel"]
    # Where this backend writes logs and debug dumps, and every secret it
    # holds beside the canary credential (a bridge or session token): C9
    # reads the first for the second.
    log_dirs: tuple[Path, ...]
    secrets: tuple[str, ...]

    def cancel_when(self, marker: str) -> None: ...

    def children(self) -> tuple[int, ...]: ...


@runtime_checkable
class Driver(Protocol):
    """What a backend kind needs to be played by a provider script."""

    kind: str
    dialect: str
    fidelity: Literal["wire", "worker_protocol"]
    bypassed_layers: tuple[str, ...]
    private_seams: tuple[str, ...]

    @property
    def inexpressible(self) -> Mapping[str, str]:
        """What the inventory pins not expressible on the kind, by scenario and step.

        The script steps the wire of the kind cannot carry and, where the
        inventory has such a row, a state of the route that no backend
        routing built for the kind can be in.
        """

    @property
    def envelopes(self) -> tuple[str, ...]:
        """Every envelope the driver's dialect renders from, by name."""

    def answers(self, script: ProviderScript) -> tuple[str, ...]:
        """The envelopes the driver's dialect answers ``script`` from, by name.

        Read without a backend and without an exchange: static cell H03 asks
        it for the grade of the replies a cell is answered from.
        """

    def preflight(self) -> None: ...

    def overlay(self, column: Column, env: Mapping[str, str]) -> Mapping[str, str]: ...

    def open(
        self,
        column: Column,
        *,
        capability: str,
        lens: Lens,
        script: ProviderScript,
        tmp_path: Path,
        without_credential: bool = False,
    ) -> ContextManager[Bound]:
        """Build the backend of ``column`` for one play.

        ``without_credential`` asks for the route without its credential: the
        backend is built by routing from the shipped route, and the place the
        route takes its credential from holds none at the last moment at
        which the route reads it. Where that moment is, and what the place
        is, is the driver's to say; a driver that cannot build the state
        raises ``MatrixSetupError``, which is no skip.
        """


@runtime_checkable
class KeeperDriver(Protocol):
    """A driver of a kind on whose wire a tool beside the adapter keeps the conversations.

    The peer of a cell stands where that tool is and holds no conversation
    of an earlier call. A handle that an invocation asks to continue is one
    the tool refuses, and what it says then is rendered by the dialect of
    the kind from an envelope. A cell whose route answers such a request is
    answered from that envelope, whatever its script holds.
    """

    def refuses(self) -> tuple[str, ...]:
        """The envelopes the dialect renders the refusal of a handle from, by name.

        Read without a backend and without an exchange, as
        :meth:`Driver.answers` is: static cell H03 asks it for the grade of
        what a cell that asks to continue a conversation is answered from.
        """


_REGISTRY: dict[str, Driver] = {}


def register(driver: Driver) -> Driver:
    """Register the one driver of a kind."""

    if driver.kind in _REGISTRY:
        raise MatrixSetupError(f"driver kind {driver.kind!r} is registered twice")
    _REGISTRY[driver.kind] = driver
    return driver


def registered_kinds() -> frozenset[str]:
    return frozenset(_REGISTRY)


def get(kind: str) -> Driver:
    try:
        return _REGISTRY[kind]
    except KeyError as exc:
        raise MatrixSetupError(
            f"no driver is registered for kind {kind!r}; registered: "
            f"{sorted(_REGISTRY)}"
        ) from exc


def refuse_the_scripted_credential(column: Column, provider: LoopbackProvider) -> None:
    """A backend that was built without a credential sent the scripted one.

    For a kind whose route reads its credential when the backend is built:
    the state of a route without its credential is then shown, not assumed.
    A request of such a play that carries the scripted credential says that
    the backend was built with it, so the play is no play of that state. The
    codes of a cell cannot tell a placeholder from the scripted credential;
    this can, and it is the matrix's own failure, never a deviation.
    """

    carried = sum(request.scripted_credential for request in provider.freeze().requests)
    if carried:
        raise MatrixSetupError(
            f"column {column.id}: {carried} request(s) of a play without the "
            "credential of the route carried the scripted credential: the backend "
            "was built with it, so the state the scenario asks for was not built"
        )


def at_the_call(bound: Bound) -> None:
    """Let a driver do what it left for the last moment before ``invoke``.

    A route that reads its credential at the call, and whose routing file
    names it as a variable, is without its credential only from that moment
    on: routing reads the same variable whenever it loads the file, and the
    runner asks routing for the route once more after the backend is built.
    A bound backend with such a step states it as ``before_call``; one
    without has no such attribute, and nothing is done.
    """

    step = getattr(bound, "before_call", None)
    if step is not None:
        step()


def bind_as_routing(
    backend: Any,
    profile: BackendFeatureSet,
    *,
    lens: Lens,
    route: str,
    admission: Literal["enforce", "diagnostic"],
) -> None:
    """Bind the limit guard and the evidence gate as routing binds them.

    For an adapter that routing does not construct. These are the two public
    binders ``resolve_for_role`` calls, in its order: the limit policy of the
    guarded lens first, then result-evidence admission.
    """

    if lens == "guarded":
        bind_invocation_limit_policy(backend, profile, GUARD_POLICY, route=route)
    bind_result_evidence_admission(backend, mode=admission)
