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
        """Script steps the kind's wire cannot carry: the inventory's rows."""

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
    ) -> ContextManager[Bound]: ...


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
