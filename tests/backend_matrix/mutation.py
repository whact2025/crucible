# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""What a cell sees of the processes its writes run in.

A backend that does not run a write tool in the process of the workflow
builds an object there that owns the process the write runs in: a *host*.
The host is built with the deadline of the invocation, started, handed each
write, and stopped. For as long as a cell plays, the watch wraps the
constructor, ``start`` and ``stop`` of every host class the matrix knows, so
that it holds each host that was built, with the deadline it was given, the
pid it showed once started and the processes that were below it when it was
stopped. When ``invoke`` has returned the cell reads what each host recorded,
whether its process is still there, and what else it left: the processes
that were below it at its stop and the processes whose parent, process group
or session it is. The watch changes nothing of what a host does, and it is
not told anything by an adapter: an adapter that builds no host leaves it
empty.

What the sweep does not see: a process that had left a host's tree, its
process group and its session before the host was stopped, and a host that
is ended otherwise than through ``stop`` shows only what is still in its
group or session. The children of the test process are the business of the
settlement clause, on every cell.

Two classes are watched. The shared mutation host of the product is a
private name of the product (:data:`PRIVATE_SEAMS`): it is wrapped in the
test process for the time of a cell and never replaced; the process it
starts is not touched. The other is the host of the toy adapter, which has
the same public names (``invocation_deadline`` of the constructor, ``start``,
``stop``, ``pid`` and ``started_write_names``).

That a write ran in the test process is not seen here. The fixture's
recorder sees it (:mod:`.toolsets`): it shadows the dispatch of the
innermost ToolSet object in this process.
"""

from __future__ import annotations

import contextlib
import functools
import os
import threading
import time
from typing import Any, Callable, Iterator

import psutil
import pytest

from .contract import WriteHost
from .drivers.base import MatrixSetupError

# The private names of the product the watch wraps, for the time of a cell.
PRODUCT_HOST = "crucible.tools._isolated_mutation_host.IsolatedMutationHost"
PRIVATE_SEAMS: tuple[str, ...] = (PRODUCT_HOST,)
# How long a host, and what it left, may take to be gone once ``invoke`` has
# returned.
GRACE_SECONDS = 1.0
_DEADLINE = "invocation_deadline"


def host_classes() -> tuple[Any, ...]:
    """Every class whose objects host writes: the product's, and the toy's."""

    from crucible.tools._isolated_mutation_host import (  # noqa: PLC0415
        IsolatedMutationHost,
    )

    from .drivers import toy  # noqa: PLC0415 - the toy driver imports the runner's modules

    return (IsolatedMutationHost, toy.WriteHost)


def _exists(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _running(process: psutil.Process) -> bool:
    """Whether the very process is still there (its pid may have a new owner)."""

    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _below(pid: int) -> list[psutil.Process]:
    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def _bound_to(pids: frozenset[int]) -> dict[int, set[int]]:
    """Per pid of ``pids``: the live processes whose parent, group or session it is."""

    found: dict[int, set[int]] = {pid: set() for pid in pids}
    for process in psutil.process_iter():
        other = process.pid
        if other in pids:
            continue  # a host itself is reported as a host
        try:
            if process.status() == psutil.STATUS_ZOMBIE:
                continue
            bound = {process.ppid(), os.getpgid(other), os.getsid(other)}
        except (psutil.Error, OSError):
            continue  # gone meanwhile, or not ours to read
        for pid in bound & pids:
            found[pid].add(other)
    return found


class Watch:
    """The hosts that were built while a cell played."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hosts: list[tuple[Any, float]] = []
        self._pids: dict[int, int | None] = {}
        self._below: dict[int, dict[int, psutil.Process]] = {}

    def built(self, host: Any, deadline: Any) -> None:
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise MatrixSetupError(
                f"a {type(host).__name__} was built without the keyword "
                f"{_DEADLINE}: the watch cannot say which deadline it is bound to"
            )
        with self._lock:
            self._hosts.append((host, float(deadline)))

    def started(self, host: Any) -> None:
        with self._lock:
            self._pids[id(host)] = host.pid

    def stopping(self, host: Any) -> None:
        """Hold what is below the host now: its stop is about to end it."""

        with self._lock:
            pid = self._pids.get(id(host))
        if pid is None:
            return
        below = {process.pid: process for process in _below(pid)}
        with self._lock:
            self._below.setdefault(id(host), {}).update(below)

    def _left_behind(self, hosts: list[Any], pids: list[int | None]) -> list[tuple[int, ...]]:
        started = frozenset(pid for pid in pids if pid is not None)
        bound = _bound_to(started) if started else {}
        with self._lock:
            below = [dict(self._below.get(id(host), {})) for host in hosts]
        return [
            tuple(
                sorted(
                    {other for other, process in held.items() if _running(process)}
                    | (bound.get(pid, set()) if pid is not None else set())
                )
            )
            for held, pid in zip(below, pids, strict=True)
        ]

    def at_return(self) -> tuple[WriteHost, ...]:
        """What every host shows now that ``invoke`` has returned.

        A process that is on its way out gets :data:`GRACE_SECONDS` to be
        gone; one that is still there after that is reported: a host as
        alive, another process as left behind by its host.
        """

        with self._lock:
            hosts = [host for host, _deadline in self._hosts]
            deadlines = [deadline for _host, deadline in self._hosts]
            pids = [self._pids.get(id(host)) for host in hosts]
        until = time.monotonic() + GRACE_SECONDS
        while True:
            alive = [pid is not None and _exists(pid) for pid in pids]
            left = self._left_behind(hosts, pids)
            if time.monotonic() >= until or not (any(alive) or any(left)):
                break
            time.sleep(0.02)
        return tuple(
            WriteHost(
                pid=pid,
                deadline=deadline,
                started_writes=tuple(str(name) for name in host.started_write_names),
                alive_at_return=still,
                left_behind=behind,
            )
            for host, deadline, pid, still, behind in zip(
                hosts, deadlines, pids, alive, left, strict=True
            )
        )


def _building(original: Callable[..., None], seen: Watch) -> Callable[..., None]:
    @functools.wraps(original)
    def built(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        seen.built(self, kwargs.get(_DEADLINE))

    return built


def _starting(original: Callable[..., Any], seen: Watch) -> Callable[..., Any]:
    @functools.wraps(original)
    def started(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        seen.started(self)
        return result

    return started


def _stopping(original: Callable[..., Any], seen: Watch) -> Callable[..., Any]:
    @functools.wraps(original)
    def stopped(self: Any, *args: Any, **kwargs: Any) -> Any:
        seen.stopping(self)
        return original(self, *args, **kwargs)

    return stopped


@contextlib.contextmanager
def watch() -> Iterator[Watch]:
    """Hold every host that is built in this process until the block ends."""

    seen = Watch()
    with pytest.MonkeyPatch.context() as wrapped:
        for host_class in host_classes():
            wrapped.setattr(host_class, "__init__", _building(host_class.__init__, seen))
            wrapped.setattr(host_class, "start", _starting(host_class.start, seen))
            wrapped.setattr(host_class, "stop", _stopping(host_class.stop, seen))
        yield seen
