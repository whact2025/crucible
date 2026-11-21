# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Network guard: only loopback targets may be connected to or resolved.

Standard library only and free of package-relative imports, so that a locked
worker interpreter can load this file by path under ``-B -I``.

While the guard is installed, a connect, a datagram send with an explicit
address or a name resolution whose target is not loopback (``127.0.0.0/8``,
``::1``, ``localhost`` or a Unix socket) raises :class:`NetworkGuardViolation`
and is recorded. The exception is deliberately not an ``OSError``: an adapter
that converts transport errors into typed results must not be able to turn a
blocked call into a result a failure cell would accept. Because any broad
``except`` can still swallow it, the record is what decides: the fixture that
installs the guard fails the test when the record is not empty at teardown.

A process that cannot hand its record to a fixture (a worker that a test
starts) passes ``install`` a callback instead: it is called with every
violation as it is recorded, before the exception is raised, so the record
survives a caller that swallows the exception and a process that is killed.

The guard sees Python sockets only. A client that opens sockets below the
``socket`` module is not stopped by it.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from typing import Any, Callable, Iterator

_SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")
_MODULE_FUNCTIONS = (
    "create_connection",
    "getaddrinfo",
    "getnameinfo",
    "gethostbyname",
    "gethostbyname_ex",
    "gethostbyaddr",
)
_ABSENT = object()

_lock = threading.Lock()
_record: list[str] = []
_saved: dict[str, Any] | None = None
_on_violation: Callable[[str], None] | None = None


class NetworkGuardViolation(Exception):
    """A test tried to reach a target that is not loopback."""


def is_loopback_host(host: Any) -> bool:
    """Return whether ``host`` names this machine's loopback interface."""

    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        try:
            host = bytes(host).decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str):
        return False
    name = host.strip().rstrip(".").lower()
    if name in {"", "localhost"}:
        return True
    try:
        address = ipaddress.ip_address(name.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(address.is_loopback or (mapped is not None and mapped.is_loopback))


def _refuse(operation: str, target: Any) -> None:
    entry = f"{operation} -> {target!r}"
    with _lock:
        _record.append(entry)
        report = _on_violation
    if report is not None:
        report(entry)
    raise NetworkGuardViolation(
        f"network guard: {entry} is not a loopback target; the matrix and "
        "the tests it guards run without network access"
    )


def _check_address(family: int, address: Any, operation: str) -> None:
    if family == getattr(socket, "AF_UNIX", _ABSENT):
        return
    if family in (socket.AF_INET, socket.AF_INET6):
        host = address[0] if isinstance(address, tuple) and address else address
        if is_loopback_host(host):
            return
    _refuse(operation, address)


def _guard_method(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(self: socket.socket, *args: Any, **kwargs: Any) -> Any:
        if name in ("connect", "connect_ex"):
            address = args[0] if args else kwargs.get("address")
            _check_address(self.family, address, name)
        elif name == "sendto":
            # sendto(data, address) or sendto(data, flags, address)
            address = args[-1] if len(args) >= 2 else kwargs.get("address")
            _check_address(self.family, address, name)
        else:
            # sendmsg(buffers[, ancdata[, flags[, address]]])
            address = args[3] if len(args) >= 4 else kwargs.get("address")
            if address is not None:
                _check_address(self.family, address, name)
        return original(self, *args, **kwargs)

    guarded.__name__ = name
    return guarded


_TARGET_KEYWORD = {
    "create_connection": "address",
    "getaddrinfo": "host",
    "getnameinfo": "sockaddr",
}


def _guard_function(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(*args: Any, **kwargs: Any) -> Any:
        target = args[0] if args else kwargs.get(_TARGET_KEYWORD.get(name, ""))
        host = target[0] if isinstance(target, tuple) and target else target
        if not is_loopback_host(host):
            _refuse(name, target)
        return original(*args, **kwargs)

    guarded.__name__ = name
    return guarded


def install(on_violation: Callable[[str], None] | None = None) -> None:
    """Install the guard. Installing twice is an error, not a no-op.

    ``on_violation`` is called with every violation as it is recorded.
    """

    global _saved, _on_violation
    with _lock:
        if _saved is not None:
            raise RuntimeError("network guard is already installed")
        saved: dict[str, Any] = {}
        for name in _SOCKET_METHODS:
            saved["socket." + name] = socket.socket.__dict__.get(name, _ABSENT)
            setattr(
                socket.socket,
                name,
                _guard_method(name, getattr(socket.socket, name)),
            )
        for name in _MODULE_FUNCTIONS:
            original = getattr(socket, name)
            saved[name] = original
            setattr(socket, name, _guard_function(name, original))
        _saved = saved
        _on_violation = on_violation


def remove() -> None:
    """Remove the guard and restore exactly what :func:`install` replaced."""

    global _saved, _on_violation
    with _lock:
        if _saved is None:
            raise RuntimeError("network guard is not installed")
        for name in _SOCKET_METHODS:
            previous = _saved["socket." + name]
            if previous is _ABSENT:
                delattr(socket.socket, name)
            else:
                setattr(socket.socket, name, previous)
        for name in _MODULE_FUNCTIONS:
            setattr(socket, name, _saved[name])
        _saved = None
        _on_violation = None


def installed() -> bool:
    with _lock:
        return _saved is not None


def take_violations() -> tuple[str, ...]:
    """Return the recorded violations and clear the record."""

    with _lock:
        taken = tuple(_record)
        _record.clear()
    return taken


def watch(fail: Callable[[str], None]) -> Iterator[None]:
    """Fixture body: the guard is on for one test, and its record must be empty.

    ``fail`` is called with the reason when the test reached beyond loopback,
    whether or not the code under test swallowed the exception.
    """

    take_violations()
    install()
    try:
        yield
    finally:
        remove()
        violations = take_violations()
        if violations:
            fail(
                "the test reached for the network (blocked by the matrix guard): "
                + "; ".join(violations)
            )
