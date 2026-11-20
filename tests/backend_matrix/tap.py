# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The origin redirect: a route's production origin is answered on loopback.

Standard library and ``httpx`` only, and free of package-relative imports, so
that a locked worker interpreter can load this file by path.

While the redirect is installed, every request that leaves through
``httpx.HTTPTransport`` or ``httpx.AsyncHTTPTransport``

- for the route's production origin is sent to the loopback target instead,
  with its path, query, headers and body unchanged. The ``Host`` header still
  names the production host, which is how the provider side can tell that the
  client addressed production;
- for a loopback address passes through untouched (a tool bridge, the
  scripted provider itself);
- for anything else raises :class:`TapViolation` and is recorded.

A vendor SDK wraps whatever its transport raises into its own connection
error, which an adapter may turn into a typed result. So the record decides,
as it does for the network guard: whoever installed the redirect must fail
when the record holds a violation.

The adapter, its SDK and the SDK's own transport are not replaced: the request
is built, serialised and parsed by them and crosses a real socket. Only DNS,
TLS and the provider are bypassed.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urlsplit

import httpx

_DEFAULT_PORTS = {"http": 80, "https": 443}


class TapViolation(Exception):
    """A request left for an origin that is neither the route's nor loopback."""


@dataclass
class TapRecord:
    """What the redirect saw while it was installed."""

    redirected: int = 0
    passed: int = 0
    violations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _State:
    origin: tuple[str, str, int]
    target: tuple[str, str, int]
    record: TapRecord
    sync_handler: Any
    async_handler: Any


_lock = threading.Lock()
_state: _State | None = None


def _origin(url: str) -> tuple[str, str, int]:
    """``(scheme, host, port)`` of ``url``, with the scheme's default port."""

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        raise ValueError(f"not an http(s) origin: {url!r}")
    return scheme, parts.hostname.lower(), parts.port or _DEFAULT_PORTS[scheme]


def _is_loopback(host: str) -> bool:
    name = host.strip().rstrip(".").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name.strip("[]")).is_loopback
    except ValueError:
        return False


def _routed(request: httpx.Request) -> httpx.Request:
    """The request to send: redirected, unchanged, or refused."""

    state = _state
    if state is None:  # removed while a request was in flight
        return request
    url = request.url
    scheme = url.scheme.lower()
    asked = (scheme, url.host.lower(), url.port or _DEFAULT_PORTS.get(scheme, 0))
    if asked == state.origin:
        with _lock:
            state.record.redirected += 1
        scheme, host, port = state.target
        return httpx.Request(
            request.method,
            url.copy_with(scheme=scheme, host=host, port=port),
            # The headers are the client's own, Host included.
            headers=request.headers,
            stream=request.stream,
            extensions=request.extensions,
        )
    if _is_loopback(url.host):
        with _lock:
            state.record.passed += 1
        return request
    entry = f"{request.method} {scheme}://{url.host}:{asked[2]}"
    with _lock:
        state.record.violations.append(entry)
    raise TapViolation(
        f"origin redirect: {entry} is neither the route's production origin "
        f"({state.origin[0]}://{state.origin[1]}:{state.origin[2]}) nor loopback"
    )


def install(origin: str, target: str, *, marker: str | os.PathLike[str] | None = None) -> None:
    """Install the redirect from ``origin`` to the loopback ``target``.

    ``marker`` names a file that is written once the redirect is in place, so
    a parent process can tell that a worker ran behind it. Installing twice is
    an error, not a no-op.
    """

    global _state
    target_origin = _origin(target)
    if not _is_loopback(target_origin[1]):
        raise ValueError(f"the redirect target must be loopback: {target!r}")
    with _lock:
        if _state is not None:
            raise RuntimeError("origin redirect is already installed")
        sync_handler = httpx.HTTPTransport.handle_request
        async_handler = httpx.AsyncHTTPTransport.handle_async_request

        def handle_request(self: Any, request: httpx.Request) -> httpx.Response:
            return sync_handler(self, _routed(request))

        async def handle_async_request(self: Any, request: httpx.Request) -> httpx.Response:
            return await async_handler(self, _routed(request))

        httpx.HTTPTransport.handle_request = handle_request  # type: ignore[method-assign]
        httpx.AsyncHTTPTransport.handle_async_request = (  # type: ignore[method-assign]
            handle_async_request
        )
        _state = _State(_origin(origin), target_origin, TapRecord(), sync_handler, async_handler)
    if marker is not None:
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(f"origin redirect installed in process {os.getpid()}\n")


def remove() -> TapRecord:
    """Remove the redirect, restore both transports and return the record."""

    global _state
    with _lock:
        if _state is None:
            raise RuntimeError("origin redirect is not installed")
        state = _state
        httpx.HTTPTransport.handle_request = state.sync_handler  # type: ignore[method-assign]
        httpx.AsyncHTTPTransport.handle_async_request = (  # type: ignore[method-assign]
            state.async_handler
        )
        _state = None
    return state.record


def installed() -> bool:
    with _lock:
        return _state is not None


@contextlib.contextmanager
def redirect(
    origin: str, target: str, *, marker: str | os.PathLike[str] | None = None
) -> Iterator[TapRecord]:
    """Install the redirect for a block; the yielded record stays readable."""

    install(origin, target, marker=marker)
    assert _state is not None
    record = _state.record
    try:
        yield record
    finally:
        remove()
