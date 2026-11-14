# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Adversarial ownership tests for workflow backend lifecycle cleanup."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from crucible.workflows import _backends


class _ReadToolSet:
    read_only_names = {"read_source"}
    write_names: set[str] = set()


class _WriteToolSet:
    read_only_names = {"read_source"}
    write_names = {"apply_patch"}


class _CloseBackend:
    def __init__(self, name: str, *, close_failures: int = 0) -> None:
        self.name = name
        self.close_failures = close_failures
        self.close_calls = 0
        self.closed = False

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls <= self.close_failures:
            raise RuntimeError(f"close failed for {self.name}")
        self.closed = True


def test_exclusive_text_reviewers_have_independent_cleanup_without_tool_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = []

    def resolve(_role, **kwargs):
        backend = _CloseBackend(f"reviewer-{len(resolved)}")
        resolved.append((backend, kwargs))
        return backend

    monkeypatch.setattr(_backends, "resolve_for_role", resolve)
    pool = _backends._BackendPool(object(), log_dir=tmp_path, cwd=str(tmp_path))
    leases = [
        pool.new_exclusive(
            "plan_judge", toolset=None, log_dir=tmp_path / str(index),
            cwd=str(tmp_path), capability="text",
        )
        for index in range(2)
    ]

    assert leases[0] is not leases[1]
    for backend, options in resolved:
        assert options["capability"] == "text"
        assert options["native_shell"] is False
        assert options["allow_trusted_project_instructions"] is False
        backend.close()
    pool.close()
    assert all(backend.closed and backend.close_calls == 1 for backend, _ in resolved)


@pytest.mark.parametrize("capability,tools", [
    ("text", _ReadToolSet()),
    ("text", _WriteToolSet()),
    ("workspace_write", None),
])
def test_exclusive_analytical_leases_reject_capability_expansion_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capability, tools,
) -> None:
    def unexpected(*_args, **_kwargs):
        pytest.fail("inadmissible lease must not resolve a provider")

    monkeypatch.setattr(_backends, "resolve_for_role", unexpected)
    pool = _backends._BackendPool(object(), log_dir=tmp_path, cwd=str(tmp_path))
    with pytest.raises(ValueError):
        pool.new_exclusive(
            "plan_judge", toolset=tools, log_dir=tmp_path,
            cwd=str(tmp_path), capability=capability,
        )


def test_pool_close_aggregates_failures_retains_only_unsettled_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backends = {
        "role-a": _CloseBackend("a", close_failures=1),
        "role-b": _CloseBackend("b", close_failures=1),
        "role-c": _CloseBackend("c"),
    }
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda role, **_kwargs: backends[role],
    )
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )
    for role in backends:
        pool.get(role)

    with pytest.raises(_backends.BackendCleanupError) as raised:
        pool.close()

    assert len(raised.value.failures) == 2
    assert backends["role-c"].closed
    assert backends["role-c"].close_calls == 1
    assert {key[0] for key in pool._cache} == {"role-a", "role-b"}  # noqa: SLF001

    pool.close()

    assert pool._cache == {}  # noqa: SLF001
    assert all(backend.closed for backend in backends.values())
    assert backends["role-a"].close_calls == 2
    assert backends["role-b"].close_calls == 2
    assert backends["role-c"].close_calls == 1


def test_pool_defers_sigint_until_backend_retry_proves_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation replays with a typed proof after exact close succeeds."""

    class _InterruptingCloseBackend(_CloseBackend):
        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise KeyboardInterrupt
            self.closed = True

    backend = _InterruptingCloseBackend("interrupt")
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda _role, **_kwargs: backend,
    )
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )
    pool.get("implementer")

    with pytest.raises(_backends.BackendSettlementInterrupted) as caught:
        pool.close()

    assert caught.value.owner is pool
    assert isinstance(caught.value.interrupt, KeyboardInterrupt)
    assert backend.close_calls == 2
    assert backend.closed
    assert pool._cache == {}  # noqa: SLF001
    assert pool._cleanup_errors == {}  # noqa: SLF001


def test_pool_sigint_retry_does_not_reclose_prior_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later interrupted owner cannot erase earlier close evidence."""

    class _CloseExactlyOnce(_CloseBackend):
        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls != 1:
                raise AssertionError("settled backend was closed twice")
            self.closed = True

    class _InterruptOnce(_CloseBackend):
        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise KeyboardInterrupt
            self.closed = True

    backends = {
        "role-a": _CloseExactlyOnce("a"),
        "role-b": _InterruptOnce("b"),
    }
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda role, **_kwargs: backends[role],
    )
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )
    for role in backends:
        pool.get(role)

    with pytest.raises(_backends.BackendSettlementInterrupted):
        pool.close()

    assert backends["role-a"].close_calls == 1
    assert backends["role-b"].close_calls == 2
    assert pool._cache == {}  # noqa: SLF001


def test_repeated_pool_sigint_returns_retryable_cleanup_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent cancellation is bounded without releasing exact ownership."""

    original_interrupt = KeyboardInterrupt()

    class _InterruptTwice(_CloseBackend):
        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls <= 2:
                raise original_interrupt
            self.closed = True

    backend = _InterruptTwice("bounded-interrupt")
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda _role, **_kwargs: backend,
    )
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )
    pool.get("implementer")

    with pytest.raises(_backends.BackendCleanupError) as pending:
        pool.close()

    assert backend.close_calls == 2
    assert pool._cache  # noqa: SLF001
    with pytest.raises(_backends.BackendCleanupError):
        pool.get("judge")
    assert backend.close_calls == 2
    with pytest.raises(_backends.BackendSettlementInterrupted) as replayed:
        pending.value.retry_cleanup()

    assert replayed.value.interrupt is original_interrupt
    assert backend.close_calls == 3
    assert backend.closed
    assert pool._cache == {}  # noqa: SLF001


def test_pool_attach_failure_retains_failed_cleanup_and_blocks_new_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[_CloseBackend] = []

    class _AttachFailureBackend(_CloseBackend):
        def attach_toolset(self, _toolset, *, context_budget=None) -> None:
            raise ValueError(f"attach failed at budget {context_budget}")

    def _resolve(_role, **_kwargs):
        backend = _AttachFailureBackend(
            "attach-failure",
            close_failures=1,
        )
        constructed.append(backend)
        return backend

    monkeypatch.setattr(_backends, "resolve_for_role", _resolve)
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )

    with pytest.raises(_backends.BackendCleanupError) as raised:
        pool.get(
            "planner",
            toolset=_ReadToolSet(),
            context_budget=123,
        )

    assert len(raised.value.failures) == 2
    assert isinstance(raised.value.failures[0][1], ValueError)
    assert isinstance(raised.value.failures[1][1], RuntimeError)
    assert len(constructed) == 1
    assert constructed[0].close_calls == 1
    assert pool._cache == {}  # noqa: SLF001
    assert len(pool._pending_cleanup) == 1  # noqa: SLF001

    with pytest.raises(_backends.BackendCleanupError):
        pool.get("other")
    assert len(constructed) == 1

    pool.close()

    assert constructed[0].closed
    assert constructed[0].close_calls == 2
    assert pool._pending_cleanup == {}  # noqa: SLF001
    assert pool._cleanup_errors == {}  # noqa: SLF001


def test_setup_cleanup_sigint_is_retained_and_replayed_after_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attach failure cannot consume a cancellation during exact cleanup."""

    class _AttachAndInterruptBackend(_CloseBackend):
        def attach_toolset(self, _toolset, *, context_budget=None) -> None:
            del context_budget
            raise ValueError("attach failed")

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise KeyboardInterrupt
            self.closed = True

    backend = _AttachAndInterruptBackend("setup-interrupt")
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda _role, **_kwargs: backend,
    )
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )

    with pytest.raises(KeyboardInterrupt) as interrupted:
        pool.get("planner", toolset=_ReadToolSet())

    assert backend.close_calls == 1
    assert len(pool._pending_cleanup) == 1  # noqa: SLF001
    with pytest.raises(_backends.BackendSettlementInterrupted) as settled:
        pool.close()

    assert settled.value.owner is pool
    assert settled.value.interrupt is interrupted.value
    assert backend.close_calls == 2
    assert backend.closed
    assert pool._pending_cleanup == {}  # noqa: SLF001
    assert pool._cleanup_errors == {}  # noqa: SLF001


def test_contextmanager_cleanup_error_is_its_own_retry_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ContextBackend(_CloseBackend):
        close_calls = 0
        closed = False

        def __init__(self) -> None:
            super().__init__("context", close_failures=1)

        def close(self) -> None:
            type(self).close_calls += 1
            if type(self).close_calls == 1:
                raise RuntimeError("context close failed")
            type(self).closed = True

    monkeypatch.setattr(_backends, "load_routing", lambda **_kwargs: object())
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda _role, **_kwargs: _ContextBackend(),
    )
    args = argparse.Namespace(backend_config=None, spec=tmp_path)

    def _use_contextmanager() -> None:
        with _backends.backend_pool(args, None, cwd=str(tmp_path)) as pool:
            pool.get("planner")

    with pytest.raises(_backends.BackendCleanupError) as caught:
        _use_contextmanager()

    cleanup_error = caught.value
    assert _ContextBackend.close_calls == 1
    assert not _ContextBackend.closed
    cleanup_error.retry_cleanup()
    assert _ContextBackend.close_calls == 2
    assert _ContextBackend.closed


@pytest.mark.parametrize("close_value", [None, "not-callable"])
def test_workspace_writer_without_callable_close_is_refused_before_attach(
    close_value: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnsettleableWriter:
        name = "unsettleable"

        def __init__(self) -> None:
            self.close = close_value
            self.attach_calls = 0

        def attach_toolset(self, _toolset, *, context_budget=None) -> None:
            del context_budget
            self.attach_calls += 1

    backend = _UnsettleableWriter()
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda _role, **_kwargs: backend,
    )
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )

    with pytest.raises(
        _backends.BackendLifecycleError,
        match="without callable close",
    ):
        pool.get("implementer", toolset=_WriteToolSet())

    assert backend.attach_calls == 0
    assert pool._cache == {}  # noqa: SLF001
    retained = pool._unbound_cleanup[id(backend)]  # noqa: SLF001
    assert retained[1] is backend

    with pytest.raises(_backends.BackendCleanupError):
        pool.close()

    assert pool._unbound_cleanup[id(backend)][1] is backend  # noqa: SLF001


def test_transient_cleanup_binding_failure_retains_until_close_proves_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TransientCloseDescriptor:
        name = "transient-close-descriptor"

        def __init__(self) -> None:
            self.close_lookups = 0
            self.close_calls = 0
            self.attach_calls = 0

        @property
        def close(self):
            self.close_lookups += 1
            if self.close_lookups == 1:
                raise RuntimeError("transient close descriptor failure")
            return self._close

        def _close(self) -> None:
            self.close_calls += 1

        def attach_toolset(self, _toolset, *, context_budget=None) -> None:
            del context_budget
            self.attach_calls += 1

    backend = _TransientCloseDescriptor()
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda _role, **_kwargs: backend,
    )
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )

    with pytest.raises(
        _backends.BackendLifecycleError,
        match="close authority could not be bound",
    ):
        pool.get("implementer", toolset=_WriteToolSet())

    assert backend.attach_calls == 0
    assert pool._unbound_cleanup[id(backend)][1] is backend  # noqa: SLF001

    pool.close()

    assert backend.close_lookups == 2
    assert backend.close_calls == 1
    assert pool._unbound_cleanup == {}  # noqa: SLF001
    assert pool._cleanup_authorities == {}  # noqa: SLF001
    assert pool._cleanup_errors == {}  # noqa: SLF001


def test_pool_uses_cleanup_authority_bound_before_writer_attach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MutableCloseWriter:
        name = "mutable-close"

        def __init__(self) -> None:
            self.close_calls = 0

        def attach_toolset(self, _toolset, *, context_budget=None) -> None:
            del context_budget

        def close(self) -> None:
            self.close_calls += 1

    backend = _MutableCloseWriter()
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda _role, **_kwargs: backend,
    )
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )
    pool.get("implementer", toolset=_WriteToolSet())

    def _revoked_close() -> None:
        raise AssertionError("cleanup authority was rediscovered")

    backend.close = _revoked_close  # type: ignore[method-assign]
    pool.close()

    assert backend.close_calls == 1
    assert pool._cache == {}  # noqa: SLF001
