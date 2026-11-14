# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Writer-epoch authority tests for provider-first settlement."""

from __future__ import annotations

import copy
from contextlib import AbstractContextManager

import pytest

from crucible.workflows._provider_settlement import (
    ProviderSettlementCleanupError,
    ProviderSettlementExitStack,
)
from crucible.workflows._writer_epoch import (
    initial_writer_epoch_record,
    inspect_writer_epoch_family,
)


class _MemoryWriterEpochStorage:
    def __init__(self, *, fail_writes: int = 0) -> None:
        self.transaction_id = "provider-journal"
        self.parent_transaction_id = None
        self.active = True
        self.record = initial_writer_epoch_record()
        self.fail_writes = fail_writes

    def read_writer_epoch_record(self) -> object:
        return copy.deepcopy(self.record)

    def compare_and_write_writer_epoch_record(
        self,
        expected: object,
        replacement: dict[str, object],
    ) -> object:
        if self.fail_writes:
            self.fail_writes -= 1
            raise RuntimeError("durable epoch write interrupted")
        if self.record != expected:
            raise RuntimeError("writer epoch compare-and-write conflict")
        self.record = copy.deepcopy(replacement)
        return copy.deepcopy(self.record)


class _RetryCloseError(RuntimeError):
    def __init__(self, context: "_ProviderContext") -> None:
        self.context = context
        super().__init__("provider close remains pending")

    def retry_cleanup(self) -> None:
        self.context.retry_close()


class _ProviderContext(AbstractContextManager[object]):
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_close: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.fail_close = fail_close
        self.entered = False
        self.closed = False
        self.close_attempts = 0

    def __enter__(self) -> object:
        self.entered = True
        self.events.append(f"enter:{self.name}")
        return self

    def __exit__(self, *_exc) -> bool:
        self.close_attempts += 1
        self.events.append(f"close:{self.name}:{self.close_attempts}")
        if self.fail_close:
            raise _RetryCloseError(self)
        self.closed = True
        return False

    def retry_close(self) -> None:
        self.close_attempts += 1
        self.events.append(f"retry:{self.name}:{self.close_attempts}")
        if self.fail_close:
            raise _RetryCloseError(self)
        self.closed = True


class _PoolContext(_ProviderContext):
    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__(name, events)
        self.get_calls: list[tuple[str, object]] = []

    def get(self, role: str, *, capability=None, **_kwargs):
        self.get_calls.append((role, capability))
        return (role, capability)


def test_provider_close_quiesces_epoch_before_mutation_callback() -> None:
    storage = _MemoryWriterEpochStorage()
    events: list[str] = []
    context = _ProviderContext("one", events)

    with ProviderSettlementExitStack() as owner:
        owner.callback(events.append, "journal-settle")
        owner.arm_provider_generation((storage,))
        assert owner.enter_provider_context(context) is context

    assert events == ["enter:one", "close:one:1", "journal-settle"]
    inspection = inspect_writer_epoch_family((storage,))
    assert inspection.state == "quiesced"
    settlement = storage.record["settlement"]
    assert settlement["kind"] == "provider_close_drain"
    assert settlement["outcome"] == "succeeded"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_provider_close_failure_retains_exact_owner_and_armed_epoch() -> None:
    storage = _MemoryWriterEpochStorage()
    events: list[str] = []
    context = _ProviderContext("retry", events, fail_close=True)

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        with ProviderSettlementExitStack() as owner:
            owner.callback(events.append, "journal-settle")
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(context)

    cleanup = captured.value
    assert cleanup.owner is owner
    assert events == ["enter:retry", "close:retry:1"]
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert owner.settlement_pending

    with pytest.raises(ProviderSettlementCleanupError) as retried:
        cleanup.retry_cleanup()
    assert retried.value is cleanup
    assert events == [
        "enter:retry",
        "close:retry:1",
        "retry:retry:2",
    ]
    assert inspect_writer_epoch_family((storage,)).state == "armed"

    context.fail_close = False
    cleanup.retry_cleanup()

    assert context.closed
    assert events == [
        "enter:retry",
        "close:retry:1",
        "retry:retry:2",
        "retry:retry:3",
        "journal-settle",
    ]
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.cleanup_complete


def test_terminal_cause_retains_cleanup_owner_without_implicit_retry() -> None:
    from crucible.tools import ToolMutationIndeterminate

    events: list[str] = []
    storage = _MemoryWriterEpochStorage()
    context = _ProviderContext("retry", events, fail_close=True)
    cleanup_owner: ProviderSettlementCleanupError | None = None
    terminal = ToolMutationIndeterminate("source settlement is unknown")

    with pytest.raises(ToolMutationIndeterminate) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.callback(events.append, "journal-settle")
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(context)
            try:
                owner.quiesce_providers()
            except ProviderSettlementCleanupError as exc:
                cleanup_owner = exc
                raise terminal from exc

    assert caught.value is terminal
    assert caught.value.__cause__ is cleanup_owner
    assert cleanup_owner is not None
    assert cleanup_owner.owner is owner
    assert events == ["enter:retry", "close:retry:1"]
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert owner.settlement_pending
    assert not owner.cleanup_complete

    context.fail_close = False
    cleanup_owner.retry_cleanup()

    assert events == [
        "enter:retry",
        "close:retry:1",
        "retry:retry:2",
        "journal-settle",
    ]
    assert owner.cleanup_complete
    assert not owner.settlement_pending
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"


def test_exit_preserves_tmi_when_provider_close_fails() -> None:
    from crucible.tools import ToolMutationIndeterminate

    events: list[str] = []
    storage = _MemoryWriterEpochStorage()
    context = _ProviderContext("tmi-close", events, fail_close=True)
    terminal = ToolMutationIndeterminate("source settlement is unknown")

    with pytest.raises(ToolMutationIndeterminate) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.callback(events.append, "journal-settle")
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(context)
            raise terminal

    assert caught.value is terminal
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.owner is owner
    assert cleanup_owner.primary_error is terminal
    assert cleanup_owner.__cause__ is cleanup_owner.cleanup_error
    assert isinstance(cleanup_owner.cleanup_error, _RetryCloseError)
    assert events == ["enter:tmi-close", "close:tmi-close:1"]
    assert context.close_attempts == 1
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert owner.settlement_pending
    assert not owner.cleanup_complete

    context.fail_close = False
    cleanup_owner.retry_cleanup()

    assert events == [
        "enter:tmi-close",
        "close:tmi-close:1",
        "retry:tmi-close:2",
        "journal-settle",
    ]
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_exit_preserves_tmi_when_settlement_callback_fails() -> None:
    from crucible.tools import ToolMutationIndeterminate

    events: list[str] = []
    storage = _MemoryWriterEpochStorage()
    context = _ProviderContext("tmi-callback", events)
    terminal = ToolMutationIndeterminate("source settlement is unknown")
    allow_rollback = False
    rollback_calls = 0

    def rollback() -> None:
        nonlocal rollback_calls
        rollback_calls += 1
        events.append(f"rollback:{rollback_calls}")
        if not allow_rollback:
            raise RuntimeError("journal rollback remains pending")

    with pytest.raises(ToolMutationIndeterminate) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.callback(rollback)
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(context)
            raise terminal

    assert caught.value is terminal
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.owner is owner
    assert cleanup_owner.primary_error is terminal
    assert cleanup_owner.__cause__ is cleanup_owner.cleanup_error
    assert isinstance(cleanup_owner.cleanup_error, RuntimeError)
    assert str(cleanup_owner.cleanup_error) == "journal rollback remains pending"
    assert events == [
        "enter:tmi-callback",
        "close:tmi-callback:1",
        "rollback:1",
    ]
    assert rollback_calls == 1
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.settlement_pending
    assert not owner.cleanup_complete

    allow_rollback = True
    cleanup_owner.retry_cleanup()

    assert events == [
        "enter:tmi-callback",
        "close:tmi-callback:1",
        "rollback:1",
        "rollback:2",
    ]
    assert rollback_calls == 2
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_retained_tmi_preserves_normal_return_when_cleanup_succeeds() -> None:
    from crucible.tools import ToolMutationIndeterminate

    events: list[str] = []
    terminal = ToolMutationIndeterminate("source settlement is unknown")
    owner = ProviderSettlementExitStack()

    def run() -> int:
        with owner:
            owner.callback(events.append, "journal-settle")
            assert owner.retain_terminal_precedence(terminal) is terminal
            return 3

    assert run() == 3
    assert events == ["journal-settle"]
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_retained_tmi_is_one_shot_and_requires_active_stack() -> None:
    from crucible.tools import ToolMutationIndeterminate

    terminal = ToolMutationIndeterminate("source settlement is unknown")
    conflicting = ToolMutationIndeterminate("different terminal")
    owner = ProviderSettlementExitStack()

    with pytest.raises(RuntimeError, match="not active"):
        owner.retain_terminal_precedence(terminal)
    with pytest.raises(TypeError, match="ToolMutationIndeterminate"):
        owner.retain_terminal_precedence(RuntimeError("wrong type"))  # type: ignore[arg-type]

    with owner:
        owner.retain_terminal_precedence(terminal)
        with pytest.raises(RuntimeError, match="already retained"):
            owner.retain_terminal_precedence(terminal)
        with pytest.raises(RuntimeError, match="already retained"):
            owner.retain_terminal_precedence(conflicting)

    with pytest.raises(RuntimeError, match="not active"):
        owner.retain_terminal_precedence(conflicting)


def test_retained_tmi_does_not_replace_explicit_body_exception() -> None:
    from crucible.tools import ToolMutationIndeterminate

    terminal = ToolMutationIndeterminate("source settlement is unknown")
    body_error = ValueError("later explicit body failure")
    allow_rollback = False

    def rollback() -> None:
        if not allow_rollback:
            raise RuntimeError("journal rollback remains pending")

    with pytest.raises(ProviderSettlementCleanupError) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.callback(rollback)
            owner.retain_terminal_precedence(terminal)
            raise body_error

    cleanup_owner = caught.value
    assert cleanup_owner.owner is owner
    assert cleanup_owner.primary_error is body_error
    assert isinstance(cleanup_owner.cleanup_error, RuntimeError)

    allow_rollback = True
    cleanup_owner.retry_cleanup()
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_retained_tmi_is_outer_when_normal_return_provider_close_fails() -> None:
    from crucible.tools import ToolMutationIndeterminate

    events: list[str] = []
    storage = _MemoryWriterEpochStorage()
    context = _ProviderContext("latched-close", events, fail_close=True)
    terminal = ToolMutationIndeterminate("source settlement is unknown")

    with pytest.raises(ToolMutationIndeterminate) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.callback(events.append, "journal-settle")
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(context)
            owner.retain_terminal_precedence(terminal)

    assert caught.value is terminal
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.owner is owner
    assert cleanup_owner.primary_error is terminal
    assert cleanup_owner.__cause__ is cleanup_owner.cleanup_error
    assert isinstance(cleanup_owner.cleanup_error, _RetryCloseError)
    assert events == ["enter:latched-close", "close:latched-close:1"]
    assert context.close_attempts == 1
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert owner.settlement_pending
    assert not owner.cleanup_complete

    context.fail_close = False
    cleanup_owner.retry_cleanup()

    assert events == [
        "enter:latched-close",
        "close:latched-close:1",
        "retry:latched-close:2",
        "journal-settle",
    ]
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_retained_tmi_is_outer_when_normal_return_callback_fails() -> None:
    from crucible.tools import ToolMutationIndeterminate

    events: list[str] = []
    terminal = ToolMutationIndeterminate("source settlement is unknown")
    allow_rollback = False
    rollback_calls = 0

    def rollback() -> None:
        nonlocal rollback_calls
        rollback_calls += 1
        events.append(f"rollback:{rollback_calls}")
        if not allow_rollback:
            raise RuntimeError("journal rollback remains pending")

    with pytest.raises(ToolMutationIndeterminate) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.callback(rollback)
            owner.retain_terminal_precedence(terminal)

    assert caught.value is terminal
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.owner is owner
    assert cleanup_owner.primary_error is terminal
    assert cleanup_owner.__cause__ is cleanup_owner.cleanup_error
    assert isinstance(cleanup_owner.cleanup_error, RuntimeError)
    assert str(cleanup_owner.cleanup_error) == "journal rollback remains pending"
    assert rollback_calls == 1
    assert events == ["rollback:1"]
    assert owner.settlement_pending
    assert not owner.cleanup_complete

    allow_rollback = True
    cleanup_owner.retry_cleanup()

    assert rollback_calls == 2
    assert events == ["rollback:1", "rollback:2"]
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_settled_backend_sigint_replays_after_epoch_and_callbacks() -> None:
    """A typed settled cancellation cannot leave provider authority armed."""

    from crucible.workflows._backends import BackendSettlementInterrupted

    storage = _MemoryWriterEpochStorage()
    events: list[str] = []
    original_interrupt = KeyboardInterrupt()
    backend_owner = object()

    class _SettledInterruptContext(AbstractContextManager[object]):
        def __enter__(self) -> object:
            events.append("enter")
            return backend_owner

        def __exit__(self, *_exc) -> bool:
            events.append("backend-settled")
            raise BackendSettlementInterrupted(
                backend_owner,
                original_interrupt
            ) from original_interrupt

    with pytest.raises(KeyboardInterrupt) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.callback(events.append, "journal-settle")
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(_SettledInterruptContext())

    assert caught.value is original_interrupt
    assert events == ["enter", "backend-settled", "journal-settle"]
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_run_after_quiescence_replays_interrupt_without_cleanup_wrapper() -> None:
    """A completed retained action is outside cleanup-error classification."""

    from crucible.workflows._backends import BackendSettlementInterrupted

    storage = _MemoryWriterEpochStorage()
    events: list[str] = []
    backend_owner = object()
    original_interrupt = KeyboardInterrupt()

    class _SettledInterruptContext(AbstractContextManager[object]):
        def __enter__(self) -> object:
            return backend_owner

        def __exit__(self, *_exc) -> bool:
            raise BackendSettlementInterrupted(
                backend_owner,
                original_interrupt,
            ) from original_interrupt

    with pytest.raises(KeyboardInterrupt) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(_SettledInterruptContext())
            owner.run_after_provider_quiescence(events.append, "retained-action")

    assert caught.value is original_interrupt
    assert events == ["retained-action"]
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_run_after_quiescence_retains_tmi_over_provider_close_failure() -> None:
    from crucible.tools import ToolMutationIndeterminate

    events: list[str] = []
    storage = _MemoryWriterEpochStorage()
    context = _ProviderContext("run-after-close", events, fail_close=True)
    terminal = ToolMutationIndeterminate("source settlement is unknown")

    with pytest.raises(ToolMutationIndeterminate) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(context)
            owner.retain_terminal_precedence(terminal)
            owner.run_after_provider_quiescence(
                events.append,
                "mutation-action",
            )

    assert caught.value is terminal
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.owner is owner
    assert cleanup_owner.primary_error is terminal
    assert cleanup_owner.__cause__ is cleanup_owner.cleanup_error
    assert isinstance(cleanup_owner.cleanup_error, _RetryCloseError)
    assert events == ["enter:run-after-close", "close:run-after-close:1"]
    assert context.close_attempts == 1
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert owner.settlement_pending
    assert not owner.cleanup_complete

    context.fail_close = False
    cleanup_owner.retry_cleanup()

    assert events == [
        "enter:run-after-close",
        "close:run-after-close:1",
        "retry:run-after-close:2",
        "mutation-action",
    ]
    assert context.close_attempts == 2
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_run_after_quiescence_retains_tmi_over_wrapped_action_failure() -> None:
    from crucible.tools import ToolMutationIndeterminate

    events: list[str] = []
    terminal = ToolMutationIndeterminate("source settlement is unknown")
    allow_action = False
    action_calls = 0

    def action() -> bool:
        nonlocal action_calls
        action_calls += 1
        events.append(f"action:{action_calls}")
        if not allow_action:
            raise RuntimeError("journal rollback remains pending")
        return False

    def caller_wrapper(
        owner: ProviderSettlementExitStack,
    ) -> bool:
        return owner.run_after_provider_quiescence(lambda: action())

    with pytest.raises(ToolMutationIndeterminate) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.retain_terminal_precedence(terminal)
            caller_wrapper(owner)

    assert caught.value is terminal
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.owner is owner
    assert cleanup_owner.primary_error is terminal
    assert cleanup_owner.__cause__ is cleanup_owner.cleanup_error
    assert isinstance(cleanup_owner.cleanup_error, RuntimeError)
    assert str(cleanup_owner.cleanup_error) == (
        "journal rollback remains pending"
    )
    assert action_calls == 1
    assert events == ["action:1"]
    assert owner.settlement_pending
    assert not owner.cleanup_complete

    allow_action = True
    cleanup_owner.retry_cleanup()

    assert action_calls == 2
    assert events == ["action:1", "action:2"]
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_run_after_quiescence_clean_false_result_is_not_terminal() -> None:
    from crucible.tools import ToolMutationIndeterminate

    terminal = ToolMutationIndeterminate("source settlement is unknown")

    def caller_wrapper(
        owner: ProviderSettlementExitStack,
    ) -> bool:
        return owner.run_after_provider_quiescence(lambda: False)

    with ProviderSettlementExitStack() as owner:
        owner.retain_terminal_precedence(terminal)
        result = caller_wrapper(owner)

    assert result is False
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_run_after_quiescence_without_latch_keeps_cleanup_owner_outer() -> None:
    allow_action = False
    action_calls = 0

    def action() -> None:
        nonlocal action_calls
        action_calls += 1
        if not allow_action:
            raise RuntimeError("journal rollback remains pending")

    with pytest.raises(ProviderSettlementCleanupError) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.run_after_provider_quiescence(action)

    cleanup_owner = caught.value
    assert cleanup_owner.owner is owner
    assert cleanup_owner.primary_error is cleanup_owner.cleanup_error
    assert isinstance(cleanup_owner.cleanup_error, RuntimeError)
    assert action_calls == 1
    assert owner.settlement_pending
    assert not owner.cleanup_complete

    allow_action = True
    cleanup_owner.retry_cleanup()

    assert action_calls == 2
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_exit_does_not_rewrap_latched_tmi_from_nested_callback() -> None:
    from crucible.tools import ToolMutationIndeterminate

    terminal = ToolMutationIndeterminate("source settlement is unknown")
    allow_action = False
    action_pending = True
    action_calls = 0
    owner = ProviderSettlementExitStack()

    def action() -> None:
        nonlocal action_calls, action_pending
        action_calls += 1
        if not allow_action:
            raise RuntimeError("journal rollback remains pending")
        action_pending = False

    def outer_callback() -> None:
        if action_pending:
            owner.run_after_provider_quiescence(action)

    with pytest.raises(ToolMutationIndeterminate) as caught:
        with owner:
            owner.retain_terminal_precedence(terminal)
            owner.callback(outer_callback)
            raise terminal

    assert caught.value is terminal
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.owner is owner
    assert cleanup_owner.primary_error is terminal
    assert isinstance(cleanup_owner.cleanup_error, RuntimeError)
    assert cleanup_owner.cleanup_error is not terminal
    assert cleanup_owner.__cause__ is cleanup_owner.cleanup_error
    assert cleanup_owner.__cause__ is not caught.value
    assert action_calls == 1
    assert owner.settlement_pending
    assert not owner.cleanup_complete

    allow_action = True
    cleanup_owner.retry_cleanup()

    assert action_calls == 2
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_unproven_raw_sigint_keeps_provider_epoch_armed() -> None:
    """Process death or an untyped interrupt alone is never close evidence."""

    storage = _MemoryWriterEpochStorage()
    events: list[str] = []

    class _UnprovenInterruptContext(AbstractContextManager[object]):
        def __enter__(self) -> object:
            events.append("enter")
            return self

        def __exit__(self, *_exc) -> bool:
            events.append("unproven-interrupt")
            raise KeyboardInterrupt

    with pytest.raises(ProviderSettlementCleanupError) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.callback(events.append, "journal-must-remain-armed")
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(_UnprovenInterruptContext())

    assert isinstance(caught.value.cleanup_error.__cause__, KeyboardInterrupt)
    assert "without typed positive close evidence" in str(
        caught.value.cleanup_error
    )
    assert events == ["enter", "unproven-interrupt"]
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert owner.settlement_pending
    assert not owner.cleanup_complete


def test_foreign_settled_interrupt_cannot_quiesce_provider_epoch() -> None:
    """A typed marker from any owner except the entered pool is rejected."""

    from crucible.workflows._backends import BackendSettlementInterrupted

    storage = _MemoryWriterEpochStorage()
    expected_owner = object()

    class _ForeignInterruptContext(AbstractContextManager[object]):
        def __enter__(self) -> object:
            return expected_owner

        def __exit__(self, *_exc) -> bool:
            raise BackendSettlementInterrupted(
                object(),
                KeyboardInterrupt(),
            )

    with pytest.raises(ProviderSettlementCleanupError) as caught:
        with ProviderSettlementExitStack() as owner:
            owner.callback(lambda: None)
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(_ForeignInterruptContext())

    assert "not bound to the entered provider owner" in str(
        caught.value.cleanup_error
    )
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert owner.settlement_pending
    with pytest.raises(ProviderSettlementCleanupError):
        caught.value.retry_cleanup()
    assert inspect_writer_epoch_family((storage,)).state == "armed"


def test_repeated_callback_sigint_returns_retryable_provider_owner() -> None:
    """Persistent cancellation cannot hot-loop or discard retained callbacks."""

    calls = 0
    original_interrupt = KeyboardInterrupt()

    def interrupted_callback() -> None:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise original_interrupt

    with pytest.raises(ProviderSettlementCleanupError) as pending:
        with ProviderSettlementExitStack() as owner:
            owner.callback(interrupted_callback)

    assert calls == 2
    assert owner.settlement_pending
    assert not owner.cleanup_complete
    with pytest.raises(KeyboardInterrupt) as replayed:
        pending.value.retry_cleanup()

    assert replayed.value is original_interrupt
    assert calls == 3
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_repeated_epoch_sigint_retains_exact_retryable_lease() -> None:
    """Bounded cancellation keeps settled-provider epoch authority retryable."""

    storage = _MemoryWriterEpochStorage()
    context = _ProviderContext("epoch-interrupt", [])
    original_interrupt = KeyboardInterrupt()
    owner = ProviderSettlementExitStack()
    owner.__enter__()
    owner.arm_provider_generation((storage,))
    owner.enter_provider_context(context)
    real_settle = owner._settle_provider_epoch
    calls = 0

    def interrupted_settle() -> None:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise original_interrupt
        real_settle()

    owner._settle_provider_epoch = interrupted_settle  # type: ignore[method-assign]

    with pytest.raises(ProviderSettlementCleanupError) as pending:
        owner.quiesce_providers()

    assert calls == 2
    assert context.closed
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert owner.settlement_pending
    assert not owner.cleanup_complete

    owner._settle_provider_epoch = real_settle  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt) as replayed:
        pending.value.retry_cleanup()

    assert replayed.value is original_interrupt
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_repeated_post_epoch_sigint_recognizes_durable_settlement() -> None:
    """Cancellation after lease clearing cannot strand an incidental error."""

    storage = _MemoryWriterEpochStorage()
    context = _ProviderContext("post-epoch-interrupt", [])
    original_interrupt = KeyboardInterrupt()
    owner = ProviderSettlementExitStack()
    owner.__enter__()
    owner.arm_provider_generation((storage,))
    owner.enter_provider_context(context)
    real_settle = owner._settle_provider_epoch
    calls = 0

    def settle_then_interrupt() -> None:
        nonlocal calls
        calls += 1
        real_settle()
        if calls <= 2:
            raise original_interrupt

    owner._settle_provider_epoch = settle_then_interrupt  # type: ignore[method-assign]

    with pytest.raises(ProviderSettlementCleanupError) as pending:
        owner.quiesce_providers()

    assert calls == 2
    assert context.closed
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.settlement_pending

    owner._settle_provider_epoch = real_settle  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt) as replayed:
        pending.value.retry_cleanup()

    assert replayed.value is original_interrupt
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_storage_sigint_during_quiescence_is_retried_then_replayed() -> None:
    """Writer-epoch wrapping cannot consume cancellation during durable I/O."""

    original_interrupt = KeyboardInterrupt()

    class _InterruptingStorage(_MemoryWriterEpochStorage):
        interrupt_next_write = False

        def compare_and_write_writer_epoch_record(
            self,
            expected: object,
            replacement: dict[str, object],
        ) -> object:
            if self.interrupt_next_write:
                self.interrupt_next_write = False
                raise original_interrupt
            return super().compare_and_write_writer_epoch_record(
                expected,
                replacement,
            )

    storage = _InterruptingStorage()
    context = _ProviderContext("storage-interrupt", [])

    with pytest.raises(KeyboardInterrupt) as replayed:
        with ProviderSettlementExitStack() as owner:
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(context)
            storage.interrupt_next_write = True
            owner.quiesce_providers()

    assert replayed.value is original_interrupt
    assert context.closed
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_storage_sigint_during_arm_is_replayed_after_failed_generation() -> None:
    """Partial arm retry retains cancellation and proves no provider dispatch."""

    original_interrupt = KeyboardInterrupt()

    class _InterruptingStorage(_MemoryWriterEpochStorage):
        def __init__(self) -> None:
            super().__init__()
            self.write_calls = 0

        def compare_and_write_writer_epoch_record(
            self,
            expected: object,
            replacement: dict[str, object],
        ) -> object:
            self.write_calls += 1
            if self.write_calls in {1, 3}:
                raise original_interrupt
            return super().compare_and_write_writer_epoch_record(
                expected,
                replacement,
            )

    storage = _InterruptingStorage()
    owner = ProviderSettlementExitStack()
    owner.__enter__()

    with pytest.raises(ProviderSettlementCleanupError) as pending:
        owner.arm_provider_generation((storage,))

    assert inspect_writer_epoch_family((storage,)).state == "never_armed"
    assert owner.settlement_pending
    with pytest.raises(KeyboardInterrupt) as replayed:
        pending.value.retry_cleanup()

    assert replayed.value is original_interrupt
    assert storage.write_calls == 4
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert storage.record["settlement"]["outcome"] == "failed"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_post_arm_sigint_retains_prepublished_exact_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation after durable arm return cannot lose cleanup authority."""

    from crucible.workflows import _provider_settlement as settlement_mod

    storage = _MemoryWriterEpochStorage()
    original_interrupt = KeyboardInterrupt()
    real_retry_arm = settlement_mod.retry_writer_epoch_arm
    calls = 0

    def arm_then_interrupt(lease):
        nonlocal calls
        calls += 1
        result = real_retry_arm(lease)
        if calls == 1:
            raise original_interrupt
        return result

    monkeypatch.setattr(
        settlement_mod,
        "retry_writer_epoch_arm",
        arm_then_interrupt,
    )
    owner = ProviderSettlementExitStack()
    owner.__enter__()

    with pytest.raises(ProviderSettlementCleanupError) as pending:
        owner.arm_provider_generation((storage,))

    assert calls == 1
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert owner.settlement_pending
    with pytest.raises(KeyboardInterrupt) as replayed:
        pending.value.retry_cleanup()

    assert replayed.value is original_interrupt
    assert calls == 2
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert storage.record["settlement"]["outcome"] == "failed"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_sigint_during_arm_authority_publication_is_not_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provisional retry owner and original cancellation survive together."""

    storage = _MemoryWriterEpochStorage()
    original_interrupt = KeyboardInterrupt()
    owner = ProviderSettlementExitStack()
    owner.__enter__()
    real_publish = owner._publish_prepared_provider_epoch

    def publish_then_interrupt(*args, **kwargs) -> None:
        real_publish(*args, **kwargs)
        raise original_interrupt

    monkeypatch.setattr(
        owner,
        "_publish_prepared_provider_epoch",
        publish_then_interrupt,
    )

    with pytest.raises(ProviderSettlementCleanupError) as pending:
        owner.arm_provider_generation((storage,))

    assert inspect_writer_epoch_family((storage,)).state == "never_armed"
    assert owner.settlement_pending
    with pytest.raises(KeyboardInterrupt) as replayed:
        pending.value.retry_cleanup()

    assert replayed.value is original_interrupt
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert storage.record["settlement"]["outcome"] == "failed"
    assert owner.cleanup_complete
    assert not owner.settlement_pending


def test_provider_generations_are_distinct_and_settle_in_order() -> None:
    storage = _MemoryWriterEpochStorage()
    events: list[str] = []
    generations: list[int] = []
    epoch_ids: list[str] = []

    with ProviderSettlementExitStack() as owner:
        for name in ("one", "two"):
            context = _ProviderContext(name, events)
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(context)
            owner.quiesce_providers()
            inspection = inspect_writer_epoch_family((storage,))
            assert inspection.state == "quiesced"
            assert inspection.generation is not None
            assert inspection.epoch_id is not None
            generations.append(inspection.generation)
            epoch_ids.append(inspection.epoch_id)

    assert events == [
        "enter:one",
        "close:one:1",
        "enter:two",
        "close:two:1",
    ]
    assert generations == [1, 2]
    assert len(set(epoch_ids)) == 2


def test_next_provider_generation_is_refused_until_prior_one_settles() -> None:
    storage = _MemoryWriterEpochStorage()
    owner = ProviderSettlementExitStack()
    owner.__enter__()
    owner.arm_provider_generation((storage,))
    owner.enter_provider_context(_ProviderContext("active", []))

    with pytest.raises(RuntimeError, match="prior provider generation"):
        owner.arm_provider_generation((storage,))

    owner.quiesce_providers()
    owner.__exit__(None, None, None)


def test_writer_provider_context_rejects_unarmed_entry() -> None:
    events: list[str] = []
    context = _ProviderContext("unarmed", events)

    with ProviderSettlementExitStack() as owner:
        with pytest.raises(RuntimeError, match="requires an exact armed"):
            owner.enter_provider_context(context)

    assert events == []


def test_read_only_generation_requires_close_and_arm_before_writer() -> None:
    storage = _MemoryWriterEpochStorage()
    events: list[str] = []
    read_context = _PoolContext("read", events)
    write_context = _PoolContext("write", events)

    with ProviderSettlementExitStack() as owner:
        read_pool = owner.enter_read_only_provider_context(read_context)
        assert read_pool.get("judge") == ("judge", "text")
        with pytest.raises(RuntimeError, match="workspace-write backend"):
            read_pool.get("implementer", capability="workspace_write")
        write_toolset = type(
            "_WriteToolSet",
            (),
            {"write_names": frozenset({"write_source"})},
        )()
        with pytest.raises(ValueError, match="mutating tools"):
            read_pool.get(
                "implementer",
                capability="read_only",
                toolset=write_toolset,
            )

        owner.quiesce_providers()
        owner.arm_provider_generation((storage,))
        write_pool = owner.enter_provider_context(write_context)
        assert write_pool.get(
            "implementer",
            capability="workspace_write",
        ) == ("implementer", "workspace_write")
        owner.quiesce_providers()

    assert events == [
        "enter:read",
        "close:read:1",
        "enter:write",
        "close:write:1",
    ]
    assert read_context.get_calls == [("judge", "text")]
    assert write_context.get_calls == [
        ("implementer", "workspace_write")
    ]


def test_second_generation_failure_before_enter_never_runs_callbacks() -> None:
    storage = _MemoryWriterEpochStorage()
    events: list[str] = []

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        with ProviderSettlementExitStack() as owner:
            owner.callback(events.append, "journal-settle")
            owner.arm_provider_generation((storage,))
            owner.enter_provider_context(_ProviderContext("first", events))
            owner.quiesce_providers()

            owner.arm_provider_generation((storage,))
            raise RuntimeError("provider context factory failed before enter")

    assert isinstance(captured.value.primary_error, RuntimeError)
    assert "factory failed" in str(captured.value.primary_error)
    assert events == ["enter:first", "close:first:1"]
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert owner.settlement_pending


def test_partial_epoch_arm_retry_proves_provider_was_not_dispatched() -> None:
    storage = _MemoryWriterEpochStorage(fail_writes=1)
    events: list[str] = []
    owner = ProviderSettlementExitStack()
    owner.__enter__()
    owner.callback(events.append, "journal-settle")

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        owner.arm_provider_generation((storage,))

    assert inspect_writer_epoch_family((storage,)).state == "never_armed"
    captured.value.retry_cleanup()

    assert events == ["journal-settle"]
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert storage.record["settlement"]["outcome"] == "failed"
    assert owner.cleanup_complete


def test_cleanup_error_identity_does_not_read_or_replace_public_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recognition uses the birth-time token, even with hostile owner hooks."""

    owner = ProviderSettlementExitStack()
    owner.__enter__()
    primary = RuntimeError("provider cleanup remains pending")
    retained = owner.retained_cleanup_error(
        primary_error=primary,
        cleanup_error=primary,
    )
    accesses = {"get": 0, "set": 0}
    armed = True

    def _get_owner(error):
        accesses["get"] += 1
        if armed:
            raise KeyboardInterrupt("owner getter must not run")
        return vars(error)["owner"]

    def _set_owner(error, value) -> None:
        accesses["set"] += 1
        if armed:
            raise KeyboardInterrupt("owner setter must not run")
        vars(error)["owner"] = value

    monkeypatch.setattr(
        ProviderSettlementCleanupError,
        "owner",
        property(_get_owner, _set_owner),
        raising=False,
    )

    assert owner.owns_cleanup_error(retained)
    assert owner._owned_cleanup_error_in_cause_chain(retained) is retained
    assert owner.__exit__(type(retained), retained, None) is False
    assert accesses == {"get": 0, "set": 0}
    assert not owner.cleanup_complete

    armed = False
    retained.retry_cleanup()
    assert accesses == {"get": 1, "set": 0}
    assert owner.cleanup_complete


def test_manually_constructed_cleanup_error_is_not_token_owned() -> None:
    """A matching public owner cannot forge the stack's birth-time token."""

    owner = ProviderSettlementExitStack()
    primary = RuntimeError("foreign cleanup error")
    unbound = ProviderSettlementCleanupError(
        owner,
        primary_error=primary,
        cleanup_error=primary,
    )

    assert owner.owns_cleanup_error(unbound) is False
    assert owner._owned_cleanup_error_in_cause_chain(unbound) is None
