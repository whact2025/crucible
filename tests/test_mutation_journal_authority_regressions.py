# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Durable adversarial regressions for journal discovery and family recovery."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys

import pytest

from .test_workflows_tier_mutation_transaction import (
    _setup,
    _simulate_process_death,
)


def _workspace_authority_state(journals) -> dict[str, object]:
    """Snapshot every thread-local and descriptor owner a lock may publish."""

    return {
        "held": dict(getattr(journals._LOCK_STATE, "held", {})),
        "authorities": dict(getattr(journals._LOCK_STATE, "authorities", {})),
        "external_common_held": dict(getattr(
            journals._LOCK_STATE,
            "external_common_held",
            {},
        )),
        "external_root_held": dict(getattr(
            journals._LOCK_STATE,
            "external_root_held",
            {},
        )),
        "external_workspace_authorities": tuple(getattr(
            journals._LOCK_STATE,
            "external_workspace_authorities",
            (),
        )),
        "lock_descriptors": set(journals._HELD_LOCK_DESCRIPTORS),
        "authority_descriptors": set(journals._HELD_AUTHORITY_DESCRIPTORS),
    }


def _assert_cross_process_recovery_lock_reacquires(
    repo: Path,
    corpus_dir: Path,
) -> None:
    """A fresh interpreter must acquire the same complete recovery chain."""

    probe = """
import sys
from pathlib import Path
from assay.corpus.store import Corpus
from crucible.workflows._mutation_journal import recovery_workspace_mutation_lock
with recovery_workspace_mutation_lock(Corpus(Path(sys.argv[2])), Path(sys.argv[1])):
    pass
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(repo), str(corpus_dir)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_retained_acquisition_owner_chain_keeps_failing_lifo_owner() -> None:
    """A failed inner release remains retryable before any outer release."""

    from crucible.workflows import _mutation_journal as journals

    events: list[str] = []
    sentinel = RuntimeError("synthetic inner release failure")

    class RetryOwner:
        def __init__(self, name: str, *, fail_once: bool = False) -> None:
            self.name = name
            self.fail_once = fail_once
            self._cleanup_complete = False

        @property
        def cleanup_complete(self) -> bool:
            return self._cleanup_complete

        def __enter__(self):
            events.append(f"enter:{self.name}")
            return self

        def retry_cleanup(self) -> None:
            events.append(f"release:{self.name}")
            if self.fail_once:
                self.fail_once = False
                raise sentinel
            self._cleanup_complete = True

    chain = journals._RetainedAcquisitionOwnerChain()
    chain.__enter__()
    chain.enter_context(RetryOwner("outer"))
    chain.enter_context(RetryOwner("inner", fail_once=True))

    with pytest.raises(journals._RetainedAcquisitionCleanupError) as captured:
        chain.__exit__(None, None, None)

    error = captured.value
    assert error.owner is chain
    assert error.cleanup_error is sentinel
    assert chain.cleanup_complete is False
    assert events == ["enter:outer", "enter:inner", "release:inner"]

    error.owner.retry_cleanup()

    assert chain.cleanup_complete
    assert events == [
        "enter:outer",
        "enter:inner",
        "release:inner",
        "release:inner",
        "release:outer",
    ]


def test_retained_source_adopts_and_retries_nested_cleanup_owner() -> None:
    """The public retained owner cannot report clean over a child failure."""

    from contextlib import contextmanager

    from crucible.workflows import _mutation_journal as journals

    events: list[str] = []
    sentinel = RuntimeError("nested release failed once")

    class RetryOwner:
        def __init__(self, name: str, *, fail_once: bool = False) -> None:
            self.name = name
            self.fail_once = fail_once
            self._cleanup_complete = False

        @property
        def cleanup_complete(self) -> bool:
            return self._cleanup_complete

        def __enter__(self):
            events.append(f"enter:{self.name}")
            return self

        def retry_cleanup(self) -> None:
            events.append(f"release:{self.name}")
            if self.fail_once:
                self.fail_once = False
                raise sentinel
            self._cleanup_complete = True

    lease = journals.WorkspaceMutationLockAcquisitionLease()

    @contextmanager
    def source():
        with journals._RetainedAcquisitionOwnerChain() as children:
            children.enter_context(RetryOwner("outer"))
            children.enter_context(RetryOwner("inner", fail_once=True))
            lease._publish()
            try:
                yield object()
            finally:
                lease._begin_settlement()

    retained = journals._RetainedAcquisition(source(), lease)
    retained.__enter__()
    with pytest.raises(journals._RetainedAcquisitionCleanupError) as captured:
        retained.__exit__(None, None, None)

    assert captured.value.cleanup_error is sentinel
    assert retained.cleanup_complete is False
    assert events == ["enter:outer", "enter:inner", "release:inner"]

    retained.retry_cleanup()

    assert retained.cleanup_complete
    assert events == [
        "enter:outer",
        "enter:inner",
        "release:inner",
        "release:inner",
        "release:outer",
    ]


def test_retained_owner_chain_does_not_mask_deferred_inner_interrupt() -> None:
    """A retry replays an inner post-release SIGINT after outer settlement."""

    from crucible.workflows import _mutation_journal as journals

    events: list[str] = []
    interrupt = KeyboardInterrupt("inner release returned")
    outer_failure = RuntimeError("outer release failed once")

    class Owner:
        def __init__(
            self,
            name: str,
            *,
            fail_before_complete: BaseException | None = None,
            fail_after_complete: BaseException | None = None,
        ) -> None:
            self.name = name
            self.fail_before_complete = fail_before_complete
            self.fail_after_complete = fail_after_complete
            self._cleanup_complete = False

        @property
        def cleanup_complete(self) -> bool:
            return self._cleanup_complete

        def __enter__(self):
            events.append(f"enter:{self.name}")
            return self

        def retry_cleanup(self) -> None:
            events.append(f"release:{self.name}")
            if self.fail_before_complete is not None:
                failure = self.fail_before_complete
                self.fail_before_complete = None
                raise failure
            self._cleanup_complete = True
            if self.fail_after_complete is not None:
                failure = self.fail_after_complete
                self.fail_after_complete = None
                raise failure

    chain = journals._RetainedAcquisitionOwnerChain()
    chain.__enter__()
    chain.enter_context(
        Owner("outer", fail_before_complete=outer_failure)
    )
    chain.enter_context(
        Owner("inner", fail_after_complete=interrupt)
    )

    with pytest.raises(journals._RetainedAcquisitionCleanupError) as captured:
        chain.__exit__(None, None, None)

    assert captured.value.cleanup_error is outer_failure
    assert not chain.cleanup_complete
    with pytest.raises(KeyboardInterrupt) as retried:
        captured.value.owner.retry_cleanup()

    assert retried.value is interrupt
    assert chain.cleanup_complete
    assert events == [
        "enter:outer",
        "enter:inner",
        "release:inner",
        "release:outer",
        "release:outer",
    ]


@pytest.mark.parametrize("interrupt_frontier", ("acquire", "release"))
def test_retained_in_process_lock_proves_post_success_interrupt(
    interrupt_frontier: str,
) -> None:
    """A post-operation SIGINT neither leaks nor invents one RLock depth."""

    from crucible.workflows import _mutation_journal as journals

    interrupt = KeyboardInterrupt(f"RLock {interrupt_frontier} returned")

    class InterruptingRLock:
        def __init__(self) -> None:
            self._lock = __import__("threading").RLock()
            self.depth = 0
            self.acquire_interrupt_pending = interrupt_frontier == "acquire"
            self.release_interrupt_pending = interrupt_frontier == "release"

        def acquire(self, blocking: bool = True) -> bool:
            acquired = self._lock.acquire(blocking)
            if acquired:
                self.depth += 1
            if acquired and self.acquire_interrupt_pending:
                self.acquire_interrupt_pending = False
                raise interrupt
            return acquired

        def release(self) -> None:
            self._lock.release()
            self.depth -= 1
            if self.release_interrupt_pending:
                self.release_interrupt_pending = False
                raise interrupt

        def _recursion_count(self) -> int:
            return self.depth

        def _is_owned(self) -> bool:
            return self.depth > 0

    lock = InterruptingRLock()
    retained = journals._retained_in_process_lock_admission(
        lock,
        nonblocking=False,
        label="fault-injected process lock",
    )

    with pytest.raises(KeyboardInterrupt) as captured:
        if interrupt_frontier == "acquire":
            retained.__enter__()
        else:
            retained.__enter__()
            retained.__exit__(None, None, None)

    assert captured.value is interrupt
    assert retained.cleanup_complete
    assert lock.depth == 0
    with journals._retained_in_process_lock_admission(
        lock,
        nonblocking=False,
        label="fault-injected process lock",
    ):
        assert lock.depth == 1
    assert lock.depth == 0


def test_retained_rlock_preserves_distinct_entry_and_cleanup_interrupts() -> None:
    """A second post-release signal remains the acquired signal's cause."""

    import threading

    from crucible.workflows import _mutation_journal as journals

    entry_interrupt = KeyboardInterrupt("RLock acquire returned")
    cleanup_interrupt = KeyboardInterrupt("RLock release returned")

    class DoublyInterruptingRLock:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self.depth = 0

        def acquire(self, blocking: bool = True) -> bool:
            acquired = self._lock.acquire(blocking)
            if acquired:
                self.depth += 1
                raise entry_interrupt
            return acquired

        def release(self) -> None:
            self._lock.release()
            self.depth -= 1
            raise cleanup_interrupt

        def _recursion_count(self) -> int:
            return self.depth

    lock = DoublyInterruptingRLock()
    owner = journals._RetainedRLockDepthOwner(
        lock,
        nonblocking=False,
        label="double-interrupt process lock",
    )

    with pytest.raises(KeyboardInterrupt) as captured:
        owner.__enter__()

    assert captured.value is entry_interrupt
    assert captured.value.__cause__ is cleanup_interrupt
    assert owner.cleanup_complete
    assert lock.depth == 0


@pytest.mark.parametrize("interrupt_frontier", ("acquire", "release"))
def test_recovery_chain_propagates_settled_rlock_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_frontier: str,
) -> None:
    """The same SIGINT crosses every parent only after complete LIFO cleanup."""

    import threading

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    baseline = _workspace_authority_state(journals)
    real_rlock = threading.RLock
    interrupt = KeyboardInterrupt(
        f"canonical RLock {interrupt_frontier} returned"
    )
    created = []

    class InterruptingRLock:
        def __init__(self, *, armed: bool) -> None:
            self._lock = real_rlock()
            self.depth = 0
            self.acquire_interrupt_pending = (
                armed and interrupt_frontier == "acquire"
            )
            self.release_interrupt_pending = (
                armed and interrupt_frontier == "release"
            )

        def acquire(self, blocking: bool = True) -> bool:
            acquired = self._lock.acquire(blocking)
            if acquired:
                self.depth += 1
            if acquired and self.acquire_interrupt_pending:
                self.acquire_interrupt_pending = False
                raise interrupt
            return acquired

        def release(self) -> None:
            self._lock.release()
            self.depth -= 1
            if self.release_interrupt_pending:
                self.release_interrupt_pending = False
                raise interrupt

        def _recursion_count(self) -> int:
            return self.depth

        def _is_owned(self) -> bool:
            return self.depth > 0

    def rlock_factory():
        # Fresh recovery creates common, root, then canonical process locks.
        lock = InterruptingRLock(armed=len(created) == 2)
        created.append(lock)
        return lock

    monkeypatch.setattr(journals.threading, "RLock", rlock_factory)
    retained = journals.recovery_workspace_mutation_lock(corpus, repo)

    with pytest.raises(KeyboardInterrupt) as captured:
        with retained:
            pass

    assert captured.value is interrupt
    assert len(created) >= 3
    assert retained.cleanup_complete
    assert all(lock.depth == 0 for lock in created)
    assert _workspace_authority_state(journals) == baseline
    with journals.recovery_workspace_mutation_lock(corpus, repo):
        assert journals._active_workspace_mutation_authority(repo) is not None
    assert _workspace_authority_state(journals) == baseline
    monkeypatch.undo()
    _assert_cross_process_recovery_lock_reacquires(repo, corpus.spec_dir)


def test_retained_cleanup_publication_avoids_hostile_exception_hooks() -> None:
    """A hostile cleanup error cannot displace the internal retry owner."""

    from crucible.workflows import _mutation_journal as journals

    class HostileError(RuntimeError):
        @property
        def owner(self):
            raise KeyboardInterrupt("hostile owner getter")

        def __str__(self) -> str:
            raise KeyboardInterrupt("hostile exception formatter")

    hostile = HostileError()

    class FailingOwner:
        cleanup_complete = False

        def __enter__(self):
            return self

        def retry_cleanup(self) -> None:
            raise hostile

    chain = journals._RetainedAcquisitionOwnerChain()
    chain.__enter__()
    chain.enter_context(FailingOwner())

    with pytest.raises(journals._RetainedAcquisitionCleanupError) as captured:
        chain.__exit__(None, None, None)

    assert captured.value.owner is chain
    assert captured.value.cleanup_error is hostile
    assert str(captured.value) == "retained acquisition cleanup failed"
    assert chain.cleanup_complete is False


def test_retained_chain_publishes_owner_when_cleanup_repeats_body_error() -> None:
    """The same hostile signal cannot escape while its owner remains live."""

    from crucible.workflows import _mutation_journal as journals

    events: list[str] = []

    class HostileInterrupt(KeyboardInterrupt):
        @property
        def owner(self):
            raise RuntimeError("hostile owner getter")

        def __str__(self) -> str:
            raise RuntimeError("hostile exception formatter")

    interrupt = HostileInterrupt()

    class Owner:
        def __init__(self, name: str, *, fail_once: bool = False) -> None:
            self.name = name
            self.fail_once = fail_once
            self._cleanup_complete = False

        @property
        def cleanup_complete(self) -> bool:
            return self._cleanup_complete

        def __enter__(self):
            events.append(f"enter:{self.name}")
            return self

        def retry_cleanup(self) -> None:
            events.append(f"release:{self.name}")
            if self.fail_once:
                self.fail_once = False
                raise interrupt
            self._cleanup_complete = True

    chain = journals._RetainedAcquisitionOwnerChain()
    chain.__enter__()
    chain.enter_context(Owner("outer"))
    chain.enter_context(Owner("inner", fail_once=True))

    with pytest.raises(journals._RetainedAcquisitionCleanupError) as captured:
        chain.__exit__(type(interrupt), interrupt, None)

    assert captured.value.owner is chain
    assert captured.value.primary_error is interrupt
    assert captured.value.cleanup_error is interrupt
    assert captured.value.__cause__ is interrupt
    assert events == ["enter:outer", "enter:inner", "release:inner"]

    captured.value.owner.retry_cleanup()

    assert chain.cleanup_complete
    assert events == [
        "enter:outer",
        "enter:inner",
        "release:inner",
        "release:inner",
        "release:outer",
    ]


def test_readable_spec_authority_preowns_caller_return_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crucible owns Assay cleanup before the retained enter call returns."""

    from crucible.workflows import _mutation_journal as journals

    _repo, corpus = _setup(tmp_path)
    retained_factory = corpus.retained_spec_authority
    assay_authority = retained_factory()

    class CallerBoundary:
        def __enter__(self):
            return assay_authority.__enter__()

        def retry_cleanup(self) -> None:
            assay_authority.retry_cleanup()

    boundary = CallerBoundary()
    monkeypatch.setattr(
        corpus,
        "retained_spec_authority",
        lambda: boundary,
    )
    enter_code = CallerBoundary.__enter__.__code__
    interrupt = KeyboardInterrupt("Assay authority caller return boundary")
    injected = False

    def interrupt_return(frame, event, _arg):
        nonlocal injected
        if (
            not injected
            and frame.f_code is enter_code
            and event == "return"
        ):
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            with journals._readable_spec_authority(corpus):
                pytest.fail("interrupted Spec authority must not enter its body")
    finally:
        sys.settrace(previous_trace)

    assert captured.value is interrupt
    assert injected
    assert assay_authority.cleanup_complete
    with retained_factory() as spec:
        assert spec == corpus.latest_spec()


@pytest.mark.parametrize("retained_inaccessible_root", (False, True))
def test_recovery_lock_source_lease_closes_callee_return_interrupt(
    tmp_path: Path,
    retained_inaccessible_root: bool,
) -> None:
    """The source-published lease owns a lock before its return is visible."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    if retained_inaccessible_root:
        journal = journals.WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "lease-return" / "mutations"
            ),
            slice_id="lease-return",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )
        _simulate_process_death(journal)
        repo.chmod(0)
    try:
        retained = journals.recovery_workspace_mutation_lock(corpus, repo)
        source = retained._context
        interrupt = KeyboardInterrupt("after recovery lock source yielded")

        class InterruptAfterSourceReturn:
            def __enter__(self):
                source.__enter__()
                raise interrupt

            def __exit__(self, exc_type, exc, traceback):
                return source.__exit__(exc_type, exc, traceback)

        retained._context = InterruptAfterSourceReturn()

        with pytest.raises(KeyboardInterrupt) as captured:
            retained.__enter__()

        assert captured.value is interrupt
        assert retained.cleanup_complete
        assert journals._active_workspace_mutation_authority(repo) is None
        if retained_inaccessible_root:
            assert stat.S_IMODE(repo.stat().st_mode) == 0
        with journals.recovery_workspace_mutation_lock(corpus, repo):
            assert journals._active_workspace_mutation_authority(repo) is not None
        if retained_inaccessible_root:
            assert stat.S_IMODE(repo.stat().st_mode) == 0
    finally:
        repo.chmod(baseline_mode)


def test_public_recovery_retains_lock_cleanup_before_entry_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public recovery caller covers its lock ``__enter__`` return gap."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    retained_contexts = []
    real_factory = journals.recovery_workspace_mutation_lock

    def observe_factory(bound_corpus, bound_repo):
        retained = real_factory(bound_corpus, bound_repo)
        retained_contexts.append(retained)
        return retained

    monkeypatch.setattr(
        journals,
        "recovery_workspace_mutation_lock",
        observe_factory,
    )
    enter_code = journals.RetainedRecoveryWorkspaceMutationLock.__enter__.__code__
    interrupt = KeyboardInterrupt("public recovery lock entry return boundary")
    injected = False

    def interrupt_return(frame, event, _arg):
        nonlocal injected
        if not injected and frame.f_code is enter_code and event == "return":
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            journals.recover_mutation_journals(corpus, repo)
    finally:
        sys.settrace(previous_trace)

    assert captured.value is interrupt
    assert injected is True
    assert len(retained_contexts) == 1
    assert retained_contexts[0].cleanup_complete
    assert journals._active_workspace_mutation_authority(repo) is None
    with real_factory(corpus, repo):
        assert journals._active_workspace_mutation_authority(repo) is not None


@pytest.mark.parametrize(
    ("boundary", "factory_name", "kind", "occurrence"),
    (
        ("inventory-bound-root", "_retained_bound_pre_root_workspace", None, 0),
        ("external-bound-root", "_retained_bound_pre_root_workspace", None, 1),
        ("external-common-rlock", "_retained_in_process_lock_admission", None, 0),
        ("external-common-file", "_retained_external_file_lock", "common", 0),
        ("external-root-rlock", "_retained_in_process_lock_admission", None, 1),
        ("external-root-file", "_retained_external_file_lock", "root", 0),
        ("external-coordination", "_retained_external_workspace_coordination", None, 0),
        ("canonical-rlock", "_retained_in_process_lock_admission", None, 2),
        ("canonical", "_retained_canonical_workspace_mutation_lock", None, 0),
        ("workspace", "_retained_workspace_mutation_lock", None, 0),
    ),
)
@pytest.mark.parametrize("return_surface", ("source", "wrapper"))
def test_recovery_source_chain_sigint_at_nested_return_releases_every_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    factory_name: str,
    kind: str | None,
    occurrence: int,
    return_surface: str,
) -> None:
    """Every source layer publishes cleanup before its return can be signaled."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    baseline = _workspace_authority_state(journals)
    original_factory = getattr(journals, factory_name)
    captured = []

    def capture_factory(*args, **kwargs):
        retained = original_factory(*args, **kwargs)
        if kind is None or kwargs.get("kind") == kind:
            captured.append(retained)
        return retained

    monkeypatch.setattr(journals, factory_name, capture_factory)
    retained = journals.recovery_workspace_mutation_lock(corpus, repo)
    injected = False

    def interrupt_return(frame, event, _arg):
        nonlocal injected
        retained_target = (
            captured[occurrence] if len(captured) > occurrence else None
        )
        target = (
            retained_target._context
            if retained_target is not None and return_surface == "source"
            else retained_target
        )
        if (
            not injected
            and target is not None
            and event == "return"
            and frame.f_locals.get("self") is target
        ):
            injected = True
            sys.settrace(None)
            signal.raise_signal(signal.SIGINT)
        return interrupt_return

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt):
            retained.__enter__()
    finally:
        sys.settrace(previous_trace)

    assert injected, boundary
    assert retained.cleanup_complete
    assert journals._active_workspace_mutation_authority(repo) is None
    assert journals._active_external_workspace_coordination(repo) is None
    assert _workspace_authority_state(journals) == baseline
    with journals.recovery_workspace_mutation_lock(corpus, repo):
        assert journals._active_workspace_mutation_authority(repo) is not None
    assert _workspace_authority_state(journals) == baseline
    _assert_cross_process_recovery_lock_reacquires(repo, corpus.spec_dir)


@pytest.mark.parametrize(
    ("factory_name", "occurrence"),
    (
        ("_retained_bound_pre_root_workspace", 0),
        ("_retained_bound_pre_root_workspace", 1),
        ("_retained_bound_pre_root_workspace", 2),
        ("_retained_external_workspace_coordination", 0),
        ("_retained_in_process_lock_admission", 2),
        ("_retained_canonical_workspace_mutation_lock", 0),
        ("_retained_workspace_mutation_lock", 0),
    ),
)
@pytest.mark.parametrize("return_surface", ("source", "wrapper"))
def test_inaccessible_recovery_nested_return_restores_mode_and_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    occurrence: int,
    return_surface: str,
) -> None:
    """Captured-root admission unwinds around its temporary chmod frontier."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "nested-inaccessible" / "mutations",
        slice_id="nested-inaccessible",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _simulate_process_death(journal)
    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    original_factory = getattr(journals, factory_name)
    captured = []

    def capture_factory(*args, **kwargs):
        retained = original_factory(*args, **kwargs)
        captured.append(retained)
        return retained

    monkeypatch.setattr(journals, factory_name, capture_factory)
    repo.chmod(0)
    baseline = _workspace_authority_state(journals)
    try:
        retained = journals.recovery_workspace_mutation_lock(corpus, repo)
        injected = False

        def interrupt_return(frame, event, _arg):
            nonlocal injected
            retained_target = (
                captured[occurrence] if len(captured) > occurrence else None
            )
            target = (
                retained_target._context
                if retained_target is not None and return_surface == "source"
                else retained_target
            )
            if (
                not injected
                and target is not None
                and event == "return"
                and frame.f_locals.get("self") is target
            ):
                injected = True
                sys.settrace(None)
                signal.raise_signal(signal.SIGINT)
            return interrupt_return

        previous_trace = sys.gettrace()
        sys.settrace(interrupt_return)
        try:
            with pytest.raises(KeyboardInterrupt):
                retained.__enter__()
        finally:
            sys.settrace(previous_trace)

        assert injected
        assert retained.cleanup_complete
        assert stat.S_IMODE(repo.stat().st_mode) == 0
        assert journals._active_workspace_mutation_authority(repo) is None
        assert journals._active_external_workspace_coordination(repo) is None
        assert _workspace_authority_state(journals) == baseline
        with journals.recovery_workspace_mutation_lock(corpus, repo):
            assert journals._active_workspace_mutation_authority(repo) is not None
        assert stat.S_IMODE(repo.stat().st_mode) == 0
        assert _workspace_authority_state(journals) == baseline
    finally:
        repo.chmod(baseline_mode)

    _assert_cross_process_recovery_lock_reacquires(repo, corpus.spec_dir)


@pytest.mark.parametrize(
    ("factory_name", "kind", "occurrence"),
    (
        ("_retained_in_process_lock_admission", None, 0),
        ("_retained_external_file_lock", "common", 0),
        ("_retained_in_process_lock_admission", None, 1),
        ("_retained_external_file_lock", "root", 0),
    ),
)
@pytest.mark.parametrize("return_surface", ("source", "wrapper"))
def test_nested_external_return_interrupt_preserves_parent_coordination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    kind: str | None,
    occurrence: int,
    return_surface: str,
) -> None:
    """Reentrant common/root layers release only their added lock depth."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    original_factory = getattr(journals, factory_name)
    captured = []

    def capture_factory(*args, **kwargs):
        retained = original_factory(*args, **kwargs)
        if kind is None or kwargs.get("kind") == kind:
            captured.append(retained)
        return retained

    monkeypatch.setattr(journals, factory_name, capture_factory)
    with journals._external_workspace_coordination(repo) as parent:
        captured.clear()
        baseline = _workspace_authority_state(journals)
        retained = journals._retained_external_workspace_coordination(repo)
        injected = False

        def interrupt_return(frame, event, _arg):
            nonlocal injected
            retained_target = (
                captured[occurrence] if len(captured) > occurrence else None
            )
            target = (
                retained_target._context
                if retained_target is not None and return_surface == "source"
                else retained_target
            )
            if (
                not injected
                and target is not None
                and event == "return"
                and frame.f_locals.get("self") is target
            ):
                injected = True
                sys.settrace(None)
                signal.raise_signal(signal.SIGINT)
            return interrupt_return

        previous_trace = sys.gettrace()
        sys.settrace(interrupt_return)
        try:
            with pytest.raises(KeyboardInterrupt):
                retained.__enter__()
        finally:
            sys.settrace(previous_trace)

        assert injected
        assert retained.cleanup_complete
        assert journals._active_workspace_mutation_authority(repo) is None
        assert journals._active_external_workspace_coordination(repo) is parent
        assert _workspace_authority_state(journals) == baseline
        with journals._retained_external_workspace_coordination(repo):
            nested = journals._active_external_workspace_coordination(repo)
            assert nested is not None
            assert nested is not parent
        assert journals._active_external_workspace_coordination(repo) is parent
        assert _workspace_authority_state(journals) == baseline

    _assert_cross_process_recovery_lock_reacquires(repo, corpus.spec_dir)


@pytest.mark.parametrize(
    "canonical_branch",
    ("fresh", "active", "held_without_authority"),
)
@pytest.mark.parametrize(
    "return_surface",
    (
        "in-process-source",
        "in-process-wrapper",
        "canonical-source",
        "canonical-wrapper",
    ),
)
def test_retained_nested_lock_closes_canonical_source_return_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_branch: str,
    return_surface: str,
) -> None:
    """A live-parent child retains the exact canonical acquisition owner."""

    from crucible.workflows import _mutation_journal as journals

    repo, _corpus = _setup(tmp_path)
    outer_context = (
        journals.workspace_mutation_lock(repo)
        if canonical_branch != "fresh"
        else journals._external_workspace_coordination(repo)
    )
    with outer_context as outer_authority:
        saved_authorities = getattr(journals._LOCK_STATE, "authorities", {})
        if canonical_branch == "held_without_authority":
            journals._LOCK_STATE.authorities = {}
        try:
            active_before = journals._active_workspace_mutation_authority(repo)
            held_before = dict(getattr(journals._LOCK_STATE, "held", {}))
            authorities_before = dict(
                getattr(journals._LOCK_STATE, "authorities", {})
            )
            held_lock_fds_before = set(journals._HELD_LOCK_DESCRIPTORS)
            held_authority_fds_before = set(
                journals._HELD_AUTHORITY_DESCRIPTORS
            )
            in_process_contexts = []
            if return_surface.startswith("in-process"):
                real_in_process_factory = (
                    journals._retained_in_process_lock_admission
                )

                def observe_in_process(*args, **kwargs):
                    context = real_in_process_factory(*args, **kwargs)
                    in_process_contexts.append(context)
                    return context

                monkeypatch.setattr(
                    journals,
                    "_retained_in_process_lock_admission",
                    observe_in_process,
                )
            retained = journals.retained_nested_workspace_mutation_lock(repo)
            interrupt = KeyboardInterrupt("canonical source return boundary")
            injected = False

            def interrupt_return(frame, event, _arg):
                nonlocal injected
                in_process = (
                    in_process_contexts[0] if in_process_contexts else None
                )
                if return_surface == "in-process-source":
                    target = (
                        in_process._context
                        if in_process is not None
                        else None
                    )
                elif return_surface == "in-process-wrapper":
                    target = in_process
                elif return_surface == "canonical-source":
                    target = retained._context
                else:
                    target = retained
                if (
                    not injected
                    and target is not None
                    and frame.f_locals.get("self") is target
                    and event == "return"
                ):
                    injected = True
                    sys.settrace(None)
                    raise interrupt
                return interrupt_return

            previous_trace = sys.gettrace()
            sys.settrace(interrupt_return)
            caller_owner = journals._RetainedAcquisitionOwnerChain()
            try:
                with pytest.raises(KeyboardInterrupt) as captured:
                    with caller_owner:
                        caller_owner.enter_context(retained)
            finally:
                sys.settrace(previous_trace)

            assert captured.value is interrupt
            assert injected
            assert retained.cleanup_complete
            assert caller_owner.cleanup_complete
            assert (
                journals._active_workspace_mutation_authority(repo)
                is active_before
            )
            assert dict(getattr(journals._LOCK_STATE, "held", {})) == held_before
            assert (
                dict(getattr(journals._LOCK_STATE, "authorities", {}))
                == authorities_before
            )
            assert set(journals._HELD_LOCK_DESCRIPTORS) == held_lock_fds_before
            assert (
                set(journals._HELD_AUTHORITY_DESCRIPTORS)
                == held_authority_fds_before
            )
            with journals.retained_nested_workspace_mutation_lock(repo):
                nested = journals._active_workspace_mutation_authority(repo)
                assert nested is not None
            assert (
                journals._active_workspace_mutation_authority(repo)
                is active_before
            )
            if canonical_branch == "active":
                assert active_before is outer_authority
            else:
                assert active_before is None
        finally:
            if canonical_branch == "held_without_authority":
                journals._LOCK_STATE.authorities = saved_authorities


@pytest.mark.parametrize(
    "swap_level",
    ("run", "durable_root", "transaction", "manifest"),
)
def test_recovery_rejects_post_discovery_identity_swap_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_level: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "victim" / "mutations",
        slice_id=f"victim-{swap_level}",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    crashed = f"# crashed {swap_level} bytes\n"
    (repo / "a.py").write_text(crashed, encoding="utf-8")
    _simulate_process_death(journal)

    transaction = journal.journal_dir
    run = corpus.spec_dir / "runs" / "victim"
    durable_root = run / "mutations"
    parked = tmp_path / f"parked-{swap_level}"
    original_manifest = (transaction / "manifest.json").read_bytes()
    original_discover = journals.discover_mutation_journals
    replacement_sentinel: list[Path] = []

    def swap_after_complete_inventory(bound_corpus, **kwargs):
        inventory = original_discover(bound_corpus, **kwargs)
        if replacement_sentinel:
            return inventory
        if swap_level == "manifest":
            manifest = transaction / "manifest.json"
            manifest.rename(parked)
            shutil.copy2(parked, manifest)
            replacement_sentinel.append(manifest)
        else:
            target = {
                "run": run,
                "durable_root": durable_root,
                "transaction": transaction,
            }[swap_level]
            target.rename(parked)
            shutil.copytree(parked, target)
            sentinel = transaction / "attacker-replacement-sentinel"
            sentinel.write_text("must not be removed\n", encoding="utf-8")
            replacement_sentinel.append(sentinel)
        return inventory

    monkeypatch.setattr(
        journals,
        "discover_mutation_journals",
        swap_after_complete_inventory,
    )
    report = journals.recover_mutation_journals(corpus, repo)

    assert report.recovered == ()
    assert report.failures
    assert (repo / "a.py").read_text(encoding="utf-8") == crashed
    assert replacement_sentinel[0].exists()
    if swap_level == "run":
        parked_manifest = parked / "mutations" / transaction.name / "manifest.json"
    elif swap_level == "durable_root":
        parked_manifest = parked / transaction.name / "manifest.json"
    elif swap_level == "transaction":
        parked_manifest = parked / "manifest.json"
    else:
        parked_manifest = parked
    assert parked_manifest.read_bytes() == original_manifest


def test_recovery_cannot_install_replacement_journal_index_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "victim" / "mutations",
        slice_id="victim-index-injection",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    crashed = "# crashed index bytes\n"
    (repo / "a.py").write_text(crashed, encoding="utf-8")
    _simulate_process_death(journal)

    transaction = journal.journal_dir
    parked = tmp_path / "parked-transaction"
    original_index = (transaction / "index").read_bytes()
    original_discover = journals.discover_mutation_journals
    injected: list[bytes] = []

    def swap_after_complete_inventory(bound_corpus, **kwargs):
        inventory = original_discover(bound_corpus, **kwargs)
        if injected:
            return inventory
        transaction.rename(parked)
        shutil.copytree(parked, transaction)
        clone = tmp_path / "alternate-clone"
        subprocess.run(
            ["git", "clone", "-q", "--no-hardlinks", str(repo), str(clone)],
            check=True,
        )
        payload = (clone / ".git" / "index").read_bytes()
        assert hashlib.sha256(payload).digest() != hashlib.sha256(
            original_index
        ).digest()
        (transaction / "index").write_bytes(payload)
        injected.append(payload)
        return inventory

    monkeypatch.setattr(
        journals,
        "discover_mutation_journals",
        swap_after_complete_inventory,
    )
    report = journals.recover_mutation_journals(corpus, repo)

    assert report.recovered == ()
    assert report.failures
    assert (repo / "a.py").read_text(encoding="utf-8") == crashed
    assert (repo / ".git" / "index").read_bytes() != injected[0]
    assert parked.exists()
    assert transaction.exists()


def _same_workspace_nested_mutation_family(tmp_path: Path):
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    repo, corpus = _setup(tmp_path)
    parent = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "absent-family" / "mutations",
        slice_id="absent-parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=parent.journal_dir / "nested_mutations",
        slice_id="absent-child",
        scope=("a.py",),
        sequence=2,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    (repo / "a.py").write_text("# accepted family bytes\n", encoding="utf-8")
    return repo, corpus, parent, child


def _assert_absent_family_rolls_back_atomically(repo, corpus, parent, child):
    from crucible.workflows._mutation_journal import (
        _unregister_live,
        recover_mutation_journals,
    )

    registry = Path(parent._manifest["registry_directory"])
    _unregister_live(child.transaction_id)
    _unregister_live(parent.transaction_id)
    report = recover_mutation_journals(corpus, repo)
    assert report.failures == ()
    assert set(report.recovered) == {"absent-parent", "absent-child"}
    assert (repo / "a.py").read_text(encoding="utf-8") == "# baseline a.py\n"
    assert not child.journal_dir.exists()
    assert not parent.journal_dir.exists()
    assert list(registry.iterdir()) == []


def test_absent_event_rolls_back_whole_fully_prepared_family(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import (
        prepare_mutation_lifecycle_settlement,
    )

    repo, corpus, parent, child = _same_workspace_nested_mutation_family(tmp_path)
    prepare_mutation_lifecycle_settlement(
        corpus,
        (parent, child),
        kind="landed",
        reason="definitively absent family",
        run_id="absent-family-run",
    )
    _assert_absent_family_rolls_back_atomically(repo, corpus, parent, child)


def test_absent_event_rolls_back_family_after_first_prepare_marker_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        prepare_mutation_lifecycle_settlement,
    )

    repo, corpus, parent, child = _same_workspace_nested_mutation_family(tmp_path)
    original = WorkspaceMutationJournal.prepare_lifecycle_acceptance
    calls: list[str] = []

    def crash_before_second_marker(current, *args, **kwargs):
        calls.append(current.transaction_id)
        if len(calls) == 2:
            raise RuntimeError("simulated crash before second family marker")
        return original(current, *args, **kwargs)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "prepare_lifecycle_acceptance",
        crash_before_second_marker,
    )
    with pytest.raises(RuntimeError, match="before second family marker"):
        prepare_mutation_lifecycle_settlement(
            corpus,
            (parent, child),
            kind="landed",
            reason="partial absent family",
            run_id="partial-absent-family-run",
        )
    _assert_absent_family_rolls_back_atomically(repo, corpus, parent, child)


def test_absent_event_family_rollback_resumes_after_first_member_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _unregister_live,
        prepare_mutation_lifecycle_settlement,
        recover_mutation_journals,
    )

    repo, corpus, parent, child = _same_workspace_nested_mutation_family(tmp_path)
    prepare_mutation_lifecycle_settlement(
        corpus,
        (parent, child),
        kind="landed",
        reason="interrupted absent family rollback",
        run_id="interrupted-absent-family-run",
    )
    registry = Path(parent._manifest["registry_directory"])
    _unregister_live(child.transaction_id)
    _unregister_live(parent.transaction_id)

    class SimulatedPowerLoss(BaseException):
        pass

    original = WorkspaceMutationJournal.restore
    restored: list[str] = []

    def die_after_first_member(current, *args, **kwargs):
        result = original(current, *args, **kwargs)
        if result and not restored:
            restored.append(current.transaction_id)
            raise SimulatedPowerLoss("after first family member terminal")
        return result

    monkeypatch.setattr(WorkspaceMutationJournal, "restore", die_after_first_member)
    with pytest.raises(SimulatedPowerLoss, match="first family member"):
        recover_mutation_journals(corpus, repo)
    monkeypatch.setattr(WorkspaceMutationJournal, "restore", original)
    assert child.journal_dir.exists()
    assert json.loads(
        (child.journal_dir / "manifest.json").read_text(encoding="utf-8")
    )["state"] == "rolled_back"

    restarted = recover_mutation_journals(corpus, repo)
    assert restarted.failures == ()
    assert not parent.journal_dir.exists()
    assert list(registry.iterdir()) == []


def test_absent_event_family_resumes_after_parent_nested_recursion_death(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _unregister_live,
        prepare_mutation_lifecycle_settlement,
        recover_mutation_journals,
    )

    repo, corpus, parent, child = _same_workspace_nested_mutation_family(tmp_path)
    prepare_mutation_lifecycle_settlement(
        corpus,
        (parent, child),
        kind="landed",
        reason="nested recursion crash family",
        run_id="nested-recursion-crash-run",
    )
    registry = Path(parent._manifest["registry_directory"])
    _unregister_live(child.transaction_id)
    _unregister_live(parent.transaction_id)

    def crash_after_parent_reenters_terminal_child() -> None:
        import os

        original = WorkspaceMutationJournal.restore
        child_calls = 0

        def exit_after_second_child_restore(current, *args, **kwargs):
            nonlocal child_calls
            result = original(current, *args, **kwargs)
            if current.transaction_id == child.transaction_id:
                child_calls += 1
                if child_calls == 2:
                    os._exit(86)
            return result

        WorkspaceMutationJournal.restore = exit_after_second_child_restore
        recover_mutation_journals(corpus, repo)
        os._exit(87)

    process = multiprocessing.get_context("fork").Process(
        target=crash_after_parent_reenters_terminal_child
    )
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("nested-recursion crash probe hung")
    assert process.exitcode == 86
    assert child.journal_dir.exists()
    assert parent.journal_dir.exists()

    restarted = recover_mutation_journals(corpus, repo)
    assert restarted.failures == ()
    assert (repo / "a.py").read_text(encoding="utf-8") == "# baseline a.py\n"
    assert not child.journal_dir.exists()
    assert not parent.journal_dir.exists()
    assert list(registry.iterdir()) == []


def _open_directory(path: Path) -> int:
    return os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )


@pytest.mark.parametrize("entry_kind", ("regular", "directory"))
def test_bound_cleanup_rejects_replacement_between_stat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._mutation_journal import MutationJournalError

    parent = tmp_path / "parent"
    target = parent / "transaction"
    target.mkdir(parents=True)
    entry = target / ("manifest.json" if entry_kind == "regular" else "snapshots")
    parked = tmp_path / f"parked-original-{entry_kind}"
    if entry_kind == "regular":
        entry.write_text("original journal manifest\n", encoding="utf-8")
    else:
        entry.mkdir()
        (entry / "owned").write_text("owned\n", encoding="utf-8")

    parent_fd = _open_directory(parent)
    target_fd = _open_directory(target)
    expected = os.fstat(target_fd)
    original_open = journals.os.open
    swapped = [False]

    def swap_before_entry_open(path, flags, mode=0o777, *, dir_fd=None):
        is_target = (
            not swapped[0]
            and path == entry.name
            and dir_fd == target_fd
            and (
                entry_kind == "regular"
                or flags & getattr(os, "O_DIRECTORY", 0)
            )
        )
        if is_target:
            entry.rename(parked)
            if entry_kind == "regular":
                entry.write_text("replacement\n", encoding="utf-8")
            else:
                entry.mkdir()
                (entry / "replacement").write_text(
                    "replacement\n",
                    encoding="utf-8",
                )
            swapped[0] = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(journals.os, "open", swap_before_entry_open)
    try:
        with pytest.raises(MutationJournalError, match="changed while opening"):
            journals._remove_bound_directory_tree(
                parent_fd,
                target.name,
                target_fd,
                expected,
            )
    finally:
        os.close(target_fd)
        os.close(parent_fd)

    assert swapped == [True]
    if entry_kind == "regular":
        assert parked.read_text(encoding="utf-8") == "original journal manifest\n"
        assert entry.read_text(encoding="utf-8") == "replacement\n"
    else:
        assert (parked / "owned").read_text(encoding="utf-8") == "owned\n"
        assert (
            entry / "replacement"
        ).read_text(encoding="utf-8") == "replacement\n"


def _wait_for_expected_child_exit(process_id: int, code: int) -> None:
    waited, status = os.waitpid(process_id, 0)
    assert waited == process_id
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == code


@pytest.mark.parametrize(
    ("entry_kind", "exit_code"),
    (("regular", 91), ("directory", 92)),
)
def test_bound_cleanup_resumes_after_quarantine_process_death(
    tmp_path: Path,
    entry_kind: str,
    exit_code: int,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    parent = tmp_path / "parent"
    target = parent / "transaction"
    target.mkdir(parents=True)
    entry = target / ("manifest.json" if entry_kind == "regular" else "snapshots")
    if entry_kind == "regular":
        entry.write_text("journal manifest\n", encoding="utf-8")
    else:
        entry.mkdir()
        (entry / "owned").write_text("owned\n", encoding="utf-8")
    parent_fd = _open_directory(parent)
    target_fd = _open_directory(target)
    expected = os.fstat(target_fd)

    process_id = os.fork()
    if process_id == 0:
        original_rename = journals.os.rename

        def die_after_quarantine(
            source,
            destination,
            *,
            src_dir_fd=None,
            dst_dir_fd=None,
        ):
            result = original_rename(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )
            if source == entry.name and str(destination).startswith(
                ".crucible-cleanup-"
            ):
                os._exit(exit_code)
            return result

        journals.os.rename = die_after_quarantine
        journals._remove_bound_directory_tree(
            parent_fd,
            target.name,
            target_fd,
            expected,
        )
        os._exit(99)

    _wait_for_expected_child_exit(process_id, exit_code)
    assert target.is_dir()
    assert not entry.exists()
    quarantined = tuple(target.iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].name.startswith(".crucible-cleanup-")
    if entry_kind == "directory":
        assert (
            quarantined[0] / "owned"
        ).read_text(encoding="utf-8") == "owned\n"
    try:
        journals._remove_bound_directory_tree(
            parent_fd,
            target.name,
            target_fd,
            expected,
        )
    finally:
        os.close(target_fd)
        os.close(parent_fd)
    assert not target.exists()


def test_bound_cleanup_resumes_after_quarantine_reservation_process_death(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    parent = tmp_path / "parent"
    target = parent / "transaction"
    target.mkdir(parents=True)
    victim = target / "manifest.json"
    victim.write_text("journal manifest\n", encoding="utf-8")
    parent_fd = _open_directory(parent)
    target_fd = _open_directory(target)
    expected = os.fstat(target_fd)

    process_id = os.fork()
    if process_id == 0:
        original_open = journals.os.open

        def die_after_reservation(path, flags, mode=0o777, *, dir_fd=None):
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            if str(path).startswith(".crucible-cleanup-") and flags & os.O_EXCL:
                os._exit(93)
            return descriptor

        journals.os.open = die_after_reservation
        journals._remove_bound_directory_tree(
            parent_fd,
            target.name,
            target_fd,
            expected,
        )
        os._exit(99)

    _wait_for_expected_child_exit(process_id, 93)
    assert victim.read_text(encoding="utf-8") == "journal manifest\n"
    assert len(tuple(target.iterdir())) == 2
    try:
        journals._remove_bound_directory_tree(
            parent_fd,
            target.name,
            target_fd,
            expected,
        )
    finally:
        os.close(target_fd)
        os.close(parent_fd)
    assert not target.exists()


def test_bound_cleanup_removes_ordinary_nested_tree(tmp_path: Path) -> None:
    from crucible.workflows import _mutation_journal as journals

    parent = tmp_path / "parent"
    target = parent / "transaction"
    nested = target / "one" / "two"
    nested.mkdir(parents=True)
    (target / "manifest.json").write_text("manifest\n", encoding="utf-8")
    (nested / "payload").write_text("payload\n", encoding="utf-8")
    parent_fd = _open_directory(parent)
    target_fd = _open_directory(target)
    expected = os.fstat(target_fd)
    try:
        journals._remove_bound_directory_tree(
            parent_fd,
            target.name,
            target_fd,
            expected,
        )
    finally:
        os.close(target_fd)
        os.close(parent_fd)
    assert not target.exists()


@pytest.mark.parametrize("entry_kind", ("symlink", "fifo"))
def test_bound_cleanup_rejects_special_entries_without_following(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._mutation_journal import MutationJournalError

    parent = tmp_path / "parent"
    target = parent / "transaction"
    target.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("protected outside bytes\n", encoding="utf-8")
    entry = target / "unsafe"
    if entry_kind == "symlink":
        entry.symlink_to(outside)
    else:
        os.mkfifo(entry)
    parent_fd = _open_directory(parent)
    target_fd = _open_directory(target)
    expected = os.fstat(target_fd)
    try:
        with pytest.raises(MutationJournalError, match="special filesystem"):
            journals._remove_bound_directory_tree(
                parent_fd,
                target.name,
                target_fd,
                expected,
            )
    finally:
        os.close(target_fd)
        os.close(parent_fd)
    assert os.path.lexists(entry)
    assert outside.read_text(encoding="utf-8") == "protected outside bytes\n"


def test_bound_cleanup_enforces_entry_budget_before_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._mutation_journal import MutationJournalError

    parent = tmp_path / "parent"
    target = parent / "transaction"
    target.mkdir(parents=True)
    (target / "a").write_text("a\n", encoding="utf-8")
    (target / "b").write_text("b\n", encoding="utf-8")
    monkeypatch.setattr(
        journals,
        "_MUTATION_JOURNAL_MAX_DISCOVERY_ARTIFACT_PATHS",
        1,
    )
    parent_fd = _open_directory(parent)
    target_fd = _open_directory(target)
    expected = os.fstat(target_fd)
    try:
        with pytest.raises(MutationJournalError, match="entry ceiling"):
            journals._remove_bound_directory_tree(
                parent_fd,
                target.name,
                target_fd,
                expected,
            )
    finally:
        os.close(target_fd)
        os.close(parent_fd)
    assert (target / "a").read_text(encoding="utf-8") == "a\n"
    assert (target / "b").read_text(encoding="utf-8") == "b\n"


@pytest.mark.parametrize(
    ("crash_point", "exit_code"),
    (
        ("after_receipt", 101),
        ("after_tombstone_rename", 102),
        ("after_manifest_delete", 103),
        ("after_tombstone_delete", 104),
    ),
)
def test_terminal_cleanup_receipt_resumes_real_process_death(
    tmp_path: Path,
    crash_point: str,
    exit_code: int,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    baseline = (repo / "a.py").read_bytes()
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / f"terminal-{crash_point}" / "mutations"
        ),
        slice_id=f"terminal-{crash_point}",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# interrupted mutation\n", encoding="utf-8")

    process_id = os.fork()
    if process_id == 0:
        if crash_point == "after_receipt":
            def die_before_marker_removal(self, **_kwargs):
                os._exit(exit_code)

            journals.WorkspaceMutationJournal._remove_registry_marker = (
                die_before_marker_removal
            )
        elif crash_point == "after_tombstone_rename":
            original_rename = journals.os.rename

            def die_after_tombstone_rename(
                source,
                destination,
                *,
                src_dir_fd=None,
                dst_dir_fd=None,
            ):
                result = original_rename(
                    source,
                    destination,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                )
                if str(destination).startswith(
                    journals._TERMINAL_TOMBSTONE_PREFIX
                ):
                    os._exit(exit_code)
                return result

            journals.os.rename = die_after_tombstone_rename
        elif crash_point == "after_manifest_delete":
            original_remove_entry = journals._remove_bound_regular_entry

            def die_after_manifest_delete(directory_fd, name, **kwargs):
                result = original_remove_entry(directory_fd, name, **kwargs)
                if name == "manifest.json":
                    os._exit(exit_code)
                return result

            journals._remove_bound_regular_entry = die_after_manifest_delete
        else:
            original_remove_tree = journals._remove_bound_directory_tree

            def die_after_tombstone_delete(
                parent_fd,
                name,
                directory_fd,
                expected,
            ):
                result = original_remove_tree(
                    parent_fd,
                    name,
                    directory_fd,
                    expected,
                )
                if str(name).startswith(journals._TERMINAL_TOMBSTONE_PREFIX):
                    os._exit(exit_code)
                return result

            journals._remove_bound_directory_tree = die_after_tombstone_delete
        journal.restore()
        os._exit(99)

    _wait_for_expected_child_exit(process_id, exit_code)
    _simulate_process_death(journal)
    receipt = (
        journal.journal_dir.parent
        / journals._terminal_receipt_name(journal.transaction_id)
    )
    assert receipt.is_file()
    assert (repo / "a.py").read_bytes() == baseline

    report = journals.recover_mutation_journals(corpus, repo)

    assert report.failures == ()
    assert (repo / "a.py").read_bytes() == baseline
    assert not journal.journal_dir.exists()
    assert not tuple(journal.journal_dir.parent.glob(
        f"{journals._TERMINAL_TOMBSTONE_PREFIX}{journal.transaction_id}-*"
    ))
    assert not receipt.exists()
    assert not journals._registry_marker_path(
        repo,
        journal.transaction_id,
    ).exists()


def test_terminal_cleanup_receipt_collision_preserves_hostile_final_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final-name race cannot replace foreign bytes or disarm rollback."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    baseline = (repo / "a.py").read_bytes()
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "terminal-collision" / "mutations"
        ),
        slice_id="terminal-collision",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# interrupted mutation\n", encoding="utf-8")
    receipt_name = journals._terminal_receipt_name(journal.transaction_id)
    receipt_path = journal.journal_dir.parent / receipt_name
    hostile_payload = b"foreign terminal authority\n"
    hostile_identity: list[tuple[int, int]] = []
    real_link = journals._link_bound_unnamed_noreplace

    def collide_before_link(
        temporary_fd,
        parent_fd,
        final_name,
        *,
        label,
    ):
        assert final_name == receipt_name
        hostile_fd = os.open(
            final_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        try:
            journals._write_all_bound(hostile_fd, hostile_payload)
            os.fsync(hostile_fd)
            metadata = os.fstat(hostile_fd)
            hostile_identity.append((metadata.st_dev, metadata.st_ino))
        finally:
            os.close(hostile_fd)
        journals._fsync_bound_directory(parent_fd)
        return real_link(
            temporary_fd,
            parent_fd,
            final_name,
            label=label,
        )

    monkeypatch.setattr(
        journals,
        "_link_bound_unnamed_noreplace",
        collide_before_link,
    )

    assert not journal.restore()

    receipt_metadata = receipt_path.lstat()
    assert hostile_identity == [
        (receipt_metadata.st_dev, receipt_metadata.st_ino)
    ]
    assert receipt_path.read_bytes() == hostile_payload
    assert journal.journal_dir.is_dir()
    assert (journal.journal_dir / "manifest.json").is_file()
    assert (repo / "a.py").read_bytes() == baseline
    assert journals._registry_marker_path(
        repo,
        journal.transaction_id,
    ).is_file()


def test_terminal_cleanup_receipt_rejects_replaced_parent_before_discovery(
    tmp_path: Path,
) -> None:
    """Copied exact entries cannot transfer a receipt to a foreign parent."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    baseline = (repo / "a.py").read_bytes()
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "terminal-parent-swap" / "mutations"
        ),
        slice_id="terminal-parent-swap",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# interrupted mutation\n", encoding="utf-8")
    exit_code = 106
    process_id = os.fork()
    if process_id == 0:
        def die_before_marker_removal(self, **_kwargs):
            os._exit(exit_code)

        journals.WorkspaceMutationJournal._remove_registry_marker = (
            die_before_marker_removal
        )
        journal.restore()
        os._exit(99)

    _wait_for_expected_child_exit(process_id, exit_code)
    _simulate_process_death(journal)
    parent = journal.journal_dir.parent
    receipt = parent / journals._terminal_receipt_name(
        journal.transaction_id
    )
    assert receipt.is_file()
    assert journal.journal_dir.is_dir()

    parked_parent = tmp_path / "parked-original-parent"
    parent.rename(parked_parent)
    replacement_parent = tmp_path / "foreign-parent"
    replacement_parent.mkdir()
    for entry in tuple(parked_parent.iterdir()):
        entry.rename(replacement_parent / entry.name)
    sentinel_payload = b"foreign parent must survive\n"
    (replacement_parent / "sentinel.bin").write_bytes(sentinel_payload)
    replacement_parent.rename(parent)
    replacement_receipt = parent / receipt.name
    replacement_journal = parent / journal.journal_dir.name

    report = journals.recover_mutation_journals(corpus, repo)

    assert report.recovered == ()
    assert any(
        "terminal cleanup receipt parent authority changed" in failure
        for failure in report.failures
    ), report.failures
    assert (parent / "sentinel.bin").read_bytes() == sentinel_payload
    assert replacement_receipt.is_file()
    assert replacement_journal.is_dir()
    assert (replacement_journal / "manifest.json").is_file()
    assert (repo / "a.py").read_bytes() == baseline
    assert journals._registry_marker_path(
        repo,
        journal.transaction_id,
    ).is_file()


def test_forged_terminal_receipt_cannot_erase_live_tombstoned_journal(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "forged-terminal" / "mutations"
        ),
        slice_id="forged-terminal",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    attacked = b"# active unaccepted mutation\n"
    (repo / "a.py").write_bytes(attacked)
    original = journal.journal_dir
    metadata = original.lstat()
    journal_fd = _open_directory(original)
    try:
        proof_digests = journals._terminal_cleanup_proof_digests(
            journal_fd,
            journal._manifest,
        )
        journal_inventory = journals._terminal_cleanup_journal_inventory(
            journal_fd
        )
    finally:
        os.close(journal_fd)
    forged_manifest = dict(journal._manifest)
    forged_manifest["state"] = "rolled_back"
    manifest_payload = json.dumps(
        forged_manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    registry_payload = (original / "registry_marker").read_bytes()
    tombstone_name = journals._terminal_tombstone_name(
        journal.transaction_id,
        "0" * 64,
    )
    tombstone = original.parent / tombstone_name
    original.rename(tombstone)
    (tombstone / "manifest.json").unlink()
    receipt_record = {
        "schema": journals._TERMINAL_CLEANUP_SCHEMA,
        "transaction_id": journal.transaction_id,
        "original_name": original.name,
        "tombstone_name": tombstone_name,
        "parent_device": original.parent.lstat().st_dev,
        "parent_inode": original.parent.lstat().st_ino,
        "journal_device": metadata.st_dev,
        "journal_inode": metadata.st_ino,
        "manifest_b64": base64.b64encode(manifest_payload).decode("ascii"),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "registry_marker_b64": base64.b64encode(registry_payload).decode(
            "ascii"
        ),
        "registry_marker_sha256": hashlib.sha256(registry_payload).hexdigest(),
        "proof_digests": proof_digests,
        "journal_inventory": [list(item) for item in journal_inventory],
    }
    receipt = original.parent / journals._terminal_receipt_name(
        journal.transaction_id
    )
    receipt.write_text(
        json.dumps(receipt_record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    _simulate_process_death(journal)

    report = journals.recover_mutation_journals(corpus, repo)

    assert report.recovered == ()
    assert report.failures
    assert (
        "disposition failed" in report.failures[0]
        or report.failures[0].endswith(
            "MutationJournalError: terminal cleanup receipt fields changed"
        )
    )
    assert tombstone.is_dir()
    assert receipt.is_file()
    assert (repo / "a.py").read_bytes() == attacked
    assert journals._registry_marker_path(
        repo,
        journal.transaction_id,
    ).is_file()


def test_forged_targetless_receipt_cannot_erase_live_journal(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "forged-targetless" / "mutations"
        ),
        slice_id="forged-targetless",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    attacked = b"# still-live mutation\n"
    (repo / "a.py").write_bytes(attacked)
    journal_fd = _open_directory(journal.journal_dir)
    try:
        proof_digests = journals._terminal_cleanup_proof_digests(
            journal_fd,
            journal._manifest,
        )
        inventory = journals._terminal_cleanup_journal_inventory(journal_fd)
    finally:
        os.close(journal_fd)
    manifest = dict(journal._manifest)
    manifest["state"] = "rolled_back"
    manifest_payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    registry_record = json.loads(
        (journal.journal_dir / "registry_marker").read_bytes()
    )
    absent_original = "forged-absent-journal"
    registry_record["journal_dir"] = str(
        journal.journal_dir.parent / absent_original
    )
    registry_payload = json.dumps(
        registry_record,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt_record = {
        "schema": journals._TERMINAL_CLEANUP_SCHEMA,
        "transaction_id": journal.transaction_id,
        "original_name": absent_original,
        "tombstone_name": journals._terminal_tombstone_name(
            journal.transaction_id,
            "1" * 64,
        ),
        "parent_device": journal.journal_dir.parent.lstat().st_dev,
        "parent_inode": journal.journal_dir.parent.lstat().st_ino,
        "journal_device": journal.journal_dir.lstat().st_dev,
        "journal_inode": journal.journal_dir.lstat().st_ino,
        "manifest_b64": base64.b64encode(manifest_payload).decode("ascii"),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "registry_marker_b64": base64.b64encode(registry_payload).decode(
            "ascii"
        ),
        "registry_marker_sha256": hashlib.sha256(registry_payload).hexdigest(),
        "proof_digests": proof_digests,
        "journal_inventory": [list(item) for item in inventory],
    }
    receipt = journal.journal_dir.parent / journals._terminal_receipt_name(
        journal.transaction_id
    )
    receipt.write_text(
        json.dumps(receipt_record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    _simulate_process_death(journal)

    report = journals.recover_mutation_journals(corpus, repo)

    assert report.recovered == ()
    assert report.failures
    assert journal.journal_dir.is_dir()
    assert receipt.is_file()
    assert (repo / "a.py").read_bytes() == attacked
    assert journals._registry_marker_path(
        repo,
        journal.transaction_id,
    ).is_file()


def test_receipt_named_replacement_tombstone_contents_are_not_deleted(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "recycled-tombstone" / "mutations"
        ),
        slice_id="recycled-tombstone",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# interrupted mutation\n", encoding="utf-8")
    process_id = os.fork()
    if process_id == 0:
        original_remove_tree = journals._remove_bound_directory_tree

        def die_after_tombstone_delete(
            parent_fd,
            name,
            directory_fd,
            expected,
        ):
            result = original_remove_tree(
                parent_fd,
                name,
                directory_fd,
                expected,
            )
            if str(name).startswith(journals._TERMINAL_TOMBSTONE_PREFIX):
                os._exit(105)
            return result

        journals._remove_bound_directory_tree = die_after_tombstone_delete
        journal.restore()
        os._exit(99)
    _wait_for_expected_child_exit(process_id, 105)
    _simulate_process_death(journal)

    receipt = journal.journal_dir.parent / journals._terminal_receipt_name(
        journal.transaction_id
    )
    record = json.loads(receipt.read_bytes())
    tombstone = journal.journal_dir.parent / str(record["tombstone_name"])
    tombstone.mkdir()
    unrelated = tombstone / "unrelated.txt"
    unrelated.write_text("must survive\n", encoding="utf-8")
    replacement_metadata = tombstone.lstat()
    record["journal_device"] = replacement_metadata.st_dev
    record["journal_inode"] = replacement_metadata.st_ino
    receipt.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    report = journals.recover_mutation_journals(corpus, repo)

    assert report.recovered == ()
    assert report.failures
    assert unrelated.read_text(encoding="utf-8") == "must survive\n"
    assert receipt.is_file()
