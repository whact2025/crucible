# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Focused races for coordination that remains reachable above the checkout."""

from __future__ import annotations

import os
from pathlib import Path
import multiprocessing
import stat
import threading
import time

import pytest

from .test_mutation_journal_writer_epoch_integration import _begin_single
from .test_workflows_tier_mutation_transaction import _simulate_process_death
from .test_workflows_tier_mutation_transaction import (
    _add_clean_submodule,
    _git,
    _setup,
)


pytestmark = pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "O_PATH"),
    reason="external workspace coordination requires POSIX O_PATH/flock",
)


def _writer_process(
    repo: str,
    entered,
    release,
    results,
    *,
    chmod_zero: bool,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    try:
        with journals.workspace_mutation_lock(Path(repo)):
            if chmod_zero:
                Path(repo).chmod(0)
            entered.set()
            if not release.wait(20):
                raise TimeoutError("writer release was not signaled")
        results.put(("writer", "ok"))
    except BaseException as exc:  # pragma: no cover - reported in parent
        results.put(("writer", f"{type(exc).__name__}: {exc}"))


def _recovery_process(
    spec_dir: str,
    repo: str,
    results,
    pause=None,
    release=None,
) -> None:
    from assay.corpus.store import Corpus
    from crucible.workflows import _mutation_journal as journals

    if pause is not None and release is not None:
        original = (
            journals.WorkspaceMutationJournal._prepare_workspace_root_access
        )

        def paused_prepare(self):
            pause.set()
            if not release.wait(20):
                raise TimeoutError("recovery release was not signaled")
            return original(self)

        journals.WorkspaceMutationJournal._prepare_workspace_root_access = (
            paused_prepare
        )
    try:
        report = journals.recover_mutation_journals(
            Corpus(Path(spec_dir)),
            Path(repo),
        )
        results.put(("recovery", report.failures, report.recovered))
    except BaseException as exc:  # pragma: no cover - reported in parent
        results.put(("recovery", f"{type(exc).__name__}: {exc}"))


def _opposing_lock_process(
    first: str,
    second: str,
    ready,
    start,
    active,
    maximum,
    counter_lock,
    results,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    try:
        with journals.workspace_mutation_lock(Path(first)):
            ready.set()
            if not start.wait(20):
                raise TimeoutError("opposing lock start was not signaled")
            try:
                with journals.workspace_mutation_lock(Path(second)):
                    with counter_lock:
                        active.value += 1
                        maximum.value = max(maximum.value, active.value)
                    time.sleep(0.1)
                    with counter_lock:
                        active.value -= 1
                results.put(("ok", ""))
            except journals.MutationJournalError as exc:
                results.put(("busy", str(exc)))
    except BaseException as exc:  # pragma: no cover - reported in parent
        results.put(("error", f"{type(exc).__name__}: {exc}"))


def test_held_writer_blocks_root_repair_before_chmod(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, journal = _begin_single(
        tmp_path,
        run_id="external-writer-blocks-repair",
    )
    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    writer_entered = threading.Event()
    release_writer = threading.Event()
    writer_errors: list[BaseException] = []
    restore_result: list[bool] = []

    def hold_writer() -> None:
        try:
            with journals.workspace_mutation_lock(repo):
                writer_entered.set()
                assert release_writer.wait(10)
        except BaseException as exc:  # pragma: no cover - reported in parent
            writer_errors.append(exc)

    def restore() -> None:
        restore_result.append(journal.restore())

    holder = threading.Thread(target=hold_writer, daemon=True)
    recovery = threading.Thread(target=restore, daemon=True)
    try:
        holder.start()
        assert writer_entered.wait(10)
        repo.chmod(0)
        recovery.start()
        recovery.join(0.25)
        assert recovery.is_alive()
        assert stat.S_IMODE(repo.stat().st_mode) == 0

        release_writer.set()
        holder.join(10)
        recovery.join(20)
        assert not holder.is_alive()
        assert not recovery.is_alive()
        assert writer_errors == []
        assert restore_result == [True]
        assert stat.S_IMODE(repo.stat().st_mode) == baseline_mode
    finally:
        release_writer.set()
        if repo.exists():
            repo.chmod(baseline_mode)


def test_recovery_holds_external_lock_from_epoch_proof_through_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, journal = _begin_single(
        tmp_path,
        run_id="external-recovery-blocks-writer",
    )
    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    recovery_paused = threading.Event()
    continue_recovery = threading.Event()
    writer_entered = threading.Event()
    restore_result: list[bool] = []
    errors: list[BaseException] = []
    original_prepare = (
        journals.WorkspaceMutationJournal._prepare_workspace_root_access
    )

    def paused_prepare(self):
        if self.transaction_id == journal.transaction_id:
            recovery_paused.set()
            assert continue_recovery.wait(10)
        return original_prepare(self)

    monkeypatch.setattr(
        journals.WorkspaceMutationJournal,
        "_prepare_workspace_root_access",
        paused_prepare,
    )

    def restore() -> None:
        try:
            restore_result.append(journal.restore())
        except BaseException as exc:  # pragma: no cover - reported in parent
            errors.append(exc)

    def writer() -> None:
        try:
            with journals.workspace_mutation_lock(repo):
                writer_entered.set()
        except BaseException as exc:  # pragma: no cover - reported in parent
            errors.append(exc)

    recovery = threading.Thread(target=restore, daemon=True)
    fresh_writer = threading.Thread(target=writer, daemon=True)
    try:
        recovery.start()
        assert recovery_paused.wait(10)
        fresh_writer.start()
        fresh_writer.join(0.25)
        assert fresh_writer.is_alive()
        assert not writer_entered.is_set()
        assert stat.S_IMODE(repo.stat().st_mode) == baseline_mode

        continue_recovery.set()
        recovery.join(20)
        fresh_writer.join(20)
        assert not recovery.is_alive()
        assert not fresh_writer.is_alive()
        assert errors == []
        assert restore_result == [True]
        assert writer_entered.is_set()
        assert stat.S_IMODE(repo.stat().st_mode) == baseline_mode
    finally:
        continue_recovery.set()
        if repo.exists():
            repo.chmod(baseline_mode)


def test_cross_process_writer_blocks_recovery_before_root_repair(
    tmp_path: Path,
) -> None:
    repo, corpus, journal = _begin_single(
        tmp_path,
        run_id="external-process-writer-blocks-recovery",
    )
    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    _simulate_process_death(journal)
    context = multiprocessing.get_context("fork")
    writer_entered = context.Event()
    release_writer = context.Event()
    results = context.Queue()
    writer = context.Process(
        target=_writer_process,
        args=(str(repo), writer_entered, release_writer, results),
        kwargs={"chmod_zero": True},
    )
    recovery = context.Process(
        target=_recovery_process,
        args=(str(corpus.spec_dir), str(repo), results),
    )
    try:
        writer.start()
        assert writer_entered.wait(10)
        recovery.start()
        recovery.join(0.5)
        assert recovery.is_alive()
        assert stat.S_IMODE(repo.stat().st_mode) == 0
        release_writer.set()
        writer.join(20)
        recovery.join(30)
        assert writer.exitcode == 0
        assert recovery.exitcode == 0
        records = [results.get(timeout=5) for _ in range(2)]
        assert {record[0] for record in records} == {"writer", "recovery"}
        assert next(record for record in records if record[0] == "writer") == (
            "writer",
            "ok",
        )
        recovery_record = next(
            record for record in records if record[0] == "recovery"
        )
        assert recovery_record[1] == ()
        assert stat.S_IMODE(repo.stat().st_mode) == baseline_mode
    finally:
        release_writer.set()
        for process in (writer, recovery):
            if process.is_alive():
                process.terminate()
                process.join(5)
        if repo.exists():
            repo.chmod(baseline_mode)


def test_cross_process_recovery_blocks_fresh_writer_after_epoch_proof(
    tmp_path: Path,
) -> None:
    repo, corpus, journal = _begin_single(
        tmp_path,
        run_id="external-process-recovery-blocks-writer",
    )
    _simulate_process_death(journal)
    context = multiprocessing.get_context("fork")
    recovery_paused = context.Event()
    release_recovery = context.Event()
    writer_entered = context.Event()
    release_writer = context.Event()
    results = context.Queue()
    recovery = context.Process(
        target=_recovery_process,
        args=(
            str(corpus.spec_dir),
            str(repo),
            results,
            recovery_paused,
            release_recovery,
        ),
    )
    writer = context.Process(
        target=_writer_process,
        args=(str(repo), writer_entered, release_writer, results),
        kwargs={"chmod_zero": False},
    )
    try:
        recovery.start()
        assert recovery_paused.wait(15)
        writer.start()
        writer.join(0.5)
        assert writer.is_alive()
        assert not writer_entered.is_set()
        release_recovery.set()
        recovery.join(30)
        assert recovery.exitcode == 0
        assert writer_entered.wait(15)
        release_writer.set()
        writer.join(20)
        assert writer.exitcode == 0
        records = [results.get(timeout=5) for _ in range(2)]
        assert {record[0] for record in records} == {"writer", "recovery"}
        recovery_record = next(
            record for record in records if record[0] == "recovery"
        )
        assert recovery_record[1] == ()
    finally:
        release_recovery.set()
        release_writer.set()
        for process in (recovery, writer):
            if process.is_alive():
                process.terminate()
                process.join(5)


def test_opposing_independent_lock_orders_fail_promptly_without_overlap(
    tmp_path: Path,
) -> None:
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    first_parent.mkdir()
    second_parent.mkdir()
    first, _first_corpus = _setup(first_parent)
    second, _second_corpus = _setup(second_parent)
    context = multiprocessing.get_context("fork")
    first_ready = context.Event()
    second_ready = context.Event()
    start = context.Event()
    active = context.Value("i", 0)
    maximum = context.Value("i", 0)
    counter_lock = context.Lock()
    results = context.Queue()
    processes = (
        context.Process(
            target=_opposing_lock_process,
            args=(
                str(first),
                str(second),
                first_ready,
                start,
                active,
                maximum,
                counter_lock,
                results,
            ),
        ),
        context.Process(
            target=_opposing_lock_process,
            args=(
                str(second),
                str(first),
                second_ready,
                start,
                active,
                maximum,
                counter_lock,
                results,
            ),
        ),
    )
    try:
        for process in processes:
            process.start()
        assert first_ready.wait(15)
        assert second_ready.wait(15)
        start.set()
        for process in processes:
            process.join(15)
            assert process.exitcode == 0
        records = [results.get(timeout=5) for _ in processes]
        assert all(record[0] in {"busy", "ok"} for record in records)
        assert any(record[0] == "busy" for record in records)
        assert maximum.value <= 1
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)


def test_reentrant_lock_uses_held_authority_after_root_becomes_inaccessible(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, journal = _begin_single(
        tmp_path,
        run_id="external-reentrant-inaccessible",
    )
    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    try:
        with journals.workspace_mutation_lock(repo):
            repo.chmod(0)
            with journals.workspace_mutation_lock(repo) as nested:
                assert nested.root_metadata.st_ino == repo.stat().st_ino
    finally:
        repo.chmod(baseline_mode)
        if journal.active:
            assert journal.restore()


def test_symlink_alias_reenters_one_inode_lock_without_self_deadlock(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, journal = _begin_single(
        tmp_path,
        run_id="external-symlink-alias",
    )
    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo, target_is_directory=True)
    with journals.workspace_mutation_lock(repo) as direct:
        with journals.workspace_mutation_lock(alias) as through_alias:
            assert direct.lock_key == through_alias.lock_key
            assert direct.root_metadata.st_ino == through_alias.root_metadata.st_ino
    assert journal.restore()


def test_startup_recovery_admits_mode_zero_checkout(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus, journal = _begin_single(
        tmp_path,
        run_id="external-startup-mode-zero",
    )
    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    _simulate_process_death(journal)
    try:
        repo.chmod(0)
        report = journals.recover_mutation_journals(corpus, repo)
        assert report.failures == ()
        assert report.recovered == (journal.slice_id,)
        assert stat.S_IMODE(repo.stat().st_mode) == baseline_mode
    finally:
        if repo.exists():
            repo.chmod(baseline_mode)


def test_startup_recovery_alias_selects_captured_root_inode(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus, journal = _begin_single(
        tmp_path,
        run_id="external-startup-alias-mode-zero",
    )
    alias = tmp_path / "startup-repo-alias"
    alias.symlink_to(repo, target_is_directory=True)
    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    _simulate_process_death(journal)
    try:
        repo.chmod(0)
        report = journals.recover_mutation_journals(corpus, alias)
        assert report.failures == ()
        assert report.recovered == (journal.slice_id,)
        assert stat.S_IMODE(repo.stat().st_mode) == baseline_mode
    finally:
        if repo.exists():
            repo.chmod(baseline_mode)


def test_parent_restore_coordinates_inaccessible_child_root(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    child_root = _add_clean_submodule(repo, tmp_path)
    child_mode = stat.S_IMODE(child_root.stat().st_mode)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir
            / "runs"
            / "external-child-mode-zero"
            / "mutations"
        ),
        slice_id="external-child-mode-zero",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert len(journal._nested_journals()) == 1
    try:
        child_root.chmod(0)
        assert journal.restore()
        assert stat.S_IMODE(child_root.stat().st_mode) == child_mode
    finally:
        if child_root.exists():
            child_root.chmod(child_mode)


def test_startup_recovery_coordinates_inaccessible_child_family(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    child_root = _add_clean_submodule(repo, tmp_path)
    child_mode = stat.S_IMODE(child_root.stat().st_mode)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir
            / "runs"
            / "external-child-startup"
            / "mutations"
        ),
        slice_id="external-child-startup",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    family = (journal, *journal._nested_journals())
    assert len(family) == 2
    for member in reversed(family):
        _simulate_process_death(member)
    try:
        child_root.chmod(0)
        report = journals.recover_mutation_journals(corpus, repo)
        assert report.failures == ()
        assert journal.slice_id in report.recovered
        assert stat.S_IMODE(child_root.stat().st_mode) == child_mode
        assert all(not member.journal_dir.exists() for member in family)
    finally:
        if child_root.exists():
            child_root.chmod(child_mode)


def test_absent_lifecycle_family_recovers_inaccessible_child_root(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    child_root = _add_clean_submodule(repo, tmp_path)
    child_mode = stat.S_IMODE(child_root.stat().st_mode)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir
            / "runs"
            / "external-child-lifecycle"
            / "mutations"
        ),
        slice_id="external-child-lifecycle",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    family = (journal, *journal._nested_journals())
    assert len(family) == 2
    journals.prepare_mutation_lifecycle_settlement(
        corpus,
        (journal,),
        kind="landed",
        reason="absent inaccessible child lifecycle",
        run_id="external-child-lifecycle",
    )
    for member in reversed(family):
        _simulate_process_death(member)
    try:
        child_root.chmod(0)
        report = journals.recover_mutation_journals(corpus, repo)
        assert report.failures == ()
        assert {member.slice_id for member in family}.issubset(
            set(report.recovered)
        )
        assert stat.S_IMODE(child_root.stat().st_mode) == child_mode
        assert all(not member.journal_dir.exists() for member in family)
    finally:
        if child_root.exists():
            child_root.chmod(child_mode)


@pytest.mark.timeout(300)
def test_attached_lifecycle_commit_cleans_nested_submodule_family(
    tmp_path: Path,
) -> None:
    from assay import LifecycleEvent

    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._run import exec_ctx_now

    repo, corpus = _setup(tmp_path)
    _add_clean_submodule(repo, tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir
            / "runs"
            / "attached-submodule-lifecycle"
            / "mutations"
        ),
        slice_id="attached-submodule-lifecycle",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    family = (journal, *journal._nested_journals())
    assert len(family) == 2
    guard = family[1]
    assert guard.parent_transaction_id == journal.transaction_id
    assert guard.journal_dir.parent == journal.journal_dir / "nested_mutations"

    accepted = "# accepted attached submodule lifecycle\n"
    (repo / "a.py").write_text(accepted, encoding="utf-8")
    reason, _settlement_ref = journals.prepare_mutation_lifecycle_settlement(
        corpus,
        (journal,),
        kind="landed",
        reason="attached nested submodule family",
        run_id="attached-submodule-lifecycle",
    )
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason=reason,
            affected_versions=(version,),
            execution=exec_ctx_now(
                "test",
                run_id="attached-submodule-lifecycle",
            ),
        ),
    )

    registry = Path(journal._manifest["registry_directory"])
    transaction_ids = {member.transaction_id for member in family}
    journal.commit(attached=True, corpus=corpus)

    assert (repo / "a.py").read_text(encoding="utf-8") == accepted
    assert all(not member.journal_dir.exists() for member in family)
    assert list(registry.iterdir()) == []
    with journals._LIVE_TRANSACTIONS_LOCK:
        assert transaction_ids.isdisjoint(journals._LIVE_TRANSACTIONS)
        assert transaction_ids.isdisjoint(journals._LIVE_JOURNAL_OBJECTS)


@pytest.mark.parametrize("startup", (False, True))
def test_two_level_nested_restore_coordinates_inaccessible_grandchild(
    tmp_path: Path,
    startup: bool,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    child_root = _add_clean_submodule(repo, tmp_path)
    grand_tmp = tmp_path / "grandchild-fixture"
    grand_tmp.mkdir()
    grandchild_root = _add_clean_submodule(child_root, grand_tmp)
    _git(repo, "add", "vendor/nested")
    _git(repo, "commit", "-qm", "advance nested dependency")
    grandchild_mode = stat.S_IMODE(grandchild_root.stat().st_mode)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir
            / "runs"
            / f"external-grandchild-{startup}"
            / "mutations"
        ),
        slice_id=f"external-grandchild-{startup}",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    family = (journal, *journal._nested_journals_children_first())
    assert len(family) == 3
    if startup:
        journals._after_fork_child()
    try:
        grandchild_root.chmod(0)
        if startup:
            report = journals.recover_mutation_journals(corpus, repo)
            assert report.failures == ()
            assert journal.slice_id in report.recovered
        else:
            assert journal.restore()
        assert stat.S_IMODE(grandchild_root.stat().st_mode) == grandchild_mode
        assert all(not member.journal_dir.exists() for member in family)
    finally:
        if grandchild_root.exists():
            grandchild_root.chmod(grandchild_mode)


def _fresh_external_lock_path(
    journals,
    tmp_path: Path,
) -> tuple[tuple[int, int], Path]:
    device = tmp_path.stat().st_dev
    namespace = Path("/tmp") / (
        f"{journals._EXTERNAL_LOCK_DIRECTORY_PREFIX}{os.geteuid()}"
    )
    namespace.mkdir(mode=0o700, exist_ok=True)
    while True:
        inode = int.from_bytes(os.urandom(8), "big") or 1
        key = (device, inode)
        path = namespace / journals._external_lock_filename("common", key)
        if not path.exists() and not path.is_symlink():
            return key, path


def test_external_lock_rejects_precreated_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    key, lock_path = _fresh_external_lock_path(journals, tmp_path)
    victim = lock_path.with_name(lock_path.name + ".victim")
    victim.write_bytes(b"unchanged\n")
    victim.chmod(0o600)
    try:
        lock_path.symlink_to(victim)
        with pytest.raises(journals.MutationJournalError):
            with journals._external_file_lock(kind="common", key=key):
                pass
        assert victim.read_bytes() == b"unchanged\n"
        lock_path.unlink()

        os.link(victim, lock_path)
        with pytest.raises(
            journals.MutationJournalError,
            match="private real file",
        ):
            with journals._external_file_lock(kind="common", key=key):
                pass
        assert victim.stat().st_nlink == 2
    finally:
        if lock_path.exists() or lock_path.is_symlink():
            lock_path.unlink()
        if victim.exists():
            victim.unlink()


def test_external_lock_detects_named_inode_replacement_while_held(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    key, lock_path = _fresh_external_lock_path(journals, tmp_path)
    parked = lock_path.with_name(lock_path.name + ".parked")
    assert not lock_path.exists() and not parked.exists()
    try:
        with pytest.raises(
            journals.MutationJournalError,
            match="namespace or inode changed",
        ):
            with journals._external_file_lock(kind="common", key=key):
                lock_path.rename(parked)
                lock_path.write_bytes(b"")
                lock_path.chmod(0o600)
    finally:
        if lock_path.exists():
            lock_path.unlink()
        if parked.exists():
            parked.unlink()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork regression")
def test_fork_child_normal_unwind_cannot_close_reacquired_lock_fds(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus = _setup(tmp_path)
    read_fd, write_fd = os.pipe()
    child_pid = -1
    child_context = None
    child_authority = None
    is_child = False
    with journals.workspace_mutation_lock(repo):
        child_pid = os.fork()
        if child_pid == 0:  # pragma: no branch - child exits below
            is_child = True
            os.close(read_fd)
            child_context = journals.workspace_mutation_lock(repo)
            child_authority = child_context.__enter__()
    if is_child:  # inherited outer frame has now unwound normally
        try:
            assert child_authority is not None and child_context is not None
            journals._assert_workspace_mutation_authority(repo)
            child_context.__exit__(None, None, None)
            os.write(write_fd, b"ok")
        except BaseException as exc:  # pragma: no cover - sent to parent
            os.write(
                write_fd,
                f"{type(exc).__name__}:{exc}".encode("utf-8", "replace"),
            )
        finally:
            os.close(write_fd)
            os._exit(0)

    os.close(write_fd)
    payload = os.read(read_fd, 4096)
    os.close(read_fd)
    waited_pid, status = os.waitpid(child_pid, 0)
    assert waited_pid == child_pid
    assert os.waitstatus_to_exitcode(status) == 0
    assert payload == b"ok"
