# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Crash/exception safety for tier Phase-1 workspace mutations."""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
import shlex
import signal
import shutil
import stat
import subprocess
import sys
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    Implementation,
    InvocationResult,
    LifecycleEvent,
    Message,
    ProjectContext,
    Spec,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG


_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_TRACE = BackendExecutionTrace(
    backend="parallel-mutation-stub",
    requested_model="stub-model",
    events=(
        BackendExecutionEvent(
            sequence=0,
            kind="terminal",
            name="end_turn",
        ),
    ),
)


def _git(cwd: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True,
    ).stdout


def _init_repo(path: Path) -> str:
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "tier@test.invalid")
    _git(path, "config", "user.name", "tier test")
    for name in ("a.py", "b.py", "prior.py"):
        (path / name).write_text(f"# baseline {name}\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "base")
    return _git(path, "rev-parse", "HEAD").decode().strip()


def _spec(base_commit: str) -> Spec:
    slices = tuple(
        Slice(
            id=f"s{index}",
            directive=ImplementationDirective(
                slice_id=f"s{index}",
                intent=f"edit slice {index}",
                files_in_scope=(f"{'a' if index == 0 else 'b'}.py",),
                creates_paths=(),
                verify_mode="skip",
                blast_radius={"max_files": 1},
            ),
        )
        for index in range(2)
    )
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier_mutation_transaction_test",
            repo_url="https://example.invalid/repo.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="exercise Phase-1 rollback",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=slices, depends_on=()),),
        ),
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _setup(tmp_path: Path) -> tuple[Path, Corpus]:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec(base))
    return repo, corpus


def _git_execution_sentinel(tmp_path: Path) -> tuple[Path, Path]:
    """Create a Git execution control that leaves an observable marker."""

    marker = tmp_path / "git-execution-control-ran"
    executable = tmp_path / "git-execution-control"
    executable.write_text(
        "#!/bin/sh\n"
        f"printf 'executed\\n' >> {shlex.quote(str(marker))}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, marker


def test_sanitized_git_environment_disables_only_optional_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows._mutation_journal import (
        _git_argv,
        _sanitized_git_env,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    _init_repo(repository)
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    monkeypatch.setenv("GIT_DIR", "/attacker-controlled")
    environment = _sanitized_git_env()

    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert "GIT_DIR" not in environment

    (repository / "a.py").write_text("# staged mutation\n", encoding="utf-8")
    index = repository / ".git" / "index"
    index_bytes = index.read_bytes()
    index_lock = repository / ".git" / "index.lock"
    index_lock.write_bytes(b"held mandatory-lock witness\n")
    blocked = subprocess.run(
        [*_git_argv(), "add", "--", "a.py"],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
    )
    assert blocked.returncode != 0
    assert b"index.lock" in blocked.stderr
    assert index.read_bytes() == index_bytes
    assert index_lock.read_bytes() == b"held mandatory-lock witness\n"

    index_lock.unlink()
    subprocess.run(
        [*_git_argv(), "add", "--", "a.py"],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
    )
    staged = subprocess.run(
        [*_git_argv(), "diff", "--cached", "--name-only"],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert staged.stdout.splitlines() == ["a.py"]
    assert not index_lock.exists()


def test_public_journal_recovery_owns_workspace_and_spec_authority(
    tmp_path: Path,
    monkeypatch,
):
    """Direct recovery cannot race either source or Spec publication."""

    from crucible.workflows import _mutation_journal

    repo, corpus = _setup(tmp_path)
    observed: list[str] = []

    def _locked_recovery(bound_corpus, cwd):
        assert Path(cwd) == repo.resolve()
        assert getattr(_mutation_journal._LOCK_STATE, "held", {})
        with pytest.raises(RuntimeError, match="locked_spec_authority"):
            bound_corpus.commit()
        observed.append("both")
        return _mutation_journal.RecoveryReport((), (), ())

    monkeypatch.setattr(
        _mutation_journal,
        "_recover_mutation_journals_locked",
        _locked_recovery,
    )

    report = _mutation_journal.recover_mutation_journals(corpus, repo)

    assert report.failures == ()
    assert observed == ["both"]


@pytest.mark.parametrize(
    "restore_method",
    (
        "_restore_control_files",
        "_restore_index",
        "_restore_registry_marker",
        "_restore_registry_topology",
        "_restore_refs_tree",
        "_restore_git_admin_topology",
    ),
    ids=(
        "control-files",
        "index",
        "registry-marker",
        "registry-topology",
        "refs-tree",
        "git-admin-topology",
    ),
)
def test_public_recovery_fails_closed_on_temporary_root_replacement(
    tmp_path: Path,
    monkeypatch,
    restore_method: str,
) -> None:
    """A temporary root replacement cannot redirect one restore phase."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows import _mutation_journal

    journal = _mutation_journal.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "bound_restore" / "mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# crashed A bytes\n", encoding="utf-8")
    _simulate_process_death(journal)

    replacement = tmp_path / "replacement-b"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--no-hardlinks",
            str(repo),
            str(replacement),
        ],
        check=True,
        capture_output=True,
    )
    _git(replacement, "config", "user.email", "b@example.invalid")
    (replacement / "b-only.txt").write_text("B staged bytes\n", encoding="utf-8")
    _git(replacement, "add", "b-only.txt")
    _git(replacement, "branch", "b-only")
    replacement_registry = (
        replacement / ".git" / "crucible-mutation-journals"
    )
    replacement_registry.mkdir()
    replacement_marker = (
        replacement_registry / f"{journal.transaction_id}.json"
    )
    replacement_marker.write_text("B independent marker\n", encoding="utf-8")
    replacement_extra = replacement_registry / "b-extra.json"
    replacement_extra.write_text("B registry extra\n", encoding="utf-8")
    replacement_admin_extra = replacement / ".git" / "b-extra-admin"
    replacement_admin_extra.write_text("B admin extra\n", encoding="utf-8")

    protected_files = (
        replacement / ".git" / "config",
        replacement / ".git" / "index",
        replacement_marker,
        replacement_extra,
        replacement / ".git" / "refs" / "heads" / "b-only",
        replacement_admin_extra,
    )
    replacement_before = {
        path.relative_to(replacement).as_posix(): path.read_bytes()
        for path in protected_files
    }

    journal_type = _mutation_journal.WorkspaceMutationJournal
    real_restore = getattr(journal_type, restore_method)
    displaced = tmp_path / "held-a"
    calls = 0

    def _swap_only_around_restore(self, *args, **kwargs):
        nonlocal calls
        if self.transaction_id != journal.transaction_id:
            return real_restore(self, *args, **kwargs)
        repo.rename(displaced)
        replacement.rename(repo)
        try:
            return real_restore(self, *args, **kwargs)
        finally:
            repo.rename(replacement)
            displaced.rename(repo)
            calls += 1

    monkeypatch.setattr(
        journal_type,
        restore_method,
        _swap_only_around_restore,
    )

    report = _mutation_journal.recover_mutation_journals(corpus, repo)

    assert calls > 0
    # Root identity is part of the public workspace authority.  Even though
    # every restore primitive is descriptor-bound and therefore cannot touch
    # B, a lexical root replacement must retain A's journal rather than report
    # an exact recovery through a path that changed identity.
    assert report.recovered == ()
    assert report.failures
    assert journal.journal_dir.exists()
    assert (repo / "a.py").read_text(encoding="utf-8") in {
        "# baseline a.py\n",
        "# crashed A bytes\n",
    }
    replacement_after = {
        path.relative_to(replacement).as_posix(): path.read_bytes()
        for path in protected_files
    }
    assert replacement_after == replacement_before


def test_public_recovery_rejects_detached_git_admin_before_source_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Grafting A's Git admin onto B cannot redirect recovery into B."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows import _mutation_journal

    replacement = tmp_path / "replacement-b"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--no-hardlinks",
            str(repo),
            str(replacement),
        ],
        check=True,
        capture_output=True,
    )
    replacement_source_before = {
        name: (replacement / name).read_bytes()
        for name in ("a.py", "b.py", "prior.py")
    }
    replacement_git_config_before = (replacement / ".git" / "config").read_bytes()

    journal = _mutation_journal.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "git_admin_graft" / "mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# crashed A bytes\n", encoding="utf-8")
    _simulate_process_death(journal)

    journal_type = _mutation_journal.WorkspaceMutationJournal
    real_assert = journal_type._assert_authority_lock_unchanged
    displaced = tmp_path / "held-a"
    replacement_git = tmp_path / "replacement-b-git"
    swapped = False

    def _graft_after_authority_check(self):
        nonlocal swapped
        result = real_assert(self)
        if self.transaction_id == journal.transaction_id and not swapped:
            repo.rename(displaced)
            replacement.rename(repo)
            (repo / ".git").rename(replacement_git)
            (displaced / ".git").rename(repo / ".git")
            swapped = True
        return result

    monkeypatch.setattr(
        journal_type,
        "_assert_authority_lock_unchanged",
        _graft_after_authority_check,
    )

    report = _mutation_journal.recover_mutation_journals(corpus, repo)

    assert swapped is True
    assert report.recovered == ()
    assert report.failures
    assert journal.journal_dir.exists()
    assert (displaced / "a.py").read_text(encoding="utf-8") == (
        "# crashed A bytes\n"
    )
    assert {
        name: (repo / name).read_bytes()
        for name in ("a.py", "b.py", "prior.py")
    } == replacement_source_before
    assert (replacement_git / "config").read_bytes() == (
        replacement_git_config_before
    )


@pytest.mark.parametrize("settlement_path", ("startup", "graceful"))
def test_journal_settlement_rejects_laundered_dropped_accepted_ref(
    tmp_path: Path,
    settlement_path: str,
) -> None:
    """A forged latest manifest cannot restore already-accepted bytes."""

    from assay.canonical import serialize

    from crucible.workflows import _mutation_journal
    from crucible.workflows._run import exec_ctx_now

    repo, corpus = _setup(tmp_path)
    journal = _mutation_journal.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "forged-latest" / "mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# accepted bytes\n", encoding="utf-8")
    implementation = Implementation(
        slice_id="s0",
        base_commit=_git(repo, "rev-parse", "HEAD").decode().strip(),
        result_commit=None,
        diff_ref=corpus.put_artifact(
            _git(repo, "diff", "--binary", "--", "a.py"),
            kind="diffs",
            name_hint="accepted.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]",
            kind="transcripts",
            name_hint="accepted.json",
        ),
        files_modified=("a.py",),
        scope_escapes=(),
        classification="success",
        execution=exec_ctx_now("test"),
    )
    journal.prepare(implementation, keep=True)
    corpus.commit(implementations=(implementation,))
    if settlement_path == "startup":
        _mutation_journal._unregister_live(journal.transaction_id)

    accepted = corpus.latest_spec()
    accepted_manifest, = tuple(
        (Path(corpus.spec_dir) / "versions").glob(
            f"v{accepted.version}_*.json"
        )
    )
    tampered = dataclasses.replace(
        accepted,
        implementation_refs=(),
    )
    # Keep the old content-addressed filename. The public reader now rejects
    # this rewritten predecessor directly, before a forged child can launder
    # the dropped reference into a superficially contiguous lineage.
    accepted_manifest.write_bytes(serialize(tampered))
    reloaded = Corpus(corpus.spec_dir)

    if settlement_path == "startup":
        report = _mutation_journal.recover_mutation_journals(reloaded, repo)
        assert report.recovered == ()
        assert any(
            "Spec" in failure
            and (
                "invalid" in failure
                or "unreadable" in failure
            )
            for failure in report.failures
        )
    else:
        with pytest.raises(
            _mutation_journal.MutationJournalError,
            match="Spec lineage|latest Spec",
        ):
            journal.settle_or_restore_outcome(reloaded)
    assert (repo / "a.py").read_text(encoding="utf-8") == (
        "# accepted bytes\n"
    )
    assert journal.journal_dir.exists()


def test_public_prepared_keep_settlement_owns_both_authorities(
    tmp_path: Path,
    monkeypatch,
):
    """Graceful negative settlement cannot race a late Spec acceptance."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows import _mutation_journal
    from crucible.workflows._run import exec_ctx_now

    journal = _mutation_journal.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "settle_lock" / "mutations",
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# unaccepted bytes\n", encoding="utf-8")
    impl = Implementation(
        slice_id="s0",
        base_commit=_git(repo, "rev-parse", "HEAD").decode().strip(),
        result_commit=None,
        diff_ref=corpus.put_artifact(
            b"unaccepted",
            kind="diffs",
            name_hint="settle-lock.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]",
            kind="transcripts",
            name_hint="settle-lock.json",
        ),
        files_modified=("a.py",),
        scope_escapes=(),
        classification="success",
        execution=exec_ctx_now("test"),
    )
    journal.prepare(impl, keep=True)
    observed: list[str] = []

    def _absent_under_lock(bound_corpus, expected, lineage, **kwargs):
        del expected, lineage, kwargs
        assert getattr(_mutation_journal._LOCK_STATE, "held", {})
        with pytest.raises(RuntimeError, match="locked_spec_authority"):
            bound_corpus.commit()
        observed.append("both")
        return None

    monkeypatch.setattr(
        _mutation_journal,
        "_expected_implementation_ref",
        _absent_under_lock,
    )

    outcome = journal.settle_or_restore_outcome(corpus)

    assert outcome.completed
    assert outcome.disposition == "restored"
    assert observed == ["both"]
    assert (repo / "a.py").read_text(encoding="utf-8") == "# baseline a.py\n"


def test_git_execution_controls_cannot_run_during_journal_recovery_or_attestation(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows.tier import attest_mutation_journal_settlement

    repo, corpus = _setup(tmp_path)
    base_commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    executable, marker = _git_execution_sentinel(tmp_path)

    # A HOME-owned global fsmonitor and ambient external diff are both
    # executable Git controls. Journal commands must ignore them rather than
    # trusting the operator process environment.
    hostile_home = tmp_path / "hostile-home"
    hostile_home.mkdir()
    _git(
        repo,
        "config",
        "--file",
        str(hostile_home / ".gitconfig"),
        "core.fsmonitor",
        str(executable),
    )
    monkeypatch.setenv("HOME", str(hostile_home))
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", str(executable))

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "git_controls" / "mutations"
        ),
        slice_id="git_controls",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert not marker.exists()

    # Install the local fsmonitor only after begin(): begin deliberately
    # rejects one that predates the transaction. Crash recovery still has to
    # inspect and restore this hostile repository metadata without invoking
    # it first.
    _git(repo, "config", "--local", "core.fsmonitor", str(executable))
    (repo / "a.py").write_text("# interrupted write\n", encoding="utf-8")
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)
    assert report.failures == ()
    assert report.mutation_journals_recovered == ("git_controls",)
    assert (repo / "a.py").read_text(encoding="utf-8") == "# baseline a.py\n"
    assert not marker.exists()

    # Exercise the post-run attestor with all three controls still hostile.
    # Recovery removed the transaction-time local config mutation, so put it
    # back to prove attestation's Git reads override it too.
    _git(repo, "config", "--local", "core.fsmonitor", str(executable))
    (repo / "a.py").write_text("# accepted candidate\n", encoding="utf-8")
    proof = attest_mutation_journal_settlement(
        corpus=corpus,
        cwd=repo,
        tier_id="t0",
        operation_id="tier-0",
        expected_base_commit=base_commit,
    )
    assert proof["settled"] is True
    assert proof["workspace_files_modified"] == ["a.py"]
    assert not marker.exists()


def test_null_device_hooks_path_is_supported_and_never_restored(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    repo, corpus = _setup(tmp_path)
    _git(repo, "config", "--local", "core.hooksPath", os.devnull)
    null_before = Path(os.devnull).lstat()

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "null_hooks" / "mutations"
        ),
        slice_id="null_hooks",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert journal._manifest["disabled_git_hooks"]["path"] == os.devnull
    assert all(
        record["name"] != "hooks"
        for record in journal._manifest["auxiliary_git_trees"]
    )

    original_remove_path = _mutation_journal._remove_path

    def guarded_remove_path(path: Path) -> None:
        assert path != Path(os.devnull)
        original_remove_path(path)

    monkeypatch.setattr(
        _mutation_journal,
        "_remove_path",
        guarded_remove_path,
    )

    # A transaction-time config redirect must still be detected and rolled
    # back, without treating the external null device as restorable state.
    _git(repo, "config", "--local", "core.hooksPath", ".githooks")
    with pytest.raises(
        MutationJournalError,
        match="changed Git admin file|changed disabled Git hooks binding",
    ):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert (
        _git(repo, "config", "--local", "--get", "core.hooksPath")
        .decode()
        .strip()
        == os.devnull
    )
    null_after = Path(os.devnull).lstat()
    assert (null_after.st_dev, null_after.st_ino, null_after.st_rdev) == (
        null_before.st_dev,
        null_before.st_ino,
        null_before.st_rdev,
    )


def test_external_hooks_path_remains_outside_journal_authority(tmp_path: Path):
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    repo, corpus = _setup(tmp_path)
    external_hooks = tmp_path / "external-hooks"
    external_hooks.mkdir()
    _git(
        repo,
        "config",
        "--local",
        "core.hooksPath",
        str(external_hooks),
    )

    with pytest.raises(
        MutationJournalError,
        match="configured Git hooks path escapes workspace/admin roots",
    ):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "external_hooks" / "mutations"
            ),
            slice_id="external_hooks",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )


def test_null_device_hooks_symlink_is_not_a_disabled_sentinel(tmp_path: Path):
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    repo, corpus = _setup(tmp_path)
    (repo / ".null-hooks").symlink_to(os.devnull)
    _git(repo, "config", "--local", "core.hooksPath", ".null-hooks")

    with pytest.raises(
        MutationJournalError,
        match="Git administrative tree must be a real directory",
    ):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "null_hooks_symlink" / "mutations"
            ),
            slice_id="null_hooks_symlink",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )


def test_diff_drivers_cannot_run_during_recovery_breach_or_attestation(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows._journal_recovery import _tree_delta_paths
    from crucible.workflows.breach import _touched_files_from_diff
    from crucible.workflows.tier import attest_mutation_journal_settlement

    repo, corpus = _setup(tmp_path)
    executable, marker = _git_execution_sentinel(tmp_path)

    (repo / ".gitattributes").write_text(
        "*.py diff=hostile\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-qm", "add hostile diff attributes")
    left = _git(repo, "rev-parse", "HEAD").decode().strip()
    (repo / "a.py").write_text("# committed delta\n", encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-qm", "change attributed file")
    right = _git(repo, "rev-parse", "HEAD").decode().strip()
    _git(
        repo,
        "config",
        "--local",
        "diff.hostile.textconv",
        str(executable),
    )
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", str(executable))

    # Recovery compares accepted trees. Neither the local textconv nor the
    # ambient external diff may participate in this authority decision.
    assert _tree_delta_paths(repo, left, right) == {"a.py"}
    assert not marker.exists()

    (repo / "b.py").write_text("# candidate delta\n", encoding="utf-8")
    patch = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "HEAD",
            "--",
            "b.py",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    assert _touched_files_from_diff(patch, cwd=str(repo)) == ("b.py",)
    assert not marker.exists()

    proof = attest_mutation_journal_settlement(
        corpus=corpus,
        cwd=repo,
        tier_id="t0",
        operation_id="tier-0",
        expected_base_commit=right,
    )
    assert proof["workspace_files_modified"] == ["b.py"]
    assert not marker.exists()


def test_tier_settlement_attestation_binds_clean_registry_and_workspace(
    tmp_path: Path,
):
    from crucible.common.benchmark_runtime import canonical_fingerprint
    from crucible.workflows.tier import attest_mutation_journal_settlement

    repo, corpus = _setup(tmp_path)
    (repo / "a.py").write_text("# accepted candidate\n", encoding="utf-8")
    run_dir = corpus.spec_dir / "runs" / "tier-child"
    run_dir.mkdir(parents=True)
    for name in ("summary.json", "result.json", "outcome.json"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")

    proof = attest_mutation_journal_settlement(
        corpus=corpus,
        cwd=repo,
        tier_id="t0",
        operation_id="tier-0",
        task_id="task",
        ext_id="ext",
        expected_base_commit=_git(repo, "rev-parse", "HEAD").decode().strip(),
        tier_run_dir=run_dir,
    )

    digest = proof.pop("attestation_sha256")
    assert digest == canonical_fingerprint(proof)
    assert proof["settled"] is True
    assert proof["registry_clear"] is True
    assert proof["remaining_journal_manifests"] == []
    assert proof["workspace_patch_bytes"] > 0
    assert proof["workspace_files_modified"] == ["a.py"]
    assert all(proof["tier_run_artifacts"].values())


def test_tier_settlement_attestation_binds_persisted_raw_spec_digest(
    tmp_path: Path,
) -> None:
    """Settlement identifies legacy manifest bytes, not their hydration."""
    from assay import PlanPolicy
    from assay.canonical import hash_record, parse, serialize

    from crucible.workflows.tier import attest_mutation_journal_settlement

    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    root = dataclasses.replace(_spec(base), plan_policy=PlanPolicy())
    legacy_payload = parse(serialize(root))
    del legacy_payload["plan_policy"][
        "max_verification_only_empty_scope_slices"
    ]
    legacy_blob = serialize(legacy_payload)
    legacy_digest = "sha256:" + hashlib.sha256(legacy_blob).hexdigest()
    (
        Path(corpus.spec_dir)
        / "versions"
        / f"v0_{legacy_digest.split(':', 1)[1][:12]}.json"
    ).write_bytes(legacy_blob)

    hydrated = corpus.latest_spec()
    assert hydrated == root
    assert str(hash_record(hydrated)) != legacy_digest

    proof = attest_mutation_journal_settlement(
        corpus=corpus,
        cwd=repo,
        tier_id="t0",
        operation_id="tier-0",
        expected_base_commit=base,
    )

    assert proof["spec_version"] == 0
    assert proof["spec_hash"] == legacy_digest


def test_tier_settlement_attestation_enforces_streaming_patch_limit(
    tmp_path: Path,
):
    from assay.primitives.implement import GitCaptureError
    from crucible.workflows.tier import attest_mutation_journal_settlement

    repo, corpus = _setup(tmp_path)
    (repo / "a.py").write_text("x" * 4096, encoding="utf-8")

    with pytest.raises(GitCaptureError, match="max_diff_bytes=128"):
        attest_mutation_journal_settlement(
            corpus=corpus,
            cwd=repo,
            tier_id="t0",
            operation_id="tier-0",
            expected_base_commit=(
                _git(repo, "rev-parse", "HEAD").decode().strip()
            ),
            max_patch_bytes=128,
        )


def test_tier_settlement_attestation_rejects_live_journal(
    tmp_path: Path,
):
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )
    from crucible.workflows.tier import attest_mutation_journal_settlement

    repo, corpus = _setup(tmp_path)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "live" / "phase1_mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    try:
        with pytest.raises(
            MutationJournalError,
            match="registry|journal|live owner",
        ):
            attest_mutation_journal_settlement(
                corpus=corpus,
                cwd=repo,
                tier_id="t0",
                operation_id="tier-0",
                expected_base_commit=(
                    _git(repo, "rev-parse", "HEAD").decode().strip()
                ),
            )
    finally:
        assert journal.restore()


def _add_clean_submodule(repo: Path, tmp_path: Path) -> Path:
    origin = tmp_path / "nested-origin"
    origin.mkdir()
    _git(origin, "init", "-q")
    _git(origin, "config", "user.email", "nested@test.invalid")
    _git(origin, "config", "user.name", "nested test")
    (origin / "nested.txt").write_text("nested baseline\n", encoding="utf-8")
    _git(origin, "add", "nested.txt")
    _git(origin, "commit", "-qm", "nested base")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(origin),
        "vendor/nested",
    )
    nested = repo / "vendor" / "nested"
    _git(nested, "config", "user.email", "nested@test.invalid")
    _git(nested, "config", "user.name", "nested test")
    _git(repo, "commit", "-qam", "add nested dependency")
    return nested


def _success(
    *,
    input_tokens: int = 1,
    output_tokens: int = 1,
    cost_usd: float = 0.0,
) -> InvocationResult:
    return InvocationResult(
        final_message="DONE",
        transcript=(),
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ),
        stop_reason="end_turn",
        error=None,
        completeness=_COMPLETE_EXECUTION,
        execution_trace=_COMPLETE_TRACE,
    )


def _complete_priced_result(
    message: str = "DONE",
    *,
    stop_reason: str = "end_turn",
) -> InvocationResult:
    usage = Usage(input_tokens=11, output_tokens=7, cost_usd=0.125)
    return InvocationResult(
        final_message=message,
        transcript=(Message(role="assistant", content=message),),
        usage=usage,
        stop_reason=stop_reason,
        error=None,
        completeness=_COMPLETE_EXECUTION,
        execution_trace=_COMPLETE_TRACE,
        pricing_evidence=UsagePricingEvidence(
            segments=(
                UsagePricingSegment(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    cost_usd=usage.cost_usd,
                ),
            ),
            complete=True,
        ),
    )


def _authoring_finalization_spec(
    base_commit: str,
) -> Spec:
    from crucible.constraints import AuthoringFinalizationConstraint
    materializes_paths = ("qualification.json",)

    directive = ImplementationDirective(
        slice_id="s0",
        intent="edit source and let the workflow create its receipt",
        files_in_scope=("a.py",),
        creates_paths=materializes_paths,
        verify_mode="tests",
        blast_radius={"max_files": 1 + len(materializes_paths)},
    )
    base = _spec(base_commit)
    return dataclasses.replace(
        base,
        project_context=dataclasses.replace(
            base.project_context,
            runners={
                "build": "tests.synthetic_build",
                "verify": "tests.synthetic_verify",
            },
        ),
        constraints=(
            AuthoringFinalizationConstraint(
                slice_id="s0",
                materializes_paths=materializes_paths,
                gate="verify",
            ),
        ),
        tier_dag=TierDAG(
            tiers=(
                Tier(
                    tier_id="t0",
                    slices=(Slice(id="s0", directive=directive),),
                    depends_on=(),
                ),
            ),
        ),
    )


def _simulate_process_death(journal) -> None:
    """Drop only the in-memory owner registry; retain the durable journal."""

    from crucible.workflows._mutation_journal import _unregister_live

    _unregister_live(journal.transaction_id)


class _MutationBackend:
    name = "parallel-mutation-stub"
    model = "stub-model"

    def __init__(self, repo: Path, *, raise_s0: bool, empty_s0: bool = False):
        self.repo = repo
        self.raise_s0 = raise_s0
        self.empty_s0 = empty_s0
        self.calls: list[str] = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        is_s0 = "slice `s0`" in invocation.user
        self.calls.append("s0" if is_s0 else "s1")
        if is_s0:
            if not self.empty_s0:
                (self.repo / "a.py").write_text("# partial s0\n")
            if self.raise_s0:
                (self.repo / "rogue.txt").write_text("unattributed partial\n")
                raise RuntimeError("implementer exploded")
        else:
            (self.repo / "b.py").write_text("# completed s1\n")
        return _success()


def _patch_backend(monkeypatch, backend) -> None:
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kwargs: backend,
    )


def _run(
    corpus: Corpus,
    repo: Path,
    *,
    skip_build_verify: bool = True,
    build_retries: int = 0,
) -> int:
    from crucible.workflows.tier import run_tier

    return run_tier(
        spec=corpus.spec_dir,
        tier_id="t0",
        cwd=str(repo),
        max_concurrent_slices=2,
        build_retries=build_retries,
        skip_build_verify=skip_build_verify,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    )


def _setup_authoring_finalization(
    tmp_path: Path,
) -> tuple[Path, Corpus]:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus = Corpus(tmp_path / "corpus")
    corpus.put_spec(_authoring_finalization_spec(base))
    return repo, corpus


def _successful_authoring_gate_outcome():
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )

    return BuildVerifyOutcome(
        classification="success",
        build=RunnerResult(
            role="build",
            runner="tests.synthetic_build",
            returncode=0,
            values={"returncode": 0.0},
            raw_output=b'{"returncode":0}\n',
            stderr=b"",
            duration_s=0.01,
        ),
        verify=RunnerResult(
            role="verify",
            runner="tests.synthetic_verify",
            returncode=0,
            values={"returncode": 0.0, "tests_failed": 0.0},
            raw_output=b'{"returncode":0,"tests_failed":0}\n',
            stderr=b"",
            duration_s=0.01,
        ),
    )


def test_authoring_finalizer_runs_after_durable_result_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A host gate materializes exact outputs inside the captured candidate."""

    from crucible.workflows import _build_verify, _gate_evidence

    repo, corpus = _setup_authoring_finalization(tmp_path)

    class Backend(_MutationBackend):
        def __init__(self, path: Path):
            super().__init__(path, raise_s0=False)
            self.invocations = []

        def invoke(self, invocation):
            self.calls.append("s0")
            self.invocations.append(invocation)
            (self.repo / "a.py").write_text(
                "# authoring finalizer source\n",
                encoding="utf-8",
            )
            return _complete_priced_result()

    backend = Backend(repo)
    _patch_backend(monkeypatch, backend)
    gate_calls = 0

    def _gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 1:
            run_dir = max((corpus.spec_dir / "runs").iterdir())
            checkpoint = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )["latest_implementer_invocation"]
            assert checkpoint["accounting_state"] == "persisted"
            assert checkpoint["completion_disposition"] == "clear"
            assert corpus.get_artifact(checkpoint["transcript_ref"])
            (repo / "qualification.json").write_text(
                '{"qualified":true}\n',
                encoding="utf-8",
            )
        _build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_authoring_finalizer",
            succeeded=True,
        )
        return _successful_authoring_gate_outcome()

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        _gate,
    )

    assert _run(corpus, repo, skip_build_verify=False) == 0
    assert backend.calls == ["s0"]
    # The exact candidate-bound finalizer outcome satisfies the ordinary
    # per-slice gate; Crucible must not rerun a materializing gate immediately
    # and then reject its own receipt write as gate mutation.
    assert gate_calls == 1
    assert "WORKFLOW-OWNED AUTHORING FINALIZATION" in (
        backend.invocations[0].user
    )
    assert (repo / "qualification.json").is_file()

    latest = corpus.latest_spec()
    implementation = corpus.get_record(
        latest.implementation_refs[-1],
        as_type=Implementation,
    )
    finalization_ref = implementation.artifacts[
        "authoring_finalization_evidence"
    ]
    evidence = json.loads(
        corpus.get_artifact(finalization_ref).decode("utf-8")
    )
    assert evidence["status"] == "succeeded"
    assert evidence["materializes_paths"] == ["qualification.json"]
    assert evidence["changed_paths"] == ["qualification.json"]
    assert evidence["transcript_binding_matches"] is True
    assert evidence["disposition_matches"] is True
    assert evidence["verification_evidence_ref"].startswith("sha256:")
    verification_ref = implementation.artifacts[
        "authoring_finalization_verification_evidence"
    ]
    assert str(verification_ref) == evidence["verification_evidence_ref"]
    verification_evidence = _gate_evidence.load_verification_evidence(
        corpus,
        str(verification_ref),
    )
    assert verification_evidence.candidate_diff_sha256 == str(
        implementation.diff_ref
    )
    assert _gate_evidence.verification_evidence_content_available(
        corpus,
        verification_evidence,
    )

    run_dir = max((corpus.spec_dir / "runs").iterdir())
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1




def test_authoring_finalizer_requires_final_durable_checkpoint_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A failed final checkpoint write grants no host-runner authority."""

    from assay.corpus.runs import RunHandle

    repo, corpus = _setup_authoring_finalization(tmp_path)
    before = (repo / "a.py").read_bytes()

    class Backend(_MutationBackend):
        def invoke(self, invocation):
            self.calls.append("s0")
            (self.repo / "a.py").write_text(
                "# candidate awaiting durable accounting\n",
                encoding="utf-8",
            )
            return _complete_priced_result()

    backend = Backend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)
    gate_calls = 0

    def _forbidden_gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        raise AssertionError(
            "failed durable checkpoint dispatched a host runner"
        )

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        _forbidden_gate,
    )
    original_write_summary = RunHandle.write_summary
    checkpoint_failure_injected = False

    def _fail_final_checkpoint_once(self, payload):
        nonlocal checkpoint_failure_injected
        latest = (
            payload.get("latest_implementer_invocation")
            if isinstance(payload, dict)
            else None
        )
        if (
            not checkpoint_failure_injected
            and isinstance(latest, dict)
            and latest.get("accounting_state") == "persisted"
        ):
            checkpoint_failure_injected = True
            raise OSError("simulated final durable checkpoint failure")
        return original_write_summary(self, payload)

    monkeypatch.setattr(
        RunHandle,
        "write_summary",
        _fail_final_checkpoint_once,
    )

    with pytest.raises(
        OSError,
        match="simulated final durable checkpoint failure",
    ):
        _run(corpus, repo, skip_build_verify=False)
    assert checkpoint_failure_injected is True
    assert gate_calls == 0
    assert backend.calls == ["s0"]
    assert (repo / "a.py").read_bytes() == before
    assert not (repo / "qualification.json").exists()


@pytest.mark.parametrize(
    "stop_reason",
    (
        None,
        "budget_exhausted",
        "cancelled",
        "content_filter",
        "context_limit",
        "delegation_forbidden",
        "error",
        "incomplete_accounting",
        "incomplete_terminal",
        "limit_contract_violated",
        "limit_evidence_incomplete",
        "max_tokens",
        "max_turns",
        "max-turns",
        "missing_model_identity",
        "model_mismatch",
        "permission_denied",
        "refusal",
        "stale_conversation",
        "stop_sequence",
        "timeout",
        "tool_budget_exhausted",
        "tool_dispatch_budget_exhausted",
        "tool_dispatch_terminal",
        "tool_use",
    ),
)
def test_authoring_finalizer_rejects_every_non_success_stop(
    tmp_path: Path,
    monkeypatch,
    stop_reason: str | None,
) -> None:
    """Only an explicit end_turn can authorize the workflow-owned runner."""

    repo, corpus = _setup_authoring_finalization(tmp_path)
    before = (repo / "a.py").read_bytes()

    class Backend(_MutationBackend):
        def invoke(self, invocation):
            self.calls.append("s0")
            (self.repo / "a.py").write_text(
                "# non-success terminal candidate\n",
                encoding="utf-8",
            )
            return _complete_priced_result(stop_reason=stop_reason)

    backend = Backend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)
    gate_calls = 0

    def _forbidden_gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        from crucible.workflows import _build_verify

        _build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_forbidden_authoring_finalizer",
            succeeded=True,
        )
        return _successful_authoring_gate_outcome()

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        _forbidden_gate,
    )

    assert _run(
        corpus,
        repo,
        skip_build_verify=False,
        build_retries=1,
    ) == 1
    assert backend.calls == ["s0"]
    assert gate_calls == 0
    assert (repo / "a.py").read_bytes() == before
    assert not (repo / "qualification.json").exists()
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == "hard_fail_authoring_finalize"
    evidence = json.loads(
        corpus.get_artifact(
            implementation.artifacts["authoring_finalization_evidence"]
        ).decode("utf-8")
    )
    assert evidence["status"] == "failed"
    assert evidence["skip_reason"] == "implementer_non_success_stop"
    assert evidence["stop_reason"] == stop_reason


def test_authoring_finalizer_never_overrides_blocked_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An explicit BLOCKED terminal revokes all host-runner authority."""

    repo, corpus = _setup_authoring_finalization(tmp_path)
    before = (repo / "a.py").read_bytes()

    class Backend(_MutationBackend):
        def invoke(self, invocation):
            self.calls.append("s0")
            (self.repo / "a.py").write_text(
                "# incomplete model mutation\n",
                encoding="utf-8",
            )
            return _complete_priced_result(
                "BLOCKED: product authority is unavailable"
            )

    backend = Backend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)
    gate_calls = 0

    def _forbidden_gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        raise AssertionError("BLOCKED completion dispatched a host runner")

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        _forbidden_gate,
    )

    assert _run(corpus, repo, skip_build_verify=False) == 1
    assert gate_calls == 0
    assert (repo / "a.py").read_bytes() == before
    assert not (repo / "qualification.json").exists()
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == "hard_fail_implementer_blocked"
    evidence = json.loads(
        corpus.get_artifact(
            implementation.artifacts["authoring_finalization_evidence"]
        ).decode("utf-8")
    )
    assert evidence["status"] == "skipped"
    assert evidence["skip_reason"] == "implementer_completion_terminal"


def test_authoring_finalizer_scope_escape_is_captured_then_restored(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Runner writes outside exact materialized paths remain diagnostic only."""

    from crucible.workflows import _build_verify

    repo, corpus = _setup_authoring_finalization(tmp_path)
    before_a = (repo / "a.py").read_bytes()
    before_b = (repo / "b.py").read_bytes()

    class Backend(_MutationBackend):
        def invoke(self, invocation):
            self.calls.append("s0")
            (self.repo / "a.py").write_text(
                "# candidate source\n",
                encoding="utf-8",
            )
            return _complete_priced_result()

    backend = Backend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)
    gate_calls = 0

    def _escaping_gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        (repo / "qualification.json").write_text(
            '{"qualified":true}\n',
            encoding="utf-8",
        )
        (repo / "b.py").write_text(
            "# escaped host runner\n",
            encoding="utf-8",
        )
        _build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_escaping_authoring_finalizer",
            succeeded=True,
        )
        return _successful_authoring_gate_outcome()

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        _escaping_gate,
    )

    assert _run(
        corpus,
        repo,
        skip_build_verify=False,
        build_retries=1,
    ) == 1
    assert backend.calls == ["s0"]
    assert gate_calls == 1
    assert (repo / "a.py").read_bytes() == before_a
    assert (repo / "b.py").read_bytes() == before_b
    assert not (repo / "qualification.json").exists()
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == "hard_fail_authoring_finalize"
    assert "b.py" in implementation.scope_escapes
    evidence = json.loads(
        corpus.get_artifact(
            implementation.artifacts["authoring_finalization_evidence"]
        ).decode("utf-8")
    )
    assert evidence["status"] == "failed"
    assert "b.py" in evidence["scope_escapes"]


def test_authoring_finalizer_requires_every_materialized_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A successful runner cannot omit one exact Spec-bound output."""

    from crucible.workflows import _build_verify

    repo, corpus = _setup_authoring_finalization(tmp_path)
    before = (repo / "a.py").read_bytes()

    class Backend(_MutationBackend):
        def invoke(self, invocation):
            self.calls.append("s0")
            (self.repo / "a.py").write_text(
                "# candidate with missing qualification\n",
                encoding="utf-8",
            )
            return _complete_priced_result()

    backend = Backend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)
    gate_calls = 0

    def _incomplete_gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        _build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_incomplete_authoring_finalizer",
            succeeded=True,
        )
        return _successful_authoring_gate_outcome()

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        _incomplete_gate,
    )

    assert _run(
        corpus,
        repo,
        skip_build_verify=False,
        build_retries=1,
    ) == 1
    assert backend.calls == ["s0"]
    assert gate_calls == 1
    assert (repo / "a.py").read_bytes() == before
    assert not (repo / "qualification.json").exists()
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == "hard_fail_authoring_finalize"
    evidence = json.loads(
        corpus.get_artifact(
            implementation.artifacts["authoring_finalization_evidence"]
        ).decode("utf-8")
    )
    assert evidence["status"] == "failed"
    assert evidence["missing_paths"] == ["qualification.json"]
    assert evidence["changed_paths"] == []


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX node types")
def test_authoring_materialization_rejects_symlink_fifo_and_directory(
    tmp_path: Path,
) -> None:
    """Non-regular outputs fail closed, and FIFO probing never blocks."""

    from crucible.workflows.tier import _authoring_materialization_state

    (tmp_path / "target.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "symlink.json").symlink_to("target.json")
    os.mkfifo(tmp_path / "fifo.json")
    (tmp_path / "directory.json").mkdir()
    root_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    started = time.monotonic()
    try:
        present, missing, invalid, digests = (
            _authoring_materialization_state(
                root_descriptor,
                ("symlink.json", "fifo.json", "directory.json"),
            )
        )
    finally:
        os.close(root_descriptor)
    assert time.monotonic() - started < 2.0
    assert present == ()
    assert missing == ()
    assert invalid == (
        "symlink.json",
        "fifo.json",
        "directory.json",
    )
    assert digests == {}


def test_serial_transaction_exception_restores_exact_pre_phase_state(
    tmp_path: Path, monkeypatch,
):
    """A raised future rolls back every unadjudicated sibling mutation."""

    repo, corpus = _setup(tmp_path)
    # Pre-existing staged and untracked user work must survive byte-for-byte.
    (repo / "prior.py").write_text("# user's staged prior-tier change\n")
    _git(repo, "add", "prior.py")
    (repo / "notes.txt").write_text("user scratch\n")
    before_status = _git(repo, "status", "--porcelain=v1", "-z")
    before_cached = _git(repo, "diff", "--cached", "--binary", "HEAD")
    before_a = (repo / "a.py").read_bytes()
    before_b = (repo / "b.py").read_bytes()

    backend = _MutationBackend(repo, raise_s0=True)
    _patch_backend(monkeypatch, backend)

    with pytest.raises(RuntimeError, match="implementer exploded"):
        _run(corpus, repo)

    assert (repo / "a.py").read_bytes() == before_a
    assert (repo / "b.py").read_bytes() == before_b
    assert (repo / "prior.py").read_text() == (
        "# user's staged prior-tier change\n"
    )
    assert (repo / "notes.txt").read_text() == "user scratch\n"
    assert not (repo / "rogue.txt").exists()
    assert _git(repo, "status", "--porcelain=v1", "-z") == before_status
    assert _git(repo, "diff", "--cached", "--binary", "HEAD") == before_cached
    assert corpus.latest_spec().implementation_refs == ()


def test_tier_retains_journal_until_persistent_backend_close_is_retried(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A live provider can never race Phase-1 exception rollback."""

    repo, corpus = _setup(tmp_path)
    before_a = (repo / "a.py").read_bytes()
    events: list[str] = []

    class CloseFailingBackend(_MutationBackend):
        def __init__(self) -> None:
            super().__init__(repo, raise_s0=True)
            self.allow_close = False
            self.closed = False

        def close(self) -> None:
            events.append("backend_close")
            if not self.allow_close:
                raise RuntimeError("backend descendants still live")
            self.closed = True

    backend = CloseFailingBackend()
    _patch_backend(monkeypatch, backend)

    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    real_settle = WorkspaceMutationJournal.settle_or_restore

    def observed_settle(self, bound_corpus) -> bool:
        events.append("journal_settle")
        assert backend.closed
        return real_settle(self, bound_corpus)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "settle_or_restore",
        observed_settle,
    )

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        _run(corpus, repo)

    cleanup_owner = captured.value
    assert isinstance(cleanup_owner.primary_error, RuntimeError)
    assert "implementer exploded" in str(cleanup_owner.primary_error)
    assert events == ["backend_close"]
    assert (repo / "a.py").read_text() == "# partial s0\n"
    assert (repo / "rogue.txt").read_text() == "unattributed partial\n"
    assert tuple(corpus.spec_dir.glob("runs/*/phase1_mutations/*/manifest.json"))
    assert tuple((repo / ".git" / "crucible-mutation-journals").glob("*.json"))

    # A second persistent failure retains the same explicit retry owner and
    # still performs no journal settlement.
    with pytest.raises(ProviderSettlementCleanupError) as retried:
        cleanup_owner.retry_cleanup()
    assert retried.value is cleanup_owner
    assert events == ["backend_close", "backend_close"]
    assert (repo / "a.py").read_text() == "# partial s0\n"

    backend.allow_close = True
    cleanup_owner.retry_cleanup()

    assert events == [
        "backend_close",
        "backend_close",
        "backend_close",
        "journal_settle",
    ]
    assert (repo / "a.py").read_bytes() == before_a
    assert not (repo / "rogue.txt").exists()
    assert not tuple(
        corpus.spec_dir.glob("runs/*/phase1_mutations/*/manifest.json")
    )
    registry = repo / ".git" / "crucible-mutation-journals"
    assert not registry.exists() or not tuple(registry.glob("*.json"))


def test_early_failure_never_invokes_later_sibling(
    tmp_path: Path, monkeypatch,
):
    """Per-slice settlement halts before a later implementer can mutate."""

    repo, corpus = _setup(tmp_path)
    before_b = (repo / "b.py").read_bytes()
    backend = _MutationBackend(repo, raise_s0=False, empty_s0=True)
    _patch_backend(monkeypatch, backend)

    assert _run(corpus, repo) == 1
    assert (repo / "b.py").read_bytes() == before_b
    # Only the adjudicated failing slice is attached; the later sibling is
    # neither invoked nor given an active transaction baseline.
    assert len(corpus.latest_spec().implementation_refs) == 1
    assert backend.calls == ["s0"]


def test_requested_parallelism_serializes_disjoint_mutation_transactions(
    tmp_path: Path, monkeypatch,
):
    """A shared checkout stages sibling mutations in deterministic order."""

    repo, corpus = _setup(tmp_path)
    backend = _MutationBackend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)

    assert _run(corpus, repo) == 0
    assert (repo / "a.py").read_text() == "# partial s0\n"
    assert (repo / "b.py").read_text() == "# completed s1\n"
    from assay import Implementation

    implementations = [
        corpus.get_record(ref, as_type=Implementation)
        for ref in corpus.latest_spec().implementation_refs
    ]
    assert [impl.classification for impl in implementations] == [
        "success", "success",
    ]
    assert all(impl.scope_escapes == () for impl in implementations)
    assert backend.calls == ["s0", "s1"]


def test_tier_refreshes_spec_after_reaper_publishes_cleanup(
    tmp_path: Path,
    monkeypatch,
):
    """Phase 1 journals and CAS bind the reaper's new Spec parent."""

    repo, corpus = _setup(tmp_path)
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._run import exec_ctx_now
    from crucible.workflows.tier import _tier_tool_authority_digests
    from crucible.tools._context import ToolContext
    from crucible.tools.profiles import build_autonomous_implementer_profile

    (repo / "a.py").write_text("# crashed prior attempt\n", encoding="utf-8")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("a.py",),
    )
    failure = Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            diff,
            kind="diffs",
            name_hint="prior-failure.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]",
            kind="transcripts",
            name_hint="prior-failure.json",
        ),
        files_modified=files,
        scope_escapes=(),
        classification="hard_fail_max_turns",
        execution=exec_ctx_now("test"),
    )
    spec = corpus.latest_spec()
    tier = spec.tier_dag.tiers[0]
    context = ToolContext(workspace_root=repo, repo_root=repo)
    profile = build_autonomous_implementer_profile(context)
    tool_digests = _tier_tool_authority_digests(context, tier, profile)
    failure = attach_resume_identity(
        corpus,
        spec,
        tier,
        tier.slices[0],
        failure,
        tool_authority_digest=tool_digests["s0"],
    )
    corpus.commit(implementations=(failure,))
    backend = _MutationBackend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)

    assert _run(corpus, repo) == 0

    latest = corpus.latest_spec()
    events = [
        corpus.get_record(ref, as_type=LifecycleEvent)
        for ref in latest.lifecycle_event_refs
    ]
    assert any(event.kind == "reaped" for event in events)
    assert latest.version >= 4
    assert (repo / "a.py").read_text() == "# partial s0\n"
    assert (repo / "b.py").read_text() == "# completed s1\n"


def test_spec_after_reaper_binds_persisted_raw_manifest_identity(
    tmp_path: Path,
) -> None:
    """Hydration-equivalent bytes cannot replace the reaper's authority."""
    from assay import PlanPolicy
    from assay.canonical import parse, serialize

    from crucible.workflows import _mutation_journal, _reaper
    from crucible.workflows.tier import _spec_after_reaper

    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    root = dataclasses.replace(_spec(base), plan_policy=PlanPolicy())
    legacy_payload = parse(serialize(root))
    del legacy_payload["plan_policy"][
        "max_verification_only_empty_scope_slices"
    ]
    legacy_blob = serialize(legacy_payload)
    legacy_digest = "sha256:" + hashlib.sha256(legacy_blob).hexdigest()
    legacy_path = (
        Path(corpus.spec_dir)
        / "versions"
        / f"v0_{legacy_digest.split(':', 1)[1][:12]}.json"
    )
    legacy_path.write_bytes(legacy_blob)
    report = _reaper.ReaperReport(
        crashed_rollbacks_reaped=(),
        scope_escapes_deleted=(),
        crashed_rollbacks_skipped=(),
        failures=(),
        authority_spec_version=0,
        authority_spec_hash=legacy_digest,
    )

    assert _spec_after_reaper(corpus, report) == root

    # Explicit-null and omitted-default manifests hydrate to the same Spec but
    # are different persisted authorities. Replacing one with the other must
    # invalidate the report's exact binding.
    legacy_path.unlink()
    replacement_blob = serialize(root)
    replacement_digest = hashlib.sha256(replacement_blob).hexdigest()
    (
        Path(corpus.spec_dir)
        / "versions"
        / f"v0_{replacement_digest[:12]}.json"
    ).write_bytes(replacement_blob)
    with pytest.raises(
        _mutation_journal.MutationJournalError,
        match="Spec authority changed",
    ):
        _spec_after_reaper(corpus, report)


def test_tier_allows_inert_filter_and_restores_filtered_scope_escape(
    tmp_path: Path,
    monkeypatch,
):
    """Assay capture stays inert while the journal protects filtered bytes."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / ".gitattributes").write_text(
        "prior.py filter=opaque-store\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-qm", "add inert filtered path")
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    corpus.put_spec(_spec(head))
    before_a = (repo / "a.py").read_bytes()
    before_prior = (repo / "prior.py").read_bytes()

    class _FilteredEscapeBackend(_MutationBackend):
        def invoke(self, invocation):
            self.calls.append("s0")
            (self.repo / "a.py").write_text("# authorized mutation\n")
            (self.repo / "prior.py").write_bytes(b"filtered scope escape\n")
            return _success()

    backend = _FilteredEscapeBackend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)

    assert _run(corpus, repo) == 1
    assert backend.calls == ["s0"]
    assert (repo / "a.py").read_bytes() == before_a
    assert (repo / "prior.py").read_bytes() == before_prior
    impl = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert impl.classification == "hard_fail_scope_escape"
    assert impl.scope_escapes == ("prior.py",)


def test_gate_checkpoint_attributes_runner_write_against_staged_siblings(
    tmp_path: Path,
    monkeypatch,
):
    """A build cannot hide its write behind another staged slice's dirt."""

    repo, corpus = _setup(tmp_path)
    before_a = (repo / "a.py").read_bytes()
    before_b = (repo / "b.py").read_bytes()
    backend = _MutationBackend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)

    from crucible.workflows._build_verify import BuildVerifyOutcome

    def _gate(*args, slice_id, **kwargs):
        if slice_id == "s0":
            (repo / "b.py").write_text("# runner overwrote sibling\n")
        from crucible.workflows import _build_verify

        _build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_stub_gate",
            succeeded=True,
        )
        return BuildVerifyOutcome(
            classification="success",
            build=None,
            verify=None,
        )

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        _gate,
    )
    assert _run(corpus, repo, skip_build_verify=False) == 1
    assert (repo / "a.py").read_bytes() == before_a
    assert (repo / "b.py").read_bytes() == before_b


def test_git_visible_content_drift_defers_to_scope_escape_detection(
    tmp_path: Path,
):
    """Tracked metadata checks must not preempt Git-visible scope escapes."""

    repo, corpus = _setup(tmp_path)
    before_b = (repo / "b.py").read_bytes()

    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "visible_scope_escape" / "mutations"
        ),
        slice_id="visible_scope_escape",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "b.py").write_text("# Git-visible runner escape\n")

    assert journal.detected_scope_escapes() == ("b.py",)
    journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert (repo / "b.py").read_bytes() == before_b


def test_absent_file_scope_does_not_authorize_new_descendants(tmp_path: Path):
    """An absent creates_path is exact even though directory scopes recurse."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "exact_absent_scope" / "mutations"
        ),
        slice_id="exact_absent_scope",
        scope=("required.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "required.py").mkdir()
    (repo / "required.py" / "child.py").write_text("ROGUE = True\n")

    assert journal.detected_scope_escapes() == ("required.py/child.py",)
    assert journal.restore()
    assert not (repo / "required.py").exists()


def test_startup_reaper_recovers_invoking_journal_after_process_death(
    tmp_path: Path,
):
    """An active write-ahead journal covers the SIGKILL callback gap."""

    repo, corpus = _setup(tmp_path)
    (repo / "prior.py").write_text("# staged user state\n")
    _git(repo, "add", "prior.py")
    before_status = _git(repo, "status", "--porcelain=v1", "-z")
    before_cached = _git(repo, "diff", "--cached", "--binary", "HEAD")
    before_a = (repo / "a.py").read_bytes()

    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "dead_tier" / "phase1_mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# killed mid-write\n")
    (repo / "unscoped.py").write_text("# killed mid-escape\n")
    assert journal.state == "invoking"
    _simulate_process_death(journal)

    from crucible.workflows._reaper import reap_orphaned_dirt

    report = reap_orphaned_dirt(corpus, repo)
    assert report.mutation_journals_recovered == ("s0",)
    assert report.failures == ()
    assert (repo / "a.py").read_bytes() == before_a
    assert not (repo / "unscoped.py").exists()
    assert _git(repo, "status", "--porcelain=v1", "-z") == before_status
    assert _git(repo, "diff", "--cached", "--binary", "HEAD") == before_cached
    assert not journal.journal_dir.exists()


def test_startup_reaper_restores_declared_test_diagnostic_child_before_parent(
    tmp_path: Path,
):
    """A crash during the diagnostic gate restores gate then candidate."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt

    journal_root = (
        corpus.spec_dir
        / "runs"
        / "dead_declared_test_diagnostic"
        / "phase1_mutations"
    )
    parent = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="declared_test_parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text(
        "# retained production candidate\n",
        encoding="utf-8",
    )
    child = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="declared_test_diagnostic_gate",
        scope=("a.py",),
        sequence=2,
        parent_transaction_id=parent.transaction_id,
    )
    (repo / "a.py").write_text(
        "# diagnostic gate mutation\n",
        encoding="utf-8",
    )

    _simulate_process_death(child)
    _simulate_process_death(parent)
    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert (repo / "a.py").read_text() == "# baseline a.py\n"
    assert not child.journal_dir.exists()
    assert not parent.journal_dir.exists()


def test_startup_reaper_recovers_unattached_record_through_journal(
    tmp_path: Path,
):
    """A put_record/Spec-attach crash uses journal chronology exactly once."""

    repo, corpus = _setup(tmp_path)
    before = (repo / "a.py").read_bytes()
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "orphan" / "mutations",
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# orphaned candidate\n", encoding="utf-8")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("a.py",),
    )
    impl = Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            diff,
            kind="diffs",
            name_hint="orphan.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]",
            kind="transcripts",
            name_hint="orphan.json",
        ),
        files_modified=files,
        scope_escapes=(),
        classification="success",
        execution=exec_ctx_now("test"),
    )
    journal.prepare(impl, keep=True)
    orphan_ref = corpus.put_record(impl, kind="implementations")
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.mutation_journals_recovered == ("s0",)
    assert report.crashed_rollbacks_reaped == ()
    assert report.unattached_legacy_skipped == (str(orphan_ref),)
    assert (repo / "a.py").read_bytes() == before
    assert not journal.journal_dir.exists()


def test_startup_reaper_preserved_journal_suppresses_legacy_reverse(
    tmp_path: Path,
):
    """Lifecycle-only acceptance blocks an overlapping legacy reverse."""

    repo, corpus = _setup(tmp_path)
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    accepted = "# accepted finalization\n"
    baseline = (repo / "a.py").read_text(encoding="utf-8")
    (repo / "a.py").write_text(accepted, encoding="utf-8")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("a.py",),
    )
    failed = Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            diff,
            kind="diffs",
            name_hint="older-failure.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]",
            kind="transcripts",
            name_hint="older-failure.json",
        ),
        files_modified=files,
        scope_escapes=(),
        classification="hard_fail_max_turns",
        execution=exec_ctx_now("test"),
    )
    spec = corpus.latest_spec()
    tier = spec.tier_dag.tiers[0]
    failed = attach_resume_identity(
        corpus,
        spec,
        tier,
        tier.slices[0],
        failed,
    )
    corpus.commit(implementations=(failed,))
    (repo / "a.py").write_text(baseline, encoding="utf-8")
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "preserved" / "mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    # The accepted bytes must be owned by this journal, not ambient state
    # captured before begin().
    (repo / "a.py").write_text(accepted, encoding="utf-8")
    reason = "accepted external finalization"
    journal.prepare_lifecycle_acceptance(kind="landed", reason=reason)
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason=reason,
            affected_versions=(version,),
            execution=exec_ctx_now("test"),
        ),
    )
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)

    assert any(
        "lifecycle-only prepared_keep" in failure
        and "journal retained" in failure
        for failure in report.failures
    )
    assert report.mutation_journals_preserved == ("s0",)
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "a.py").read_text(encoding="utf-8") == accepted
    assert journal.journal_dir.exists()


@pytest.mark.parametrize(
    "journal_root_name",
    (
        "post_breach_mutations",
        "post_review_mutations",
        "breach_test_mutations",
        "final_gate_mutations",
    ),
)
def test_startup_reaper_discovers_tier_finalization_journal_roots(
    tmp_path: Path,
    journal_root_name: str,
):
    """Every durable post-breach checkpoint is recoverable after SIGKILL."""

    repo, corpus = _setup(tmp_path)
    before = (repo / "a.py").read_bytes()
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / f"dead_{journal_root_name}"
            / journal_root_name
        ),
        slice_id=journal_root_name,
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text(f"# interrupted {journal_root_name}\n")
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)
    assert report.mutation_journals_recovered == (journal_root_name,)
    assert report.failures == ()
    assert (repo / "a.py").read_bytes() == before
    assert not journal.journal_dir.exists()


@pytest.mark.parametrize(
    "journal_root_name",
    (
        "post_breach_mutations",
        "post_review_mutations",
        "breach_test_mutations",
        "final_gate_mutations",
    ),
)
def test_startup_reaper_preserves_attached_finalization_journal(
    tmp_path: Path,
    journal_root_name: str,
):
    """Prepared acceptance in every discovered root survives settle death."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / f"settled_{journal_root_name}"
            / journal_root_name
        ),
        slice_id=journal_root_name,
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    accepted = f"# accepted {journal_root_name}\n"
    (repo / "a.py").write_text(accepted)
    reason = f"accepted {journal_root_name}"
    journal.prepare_lifecycle_acceptance(kind="landed", reason=reason)
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason=reason,
            affected_versions=(version,),
            execution=exec_ctx_now("test"),
        ),
    )
    _simulate_process_death(journal)

    first = reap_orphaned_dirt(corpus, repo)
    second = reap_orphaned_dirt(corpus, repo)
    for report in (first, second):
        assert report.mutation_journals_preserved == (journal_root_name,)
        assert any(
            "lifecycle-only prepared_keep" in failure
            and "journal retained" in failure
            for failure in report.failures
        )
    assert (repo / "a.py").read_text() == accepted
    assert journal.journal_dir.exists()


def test_real_sigkill_leaves_recoverable_mutation_journal(tmp_path: Path):
    """The kernel-death path is exercised without Python cleanup hooks."""

    repo, corpus = _setup(tmp_path)
    before = (repo / "a.py").read_bytes()
    journal_root = (
        corpus.spec_dir / "runs" / "sigkill_tier" / "mutations"
    )
    script = "\n".join((
        "import os, signal, sys",
        "from pathlib import Path",
        "from crucible.workflows._mutation_journal import MutationTransaction",
        "repo, root = Path(sys.argv[1]), Path(sys.argv[2])",
        "MutationTransaction.begin(cwd=repo, journal_root=root, "
        "slice_id='killed', scope=('a.py',), sequence=1)",
        "(repo / 'a.py').write_text('# partial bytes\\n', encoding='utf-8')",
        "(repo / 'rogue.py').write_text('# rogue\\n', encoding='utf-8')",
        "os.kill(os.getpid(), signal.SIGKILL)",
    ))
    child = subprocess.run(
        [sys.executable, "-c", script, str(repo), str(journal_root)],
        check=False,
        timeout=30,
    )
    assert child.returncode == -signal.SIGKILL
    assert (repo / "a.py").read_text() == "# partial bytes\n"

    from crucible.workflows._reaper import reap_orphaned_dirt

    report = reap_orphaned_dirt(corpus, repo)
    assert report.mutation_journals_recovered == ("killed",)
    assert report.failures == ()
    assert (repo / "a.py").read_bytes() == before
    assert not (repo / "rogue.py").exists()


def test_sigkill_before_breach_acceptance_restores_deferred_seed(tmp_path: Path):
    """prepared child intent is not acceptance without its landed lifecycle."""

    repo, corpus = _setup(tmp_path)
    before = (repo / "a.py").read_bytes()
    journal_root = (
        corpus.spec_dir / "runs" / "deferred_sigkill" / "mutations"
    )
    script = "\n".join((
        "import os, signal, sys",
        "from pathlib import Path",
        "from assay.corpus.store import Corpus",
        "from crucible.workflows._mutation_journal import MutationTransaction",
        "repo, corpus_dir, root = map(Path, sys.argv[1:4])",
        "corpus = Corpus(corpus_dir)",
        "tx = MutationTransaction.begin(cwd=repo, journal_root=root, "
        "slice_id='deferred', scope=('a.py',), sequence=1, "
        "spec=corpus.latest_spec())",
        "(repo / 'a.py').write_text('# broken seed\\n', encoding='utf-8')",
        "tx.prepare_lifecycle_acceptances(kind='landed', "
        "reasons=('breach loop proceed',))",
        "os.kill(os.getpid(), signal.SIGKILL)",
    ))
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(repo),
            str(corpus.spec_dir),
            str(journal_root),
        ],
        check=False,
        timeout=30,
    )
    assert child.returncode == -signal.SIGKILL
    assert (repo / "a.py").read_text() == "# broken seed\n"

    from crucible.workflows._reaper import reap_orphaned_dirt

    report = reap_orphaned_dirt(corpus, repo)
    assert report.mutation_journals_recovered == ("deferred",)
    assert report.mutation_journals_preserved == ()
    assert (repo / "a.py").read_bytes() == before


def test_attached_failed_seed_is_not_child_acceptance_proof(tmp_path: Path):
    """Switching to child-lifecycle settlement clears the failed impl proof."""

    repo, corpus = _setup(tmp_path)
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "deferred" / "mutations",
        slice_id="deferred",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# broken seed\n")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo), path_scope=("a.py",),
    )
    failed = Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            diff, kind="diffs", name_hint="failed-seed.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint="failed-seed.json",
        ),
        files_modified=files,
        scope_escapes=(),
        classification="dev_verify_failed",
        execution=exec_ctx_now("test"),
    )
    journal.prepare(failed, keep=False)
    corpus.commit(implementations=(failed,))
    journal.prepare_lifecycle_acceptances(
        kind="landed",
        reasons=("breach loop proceed",),
    )
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)
    assert report.mutation_journals_preserved == ()
    assert report.mutation_journals_recovered == ("deferred",)
    assert (repo / "a.py").read_text() == "# baseline a.py\n"


def test_full_rollback_excludes_workflow_metadata_inside_checkout(
    tmp_path: Path,
):
    """A repo-local corpus/run directory is metadata, never agent dirt."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    run_dir = repo / "local-corpus" / "runs" / "active-tier"
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run_dir / "phase1_mutations",
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
    )
    log = run_dir / "stderr.log"
    log.write_text("diagnostic must survive rollback\n")
    (repo / "a.py").write_text("# interrupted\n")
    assert journal.restore(full=True)
    assert (repo / "a.py").read_text() == "# baseline a.py\n"
    assert log.read_text() == "diagnostic must survive rollback\n"


def _metadata_only_scope_escape_implementation(
    *,
    repo: Path,
    corpus: Corpus,
    journal,
) -> Implementation:
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._run import exec_ctx_now

    metadata_log = journal.journal_dir.parent / "sdk.log"
    metadata_log.write_text("trusted workflow output\n", encoding="utf-8")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo), path_scope=("a.py",),
    )
    return Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            diff, kind="diffs", name_hint="metadata_escape.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint="metadata_escape.json",
        ),
        files_modified=files,
        scope_escapes=(metadata_log.relative_to(repo).as_posix(),),
        classification="hard_fail_scope_escape",
        execution=exec_ctx_now("test"),
    )


def test_metadata_escape_filter_does_not_launder_inherited_diff_as_success(
    tmp_path: Path,
):
    """Cumulative scoped dirt is not proof this invocation changed it."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    (repo / "a.py").write_text("# inherited tier change\n", encoding="utf-8")
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            repo / "local-corpus" / "runs" / "metadata_empty" / "mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    impl = _metadata_only_scope_escape_implementation(
        repo=repo, corpus=corpus, journal=journal,
    )

    validated = journal.validate_implementation(impl)

    assert validated.scope_escapes == ()
    assert validated.files_modified == ("a.py",)
    assert validated.classification == "hard_fail_empty_diff"
    assert journal.restore()


def test_metadata_escape_filter_preserves_empty_soft_failure(tmp_path: Path):
    """The legacy no-files result remains soft after metadata filtering."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            repo / "local-corpus" / "runs" / "metadata_soft" / "mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    impl = _metadata_only_scope_escape_implementation(
        repo=repo, corpus=corpus, journal=journal,
    )

    validated = journal.validate_implementation(impl)

    assert validated.scope_escapes == ()
    assert validated.files_modified == ()
    assert validated.classification == "soft_fail_no_changes"
    assert journal.restore()


def test_metadata_escape_filter_keeps_genuine_scoped_edit_as_success(
    tmp_path: Path,
):
    """Metadata filtering still accepts an invocation-local scoped edit."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    (repo / "a.py").write_text("# inherited tier change\n", encoding="utf-8")
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            repo / "local-corpus" / "runs" / "metadata_edit" / "mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text(
        "# inherited tier change\n# local invocation edit\n",
        encoding="utf-8",
    )
    impl = _metadata_only_scope_escape_implementation(
        repo=repo, corpus=corpus, journal=journal,
    )

    validated = journal.validate_implementation(impl)

    assert validated.scope_escapes == ()
    assert validated.files_modified == ("a.py",)
    assert validated.classification == "success"
    assert journal.restore()


def test_journal_rejects_and_rewinds_model_authored_commit(tmp_path: Path):
    """A backend cannot evade diff rollback by moving HEAD."""

    repo = tmp_path / "repo"
    repo.mkdir()
    baseline_head = _init_repo(repo)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "run" / "phase1_mutations",
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
    )
    (repo / "a.py").write_text("# committed by model\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-qm", "forbidden model commit")
    with pytest.raises(MutationJournalError, match="changed Git HEAD"):
        journal.assert_head_unchanged()
    assert journal.restore()
    assert _git(repo, "rev-parse", "HEAD").decode().strip() == baseline_head
    assert (repo / "a.py").read_text() == "# baseline a.py\n"


@pytest.mark.parametrize(
    (
        "classification",
        "keep",
        "expected_text",
        "report_field",
        "diagnostic",
    ),
    [
        (
            "success",
            True,
            "# attached success\n",
            "mutation_journals_preserved",
            False,
        ),
        (
            "hard_fail_smoke",
            False,
            "# baseline a.py\n",
            "mutation_journals_recovered",
            False,
        ),
        (
            "hard_fail_missing_declared_test_mutation",
            False,
            "# baseline a.py\n",
            "mutation_journals_recovered",
            True,
        ),
    ],
)
def test_startup_reaper_reconciles_prepare_attach_crash_window(
    tmp_path: Path,
    classification: str,
    keep: bool,
    expected_text: str,
    report_field: str,
    diagnostic: bool,
):
    """Recovery decides from the exact attached record, not timing luck."""

    repo, corpus = _setup(tmp_path)
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._gate_evidence import (
        DECLARED_TEST_EVIDENCE_PENDING_ARTIFACT,
    )
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "dead_prepare" / "phase1_mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# attached success\n")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo), path_scope=("a.py",),
    )
    impl = Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(diff, kind="diffs", name_hint="s0.diff"),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint="s0.json",
        ),
        files_modified=files,
        scope_escapes=(),
        classification=classification,
        execution=exec_ctx_now("test"),
        artifacts=(
            {
                DECLARED_TEST_EVIDENCE_PENDING_ARTIFACT: corpus.put_artifact(
                    b"diagnostic",
                    kind="verification_evidence",
                    name_hint="s0.declared-test-diagnostic.json",
                ),
            }
            if diagnostic
            else {}
        ),
    )
    journal.prepare(impl, keep=keep)
    corpus.commit(implementations=(impl,))
    # Simulate death after the atomic Spec attach but before journal.commit().
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)
    assert getattr(report, report_field) == ("s0",)
    assert (repo / "a.py").read_text() == expected_text
    assert not journal.journal_dir.exists()


def test_preexisting_identical_record_cannot_authorize_prepared_keep(
    tmp_path: Path,
):
    """Recovery requires an append after this transaction's Spec baseline."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    impl = Implementation(
        slice_id="s0",
        base_commit=_git(repo, "rev-parse", "HEAD").decode().strip(),
        result_commit=None,
        diff_ref=corpus.put_artifact(
            b"preexisting", kind="diffs", name_hint="existing.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint="existing.json",
        ),
        files_modified=("a.py",),
        scope_escapes=(),
        classification="success",
        execution=exec_ctx_now("test"),
    )
    corpus.commit(implementations=(impl,))
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "false_lineage" / "mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# uncommitted transaction\n")
    journal.prepare(impl, keep=True)
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)
    assert report.mutation_journals_preserved == ()
    assert report.mutation_journals_recovered == ("s0",)
    assert (repo / "a.py").read_text() == "# baseline a.py\n"


@pytest.mark.parametrize(
    ("legacy_projection", "expected_field", "expected_text"),
    [
        (False, "mutation_journals_preserved", "# exact candidate\n"),
        (True, None, "# exact candidate\n"),
    ],
)
def test_prepared_keep_uses_exact_record_ref_not_legacy_field_collision(
    tmp_path: Path,
    legacy_projection: bool,
    expected_field: str | None,
    expected_text: str,
):
    """The former five-field projection can collide across real records."""

    repo, corpus = _setup(tmp_path)
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "ref_collision" / "mutations",
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# exact candidate\n", encoding="utf-8")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("a.py",),
    )
    impl = Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(diff, kind="diffs", name_hint="exact.diff"),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint="exact.json",
        ),
        files_modified=files,
        scope_escapes=(),
        classification="success",
        execution=exec_ctx_now("test"),
    )
    collision = dataclasses.replace(impl, cost_usd=1.0)
    journal.prepare(impl, keep=True)
    if legacy_projection:
        journal._manifest["expected_implementation"].pop("record_ref")
        journal._write()
    corpus.commit(implementations=(impl, collision))
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)

    if expected_field is None:
        assert report.mutation_journals_recovered == ()
        assert report.mutation_journals_preserved == ()
        assert any(
            "legacy prepared keep matches multiple attached Implementations"
            in failure
            for failure in report.failures
        )
        assert journal.journal_dir.exists()
    else:
        assert getattr(report, expected_field) == ("s0",)
    assert (repo / "a.py").read_text(encoding="utf-8") == expected_text


def test_unreadable_attached_expected_record_retains_prepared_keep(
    tmp_path: Path,
):
    """Corrupt accepted evidence cannot authorize destructive recovery."""

    repo, corpus = _setup(tmp_path)
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "corrupt_accept" / "mutations",
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# accepted bytes\n", encoding="utf-8")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("a.py",),
    )
    impl = Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            diff,
            kind="diffs",
            name_hint="corrupt-accept.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]",
            kind="transcripts",
            name_hint="corrupt-accept.json",
        ),
        files_modified=files,
        scope_escapes=(),
        classification="success",
        execution=exec_ctx_now("test"),
    )
    journal.prepare(impl, keep=True)
    committed = corpus.commit(implementations=(impl,))
    ref = str(committed.refs.implementation_refs[0])
    record_path = (
        corpus.spec_dir
        / "artifacts"
        / "implementations"
        / f"{ref.split(':', 1)[1]}.bin"
    )
    record_path.write_bytes(b"corrupt")
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.mutation_journals_recovered == ()
    assert report.mutation_journals_preserved == ()
    assert any(
        "attached expected Implementation is unreadable" in failure
        for failure in report.failures
    )
    assert journal.journal_dir.exists()
    assert (repo / "a.py").read_text(encoding="utf-8") == "# accepted bytes\n"


def test_unreadable_prepared_keep_baseline_retains_source(
    tmp_path: Path,
    monkeypatch,
):
    """Unreadable lineage is ambiguous, not proof that acceptance is absent."""

    repo, corpus = _setup(tmp_path)
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "bad_lineage" / "mutations",
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    baseline_version = journal._manifest["lineage"]["spec_version"]
    (repo / "a.py").write_text("# accepted bytes\n", encoding="utf-8")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("a.py",),
    )
    impl = Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(diff, kind="diffs", name_hint="lineage.diff"),
        transcript_ref=corpus.put_artifact(
            b"[]",
            kind="transcripts",
            name_hint="lineage.json",
        ),
        files_modified=files,
        scope_escapes=(),
        classification="success",
        execution=exec_ctx_now("test"),
    )
    journal.prepare(impl, keep=True)
    corpus.commit(implementations=(impl,))
    _simulate_process_death(journal)
    original_get_spec = corpus.get_spec
    original_get_spec_with_digest = corpus.get_spec_with_digest

    def _unreadable_baseline(version):
        if version == baseline_version:
            raise OSError("baseline unavailable")
        return original_get_spec(version)

    def _unreadable_baseline_with_digest(version):
        if version == baseline_version:
            raise OSError("baseline unavailable")
        return original_get_spec_with_digest(version)

    monkeypatch.setattr(corpus, "get_spec", _unreadable_baseline)
    monkeypatch.setattr(
        corpus,
        "get_spec_with_digest",
        _unreadable_baseline_with_digest,
    )

    report = reap_orphaned_dirt(corpus, repo)

    assert report.mutation_journals_recovered == ()
    assert any(
        "latest Spec lineage is invalid before recovery" in failure
        or (
            "preserved recovery journal Implementation lacks" in failure
            and "authority" in failure
        )
        for failure in report.failures
    )
    assert journal.journal_dir.exists()
    assert (repo / "a.py").read_text(encoding="utf-8") == "# accepted bytes\n"


def test_accepted_transaction_without_spec_lineage_is_rejected(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "no_lineage" / "mutations",
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
    )
    impl = Implementation(
        slice_id="s0",
        base_commit=_git(repo, "rev-parse", "HEAD").decode().strip(),
        result_commit=None,
        diff_ref=corpus.put_artifact(b"x", kind="diffs", name_hint="x.diff"),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint="x.json",
        ),
        files_modified=("a.py",),
        scope_escapes=(),
        classification="success",
        execution=exec_ctx_now("test"),
    )
    with pytest.raises(MutationJournalError, match="no Spec lineage"):
        journal.prepare(impl, keep=True)
    assert journal.restore()


def test_workspace_mutation_lock_is_reentrant_for_composed_workflows(
    tmp_path: Path,
):
    repo, _ = _setup(tmp_path)
    from crucible.workflows._mutation_journal import workspace_mutation_lock

    with workspace_mutation_lock(repo):
        with workspace_mutation_lock(repo):
            assert (repo / "a.py").is_file()


def test_tier_retained_recovery_lock_return_interrupt_releases_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source lease closes Tier's acquired-lock return-value gap."""

    from types import SimpleNamespace

    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows import tier as tier_workflow

    repo, corpus = _setup(tmp_path)
    retained = journals.recovery_workspace_mutation_lock(corpus, repo)
    monkeypatch.setattr(
        tier_workflow._mutation_journal,
        "recovery_workspace_mutation_lock",
        lambda *_args, **_kwargs: retained,
    )
    enter_code = type(retained).__enter__.__code__
    interrupt = KeyboardInterrupt("Tier lock __enter__ return boundary")
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
        with pytest.raises(KeyboardInterrupt) as caught:
            tier_workflow._run_tier_body(
                SimpleNamespace(),
                str(repo),
                object(),
                corpus,
                object(),
                object(),
            )
    finally:
        sys.settrace(previous_trace)

    assert caught.value is interrupt
    assert injected
    assert retained.cleanup_complete
    assert journals._active_workspace_mutation_authority(repo) is None


def test_tier_attestation_lock_return_interrupt_releases_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The post-run attestor uses the same pre-registered recovery owner."""

    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows import tier as tier_workflow

    repo, corpus = _setup(tmp_path)
    retained = journals.recovery_workspace_mutation_lock(corpus, repo)
    monkeypatch.setattr(
        tier_workflow._mutation_journal,
        "recovery_workspace_mutation_lock",
        lambda *_args, **_kwargs: retained,
    )
    enter_code = type(retained).__enter__.__code__
    interrupt = KeyboardInterrupt("attestation lock return boundary")
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
        with pytest.raises(KeyboardInterrupt) as caught:
            tier_workflow.attest_mutation_journal_settlement(
                corpus=corpus,
                cwd=repo,
                tier_id="t0",
                operation_id="tier-0",
            )
    finally:
        sys.settrace(previous_trace)

    assert caught.value is interrupt
    assert injected
    assert retained.cleanup_complete
    assert journals._active_workspace_mutation_authority(repo) is None


def _stub_tier_terminal_boundary(
    monkeypatch,
    *,
    lock_error=None,
    tier_error=None,
    staged_status: str | None = None,
    stage_result: bool = True,
    summary_cost_usd: float = 4.25,
    summary_debt: bool = False,
    summary_write_error: BaseException | None = None,
    staged_error: BaseException | None = None,
    tier_error_factory=None,
    run_entry_error: BaseException | None = None,
    publish_run_entry_failure: bool = False,
):
    """Drive Tier's outer authority boundary without its deep phase cascade."""

    from contextlib import contextmanager
    from types import SimpleNamespace

    from assay.corpus.runs import RunResult
    from crucible.workflows import tier as tier_workflow
    events: list[object] = []

    class _RawHandle:
        finalized = False
        elapsed_s = 1.0
        run_dir = Path("/tmp/stub-tier-run")

        def __init__(self) -> None:
            self.results: list[RunResult] = []
            self.summaries: list[dict] = []
            self.summary_write_error = summary_write_error

        def write_summary(self, summary) -> None:
            self.summaries.append(dict(summary))
            error = self.summary_write_error
            self.summary_write_error = None
            if error is not None:
                raise error

        def finalize_elapsed(self, result: RunResult) -> None:
            assert not self.finalized
            self.finalized = True
            self.results.append(result)
            events.append(("publish", result.status, result.cost_usd))

    raw_handle = _RawHandle()
    locked_spec = object()
    locked_tier = SimpleNamespace(tier_id="t0", slices=())

    class _Corpus:
        @contextmanager
        def locked_spec_authority(self):
            yield locked_spec

    @contextmanager
    def _workspace_lock(_corpus, _cwd):
        events.append("workspace_enter")
        try:
            yield SimpleNamespace(root_descriptor=73)
        finally:
            events.append("workspace_exit")
            if lock_error is not None:
                raise lock_error

    @contextmanager
    def _open_run(_corpus, _workflow, **kwargs):
        events.append("run_enter")
        if run_entry_error is not None:
            if publish_run_entry_failure:
                kwargs["failure_terminal"](run_entry_error)
                events.append("run_terminal_published")
            raise run_entry_error
        try:
            yield raw_handle
        except BaseException as exc:  # noqa: BLE001 - identity seam
            events.append(("run_exception", exc))
            if not raw_handle.finalized:
                failure_terminal = kwargs["failure_terminal"](exc)
                raw_handle.finalize_elapsed(
                    RunResult(
                        status=(
                            failure_terminal.status
                            if failure_terminal is not None
                            else "failed"
                        ),
                        cost_usd=(
                            kwargs["failure_cost_usd"]()
                            if kwargs.get("failure_cost_usd") is not None
                            else 0.0
                        ),
                        error_kind=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
            raise
        finally:
            events.append("run_exit")

    def _locked_body(
        *_args,
        handle,
        phase1_rollback_stack,
        **_kwargs,
    ):
        with phase1_rollback_stack:
            effective_tier_error = (
                tier_error_factory(phase1_rollback_stack)
                if tier_error_factory is not None
                else tier_error
            )
            events.append("provider_body")
            handle.write_summary({
                "status": "tier_complete",
                "costs": {
                    "total_usd": summary_cost_usd,
                    "accounting_debt": summary_debt,
                    "rollup_incomplete": summary_debt,
                },
            })
            status = staged_status or (
                tier_workflow._outcome.TOOL_MUTATION_INDETERMINATE_CLASSIFICATION
                if effective_tier_error is not None
                else "tier_complete"
            )
            if stage_result:
                binding = staged_error
                if binding is None and status != "tier_complete":
                    binding = effective_tier_error
                if binding is not None:
                    handle.bind_terminal_error(binding)
                handle.finalize_elapsed(RunResult(
                    status=status,
                    cost_usd=summary_cost_usd,
                    error_kind=(
                        type(binding).__name__ if binding is not None else None
                    ),
                    error_message=(
                        str(binding)[:2000] if binding is not None else None
                    ),
                ))
            events.append("provider_cleanup")
            if effective_tier_error is not None:
                raise effective_tier_error
        return 0

    monkeypatch.setattr(
        tier_workflow._mutation_journal,
        "recovery_workspace_mutation_lock",
        _workspace_lock,
    )
    monkeypatch.setattr(tier_workflow, "open_run_with_capture", _open_run)
    monkeypatch.setattr(tier_workflow, "_run_tier_locked_body", _locked_body)
    monkeypatch.setattr(
        tier_workflow,
        "_persisted_spec_digest",
        lambda *_args: "sha256:stub",
    )
    monkeypatch.setattr(
        tier_workflow,
        "_find_tier",
        lambda *_args: locked_tier,
    )
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "runner_parameters",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        tier_workflow,
        "_spec_execution_structure_hash",
        lambda *_args: "sha256:structure",
    )
    args = SimpleNamespace(
        breach_policy=SimpleNamespace(enabled=False),
        skip_build_verify=False,
        tier_id="t0",
        writer_frontier=None,
    )
    ctx = SimpleNamespace(container_id=None)
    return (
        tier_workflow,
        args,
        ctx,
        _Corpus(),
        raw_handle,
        events,
    )


def test_tier_publishes_terminal_after_provider_and_workspace_cleanup(
    monkeypatch,
) -> None:
    tier_workflow, args, ctx, corpus, handle, events = _stub_tier_terminal_boundary(
        monkeypatch
    )

    assert (
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )
        == 0
    )

    assert handle.results[0].status == "tier_complete"
    assert events.index("provider_cleanup") < events.index("workspace_exit")
    assert events.index("workspace_exit") < events.index(
        ("publish", "tier_complete", 4.25)
    )


def test_tier_late_workspace_baseexception_replaces_staged_success(
    monkeypatch,
) -> None:
    class _LateWorkspaceExit(BaseException):
        pass

    sentinel = _LateWorkspaceExit("late workspace release failed")
    tier_workflow, args, ctx, corpus, handle, events = _stub_tier_terminal_boundary(
        monkeypatch, lock_error=sentinel
    )

    with pytest.raises(_LateWorkspaceExit) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is sentinel
    assert [result.status for result in handle.results] == [
        "incomplete_accounting"
    ]
    assert handle.results[0].cost_usd == 4.25
    assert ("publish", "tier_complete", 4.25) not in events
    assert ("run_exception", sentinel) in events


def test_tier_late_workspace_failure_preserves_accounting_debt_terminal(
    monkeypatch,
) -> None:
    class _LateWorkspaceExit(BaseException):
        pass

    sentinel = _LateWorkspaceExit("late workspace release failed")
    tier_workflow, args, ctx, corpus, handle, _events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            lock_error=sentinel,
            staged_status="incomplete_accounting",
        )
    )

    with pytest.raises(_LateWorkspaceExit) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is sentinel
    assert [result.status for result in handle.results] == [
        "incomplete_accounting"
    ]
    assert handle.results[0].cost_usd == 4.25


def test_tier_workspace_failure_preserves_existing_tmi_retry_owner(
    monkeypatch,
) -> None:
    from crucible.tools import ToolMutationIndeterminate

    retry_owner = RuntimeError("writer cleanup retry authority")
    workspace_error = RuntimeError("workspace release also failed")
    terminal = ToolMutationIndeterminate("exact writer mutation terminal")
    terminal.__cause__ = retry_owner
    terminal.__suppress_context__ = True
    tier_workflow, args, ctx, corpus, handle, _events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            lock_error=workspace_error,
            tier_error=terminal,
        )
    )

    with pytest.raises(ToolMutationIndeterminate) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is terminal
    assert caught.value.__cause__ is retry_owner
    assert any(
        "secondary workspace release failure: RuntimeError" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    workspace_owner = getattr(
        caught.value,
        tier_workflow._WORKSPACE_CLEANUP_OWNER_ATTR,
    )
    assert workspace_owner.cleanup_complete is False
    with pytest.raises(RuntimeError) as repeated:
        workspace_owner.retry_cleanup()
    assert repeated.value is workspace_error
    assert workspace_owner.cleanup_complete is False
    assert [result.status for result in handle.results] == [
        "incomplete_accounting"
    ]


def test_tier_failure_without_staged_result_keeps_checkpointed_cost(
    monkeypatch,
) -> None:
    sentinel = RuntimeError("failure after paid call checkpoint")
    tier_workflow, args, ctx, corpus, handle, _events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            tier_error=sentinel,
            stage_result=False,
        )
    )

    with pytest.raises(RuntimeError) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is sentinel
    assert [result.status for result in handle.results] == ["failed"]
    assert handle.results[0].cost_usd == 4.25


def test_tier_summary_write_failure_keeps_in_memory_checkpointed_cost(
    monkeypatch,
) -> None:
    sentinel = OSError("summary checkpoint persistence failed")
    tier_workflow, args, ctx, corpus, handle, _events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            summary_write_error=sentinel,
            stage_result=False,
        )
    )

    with pytest.raises(OSError) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is sentinel
    assert [result.status for result in handle.results] == ["failed"]
    assert handle.results[0].cost_usd == 4.25


def test_tier_malformed_later_checkpoint_retains_prior_cost_lower_bound(
) -> None:
    from types import SimpleNamespace

    from crucible.workflows import tier as tier_workflow

    raw = SimpleNamespace(
        finalized=False,
        write_summary=lambda _summary: None,
    )
    handle = tier_workflow._DeferredTierRunHandle(raw)
    handle.write_summary({"costs": {"total_usd": 3.5}})

    with pytest.raises(TypeError, match="total_usd must be a real number"):
        handle.write_summary({"costs": {"total_usd": True}})

    assert handle.failure_cost_usd() == 3.5
    terminal = handle.cleanup_failure_terminal(
        RuntimeError("malformed checkpoint")
    )
    assert terminal is not None
    assert terminal.status == "incomplete_accounting"
    assert "TypeError" in terminal.summary_fields["checkpoint_cost_error"]


def test_tier_unstaged_failure_preserves_checkpointed_accounting_debt(
    monkeypatch,
) -> None:
    sentinel = RuntimeError("failure after incomplete paid checkpoint")
    tier_workflow, args, ctx, corpus, handle, _events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            tier_error=sentinel,
            stage_result=False,
            summary_debt=True,
        )
    )

    with pytest.raises(RuntimeError) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is sentinel
    assert [result.status for result in handle.results] == [
        "incomplete_accounting"
    ]
    assert handle.results[0].cost_usd == 4.25


def test_tier_publishes_exact_staged_failure_after_cleanup_then_reraises(
    monkeypatch,
) -> None:
    sentinel = RuntimeError("post-call attestation failed")
    tier_workflow, args, ctx, corpus, handle, events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            tier_error=sentinel,
            staged_status="slice_failed",
        )
    )

    with pytest.raises(RuntimeError) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is sentinel
    assert [result.status for result in handle.results] == ["slice_failed"]
    assert events.index("workspace_exit") < events.index(
        ("publish", "slice_failed", 4.25)
    )


def test_tier_rejects_distinct_same_signature_staged_failure(
    monkeypatch,
) -> None:
    staged = RuntimeError("same failure signature")
    escaped = RuntimeError("same failure signature")
    tier_workflow, args, ctx, corpus, handle, _events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            tier_error=escaped,
            staged_status="slice_failed",
            staged_error=staged,
        )
    )

    with pytest.raises(RuntimeError) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is escaped
    assert [result.status for result in handle.results] == ["failed"]
    assert handle.results[0].error_kind == "RuntimeError"


def test_tier_exact_tmi_and_retry_owner_cross_workspace_before_publication(
    monkeypatch,
) -> None:
    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    terminal = ToolMutationIndeterminate("exact mutation terminal")
    cleanup_error = RuntimeError("provider close remains pending")
    retained: dict[str, ProviderSettlementCleanupError] = {}

    def _owned_terminal(stack: ProviderSettlementExitStack):
        owner = stack.retained_cleanup_error(
            primary_error=terminal,
            cleanup_error=cleanup_error,
        )
        retained["owner"] = owner
        terminal.__cause__ = owner
        terminal.__suppress_context__ = True
        return terminal

    tier_workflow, args, ctx, corpus, handle, events = _stub_tier_terminal_boundary(
        monkeypatch,
        tier_error_factory=_owned_terminal,
    )

    with pytest.raises(ToolMutationIndeterminate) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is terminal
    owner = retained["owner"]
    assert caught.value.__cause__ is owner
    assert owner.primary_error is terminal
    assert [result.status for result in handle.results] == [
        "incomplete_accounting"
    ]
    assert "workspace_exit" not in events
    assert ("run_exception", terminal) in events
    owner.retry_cleanup()
    assert events[-1] == "workspace_exit"
    assert owner.owner.cleanup_complete


def test_tier_provider_cleanup_owner_replaces_staged_success_exactly(
    monkeypatch,
) -> None:
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    cleanup_error = RuntimeError("provider generation did not settle")
    retained: dict[str, ProviderSettlementCleanupError] = {}
    retained_stack: list[ProviderSettlementExitStack] = []

    def _owned_cleanup(stack: ProviderSettlementExitStack):
        retained_stack.append(stack)
        owner = stack.retained_cleanup_error(
            primary_error=cleanup_error,
            cleanup_error=cleanup_error,
        )
        retained["owner"] = owner
        return owner

    tier_workflow, args, ctx, corpus, handle, events = _stub_tier_terminal_boundary(
        monkeypatch,
        tier_error_factory=_owned_cleanup,
        staged_status="tier_complete",
    )

    with pytest.raises(ProviderSettlementCleanupError) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    owner = retained["owner"]
    assert caught.value is owner
    assert retained_stack == [caught.value.owner]
    assert caught.value.cleanup_error is cleanup_error
    assert [result.status for result in handle.results] == [
        "incomplete_accounting"
    ]
    assert handle.results[0].cost_usd == 4.25
    assert ("publish", "tier_complete", 4.25) not in events
    assert "workspace_exit" not in events
    assert ("run_exception", owner) in events
    owner.retry_cleanup()
    assert events[-1] == "workspace_exit"
    assert owner.owner.cleanup_complete


def test_tier_outer_owner_return_interrupt_republishes_prebound_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGINT at the outer recognition return cannot strand final cleanup."""

    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    cleanup_error = RuntimeError("provider generation did not settle")
    retained: dict[str, ProviderSettlementCleanupError] = {}
    retained_stack: list[ProviderSettlementExitStack] = []

    def _owned_cleanup(stack: ProviderSettlementExitStack):
        retained_stack.append(stack)
        owner = stack.retained_cleanup_error(
            primary_error=cleanup_error,
            cleanup_error=cleanup_error,
        )
        retained["owner"] = owner
        return owner

    tier_workflow, args, ctx, corpus, handle, events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            tier_error_factory=_owned_cleanup,
            staged_status="tier_complete",
        )
    )
    helper_code = (
        tier_workflow._defer_tier_workspace_cleanup_to_provider_owner.__code__
    )
    interrupt = KeyboardInterrupt("Tier final-owner return boundary")
    injected = False

    def interrupt_return(frame, event, value):
        nonlocal injected
        if (
            not injected
            and frame.f_code is helper_code
            and event == "return"
            and isinstance(value, ProviderSettlementCleanupError)
        ):
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            tier_workflow._run_tier_body(
                args,
                "/stub/repo",
                ctx,
                corpus,
                object(),
                object(),
            )
    finally:
        sys.settrace(previous_trace)

    owner = retained["owner"]
    assert caught.value is interrupt
    assert injected
    assert caught.value.__cause__ is owner
    assert retained_stack == [owner.owner]
    assert [result.status for result in handle.results] == [
        "incomplete_accounting"
    ]
    assert "workspace_exit" not in events

    owner.retry_cleanup()
    assert events[-1] == "workspace_exit"
    assert owner.owner.cleanup_complete


def test_tier_publication_recognition_return_interrupt_uses_prebound_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGINT inside the outer boundary still exposes its cached final owner."""

    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    cleanup_error = RuntimeError("provider generation did not settle")
    retained: dict[str, ProviderSettlementCleanupError] = {}

    def _owned_cleanup(stack: ProviderSettlementExitStack):
        owner = stack.retained_cleanup_error(
            primary_error=cleanup_error,
            cleanup_error=cleanup_error,
        )
        retained["direct"] = owner
        return owner

    tier_workflow, args, ctx, corpus, handle, events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            tier_error_factory=_owned_cleanup,
            staged_status="tier_complete",
        )
    )
    recognition_code = (
        tier_workflow._TierSettlementStack.retain_pending_cleanup_error.__code__
    )
    boundary_exit_code = tier_workflow._TierCleanupPublicationBoundary.__exit__.__code__
    interrupt = KeyboardInterrupt("Tier publication recognition return boundary")
    injected = False

    def interrupt_return(frame, event, _value):
        nonlocal injected
        caller = frame.f_back
        if (
            not injected
            and frame.f_code is recognition_code
            and event == "return"
            and caller is not None
            and caller.f_code is boundary_exit_code
        ):
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            tier_workflow._run_tier_body(
                args,
                "/stub/repo",
                ctx,
                corpus,
                object(),
                object(),
            )
    finally:
        sys.settrace(previous_trace)

    direct = retained["direct"]
    final = caught.value.__cause__
    assert caught.value is interrupt
    assert injected
    assert isinstance(final, ProviderSettlementCleanupError)
    assert final is not direct
    assert final.owner is direct.owner
    assert final.primary_error is interrupt
    assert final.cleanup_error is direct
    assert [result.status for result in handle.results] == [
        "incomplete_accounting"
    ]
    assert "workspace_exit" not in events

    final.retry_cleanup()
    assert events[-1] == "workspace_exit"
    assert final.owner.cleanup_complete


def test_tier_direct_cleanup_factory_return_interrupt_uses_prebound_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGINT before direct-owner publication cannot strand Tier rollback."""

    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    tier_workflow, args, ctx, corpus, handle, events = (
        _stub_tier_terminal_boundary(monkeypatch, stage_result=False)
    )
    rollback_error = RuntimeError("Tier rollback remains pending")
    failure_pending = True

    def _rollback() -> None:
        events.append("parent_rollback")
        if failure_pending:
            raise rollback_error

    def _locked_body(
        *_args,
        phase1_rollback_stack,
        **_kwargs,
    ) -> int:
        with phase1_rollback_stack:
            phase1_rollback_stack.callback(_rollback)
        return 0

    monkeypatch.setattr(tier_workflow, "_run_tier_locked_body", _locked_body)
    factory_code = ProviderSettlementExitStack._new_cleanup_error.__code__
    publisher_code = (
        ProviderSettlementExitStack._raise_retained_cleanup_owner.__code__
    )
    interrupt = KeyboardInterrupt("direct cleanup owner factory return boundary")
    injected = False

    def interrupt_return(frame, event, value):
        nonlocal injected
        caller = frame.f_back
        if (
            not injected
            and frame.f_code is factory_code
            and event == "return"
            and isinstance(value, ProviderSettlementCleanupError)
            and caller is not None
            and caller.f_code is publisher_code
        ):
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            tier_workflow._run_tier_body(
                args,
                "/stub/repo",
                ctx,
                corpus,
                object(),
                object(),
            )
    finally:
        sys.settrace(previous_trace)

    final = caught.value.__cause__
    assert caught.value is interrupt
    assert injected
    assert isinstance(final, ProviderSettlementCleanupError)
    assert final.primary_error is interrupt
    assert final.cleanup_error is interrupt
    assert [result.status for result in handle.results] == [
        "incomplete_accounting"
    ]
    assert events == [
        "workspace_enter",
        "run_enter",
        "parent_rollback",
        ("run_exception", interrupt),
        ("publish", "incomplete_accounting", 0.0),
        "run_exit",
    ]

    failure_pending = False
    final.retry_cleanup()
    assert events[-2:] == ["parent_rollback", "workspace_exit"]
    assert final.owner.cleanup_complete


def test_tier_foreign_provider_owner_cannot_retain_workspace_lock(
    monkeypatch,
) -> None:
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    foreign_stack = ProviderSettlementExitStack()
    cleanup_error = RuntimeError("foreign provider cleanup remains pending")
    owner = ProviderSettlementCleanupError(
        foreign_stack,
        primary_error=cleanup_error,
        cleanup_error=cleanup_error,
    )
    tier_workflow, args, ctx, corpus, handle, events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            tier_error=owner,
            staged_status="tier_complete",
        )
    )

    with pytest.raises(ProviderSettlementCleanupError) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is owner
    assert owner.owner is foreign_stack
    assert [result.status for result in handle.results] == ["failed"]
    assert events.index("workspace_exit") < events.index(
        ("run_exception", owner)
    )


@pytest.mark.parametrize("outer_kind", ("cleanup_owner", "mutation_terminal"))
@pytest.mark.parametrize(
    "failure_kind",
    ("child_error", "parent_error", "child_interrupt", "parent_interrupt"),
)
def test_tier_breach_child_owner_precedes_parent_rollback_and_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outer_kind: str,
    failure_kind: str,
) -> None:
    """The prebound Tier owner settles child before every parent authority."""

    from types import SimpleNamespace

    from crucible.common.cost import CostAccumulator
    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    events: list[str] = []
    failure_pending = True
    interrupt = KeyboardInterrupt(f"{failure_kind} during cleanup")

    def _explicit_cause_chain(error: BaseException) -> list[BaseException]:
        chain: list[BaseException] = []
        current: BaseException | None = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(current)
            cause = current.__cause__
            current = cause if isinstance(cause, BaseException) else None
        return chain

    class _ChildOwner:
        def __init__(self) -> None:
            self.cleanup_complete = False

        def retry_cleanup(self) -> None:
            nonlocal failure_pending
            events.append("child_cleanup")
            self.cleanup_complete = True
            if failure_pending and failure_kind == "child_error":
                self.cleanup_complete = False
                raise RuntimeError("child provider cleanup remains pending")
            if failure_pending and failure_kind == "child_interrupt":
                failure_pending = False
                raise interrupt

    def _parent_rollback() -> None:
        nonlocal failure_pending
        events.append("parent_rollback")
        if failure_pending and failure_kind == "parent_error":
            raise RuntimeError("parent rollback remains pending")
        if failure_pending and failure_kind == "parent_interrupt":
            failure_pending = False
            raise interrupt

    class _HostileChildCleanup(ProviderSettlementCleanupError):
        """Fail if Tier reads or replaces the child's public owner."""

        def __init__(self, owner: _ChildOwner) -> None:
            self._bound_owner = None
            self.owner_reads = 0
            self.owner_writes = 0
            self.owner_access_armed = False
            super().__init__(
                owner,  # type: ignore[arg-type]
                primary_error=child_cause,
                cleanup_error=child_cause,
            )
            self.owner_reads = 0
            self.owner_writes = 0
            self.owner_access_armed = True

        @property
        def owner(self):
            self.owner_reads += 1
            if self.owner_access_armed:
                raise AssertionError("Tier read the child cleanup owner")
            return self._bound_owner

        @owner.setter
        def owner(self, value):
            self.owner_writes += 1
            if self.owner_access_armed:
                raise AssertionError("Tier replaced the child cleanup owner")
            self._bound_owner = value

    child_cause = RuntimeError("child provider descendants remain live")
    original_child_owner = _ChildOwner()
    child_cleanup = _HostileChildCleanup(original_child_owner)
    child_cleanup.__cause__ = child_cause
    child_cleanup.__suppress_context__ = True
    if outer_kind == "mutation_terminal":
        outer: BaseException = ToolMutationIndeterminate(
            "child mutation settlement is unknown"
        )
        outer.__cause__ = child_cleanup
        outer.__suppress_context__ = True
    else:
        outer = child_cleanup

    class _Handle:
        run_dir = tmp_path

        def __init__(self) -> None:
            self.summaries: list[dict] = []

        def write_summary(self, summary) -> None:
            self.summaries.append(
                json.loads(json.dumps(summary, default=str))
            )

    handle = _Handle()
    costs = CostAccumulator()
    summary = {
        "tier_id": "t0",
        "phase_3_breach": None,
        "token_rollup_incomplete": False,
    }
    committed_outcome = SimpleNamespace(
        cost_usd=0.625,
        detail_status="incomplete_accounting",
        terminal_class=SimpleNamespace(value="inconclusive"),
    )

    monkeypatch.setattr(
        tier_workflow._outcome,
        "child_operation",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        tier_workflow._outcome,
        "child_outcome",
        lambda *_args, **_kwargs: committed_outcome,
    )
    monkeypatch.setattr(
        tier_workflow._outcome,
        "child_run_dir",
        lambda *_args, **_kwargs: pytest.fail(
            "live-child run details must not be read"
        ),
    )

    def _escape_child(*_args, **_kwargs) -> int:
        raise outer

    monkeypatch.setattr(tier_workflow.breach_workflow, "main", _escape_child)

    workspace = tier_workflow._RetainedTierCleanupStack()
    workspace.callback(events.append, "workspace_release")
    parent = tier_workflow._TierSettlementStack(workspace)
    retained_parent = None
    with pytest.raises(BaseException) as captured:
        try:
            with parent:
                parent.callback(_parent_rollback)
                tier_workflow._invoke_tier_breach_child(
                    argv=["--stub"],
                    tier_id="t0",
                    parent_run_dir=tmp_path,
                    fresh_lineage=False,
                    development_budget_ledger=None,
                    phase1_rollback_stack=parent,
                    costs=costs,
                    summary=summary,
                    handle=handle,
                )
        except BaseException as escaped:  # noqa: BLE001 - identity seam
            retained_parent = (
                tier_workflow._defer_tier_workspace_cleanup_to_provider_owner(
                    escaped,
                    expected_owner=parent,
                    workspace_release=workspace,
                )
            )
            raise

    assert captured.value is outer
    parent_hold = retained_parent
    assert isinstance(parent_hold, ProviderSettlementCleanupError)
    assert retained_parent is parent_hold
    assert parent_hold.primary_error is outer
    assert parent_hold.__cause__ is child_cause
    assert parent_hold.owner is parent
    assert outer.__cause__ is parent_hold
    assert parent_hold in _explicit_cause_chain(outer)
    assert parent_hold.cleanup_error is child_cleanup
    assert child_cleanup._bound_owner is original_child_owner
    assert child_cleanup.owner_reads == 0
    assert child_cleanup.owner_writes == 0
    assert events == []
    assert costs.total_cost_usd == pytest.approx(0.625)
    assert costs.to_dict()["accounting_debt_by_role"] == {
        "breach_subprocess": ["cost_complete", "token_usage_complete"],
    }
    assert handle.summaries[-1]["phase_3_breach"][
        "cost_reconciliation"
    ]["source"] == "committed_child_outcome_lower_bound"

    # The child's owner is deliberately left untouched.  Its own public retry
    # can settle only the child; the supported final retry surface is the Tier
    # error that was prebound before Breach dispatched.
    child_cleanup.owner_access_armed = False
    retry_error = parent_hold
    if failure_kind.endswith("_interrupt"):
        with pytest.raises(KeyboardInterrupt) as retry_interrupted:
            retry_error.retry_cleanup()
        assert retry_interrupted.value is interrupt
        expected_events = (
            ["child_cleanup", "parent_rollback", "workspace_release"]
            if failure_kind == "child_interrupt"
            else [
                "child_cleanup",
                "parent_rollback",
                "parent_rollback",
                "workspace_release",
            ]
        )
        assert events == expected_events
    else:
        with pytest.raises(ProviderSettlementCleanupError) as retry_failed:
            retry_error.retry_cleanup()
        assert retry_failed.value is retry_error
        assert events == (
            ["child_cleanup"]
            if failure_kind == "child_error"
            else ["child_cleanup", "parent_rollback"]
        )

        failure_pending = False
        retry_error.retry_cleanup()
        assert events == (
            [
                "child_cleanup",
                "child_cleanup",
                "parent_rollback",
                "workspace_release",
            ]
            if failure_kind == "child_error"
            else [
                "child_cleanup",
                "parent_rollback",
                "parent_rollback",
                "workspace_release",
            ]
        )
    assert parent.cleanup_complete
    assert child_cleanup.owner is original_child_owner
    assert parent_hold.owner is parent


def test_tier_breach_child_transfer_return_interrupt_keeps_final_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGINT at child-transfer return retains child -> parent -> workspace."""

    from crucible.common.cost import CostAccumulator
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    events: list[str] = []

    class _ChildOwner:
        cleanup_complete = False

        def retry_cleanup(self) -> None:
            events.append("child_cleanup")
            self.cleanup_complete = True

    child_owner = _ChildOwner()
    child_error = RuntimeError("child provider descendants remain live")
    child_cleanup = ProviderSettlementCleanupError(
        child_owner,  # type: ignore[arg-type]
        primary_error=child_error,
        cleanup_error=child_error,
    )
    child_cleanup.__cause__ = child_error
    child_cleanup.__suppress_context__ = True

    monkeypatch.setattr(
        tier_workflow._outcome,
        "child_operation",
        lambda *_args, **_kwargs: nullcontext(),
    )

    def _escape_child(*_args, **_kwargs) -> int:
        raise child_cleanup

    monkeypatch.setattr(tier_workflow.breach_workflow, "main", _escape_child)

    workspace = tier_workflow._RetainedTierCleanupStack()
    workspace.callback(events.append, "workspace_release")
    parent = tier_workflow._TierSettlementStack(workspace)
    transfer_code = parent.retain_child_cleanup.__code__
    interrupt = KeyboardInterrupt("child cleanup transfer return boundary")
    injected = False

    def interrupt_return(frame, event, _arg):
        nonlocal injected
        if not injected and frame.f_code is transfer_code and event == "return":
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            with parent:
                parent.callback(events.append, "parent_rollback")
                tier_workflow._invoke_tier_breach_child(
                    argv=["--stub"],
                    tier_id="t0",
                    parent_run_dir=tmp_path,
                    fresh_lineage=False,
                    development_budget_ledger=None,
                    phase1_rollback_stack=parent,
                    costs=CostAccumulator(),
                    summary={"tier_id": "t0", "phase_3_breach": None},
                    handle=object(),
                )
    finally:
        sys.settrace(previous_trace)

    assert caught.value is interrupt
    assert injected
    final_cleanup = caught.value.__cause__
    assert isinstance(final_cleanup, ProviderSettlementCleanupError)
    assert final_cleanup.owner is parent
    assert final_cleanup.cleanup_error is child_cleanup
    assert events == []

    final_cleanup.retry_cleanup()
    assert events == [
        "child_cleanup",
        "parent_rollback",
        "workspace_release",
    ]
    assert parent.cleanup_complete


def test_tier_successful_breach_does_not_activate_child_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful child leaves only ordinary parent/workspace cleanup."""

    from crucible.common.cost import CostAccumulator
    from crucible.workflows import tier as tier_workflow

    monkeypatch.setattr(
        tier_workflow._outcome,
        "child_operation",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        tier_workflow.breach_workflow,
        "main",
        lambda *_args, **_kwargs: 0,
    )

    events: list[str] = []
    workspace = tier_workflow._RetainedTierCleanupStack()
    workspace.callback(events.append, "workspace_release")
    parent = tier_workflow._TierSettlementStack(workspace)

    with parent:
        parent.callback(events.append, "parent_rollback")
        result = tier_workflow._invoke_tier_breach_child(
            argv=["--stub"],
            tier_id="t0",
            parent_run_dir=tmp_path,
            fresh_lineage=False,
            development_budget_ledger=None,
            phase1_rollback_stack=parent,
            costs=CostAccumulator(),
            summary={"tier_id": "t0", "phase_3_breach": None},
            handle=object(),
        )
        assert result == (0, None)
        assert parent.child_cleanup_pending is False
        assert events == []

    assert events == ["parent_rollback"]
    assert not parent.cleanup_complete
    workspace.close()
    assert events == ["parent_rollback", "workspace_release"]
    assert parent.cleanup_complete


def test_tier_ordinary_body_failure_runs_provider_and_mutation_cleanup() -> None:
    """A body error is not misclassified as an unpublished cleanup failure."""

    from crucible.workflows import tier as tier_workflow

    events: list[str] = []
    workspace = tier_workflow._RetainedTierCleanupStack()
    workspace.callback(events.append, "workspace_release")
    owner = tier_workflow._TierSettlementStack(workspace)
    primary = OSError("durable checkpoint write failed")

    with pytest.raises(OSError) as caught:
        with owner:
            owner.callback(events.append, "mutation_rollback")
            raise primary

    assert caught.value is primary
    assert caught.value.__cause__ is None
    assert events == ["mutation_rollback"]
    assert not owner.settlement_pending
    workspace.close()
    assert events == ["mutation_rollback", "workspace_release"]
    assert owner.cleanup_complete


def test_tier_cleanup_owner_replays_settled_interrupt_after_lock_release(
) -> None:
    from crucible.workflows import tier as tier_workflow

    events: list[str] = []
    interrupt = KeyboardInterrupt("provider settled during cancellation")

    workspace = tier_workflow._RetainedTierCleanupStack()
    workspace.callback(events.append, "workspace_released")
    owner = tier_workflow._TierSettlementStack(workspace)
    owner.__enter__()
    owner.callback(events.append, "provider_settled")
    owner._deferred_interrupt = interrupt
    cleanup = owner.retained_cleanup_error(
        primary_error=interrupt,
        cleanup_error=interrupt,
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        cleanup.retry_cleanup()

    assert caught.value is interrupt
    assert events == ["provider_settled", "workspace_released"]
    assert owner.cleanup_complete


def test_tier_retained_generator_release_never_false_completes_on_retry(
) -> None:
    from contextlib import contextmanager

    from crucible.workflows import tier as tier_workflow

    sentinel = RuntimeError("generator lock exit failed")
    exit_attempts = 0

    @contextmanager
    def _failing_generator_lock():
        nonlocal exit_attempts
        try:
            yield object()
        finally:
            exit_attempts += 1
            raise sentinel

    acquisition = tier_workflow._RetainedTierCleanupStack()
    acquisition.enter_context(_failing_generator_lock())
    release = acquisition.pop_all()

    for _attempt in range(2):
        with pytest.raises(RuntimeError) as caught:
            release.retry_cleanup()
        assert caught.value is sentinel
        assert not release.cleanup_complete
    assert exit_attempts == 1


@pytest.mark.parametrize("error_kind", ("runtime", "interrupt"))
def test_tier_cleanup_owner_retains_failed_workspace_release_for_retry(
    error_kind: str,
) -> None:
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    release_error: BaseException = (
        RuntimeError("workspace release failed")
        if error_kind == "runtime"
        else KeyboardInterrupt("workspace release interrupted")
    )
    fail_release = True
    attempts = 0

    def _release_workspace() -> None:
        nonlocal attempts
        attempts += 1
        if fail_release:
            raise release_error

    workspace = tier_workflow._RetainedTierCleanupStack()
    workspace.callback(_release_workspace)
    owner = tier_workflow._TierSettlementStack(workspace)
    owner.__enter__()
    primary = RuntimeError("provider cleanup originally failed")
    cleanup = owner.retained_cleanup_error(
        primary_error=primary,
        cleanup_error=primary,
    )

    with pytest.raises(ProviderSettlementCleanupError) as caught:
        cleanup.retry_cleanup()

    assert caught.value is cleanup
    assert caught.value.__cause__ is release_error
    assert attempts == 1
    assert not owner.cleanup_complete

    fail_release = False
    cleanup.retry_cleanup()
    assert attempts == 2
    assert owner.cleanup_complete


def test_tier_run_entry_failure_releases_workspace_without_second_run(
    monkeypatch,
) -> None:
    sentinel = KeyboardInterrupt("run capture entry interrupted")
    tier_workflow, args, ctx, corpus, handle, events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            run_entry_error=sentinel,
        )
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is sentinel
    assert events == ["workspace_enter", "run_enter", "workspace_exit"]
    assert handle.results == []


def test_tier_preyield_run_failure_releases_workspace_before_publication(
    monkeypatch,
) -> None:
    sentinel = RuntimeError("capture entry failed before yielding run handle")
    tier_workflow, args, ctx, corpus, handle, events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            run_entry_error=sentinel,
            publish_run_entry_failure=True,
        )
    )

    with pytest.raises(RuntimeError) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is sentinel
    assert events == [
        "workspace_enter",
        "run_enter",
        "workspace_exit",
        "run_terminal_published",
    ]
    assert handle.results == []


def test_tier_preyield_cleanup_baseexception_retains_exact_owner(
    monkeypatch,
) -> None:
    run_error = RuntimeError("capture entry failed before yield")
    cleanup_error = KeyboardInterrupt("workspace release interrupted")
    tier_workflow, args, ctx, corpus, handle, events = (
        _stub_tier_terminal_boundary(
            monkeypatch,
            lock_error=cleanup_error,
            run_entry_error=run_error,
            publish_run_entry_failure=True,
        )
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            ctx,
            corpus,
            object(),
            object(),
        )

    assert caught.value is cleanup_error
    assert caught.value.__cause__ is run_error
    owner = getattr(
        caught.value,
        tier_workflow._WORKSPACE_CLEANUP_OWNER_ATTR,
    )
    assert owner.cleanup_complete is False
    with pytest.raises(KeyboardInterrupt) as repeated:
        owner.retry_cleanup()
    assert repeated.value is cleanup_error
    assert owner.cleanup_complete is False
    assert events == [
        "workspace_enter",
        "run_enter",
        "workspace_exit",
        "run_terminal_published",
    ]
    assert handle.results == []


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_tier_real_capture_entry_failure_releases_before_result_publication(
    tmp_path: Path,
    monkeypatch,
    cleanup_fails: bool,
) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    from crucible.workflows import _run as run_workflow
    from crucible.workflows import tier as tier_workflow

    repo, corpus = _setup(tmp_path)
    sentinel = RuntimeError("real capture context failed before yield")
    cleanup_error = KeyboardInterrupt("real workspace release interrupted")
    events: list[str] = []

    @contextmanager
    def _workspace_lock(_corpus, _cwd):
        events.append("workspace_enter")
        try:
            yield SimpleNamespace(root_descriptor=73)
        finally:
            events.append("workspace_exit")
            assert not any(
                path.name == "result.json"
                for run_dir in (corpus.spec_dir / "runs").glob("*_tier_*")
                for path in run_dir.iterdir()
            )
            if cleanup_fails:
                raise cleanup_error

    @contextmanager
    def _failing_capture(_run_dir):
        events.append("capture_enter")
        raise sentinel
        yield  # pragma: no cover - contextmanager shape only

    monkeypatch.setattr(
        tier_workflow._mutation_journal,
        "recovery_workspace_mutation_lock",
        _workspace_lock,
    )
    monkeypatch.setattr(run_workflow, "capture_to_run_dir", _failing_capture)
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "runner_parameters",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        tier_workflow,
        "_spec_execution_structure_hash",
        lambda *_args: "sha256:structure",
    )
    args = SimpleNamespace(
        breach_policy=SimpleNamespace(enabled=False),
        skip_build_verify=False,
        tier_id="t0",
        writer_frontier=None,
    )

    with pytest.raises(BaseException) as caught:
        tier_workflow._run_tier_body(
            args,
            str(repo),
            SimpleNamespace(container_id=None),
            corpus,
            object(),
            object(),
        )

    expected_error = cleanup_error if cleanup_fails else sentinel
    assert caught.value is expected_error
    assert events == ["workspace_enter", "capture_enter", "workspace_exit"]
    run_dir = next((corpus.spec_dir / "runs").glob("*_tier_*"))
    result = json.loads((run_dir / "result.json").read_text())
    assert result["status"] == (
        "incomplete_accounting" if cleanup_fails else "failed"
    )
    if cleanup_fails:
        assert caught.value.__cause__ is sentinel
        owner = getattr(
            caught.value,
            tier_workflow._WORKSPACE_CLEANUP_OWNER_ATTR,
        )
        assert owner.cleanup_complete is False
        with pytest.raises(KeyboardInterrupt) as repeated:
            owner.retry_cleanup()
        assert repeated.value is cleanup_error
        assert owner.cleanup_complete is False


def test_tier_setup_terminal_records_only_after_workspace_cleanup(
    monkeypatch,
) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    from crucible.workflows import tier as tier_workflow

    events: list[str] = []

    class _Corpus:
        @contextmanager
        def locked_spec_authority(self):
            yield None

    @contextmanager
    def _workspace_lock(_corpus, _cwd):
        try:
            yield SimpleNamespace(root_descriptor=31)
        finally:
            events.append("workspace_exit")

    def _record(*_args, **kwargs):
        events.append("record_setup")
        assert kwargs["status"] == "setup_error"
        assert kwargs["error_kind"] == "missing_spec"

    monkeypatch.setattr(
        tier_workflow._mutation_journal,
        "recovery_workspace_mutation_lock",
        _workspace_lock,
    )
    monkeypatch.setattr(tier_workflow, "record_handled_terminal", _record)
    args = SimpleNamespace(
        breach_policy=SimpleNamespace(enabled=False),
        tier_id="t0",
    )

    assert (
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            SimpleNamespace(container_id=None),
            _Corpus(),
            object(),
            object(),
        )
        == 2
    )
    assert events == ["workspace_exit", "record_setup"]


def test_tier_setup_and_workspace_failure_preserve_both_causes(
    monkeypatch,
) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    from crucible.workflows import tier as tier_workflow

    cleanup_error = RuntimeError("workspace cleanup failed after setup")
    events: list[object] = []

    class _Corpus:
        @contextmanager
        def locked_spec_authority(self):
            yield None

    @contextmanager
    def _workspace_lock(_corpus, _cwd):
        try:
            yield SimpleNamespace(root_descriptor=31)
        finally:
            events.append("workspace_exit")
            raise cleanup_error

    @contextmanager
    def _open_run(*_args, **kwargs):
        try:
            yield object()
        except BaseException as exc:  # noqa: BLE001 - identity seam
            terminal = kwargs["failure_terminal"](exc)
            events.append(("failure_terminal", exc, terminal.status))
            raise

    monkeypatch.setattr(
        tier_workflow._mutation_journal,
        "recovery_workspace_mutation_lock",
        _workspace_lock,
    )
    monkeypatch.setattr(tier_workflow, "open_run_with_capture", _open_run)
    monkeypatch.setattr(
        tier_workflow,
        "record_handled_terminal",
        lambda *_args, **_kwargs: pytest.fail(
            "setup success terminal must not replace cleanup failure"
        ),
    )
    args = SimpleNamespace(
        breach_policy=SimpleNamespace(enabled=False),
        tier_id="t0",
    )

    with pytest.raises(RuntimeError) as caught:
        tier_workflow._run_tier_body(
            args,
            "/stub/repo",
            SimpleNamespace(container_id=None),
            _Corpus(),
            object(),
            object(),
        )

    assert caught.value is cleanup_error
    assert isinstance(caught.value.__cause__, tier_workflow._TierSetupTerminal)
    assert caught.value.__cause__.error_kind == "missing_spec"
    assert any(
        "workspace cleanup replaced setup terminal" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert events == [
        "workspace_exit",
        (
            "failure_terminal",
            cleanup_error,
            "incomplete_accounting",
        ),
    ]


def test_workspace_mutation_lock_rejects_config_include_fifo(
    tmp_path: Path,
) -> None:
    repo, _ = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        workspace_mutation_lock,
    )

    include_fifo = tmp_path / "include.fifo"
    _git(repo, "config", "include.path", str(include_fifo))
    os.mkfifo(include_fifo)

    with pytest.raises(
        MutationJournalError,
        match="config include/includeIf directives are unsupported",
    ):
        with workspace_mutation_lock(repo):
            pytest.fail("unsafe config include unexpectedly acquired lock")

    assert stat.S_ISFIFO(include_fifo.lstat().st_mode)


@pytest.mark.parametrize("entry_kind", ("symlink", "hardlink", "fifo"))
def test_workspace_mutation_lock_rejects_unsafe_lock_entry(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    repo, _ = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        workspace_mutation_lock,
    )

    lock_path = repo / ".git" / "crucible-workspace.lock"
    victim = tmp_path / "lock-victim"
    if entry_kind == "symlink":
        victim.write_text("victim\n")
        lock_path.symlink_to(victim)
    elif entry_kind == "hardlink":
        victim.write_text("victim\n")
        os.link(victim, lock_path)
    else:
        os.mkfifo(lock_path)

    with pytest.raises(MutationJournalError):
        with workspace_mutation_lock(repo):
            pytest.fail("unsafe lock entry unexpectedly granted authority")

    if entry_kind != "fifo":
        assert victim.read_text() == "victim\n"


def test_workspace_mutation_lock_rejects_lock_replacement_after_flock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fcntl

    repo, _ = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        workspace_mutation_lock,
    )

    with workspace_mutation_lock(repo):
        pass
    lock_path = repo / ".git" / "crucible-workspace.lock"
    real_flock = fcntl.flock
    replaced = False

    def replace_lock(descriptor, operation):
        nonlocal replaced
        if operation == fcntl.LOCK_EX and not replaced:
            replaced = True
            lock_path.unlink()
            lock_path.write_text("replacement\n")
        return real_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", replace_lock)
    with pytest.raises(
        MutationJournalError,
        match="workspace authority lock changed identity",
    ):
        with workspace_mutation_lock(repo):
            pytest.fail("replaced lock unexpectedly granted authority")

    assert replaced
    assert lock_path.read_text() == "replacement\n"


def test_workspace_mutation_lock_rejects_common_dir_replacement_after_flock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fcntl

    repo, _ = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        workspace_mutation_lock,
    )

    real_flock = fcntl.flock
    displaced = tmp_path / "displaced-git-dir"
    replaced = False

    def replace_common_dir(descriptor, operation):
        nonlocal replaced
        if operation == fcntl.LOCK_EX and not replaced:
            replaced = True
            (repo / ".git").rename(displaced)
            (repo / ".git").mkdir()
        return real_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", replace_common_dir)
    with pytest.raises(
        MutationJournalError,
        match="workspace authority common dir changed identity",
    ):
        with workspace_mutation_lock(repo):
            pytest.fail("replaced common dir unexpectedly granted authority")

    assert replaced
    assert displaced.is_dir()


def _assert_retained_root_swap_cleanup(
    error: BaseException,
    *,
    primary_message_prefix: str | None,
    roots: tuple[Path, ...],
) -> None:
    """Prove a checkout swap retains its exact, non-settled cleanup owner."""

    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import MutationJournalError

    assert type(error) is _mutation_journal._RetainedAcquisitionCleanupError
    assert str(error) == "retained acquisition cleanup failed"
    if primary_message_prefix is None:
        assert error.primary_error is None
        assert error.__cause__ is None
    else:
        assert type(error.primary_error) is MutationJournalError
        assert str(error.primary_error).startswith(primary_message_prefix)
        assert error.__cause__ is error.primary_error

    nested = error.cleanup_error
    assert type(nested) is _mutation_journal._RetainedAcquisitionCleanupError
    assert str(nested) == "retained acquisition cleanup failed"
    assert type(nested.primary_error) is MutationJournalError
    assert str(nested.primary_error) == "external workspace root changed identity"
    assert type(nested.cleanup_error) is MutationJournalError
    assert str(nested.cleanup_error) == "external workspace root changed identity"
    assert nested.primary_error is not nested.cleanup_error
    assert nested.__cause__ is nested.primary_error
    assert error.owner is not nested.owner
    assert error.owner.cleanup_complete is False
    assert nested.owner.cleanup_complete is False

    for _attempt in range(2):
        with pytest.raises(MutationJournalError) as retry:
            error.owner.retry_cleanup()
        assert retry.value is nested.cleanup_error
        assert error.owner.cleanup_complete is False
        assert nested.owner.cleanup_complete is False

    for root in roots:
        assert _mutation_journal._active_workspace_mutation_authority(root) is None
        assert (
            _mutation_journal._active_external_workspace_coordination(root) is None
        )


def test_workspace_mutation_lock_rejects_checkout_root_replacement_after_flock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fcntl

    repo, _ = _setup(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _init_repo(other)
    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        workspace_mutation_lock,
    )

    real_flock = fcntl.flock
    displaced = tmp_path / "displaced-repo"
    swapped = False

    def replace_checkout(descriptor, operation):
        nonlocal swapped
        if operation == fcntl.LOCK_EX and not swapped:
            swapped = True
            repo.rename(displaced)
            other.rename(repo)
        return real_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", replace_checkout)
    with pytest.raises(
        _mutation_journal._RetainedAcquisitionCleanupError,
    ) as caught:
        with workspace_mutation_lock(repo):
            pytest.fail("replaced checkout unexpectedly granted authority")

    _assert_retained_root_swap_cleanup(
        caught.value,
        primary_message_prefix="workspace authority root changed identity",
        roots=(repo, displaced),
    )

    assert swapped
    assert displaced.is_dir()


def test_workspace_mutation_lock_rejects_reentry_after_checkout_root_swap(
    tmp_path: Path,
) -> None:
    repo, _ = _setup(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _init_repo(other)
    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        workspace_mutation_lock,
    )

    displaced = tmp_path / "displaced-repo"
    root_swap_message = (
        "(?:workspace authority|external workspace) root changed identity"
    )
    with pytest.raises(
        _mutation_journal._RetainedAcquisitionCleanupError,
    ) as caught:
        with workspace_mutation_lock(repo):
            repo.rename(displaced)
            other.rename(repo)
            with pytest.raises(MutationJournalError, match=root_swap_message):
                with workspace_mutation_lock(repo):
                    pytest.fail("swapped checkout acquired a second lock")

    _assert_retained_root_swap_cleanup(
        caught.value,
        primary_message_prefix=None,
        roots=(repo, displaced),
    )

    assert displaced.is_dir()
    assert repo.is_dir()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_workspace_mutation_lock_closes_authority_fds_in_fork_child(
    tmp_path: Path,
) -> None:
    repo, _ = _setup(tmp_path)
    from crucible.workflows import _mutation_journal

    read_fd, write_fd = os.pipe()
    try:
        with _mutation_journal.workspace_mutation_lock(repo) as authority:
            authority_fds = (
                authority.root_descriptor,
                authority.common_descriptor,
                authority.lock_descriptor,
            )
            child_pid = os.fork()
            if child_pid == 0:
                os.close(read_fd)
                closed = []
                for descriptor in authority_fds:
                    try:
                        os.fstat(descriptor)
                    except OSError as exc:
                        closed.append(exc.errno == errno.EBADF)
                    else:
                        closed.append(False)
                state_cleared = not getattr(
                    _mutation_journal._LOCK_STATE,
                    "authorities",
                    {},
                )
                os.write(
                    write_fd,
                    json.dumps({
                        "closed": closed,
                        "state_cleared": state_cleared,
                    }).encode(),
                )
                os.close(write_fd)
                os._exit(0)

            os.close(write_fd)
            write_fd = -1
            _, status = os.waitpid(child_pid, 0)
            child_report = json.loads(os.read(read_fd, 4096))
            assert os.WIFEXITED(status)
            assert os.WEXITSTATUS(status) == 0
            assert child_report == {
                "closed": [True, True, True],
                "state_cleared": True,
            }
            for descriptor in authority_fds:
                os.fstat(descriptor)
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_journal_rejects_symbolic_head_switch_at_same_commit(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    baseline_ref = _git(repo, "symbolic-ref", "HEAD").decode().strip()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "head_ref" / "mutations",
        slice_id="head_ref",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _git(repo, "switch", "-qc", "other")
    with pytest.raises(MutationJournalError, match="changed Git HEAD"):
        journal.assert_head_unchanged()
    assert journal.restore()
    assert _git(repo, "symbolic-ref", "HEAD").decode().strip() == baseline_ref


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_journal_rejects_index_flags_with_unchanged_tree(
    tmp_path: Path,
    flag: str,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "index_flag" / "mutations",
        slice_id="index_flag",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _git(repo, "update-index", flag, "a.py")
    with pytest.raises(MutationJournalError, match="staged Git index state"):
        journal.assert_index_unchanged()
    assert journal.restore()


def test_journal_rejects_hardlinked_index(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "index_link" / "mutations",
        slice_id="index_link",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    index = Path(_git(repo, "rev-parse", "--git-path", "index").decode().strip())
    if not index.is_absolute():
        index = repo / index
    original = index.with_name("index.original")
    index.rename(original)
    index.hardlink_to(original)
    with pytest.raises(
        MutationJournalError,
        match="hard-linked|removed or replaced Git index",
    ):
        journal.assert_index_unchanged()
    assert journal.restore()
    assert index.stat().st_nlink == 1


def test_journal_restores_non_head_refs_and_new_objects(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "git_metadata" / "mutations",
        slice_id="git_metadata",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _git(repo, "branch", "rogue")
    created_oid = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=b"rogue object\n",
        check=True,
        capture_output=True,
    ).stdout.decode().strip()
    with pytest.raises(MutationJournalError):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert subprocess.run(
        ["git", "show-ref", "--verify", "refs/heads/rogue"],
        cwd=repo,
        capture_output=True,
        check=False,
    ).returncode != 0
    objects = Path(_git(
        repo, "rev-parse", "--path-format=absolute", "--git-path", "objects",
    ).decode().strip())
    assert not (objects / created_oid[:2] / created_oid[2:]).exists()


def test_journal_retains_evidence_when_existing_object_disappears(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "object_loss" / "mutations",
        slice_id="object_loss",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    objects = Path(_git(
        repo, "rev-parse", "--path-format=absolute", "--git-path", "objects",
    ).decode().strip())
    object_path = objects / head[:2] / head[2:]
    assert object_path.is_file()
    object_path.unlink()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()


def test_journal_rejects_object_root_symlink_and_retains_evidence(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "object_alias" / "mutations",
        slice_id="object_alias",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    objects = Path(_git(
        repo, "rev-parse", "--path-format=absolute", "--git-path", "objects",
    ).decode().strip())
    moved = objects.with_name("objects-moved")
    objects.rename(moved)
    objects.symlink_to(moved, target_is_directory=True)

    assert not journal.restore()
    assert journal.journal_dir.is_dir()
    assert objects.is_symlink()


def test_journal_recovers_deleted_current_branch_ref(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    baseline_ref = _git(repo, "symbolic-ref", "HEAD").decode().strip()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "deleted_ref" / "mutations",
        slice_id="deleted_ref",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _git(repo, "update-ref", "-d", baseline_ref)
    assert journal.restore()
    assert _git(repo, "symbolic-ref", "HEAD").decode().strip() == baseline_ref
    assert _git(repo, "rev-parse", "HEAD").decode().strip()


def test_graceful_exception_after_atomic_attach_preserves_workspace(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "post_attach" / "mutations",
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# atomically attached\n")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo), path_scope=("a.py",),
    )
    impl = Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(diff, kind="diffs", name_hint="x.diff"),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint="x.json",
        ),
        files_modified=files,
        scope_escapes=(),
        classification="success",
        execution=exec_ctx_now("test"),
    )
    journal.prepare(impl, keep=True)
    with pytest.raises(KeyboardInterrupt):
        with ExitStack() as stack:
            stack.callback(journal.settle_or_restore, corpus)
            corpus.commit(implementations=(impl,))
            raise KeyboardInterrupt
    assert (repo / "a.py").read_text() == "# atomically attached\n"
    assert not journal.journal_dir.exists()


def test_unconfigured_effective_filter_uses_physical_snapshot_rollback(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    (repo / ".gitattributes").write_text(
        "a.py filter=untrusted-smudge\n",
        encoding="utf-8",
    )
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    baseline = (repo / "a.py").read_bytes()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "attrs" / "mutations",
        slice_id="attrs",
        scope=("b.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert "a.py" in journal._manifest["entries"]
    assert "a.py" not in journal._manifest["raw_object_restore"]
    assert journal._manifest["external_filter_paths"] == ["a.py"]

    (repo / "a.py").write_text("# escaped mutation\n", encoding="utf-8")
    assert journal.detected_scope_escapes() == ("a.py",)
    assert journal.restore()
    assert (repo / "a.py").read_bytes() == baseline


def test_external_filter_status_is_excluded_but_physical_bytes_are_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, corpus = _setup(tmp_path)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    (repo / ".gitattributes").write_text(
        "a.py filter=uppercase\n",
        encoding="utf-8",
    )
    _git(repo, "config", "--global", "filter.uppercase.clean", "tr a-z A-Z")
    _git(repo, "config", "--global", "filter.uppercase.smudge", "cat")
    _git(repo, "config", "--global", "filter.uppercase.required", "true")
    (repo / "a.py").write_text("# BASELINE A.PY\n", encoding="utf-8")
    _git(repo, "add", ".gitattributes", "a.py")
    _git(repo, "commit", "-qm", "filtered baseline")
    (repo / "a.py").write_text("# baseline a.py\n", encoding="utf-8")
    assert _git(repo, "status", "--porcelain", "--", "a.py") == b""

    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "filtered" / "mutations",
        slice_id="filtered",
        scope=("b.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert journal._manifest["external_filter_paths"] == ["a.py"]
    baseline = (repo / "a.py").read_bytes()
    (repo / "a.py").write_text("# escaped mutation\n", encoding="utf-8")
    (repo / "b.py").write_text("# mutation\n", encoding="utf-8")

    assert journal.restore()
    assert not journal.journal_dir.exists()
    assert (repo / "a.py").read_bytes() == baseline
    assert _git(repo, "status", "--porcelain", "--", "a.py") == b""


def test_unbound_external_filter_path_fails_closed(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "unbound_filter" / "mutations",
        slice_id="unbound_filter",
        scope=("b.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    journal._manifest["external_filter_paths"] = ["unbound.bin"]
    journal._write()
    (repo / "b.py").write_text("# mutation\n", encoding="utf-8")

    assert not journal.restore()
    assert journal.journal_dir.is_dir()


def test_effective_filter_attribute_is_rejected_inside_mutation_scope(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    (repo / ".gitattributes").write_text(
        "a.py filter=unconfigured-store\n",
        encoding="utf-8",
    )
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    with pytest.raises(MutationJournalError, match="mutation scope includes"):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "in_scope_attrs" / "mutations"
            ),
            slice_id="in_scope_attrs",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )


def test_wildcard_filter_rejects_absent_create_path(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    (repo / ".gitattributes").write_text(
        "*.bin filter=opaque-store\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-qm", "add wildcard filter policy")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    with pytest.raises(MutationJournalError, match="mutation scope includes"):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "absent_filter" / "mutations"
            ),
            slice_id="absent_filter",
            scope=("future.bin",),
            sequence=1,
            spec=corpus.latest_spec(),
        )


def test_git_lfs_attribute_snapshots_physical_checkout_image(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    payload = (b"retained binary witness\x00" * 4096) + b"tail\n"
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        + b"oid sha256:"
        + hashlib.sha256(payload).hexdigest().encode("ascii")
        + b"\nsize "
        + str(len(payload)).encode("ascii")
        + b"\n"
    )
    clean_filter = tmp_path / "lfs-clean-filter"
    clean_filter.write_bytes(
        b"#!/bin/sh\ncat >/dev/null\ncat <<'EOF'\n" + pointer + b"EOF\n"
    )
    clean_filter.chmod(0o755)
    status_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    status_env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
    })

    (repo / ".gitattributes").write_text(
        "large.bin filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    (repo / "large.bin").write_bytes(payload)
    subprocess.run(
        [
            "git",
            "-c",
            f"filter.lfs.clean={clean_filter}",
            "-c",
            "filter.lfs.required=false",
            "add",
            ".gitattributes",
            "large.bin",
        ],
        cwd=repo,
        env=status_env,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "add smudged LFS fixture"],
        cwd=repo,
        env=status_env,
        check=True,
    )
    assert _git(repo, "show", "HEAD:large.bin") == pointer
    assert (repo / "large.bin").read_bytes() == payload

    def _lfs_status() -> bytes:
        return subprocess.run(
            [
                "git",
                "-c",
                f"filter.lfs.clean={clean_filter}",
                "-c",
                "filter.lfs.required=false",
                "status",
                "--porcelain",
            ],
            cwd=repo,
            env=status_env,
            check=True,
            capture_output=True,
        ).stdout

    assert _lfs_status() == b""
    # The filtered add caches the physical payload's stat while indexing the
    # pointer bytes.  Invalidate that cache deterministically so the journal's
    # filter-free Git view hashes the worktree and records ``large.bin`` in the
    # transaction baseline.  Without this, Git may trust the racy-clean cache
    # on one runner and discover the physical-vs-pointer difference only after
    # restore.
    physical = (repo / "large.bin").stat()
    os.utime(
        repo / "large.bin",
        ns=(physical.st_atime_ns, physical.st_mtime_ns - 2_000_000_000),
    )
    assert subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        env=status_env,
        check=True,
        capture_output=True,
    ).stdout == b" M large.bin\n"

    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "lfs_attr_snapshot" / "mutations"
        ),
        slice_id="lfs_attr_snapshot",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert "large.bin" not in journal._manifest["raw_object_restore"]
    assert journal._manifest["entries"]["large.bin"]["kind"] == "file"

    # Mutate the physical checkout directly.  The assertions above already
    # prove that HEAD stores the pointer while the worktree stores the smudged
    # payload.  checkout-index can trust the clean index's payload stat cache
    # and skip materialization, so using it here would test host Git timing
    # rather than the journal's byte-exact snapshot contract.
    (repo / "large.bin").write_bytes(pointer)
    assert (repo / "large.bin").read_bytes() == pointer
    assert journal.restore()
    assert (repo / "large.bin").read_bytes() == payload

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "lfs_attr_recovery" / "mutations"
        ),
        slice_id="lfs_attr_recovery",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )

    before = (repo / "large.bin").stat()
    (repo / "large.bin").write_bytes(b"X" * len(payload))
    os.utime(
        repo / "large.bin",
        ns=(before.st_atime_ns, before.st_mtime_ns),
    )
    assert journal.detected_scope_escapes() == ("large.bin",)
    _simulate_process_death(journal)
    from crucible.workflows._reaper import reap_orphaned_dirt

    report = reap_orphaned_dirt(corpus, repo)
    assert report.failures == ()
    assert report.mutation_journals_recovered == ("lfs_attr_recovery",)
    assert (repo / "large.bin").read_bytes() == payload
    assert _lfs_status() == b""


def test_recursive_scope_rejects_descendant_filter_attribute(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    (repo / "assets").mkdir()
    (repo / "assets" / "large.bin").write_bytes(b"opaque payload\n")
    (repo / ".gitattributes").write_text(
        "assets/large.bin filter=opaque-store\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".gitattributes", "assets/large.bin")
    _git(repo, "commit", "-qm", "add filtered asset")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    with pytest.raises(MutationJournalError, match="mutation scope includes"):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "recursive_filter" / "mutations"
            ),
            slice_id="recursive_filter",
            scope=("assets",),
            sequence=1,
            spec=corpus.latest_spec(),
        )


def test_recursive_scope_rejects_new_filtered_descendant_before_capture(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    (repo / "assets").mkdir()
    (repo / "assets" / "keep.txt").write_text("baseline\n")
    (repo / ".gitattributes").write_text(
        "*.bin filter=opaque-store\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".gitattributes", "assets/keep.txt")
    _git(repo, "commit", "-qm", "add filtered recursive policy")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "new_recursive_filter" / "mutations"
        ),
        slice_id="new_recursive_filter",
        scope=("assets",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "assets" / "new.bin").write_bytes(b"new payload\n")
    with pytest.raises(MutationJournalError, match="mutation scope contains"):
        journal.prepare_git_capture()
    assert journal.restore()
    assert not (repo / "assets" / "new.bin").exists()


def test_tier_reports_preinvoke_journal_rejection_with_complete_zero_cost(
    tmp_path: Path,
    monkeypatch,
):
    """Unsupported checkout transforms are setup failures, not lost spend."""

    repo, corpus = _setup(tmp_path)
    (repo / ".gitattributes").write_text(
        "a.py filter=untrusted-smudge\n",
        encoding="utf-8",
    )
    _git(repo, "config", "filter.untrusted-smudge.clean", "cat")
    backend = _MutationBackend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)

    assert _run(corpus, repo) == 2
    assert backend.calls == []

    run_dir = max((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())

    assert summary["status"] == "setup_error"
    assert summary["setup_stage"] == "mutation_journal_begin"
    assert summary["setup_slice_id"] == "s0"
    assert summary["error_kind"] == "MutationJournalError"
    assert "configured Git filter drivers" in summary["error_message"]
    assert summary["costs"]["total_usd"] == 0.0
    assert summary["costs"]["total_input_tokens"] == 0
    assert summary["costs"]["total_output_tokens"] == 0
    assert summary["costs"]["rollup_incomplete"] is False
    assert result["status"] == "setup_error"
    assert result["cost_usd"] == 0.0
    assert outcome["detail_status"] == "setup_error"
    assert outcome["terminal_class"] == "permanent_failure"
    assert outcome["cost_usd"] == 0.0


def test_tier_persists_returned_usage_before_journal_validation_rejects(
    tmp_path: Path,
    monkeypatch,
):
    """A post-call journal error retains exact child usage and stays primary."""

    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    repo, corpus = _setup(tmp_path)

    class _CostedMutationBackend(_MutationBackend):
        def invoke(self, invocation):
            super().invoke(invocation)
            return _success(
                input_tokens=123,
                output_tokens=45,
                cost_usd=1.234567,
            )

    backend = _CostedMutationBackend(repo, raise_s0=False)
    _patch_backend(monkeypatch, backend)
    original_validate = WorkspaceMutationJournal.validate_implementation

    def _reject_after_return(self, impl):
        if self.slice_id == "s0":
            raise MutationJournalError("post-call repository attestation failed")
        return original_validate(self, impl)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "validate_implementation",
        _reject_after_return,
    )

    with pytest.raises(
        MutationJournalError,
        match="post-call repository attestation failed",
    ):
        _run(corpus, repo)

    assert backend.calls == ["s0"]
    assert (repo / "a.py").read_text(encoding="utf-8") == "# baseline a.py\n"
    run_dir = max((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    outcome = json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))

    assert summary["status"] == "slice_failed"
    assert summary["failure_phase"] == "initial_mutation_validation"
    assert summary["failure_slice_id"] == "s0"
    assert summary["error_kind"] == "MutationJournalError"
    assert summary["error_message"] == "post-call repository attestation failed"
    assert summary["costs"]["total_usd"] == pytest.approx(1.234567)
    assert summary["costs"]["total_input_tokens"] == 123
    assert summary["costs"]["total_output_tokens"] == 45
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["rollup_incomplete"] is False
    assert result["status"] == "slice_failed"
    assert result["cost_usd"] == pytest.approx(1.234567)
    assert result["error_kind"] == "MutationJournalError"
    assert outcome["detail_status"] == "slice_failed"
    assert outcome["error_kind"] == "MutationJournalError"
    assert outcome["cost_usd"] == pytest.approx(1.234567)


def _setup_text_attribute_repo(
    tmp_path: Path,
    *,
    eol: str,
) -> tuple[Path, Corpus]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / ".gitattributes").write_text(
        f"*.js text eol={eol}\n",
        encoding="utf-8",
    )
    (repo / "Gruntfile.js").write_bytes(b"line one\nline two\n")
    (repo / "Gruntfile.js").chmod(0o755)
    _git(repo, "add", ".gitattributes", "Gruntfile.js")
    _git(repo, "commit", "-qm", "add text attributes")
    # Materialize the canonical checkout image selected by the attribute.
    (repo / "Gruntfile.js").unlink()
    _git(repo, "checkout", "HEAD", "--", "Gruntfile.js")

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    corpus.put_spec(_spec(head))
    return repo, corpus


def test_baseline_missing_target_rejects_replaced_real_parent(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    baseline_parent = repo / "pkg"
    baseline_parent.mkdir()
    (baseline_parent / "keep.py").write_text("# captured parent\n")
    _git(repo, "add", "pkg/keep.py")
    _git(repo, "commit", "-qm", "add captured parent")
    corpus = Corpus(tmp_path / "corpus")
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    corpus.put_spec(_spec(head))
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "replaced_missing_parent" / "mutations"
        ),
        slice_id="replaced_missing_parent",
        scope=("pkg/new.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert journal._manifest["entries"]["pkg/new.py"]["kind"] == "missing"
    parked = tmp_path / "captured-pkg"
    baseline_parent.rename(parked)
    replacement = repo / "pkg"
    replacement.mkdir(mode=parked.stat().st_mode & 0o777)
    payload = replacement / "new.py"
    payload.write_text("# provider replacement\n")

    with pytest.raises(
        MutationJournalError,
        match="restore parent differs from its captured identity",
    ):
        journal.restore_paths_from_snapshot(("pkg/new.py",))

    assert payload.read_text() == "# provider replacement\n"
    assert (parked / "keep.py").read_text() == "# captured parent\n"
    shutil.rmtree(replacement)
    parked.rename(repo / "pkg")
    assert journal.restore()


def test_restore_accepts_tracked_absence_beneath_already_absent_parent(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "a.py").write_text("# baseline\n")
    donor = repo / "donor"
    donor.mkdir()
    (donor / "child.py").write_text("# tracked donor\n")
    _git(repo, "add", "a.py", "donor/child.py")
    _git(repo, "commit", "-qm", "add donor")
    shutil.rmtree(donor)
    corpus = Corpus(tmp_path / "corpus")
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    corpus.put_spec(_spec(head))
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "already_absent_parent" / "mutations"
        ),
        slice_id="already_absent_parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert journal._manifest["entries"]["donor/child.py"]["kind"] == "missing"
    assert journal._manifest["tracked_path_metadata"]["donor/child.py"][
        "kind"
    ] == "missing"
    assert "donor" not in journal._manifest["directory_identities"]
    (repo / "a.py").write_text("# provider mutation\n")

    assert journal.restore()
    assert (repo / "a.py").read_text() == "# baseline\n"
    assert not donor.exists()


@pytest.mark.parametrize("create_leaf", [False, True])
def test_restore_handles_recreated_parent_of_captured_tracked_absence(
    tmp_path: Path,
    create_leaf: bool,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "a.py").write_text("# baseline\n")
    donor = repo / "donor"
    donor.mkdir()
    (donor / "child.py").write_text("# tracked donor\n")
    _git(repo, "add", "a.py", "donor/child.py")
    _git(repo, "commit", "-qm", "add donor")
    shutil.rmtree(donor)
    corpus = Corpus(tmp_path / "corpus")
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    corpus.put_spec(_spec(head))
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "recreated_absent_parent" / "mutations"
        ),
        slice_id="recreated_absent_parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    donor.mkdir()
    if create_leaf:
        (donor / "child.py").write_text("# replacement\n")

    if not create_leaf:
        journal.restore_paths_from_snapshot(("a.py",))
        assert not donor.exists()
        assert journal.restore()
        return

    with pytest.raises(
        MutationJournalError,
        match="workspace directory addition is not empty",
    ):
        journal.restore_paths_from_snapshot(("a.py",))

    assert (donor / "child.py").read_text() == "# replacement\n"
    shutil.rmtree(donor)
    assert journal.restore()


def test_builtin_text_attribute_uses_raw_object_rollback(tmp_path: Path):
    repo, corpus = _setup_text_attribute_repo(tmp_path, eol="lf")
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    baseline = (repo / "Gruntfile.js").read_bytes()
    baseline_mode = (repo / "Gruntfile.js").stat().st_mode & 0o777
    assert baseline == b"line one\nline two\n"
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "text_attr_object" / "mutations"
        ),
        slice_id="text_attr_object",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert "Gruntfile.js" in journal._manifest["raw_object_restore"]
    assert "Gruntfile.js" not in journal._manifest["entries"]

    (repo / "Gruntfile.js").write_bytes(b"model mutation\n")
    assert journal.detected_scope_escapes() == ("Gruntfile.js",)
    journal.restore_paths_from_snapshot(("Gruntfile.js",))
    assert (repo / "Gruntfile.js").read_bytes() == baseline
    assert (repo / "Gruntfile.js").stat().st_mode & 0o777 == baseline_mode

    (repo / "Gruntfile.js").write_bytes(b"second model mutation\n")
    assert journal.restore()
    assert (repo / "Gruntfile.js").read_bytes() == baseline
    assert (repo / "Gruntfile.js").stat().st_mode & 0o777 == baseline_mode
    assert _git(repo, "status", "--porcelain") == b""


def test_builtin_text_attribute_snapshots_physical_checkout_image(
    tmp_path: Path,
):
    repo, corpus = _setup_text_attribute_repo(tmp_path, eol="crlf")
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    baseline = (repo / "Gruntfile.js").read_bytes()
    assert baseline == b"line one\r\nline two\r\n"
    assert _git(repo, "status", "--porcelain") == b""
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "text_attr_snapshot" / "mutations"
        ),
        slice_id="text_attr_snapshot",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert "Gruntfile.js" not in journal._manifest["raw_object_restore"]
    assert journal._manifest["entries"]["Gruntfile.js"]["kind"] == "file"

    (repo / "Gruntfile.js").write_bytes(b"model mutation\r\n")
    assert journal.restore()
    assert (repo / "Gruntfile.js").read_bytes() == baseline
    assert _git(repo, "status", "--porcelain") == b""


def test_configured_clean_filter_is_rejected_before_invocation(tmp_path: Path):
    repo, corpus = _setup_text_attribute_repo(tmp_path, eol="lf")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    sentinel = tmp_path / "filter-executed"
    filter_script = tmp_path / "clean-filter.sh"
    filter_script.write_text(
        "#!/bin/sh\ntouch \"$1\"\ncat\n",
        encoding="utf-8",
    )
    filter_script.chmod(0o755)
    _git(
        repo,
        "config",
        "filter.hostile.clean",
        f"{filter_script} {sentinel}",
    )
    with pytest.raises(MutationJournalError, match="configured Git filter"):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "attribute_filter" / "mutations"
            ),
            slice_id="attribute_filter",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )
    assert not sentinel.exists()


def test_worktree_configured_filter_is_rejected_before_invocation(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    _git(repo, "config", "extensions.worktreeConfig", "true")
    _git(repo, "config", "--worktree", "FiLtEr.Hostile.ClEaN", "cat")
    with pytest.raises(MutationJournalError, match="configured Git filter"):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "worktree_filter" / "mutations"
            ),
            slice_id="worktree_filter",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )


def test_conditional_config_include_is_rejected_without_loading_it(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    included = tmp_path / "included.gitconfig"
    included.write_text("[filter \"hostile\"]\n\tclean = cat\n")
    _git(
        repo,
        "config",
        "--local",
        f"InClUdEiF.gitdir:{repo.as_posix()}/.path",
        str(included),
    )
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    with pytest.raises(MutationJournalError, match="config includes"):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "conditional_include" / "mutations"
            ),
            slice_id="conditional_include",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )


def test_initialized_submodule_filter_is_rejected_without_execution(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    sentinel = tmp_path / "nested-filter-ran"
    driver = tmp_path / "nested-filter"
    driver.write_text(
        "#!/bin/sh\n"
        f"touch {shlex.quote(str(sentinel))}\n"
        "cat\n",
        encoding="utf-8",
    )
    driver.chmod(0o755)
    _git(nested, "config", "filter.hostile.clean", str(driver))
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    with pytest.raises(MutationJournalError, match="configured Git filter"):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "nested_filter" / "mutations"
            ),
            slice_id="nested_filter",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )
    assert not sentinel.exists()


def test_home_attributes_are_ignored_by_journal_git_commands(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus = _setup(tmp_path)
    hostile_home = tmp_path / "hostile-home"
    attributes = hostile_home / ".config" / "git" / "attributes"
    attributes.parent.mkdir(parents=True)
    attributes.write_text("a.py working-tree-encoding=not-a-real-encoding\n")
    monkeypatch.setenv("HOME", str(hostile_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(hostile_home / ".config"))
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "home_attributes" / "mutations"
        ),
        slice_id="home_attributes",
        scope=("b.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert "a.py" not in journal._manifest["transforming_paths"]
    assert journal.restore()


def test_assay_pre_capture_hook_blocks_model_added_filter(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from assay.primitives.implement import implement
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    sentinel = tmp_path / "model-filter-executed"
    filter_script = tmp_path / "model-clean-filter.sh"
    filter_script.write_text(
        "#!/bin/sh\ntouch \"$1\"\ncat\n",
        encoding="utf-8",
    )
    filter_script.chmod(0o755)

    class _PolicyMutationBackend:
        name = "policy-mutation-stub"
        model = "stub-model"

        def invoke(self, invocation):
            _git(
                repo,
                "config",
                "filter.hostile.clean",
                f"{filter_script} {sentinel}",
            )
            (repo / ".gitattributes").write_text(
                "a.py filter=hostile\n",
                encoding="utf-8",
            )
            (repo / "a.py").write_text("# model mutation\n", encoding="utf-8")
            return _success()

    spec = corpus.latest_spec()
    assert spec is not None and spec.tier_dag is not None
    directive = spec.tier_dag.tiers[0].slices[0].directive
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "assay_capture_filter" / "mutations"
        ),
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
        spec=spec,
    )
    impl = implement(
        spec,
        "s0",
        directive,
        backend=_PolicyMutationBackend(),
        corpus=corpus,
        cwd=str(repo),
        before_capture=journal.prepare_git_capture,
    )

    assert impl.files_modified == ("a.py",)
    assert not sentinel.exists()
    with pytest.raises(MutationJournalError, match="config/info attributes"):
        journal.validate_implementation(impl)
    assert not sentinel.exists()
    assert journal.restore()
    assert not (repo / ".gitattributes").exists()
    assert not sentinel.exists()


def test_initialized_gitlink_admin_mutations_restore_child_first(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _registry_directory,
    )

    baseline_head = _git(nested, "rev-parse", "HEAD")
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "nested" / "mutations",
        slice_id="nested",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    children = journal._nested_journals()
    assert len(children) == 1
    _git(nested, "config", "nested.guard-probe", "changed")
    _git(nested, "commit", "--allow-empty", "-qm", "rogue nested commit")
    with pytest.raises(MutationJournalError):
        journal.assert_repository_metadata_unchanged()

    assert journal.restore()
    assert _git(nested, "rev-parse", "HEAD") == baseline_head
    assert subprocess.run(
        ["git", "config", "--get", "nested.guard-probe"],
        cwd=nested,
        capture_output=True,
        check=False,
    ).returncode == 1
    assert list(_registry_directory(repo).iterdir()) == []

    retry = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "nested_retry" / "mutations",
        slice_id="nested_retry",
        scope=("a.py",),
        sequence=2,
        spec=corpus.latest_spec(),
    )
    assert retry.restore()


def _gitlink_index(nested: Path) -> Path:
    return Path(
        _git(nested, "rev-parse", "--absolute-git-dir").decode().strip()
    ) / "index"


def _replace_with_semantically_equal_index(nested: Path, index: Path) -> None:
    replacement = index.with_name("r19-benign-index-rewrite")
    git_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    git_env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_INDEX_FILE": str(replacement),
    })
    previous_umask = os.umask(0o002)
    try:
        subprocess.run(
            ["git", "read-tree", "HEAD"],
            cwd=nested,
            env=git_env,
            check=True,
            capture_output=True,
        )
    finally:
        os.umask(previous_umask)
    os.replace(replacement, index)


def test_initialized_gitlink_private_wider_cache_churn_is_normalized(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from assay.primitives.implement import capture_workspace_delta
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _index_semantic_views,
        _nested_clean_status,
        recovery_workspace_mutation_lock,
    )
    from crucible.workflows._run import exec_ctx_now

    repo.chmod(0o700)
    index = _gitlink_index(nested)
    index.chmod(0o600)
    with recovery_workspace_mutation_lock(corpus, repo):
        journal = WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "nested_index_wider" / "mutations"
            ),
            slice_id="nested_index_wider",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )
        child = next(
            candidate
            for candidate in journal._nested_journals()
            if "::gitlink::" in candidate.slice_id
        )
        baseline = index.read_bytes()
        baseline_mode = stat.S_IMODE(index.stat().st_mode)
        baseline_views = _index_semantic_views(nested, index)
        baseline_status = _nested_clean_status(nested)
        (repo / "a.py").write_text(
            "# retained candidate bytes\n",
            encoding="utf-8",
        )
        diff, files, base = capture_workspace_delta(
            cwd=str(repo),
            path_scope=("a.py",),
        )
        candidate_diff = subprocess.run(
            ["git", "diff", "--binary", "--", "a.py"],
            cwd=repo,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            check=True,
            capture_output=True,
        ).stdout
        implementation = Implementation(
            slice_id="nested_index_wider",
            base_commit=base,
            result_commit=None,
            diff_ref=corpus.put_artifact(
                diff,
                kind="diffs",
                name_hint="nested-index-wider.diff",
            ),
            transcript_ref=corpus.put_artifact(
                b"[]",
                kind="transcripts",
                name_hint="nested-index-wider.json",
            ),
            files_modified=files,
            scope_escapes=(),
            classification="success",
            execution=exec_ctx_now("test"),
        )

        _replace_with_semantically_equal_index(nested, index)

        assert baseline_mode == 0o600
        assert stat.S_IMODE(index.stat().st_mode) == 0o664
        assert index.read_bytes() != baseline
        assert _index_semantic_views(nested, index) == baseline_views
        assert _nested_clean_status(nested) == baseline_status

        validated = journal.validate_implementation(implementation)

        assert validated == implementation
        assert index.read_bytes() == baseline
        assert stat.S_IMODE(index.stat().st_mode) == baseline_mode
        assert subprocess.run(
            ["git", "diff", "--binary", "--", "a.py"],
            cwd=repo,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            check=True,
            capture_output=True,
        ).stdout == candidate_diff
        assert child.assert_index_unchanged() is None
        assert journal.restore()


@pytest.mark.parametrize(
    "mutation",
    (
        "mode-only",
        "non-private",
        "no-authority",
        "executable",
        "special",
        "mixed",
        "legacy-identity",
        "identity-device",
        "identity-uid",
        "identity-gid",
    ),
)
def test_initialized_gitlink_wider_index_repair_rejects_unsafe_modes(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    repo.chmod(0o755 if mutation == "non-private" else 0o700)
    index = _gitlink_index(nested)
    index.chmod(0o640 if mutation == "mixed" else 0o600)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / f"nested_index_{mutation}" / "mutations"
        ),
        slice_id=f"nested_index_{mutation}",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = next(
        candidate
        for candidate in journal._nested_journals()
        if "::gitlink::" in candidate.slice_id
    )
    if mutation == "mode-only":
        index.chmod(0o664)
    else:
        _replace_with_semantically_equal_index(nested, index)
        if mutation == "executable":
            index.chmod(0o764)
        elif mutation == "special":
            index.chmod(0o2664)
        elif mutation == "mixed":
            index.chmod(0o602)
        elif mutation == "legacy-identity":
            for field in ("index_device", "index_uid", "index_gid"):
                child._manifest.pop(field)
            child._write()
        elif mutation.startswith("identity-"):
            field = f"index_{mutation.removeprefix('identity-')}"
            child._manifest[field] = int(child._manifest[field]) + 1
            child._write()

    authority = (
        nullcontext()
        if mutation == "no-authority"
        else workspace_mutation_lock(repo)
    )
    with authority:
        with pytest.raises(
            MutationJournalError,
            match="changed Git (?:admin file|index mode)",
        ):
            journal.assert_repository_metadata_unchanged()

    if mutation.startswith("identity-"):
        field = f"index_{mutation.removeprefix('identity-')}"
        child._manifest[field] = int(child._manifest[field]) - 1
        child._write()
    assert journal.restore()


@pytest.mark.parametrize(
    "mutation",
    ("config", "head", "refs", "hooks", "objects", "index-lock"),
)
def test_initialized_gitlink_cache_repair_rejects_admin_comutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    repo.chmod(0o700)
    index = _gitlink_index(nested)
    index.chmod(0o600)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / f"nested_admin_{mutation}" / "mutations"
        ),
        slice_id=f"nested_admin_{mutation}",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _replace_with_semantically_equal_index(nested, index)
    git_dir = index.parent
    if mutation == "config":
        config = git_dir / "config"
        config.write_bytes(config.read_bytes() + b"\n[r19]\n\tprobe = true\n")
    elif mutation == "head":
        head = git_dir / "HEAD"
        head.write_bytes(head.read_bytes() + b" ")
    elif mutation == "refs":
        rogue_ref = git_dir / "refs" / "heads" / "r19-probe"
        rogue_ref.write_text("0" * 40 + "\n", encoding="ascii")
    elif mutation == "hooks":
        hook = git_dir / "hooks" / "pre-commit.sample"
        hook.write_bytes(hook.read_bytes() + b"\n# r19 probe\n")
    elif mutation == "objects":
        (git_dir / "objects" / "info" / "r19-probe").write_bytes(b"probe\n")
    else:
        (git_dir / "index.lock").write_bytes(b"r19 held lock\n")

    with pytest.raises(MutationJournalError):
        journal.assert_repository_metadata_unchanged()

    assert journal.restore()


def test_initialized_gitlink_cache_repair_rejects_index_xattrs(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "setxattr"):
        pytest.skip("xattrs unavailable")
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    repo.chmod(0o700)
    index = _gitlink_index(nested)
    index.chmod(0o600)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_index_xattr" / "mutations"
        ),
        slice_id="nested_index_xattr",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _replace_with_semantically_equal_index(nested, index)
    try:
        os.setxattr(index, "user.crucible-r19", b"probe")
    except OSError as exc:
        if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM}:
            pytest.skip("test filesystem does not permit user xattrs")
        raise

    with pytest.raises(MutationJournalError, match="Git admin xattrs"):
        journal.assert_repository_metadata_unchanged()

    assert journal.restore()


def test_initialized_gitlink_cache_repair_rejects_private_root_xattrs(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "setxattr"):
        pytest.skip("xattrs unavailable")
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    repo.chmod(0o700)
    index = _gitlink_index(nested)
    index.chmod(0o600)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_private_root_xattr" / "mutations"
        ),
        slice_id="nested_private_root_xattr",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _replace_with_semantically_equal_index(nested, index)
    try:
        os.setxattr(repo, "user.crucible-r19", b"probe")
    except OSError as exc:
        if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM}:
            pytest.skip("test filesystem does not permit user xattrs")
        raise

    with workspace_mutation_lock(repo):
        with pytest.raises(
            MutationJournalError,
            match="changed Git (?:admin file|index mode)",
        ):
            journal.assert_repository_metadata_unchanged()

    assert journal.restore()


def test_initialized_gitlink_cache_repair_requires_process_owned_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    repo.chmod(0o700)
    index = _gitlink_index(nested)
    index.chmod(0o600)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_foreign_private_root" / "mutations"
        ),
        slice_id="nested_foreign_private_root",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = next(
        candidate
        for candidate in journal._nested_journals()
        if "::gitlink::" in candidate.slice_id
    )
    _replace_with_semantically_equal_index(nested, index)
    metadata = index.stat()
    process_uid = os.geteuid()

    with workspace_mutation_lock(repo), monkeypatch.context() as isolated:
        isolated.setattr(os, "geteuid", lambda: process_uid + 1)
        assert not child._nested_index_mode_repair_is_privately_confined(
            metadata
        )

    assert journal.restore()


def test_initialized_gitlink_index_aliases_fail_without_touching_target(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    repo.chmod(0o700)
    index = _gitlink_index(nested)
    index.chmod(0o600)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_index_aliases" / "mutations"
        ),
        slice_id="nested_index_aliases",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    baseline = index.read_bytes()
    external = tmp_path / "external-index-target"
    external.write_bytes(baseline)

    for replacement in ("missing", "symlink", "hardlink", "fifo", "directory"):
        if index.exists() or index.is_symlink():
            index.unlink()
        if replacement == "symlink":
            index.symlink_to(external)
        elif replacement == "hardlink":
            os.link(external, index)
        elif replacement == "fifo":
            os.mkfifo(index)
        elif replacement == "directory":
            index.mkdir()
        with pytest.raises((MutationJournalError, OSError)):
            journal.assert_repository_metadata_unchanged()
        assert external.read_bytes() == baseline
        if index.is_dir() and not index.is_symlink():
            index.rmdir()
        elif index.exists() or index.is_symlink():
            index.unlink()
        index.write_bytes(baseline)
        index.chmod(0o600)

    journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert external.read_bytes() == baseline


@pytest.mark.parametrize(
    "race_point",
    ("semantic-proof", "restore", "remove", "remove-in-place"),
)
def test_initialized_gitlink_cache_repair_rejects_index_replacement_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race_point: str,
) -> None:
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    repo.chmod(0o700)
    index = _gitlink_index(nested)
    index.chmod(0o600)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / f"nested_index_race_{race_point}"
            / "mutations"
        ),
        slice_id=f"nested_index_race_{race_point}",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = next(
        candidate
        for candidate in journal._nested_journals()
        if "::gitlink::" in candidate.slice_id
    )
    _replace_with_semantically_equal_index(nested, index)
    raced = False

    def replace_current_index() -> None:
        nonlocal raced
        replacement = index.with_name("r19-racing-index")
        shutil.copy2(index, replacement)
        os.replace(replacement, index)
        raced = True

    def rewrite_current_index_in_place() -> None:
        nonlocal raced
        before = index.stat()
        payload = bytearray(index.read_bytes())
        payload[-1] ^= 0x01
        with index.open("r+b", buffering=0) as stream:
            stream.write(payload)
            os.fsync(stream.fileno())
        after = index.stat()
        assert (after.st_dev, after.st_ino, after.st_size) == (
            before.st_dev,
            before.st_ino,
            before.st_size,
        )
        raced = True

    with monkeypatch.context() as isolated:
        if race_point == "semantic-proof":
            original_views = _mutation_journal._index_semantic_views

            def racing_views(cwd: Path, current_index: Path):
                views = original_views(cwd, current_index)
                if not raced and cwd.resolve() == nested.resolve():
                    replace_current_index()
                return views

            isolated.setattr(
                _mutation_journal,
                "_index_semantic_views",
                racing_views,
            )
            match = "during semantic proof"
        elif race_point == "restore":
            original_restore = WorkspaceMutationJournal._restore_bound_git_admin_file

            def racing_restore(owner, *args, **kwargs):
                if (
                    not raced
                    and owner.transaction_id == child.transaction_id
                    and kwargs.get("label") == "Git index normalization"
                ):
                    replace_current_index()
                return original_restore(owner, *args, **kwargs)

            isolated.setattr(
                WorkspaceMutationJournal,
                "_restore_bound_git_admin_file",
                racing_restore,
            )
            match = "before normalization"
        elif race_point == "remove":
            original_remove = _mutation_journal._remove_bound_workspace_entry

            def racing_remove(*args, **kwargs):
                if not raced and kwargs.get("expected_top_metadata") is not None:
                    replace_current_index()
                return original_remove(*args, **kwargs)

            isolated.setattr(
                _mutation_journal,
                "_remove_bound_workspace_entry",
                racing_remove,
            )
            match = "changed before removal"
        else:
            original_remove = _mutation_journal._remove_bound_workspace_entry

            def racing_remove_in_place(*args, **kwargs):
                if not raced and kwargs.get("expected_top_metadata") is not None:
                    rewrite_current_index_in_place()
                return original_remove(*args, **kwargs)

            isolated.setattr(
                _mutation_journal,
                "_remove_bound_workspace_entry",
                racing_remove_in_place,
            )
            match = "changed before removal"
        with workspace_mutation_lock(repo):
            with pytest.raises(MutationJournalError, match=match):
                journal.assert_repository_metadata_unchanged()

    assert raced
    assert journal.restore()


def test_initialized_gitlink_cache_repair_rejects_status_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    repo.chmod(0o700)
    index = _gitlink_index(nested)
    index.chmod(0o600)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_index_status_divergence"
            / "mutations"
        ),
        slice_id="nested_index_status_divergence",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    baseline = index.read_bytes()
    _replace_with_semantically_equal_index(nested, index)
    original_status = _mutation_journal._nested_clean_status

    def divergent_status(cwd: Path, *, excluded_roots=()):
        status = original_status(cwd, excluded_roots=excluded_roots)
        if cwd.resolve() == nested.resolve() and index.read_bytes() == baseline:
            return status + b"forced-post-normalization-divergence\0"
        return status

    with monkeypatch.context() as isolated:
        isolated.setattr(
            _mutation_journal,
            "_nested_clean_status",
            divergent_status,
        )
        with workspace_mutation_lock(repo):
            with pytest.raises(
                MutationJournalError,
                match="changed Git status after index normalization",
            ):
                journal.assert_repository_metadata_unchanged()

    assert index.read_bytes() == baseline
    assert journal.restore()


def test_initialized_gitlink_benign_index_cache_churn_is_normalized(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_index_cache" / "mutations"
        ),
        slice_id="nested_index_cache",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = next(
        candidate
        for candidate in journal._nested_journals()
        if "::gitlink::" in candidate.slice_id
    )
    index = Path(str(child._manifest["index_path"]))
    baseline = index.read_bytes()
    rebuilt = tmp_path / "rebuilt-nested-index"
    git_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    git_env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_INDEX_FILE": str(rebuilt),
    })
    subprocess.run(
        ["git", "read-tree", "HEAD"],
        cwd=nested,
        env=git_env,
        check=True,
    )
    rebuilt_payload = rebuilt.read_bytes()
    assert rebuilt_payload != baseline
    index.write_bytes(rebuilt_payload)
    baseline_mode = stat.S_IMODE(index.stat().st_mode)
    narrowed_mode = baseline_mode & ~0o077
    assert narrowed_mode != baseline_mode
    index.chmod(narrowed_mode)
    assert _git(nested, "ls-files", "--stage", "-z") == (
        subprocess.run(
            ["git", "ls-files", "--stage", "-z"],
            cwd=nested,
            env=git_env,
            check=True,
            capture_output=True,
        ).stdout
    )

    journal.assert_repository_metadata_unchanged()

    assert index.read_bytes() == baseline
    assert stat.S_IMODE(index.stat().st_mode) == baseline_mode
    assert journal.restore()


def test_top_level_benign_index_cache_churn_is_normalized(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "root_index_cache" / "mutations"
        ),
        slice_id="root_index_cache",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    index = Path(str(journal._manifest["index_path"]))
    baseline = index.read_bytes()
    rebuilt = tmp_path / "rebuilt-root-index"
    git_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    git_env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_INDEX_FILE": str(rebuilt),
    })
    subprocess.run(
        ["git", "read-tree", "HEAD"],
        cwd=repo,
        env=git_env,
        check=True,
    )
    rebuilt_payload = rebuilt.read_bytes()
    assert rebuilt_payload != baseline
    index.write_bytes(rebuilt_payload)
    baseline_mode = stat.S_IMODE(index.stat().st_mode)
    narrowed_mode = baseline_mode & ~0o077
    assert narrowed_mode != baseline_mode
    index.chmod(narrowed_mode)

    journal.assert_repository_metadata_unchanged()

    assert index.read_bytes() == baseline
    assert stat.S_IMODE(index.stat().st_mode) == baseline_mode
    assert journal.restore()


def test_initialized_gitlink_index_mode_only_mutation_still_fails_closed(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup(tmp_path)
    _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_index_mode" / "mutations"
        ),
        slice_id="nested_index_mode",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = next(
        candidate
        for candidate in journal._nested_journals()
        if "::gitlink::" in candidate.slice_id
    )
    index = Path(str(child._manifest["index_path"]))
    baseline_mode = stat.S_IMODE(index.stat().st_mode)
    narrowed_mode = baseline_mode & ~0o077
    assert narrowed_mode != baseline_mode
    index.chmod(narrowed_mode)

    with pytest.raises(MutationJournalError, match="changed Git index mode"):
        journal.assert_repository_metadata_unchanged()

    assert journal.restore()


@pytest.mark.parametrize(
    "mutation",
    [
        "staged-mode",
        "--skip-worktree",
        "--assume-unchanged",
    ],
)
def test_initialized_gitlink_semantic_index_mutation_still_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_index_semantic" / "mutations"
        ),
        slice_id="nested_index_semantic",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    if mutation == "staged-mode":
        oid = _git(nested, "rev-parse", "HEAD:nested.txt").decode().strip()
        _git(
            nested,
            "update-index",
            "--cacheinfo",
            "100755",
            oid,
            "nested.txt",
        )
    else:
        _git(nested, "update-index", mutation, "nested.txt")

    with pytest.raises(MutationJournalError, match="staged Git index state"):
        journal.assert_repository_metadata_unchanged()

    assert journal.restore()


def test_initialized_gitlink_fsmonitor_view_mutation_still_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_index_fsmonitor" / "mutations"
        ),
        slice_id="nested_index_fsmonitor",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    original = _mutation_journal._index_semantic_views

    def changed_fsmonitor_view(cwd: Path, index: Path):
        views = original(cwd, index)
        if cwd.resolve() == nested.resolve():
            return views[0], views[1], views[2] + b"r19-fsmonitor-drift\0"
        return views

    with monkeypatch.context() as isolated:
        isolated.setattr(
            _mutation_journal,
            "_index_semantic_views",
            changed_fsmonitor_view,
        )
        with pytest.raises(
            MutationJournalError,
            match="staged Git index state",
        ):
            journal.assert_repository_metadata_unchanged()

    assert journal.restore()


def test_nested_gitlink_restore_never_mutates_replacement_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late child-root swap cannot redirect cleanup into checkout B."""
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    replacement = tmp_path / "replacement-nested"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--no-hardlinks",
            str(nested),
            str(replacement),
        ],
        check=True,
        capture_output=True,
    )
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_root_swap" / "mutations"
        ),
        slice_id="nested_root_swap",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    attacked_relative = "late-untracked.txt"
    (nested / attacked_relative).write_text(
        "untracked A\n",
        encoding="utf-8",
    )
    replacement_target = replacement / attacked_relative
    replacement_target.write_text("protected B\n", encoding="utf-8")
    replacement_before = replacement_target.read_bytes()

    real_remove = _mutation_journal._remove_bound_workspace_entry
    displaced = tmp_path / "held-nested-a"
    calls = 0

    def _swap_only_around_remove(
        root_fd: int,
        relative: str,
        parent_fd: int,
        name: str,
        parent_chain: tuple[tuple[int, int], ...],
        *,
        allow_terminal_vcs_metadata: bool = False,
    ) -> None:
        nonlocal calls
        if relative == attacked_relative:
            nested.rename(displaced)
            replacement.rename(nested)
            try:
                real_remove(
                    root_fd,
                    relative,
                    parent_fd,
                    name,
                    parent_chain,
                    allow_terminal_vcs_metadata=allow_terminal_vcs_metadata,
                )
            finally:
                nested.rename(replacement)
                displaced.rename(nested)
                calls += 1
            return
        real_remove(
            root_fd,
            relative,
            parent_fd,
            name,
            parent_chain,
            allow_terminal_vcs_metadata=allow_terminal_vcs_metadata,
        )

    monkeypatch.setattr(
        _mutation_journal,
        "_remove_bound_workspace_entry",
        _swap_only_around_remove,
    )
    with workspace_mutation_lock(repo):
        restored = journal.restore()

    assert calls == 1
    assert replacement_target.read_bytes() == replacement_before
    # The authorized A cleanup either completes through a held descriptor or
    # fails closed with its journal armed; it can never consume B's bytes.
    if restored:
        assert not (nested / attacked_relative).exists()
        assert not journal.journal_dir.exists()
    else:
        assert (nested / attacked_relative).read_text(encoding="utf-8") == (
            "untracked A\n"
        )
        assert journal.journal_dir.exists()


def test_linked_peer_restore_never_mutates_replacement_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late linked-peer swap cannot redirect cleanup into checkout B."""
    repo, corpus = _setup(tmp_path)
    linked = tmp_path / "linked-peer-a"
    _git(repo, "worktree", "add", "-q", "--detach", str(linked), "HEAD")
    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    replacement = tmp_path / "linked-peer-b"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--no-hardlinks",
            str(repo),
            str(replacement),
        ],
        check=True,
        capture_output=True,
    )
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "linked_peer_swap" / "mutations"
        ),
        slice_id="linked_peer_swap",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _git(replacement, "branch", "b-protected")
    protected_ref = replacement / ".git" / "refs" / "heads" / "b-protected"
    protected_before = protected_ref.read_bytes()
    git_entry = linked / ".git"
    git_entry_before = git_entry.read_bytes()
    journal_type = _mutation_journal.WorkspaceMutationJournal
    real_restore_refs = journal_type._restore_refs
    calls = 0

    def _graft_only_around_refs(self, cwd: Path) -> None:
        nonlocal calls
        if "::linked-peer-worktree::" not in self.slice_id:
            return real_restore_refs(self, cwd)
        git_entry.write_text(
            f"gitdir: {replacement / '.git'}\n",
            encoding="utf-8",
        )
        try:
            real_restore_refs(self, cwd)
        finally:
            git_entry.write_bytes(git_entry_before)
            calls += 1

    monkeypatch.setattr(
        journal_type,
        "_restore_refs",
        _graft_only_around_refs,
    )
    with workspace_mutation_lock(repo):
        restored = journal.restore()

    assert calls == 1
    assert protected_ref.read_bytes() == protected_before
    assert git_entry.read_bytes() == git_entry_before
    if restored:
        assert not journal.journal_dir.exists()
    else:
        assert journal.journal_dir.exists()


def test_nested_gitlink_restore_rejects_late_git_entry_graft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child .git graft cannot redirect ref restoration into repo B."""
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    replacement = tmp_path / "nested-git-graft-b"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--no-hardlinks",
            str(nested),
            str(replacement),
        ],
        check=True,
        capture_output=True,
    )
    _git(replacement, "branch", "b-protected")
    protected_ref = replacement / ".git" / "refs" / "heads" / "b-protected"
    protected_before = protected_ref.read_bytes()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_git_graft" / "mutations"
        ),
        slice_id="nested_git_graft",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    git_entry = nested / ".git"
    git_entry_before = git_entry.read_bytes()
    journal_type = _mutation_journal.WorkspaceMutationJournal
    real_restore_refs = journal_type._restore_refs
    calls = 0

    def _graft_only_around_refs(self, cwd: Path) -> None:
        nonlocal calls
        if "::gitlink::" not in self.slice_id:
            return real_restore_refs(self, cwd)
        git_entry.write_text(
            f"gitdir: {replacement / '.git'}\n",
            encoding="utf-8",
        )
        try:
            real_restore_refs(self, cwd)
        finally:
            git_entry.write_bytes(git_entry_before)
            calls += 1

    monkeypatch.setattr(journal_type, "_restore_refs", _graft_only_around_refs)
    with workspace_mutation_lock(repo):
        restored = journal.restore()

    assert calls == 1
    assert protected_ref.read_bytes() == protected_before
    assert git_entry.read_bytes() == git_entry_before
    if restored:
        assert not journal.journal_dir.exists()
    else:
        assert journal.journal_dir.exists()


def test_nested_gitlink_restore_binds_common_dir_across_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child common-dir swap cannot redirect control restoration into B."""
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows import _mutation_journal
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        workspace_mutation_lock,
    )

    replacement = tmp_path / "nested-common-b"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--no-hardlinks",
            str(nested),
            str(replacement),
        ],
        check=True,
        capture_output=True,
    )
    _git(replacement, "config", "b.protected", "true")
    replacement_config = replacement / ".git" / "config"
    replacement_before = replacement_config.read_bytes()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_common_swap" / "mutations"
        ),
        slice_id="nested_common_swap",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = next(
        candidate
        for candidate in journal._nested_journals()
        if "::gitlink::" in candidate.slice_id
    )
    child_common = Path(str(child._manifest["git_common_dir"]))
    displaced_common = tmp_path / "held-child-common-a"
    journal_type = _mutation_journal.WorkspaceMutationJournal
    real_restore_controls = journal_type._restore_control_files
    calls = 0

    def _swap_only_around_controls(self) -> None:
        nonlocal calls
        if "::gitlink::" not in self.slice_id:
            return real_restore_controls(self)
        child_common.rename(displaced_common)
        (replacement / ".git").rename(child_common)
        try:
            real_restore_controls(self)
        finally:
            child_common.rename(replacement / ".git")
            displaced_common.rename(child_common)
            calls += 1

    monkeypatch.setattr(
        journal_type,
        "_restore_control_files",
        _swap_only_around_controls,
    )
    with workspace_mutation_lock(repo):
        restored = journal.restore()

    assert calls > 0
    assert replacement_config.read_bytes() == replacement_before
    if restored:
        assert not journal.journal_dir.exists()
    else:
        assert journal.journal_dir.exists()


def test_initialized_gitlink_worktree_change_fails_and_retains_journal(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "nested_drift" / "mutations",
        slice_id="nested_drift",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (nested / "nested.txt").write_text("nested drift\n", encoding="utf-8")

    assert not journal.restore()
    assert journal.journal_dir.is_dir()
    assert (nested / "nested.txt").read_text(encoding="utf-8") == "nested drift\n"


def test_new_out_of_scope_nested_vcs_marker_is_removed_globally(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    (repo / "other").mkdir()
    (repo / "other" / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "other/tracked.txt")
    _git(repo, "commit", "-qm", "tracked outside scope")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "nested_marker" / "mutations",
        slice_id="nested_marker",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _git(repo / "other", "init", "-q")
    with pytest.raises(
        MutationJournalError,
        match="nested VCS|directory topology",
    ):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not (repo / "other" / ".git").exists()
    assert (repo / "other" / "tracked.txt").read_text() == "baseline\n"


def test_direct_checkout_journal_root_does_not_delete_own_snapshot(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=repo / "journal",
        slice_id="direct_root",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# changed\n", encoding="utf-8")
    assert journal.restore()
    assert (repo / "a.py").read_text() == "# baseline a.py\n"
    assert list((repo / "journal").iterdir()) == []


def test_journal_restores_commit_message_control(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    commit_message = Path(
        _git(repo, "rev-parse", "--git-path", "COMMIT_EDITMSG").decode().strip()
    )
    if not commit_message.is_absolute():
        commit_message = repo / commit_message
    before = commit_message.read_bytes() if commit_message.exists() else None
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "commit_message" / "mutations",
        slice_id="commit_message",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _git(repo, "commit", "--allow-empty", "-qm", "rogue commit message")
    assert journal.restore()
    assert (
        commit_message.read_bytes() if commit_message.exists() else None
    ) == before


def test_journal_rejects_and_restores_git_invisible_directory_modes(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    (repo / "outside").mkdir()
    (repo / "outside" / "tracked.txt").write_text("baseline\n")
    _git(repo, "add", "outside/tracked.txt")
    _git(repo, "commit", "-qm", "tracked directory")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    root_mode = repo.stat().st_mode & 0o777
    outside_mode = (repo / "outside").stat().st_mode & 0o777
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "directory_modes" / "mutations",
        slice_id="directory_modes",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    repo.chmod(0o700 if root_mode != 0o700 else 0o755)
    (repo / "outside").chmod(0o700 if outside_mode != 0o700 else 0o755)
    with pytest.raises(MutationJournalError, match="directory topology or modes"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert repo.stat().st_mode & 0o777 == root_mode
    assert (repo / "outside").stat().st_mode & 0o777 == outside_mode


def test_journal_rejects_empty_directory_even_inside_recursive_scope(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    (repo / "scoped").mkdir()
    (repo / "scoped" / "tracked.txt").write_text("baseline\n")
    _git(repo, "add", "scoped/tracked.txt")
    _git(repo, "commit", "-qm", "recursive scope")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "empty_directory" / "mutations",
        slice_id="empty_directory",
        scope=("scoped",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "scoped" / "empty").mkdir()
    with pytest.raises(MutationJournalError, match="directory topology or modes"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not (repo / "scoped" / "empty").exists()


@pytest.mark.parametrize("workspace_prefixed", [False, True])
def test_absent_trailing_slash_scope_authorizes_git_visible_tree(
    tmp_path: Path,
    workspace_prefixed: bool,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    rendered_scope = (
        f"{repo.name}/generated/" if workspace_prefixed else "generated/"
    )
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "absent_recursive" / "mutations"
        ),
        slice_id="absent_recursive",
        scope=(rendered_scope,),
        sequence=1,
        spec=corpus.latest_spec(),
    )

    assert journal._manifest["scope"] == ["generated"]
    assert journal._manifest["recursive_scope"] == ["generated"]
    (repo / "generated").mkdir()
    (repo / "generated" / "partial.py").write_text("PARTIAL = True\n")

    assert journal.detected_scope_escapes() == ()
    journal.assert_repository_metadata_unchanged()
    assert journal.locally_changed_scope_paths() == ("generated",)
    assert journal.restore()
    assert not (repo / "generated").exists()


def test_absent_recursive_scope_rejects_empty_directory_topology(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "absent_empty_directory" / "mutations"
        ),
        slice_id="absent_empty_directory",
        scope=("generated/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "generated").mkdir()

    with pytest.raises(MutationJournalError, match="directory topology or modes"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not (repo / "generated").exists()


def test_absent_recursive_scope_rejects_symlink_in_place_of_directory(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    external = tmp_path / "external-generated"
    external.mkdir()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "recursive_symlink" / "mutations"
        ),
        slice_id="recursive_symlink",
        scope=("generated/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "generated").symlink_to(external, target_is_directory=True)

    with pytest.raises(MutationJournalError, match="directory topology or modes"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not (repo / "generated").exists()
    assert external.is_dir()


def test_exact_absent_file_scope_authorizes_git_visible_parent_creation(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "exact_parent_create" / "mutations"
        ),
        slice_id="exact_parent_create",
        scope=("tests/test_breach.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_breach.py").write_text("def test_ok(): pass\n")

    assert journal.detected_scope_escapes() == ()
    journal.assert_repository_metadata_unchanged()
    assert journal.locally_changed_scope_paths() == ("tests/test_breach.py",)
    assert journal.restore()
    assert not (repo / "tests").exists()


def test_partial_restore_removes_empty_parents_of_absent_exact_files(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    paths = (
        "native/contracts/provider/schema.json",
        "native/contracts/provider/contract.json",
    )
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "partial_parent_restore" / "mutations"
        ),
        slice_id="partial_parent_restore",
        scope=paths,
        sequence=1,
        spec=corpus.latest_spec(),
    )
    for path in paths:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")

    journal.assert_repository_metadata_unchanged()
    journal.restore_paths_from_snapshot(paths)

    assert not (repo / "native").exists()
    journal.assert_repository_metadata_unchanged()


def test_exact_file_scope_authorizes_git_visible_parent_deletion(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    target = repo / "tests" / "test_breach.py"
    target.parent.mkdir()
    target.write_text("def test_baseline(): pass\n")
    _git(repo, "add", target.relative_to(repo).as_posix())
    _git(repo, "commit", "-qm", "nested exact file")
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "exact_parent_delete" / "mutations"
        ),
        slice_id="exact_parent_delete",
        scope=("tests/test_breach.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    target.unlink()
    target.parent.rmdir()

    assert journal.detected_scope_escapes() == ()
    journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert target.read_text() == "def test_baseline(): pass\n"


def test_scoped_recursive_delete_tool_is_restored_by_mutation_journal(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    (donor / "nested").mkdir(parents=True)
    (donor / ".gitattributes").write_text("* text=auto eol=lf\n")
    (donor / ".gitignore").write_text("cache/\n")
    (donor / "root.txt").write_text("root\n")
    (donor / "nested" / "child.txt").write_text("child\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add donor")
    (donor / "cache").mkdir()
    (donor / "cache" / "ignored.bin").write_bytes(b"ignored\x00payload")
    from crucible.tools import ToolContext
    from crucible.tools.write import build_write_tools
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "recursive_delete" / "mutations"
        ),
        slice_id="recursive_delete",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    result = build_write_tools(ToolContext(
        workspace_root=repo,
        repo_root=repo,
        write_scope=("donor",),
    )).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })

    assert result.startswith("OK: recursively deleted")
    assert not donor.exists()
    assert journal.detected_scope_escapes() == ()
    journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert (donor / ".gitattributes").read_text() == "* text=auto eol=lf\n"
    assert (donor / ".gitignore").read_text() == "cache/\n"
    assert (donor / "root.txt").read_text() == "root\n"
    assert (donor / "nested" / "child.txt").read_text() == "child\n"
    assert (donor / "cache" / "ignored.bin").read_bytes() == (
        b"ignored\x00payload"
    )


def test_absent_attribute_policy_leaf_accepts_recreated_empty_parent(
    tmp_path: Path,
):
    """Recovery treats a still-absent tracked policy leaf as a no-op."""

    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    donor.mkdir()
    policy = donor / ".gitattributes"
    policy.write_text("* text=auto eol=lf\n")
    _git(repo, "add", "donor/.gitattributes")
    _git(repo, "commit", "-qm", "add donor attribute policy")
    shutil.rmtree(donor)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "absent_attribute_leaf" / "mutations"
        ),
        slice_id="absent_attribute_leaf",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert {
        (record["path"], record["exists"])
        for record in journal._manifest["attribute_policy"]
    } >= {("donor/.gitattributes", False)}

    # A prior partial restore can legitimately recreate the historical parent
    # before attribute-policy settlement reaches its absent child.
    donor.mkdir()
    baseline = (repo / "a.py").read_bytes()
    (repo / "a.py").write_bytes(b"provider mutation\n")
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.mutation_journals_recovered == ("absent_attribute_leaf",)
    assert report.mutation_journals_preserved == ()
    assert not journal.journal_dir.exists()
    assert (repo / "a.py").read_bytes() == baseline
    assert not donor.exists()


def test_recursive_scope_partial_attribute_parent_delete_rolls_back(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    nested = donor / "nested"
    nested.mkdir(parents=True)
    (nested / ".gitattributes").write_text("* text=auto eol=lf\n")
    (nested / "child.txt").write_text("child\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add nested attribute policy")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "partial_attribute_parent" / "mutations"
        ),
        slice_id="partial_attribute_parent",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    shutil.rmtree(nested)

    with pytest.raises(
        MutationJournalError,
        match="Git attribute policy parent disappeared",
    ):
        journal.detected_scope_escapes()
    assert journal.restore()
    assert (nested / ".gitattributes").read_text() == "* text=auto eol=lf\n"
    assert (nested / "child.txt").read_text() == "child\n"


def test_recursive_attribute_tombstone_survives_selective_restore(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    donor.mkdir()
    (donor / ".gitattributes").write_text("* text=auto eol=lf\n")
    (donor / "child.txt").write_text("child\n")
    nested = donor / "nested"
    nested.mkdir()
    (nested / "nested.txt").write_text("nested\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add selectively restored donor")
    from crucible.tools import ToolContext
    from crucible.tools.write import build_write_tools
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "selective_attribute_tree" / "mutations"
        ),
        slice_id="selective_attribute_tree",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    donor_fd = os.open(donor, os.O_RDONLY | os.O_DIRECTORY)
    nested_fd = os.open(nested, os.O_RDONLY | os.O_DIRECTORY)
    result = build_write_tools(ToolContext(
        workspace_root=repo,
        repo_root=repo,
        write_scope=("donor",),
    )).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })
    assert result.startswith("OK: recursively deleted")

    journal.prepare_git_capture()
    assert journal._manifest[
        "deferred_recursive_attribute_policy_changes"
    ] == {"donor/.gitattributes": "donor"}
    _simulate_process_death(journal)
    journal = WorkspaceMutationJournal.load(journal.journal_dir)

    journal.restore_paths_from_snapshot(("donor",))
    os.close(nested_fd)
    os.close(donor_fd)

    assert (donor / ".gitattributes").read_text() == "* text=auto eol=lf\n"
    assert (donor / "child.txt").read_text() == "child\n"
    assert (nested / "nested.txt").read_text() == "nested\n"
    assert journal._manifest["pre_capture_attribute_changes"] == []
    assert journal._manifest[
        "deferred_recursive_attribute_policy_changes"
    ] == {}
    restored_authority = journal._manifest[
        "restored_workspace_directory_authority"
    ]
    assert "donor" in restored_authority
    restored_identity = donor.stat()
    assert restored_authority["donor"] == {
        "device": restored_identity.st_dev,
        "inode": restored_identity.st_ino,
    }
    nested_identity = nested.stat()
    assert restored_authority["donor/nested"] == {
        "device": nested_identity.st_dev,
        "inode": nested_identity.st_ino,
    }
    restored_entries = journal._manifest["restored_tracked_entry_authority"]
    for relative in (
        "donor/.gitattributes",
        "donor/child.txt",
        "donor/nested/nested.txt",
    ):
        identity = (repo / relative).lstat()
        assert restored_entries[relative] == {
            "device": identity.st_dev,
            "inode": identity.st_ino,
        }
    _simulate_process_death(journal)
    journal = WorkspaceMutationJournal.load(journal.journal_dir)
    (donor / "child.txt").write_text("changed within rebound inode\n")
    journal.restore_paths_from_snapshot(("donor/child.txt",))
    assert (donor / "child.txt").read_text() == "child\n"
    child_identity = (donor / "child.txt").stat()
    assert journal._manifest["restored_tracked_entry_authority"][
        "donor/child.txt"
    ] == {
        "device": child_identity.st_dev,
        "inode": child_identity.st_ino,
    }
    authorized_donor = tmp_path / "authorized-donor"
    donor.rename(authorized_donor)
    shutil.copytree(authorized_donor, donor, copy_function=shutil.copy2)
    assert not journal.restore()
    shutil.rmtree(donor)
    authorized_donor.rename(donor)
    assert journal.detected_scope_escapes() == ()
    assert journal.restore()


def test_selective_restore_publishes_directory_authority_before_later_failure(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    donor.mkdir()
    (donor / ".gitattributes").write_text("* text=auto eol=lf\n")
    (donor / "child.txt").write_text("child\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add interrupted selective restore tree")
    from crucible.tools import ToolContext
    from crucible.tools.write import build_write_tools
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "interrupted_selective_restore" / "mutations"
        ),
        slice_id="interrupted_selective_restore",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    # Keep the deleted inode alive until its replacement is published so the
    # filesystem allocator cannot reuse it and bypass this test's replacement
    # authority branch.
    donor_fd = os.open(donor, os.O_RDONLY | os.O_DIRECTORY)
    result = build_write_tools(ToolContext(
        workspace_root=repo,
        repo_root=repo,
        write_scope=("donor",),
    )).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })
    assert result.startswith("OK: recursively deleted")
    journal.prepare_git_capture()
    _simulate_process_death(journal)
    journal = WorkspaceMutationJournal.load(journal.journal_dir)

    def fail_after_directory_publication(*_args, **_kwargs):
        raise MutationJournalError("injected tracked-metadata failure")

    monkeypatch.setattr(
        journal,
        "_restore_tracked_path_metadata",
        fail_after_directory_publication,
    )
    with pytest.raises(
        MutationJournalError,
        match="injected tracked-metadata failure",
    ):
        journal.restore_paths_from_snapshot(("donor",))
    os.close(donor_fd)
    restored_identity = donor.stat()
    assert journal._manifest[
        "restored_workspace_directory_authority"
    ]["donor"] == {
        "device": restored_identity.st_dev,
        "inode": restored_identity.st_ino,
    }

    _simulate_process_death(journal)
    journal = WorkspaceMutationJournal.load(journal.journal_dir)
    assert journal.restore()
    assert (donor / "child.txt").read_text() == "child\n"


def test_recursive_attribute_tombstone_rejects_changed_recreated_root(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    donor.mkdir()
    (donor / ".gitattributes").write_text("* text=auto eol=lf\n")
    (donor / "child.txt").write_text("child\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add recreated donor")
    from crucible.tools import ToolContext
    from crucible.tools.write import build_write_tools
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "changed_attribute_tree" / "mutations"
        ),
        slice_id="changed_attribute_tree",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    result = build_write_tools(ToolContext(
        workspace_root=repo,
        repo_root=repo,
        write_scope=("donor",),
    )).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })
    assert result.startswith("OK: recursively deleted")
    journal.prepare_git_capture()

    donor.mkdir()
    (donor / ".gitattributes").write_text("* text=auto eol=lf\n")
    (donor / "child.txt").write_text("changed after tombstone\n")

    with pytest.raises(
        MutationJournalError,
        match="mutation of Git attribute policy is unsupported",
    ):
        journal.detected_scope_escapes()
    assert not journal.restore()
    shutil.rmtree(donor)
    assert journal.restore()
    assert (donor / "child.txt").read_text() == "child\n"


@pytest.mark.parametrize(
    "malformation",
    ("root", "unknown", "bool", "missing_component", "duplicate_identity"),
)
def test_restored_directory_authority_rejects_malformed_records(
    tmp_path: Path,
    malformation: str,
):
    repo, corpus = _setup(tmp_path)
    package = repo / "package"
    package.mkdir()
    (package / "child.txt").write_text("child\n")
    _git(repo, "add", "package")
    _git(repo, "commit", "-qm", "add authority directory")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "malformed_directory_authority" / "mutations"
        ),
        slice_id=f"malformed_directory_authority_{malformation}",
        scope=("package/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    identity = package.stat()
    record: dict[str, object]
    if malformation == "root":
        record = {".": {"device": identity.st_dev, "inode": identity.st_ino}}
    elif malformation == "unknown":
        record = {
            "unknown": {"device": identity.st_dev, "inode": identity.st_ino}
        }
    elif malformation == "bool":
        record = {"package": {"device": True, "inode": identity.st_ino}}
    elif malformation == "duplicate_identity":
        root_identity = journal._manifest["directory_identities"]["."]
        record = {"package": dict(root_identity)}
    else:
        record = {"package": {"device": identity.st_dev}}
    journal._manifest["restored_workspace_directory_authority"] = record

    with pytest.raises(
        MutationJournalError,
        match="restored workspace directory authority is malformed",
    ):
        journal._effective_workspace_directory_identities(
            journal._manifest["directory_modes"]
        )
    journal._manifest["restored_workspace_directory_authority"] = {}
    assert journal.restore()


@pytest.mark.parametrize(
    "malformation",
    (
        "unknown",
        "bool",
        "missing_component",
        "directory_alias",
        "duplicate_identity",
    ),
)
def test_restored_tracked_entry_authority_rejects_malformed_records(
    tmp_path: Path,
    malformation: str,
):
    repo, corpus = _setup(tmp_path)
    package = repo / "package"
    package.mkdir()
    first = package / "first.txt"
    second = package / "second.txt"
    first.write_text("first\n")
    second.write_text("second\n")
    _git(repo, "add", "package")
    _git(repo, "commit", "-qm", "add tracked authority files")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "malformed_tracked_authority" / "mutations"
        ),
        slice_id=f"malformed_tracked_authority_{malformation}",
        scope=("package/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    first_identity = first.stat()
    if malformation == "unknown":
        record = {
            "unknown.txt": {
                "device": first_identity.st_dev,
                "inode": first_identity.st_ino,
            }
        }
    elif malformation == "bool":
        record = {
            "package/first.txt": {
                "device": True,
                "inode": first_identity.st_ino,
            }
        }
    elif malformation == "missing_component":
        record = {"package/first.txt": {"device": first_identity.st_dev}}
    elif malformation == "directory_alias":
        directory_identity = package.stat()
        record = {
            "package/first.txt": {
                "device": directory_identity.st_dev,
                "inode": directory_identity.st_ino,
            }
        }
    else:
        record = {
            relative: {
                "device": first_identity.st_dev,
                "inode": first_identity.st_ino,
            }
            for relative in (
                "package/first.txt",
                "package/second.txt",
            )
        }
    journal._manifest["restored_tracked_entry_authority"] = record

    with pytest.raises(
        MutationJournalError,
        match="restored tracked entry authority is malformed",
    ):
        journal._validated_restored_tracked_entry_authority()
    journal._manifest["restored_tracked_entry_authority"] = {}
    assert journal.restore()


def test_selective_restore_does_not_clear_direct_attribute_mutation(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    donor.mkdir()
    attribute = donor / ".gitattributes"
    attribute.write_text("* text=auto eol=lf\n")
    (donor / "child.txt").write_text("child\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add directly mutated attribute tree")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "direct_attribute_tree" / "mutations"
        ),
        slice_id="direct_attribute_tree",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    attribute.write_text("*.bin -text\n")
    journal.prepare_git_capture()

    assert attribute.read_text() == "* text=auto eol=lf\n"
    assert journal._manifest["pre_capture_attribute_changes"] == [
        "donor/.gitattributes"
    ]
    assert journal._manifest[
        "deferred_recursive_attribute_policy_changes"
    ] == {}
    with pytest.raises(
        MutationJournalError,
        match="mutation of Git attribute policy is unsupported",
    ):
        journal.restore_paths_from_snapshot(("donor",))
    assert journal._manifest["pre_capture_attribute_changes"] == [
        "donor/.gitattributes"
    ]
    assert journal.restore()


def test_malformed_attribute_tombstone_fails_before_policy_repair(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    donor.mkdir()
    attribute = donor / ".gitattributes"
    attribute.write_text("* text=auto eol=lf\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add malformed-proof attribute")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "malformed_attribute_proof" / "mutations"
        ),
        slice_id="malformed_attribute_proof",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    attribute.write_text("*.bin -text\n")
    journal._manifest[
        "deferred_recursive_attribute_policy_changes"
    ] = {"donor/.gitattributes": 7}
    journal._write()

    with pytest.raises(
        MutationJournalError,
        match="deferred recursive Git attribute policy proof is malformed",
    ):
        journal.prepare_git_capture()
    assert attribute.read_text() == "*.bin -text\n"

    journal._manifest[
        "deferred_recursive_attribute_policy_changes"
    ] = {}
    journal._write()
    journal.prepare_git_capture()
    assert attribute.read_text() == "* text=auto eol=lf\n"
    assert journal.restore()


def test_failed_rollback_does_not_resurrect_attribute_tombstone_proof(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    donor.mkdir()
    attribute = donor / ".gitattributes"
    attribute.write_text("* text=auto eol=lf\n")
    (donor / "child.txt").write_text("child\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add rollback attribute tree")
    from crucible.tools import ToolContext
    from crucible.tools.write import build_write_tools
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "failed_attribute_restore" / "mutations"
        ),
        slice_id="failed_attribute_restore",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    result = build_write_tools(ToolContext(
        workspace_root=repo,
        repo_root=repo,
        write_scope=("donor",),
    )).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })
    assert result.startswith("OK: recursively deleted")
    journal.prepare_git_capture()
    assert journal._manifest[
        "deferred_recursive_attribute_policy_changes"
    ] == {"donor/.gitattributes": "donor"}

    restore_index = journal._restore_index
    injected = False

    def restore_index_then_mutate(cwd: Path) -> None:
        nonlocal injected
        restore_index(cwd)
        if not injected:
            injected = True
            attribute.write_text("*.bin -text\n")

    monkeypatch.setattr(journal, "_restore_index", restore_index_then_mutate)
    assert not journal.restore()
    assert journal._manifest["pre_capture_attribute_changes"] == [
        "donor/.gitattributes"
    ]
    assert journal._manifest[
        "deferred_recursive_attribute_policy_changes"
    ] == {}

    _simulate_process_death(journal)
    journal = WorkspaceMutationJournal.load(journal.journal_dir)
    with pytest.raises(
        MutationJournalError,
        match="mutation of Git attribute policy is unsupported",
    ):
        journal.detected_scope_escapes()


def test_recursive_scope_attribute_tree_can_attach_as_deleted(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    donor = repo / "donor"
    donor.mkdir()
    (donor / ".gitattributes").write_text("* text=auto eol=lf\n")
    (donor / "child.txt").write_text("child\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add attachable attribute tree")
    base = _git(repo, "rev-parse", "HEAD").decode().strip()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    spec = _spec(base)
    original_tier = spec.tier_dag.tiers[0]
    original_slice = original_tier.slices[0]
    donor_slice = dataclasses.replace(
        original_slice,
        directive=dataclasses.replace(
            original_slice.directive,
            files_in_scope=("donor/",),
            blast_radius={"max_files": 2},
        ),
    )
    donor_tier = dataclasses.replace(original_tier, slices=(donor_slice,))
    spec = dataclasses.replace(
        spec,
        tier_dag=TierDAG(tiers=(donor_tier,)),
    )
    corpus.put_spec(spec)
    from assay.primitives.implement import capture_workspace_delta
    from crucible.tools import ToolContext
    from crucible.tools.write import build_write_tools
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "attribute_tree_attach" / "mutations"
        ),
        slice_id="s0",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    result = build_write_tools(ToolContext(
        workspace_root=repo,
        repo_root=repo,
        write_scope=("donor",),
    )).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })
    assert result.startswith("OK: recursively deleted")

    journal.prepare_git_capture()
    diff, files, base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("donor",),
    )
    assert files == ("donor/.gitattributes", "donor/child.txt")
    impl = Implementation(
        slice_id="s0",
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            diff,
            kind="diffs",
            name_hint="attribute-tree-delete.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]",
            kind="transcripts",
            name_hint="attribute-tree-delete.json",
        ),
        files_modified=files,
        scope_escapes=(),
        classification="success",
        execution=exec_ctx_now("test"),
    )
    impl = attach_resume_identity(
        corpus,
        spec,
        donor_tier,
        donor_slice,
        impl,
    )
    impl = journal.validate_implementation(impl)
    journal.prepare(impl, keep=True)
    operation = journal._manifest["expected_postimage"]
    assert isinstance(operation, dict)
    postimages = {
        record["path"]: record["state"]["kind"]
        for record in operation["postimages"]
    }
    assert postimages == {
        "donor": "missing",
        "donor/.gitattributes": "missing",
        "donor/child.txt": "missing",
    }
    corpus.commit(implementations=(impl,))
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.mutation_journals_recovered == ()
    assert report.mutation_journals_preserved == ("s0",)
    assert not donor.exists()
    assert not journal.journal_dir.exists()


def test_startup_reaper_restores_recursive_attribute_tree(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    (donor / "nested").mkdir(parents=True)
    (donor / ".gitignore").write_text("cache/\n")
    (donor / "nested" / ".gitattributes").write_text(
        "* text=auto eol=lf\n"
    )
    (donor / "nested" / "child.txt").write_text("child\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add startup recovery tree")
    (donor / "cache").mkdir()
    (donor / "cache" / "ignored.bin").write_bytes(b"ignored\x00payload")
    from crucible.tools import ToolContext
    from crucible.tools.write import build_write_tools
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "startup_attribute_tree" / "mutations"
        ),
        slice_id="startup_attribute_tree",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    result = build_write_tools(ToolContext(
        workspace_root=repo,
        repo_root=repo,
        write_scope=("donor",),
    )).dispatch("delete_source", {
        "file_path": "donor",
        "recursive": True,
    })
    assert result.startswith("OK: recursively deleted")
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.mutation_journals_recovered == ("startup_attribute_tree",)
    assert report.mutation_journals_preserved == ()
    assert not journal.journal_dir.exists()
    assert (donor / ".gitignore").read_text() == "cache/\n"
    assert (donor / "nested" / ".gitattributes").read_text() == (
        "* text=auto eol=lf\n"
    )
    assert (donor / "nested" / "child.txt").read_text() == "child\n"
    assert (donor / "cache" / "ignored.bin").read_bytes() == (
        b"ignored\x00payload"
    )


def test_recursive_scope_attribute_parent_missing_above_root_fails_closed(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    outer = repo / "outer"
    donor = outer / "donor"
    donor.mkdir(parents=True)
    (donor / ".gitattributes").write_text("* text=auto eol=lf\n")
    (donor / "child.txt").write_text("child\n")
    _git(repo, "add", "outer")
    _git(repo, "commit", "-qm", "add nested recursive root")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "attribute_ancestor" / "mutations"
        ),
        slice_id="attribute_ancestor",
        scope=("outer/donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    held_outer = tmp_path / "held-outer"
    outer.rename(held_outer)

    with pytest.raises(
        MutationJournalError,
        match="Git attribute policy parent disappeared",
    ):
        journal.detected_scope_escapes()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()
    held_outer.rename(outer)
    assert journal.restore()


def test_hosted_recursive_delete_settles_durable_quarantine_receipt(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    (donor / "nested").mkdir(parents=True)
    (donor / "root.txt").write_text("root\n")
    (donor / "nested" / "child.txt").write_text("child\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add hosted-delete donor")
    from crucible.tools import ToolContext
    from crucible.tools._isolated_mutation_host import (
        IsolatedMutationHost,
        build_isolated_toolset_spec,
    )
    from crucible.tools.write import build_write_tools
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "hosted_recursive_delete" / "mutations"
        ),
        slice_id="hosted_recursive_delete",
        scope=("donor",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    context = ToolContext(
        workspace_root=repo,
        repo_root=repo,
        write_scope=("donor",),
        mutation_journal_dir=journal.journal_dir,
        mutation_journal_transaction_id=journal.transaction_id,
    )
    toolset = build_write_tools(context).filter_to({
        "write_source",
        "apply_patch",
        "delete_source",
    })
    isolated = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset(toolset.write_names),
    )
    with IsolatedMutationHost(
        isolated,
        invocation_deadline=time.monotonic() + 10.0,
    ) as host:
        result = host.dispatch(
            "delete_source",
            {"file_path": "donor", "recursive": True},
        )

    assert result.startswith("OK: recursively deleted")
    assert not donor.exists()
    assert journal._manifest["pending_delete_quarantines"] == []
    assert journal.restore()
    assert (donor / "root.txt").read_text() == "root\n"
    assert (donor / "nested" / "child.txt").read_text() == "child\n"


def test_journal_restores_partial_delete_after_publish_barrier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, corpus = _setup(tmp_path)
    donor = repo / "donor"
    donor.mkdir()
    first = donor / "a.txt"
    second = donor / "b.txt"
    first.write_text("alpha\n")
    second.write_text("beta\n")
    _git(repo, "add", "donor")
    _git(repo, "commit", "-qm", "add partial-delete donor")
    from crucible.tools import ToolContext
    from crucible.tools.write import handlers as write_handlers
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "partial_delete" / "mutations"
        ),
        slice_id="partial_delete",
        scope=("donor/",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    barrier_seen = False

    def barrier() -> None:
        nonlocal barrier_seen
        barrier_seen = True

    real_delete = write_handlers._delete_tree_contents

    def fail_after_first_unlink(
        directory_fd,
        *,
        manifest,
        root_device,
        root_mount_id,
    ):
        del root_device, root_mount_id
        assert barrier_seen
        assert manifest[0].name == "a.txt"
        os.unlink(manifest[0].name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        raise OSError("injected failure after first unlink")

    monkeypatch.setattr(
        write_handlers,
        "_delete_tree_contents",
        fail_after_first_unlink,
    )
    with pytest.raises(
        write_handlers.DeleteMutationIndeterminate,
        match="journal rollback",
    ):
        write_handlers._delete_source_impl(
            ToolContext(
                workspace_root=repo,
                repo_root=repo,
                write_scope=("donor",),
            ),
            file_path="donor",
            recursive=True,
            publication_barrier=barrier,
        )
    monkeypatch.setattr(
        write_handlers,
        "_delete_tree_contents",
        real_delete,
    )

    assert barrier_seen
    assert donor.is_dir()
    assert not first.exists()
    assert second.read_text() == "beta\n"
    assert journal.restore()
    assert first.read_text() == "alpha\n"
    assert second.read_text() == "beta\n"


def test_exact_absent_file_scope_rejects_directory_substitution(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "exact_directory" / "mutations"
        ),
        slice_id="exact_directory",
        scope=("tests/test_breach.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "tests" / "test_breach.py").mkdir(parents=True)
    (repo / "tests" / "test_breach.py" / "child.py").write_text("ROGUE = True\n")

    with pytest.raises(MutationJournalError, match="directory topology or modes"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not (repo / "tests").exists()


def test_exact_absent_file_scope_rejects_unrelated_directory_topology(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "exact_unrelated" / "mutations"
        ),
        slice_id="exact_unrelated",
        scope=("tests/test_breach.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_breach.py").write_text("def test_ok(): pass\n")
    (repo / "unrelated-empty").mkdir()

    with pytest.raises(MutationJournalError, match="directory topology or modes"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not (repo / "tests").exists()
    assert not (repo / "unrelated-empty").exists()


def test_exact_absent_file_scope_rejects_empty_directory_substitution(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "exact_empty_directory" / "mutations"
        ),
        slice_id="exact_empty_directory",
        scope=("tests/test_breach.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "tests" / "test_breach.py").mkdir(parents=True)

    with pytest.raises(MutationJournalError, match="directory topology or modes"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not (repo / "tests").exists()


def test_new_registered_external_worktree_fails_and_retains_without_deleting(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    external = tmp_path / "preexisting-empty"
    external.mkdir()
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "worktree_add" / "mutations",
        slice_id="worktree_add",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _git(repo, "worktree", "add", "--detach", str(external), "HEAD")
    with pytest.raises(
        MutationJournalError,
        match="registered Git worktrees|admin top-level topology",
    ):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert external.is_dir()
    assert (external / ".git").exists()
    assert journal.journal_dir.is_dir()


def test_linked_worktree_guards_main_checkout_changes(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-q", "--detach", str(linked), "HEAD")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=linked,
        journal_root=corpus.spec_dir / "runs" / "linked_guard" / "mutations",
        slice_id="linked_guard",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# main checkout drift\n")
    with pytest.raises(
        MutationJournalError,
        match="nested Git worktree|tracked path",
    ):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()


def test_begin_and_commit_leave_raw_index_byte_identical(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    index = Path(_git(repo, "rev-parse", "--git-path", "index").decode().strip())
    if not index.is_absolute():
        index = repo / index
    before = index.read_bytes()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "index_noop" / "mutations",
        slice_id="index_noop",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    journal.commit()
    assert index.read_bytes() == before


def test_linked_begin_and_commit_preserve_both_raw_indexes(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    linked = tmp_path / "linked-index"
    _git(repo, "worktree", "add", "-q", "--detach", str(linked), "HEAD")
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    def index_path(cwd: Path) -> Path:
        path = Path(_git(cwd, "rev-parse", "--git-path", "index").decode().strip())
        return path if path.is_absolute() else cwd / path

    main_index = index_path(repo)
    linked_index = index_path(linked)
    before_main = main_index.read_bytes()
    before_linked = linked_index.read_bytes()
    journal = WorkspaceMutationJournal.begin(
        cwd=linked,
        journal_root=corpus.spec_dir / "runs" / "linked_index" / "mutations",
        slice_id="linked_index",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    journal.commit()
    assert main_index.read_bytes() == before_main
    assert linked_index.read_bytes() == before_linked


def test_transaction_created_hardlinked_submodule_admin_is_retained_fail_closed(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    origin = tmp_path / "added-origin"
    origin.mkdir()
    _git(origin, "init", "-q")
    _git(origin, "config", "user.email", "added@test.invalid")
    _git(origin, "config", "user.name", "added test")
    (origin / "lib.txt").write_text("origin\n")
    _git(origin, "add", "lib.txt")
    _git(origin, "commit", "-qm", "origin")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "module_add" / "mutations",
        slice_id="module_add",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(origin),
        "vendor/child",
    )
    with pytest.raises(MutationJournalError):
        journal.assert_repository_metadata_unchanged()
    # A local submodule clone hard-links object files to ``origin``.  Removing
    # that administration tree would mutate external inode link counts, so
    # exact rollback retains the journal instead of claiming a lossy success.
    assert not journal.restore()
    assert journal.journal_dir.exists()
    assert (origin / "lib.txt").read_text(encoding="utf-8") == "origin\n"
    _git(origin, "fsck", "--full", "--no-dangling", "HEAD")


def test_unknown_git_admin_directory_is_rejected_and_retained(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_dir,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "unknown_admin" / "mutations",
        slice_id="unknown_admin",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    admin = _git_dir(repo)
    (admin / "unknown-control").write_text("rogue\n")
    (admin / "unknown-tree").mkdir()
    (admin / "unknown-tree" / "state").write_text("rogue\n")
    with pytest.raises(MutationJournalError, match="admin top-level topology"):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert journal.journal_dir.exists()
    assert (admin / "unknown-control").read_text(encoding="utf-8") == "rogue\n"
    assert (admin / "unknown-tree" / "state").read_text(
        encoding="utf-8"
    ) == "rogue\n"


def test_custom_git_admin_tree_is_guarded_exactly(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_dir,
    )

    custom = _git_dir(repo) / "gk"
    custom.mkdir()
    (custom / "config").write_text("baseline\n")
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "custom_admin" / "mutations",
        slice_id="custom_admin",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (custom / "config").write_text("changed\n")
    with pytest.raises(MutationJournalError, match="custom Git admin tree"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert (custom / "config").read_text() == "baseline\n"


def test_registry_and_lease_residual_entries_are_rejected_and_removed(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
        _registry_directory,
    )

    leases = _git_common_dir(repo) / "crucible-worktree-leases"
    leases.mkdir()
    baseline_lease = leases / "baseline.json"
    baseline_lease.write_text("{}")
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "registry_residual" / "mutations",
        slice_id="registry_residual",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    registry_rogue = _registry_directory(repo) / "rogue.json"
    registry_rogue.write_text("{}")
    lease_rogue = leases / "rogue.json"
    lease_rogue.write_text("{}")
    with pytest.raises(MutationJournalError):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not registry_rogue.exists()
    assert not lease_rogue.exists()
    assert baseline_lease.read_text() == "{}"


def test_two_initialized_gitlink_children_share_registry_family(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    _add_clean_submodule(repo, tmp_path)
    second = tmp_path / "second-origin"
    second.mkdir()
    _git(second, "init", "-q")
    _git(second, "config", "user.email", "second@test.invalid")
    _git(second, "config", "user.name", "second test")
    (second / "second.txt").write_text("second\n")
    _git(second, "add", "second.txt")
    _git(second, "commit", "-qm", "second")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(second),
        "vendor/second",
    )
    _git(repo, "commit", "-qam", "add second nested dependency")
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _registry_directory,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "two_children" / "mutations",
        slice_id="two_children",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert len(journal._nested_journals()) == 2
    journal.assert_repository_metadata_unchanged()
    journal.commit()
    assert list(_registry_directory(repo).iterdir()) == []


def test_out_of_scope_tracked_mode_and_xattrs_restore_exactly(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    target = repo / "b.py"
    baseline_mode = target.stat().st_mode & 0o7777
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "tracked_metadata" / "mutations",
        slice_id="tracked_metadata",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    if hasattr(os, "setxattr"):
        os.setxattr(target, "user.crucible-test", b"rogue")
    target.chmod((baseline_mode & ~0o200) or 0o400)
    with pytest.raises(MutationJournalError, match="Git-invisible mode|xattr"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert target.stat().st_mode & 0o7777 == baseline_mode
    if hasattr(os, "listxattr"):
        assert "user.crucible-test" not in os.listxattr(target)


def test_same_size_same_mtime_poison_is_found_without_hashing_all_files(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    target = repo / "b.py"
    baseline = target.read_bytes()
    before = target.stat()
    poison = b"!" * len(baseline)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "stat_poison" / "mutations",
        slice_id="stat_poison",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    target.write_bytes(poison)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    # Exercise the fallback itself rather than depending on whether this
    # filesystem/Git build happens to trust the colliding stat tuple.
    monkeypatch.setattr(
        "crucible.workflows._mutation_journal._git_status_paths",
        lambda _cwd: set(),
    )
    with pytest.raises(MutationJournalError, match="tracked path"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert target.read_bytes() == baseline


def test_predirty_untracked_xattrs_round_trip_and_hardlinks_fail_closed(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    scratch = repo / "scratch.txt"
    scratch.write_text("user baseline\n")
    if hasattr(os, "setxattr"):
        os.setxattr(scratch, "user.crucible-test", b"baseline")
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "predirty_xattr" / "mutations",
        slice_id="predirty_xattr",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    if hasattr(os, "setxattr"):
        os.setxattr(scratch, "user.crucible-test", b"changed")
    assert journal.restore()
    if hasattr(os, "getxattr"):
        assert os.getxattr(scratch, "user.crucible-test") == b"baseline"

    alias = tmp_path / "scratch-alias"
    os.link(scratch, alias)
    with pytest.raises(MutationJournalError, match="hard-linked"):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=corpus.spec_dir / "runs" / "predirty_link" / "mutations",
            slice_id="predirty_link",
            scope=("a.py",),
            sequence=2,
            spec=corpus.latest_spec(),
        )


def test_hardlinked_tracked_path_is_rejected_and_alias_is_broken(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    target = repo / "b.py"
    external = tmp_path / "external-copy"
    external.write_bytes(target.read_bytes())
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "hardlink" / "mutations",
        slice_id="hardlink",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    target.unlink()
    os.link(external, target)
    with pytest.raises(MutationJournalError, match="link count"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert target.stat().st_nlink == 1
    assert external.stat().st_nlink == 1


@pytest.mark.parametrize("target_name", [None, "outside"])
def test_chmod_zero_directory_does_not_strand_rollback(
    tmp_path: Path,
    target_name: str | None,
):
    repo, corpus = _setup(tmp_path)
    (repo / "outside").mkdir()
    (repo / "outside" / "tracked.txt").write_text("baseline\n")
    _git(repo, "add", "outside/tracked.txt")
    _git(repo, "commit", "-qm", "outside directory")
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    target = repo if target_name is None else repo / target_name
    baseline_mode = target.stat().st_mode & 0o7777
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "chmod_zero" / "mutations",
        slice_id="chmod_zero",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    target.chmod(0)
    assert journal.restore()
    assert target.stat().st_mode & 0o7777 == baseline_mode


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "O_PATH"),
    reason="POSIX descriptor-bound inaccessible-root recovery",
)
def test_prelock_root_access_adds_only_owner_read_and_traversal(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows import _mutation_journal

    WorkspaceMutationJournal = _mutation_journal.WorkspaceMutationJournal

    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "minimal_root_access" / "mutations"
        ),
        slice_id="minimal_root_access",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    try:
        repo.chmod(0)
        with _mutation_journal._external_workspace_coordination(
            repo,
            captured_authority=journal._manifest["authority_lock"],
        ):
            journal._require_writer_epoch_quiescence(
                frontier="test root access preparation"
            )
            journal._prepare_workspace_root_access()
            assert stat.S_IMODE(repo.stat().st_mode) == (
                stat.S_IRUSR | stat.S_IXUSR
            )
    finally:
        repo.chmod(baseline_mode)
    assert journal.restore()


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "O_PATH"),
    reason="POSIX descriptor-bound inaccessible-root recovery",
)
def test_prelock_root_access_rejects_malformed_mode_before_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    from crucible.workflows import _mutation_journal as mutation_journal
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
    )

    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "malformed_root_mode" / "mutations"
        ),
        slice_id="malformed_root_mode",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    original_modes = dict(journal._manifest["directory_modes"])
    journal._manifest["directory_modes"]["."] = True
    journal._write()
    chmod_calls: list[tuple[object, object]] = []
    real_chmod = mutation_journal.os.chmod

    def recording_chmod(path, mode, *args, **kwargs):
        chmod_calls.append((path, mode))
        return real_chmod(path, mode, *args, **kwargs)

    try:
        repo.chmod(0)
        with monkeypatch.context() as guarded:
            guarded.setattr(mutation_journal.os, "chmod", recording_chmod)
            assert not journal.restore()
        assert chmod_calls == []
        assert stat.S_IMODE(repo.stat().st_mode) == 0
    finally:
        repo.chmod(baseline_mode)
        journal._manifest["directory_modes"] = original_modes
        journal._write()
    assert journal.restore()


def test_chmod_zero_added_object_directory_does_not_strand_rollback(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    journal_root = corpus.spec_dir / "runs" / "object_chmod_zero" / "mutations"
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="object_chmod_zero",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    addition = _git_common_dir(repo) / "objects" / "transaction-added"
    addition.mkdir()
    (addition / "payload").write_bytes(b"not a Git object\n")
    external = tmp_path / "object-symlink-target"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_bytes(b"outside object authority\n")
    external.chmod(0o500)
    external_mode = stat.S_IMODE(external.stat().st_mode)
    (addition / "external-alias").symlink_to(
        external,
        target_is_directory=True,
    )
    addition.chmod(0)

    assert journal.restore()
    assert not addition.exists()
    assert stat.S_IMODE(external.stat().st_mode) == external_mode
    assert sentinel.read_bytes() == b"outside object authority\n"

    # Recovery removed both the hostile addition and the durable transaction,
    # so the same checkout can immediately start another exact journal.
    retry = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="object_chmod_zero_retry",
        scope=("a.py",),
        sequence=2,
        spec=corpus.latest_spec(),
    )
    assert retry.restore()


def test_unreadable_baseline_object_directory_is_rejected_before_capture(
    tmp_path: Path,
):
    if os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("requires non-root POSIX directory permission enforcement")
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
        _registry_directory,
    )

    baseline_directory = _git_common_dir(repo) / "objects" / "unreadable-baseline"
    baseline_directory.mkdir()
    payload = baseline_directory / "payload"
    payload.write_bytes(b"must remain represented by the baseline\n")
    journal_root = corpus.spec_dir / "runs" / "unreadable_objects" / "mutations"
    journal = None
    baseline_directory.chmod(0)
    try:
        with pytest.raises(
            MutationJournalError,
            match="cannot completely traverse Git object baseline inventory",
        ):
            journal = WorkspaceMutationJournal.begin(
                cwd=repo,
                journal_root=journal_root,
                slice_id="unreadable_objects",
                scope=("a.py",),
                sequence=1,
                spec=corpus.latest_spec(),
            )
    finally:
        baseline_directory.chmod(0o700)
        if journal is not None:
            journal.restore()

    assert payload.read_bytes() == b"must remain represented by the baseline\n"
    assert not journal_root.exists() or not any(journal_root.iterdir())
    assert not any(_registry_directory(repo).iterdir())

    retry = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="readable_objects_retry",
        scope=("a.py",),
        sequence=2,
        spec=corpus.latest_spec(),
    )
    assert retry.restore()


def test_unreadable_baseline_refs_directory_is_rejected_before_capture(
    tmp_path: Path,
):
    if os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("requires non-root POSIX directory permission enforcement")
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
        _registry_directory,
    )

    baseline_directory = _git_common_dir(repo) / "refs" / "unreadable-baseline"
    baseline_directory.mkdir()
    payload = baseline_directory / "payload"
    payload.write_bytes(b"must not disappear from the refs baseline\n")
    journal_root = corpus.spec_dir / "runs" / "unreadable_refs" / "mutations"
    journal = None
    baseline_directory.chmod(0)
    try:
        with pytest.raises(
            MutationJournalError,
            match=r"cannot completely traverse .*Git administrative tree",
        ):
            journal = WorkspaceMutationJournal.begin(
                cwd=repo,
                journal_root=journal_root,
                slice_id="unreadable_refs",
                scope=("a.py",),
                sequence=1,
                spec=corpus.latest_spec(),
            )
    finally:
        baseline_directory.chmod(0o700)
        if journal is not None:
            journal.restore()

    assert payload.read_bytes() == b"must not disappear from the refs baseline\n"
    assert not journal_root.exists() or not any(journal_root.iterdir())
    assert not any(_registry_directory(repo).iterdir())


def test_chmod_zero_added_registry_directory_does_not_strand_rollback(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _registry_directory,
    )

    journal_root = corpus.spec_dir / "runs" / "registry_chmod_zero" / "mutations"
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="registry_chmod_zero",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    addition = _registry_directory(repo) / "transaction-added"
    addition.mkdir()
    (addition / "payload").write_bytes(b"not a registry marker\n")
    addition.chmod(0)

    try:
        restored = journal.restore()
    finally:
        if addition.exists():
            addition.chmod(0o700)
    assert restored
    assert not addition.exists()

    retry = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="registry_chmod_zero_retry",
        scope=("a.py",),
        sequence=2,
        spec=corpus.latest_spec(),
    )
    assert retry.restore()


@pytest.mark.parametrize(
    "relative",
    (
        "transaction-added-admin",
        "modules/transaction-added",
        "worktrees/transaction-added",
    ),
)
def test_chmod_zero_added_git_admin_directory_is_retained_fail_closed(
    tmp_path: Path,
    relative: str,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    common_dir = _git_common_dir(repo)
    addition = common_dir / relative
    addition.parent.mkdir(parents=True, exist_ok=True)
    journal_root = corpus.spec_dir / "runs" / "admin_chmod_zero" / "mutations"
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="admin_chmod_zero",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    addition.mkdir()
    (addition / "payload").write_bytes(b"transaction-added admin payload\n")
    addition.chmod(0)

    try:
        restored = journal.restore()
    finally:
        if addition.exists():
            addition.chmod(0o700)
    assert not restored
    assert journal.journal_dir.exists()
    assert addition.is_dir()
    assert (addition / "payload").read_bytes() == (
        b"transaction-added admin payload\n"
    )


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX descriptor and mode-access contract",
)
def test_access_preparation_reopens_wide_tree_with_bounded_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.workflows import _mutation_journal as mutation_journal
    from crucible.workflows._mutation_journal import (
        _make_real_directory_tree_owner_accessible,
    )

    root = tmp_path / "wide-controlled-tree"
    root.mkdir()
    children = []
    for index in range(32):
        child = root / f"child-{index:02d}"
        child.mkdir()
        child.chmod(0)
        children.append(child)

    real_open = os.open
    real_dup = os.dup
    real_close = os.close
    tracked: set[int] = set()
    descriptor_limit = 6

    def ensure_capacity() -> None:
        if len(tracked) >= descriptor_limit:
            raise OSError(errno.EMFILE, "modeled descriptor limit")

    def tracked_open(*args, **kwargs):
        ensure_capacity()
        descriptor = real_open(*args, **kwargs)
        tracked.add(descriptor)
        return descriptor

    def tracked_dup(descriptor: int):
        ensure_capacity()
        duplicate = real_dup(descriptor)
        tracked.add(duplicate)
        return duplicate

    def tracked_close(descriptor: int):
        tracked.discard(descriptor)
        return real_close(descriptor)

    prepared_modes: list[int] | None = None
    try:
        with monkeypatch.context() as modeled:
            modeled.setattr(mutation_journal.os, "open", tracked_open)
            modeled.setattr(mutation_journal.os, "dup", tracked_dup)
            modeled.setattr(mutation_journal.os, "close", tracked_close)
            modeled.setattr(
                mutation_journal.os,
                "supports_dir_fd",
                {*mutation_journal.os.supports_dir_fd, tracked_open},
            )
            _make_real_directory_tree_owner_accessible(
                root,
                tree_label="modeled controlled tree",
            )
            prepared_modes = [
                stat.S_IMODE(child.stat().st_mode) for child in children
            ]
    finally:
        for child in children:
            child.chmod(0o700)

    assert not tracked
    assert prepared_modes is not None
    assert all(mode == 0o700 for mode in prepared_modes)


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "geteuid"),
    reason="ordinary-POSIX ownership contract",
)
def test_foreign_owned_initialized_gitlink_is_rejected_by_child_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    journal_root = corpus.spec_dir / "runs" / "foreign_gitlink" / "mutations"

    from crucible.workflows import _mutation_journal as mutation_journal
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _registry_directory,
    )

    real_lstat = Path.lstat
    real_restore_control = mutation_journal._assert_directory_restore_control
    modeled_euid = os.geteuid()
    modeled_foreign_uid = modeled_euid + 1

    def modeled_lstat(path: Path):
        metadata = real_lstat(path)
        in_repo = path == repo or repo in path.parents
        if not in_repo or not stat.S_ISDIR(metadata.st_mode):
            return metadata
        fields = list(metadata)
        modeled_uid = modeled_foreign_uid if path == nested else modeled_euid
        fields[4] = modeled_uid
        fields[5] = modeled_uid
        return os.stat_result(fields)

    def modeled_restore_control(path: Path, metadata: os.stat_result):
        fields = list(metadata)
        try:
            is_nested = path == nested or path.resolve() == nested
        except OSError:
            is_nested = path == nested
        modeled_uid = modeled_foreign_uid if is_nested else modeled_euid
        fields[4] = modeled_uid
        fields[5] = modeled_uid
        return real_restore_control(path, os.stat_result(fields))

    with monkeypatch.context() as modeled:
        modeled.setattr(Path, "lstat", modeled_lstat)
        modeled.setattr(
            mutation_journal,
            "_assert_directory_restore_control",
            modeled_restore_control,
        )
        modeled.setattr(
            mutation_journal.os,
            "geteuid",
            lambda: modeled_euid,
        )
        with pytest.raises(
            MutationJournalError,
            match="cannot control modeled baseline directory",
        ) as raised:
            WorkspaceMutationJournal.begin(
                cwd=repo,
                journal_root=journal_root,
                slice_id="foreign_gitlink",
                scope=("a.py",),
                sequence=1,
                spec=corpus.latest_spec(),
            )

    assert f"path={str(nested)!r}" in str(raised.value)
    assert not journal_root.exists() or not any(journal_root.iterdir())
    assert not any(_registry_directory(repo).iterdir())
    retry = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="foreign_gitlink_retry",
        scope=("a.py",),
        sequence=2,
        spec=corpus.latest_spec(),
    )
    assert retry.restore()


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "geteuid"),
    reason="ordinary-POSIX ownership contract",
)
def test_foreign_owned_ignored_root_remains_outside_restore_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, corpus = _setup(tmp_path)
    (repo / ".gitignore").write_text("ignored-cache/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "ignore cache")
    ignored = repo / "ignored-cache"
    ignored.mkdir(mode=0o750)
    (ignored / "payload").write_text("outside journal authority\n")
    baseline_mode = stat.S_IMODE(ignored.stat().st_mode)

    from crucible.workflows import _mutation_journal as mutation_journal
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    real_lstat = Path.lstat
    real_chmod = Path.chmod
    real_restore_control = mutation_journal._assert_directory_restore_control
    modeled_euid = os.geteuid()
    modeled_foreign_uid = modeled_euid + 1

    def modeled_lstat(path: Path):
        metadata = real_lstat(path)
        in_repo = path == repo or repo in path.parents
        if not in_repo or not stat.S_ISDIR(metadata.st_mode):
            return metadata
        fields = list(metadata)
        modeled_uid = modeled_foreign_uid if path == ignored else modeled_euid
        fields[4] = modeled_uid
        fields[5] = modeled_uid
        return os.stat_result(fields)

    def guarded_chmod(path: Path, mode: int, *args, **kwargs):
        if path == ignored or ignored in path.parents:
            raise AssertionError(f"ignored directory was chmodded: {path}")
        return real_chmod(path, mode, *args, **kwargs)

    def modeled_restore_control(path: Path, metadata: os.stat_result):
        if path == ignored or ignored in path.parents:
            raise AssertionError(
                f"ignored directory entered restore authority: {path}"
            )
        fields = list(metadata)
        fields[4] = modeled_euid
        fields[5] = modeled_euid
        return real_restore_control(path, os.stat_result(fields))

    with monkeypatch.context() as modeled:
        modeled.setattr(Path, "lstat", modeled_lstat)
        modeled.setattr(Path, "chmod", guarded_chmod)
        modeled.setattr(
            mutation_journal,
            "_assert_directory_restore_control",
            modeled_restore_control,
        )
        journal = WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(corpus.spec_dir / "runs" / "foreign_ignored" / "mutations"),
            slice_id="foreign_ignored",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )
        assert journal.restore()

    assert ignored.is_dir()
    assert stat.S_IMODE(ignored.stat().st_mode) == baseline_mode
    assert (ignored / "payload").read_text() == "outside journal authority\n"


def test_baseline_ignored_recursive_roots_remain_opaque_to_topology_guard(
    tmp_path: Path,
):
    """Recursive build/cache growth cannot become guarded topology drift."""

    repo, corpus = _setup(tmp_path)
    (repo / ".gitignore").write_text(
        "build/\n__pycache__/\n",
        encoding="utf-8",
    )
    cache_root = repo / "internal" / "tools" / "__pycache__"
    build_root = repo / "ov-api" / "test_package" / "build"
    cache_root.mkdir(parents=True)
    build_root.mkdir(parents=True)
    (cache_root.parent / "keep.txt").write_text("tracked parent\n")
    (build_root.parent / "keep.txt").write_text("tracked parent\n")
    (cache_root / "baseline.pyc").write_bytes(b"baseline cache\n")
    (build_root / "baseline.o").write_bytes(b"baseline build\n")
    _git(
        repo,
        "add",
        ".gitignore",
        "internal/tools/keep.txt",
        "ov-api/test_package/keep.txt",
    )
    _git(repo, "commit", "-qm", "ignore recursive build products")

    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "ignored_recursive" / "mutations"
        ),
        slice_id="ignored_recursive",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert {"internal/tools/__pycache__", "ov-api/test_package/build"} <= set(
        journal._manifest["ignored_nested_vcs_roots"]
    )

    (repo / "a.py").write_text("# accepted source edit\n", encoding="utf-8")
    generated_cache = cache_root / "nested" / "generated.pyc"
    generated_cache.parent.mkdir()
    generated_cache.write_bytes(b"generated cache\n")
    generated_build = build_root / "linux" / "Release" / "artifact.o"
    generated_build.parent.mkdir(parents=True)
    generated_build.write_bytes(b"generated build\n")

    journal.assert_repository_metadata_unchanged()

    unrelated = repo / "unrelated-empty"
    unrelated.mkdir()
    with pytest.raises(MutationJournalError, match="directory topology or modes"):
        journal.assert_repository_metadata_unchanged()

    assert journal.restore()
    assert (repo / "a.py").read_text(encoding="utf-8") == "# baseline a.py\n"
    assert not unrelated.exists()
    assert generated_cache.read_bytes() == b"generated cache\n"
    assert generated_build.read_bytes() == b"generated build\n"


def test_declared_ignored_root_has_exact_directory_authority(tmp_path: Path):
    """An explicitly scoped ignored tree is exact, not opaque build dirt."""

    repo, corpus = _setup(tmp_path)
    (repo / ".gitignore").write_text("ignored-cache/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "ignore cache")
    ignored = repo / "ignored-cache"
    ignored.mkdir(mode=0o750)
    baseline = ignored / "payload"
    baseline.write_text("declared baseline\n", encoding="utf-8")

    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "declared_ignored" / "mutations"
        ),
        slice_id="declared_ignored",
        scope=("ignored-cache",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    assert "ignored-cache" in journal._manifest["directory_identities"]

    baseline.write_text("overwritten\n", encoding="utf-8")
    (ignored / ".git").mkdir()
    (ignored / "created").mkdir()
    (ignored / "created" / "artifact").write_text(
        "created\n",
        encoding="utf-8",
    )
    with pytest.raises(
        MutationJournalError,
        match="directory topology|nested VCS",
    ):
        journal.assert_repository_metadata_unchanged()

    assert journal.restore(full=True)
    assert baseline.read_text(encoding="utf-8") == "declared baseline\n"
    assert not (ignored / ".git").exists()
    assert not (ignored / "created").exists()
    assert stat.S_IMODE(ignored.stat().st_mode) == 0o750


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "geteuid"),
    reason="ordinary-POSIX ownership contract",
)
def test_foreign_owned_modeled_directory_is_rejected_during_begin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, corpus = _setup(tmp_path)
    foreign = repo / "foreign-baseline"
    foreign.mkdir(mode=0o755)
    foreign.chmod(0o755)
    journal_root = corpus.spec_dir / "runs" / "foreign_directory" / "mutations"

    from crucible.workflows import _mutation_journal as mutation_journal
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    real_lstat = Path.lstat
    real_restore_control = mutation_journal._assert_directory_restore_control
    modeled_euid = os.geteuid()
    modeled_foreign_uid = modeled_euid + 1
    modeled_foreign_gid = modeled_euid + 2

    def modeled_lstat(path: Path):
        metadata = real_lstat(path)
        in_repo = path == repo or repo in path.parents
        if not in_repo or metadata.st_mode & 0o170000 != 0o040000:
            return metadata
        fields = list(metadata)
        fields[4] = modeled_foreign_uid if path == foreign else modeled_euid
        fields[5] = modeled_foreign_gid if path == foreign else modeled_euid
        return os.stat_result(fields)

    def modeled_restore_control(path: Path, metadata: os.stat_result):
        fields = list(metadata)
        try:
            is_foreign = path == foreign or path.resolve() == foreign
        except OSError:
            is_foreign = path == foreign
        fields[4] = modeled_foreign_uid if is_foreign else modeled_euid
        fields[5] = modeled_foreign_gid if is_foreign else modeled_euid
        return real_restore_control(path, os.stat_result(fields))

    with monkeypatch.context() as modeled:
        modeled.setattr(Path, "lstat", modeled_lstat)
        modeled.setattr(
            mutation_journal,
            "_assert_directory_restore_control",
            modeled_restore_control,
        )
        modeled.setattr(
            mutation_journal.os,
            "geteuid",
            lambda: modeled_euid,
        )
        with pytest.raises(
            MutationJournalError,
            match="cannot control modeled baseline directory",
        ) as raised:
            WorkspaceMutationJournal.begin(
                cwd=repo,
                journal_root=journal_root,
                slice_id="foreign_directory",
                scope=("a.py",),
                sequence=1,
                spec=corpus.latest_spec(),
            )

    message = str(raised.value)
    assert f"path={str(foreign)!r}" in message
    assert f"uid={modeled_foreign_uid}" in message
    assert f"gid={modeled_foreign_gid}" in message
    assert "mode=0o0755" in message
    assert f"journal_euid={modeled_euid}" in message
    assert not journal_root.exists() or not any(journal_root.iterdir())

    # A failed preflight leaves no live/durable transaction behind, and the
    # same ordinary owned checkout can start and settle a fresh journal.
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="owned_directory",
        scope=("a.py",),
        sequence=2,
        spec=corpus.latest_spec(),
    )
    assert journal.restore()


@pytest.mark.parametrize(
    "relative_admin",
    (
        ".git",
        ".git/objects/pack",
        ".git/logs/refs",
    ),
)
@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "geteuid"),
    reason="ordinary-POSIX ownership contract",
)
def test_foreign_owned_git_admin_directory_is_rejected_during_begin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_admin: str,
):
    repo, corpus = _setup(tmp_path)
    foreign = repo / relative_admin
    assert foreign.is_dir()
    journal_root = corpus.spec_dir / "runs" / "foreign_admin" / "mutations"

    from crucible.workflows import _mutation_journal as mutation_journal
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    real_lstat = Path.lstat
    modeled_euid = os.geteuid()
    modeled_foreign_uid = modeled_euid + 1
    modeled_foreign_gid = modeled_euid + 2

    def modeled_lstat(path: Path):
        metadata = real_lstat(path)
        in_repo = path == repo or repo in path.parents
        if not in_repo or not stat.S_ISDIR(metadata.st_mode):
            return metadata
        fields = list(metadata)
        fields[4] = modeled_foreign_uid if path == foreign else modeled_euid
        fields[5] = modeled_foreign_gid if path == foreign else modeled_euid
        return os.stat_result(fields)

    with monkeypatch.context() as modeled:
        modeled.setattr(Path, "lstat", modeled_lstat)
        modeled.setattr(
            mutation_journal.os,
            "geteuid",
            lambda: modeled_euid,
        )
        with pytest.raises(
            MutationJournalError,
            match="ordinary-POSIX ownership contract",
        ) as raised:
            WorkspaceMutationJournal.begin(
                cwd=repo,
                journal_root=journal_root,
                slice_id="foreign_admin",
                scope=("a.py",),
                sequence=1,
                spec=corpus.latest_spec(),
            )

    message = str(raised.value)
    assert f"path={str(foreign)!r}" in message
    assert f"uid={modeled_foreign_uid}" in message
    assert f"gid={modeled_foreign_gid}" in message
    assert f"journal_euid={modeled_euid}" in message
    assert not journal_root.exists() or not any(journal_root.iterdir())

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=journal_root,
        slice_id="owned_admin",
        scope=("a.py",),
        sequence=2,
        spec=corpus.latest_spec(),
    )
    assert journal.restore()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("core.ignoreCase", "true", "ignoreCase"),
        ("core.trustctime", "false", "trustctime"),
        ("core.checkStat", "minimal", "checkStat"),
        ("core.ignoreStat", "true", "ignoreStat"),
    ],
)
def test_git_status_configs_that_hide_writes_are_rejected(
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
):
    repo, corpus = _setup(tmp_path)
    _git(repo, "config", key, value)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    with pytest.raises(MutationJournalError, match=message):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=corpus.spec_dir / "runs" / "unsafe_stat" / "mutations",
            slice_id="unsafe_stat",
            scope=("a.py",),
            sequence=1,
            spec=corpus.latest_spec(),
        )


def test_submodule_journal_guards_ancestor_source_but_allows_scoped_edit(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=nested,
        journal_root=corpus.spec_dir / "runs" / "ancestor" / "mutations",
        slice_id="ancestor",
        scope=("nested.txt",),
        sequence=1,
    )
    with (repo / "a.py").open("rb") as pinned_ancestor:
        pinned_identity = (
            os.fstat(pinned_ancestor.fileno()).st_dev,
            os.fstat(pinned_ancestor.fileno()).st_ino,
        )
        (nested / "nested.txt").write_text("authorized child edit\n")
        journal.assert_repository_metadata_unchanged()
        (repo / "a.py").write_text("ancestor escape\n")
        with pytest.raises(MutationJournalError):
            journal.assert_repository_metadata_unchanged()
        assert not journal.restore()
        assert (repo / "a.py").read_text() == "ancestor escape\n"
        _git(repo, "checkout", "--", "a.py")
        current = (repo / "a.py").stat()
        assert (current.st_dev, current.st_ino) != pinned_identity
        assert journal.restore()
    assert (nested / "nested.txt").read_text() == "nested baseline\n"


@pytest.mark.parametrize("metadata_fault", ("mode", "xattr", "hardlink"))
def test_submodule_guard_rebind_rejects_nonbaseline_ancestor_metadata(
    tmp_path: Path,
    metadata_fault: str,
):
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    journal = WorkspaceMutationJournal.begin(
        cwd=nested,
        journal_root=(corpus.spec_dir / "runs" / "ancestor_metadata" / "mutations"),
        slice_id=f"ancestor_metadata_{metadata_fault}",
        scope=("nested.txt",),
        sequence=1,
    )
    (repo / "a.py").write_text("ancestor escape\n")
    assert not journal.restore()
    _git(repo, "checkout", "--", "a.py")
    if metadata_fault == "mode":
        (repo / "a.py").chmod(0o600)
    elif metadata_fault == "hardlink":
        os.link(repo / "a.py", repo / "a.py.alias")
    else:
        try:
            os.setxattr(repo / "a.py", "user.crucible_probe", b"changed")
        except OSError as exc:
            if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
                pytest.skip("filesystem does not support user xattrs")
            raise

    restored_identity = (repo / "a.py").stat()
    assert not journal.restore()
    current = (repo / "a.py").stat()
    assert (current.st_dev, current.st_ino) == (
        restored_identity.st_dev,
        restored_identity.st_ino,
    )


def test_submodule_journal_restores_ancestor_git_config(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=nested,
        journal_root=corpus.spec_dir / "runs" / "ancestor_config" / "mutations",
        slice_id="ancestor_config",
        scope=("nested.txt",),
        sequence=1,
    )
    _git(repo, "config", "crucible.ancestor-probe", "rogue")
    with pytest.raises(MutationJournalError):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    probe = subprocess.run(
        ["git", "config", "--get", "crucible.ancestor-probe"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 1


def test_cached_module_controls_are_exact_without_copying_objects(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    _add_clean_submodule(repo, tmp_path)
    _git(repo, "submodule", "deinit", "-f", "vendor/nested")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    config = _git_common_dir(repo) / "modules" / "vendor" / "nested" / "config"
    baseline = config.read_bytes()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "cached_module" / "mutations",
        slice_id="cached_module",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    config.write_bytes(baseline + b"\n[rogue]\n\tvalue = true\n")
    with pytest.raises(MutationJournalError, match="cached module control"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert config.read_bytes() == baseline


def test_git_admin_config_xattrs_are_restored_with_bytes(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    if not hasattr(os, "setxattr"):
        pytest.skip("xattrs unavailable")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_dir,
    )

    config = _git_dir(repo) / "config"
    os.setxattr(config, "user.crucible-test", b"baseline")
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "admin_xattr" / "mutations",
        slice_id="admin_xattr",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    os.setxattr(config, "user.crucible-test", b"changed")
    with pytest.raises(MutationJournalError, match="admin xattrs"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert os.getxattr(config, "user.crucible-test") == b"baseline"


def test_packed_object_permission_damage_retains_evidence(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    _git(repo, "gc", "--prune=now")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    pack = next((_git_common_dir(repo) / "objects" / "pack").glob("*.pack"))
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "packed_mode" / "mutations",
        slice_id="packed_mode",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    pack.chmod(0)
    with pytest.raises(MutationJournalError):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()
    assert pack.stat().st_mode & 0o7777 == 0


def test_pack_timestamp_only_touch_passes_full_fsck(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    _git(repo, "gc", "--prune=now")
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    pack = next((_git_common_dir(repo) / "objects" / "pack").glob("*.pack"))
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "pack_timestamp" / "mutations",
        slice_id="pack_timestamp",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    before = pack.stat()
    os.utime(
        pack,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
    )
    after = pack.stat()
    assert (after.st_dev, after.st_ino, after.st_nlink, after.st_size) == (
        before.st_dev,
        before.st_ino,
        before.st_nlink,
        before.st_size,
    )
    assert after.st_mtime_ns != before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns

    journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert pack.stat().st_mtime_ns == after.st_mtime_ns


def test_loose_object_timestamp_only_touch_passes_full_fsck(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    objects = _git_common_dir(repo) / "objects"
    loose = next(
        path
        for path in objects.glob("[0-9a-f][0-9a-f]/*")
        if len(path.parent.name + path.name) in {40, 64}
    )
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "loose_timestamp" / "mutations",
        slice_id="loose_timestamp",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    before = loose.stat()
    os.utime(
        loose,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
    )
    after = loose.stat()
    assert (after.st_dev, after.st_ino, after.st_nlink, after.st_size) == (
        before.st_dev,
        before.st_ino,
        before.st_nlink,
        before.st_size,
    )
    assert after.st_mtime_ns != before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns

    journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert loose.stat().st_mtime_ns == after.st_mtime_ns


def _setup_stale_shallow_commit_graph(
    tmp_path: Path,
) -> tuple[Path, Corpus, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    ancestor = _init_repo(repo)
    (repo / "a.py").write_text("# second commit\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "second")
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec(head))

    # Mirror DeepSWE history sanitization: the graph is written while the
    # ancestor exists, then the checkout becomes depth-one and that ancestor
    # is pruned.  Native fsck must ignore this unchanged acceleration cache
    # while checking the authoritative object store itself.
    _git(repo, "commit-graph", "write", "--reachable")
    git_dir = repo / ".git"
    (git_dir / "shallow").write_text(head + "\n", encoding="ascii")
    _git(repo, "reflog", "expire", "--expire=now", "--all")
    _git(repo, "repack", "-Ad")
    _git(repo, "prune", "--expire=now")
    graph = git_dir / "objects" / "info" / "commit-graph"
    assert graph.is_file()
    assert subprocess.run(
        ["git", "cat-file", "-e", ancestor],
        cwd=repo,
        capture_output=True,
        check=False,
    ).returncode != 0
    baseline_fsck = subprocess.run(
        ["git", "fsck", "--full", "--no-dangling", "HEAD"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    assert baseline_fsck.returncode != 0
    assert b"from object database for commit-graph" in baseline_fsck.stderr
    return repo, corpus, graph


def test_pack_timestamp_touch_ignores_stale_shallow_commit_graph(
    tmp_path: Path,
):
    repo, corpus, graph = _setup_stale_shallow_commit_graph(tmp_path)

    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    pack = next((_git_common_dir(repo) / "objects" / "pack").glob("*.pack"))
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "shallow_pack_timestamp" / "mutations"
        ),
        slice_id="shallow_pack_timestamp",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    before = pack.stat()
    os.utime(
        pack,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
    )

    journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert graph.is_file()


def test_stale_shallow_commit_graph_does_not_mask_object_drift_or_rollback(
    tmp_path: Path,
):
    repo, corpus, graph = _setup_stale_shallow_commit_graph(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    objects = _git_common_dir(repo) / "objects"
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "shallow_object_addition" / "mutations"
        ),
        slice_id="shallow_object_addition",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    created_oid = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=b"transaction-created object\n",
        check=True,
        capture_output=True,
    ).stdout.decode().strip()
    created = objects / created_oid[:2] / created_oid[2:]
    assert created.is_file()

    with pytest.raises(MutationJournalError, match="changed Git objects") as caught:
        journal.assert_repository_metadata_unchanged()
    assert "commit-graph" not in str(caught.value)
    assert journal.restore()
    assert not created.exists()
    assert graph.is_file()


def test_pack_timestamp_touch_ignores_preexisting_dangling_symbolic_ref(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    _git(repo, "gc", "--prune=now")
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _git_common_dir,
        _git_dir,
    )

    dangling = _git_dir(repo) / "refs" / "remotes" / "origin" / "HEAD"
    dangling.parent.mkdir(parents=True)
    dangling.write_text("ref: refs/remotes/origin/main\n", encoding="ascii")
    pack = next((_git_common_dir(repo) / "objects" / "pack").glob("*.pack"))
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "pack_timestamp_dangling_ref" / "mutations"
        ),
        slice_id="pack_timestamp_dangling_ref",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    before = pack.stat()
    os.utime(
        pack,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
    )

    journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert dangling.read_text(encoding="ascii") == (
        "ref: refs/remotes/origin/main\n"
    )


def test_dangling_symbolic_ref_target_creation_is_still_rejected(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_dir,
    )

    remote_dir = _git_dir(repo) / "refs" / "remotes" / "origin"
    remote_dir.mkdir(parents=True)
    dangling = remote_dir / "HEAD"
    dangling.write_text("ref: refs/remotes/origin/main\n", encoding="ascii")
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "dangling_ref_mutation" / "mutations"
        ),
        slice_id="dangling_ref_mutation",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (remote_dir / "main").write_text(
        _git(repo, "rev-parse", "HEAD").decode().strip() + "\n",
        encoding="ascii",
    )

    with pytest.raises(MutationJournalError, match="Git admin|Git loose refs"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not (remote_dir / "main").exists()
    assert dangling.read_text(encoding="ascii") == (
        "ref: refs/remotes/origin/main\n"
    )


def test_pack_timestamp_touch_composes_with_new_object_cleanup(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    _git(repo, "gc", "--prune=now")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    objects = _git_common_dir(repo) / "objects"
    pack = next((objects / "pack").glob("*.pack"))
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "pack_timestamp_addition" / "mutations"
        ),
        slice_id="pack_timestamp_addition",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    before = pack.stat()
    os.utime(
        pack,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
    )
    touched_mtime = pack.stat().st_mtime_ns
    created_oid = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=b"transaction-created object\n",
        check=True,
        capture_output=True,
    ).stdout.decode().strip()
    created = objects / created_oid[:2] / created_oid[2:]
    assert created.is_file()

    with pytest.raises(MutationJournalError, match="changed Git objects"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not created.exists()
    assert pack.stat().st_mtime_ns == touched_mtime


def test_same_size_pack_corruption_fails_timestamp_exception(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    _git(repo, "gc", "--prune=now")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
        _git_dir,
    )

    # A malformed auxiliary ref must not mask independent pack corruption.
    dangling = _git_dir(repo) / "refs" / "remotes" / "origin" / "HEAD"
    dangling.parent.mkdir(parents=True)
    dangling.write_text("ref: refs/remotes/origin/main\n", encoding="ascii")
    pack = next((_git_common_dir(repo) / "objects" / "pack").glob("*.pack"))
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "pack_corrupt" / "mutations",
        slice_id="pack_corrupt",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    before = pack.stat()
    baseline_mode = before.st_mode & 0o7777
    pack.chmod(baseline_mode | 0o200)
    with pack.open("r+b") as stream:
        stream.seek(min(64, before.st_size // 2))
        original = stream.read(1)
        assert original
        stream.seek(-1, os.SEEK_CUR)
        stream.write(bytes([original[0] ^ 0xFF]))
        stream.flush()
        os.fsync(stream.fileno())
    pack.chmod(baseline_mode)
    os.utime(pack, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = pack.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )

    with pytest.raises(MutationJournalError, match="failed full fsck"):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()


def test_unreachable_pack_corruption_fails_explicit_head_fsck(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
        _git_dir,
    )

    objects = _git_common_dir(repo) / "objects"
    pack_dir = objects / "pack"
    pack_dir.mkdir(exist_ok=True)
    orphan_oid = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=b"unreachable baseline object\n",
        check=True,
        capture_output=True,
    ).stdout.decode().strip()
    pack_hash = subprocess.run(
        ["git", "pack-objects", str(pack_dir / "pack")],
        cwd=repo,
        input=f"{orphan_oid}\n".encode("ascii"),
        check=True,
        capture_output=True,
    ).stdout.decode().strip()
    (objects / orphan_oid[:2] / orphan_oid[2:]).unlink()
    pack = pack_dir / f"pack-{pack_hash}.pack"

    dangling = _git_dir(repo) / "refs" / "remotes" / "origin" / "HEAD"
    dangling.parent.mkdir(parents=True)
    dangling.write_text("ref: refs/remotes/origin/main\n", encoding="ascii")
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "unreachable_pack_corrupt" / "mutations"
        ),
        slice_id="unreachable_pack_corrupt",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    before = pack.stat()
    baseline_mode = before.st_mode & 0o7777
    pack.chmod(baseline_mode | 0o200)
    with pack.open("r+b") as stream:
        stream.seek(min(20, before.st_size // 2))
        original = stream.read(1)
        assert original
        stream.seek(-1, os.SEEK_CUR)
        stream.write(bytes([original[0] ^ 0xFF]))
        stream.flush()
        os.fsync(stream.fileno())
    pack.chmod(baseline_mode)
    os.utime(pack, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(MutationJournalError, match="failed full fsck"):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()


def test_unindexed_canonical_pack_cannot_use_timestamp_exception(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    _git(repo, "gc", "--prune=now")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    pack_dir = _git_common_dir(repo) / "objects" / "pack"
    fake_oid = "1" * 40
    fake_pack = pack_dir / f"pack-{fake_oid}.pack"
    fake_index = pack_dir / f"pack-{fake_oid}.idx"
    fake_pack.write_bytes(b"unindexed fake pack payload\n")
    fake_pack.chmod(0o444)
    assert not fake_index.exists()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "unindexed_pack_corrupt" / "mutations"
        ),
        slice_id="unindexed_pack_corrupt",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    before = fake_pack.stat()
    fake_pack.chmod(0o644)
    with fake_pack.open("r+b") as stream:
        original = stream.read(1)
        assert original
        stream.seek(0)
        stream.write(bytes([original[0] ^ 0x01]))
        stream.flush()
        os.fsync(stream.fileno())
    fake_pack.chmod(0o444)
    os.utime(fake_pack, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = fake_pack.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )

    with pytest.raises(MutationJournalError, match="changed Git objects"):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()


def test_same_size_info_packs_corruption_is_not_timestamp_exception(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    _git(repo, "gc", "--prune=now")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    info_packs = _git_common_dir(repo) / "objects" / "info" / "packs"
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "info_packs_corrupt" / "mutations",
        slice_id="info_packs_corrupt",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    before = info_packs.stat()
    with info_packs.open("r+b") as stream:
        original = stream.read(1)
        assert original
        stream.seek(0)
        stream.write(bytes([original[0] ^ 0x01]))
        stream.flush()
        os.fsync(stream.fileno())
    os.utime(info_packs, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = info_packs.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )

    with pytest.raises(MutationJournalError, match="changed Git objects"):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()


def test_pack_replacement_remains_irreplaceable(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    _git(repo, "gc", "--prune=now")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    pack_dir = _git_common_dir(repo) / "objects" / "pack"
    before = {
        path.name: path.stat().st_ino for path in pack_dir.glob("*.pack")
    }
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "pack_replace" / "mutations",
        slice_id="pack_replace",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    _git(repo, "repack", "-Adf")
    after = {
        path.name: path.stat().st_ino for path in pack_dir.glob("*.pack")
    }
    assert after != before

    with pytest.raises(MutationJournalError, match="changed Git objects"):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()


def test_pack_directory_symlink_remains_irreplaceable(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    _git(repo, "gc", "--prune=now")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    pack_dir = _git_common_dir(repo) / "objects" / "pack"
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "pack_alias" / "mutations",
        slice_id="pack_alias",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    external = tmp_path / "external-pack"
    pack_dir.rename(external)
    pack_dir.symlink_to(external, target_is_directory=True)

    with pytest.raises(MutationJournalError, match="changed Git objects"):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()


def test_packed_object_file_xattr_damage_retains_evidence(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    if not hasattr(os, "setxattr"):
        pytest.skip("xattrs unavailable")
    _git(repo, "gc", "--prune=now")
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
    )

    pack = next((_git_common_dir(repo) / "objects" / "pack").glob("*.pack"))
    baseline_mode = pack.stat().st_mode & 0o7777
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "packed_xattr" / "mutations",
        slice_id="packed_xattr",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    pack.chmod(baseline_mode | 0o200)
    os.setxattr(pack, "user.crucible-test", b"rogue")
    pack.chmod(baseline_mode)

    with pytest.raises(MutationJournalError, match="changed Git objects"):
        journal.assert_repository_metadata_unchanged()
    assert not journal.restore()
    assert journal.journal_dir.is_dir()
    assert "user.crucible-test" in os.listxattr(pack)
    assert pack.stat().st_mode & 0o7777 == baseline_mode


def test_missing_registered_peer_admin_is_guarded_exactly(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    linked = tmp_path / "missing-linked"
    available = tmp_path / "available-linked"
    _git(repo, "worktree", "add", "-q", "--detach", str(linked), "HEAD")
    _git(repo, "worktree", "add", "-q", "--detach", str(available), "HEAD")
    shutil.rmtree(linked)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _git_common_dir,
        _registered_worktree_admin_roots,
    )

    admin_name = _registered_worktree_admin_roots(repo)[linked.resolve()]
    admin_root = _git_common_dir(repo) / "worktrees" / admin_name
    # Preserve an alias spelling in Git's stale gitdir record. Production
    # has /home/... records whose existing ancestor resolves to /data/...;
    # lookup must compare canonical paths while retaining the exact admin.
    alias_parent = tmp_path / "worktree-alias"
    alias_parent.symlink_to(tmp_path, target_is_directory=True)
    (admin_root / "gitdir").write_text(
        str(alias_parent / linked.name / ".git") + "\n"
    )
    assert _registered_worktree_admin_roots(repo)[linked.resolve()] == admin_name
    head = admin_root / "HEAD"
    baseline = head.read_bytes()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "missing_peer" / "mutations",
        slice_id="missing_peer",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    head.write_text("0" * 40 + "\n")
    with pytest.raises(MutationJournalError, match="missing-worktree admin"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert head.read_bytes() == baseline


def test_guard_only_peer_supports_sparse_index_but_direct_mutation_rejects_it(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    (repo / "nested").mkdir()
    (repo / "nested" / ".gitignore").write_text("generated/\n")
    _git(repo, "add", "nested/.gitignore")
    _git(repo, "commit", "-qm", "tracked ignore policy")
    linked = tmp_path / "sparse-linked"
    _git(repo, "worktree", "add", "-q", "--detach", str(linked), "HEAD")
    _git(linked, "update-index", "--skip-worktree", "b.py")
    _git(linked, "update-index", "--skip-worktree", "nested/.gitignore")
    (linked / "nested" / ".gitignore").unlink()
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "sparse_peer" / "mutations",
        slice_id="sparse_peer",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    journal.assert_repository_metadata_unchanged()
    journal.commit()

    with pytest.raises(MutationJournalError, match="skip-worktree"):
        WorkspaceMutationJournal.begin(
            cwd=linked,
            journal_root=corpus.spec_dir / "runs" / "sparse_direct" / "mutations",
            slice_id="sparse_direct",
            scope=("a.py",),
            sequence=2,
            spec=corpus.latest_spec(),
        )


def test_nested_peer_journal_controls_do_not_force_fsck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, corpus = _setup(tmp_path)
    nested = _add_clean_submodule(repo, tmp_path)
    linked = tmp_path / "nested-linked"
    _git(nested, "worktree", "add", "-q", "--detach", str(linked), "HEAD")
    from crucible.workflows import _mutation_journal as mutation_journal
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    nested_common = Path(
        _git(
            nested,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ).decode().strip()
    )
    journal = WorkspaceMutationJournal.begin(
        cwd=nested,
        journal_root=(
            corpus.spec_dir / "runs" / "nested_peer_controls" / "mutations"
        ),
        slice_id="nested_peer_controls",
        scope=(),
        sequence=1,
        _nested_gitlink_guard=True,
        _external_worktree_guard=True,
        _suppress_superproject_guard=True,
    )
    expected_root = next(
        record
        for record in journal._manifest["git_admin_topology"]
        if Path(str(record["root"])) == nested_common
    )
    journal_owned = {
        "crucible-mutation-journals",
        "crucible-workspace.lock",
    }
    assert journal_owned.isdisjoint(expected_root["entries"])
    assert journal_owned.issubset(
        path.name for path in nested_common.iterdir()
    )
    assert journal._initial_fsck_reasons(nested) == set()

    original_git = mutation_journal._git
    fsck_calls: list[tuple[str, ...]] = []

    def recording_git(
        cwd: Path,
        *args: str,
        check: bool = True,
        timeout_s: float | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> bytes:
        if args and args[0] == "fsck":
            fsck_calls.append(args)
        return original_git(
            cwd,
            *args,
            check=check,
            timeout_s=timeout_s,
            pass_fds=pass_fds,
        )

    monkeypatch.setattr(mutation_journal, "_git", recording_git)
    assert journal.restore()
    assert fsck_calls == []
    # The per-common-dir coordination substrate is intentionally persistent;
    # terminal cleanup removes every marker, while retaining the exact empty
    # registry inode and single-link lock for later writers.
    assert journal_owned.issubset(
        path.name for path in nested_common.iterdir()
    )
    registry = nested_common / "crucible-mutation-journals"
    assert registry.is_dir()
    assert list(registry.iterdir()) == []
    lock = nested_common / "crucible-workspace.lock"
    assert lock.is_file()
    assert lock.stat().st_nlink == 1


def test_parent_registry_and_lock_with_linked_peer_do_not_force_fsck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, corpus = _setup(tmp_path)
    linked = tmp_path / "linked-with-parent-registry"
    _git(repo, "worktree", "add", "-q", "--detach", str(linked), "HEAD")
    from crucible.workflows import _mutation_journal as mutation_journal
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    common = Path(
        _git(
            repo,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ).decode().strip()
    )
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "parent_peer_controls" / "mutations"
        ),
        slice_id="parent_peer_controls",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    expected_root = next(
        record
        for record in journal._manifest["git_admin_topology"]
        if Path(str(record["root"])) == common
    )
    parent_controls = {
        "crucible-mutation-journals",
        "crucible-workspace.lock",
    }
    assert parent_controls.issubset(expected_root["entries"])
    children = journal._nested_journals()
    assert len(children) == 1
    assert {
        f"{journal.transaction_id}.json",
        f"{children[0].transaction_id}.json",
    }.issubset(
        path.name
        for path in (common / "crucible-mutation-journals").iterdir()
    )
    assert journal._initial_fsck_reasons(repo) == set()
    rogue_control = common / "rogue-control"
    rogue_control.write_text("not journal-owned\n")
    assert journal._initial_fsck_reasons(repo) == {"controls"}
    rogue_control.unlink()
    assert journal._initial_fsck_reasons(repo) == set()

    original_git = mutation_journal._git
    fsck_calls: list[tuple[str, ...]] = []

    def recording_git(
        cwd: Path,
        *args: str,
        check: bool = True,
        timeout_s: float | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> bytes:
        if args and args[0] == "fsck":
            fsck_calls.append(args)
        return original_git(
            cwd,
            *args,
            check=check,
            timeout_s=timeout_s,
            pass_fds=pass_fds,
        )

    monkeypatch.setattr(mutation_journal, "_git", recording_git)
    assert journal.restore()
    assert fsck_calls == []


def test_discovered_parent_cleans_descendants_under_bound_journal_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, corpus = _setup(tmp_path)
    linked = tmp_path / "bound-cleanup-linked"
    _git(repo, "worktree", "add", "-q", "--detach", str(linked), "HEAD")
    from crucible.workflows import _mutation_journal

    journal = _mutation_journal.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "bound_descendant_cleanup" / "mutations"
        ),
        slice_id="bound_descendant_cleanup",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    family = (journal, *journal._nested_journals())
    assert len(family) == 2
    for member in family:
        _mutation_journal._unregister_live(member.transaction_id)
    discovered = _mutation_journal.discover_mutation_journals(
        corpus,
        workspace=repo,
    )
    recovered_parent = next(
        member
        for member in discovered
        if member.parent_transaction_id is None
    )
    observed_roots: list[str] = []
    original = (
        _mutation_journal.WorkspaceMutationJournal
        ._nested_journals_children_first
    )

    def record_bound_root(current):
        if current.transaction_id == recovered_parent.transaction_id:
            observed_roots.append(os.fspath(current.journal_dir))
        return original(current)

    monkeypatch.setattr(
        _mutation_journal.WorkspaceMutationJournal,
        "_nested_journals_children_first",
        record_bound_root,
    )
    missing = object()
    previous = getattr(
        _mutation_journal._LOCK_STATE,
        "writer_epoch_recovery_members",
        missing,
    )
    _mutation_journal._LOCK_STATE.writer_epoch_recovery_members = {
        member.transaction_id: member for member in discovered
    }
    try:
        assert recovered_parent.restore()
    finally:
        if previous is missing:
            del _mutation_journal._LOCK_STATE.writer_epoch_recovery_members
        else:
            _mutation_journal._LOCK_STATE.writer_epoch_recovery_members = previous

    assert observed_roots
    assert all(root.startswith("/proc/self/fd/") for root in observed_roots)
    assert not journal.journal_dir.exists()


def test_child_cleanup_crash_before_parent_ack_resumes_from_terminal_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, corpus = _setup(tmp_path)
    linked = tmp_path / "cleanup-ack-crash-linked"
    _git(repo, "worktree", "add", "-q", "--detach", str(linked), "HEAD")
    from crucible.workflows import _mutation_journal

    journal = _mutation_journal.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "cleanup_ack_crash" / "mutations"
        ),
        slice_id="cleanup_ack_crash",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    children = journal._nested_journals()
    assert len(children) == 1
    child_dir = children[0].journal_dir
    original = (
        _mutation_journal.WorkspaceMutationJournal
        ._acknowledge_resolved_descendants
    )
    crashed = False

    def crash_before_ack(current, transaction_ids):
        nonlocal crashed
        resolved = tuple(transaction_ids)
        if current.transaction_id == journal.transaction_id and not crashed:
            crashed = True
            assert not child_dir.exists()
            raise RuntimeError("simulated crash before parent family ack")
        return original(current, resolved)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            _mutation_journal.WorkspaceMutationJournal,
            "_acknowledge_resolved_descendants",
            crash_before_ack,
        )
        assert not journal.restore()

    assert crashed
    assert journal.state == "rolled_back"
    assert journal.journal_dir.is_dir()
    assert not child_dir.exists()
    _mutation_journal._unregister_live(journal.transaction_id)
    report = _mutation_journal.recover_mutation_journals(corpus, repo)

    assert report.failures == ()
    assert not journal.journal_dir.exists()
    assert list(_mutation_journal._registry_directory(repo).iterdir()) == []


def test_registry_family_shrinks_after_child_marker_is_removed(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    parent = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "family_parent" / "mutations",
        slice_id="family_parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "family_child" / "mutations",
        slice_id="family_child",
        scope=("b.py",),
        sequence=2,
        parent_transaction_id=parent.transaction_id,
    )
    assert child.transaction_id in parent._manifest["registry_transaction_family"]
    child.commit()
    persisted = json.loads((parent.journal_dir / "manifest.json").read_text())
    assert persisted["registry_transaction_family"] == [parent.transaction_id]
    assert persisted["failed_family_members"] == []
    parent.commit()


def test_restore_failure_retries_interrupted_family_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, corpus = _setup(tmp_path)
    from crucible.workflows import _mutation_journal as mutation_journal
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal

    parent = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "failure_retry_parent" / "mutations"
        ),
        slice_id="failure_retry_parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "failure_retry_child" / "mutations"
        ),
        slice_id="failure_retry_child",
        scope=("b.py",),
        sequence=2,
        parent_transaction_id=parent.transaction_id,
    )
    real_propagate = mutation_journal._mark_live_family_failed
    attempts = 0

    def interrupted_once(transaction_id: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("modeled interruption before family propagation")
        real_propagate(transaction_id)

    monkeypatch.setattr(
        mutation_journal,
        "_mark_live_family_failed",
        interrupted_once,
    )
    child._record_restore_failure()
    assert attempts == 1
    child_persisted = json.loads(
        (child.journal_dir / "manifest.json").read_text()
    )
    assert child.transaction_id in child_persisted["failed_family_members"]
    parent_persisted = json.loads(
        (parent.journal_dir / "manifest.json").read_text()
    )
    assert child.transaction_id not in parent_persisted["failed_family_members"]

    child._record_restore_failure()

    assert attempts == 2
    parent_persisted = json.loads(
        (parent.journal_dir / "manifest.json").read_text()
    )
    assert child.transaction_id in parent_persisted["failed_family_members"]
    monkeypatch.setattr(
        mutation_journal,
        "_mark_live_family_failed",
        real_propagate,
    )
    assert child.restore()
    assert parent.restore()


def test_nested_journal_rejects_asserted_non_live_parent(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
    )

    with pytest.raises(MutationJournalError, match="not live in this process"):
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=(
                corpus.spec_dir / "runs" / "missing_parent" / "mutations"
            ),
            slice_id="missing_parent_child",
            scope=("a.py",),
            sequence=1,
            parent_transaction_id="not-a-live-parent",
        )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_fork_child_cannot_reuse_live_parent_transaction(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        validate_live_parent_transaction,
    )

    parent = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "fork_parent" / "mutations",
        slice_id="fork_parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# parent mutation\n")
    read_fd, write_fd = os.pipe()
    try:
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            exit_code = 1
            message = b"parent transaction was unexpectedly accepted"
            try:
                validate_live_parent_transaction(
                    parent.transaction_id,
                    cwd=repo,
                )
            except MutationJournalError as exc:
                message = str(exc).encode()
                exit_code = (
                    0 if "not live in this process" in str(exc) else 2
                )
            except BaseException as exc:
                message = f"unexpected error: {exc!r}".encode()
                exit_code = 3
            try:
                os.write(write_fd, message)
            finally:
                os.close(write_fd)
                os._exit(exit_code)

        os.close(write_fd)
        write_fd = -1
        _, status = os.waitpid(child_pid, 0)
        child_message = os.read(read_fd, 4096).decode()
        assert os.WIFEXITED(status), child_message
        assert os.WEXITSTATUS(status) == 0, child_message
        assert "not live in this process" in child_message
        assert parent.active
        assert (repo / "a.py").read_text() == "# parent mutation\n"
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        if parent.active:
            assert parent.restore()
    assert (repo / "a.py").read_text() == "# baseline a.py\n"


def test_child_workflow_parent_must_own_same_checkout(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    other = tmp_path / "other-repo"
    other.mkdir()
    _init_repo(other)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        validate_live_parent_transaction,
    )

    parent = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "cwd_parent" / "mutations",
        slice_id="cwd_parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    try:
        with pytest.raises(MutationJournalError, match="different checkout"):
            validate_live_parent_transaction(
                parent.transaction_id,
                cwd=other,
            )
    finally:
        assert parent.restore()


def test_failed_external_child_keeps_ancestor_family_armed(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _git_common_dir,
        _registry_directory,
    )

    parent = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "failed_parent" / "mutations",
        slice_id="failed_parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    payload = b"transaction-family-object\n"
    object_id = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=payload,
        check=True,
        capture_output=True,
    ).stdout.decode().strip()
    child = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "failed_child" / "mutations",
        slice_id="failed_child",
        scope=("b.py",),
        sequence=2,
        parent_transaction_id=parent.transaction_id,
    )
    object_path = _git_common_dir(repo) / "objects" / object_id[:2] / object_id[2:]
    object_path.unlink()
    assert not child.restore()
    assert not parent.restore()
    assert len(list(_registry_directory(repo).glob("*.json"))) == 2

    # Re-creating the same object bytes cannot recreate its exact inode/time
    # baseline. Both durable markers intentionally remain for inspection.


def test_untrusted_manifest_cannot_forge_registry_family_membership(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        WorkspaceMutationJournal,
        _registry_directory,
    )

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "registry_forgery" / "mutations",
        slice_id="registry_forgery",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    fake_id = "f" * 32
    fake_dir = tmp_path / "fake-journal"
    fake_dir.mkdir()
    (fake_dir / "manifest.json").write_text(json.dumps({
        "schema": "crucible.workspace-mutation-journal.v2",
        "transaction_id": fake_id,
        "parent_transaction_id": journal.transaction_id,
    }))
    rogue = _registry_directory(repo) / f"{fake_id}.json"
    rogue.write_text(json.dumps({
        "transaction_id": fake_id,
        "journal_dir": str(fake_dir),
        "cwd": str(repo),
        "spec_dir": str(corpus.spec_dir),
    }))
    with pytest.raises(MutationJournalError, match="registry topology"):
        journal.assert_repository_metadata_unchanged()
    assert journal.restore()
    assert not rogue.exists()


def test_nested_child_recovery_preserves_live_parent_transaction(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt

    parent = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "parent" / "mutations",
        slice_id="parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# child seed\n")
    parent.prepare_lifecycle_acceptances(
        kind="landed",
        reasons=("breach loop proceed",),
    )

    report = reap_orphaned_dirt(corpus, repo)
    assert report.mutation_journals_recovered == ()
    assert report.mutation_journals_preserved == ()
    assert (repo / "a.py").read_text() == "# child seed\n"
    assert parent.restore()


def test_lifecycle_attach_settles_prepared_mutation_after_process_death(
    tmp_path: Path,
):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "lifecycle_crash" / "mutations"
        ),
        slice_id="breach",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# accepted breach repair\n")
    reason = "breach loop proceed"
    journal.prepare_lifecycle_acceptance(kind="landed", reason=reason)
    corpus.commit(lifecycle_events=(LifecycleEvent(
        kind="landed",
        reason=reason,
        affected_versions=(1,),
        execution=exec_ctx_now("test"),
    ),))
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)
    assert report.mutation_journals_preserved == ("breach",)
    assert any(
        "lifecycle-only prepared_keep" in failure
        and "journal retained" in failure
        for failure in report.failures
    )
    assert (repo / "a.py").read_text() == "# accepted breach repair\n"


def test_ambiguous_lifecycle_attach_retains_prepared_mutation(
    tmp_path: Path,
):
    """Multiple matching lifecycle refs are ambiguity, not rollback proof."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "lifecycle_collision" / "mutations",
        slice_id="breach",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# accepted breach repair\n")
    reason = "breach loop proceed"
    journal.prepare_lifecycle_acceptance(kind="landed", reason=reason)
    for _ in range(2):
        corpus.commit(
            lifecycle_event_factory=lambda version: LifecycleEvent(
                kind="landed",
                reason=reason,
                affected_versions=(version,),
                execution=exec_ctx_now("test"),
            ),
        )
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.mutation_journals_preserved == ()
    assert report.mutation_journals_recovered == ()
    assert any(
        "prepared lifecycle matches multiple attached events" in failure
        for failure in report.failures
    )
    assert journal.journal_dir.exists()
    assert (repo / "a.py").read_text() == "# accepted breach repair\n"


def test_missing_lifecycle_lineage_retains_prepared_mutation(
    tmp_path: Path,
):
    """Corrupt lifecycle lineage cannot authorize destructive recovery."""

    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "missing_lineage" / "mutations",
        slice_id="breach",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# accepted breach repair\n")
    reason = "breach loop proceed"
    journal.prepare_lifecycle_acceptance(kind="landed", reason=reason)
    journal._manifest["lineage"] = None
    journal._write()
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason=reason,
            affected_versions=(version,),
            execution=exec_ctx_now("test"),
        ),
    )
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.mutation_journals_preserved == ()
    assert report.mutation_journals_recovered == ()
    assert any(
        "prepared lifecycle keep has no baseline Spec lineage" in failure
        for failure in report.failures
    )
    assert journal.journal_dir.exists()
    assert (repo / "a.py").read_text() == "# accepted breach repair\n"


def test_foreign_run_lifecycle_cannot_settle_prepared_mutation(tmp_path: Path):
    repo, corpus = _setup(tmp_path)
    from crucible.workflows._mutation_journal import WorkspaceMutationJournal
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "foreign_run" / "mutations",
        slice_id="foreign_run",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.py").write_text("# ours but not attached\n")
    reason = "tier t0: complete"
    journal.prepare_lifecycle_acceptance(
        kind="landed",
        reason=reason,
        run_id="our-run",
    )
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason=reason,
            affected_versions=(version,),
            execution=exec_ctx_now("test", run_id="foreign-run"),
        ),
    )
    _simulate_process_death(journal)

    report = reap_orphaned_dirt(corpus, repo)
    assert report.mutation_journals_preserved == ()
    assert report.mutation_journals_recovered == ("foreign_run",)
    assert (repo / "a.py").read_text() == "# baseline a.py\n"
