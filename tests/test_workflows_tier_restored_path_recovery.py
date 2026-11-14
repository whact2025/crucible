# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fresh scoped retries own source bytes, not permanent restore observations."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from assay import Implementation
from assay.corpus.store import Corpus
from crucible.workflows import _mutation_journal as journal_module
from tests.test_workflows_tier_diagnostic_recovery import _SettledMaxTurnsBackend
from tests.test_workflows_tier_empty_diff_retry import _COMPLETE_EXECUTION, _run
from tests.test_workflows_tier_mutation_transaction import _setup


@pytest.mark.parametrize("inherited_dirty", [False, True])
@pytest.mark.parametrize("atomic_replacement", [False, True])
def test_post_rollback_recovery_can_replace_the_same_tracked_path(
    tmp_path, monkeypatch, inherited_dirty, atomic_replacement,
):
    class RecoveryBackend(_SettledMaxTurnsBackend):
        def invoke(self, invocation):
            result = super().invoke(invocation)
            if len(self.invocations) == 2 and atomic_replacement:
                replacement = self.repo / "replacement.py"
                replacement.write_text("value = recovered\n")
                replacement.replace(self.repo / "target.py")
            return result

    backend = RecoveryBackend(
        mutate=True,
        completeness=replace(_COMPLETE_EXECUTION, terminal_complete=False),
    )
    rc, summary, repo = _run(
        tmp_path, monkeypatch, backend, inherited_dirty=inherited_dirty,
        budget_usd=12, max_turns=90,
    )
    assert rc == 0
    assert len(backend.invocations) == 2
    assert backend.invocations[1].max_turns == 24
    assert repo.joinpath("target.py").read_text() == "value = recovered\n"
    assert summary["costs"]["total_usd"] == 0.5
    assert summary["costs"]["accounting_debt"] is False
    recovery = summary["phase_1_slices"][0]["incomplete_terminal_recovery"]
    assert recovery["restored_to_pre_call_snapshot"] is True
    assert recovery["retry_attempted"] is True
    corpus = Corpus(tmp_path / "corpus")
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1], as_type=Implementation,
    )
    assert implementation.classification.startswith("success")
    from crucible.workflows.tier import attest_mutation_journal_settlement

    settlement = attest_mutation_journal_settlement(
        corpus=corpus, cwd=repo, tier_id="t0", operation_id="same-path-recovery-test",
    )
    assert settlement["settled"] is True
    assert settlement["registry_clear"] is True
    assert settlement["remaining_journal_manifests"] == []


def _restored_parent_and_child(tmp_path, *, child_scope=("a.py",)):
    repo, corpus = _setup(tmp_path)
    root = corpus.spec_dir / "runs" / "restored-path" / "mutations"
    parent = journal_module.WorkspaceMutationJournal.begin(
        cwd=repo, journal_root=root, slice_id="parent", scope=("a.py",),
        sequence=1, spec=corpus.latest_spec(),
    )
    repo.joinpath("a.py").write_text("# partial first attempt\n")
    parent.restore_paths_from_snapshot(("a.py",))
    child = journal_module.WorkspaceMutationJournal.begin(
        cwd=repo, journal_root=root, slice_id="child", scope=child_scope,
        sequence=2, spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    return repo, parent, child


def test_child_handoff_retires_only_shared_scope_observations(tmp_path):
    repo, parent, child = _restored_parent_and_child(tmp_path)
    original_manifest = parent.journal_dir.joinpath("manifest.json").read_bytes()
    parent.admit_restored_path_child(child)
    assert "a.py" not in parent._restored_tracked_metadata_observations
    assert "b.py" in parent._restored_tracked_metadata_observations
    assert parent.journal_dir.joinpath("manifest.json").read_bytes() == original_manifest
    repo.joinpath("a.py").write_text("# permitted retry edit\n")
    parent.assert_repository_metadata_unchanged()
    child.assert_repository_metadata_unchanged()
    assert child.restore()
    assert parent.restore()


def test_previous_same_path_failure_is_reproduced_without_the_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(
        journal_module.WorkspaceMutationJournal,
        "admit_restored_path_child", lambda *args, **kwargs: None,
    )
    backend = _SettledMaxTurnsBackend(mutate=True, completeness=_COMPLETE_EXECUTION)
    with pytest.raises(journal_module.MutationJournalError, match="tracked path"):
        _run(tmp_path, monkeypatch, backend, budget_usd=12, max_turns=90)
    assert len(backend.invocations) == 2
    assert tmp_path.joinpath("repo", "target.py").read_text() == "value = 1\n"


def test_child_scope_does_not_expand_ancestor_write_authority(tmp_path):
    repo, parent, child = _restored_parent_and_child(tmp_path, child_scope=("b.py",))
    retained = dict(parent._restored_tracked_metadata_observations)
    parent.admit_restored_path_child(child)
    assert parent._restored_tracked_metadata_observations == retained
    repo.joinpath("b.py").write_text("# cannot expand the ancestor scope\n")
    with pytest.raises(journal_module.MutationJournalError, match="tracked path"):
        parent.assert_repository_metadata_unchanged()
    assert child.restore()
    assert parent.restore()


def test_armed_writer_cannot_retire_restored_path_observations(tmp_path):
    from crucible.workflows._writer_epoch import quiesce_writer_epoch_family
    from tests.test_writer_epoch import _arm, _scheduler_receipt

    _repo, parent, child = _restored_parent_and_child(tmp_path)
    retained = dict(parent._restored_tracked_metadata_observations)
    lease = _arm((parent, child))
    with pytest.raises(journal_module.MutationJournalError, match="writer epoch remains armed"):
        parent.admit_restored_path_child(child)
    assert parent._restored_tracked_metadata_observations == retained
    quiesce_writer_epoch_family(lease, _scheduler_receipt(lease))
    assert child.restore()
    assert parent.restore()


def test_unrelated_child_cannot_retire_restored_path_observations(tmp_path):
    _repo, parent, child = _restored_parent_and_child(tmp_path)
    retained = dict(parent._restored_tracked_metadata_observations)
    with pytest.raises(journal_module.MutationJournalError, match="exact live child"):
        parent.admit_restored_path_child(parent)
    assert parent._restored_tracked_metadata_observations == retained
    assert child.restore()
    assert parent.restore()


def test_nested_recovery_retires_only_common_ancestor_scope(tmp_path):
    repo, parent, child = _restored_parent_and_child(tmp_path)
    grandchild = journal_module.WorkspaceMutationJournal.begin(
        cwd=repo, journal_root=child.journal_dir.parent,
        slice_id="grandchild", scope=("a.py",), sequence=3,
        parent_transaction_id=child.transaction_id,
    )
    child.admit_restored_path_child(grandchild)
    assert "a.py" not in parent._restored_tracked_metadata_observations
    assert "b.py" in parent._restored_tracked_metadata_observations
    repo.joinpath("a.py").write_text("# nested permitted retry\n")
    parent.assert_repository_metadata_unchanged()
    child.assert_repository_metadata_unchanged()
    grandchild.assert_repository_metadata_unchanged()
    assert grandchild.restore()
    assert child.restore()
    assert parent.restore()


def test_previously_armed_child_is_not_a_fresh_continuation(tmp_path):
    from crucible.workflows._writer_epoch import quiesce_writer_epoch_family
    from tests.test_writer_epoch import _arm, _scheduler_receipt

    _repo, parent, child = _restored_parent_and_child(tmp_path)
    retained = dict(parent._restored_tracked_metadata_observations)
    lease = _arm((parent, child))
    quiesce_writer_epoch_family(lease, _scheduler_receipt(lease))
    with pytest.raises(journal_module.MutationJournalError, match="pristine child checkpoint"):
        parent.admit_restored_path_child(child)
    assert parent._restored_tracked_metadata_observations == retained
    assert child.restore()
    assert parent.restore()


def test_child_baseline_mismatch_does_not_retire_observations(tmp_path, monkeypatch):
    _repo, parent, child = _restored_parent_and_child(tmp_path)
    retained = dict(parent._restored_tracked_metadata_observations)
    read = child.read_writer_epoch_record

    def mismatched_baseline():
        raw = read()
        child._manifest["tracked_path_metadata"]["a.py"]["size"] += 1
        return raw

    with monkeypatch.context() as fault:
        fault.setattr(child, "read_writer_epoch_record", mismatched_baseline)
        with pytest.raises(journal_module.MutationJournalError, match="child baseline differs"):
            parent.admit_restored_path_child(child)
    assert parent._restored_tracked_metadata_observations == retained
    assert child.restore()
    assert parent.restore()


def test_late_ancestor_rejection_keeps_every_observation(tmp_path, monkeypatch):
    repo, parent, child = _restored_parent_and_child(tmp_path)
    repo.joinpath("a.py").write_text("# nested failed attempt\n")
    child.restore_paths_from_snapshot(("a.py",))
    grandchild = journal_module.WorkspaceMutationJournal.begin(
        cwd=repo, journal_root=child.journal_dir.parent,
        slice_id="grandchild", scope=("a.py",), sequence=3,
        parent_transaction_id=child.transaction_id,
    )
    parent_retained = dict(parent._restored_tracked_metadata_observations)
    child_retained = dict(child._restored_tracked_metadata_observations)
    checked = []
    check_child = child.assert_repository_metadata_unchanged

    def child_passes():
        check_child()
        checked.append("child")

    def parent_fails():
        checked.append("parent")
        raise journal_module.MutationJournalError("late ancestor drift")

    with monkeypatch.context() as fault:
        fault.setattr(child, "assert_repository_metadata_unchanged", child_passes)
        fault.setattr(parent, "assert_repository_metadata_unchanged", parent_fails)
        with pytest.raises(journal_module.MutationJournalError, match="late ancestor drift"):
            child.admit_restored_path_child(grandchild)
    assert checked == ["child", "parent"]
    assert parent._restored_tracked_metadata_observations == parent_retained
    assert child._restored_tracked_metadata_observations == child_retained
    assert grandchild.restore()
    assert child.restore()
    assert parent.restore()


@pytest.mark.parametrize("drift", ["content", "inode", "mode", "hardlink", "xattr"])
def test_parent_rejects_restored_path_drift_before_child_admission(tmp_path, drift):
    repo, parent, child = _restored_parent_and_child(tmp_path)
    path = repo / "a.py"
    link = tmp_path / "outside-link"
    if drift == "content":
        path.write_text("# outsider replacement\n")
    elif drift == "inode":
        replacement = repo / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif drift == "mode":
        path.chmod(0o600)
    elif drift == "hardlink":
        os.link(path, link)
    else:
        os.setxattr(path, "user.crucible_test", b"outsider")
    retained = dict(parent._restored_tracked_metadata_observations)
    with pytest.raises(journal_module.MutationJournalError, match="tracked path"):
        parent.admit_restored_path_child(child)
    assert parent._restored_tracked_metadata_observations == retained
    if link.exists():
        link.unlink()
    assert child.restore()
    assert parent.restore()


@pytest.mark.parametrize("target", ["a.py", "b.py"])
@pytest.mark.parametrize("drift", ["mode", "hardlink", "xattr"])
def test_child_admission_does_not_waive_metadata_rules(tmp_path, target, drift):
    repo, parent, child = _restored_parent_and_child(tmp_path)
    parent.admit_restored_path_child(child)
    path = repo / target
    link = tmp_path / "outside-link"
    if drift == "mode":
        path.chmod(0o600)
    elif drift == "hardlink":
        os.link(path, link)
    else:
        os.setxattr(path, "user.crucible_test", b"outsider")
    with pytest.raises(journal_module.MutationJournalError, match="tracked path"):
        parent.assert_repository_metadata_unchanged()
    if link.exists():
        link.unlink()
    assert child.restore()
    assert parent.restore()


def test_restored_sibling_content_is_not_child_authority(tmp_path):
    repo, parent, child = _restored_parent_and_child(tmp_path)
    parent.admit_restored_path_child(child)
    repo.joinpath("b.py").write_text("# outsider sibling edit\n")
    with pytest.raises(journal_module.MutationJournalError, match="tracked path"):
        parent.assert_repository_metadata_unchanged()
    assert child.restore()
    assert parent.restore()
