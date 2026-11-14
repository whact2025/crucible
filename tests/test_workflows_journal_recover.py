# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Exact external-Git recovery for retained mutation journals."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from assay import ProjectContext, Spec
from assay.corpus.store import Corpus
from assay.primitives.implement import capture_workspace_delta

from crucible.workflows._journal_recovery import (
    AcceptedAncestorSubmodule,
    ExternalRecoveryRequest,
    inspect_external_recovery,
    reap_external_recovery_tombstones,
)
from crucible.workflows._mutation_journal import (
    MutationJournalError,
    WorkspaceMutationJournal,
    _unregister_live,
    recover_mutation_journals,
    workspace_mutation_lock,
)
from crucible.workflows.journal_recover import main as journal_recover_main


def _git(cwd: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    ).stdout


def _spec(base_commit: str) -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="journal-recovery-test",
            repo_url="https://example.invalid/source.git",
            base_commit=base_commit,
            branch="main",
            submodule_path="source",
            runners={},
        ),
        execution=None,
        intent="exercise exact external journal recovery",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=None,
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _family(journal: WorkspaceMutationJournal) -> tuple[WorkspaceMutationJournal, ...]:
    result = [journal]
    for child in journal._nested_journals():
        result.extend(_family(child))
    return tuple(result)


def _simulate_death(journal: WorkspaceMutationJournal) -> None:
    for member in _family(journal):
        _unregister_live(member.transaction_id)


def _tree_image(root: Path) -> dict[str, tuple[object, ...]]:
    image: dict[str, tuple[object, ...]] = {}
    for path in sorted((root, *root.rglob("*"))):
        relative = "." if path == root else path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            image[relative] = ("symlink", mode, os.readlink(path))
        elif path.is_dir():
            image[relative] = ("directory", mode)
        elif path.is_file():
            image[relative] = ("file", mode, path.read_bytes())
        else:
            image[relative] = ("other", mode)
    return image


def _setup(
    tmp_path: Path,
    *,
    keep_peer: bool = False,
    late_peer_extra: bool = False,
    with_ancestor_addition: bool = False,
) -> dict[str, object]:
    outer = tmp_path / "outer"
    source = outer / "source"
    if with_ancestor_addition:
        source_origin = tmp_path / "source-origin"
        source_origin.mkdir()
        _git(source_origin, "init", "-q", "-b", "main")
        _git(source_origin, "config", "user.email", "recover@test.invalid")
        _git(source_origin, "config", "user.name", "Recovery Test")
        for name in ("a.txt", "peer.txt", "staged.txt"):
            (source_origin / name).write_text(
                f"baseline {name}\n", encoding="utf-8"
            )
        _git(source_origin, "add", "-A")
        _git(source_origin, "commit", "-qm", "baseline")
        outer.mkdir()
        _git(outer, "init", "-q", "-b", "main")
        _git(outer, "config", "user.email", "recover@test.invalid")
        _git(outer, "config", "user.name", "Recovery Test")
        (outer / "outer.txt").write_text("outer baseline\n", encoding="utf-8")
        _git(outer, "add", "outer.txt")
        _git(outer, "commit", "-qm", "outer baseline")
        _git(
            outer,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            source_origin.as_uri(),
            "source",
        )
        _git(outer, "commit", "-qam", "add source submodule")
    else:
        source.mkdir(parents=True)
        _git(source, "init", "-q", "-b", "main")
        _git(source, "config", "user.email", "recover@test.invalid")
        _git(source, "config", "user.name", "Recovery Test")
        for name in ("a.txt", "peer.txt", "staged.txt"):
            (source / name).write_text(f"baseline {name}\n", encoding="utf-8")
        _git(source, "add", "-A")
        _git(source, "commit", "-qm", "baseline")
    _git(source, "config", "user.email", "recover@test.invalid")
    _git(source, "config", "user.name", "Recovery Test")
    base = _git(source, "rev-parse", "HEAD").decode().strip()

    # The accepted operation must be an append to an existing stash, never
    # creation/rewrite of the namespace.
    (source / "a.txt").write_text("pre-existing stash\n", encoding="utf-8")
    _git(source, "stash", "push", "-m", "baseline stash")

    clean_peer = tmp_path / "clean-peer"
    dirty_peer = tmp_path / "dirty-peer"
    survivor_peer = tmp_path / "survivor-peer"
    _git(source, "worktree", "add", "-q", "-b", "clean-peer", str(clean_peer), "HEAD")
    _git(source, "worktree", "add", "-q", "--detach", str(dirty_peer), "HEAD")
    if keep_peer:
        _git(
            source,
            "worktree",
            "add",
            "-q",
            "-b",
            "survivor-peer",
            str(survivor_peer),
            "HEAD",
        )
    (dirty_peer / "peer.txt").write_text("dirty peer worktree\n", encoding="utf-8")
    (dirty_peer / "staged.txt").write_text("dirty staged peer\n", encoding="utf-8")
    _git(dirty_peer, "add", "staged.txt")
    untracked = dirty_peer / "untracked.txt"
    untracked.write_text("dirty untracked peer\n", encoding="utf-8")

    corpus_dir = tmp_path / "corpus"
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec(base))
    run_id = "run-external-recovery"
    journal = WorkspaceMutationJournal.begin(
        cwd=source,
        journal_root=corpus_dir / "runs" / run_id / "phase1_mutations",
        slice_id="s0",
        scope=("a.txt", "new.txt"),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (corpus_dir / "runs" / run_id / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": "crucible.workflow-outcome.v1",
                "workflow": "tier",
                "run_id": run_id,
                "terminal_class": "permanent_failure",
                "detail_status": "failed",
                "cost_usd": 0.0,
                "spec_version": 0,
                "refs": [],
                "error_kind": "MutationJournalError",
                "error_message": "retained for recovery test",
                "parent_run_id": None,
                "operation_id": None,
                "completed_at": "2026-07-02T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (source / "a.txt").write_text("model postimage\n", encoding="utf-8")
    (source / "new.txt").write_text("model create\n", encoding="utf-8")
    diff, files, captured_base = capture_workspace_delta(
        cwd=str(source),
        path_scope=("a.txt", "new.txt"),
    )
    assert set(files) == {"a.txt", "new.txt"}
    assert captured_base == base
    diff_ref = str(corpus.put_artifact(diff, kind="diffs", name_hint="s0.diff"))

    accepted_ancestor = None
    accepted_module = None
    if with_ancestor_addition:
        nested_origin = tmp_path / "ovstage-origin"
        nested_origin.mkdir()
        _git(nested_origin, "init", "-q", "-b", "main")
        _git(nested_origin, "config", "user.email", "recover@test.invalid")
        _git(nested_origin, "config", "user.name", "Recovery Test")
        (nested_origin / "nested.txt").write_text(
            "nested module\n", encoding="utf-8"
        )
        _git(nested_origin, "add", "nested.txt")
        _git(nested_origin, "commit", "-qm", "nested module")

        module_origin = tmp_path / "accepted-module-origin"
        module_origin.mkdir()
        _git(module_origin, "init", "-q", "-b", "main")
        _git(module_origin, "config", "user.email", "recover@test.invalid")
        _git(module_origin, "config", "user.name", "Recovery Test")
        (module_origin / "module.txt").write_text(
            "accepted module\n", encoding="utf-8"
        )
        (module_origin / ".gitignore").write_text(
            "__pycache__/\n", encoding="utf-8"
        )
        _git(module_origin, "add", "module.txt", ".gitignore")
        _git(module_origin, "commit", "-qm", "accepted module")
        _git(
            module_origin,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            nested_origin.as_uri(),
            "ovstage-internal",
        )
        _git(module_origin, "commit", "-qam", "add initialized descendant")

        accepted_path = "dvangelder-isaacsim-m2"
        accepted_url = (
            "git@github.com:NVIDIA-dev/dvangelder-isaacsim-m2.git"
        )
        _git(
            outer,
            "-c",
            f"url.{module_origin.as_uri()}.insteadOf={accepted_url}",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            accepted_url,
            accepted_path,
        )
        _git(
            outer,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--",
            accepted_path,
        )
        accepted_module = outer / accepted_path
        ignored = accepted_module / "__pycache__"
        ignored.mkdir()
        (ignored / "generated.pyc").write_bytes(b"ignored runtime payload\n")
        ancestor_index = Path(
            _git(
                outer,
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "index",
            )
            .decode()
            .strip()
        ).read_bytes()
        index_ref = str(
            corpus.put_artifact(
                ancestor_index,
                kind="journal_recovery_indexes",
                name_hint="ancestor.index",
            )
        )
        gitmodules_ref = str(
            corpus.put_artifact(
                (outer / ".gitmodules").read_bytes(),
                kind="journal_recovery_gitmodules",
                name_hint="ancestor.gitmodules",
            )
        )
        gitlink_oid = _git(accepted_module, "rev-parse", "HEAD").decode().strip()
        ancestor_journal = next(
            member
            for member in _family(journal)
            if "::ancestor-superproject" in member.slice_id
        )
        acceptance_payload = json.dumps(
            {
                "schema": "crucible.accepted-ancestor-submodule.v1",
                "run_id": run_id,
                "transaction_id": journal.transaction_id,
                "ancestor_transaction_id": ancestor_journal.transaction_id,
                "ancestor_root": str(outer),
                "submodule_path": accepted_path,
                "submodule_url": accepted_url,
                "gitlink_oid": gitlink_oid,
                "index_ref": index_ref,
                "gitmodules_ref": gitmodules_ref,
            },
            sort_keys=True,
        ).encode("utf-8")
        acceptance_ref = str(
            corpus.put_artifact(
                acceptance_payload,
                kind="journal_recovery_acceptances",
                name_hint="ancestor.acceptance.json",
            )
        )
        accepted_ancestor = AcceptedAncestorSubmodule(
            run_id=run_id,
            transaction_id=journal.transaction_id,
            ancestor_transaction_id=ancestor_journal.transaction_id,
            ancestor_root=str(outer),
            submodule_path=accepted_path,
            submodule_url=accepted_url,
            gitlink_oid=gitlink_oid,
            index_ref=index_ref,
            gitmodules_ref=gitmodules_ref,
            acceptance_ref=acceptance_ref,
        ).normalized()

    if late_peer_extra:
        (dirty_peer / "late-untracked.txt").write_text(
            "not captured by the journal\n",
            encoding="utf-8",
        )
    _git(dirty_peer, "stash", "push", "--include-untracked", "-m", "cleanup peer")
    stash_tip = _git(source, "rev-parse", "refs/stash").decode().strip()
    _git(source, "worktree", "remove", str(dirty_peer))
    _git(source, "worktree", "remove", str(clean_peer))
    _simulate_death(journal)
    request = ExternalRecoveryRequest(
        run_id=run_id,
        transaction_id=journal.transaction_id,
        expected_diff_ref=diff_ref,
        removed_linked_worktrees=(str(clean_peer), str(dirty_peer)),
        appended_stash_tip=stash_tip,
        reason="operator intentionally stashed and removed obsolete peers",
        ancestor_submodule=accepted_ancestor,
    ).normalized()
    return {
        "outer": outer,
        "source": source,
        "clean_peer": clean_peer,
        "dirty_peer": dirty_peer,
        "survivor_peer": survivor_peer if keep_peer else None,
        "corpus": corpus,
        "corpus_dir": corpus_dir,
        "journal": journal,
        "request": request,
        "stash_tip": stash_tip,
        "accepted_module": accepted_module,
        "accepted_ancestor": accepted_ancestor,
    }


def _apply(state: dict[str, object]):
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    with workspace_mutation_lock(source):
        return inspect_external_recovery(corpus, source, request, apply=True)


def _replace_ancestor_acceptance(
    state: dict[str, object],
    **changes: str,
) -> AcceptedAncestorSubmodule:
    corpus = state["corpus"]
    request = state["request"]
    accepted = state["accepted_ancestor"]
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    record = accepted.to_manifest_json()
    record.update(changes)
    payload = json.dumps(record, sort_keys=True).encode("utf-8")
    acceptance_ref = str(
        corpus.put_artifact(
            payload,
            kind="journal_recovery_acceptances",
            name_hint="ancestor.acceptance.json",
        )
    )
    replacement = AcceptedAncestorSubmodule(
        run_id=record["run_id"],
        transaction_id=record["transaction_id"],
        ancestor_transaction_id=record["ancestor_transaction_id"],
        ancestor_root=record["ancestor_root"],
        submodule_path=record["submodule_path"],
        submodule_url=record["submodule_url"],
        gitlink_oid=record["gitlink_oid"],
        index_ref=record["index_ref"],
        gitmodules_ref=record["gitmodules_ref"],
        acceptance_ref=acceptance_ref,
    ).normalized()
    state["accepted_ancestor"] = replacement
    state["request"] = replace(
        request,
        ancestor_submodule=replacement,
    ).normalized()
    return replacement


def _accept_current_ancestor_index(state: dict[str, object]) -> str:
    corpus = state["corpus"]
    accepted = state["accepted_ancestor"]
    assert isinstance(corpus, Corpus)
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    index = Path(
        _git(
            Path(accepted.ancestor_root),
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "index",
        )
        .decode()
        .strip()
    )
    index_ref = str(
        corpus.put_artifact(
            index.read_bytes(),
            kind="journal_recovery_indexes",
            name_hint="ancestor.index",
        )
    )
    _replace_ancestor_acceptance(state, index_ref=index_ref)
    return index_ref


def _accept_current_ancestor_gitmodules(state: dict[str, object]) -> str:
    corpus = state["corpus"]
    accepted = state["accepted_ancestor"]
    assert isinstance(corpus, Corpus)
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    gitmodules_ref = str(
        corpus.put_artifact(
            (Path(accepted.ancestor_root) / ".gitmodules").read_bytes(),
            kind="journal_recovery_gitmodules",
            name_hint="ancestor.gitmodules",
        )
    )
    _replace_ancestor_acceptance(state, gitmodules_ref=gitmodules_ref)
    return gitmodules_ref


def _assert_ancestor_rejected(
    state: dict[str, object],
    message: str,
) -> None:
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match=message):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    assert (source / "new.txt").is_file()


def test_check_is_read_only_then_apply_preserves_stash_and_removals(
    tmp_path: Path,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    journal = state["journal"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(journal, WorkspaceMutationJournal)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    manifest_before = (journal.journal_dir / "manifest.json").read_bytes()
    source_before = _tree_image(source)
    journal_before = _tree_image(journal.journal_dir)
    stash_log = Path(
        _git(
            source,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "logs/refs/stash",
        )
        .decode()
        .strip()
    )
    stash_log_before = stash_log.read_bytes()
    with workspace_mutation_lock(source):
        checked = inspect_external_recovery(corpus, source, request, apply=False)
    assert checked.status == "ready"
    assert (source / "a.txt").read_text() == "model postimage\n"
    assert (source / "new.txt").read_text() == "model create\n"
    assert (journal.journal_dir / "manifest.json").read_bytes() == manifest_before
    assert _tree_image(source) == source_before
    assert _tree_image(journal.journal_dir) == journal_before
    assert not (journal.journal_dir / "external_recovery.json").exists()
    assert not (journal.journal_dir.parent.parent / "journal_recovery").exists()
    assert stash_log.read_bytes() == stash_log_before

    report = _apply(state)
    assert report.status == "recovered"
    assert (source / "a.txt").read_text() == "baseline a.txt\n"
    assert not (source / "new.txt").exists()
    assert _git(source, "status", "--porcelain") == b""
    assert (
        _git(source, "rev-parse", "refs/stash").decode().strip() == state["stash_tip"]
    )
    assert stash_log.read_bytes() == stash_log_before
    assert not state["clean_peer"].exists()
    assert not state["dirty_peer"].exists()
    assert not journal.journal_dir.exists()
    assert report.object_addition_oids
    audit = json.loads(Path(str(report.audit_path)).read_text())
    assert audit["stage"] == "complete"
    assert audit["request"]["reason"] == request.reason


def test_exact_ancestor_submodule_addition_is_preserved(
    tmp_path: Path,
) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    accepted_module = state["accepted_module"]
    accepted = state["accepted_ancestor"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(accepted_module, Path)
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    outer_index = Path(
        _git(
            Path(accepted.ancestor_root),
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "index",
        )
        .decode()
        .strip()
    )
    index_before = outer_index.read_bytes()
    gitmodules_before = (
        Path(accepted.ancestor_root) / ".gitmodules"
    ).read_bytes()
    with workspace_mutation_lock(source):
        checked = inspect_external_recovery(corpus, source, request, apply=False)
    assert checked.status == "ready"
    assert checked.ancestor_root == accepted.ancestor_root
    assert checked.added_submodule_path == accepted.submodule_path
    assert checked.ancestor_transaction_id
    report = _apply(state)
    assert report.status == "recovered"
    assert outer_index.read_bytes() == index_before
    assert (
        Path(accepted.ancestor_root) / ".gitmodules"
    ).read_bytes() == gitmodules_before
    assert _git(accepted_module, "rev-parse", "HEAD").decode().strip() == (
        accepted.gitlink_oid
    )
    assert (accepted_module / "__pycache__" / "generated.pyc").is_file()


def test_cli_uses_one_transaction_bound_ancestor_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    request = state["request"]
    accepted = state["accepted_ancestor"]
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    args = [
        "--spec",
        str(state["corpus_dir"]),
        "--cwd",
        str(state["outer"]),
        "--run-id",
        request.run_id,
        "--transaction-id",
        request.transaction_id,
        "--expected-diff-ref",
        request.expected_diff_ref,
        "--accept-removed-linked-worktree",
        request.removed_linked_worktrees[0],
        "--accept-removed-linked-worktree",
        request.removed_linked_worktrees[1],
        "--accept-appended-stash-tip",
        request.appended_stash_tip,
        "--accept-ancestor-change-ref",
        accepted.acceptance_ref,
        "--reason",
        request.reason,
    ]
    assert journal_recover_main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready"
    assert result["ancestor_root"] == accepted.ancestor_root
    assert result["added_submodule_path"] == accepted.submodule_path


@pytest.mark.parametrize("mutation", ["extra_path", "hidden_flag"])
def test_ancestor_index_semantics_reject_extra_path_or_flag(
    tmp_path: Path,
    mutation: str,
) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    accepted = state["accepted_ancestor"]
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    outer = Path(accepted.ancestor_root)
    if mutation == "extra_path":
        (outer / "extra.txt").write_text("extra staged path\n", encoding="utf-8")
        _git(outer, "add", "extra.txt")
        message = "staged delta"
    else:
        _git(outer, "update-index", "--assume-unchanged", "outer.txt")
        message = "index flags"
    _accept_current_ancestor_index(state)
    _assert_ancestor_rejected(state, message)


@pytest.mark.parametrize("mutation", ["noncanonical", "second_section"])
def test_ancestor_gitmodules_must_be_one_canonical_append(
    tmp_path: Path,
    mutation: str,
) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    accepted = state["accepted_ancestor"]
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    outer = Path(accepted.ancestor_root)
    path = outer / ".gitmodules"
    payload = path.read_bytes()
    if mutation == "noncanonical":
        marker = f'[submodule "{accepted.submodule_path}"]'.encode("ascii")
        payload = payload.replace(marker, b"\n" + marker, 1)
    else:
        payload += (
            '[submodule "unexpected"]\n'
            "\tpath = unexpected\n"
            "\turl = https://example.invalid/unexpected.git\n"
        ).encode("ascii")
    path.write_bytes(payload)
    _git(outer, "add", ".gitmodules")
    _accept_current_ancestor_index(state)
    _accept_current_ancestor_gitmodules(state)
    _assert_ancestor_rejected(state, "one canonical section append")


def test_ancestor_config_rejects_any_extra_delta(tmp_path: Path) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    accepted = state["accepted_ancestor"]
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    _git(
        Path(accepted.ancestor_root),
        "config",
        "crucible.unaccepted",
        "true",
    )
    _assert_ancestor_rejected(state, "config is not exactly")


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_ancestor_object_delta_is_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    accepted = state["accepted_ancestor"]
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    outer = Path(accepted.ancestor_root)
    if mutation == "extra":
        subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=outer,
            input=b"unaccepted loose object\n",
            capture_output=True,
            check=True,
        )
        message = "object delta"
    else:
        record = _git(outer, "ls-files", "--stage", "--", ".gitmodules")
        oid = record.split()[1].decode()
        object_path = Path(
            _git(
                outer,
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "objects",
            )
            .decode()
            .strip()
        ) / oid[:2] / oid[2:]
        object_path.unlink()
        message = "object delta|objects changed|cat-file"
    _assert_ancestor_rejected(state, message)


@pytest.mark.parametrize("mutation", ["gitlink", "url", "head", "descendant_url"])
def test_accepted_module_identity_is_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    accepted = state["accepted_ancestor"]
    module = state["accepted_module"]
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    assert isinstance(module, Path)
    if mutation == "gitlink":
        _replace_ancestor_acceptance(state, gitlink_oid="f" * 40)
        message = "staged delta"
    elif mutation == "url":
        _git(module, "remote", "set-url", "origin", "https://example.invalid/wrong")
        message = "origin URL"
    elif mutation == "head":
        _git(module, "checkout", "-q", "HEAD^")
        message = "status differs|HEAD differs|dirty"
    else:
        nested = module / "ovstage-internal"
        _git(nested, "remote", "set-url", "origin", "https://example.invalid/wrong")
        message = "origin URL"
    _assert_ancestor_rejected(state, message)


def test_accepted_module_rejects_nonignored_dirty_state(tmp_path: Path) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    module = state["accepted_module"]
    assert isinstance(module, Path)
    (module / "untracked.txt").write_text("not accepted\n", encoding="utf-8")
    _assert_ancestor_rejected(state, "status differs|dirty")


@pytest.mark.parametrize(
    "mutation",
    ["alternates", "linked_worktree", "operation", "hooks_path", "include"],
)
def test_accepted_module_rejects_unsafe_admin_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    module = state["accepted_module"]
    assert isinstance(module, Path)
    admin = Path(
        _git(module, "rev-parse", "--path-format=absolute", "--git-dir")
        .decode()
        .strip()
    )
    if mutation == "alternates":
        alternates = admin / "objects" / "info" / "alternates"
        alternates.write_text("/tmp/unaccepted-objects\n", encoding="utf-8")
        message = "alternate object store"
    elif mutation == "linked_worktree":
        _git(module, "worktree", "add", "-q", "--detach", str(tmp_path / "linked"))
        message = "linked worktrees|linked-worktree admin"
    elif mutation == "operation":
        (admin / "MERGE_HEAD").write_text("0" * 40 + "\n", encoding="ascii")
        message = "operation state"
    elif mutation == "hooks_path":
        _git(module, "config", "core.hooksPath", str(tmp_path / "hooks"))
        message = "redirects Git execution"
    else:
        _git(module, "config", "include.path", str(tmp_path / "untrusted.config"))
        message = "config contains includes"
    _assert_ancestor_rejected(state, message)


def test_ignored_payload_cannot_introduce_git_admin_identity(tmp_path: Path) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    module = state["accepted_module"]
    assert isinstance(module, Path)
    rogue = module / "__pycache__" / "rogue" / ".git"
    rogue.mkdir(parents=True)
    _assert_ancestor_rejected(state, "unbound nested Git administration")


@pytest.mark.parametrize("binding", ["run_id", "transaction_id", "ancestor_transaction_id", "ancestor_root"])
def test_ancestor_acceptance_manifest_is_transaction_bound(
    tmp_path: Path,
    binding: str,
) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    replacement = (
        "wrong-run"
        if binding == "run_id"
        else str(tmp_path / "wrong-ancestor")
        if binding == "ancestor_root"
        else "0" * 32
    )
    _replace_ancestor_acceptance(state, **{binding: replacement})
    _assert_ancestor_rejected(
        state,
        "differs from the requested transaction|different run or transaction|direct guard",
    )


@pytest.mark.parametrize("artifact", ["acceptance", "index", "gitmodules"])
def test_ancestor_artifact_tamper_is_rejected(
    tmp_path: Path,
    artifact: str,
) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    corpus = state["corpus"]
    accepted = state["accepted_ancestor"]
    assert isinstance(corpus, Corpus)
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    if artifact == "acceptance":
        kind = "journal_recovery_acceptances"
        reference = accepted.acceptance_ref
    elif artifact == "index":
        kind = "journal_recovery_indexes"
        reference = accepted.index_ref
    else:
        kind = "journal_recovery_gitmodules"
        reference = accepted.gitmodules_ref
    digest = reference.split(":", 1)[1]
    path = Path(corpus.spec_dir) / "artifacts" / kind / f"{digest}.bin"
    path.write_bytes(path.read_bytes() + b"tamper")
    _assert_ancestor_rejected(state, "CAS validation")


def test_prepared_ancestor_acceptance_is_revalidated_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path, with_ancestor_addition=True)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    accepted = state["accepted_ancestor"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(accepted, AcceptedAncestorSubmodule)
    assert isinstance(journal, WorkspaceMutationJournal)
    from crucible.workflows import _journal_recovery

    original = _journal_recovery._write_event

    def stop_after_source_restore(current, event):
        original(current, event)
        if event.get("stage") == "source_restored":
            raise RuntimeError("simulated ancestor recovery death")

    monkeypatch.setattr(
        _journal_recovery,
        "_write_event",
        stop_after_source_restore,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="ancestor recovery death"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(_journal_recovery, "_write_event", original)
    _git(
        Path(accepted.ancestor_root),
        "config",
        "crucible.after-prepare",
        "true",
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="config is not exactly"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert journal.journal_dir.is_dir()
    assert (source / "a.txt").read_text(encoding="utf-8") == "baseline a.txt\n"


def test_cli_resolves_outer_submodule_once_and_requires_apply(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _setup(tmp_path)
    request = state["request"]
    assert isinstance(request, ExternalRecoveryRequest)
    args = [
        "--spec",
        str(state["corpus_dir"]),
        "--cwd",
        str(state["outer"]),
        "--run-id",
        request.run_id,
        "--transaction-id",
        request.transaction_id,
        "--expected-diff-ref",
        request.expected_diff_ref,
        "--accept-removed-linked-worktree",
        request.removed_linked_worktrees[0],
        "--accept-removed-linked-worktree",
        request.removed_linked_worktrees[1],
        "--accept-appended-stash-tip",
        request.appended_stash_tip,
        "--reason",
        request.reason,
    ]
    assert journal_recover_main(args) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["status"] == "ready"
    assert (state["source"] / "new.txt").exists()
    assert journal_recover_main([*args, "--apply"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["status"] == "recovered"
    assert not (state["source"] / "new.txt").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("wrong_diff", "artifact|postimage"),
        ("wrong_tip", "refs changed|accepted transition"),
        ("missing_peer", "every and only|worktree delta"),
        ("empty_reason", "operator reason"),
    ],
)
def test_acceptance_is_exact_and_fails_before_source_write(
    tmp_path: Path,
    change: str,
    message: str,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    original = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(original, ExternalRecoveryRequest)
    values = original.to_json()
    if change == "wrong_diff":
        values["expected_diff_ref"] = "sha256:" + "0" * 64
    elif change == "wrong_tip":
        values["appended_stash_tip"] = "1" * 40
    elif change == "missing_peer":
        values["removed_linked_worktrees"] = [original.removed_linked_worktrees[0]]
    elif change == "empty_reason":
        values["reason"] = ""
    with workspace_mutation_lock(source):
        with pytest.raises((MutationJournalError, FileNotFoundError), match=message):
            request = ExternalRecoveryRequest.from_json(values)
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text() == "model postimage\n"
    assert (source / "new.txt").exists()


def test_stash_rewrite_is_rejected(tmp_path: Path) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    log = Path(
        _git(
            source,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "logs/refs/stash",
        )
        .decode()
        .strip()
    )
    data = log.read_bytes()
    replacement = b"f" if data[10:11] != b"f" else b"e"
    log.write_bytes(data[:10] + replacement + data[11:])
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="not an append"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text() == "model postimage\n"


def test_prepared_recovery_resumes_after_source_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    from crucible.workflows import _journal_recovery

    original = _journal_recovery._write_event
    raised = False

    def fail_after_source(current, event):
        nonlocal raised
        original(current, event)
        if event.get("stage") == "source_restored" and not raised:
            raised = True
            raise RuntimeError("simulated process death")

    monkeypatch.setattr(_journal_recovery, "_write_event", fail_after_source)
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="simulated process death"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text() == "baseline a.txt\n"
    assert not (source / "new.txt").exists()
    monkeypatch.setattr(_journal_recovery, "_write_event", original)
    with workspace_mutation_lock(source):
        recovery = recover_mutation_journals(corpus, source)
    assert recovery.failures == ()
    assert recovery.recovered == ("s0",)
    assert not journal.journal_dir.exists()


def test_source_restored_stage_rejects_reintroduced_postimage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    from crucible.workflows import _journal_recovery

    original = _journal_recovery._write_event

    def stop_after_source_restore(current, event):
        original(current, event)
        if event.get("stage") == "source_restored":
            raise RuntimeError("simulated death after source restore")

    monkeypatch.setattr(
        _journal_recovery,
        "_write_event",
        stop_after_source_restore,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="after source restore"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(_journal_recovery, "_write_event", original)
    (source / "a.txt").write_text("model postimage\n", encoding="utf-8")
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="regressed after source restore"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert journal.journal_dir.is_dir()
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"


def test_verified_stage_rejects_reintroduced_postimage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    from crucible.workflows import _journal_recovery

    original = _journal_recovery._cleanup_verified

    def stop_before_cleanup(
        _inspection,
        _event,
        *,
        members_preinstalled=False,
    ):
        del members_preinstalled
        raise RuntimeError("simulated death after verification")

    monkeypatch.setattr(
        _journal_recovery,
        "_cleanup_verified",
        stop_before_cleanup,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="after verification"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(_journal_recovery, "_cleanup_verified", original)
    (source / "a.txt").write_text("model postimage\n", encoding="utf-8")
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="regressed after source restore"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert journal.journal_dir.is_dir()
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"


def test_recovery_specific_loose_object_timestamp_drift_is_fsck_tolerated(
    tmp_path: Path,
) -> None:
    state = _setup(tmp_path)
    journal = state["journal"]
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(journal, WorkspaceMutationJournal)
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    objects = Path(str(journal._manifest["objects_path"]))
    loose = next(
        objects / relative
        for relative, record in journal._manifest["object_inventory"].items()
        if record.get("kind") == "file"
        and len(relative.split("/")) == 2
        and len(relative.replace("/", "")) in {40, 64}
    )
    before = loose.stat()
    os.utime(loose, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    with workspace_mutation_lock(source):
        report = inspect_external_recovery(corpus, source, request, apply=False)
    assert report.status == "ready"


def test_corrupt_added_stash_object_fails_before_source_write(tmp_path: Path) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    with workspace_mutation_lock(source):
        checked = inspect_external_recovery(corpus, source, request, apply=False)
    oid = checked.object_addition_oids[0]
    object_file = Path(str(journal._manifest["objects_path"])) / oid[:2] / oid[2:]
    mode = stat.S_IMODE(object_file.stat().st_mode)
    object_file.chmod(mode | stat.S_IWUSR)
    payload = object_file.read_bytes()
    object_file.write_bytes(bytes([payload[0] ^ 0x01]) + payload[1:])
    object_file.chmod(mode)
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text() == "model postimage\n"
    assert (source / "new.txt").exists()


@pytest.mark.parametrize("control", ["config", "packed-refs", "index"])
def test_unrelated_primary_admin_change_is_rejected(
    tmp_path: Path,
    control: str,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    path = Path(
        _git(source, "rev-parse", "--path-format=absolute", "--git-path", control)
        .decode()
        .strip()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(b"# unauthorized admin change\n")
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text() == "model postimage\n"


def test_unaffected_linked_peer_is_validated_not_adopted(tmp_path: Path) -> None:
    state = _setup(tmp_path, keep_peer=True)
    survivor = state["survivor_peer"]
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(survivor, Path)
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    (survivor / "peer.txt").write_text("unaccepted survivor drift\n")
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="every and only"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text() == "model postimage\n"


def test_verified_marker_cleanup_resumes_after_partial_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    from crucible.workflows import _journal_recovery

    original = _journal_recovery._cleanup_verified
    raised = False

    def partial_cleanup(
        inspection,
        event,
        *,
        members_preinstalled=False,
    ):
        nonlocal raised
        if not raised:
            raised = True
            child = inspection.family[-1]
            Path(str(child._manifest["registry_marker"])).unlink()
            raise RuntimeError("simulated marker cleanup death")
        return original(
            inspection,
            event,
            members_preinstalled=members_preinstalled,
        )

    monkeypatch.setattr(_journal_recovery, "_cleanup_verified", partial_cleanup)
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="marker cleanup death"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(_journal_recovery, "_cleanup_verified", original)
    with workspace_mutation_lock(source):
        report = inspect_external_recovery(corpus, source, request, apply=True)
    assert report.status == "recovered"
    assert not journal.journal_dir.exists()


def test_ordinary_reaper_without_durable_acceptance_remains_fail_closed(
    tmp_path: Path,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(journal, WorkspaceMutationJournal)
    with workspace_mutation_lock(source):
        recovery = recover_mutation_journals(corpus, source)
    assert recovery.recovered == ()
    assert recovery.failures
    assert journal.journal_dir.is_dir()
    assert not (journal.journal_dir / "external_recovery.json").exists()
    assert (source / "a.txt").read_text() == "model postimage\n"
    assert (source / "new.txt").exists()


def test_prepared_recovery_resumes_from_mixed_pre_and_postimage_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    original = WorkspaceMutationJournal._restore_snapshot_root
    calls = 0

    def die_after_first(current, cwd, relative):
        nonlocal calls
        original(current, cwd, relative)
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated mid-source death")

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "_restore_snapshot_root",
        die_after_first,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="mid-source death"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (journal.journal_dir / "external_recovery.json").is_file()
    assert (source / "a.txt").read_text() == "baseline a.txt\n"
    assert (source / "new.txt").exists()
    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "_restore_snapshot_root",
        original,
    )
    with workspace_mutation_lock(source):
        report = inspect_external_recovery(corpus, source, request, apply=True)
    assert report.status == "recovered"
    assert not (source / "new.txt").exists()


def test_stash_cannot_launder_a_path_added_after_peer_snapshot(tmp_path: Path) -> None:
    state = _setup(tmp_path, late_peer_extra=True)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="untracked tree differs"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text() == "model postimage\n"


def test_missing_terminal_outcome_is_rejected(tmp_path: Path) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    outcome = Path(corpus.spec_dir) / "runs" / request.run_id / "outcome.json"
    outcome.unlink()
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="terminal outcome"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text() == "model postimage\n"


@pytest.mark.parametrize(
    "reserved_name",
    ["journal_recovery", "journal_recovery_tombstones"],
)
def test_reserved_recovery_directory_symlink_is_rejected_before_source_write(
    tmp_path: Path,
    reserved_name: str,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    outside = tmp_path / f"outside-{reserved_name}"
    outside.mkdir()
    reserved = (
        Path(corpus.spec_dir) / "runs" / request.run_id / reserved_name
    )
    reserved.symlink_to(outside, target_is_directory=True)
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="directory is aliased"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text() == "model postimage\n"
    assert (source / "new.txt").is_file()
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("parent_slice_id", "different-slice", "parent slice"),
        ("source_root", "/different-source", "source-root"),
        ("journal_dir", "/different-journal", "journal-directory"),
        ("postimage_records", {}, "postimage scope"),
    ],
)
def test_durable_event_identity_fields_are_revalidated_before_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    from crucible.workflows import _journal_recovery

    original = _journal_recovery._write_event

    def stop_after_prepare(current, event):
        original(current, event)
        if event.get("stage") == "prepared":
            raise RuntimeError("simulated death after durable prepare")

    monkeypatch.setattr(_journal_recovery, "_write_event", stop_after_prepare)
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="durable prepare"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(_journal_recovery, "_write_event", original)
    event_path = journal.journal_dir / "external_recovery.json"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event[field] = replacement
    event_path.write_text(json.dumps(event), encoding="utf-8")
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match=message):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_text() == "model postimage\n"
    assert (source / "new.txt").is_file()


def test_atomic_tombstone_cleanup_resumes_after_partial_rmtree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    from crucible.workflows import _journal_recovery

    original = _journal_recovery._remove_bound_directory_tree
    raised = False

    def partial_rmtree(parent_fd, name, directory_fd, expected, **kwargs):
        nonlocal raised
        if not raised and name == request.transaction_id:
            raised = True
            candidate = Path(f"/proc/self/fd/{directory_fd}")
            victim = next(item for item in candidate.rglob("*") if item.is_file())
            victim.unlink()
            raise RuntimeError("simulated tombstone deletion death")
        return original(parent_fd, name, directory_fd, expected, **kwargs)

    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_directory_tree",
        partial_rmtree,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="tombstone deletion death"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert not journal.journal_dir.exists()
    tombstone = (
        Path(corpus.spec_dir)
        / "runs"
        / request.run_id
        / "journal_recovery_tombstones"
        / request.transaction_id
    )
    assert tombstone.is_dir()
    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_directory_tree",
        original,
    )
    with workspace_mutation_lock(source):
        recovery = recover_mutation_journals(corpus, source)
    assert recovery.failures == ()
    assert recovery.recovered == ("s0",)
    assert not tombstone.exists()
    with workspace_mutation_lock(source):
        second = recover_mutation_journals(corpus, source)
    assert second.failures == ()
    assert second.recovered == ()


def test_tombstone_reaper_refuses_replaced_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    from crucible.workflows import _journal_recovery

    original = _journal_recovery._remove_bound_directory_tree

    def stop_before_delete(parent_fd, name, directory_fd, expected, **kwargs):
        if name == request.transaction_id:
            raise RuntimeError("simulated death before tombstone deletion")
        return original(parent_fd, name, directory_fd, expected, **kwargs)

    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_directory_tree",
        stop_before_delete,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="tombstone deletion"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_directory_tree",
        original,
    )
    tombstone = (
        Path(corpus.spec_dir)
        / "runs"
        / request.run_id
        / "journal_recovery_tombstones"
        / request.transaction_id
    )
    displaced = tombstone.with_name(f"{tombstone.name}.original")
    tombstone.rename(displaced)
    tombstone.mkdir()
    sentinel = tombstone / "must-not-delete"
    sentinel.write_text("unrelated directory\n", encoding="utf-8")
    with workspace_mutation_lock(source):
        recovery = recover_mutation_journals(corpus, source)
    assert any("tombstone identity changed" in item for item in recovery.failures)
    assert sentinel.read_text(encoding="utf-8") == "unrelated directory\n"


def _prepare_external_recovery_event(
    state: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Stop immediately after the durable prepared event is written."""

    from crucible.workflows import _journal_recovery

    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    original = _journal_recovery._write_event

    def stop_after_prepare(current, event):
        original(current, event)
        if event.get("stage") == "prepared":
            raise RuntimeError("stop after durable recovery prepare")

    monkeypatch.setattr(_journal_recovery, "_write_event", stop_after_prepare)
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="durable recovery prepare"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(_journal_recovery, "_write_event", original)
    event = journal.journal_dir / "external_recovery.json"
    assert event.is_file()
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    return event


def test_resumable_source_reuses_live_capture_and_constant_journal_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery, _mutation_journal

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    event_path = _prepare_external_recovery_event(state, monkeypatch)
    event = json.loads(event_path.read_text(encoding="utf-8"))
    with workspace_mutation_lock(source):
        journal = _journal_recovery._find_journal(corpus, source, request)
    original_capture = _journal_recovery._current_postimage_state
    original_inventory = _mutation_journal._mutation_journal_artifact_inventory
    captures = [0]
    inventories = [0]

    def count_capture(*args, **kwargs):
        captures[0] += 1
        return original_capture(*args, **kwargs)

    def count_inventory(*args, **kwargs):
        inventories[0] += 1
        return original_inventory(*args, **kwargs)

    monkeypatch.setattr(
        _journal_recovery,
        "_current_postimage_state",
        count_capture,
    )
    monkeypatch.setattr(
        _mutation_journal,
        "_mutation_journal_artifact_inventory",
        count_inventory,
    )
    with workspace_mutation_lock(source):
        _journal_recovery._assert_resumable_source(journal, event)
    assert captures == [2]
    assert inventories == [2]


def _leave_completed_recovery_tombstone(
    state: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Crash after tombstoning and writing the complete recovery audit."""

    from crucible.workflows import _journal_recovery

    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    original = _journal_recovery._remove_bound_directory_tree

    def stop_before_bound_delete(
        parent_fd, name, directory_fd, expected, **kwargs
    ):
        if name == request.transaction_id:
            raise RuntimeError("stop before bound tombstone delete")
        return original(parent_fd, name, directory_fd, expected, **kwargs)

    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_directory_tree",
        stop_before_bound_delete,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="bound tombstone delete"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_directory_tree",
        original,
    )
    run = Path(corpus.spec_dir) / "runs" / request.run_id
    tombstone = (
        run / "journal_recovery_tombstones" / request.transaction_id
    )
    audit = run / "journal_recovery" / f"{request.transaction_id}.json"
    assert tombstone.is_dir()
    assert json.loads(audit.read_text(encoding="utf-8"))["stage"] == "complete"
    return tombstone, audit


def _grow_file_during_bounded_read(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> list[bool]:
    """Append after the first read from the exact control-file inode."""

    identity = path.stat()
    original = module.os.read
    changed = [False]

    def grow_during_read(descriptor, count):
        chunk = original(descriptor, count)
        opened = os.fstat(descriptor)
        if (
            chunk
            and not changed[0]
            and (opened.st_dev, opened.st_ino)
            == (identity.st_dev, identity.st_ino)
        ):
            with path.open("ab") as stream:
                stream.write(b" ")
            changed[0] = True
        return chunk

    monkeypatch.setattr(module.os, "read", grow_during_read)
    return changed


def test_namespaced_artifact_read_is_bounded_and_namespace_exact(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _journal_recovery

    corpus = Corpus(tmp_path / "corpus")
    payload = b"exact recovery artifact\n"
    reference = str(corpus.put_artifact(payload, kind="diffs"))

    assert (
        _journal_recovery._load_namespaced_artifact(
            corpus,
            reference,
            kind="diffs",
            max_bytes=len(payload),
        )
        == payload
    )
    with pytest.raises(MutationJournalError, match="byte ceiling"):
        _journal_recovery._load_namespaced_artifact(
            corpus,
            reference,
            kind="diffs",
            max_bytes=len(payload) - 1,
        )
    with pytest.raises(MutationJournalError, match="artifact namespace"):
        _journal_recovery._load_namespaced_artifact(
            corpus,
            reference,
            kind="journal_recovery_acceptances",
        )


def test_namespaced_artifact_directory_swap_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    corpus = Corpus(tmp_path / "corpus")
    payload = b"authoritative artifact\n"
    reference = str(corpus.put_artifact(payload, kind="diffs"))
    digest = reference.removeprefix("sha256:")
    namespace = Path(corpus.spec_dir) / "artifacts" / "diffs"
    parked = tmp_path / "parked-diffs"
    outside = tmp_path / "outside-diffs"
    outside.mkdir()
    forged = outside / f"{digest}.bin"
    forged.write_bytes(payload)
    original = _journal_recovery._open_authority_directory_at
    swapped = [False]

    def swap_namespace_after_open(parent_fd, name, *, label):
        descriptor, opened = original(parent_fd, name, label=label)
        if name == "diffs" and not swapped[0]:
            namespace.rename(parked)
            namespace.symlink_to(outside, target_is_directory=True)
            swapped[0] = True
        return descriptor, opened

    monkeypatch.setattr(
        _journal_recovery,
        "_open_authority_directory_at",
        swap_namespace_after_open,
    )
    with pytest.raises(MutationJournalError, match="changed during artifact read"):
        _journal_recovery._load_namespaced_artifact(
            corpus,
            reference,
            kind="diffs",
        )
    assert swapped == [True]
    assert forged.read_bytes() == payload
    assert (parked / f"{digest}.bin").read_bytes() == payload


def test_namespaced_artifact_growth_during_read_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    corpus = Corpus(tmp_path / "corpus")
    payload = b"bounded artifact\n"
    reference = str(corpus.put_artifact(payload, kind="diffs"))
    digest = reference.removeprefix("sha256:")
    artifact = Path(corpus.spec_dir) / "artifacts" / "diffs" / f"{digest}.bin"
    changed = _grow_file_during_bounded_read(
        artifact,
        monkeypatch,
        _journal_recovery,
    )

    with pytest.raises(MutationJournalError, match="changed during read"):
        _journal_recovery._load_namespaced_artifact(
            corpus,
            reference,
            kind="diffs",
        )
    assert changed == [True]


def test_recovery_artifact_budget_is_aggregate_across_namespaces(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _journal_recovery

    corpus = Corpus(tmp_path / "corpus")
    first = b"first"
    second = b"second"
    first_ref = str(corpus.put_artifact(first, kind="diffs"))
    second_ref = str(
        corpus.put_artifact(second, kind="journal_recovery_indexes")
    )
    budget = _journal_recovery._RecoveryArtifactBudget(
        max_bytes=len(first) + len(second) - 1,
        max_artifacts=2,
    )

    assert (
        _journal_recovery._load_namespaced_artifact(
            corpus,
            first_ref,
            kind="diffs",
            max_bytes=len(first),
            budget=budget,
        )
        == first
    )
    with pytest.raises(MutationJournalError, match="byte ceiling"):
        _journal_recovery._load_namespaced_artifact(
            corpus,
            second_ref,
            kind="journal_recovery_indexes",
            max_bytes=len(second),
            budget=budget,
        )
    assert budget.bytes == len(first)
    assert budget.artifacts == 1


def test_accepted_ancestor_manifest_has_a_small_production_ceiling(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _journal_recovery

    corpus = Corpus(tmp_path / "corpus")
    payload = b"{}"
    reference = str(
        corpus.put_artifact(
            payload,
            kind="journal_recovery_acceptances",
        )
    )
    digest = reference.removeprefix("sha256:")
    artifact = (
        Path(corpus.spec_dir)
        / "artifacts"
        / "journal_recovery_acceptances"
        / f"{digest}.bin"
    )
    with artifact.open("r+b") as stream:
        stream.truncate(_journal_recovery._MAX_ANCESTOR_ACCEPTANCE_BYTES + 1)

    with pytest.raises(MutationJournalError, match="byte ceiling"):
        _journal_recovery.load_accepted_ancestor_submodule(corpus, reference)


def test_primary_source_recovery_diff_is_bounded_by_expected_artifact(
    tmp_path: Path,
) -> None:
    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    oversized = b"x" * (1024 * 1024)
    (source / "a.txt").write_bytes(oversized)

    with workspace_mutation_lock(source):
        with pytest.raises(
            MutationJournalError,
            match="bounded primary source recovery diff",
        ):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert (source / "a.txt").read_bytes() == oversized
    assert (source / "new.txt").read_text(encoding="utf-8") == "model create\n"


def _setup_dirty_nested_journal(
    tmp_path: Path,
    *,
    corrupt_blob_before_discovery: bool = False,
) -> tuple[Path, Corpus, WorkspaceMutationJournal]:
    """Create one crashed journal whose baseline payload is not in Git."""

    from crucible.workflows import _mutation_journal

    source = tmp_path / "nested-source"
    source.mkdir()
    _git(source, "init", "-q", "-b", "main")
    _git(source, "config", "user.email", "recover@test.invalid")
    _git(source, "config", "user.name", "Recovery Test")
    (source / "dir").mkdir()
    nested = source / "dir" / "a.txt"
    nested.write_text("committed baseline\n", encoding="utf-8")
    _git(source, "add", "-A")
    _git(source, "commit", "-qm", "baseline")
    base = _git(source, "rev-parse", "HEAD").decode().strip()
    nested.write_text("dirty baseline\n", encoding="utf-8")

    corpus = Corpus(tmp_path / "nested-corpus")
    spec = _spec(base)
    corpus.put_spec(spec)
    with workspace_mutation_lock(source):
        journal = WorkspaceMutationJournal.begin(
            cwd=source,
            journal_root=(
                Path(corpus.spec_dir) / "runs" / "nested-run" / "phase1_mutations"
            ),
            slice_id="nested",
            scope=("dir/a.txt",),
            sequence=1,
            spec=spec,
        )
    nested.write_text("model postimage\n", encoding="utf-8")
    _simulate_death(journal)
    if corrupt_blob_before_discovery:
        entry = journal._manifest["entries"]["dir/a.txt"]
        digest = str(entry["sha256"])
        size = int(entry["size"])
        (journal.journal_dir / "blobs" / digest).write_bytes(b"x" * size)
    with workspace_mutation_lock(source):
        discovered = _mutation_journal.discover_mutation_journals(
            corpus,
            workspace=source,
        )
    assert len(discovered) == 1
    assert "dir/a.txt" in discovered[0]._manifest["entries"]
    return source, corpus, discovered[0]


def test_baseline_match_rejects_intermediate_symlink_to_matching_outside_file(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _journal_recovery

    source, _corpus, journal = _setup_dirty_nested_journal(tmp_path)
    parked = tmp_path / "parked-source-dir"
    outside = tmp_path / "outside-source-dir"
    outside.mkdir()
    outside_file = outside / "a.txt"
    outside_file.write_text("dirty baseline\n", encoding="utf-8")
    (source / "dir").rename(parked)
    (source / "dir").symlink_to(outside, target_is_directory=True)
    outside_identity = outside_file.stat()

    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError):
            _journal_recovery._matches_baseline(journal, "dir/a.txt")
    after = outside_file.stat()
    assert (after.st_dev, after.st_ino) == (
        outside_identity.st_dev,
        outside_identity.st_ino,
    )
    assert outside_file.read_text(encoding="utf-8") == "dirty baseline\n"
    assert (parked / "a.txt").read_text(encoding="utf-8") == "model postimage\n"


def test_baseline_blob_directory_swap_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    source, _corpus, journal = _setup_dirty_nested_journal(tmp_path)
    (source / "dir" / "a.txt").write_text("dirty baseline\n", encoding="utf-8")
    blobs = journal.journal_dir / "blobs"
    parked = tmp_path / "parked-baseline-blobs"
    outside = tmp_path / "outside-baseline-blobs"
    outside.mkdir()
    expected = journal._manifest["entries"]["dir/a.txt"]
    digest = str(expected["sha256"])
    forged = outside / digest
    forged.write_text("dirty baseline\n", encoding="utf-8")
    original = _journal_recovery._open_authority_directory_at
    swapped = [False]

    def swap_blobs_after_open(parent_fd, name, *, label):
        descriptor, opened = original(parent_fd, name, label=label)
        if name == "blobs" and not swapped[0]:
            blobs.rename(parked)
            blobs.symlink_to(outside, target_is_directory=True)
            swapped[0] = True
        return descriptor, opened

    monkeypatch.setattr(
        _journal_recovery,
        "_open_authority_directory_at",
        swap_blobs_after_open,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="blob directory changed"):
            _journal_recovery._matches_baseline(journal, "dir/a.txt")
    assert swapped == [True]
    assert forged.read_text(encoding="utf-8") == "dirty baseline\n"
    assert (parked / digest).read_text(encoding="utf-8") == "dirty baseline\n"


def test_malformed_baseline_blob_digest_cannot_escape_journal(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _journal_recovery

    source, _corpus, journal = _setup_dirty_nested_journal(tmp_path)
    (source / "dir" / "a.txt").write_text("dirty baseline\n", encoding="utf-8")
    outside = journal.journal_dir.parent / "outside-baseline"
    outside.write_text("outside sentinel\n", encoding="utf-8")
    journal._manifest["entries"]["dir/a.txt"]["sha256"] = "../../outside-baseline"
    outside_identity = outside.stat()

    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="baseline file is malformed"):
            _journal_recovery._matches_baseline(journal, "dir/a.txt")
    after = outside.stat()
    assert (after.st_dev, after.st_ino) == (
        outside_identity.st_dev,
        outside_identity.st_ino,
    )
    assert outside.read_text(encoding="utf-8") == "outside sentinel\n"


def test_corrupt_baseline_blob_bound_during_discovery_is_still_rejected(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _journal_recovery

    source, _corpus, journal = _setup_dirty_nested_journal(
        tmp_path,
        corrupt_blob_before_discovery=True,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="corrupt mutation snapshot"):
            with _journal_recovery._bound_recovery_baseline_blobs(journal):
                pass


@pytest.mark.parametrize("control_kind", ("fifo", "symlink", "oversize", "growth"))
def test_terminal_outcome_bounded_read_failures_precede_source_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_kind: str,
) -> None:
    from crucible.workflows import _mutation_journal

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    outcome = Path(corpus.spec_dir) / "runs" / request.run_id / "outcome.json"
    changed: list[bool] | None = None
    if control_kind == "fifo":
        outcome.unlink()
        os.mkfifo(outcome)
    elif control_kind == "symlink":
        outcome.unlink()
        forged = tmp_path / "forged-outcome.json"
        forged.write_text("{}", encoding="utf-8")
        outcome.symlink_to(forged)
    elif control_kind == "oversize":
        outcome.write_bytes(
            b"x" * (_mutation_journal._MUTATION_JOURNAL_MAX_RUN_CONTROL_BYTES + 1)
        )
    else:
        changed = _grow_file_during_bounded_read(
            outcome,
            monkeypatch,
            _mutation_journal,
        )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="terminal outcome"):
            inspect_external_recovery(corpus, source, request, apply=True)
    if changed is not None:
        assert changed == [True]
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    assert (source / "new.txt").is_file()


@pytest.mark.parametrize("control_kind", ("fifo", "symlink", "oversize", "growth"))
def test_durable_recovery_event_bounded_read_failures_precede_source_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_kind: str,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    event = _prepare_external_recovery_event(state, monkeypatch)
    changed: list[bool] | None = None
    if control_kind == "fifo":
        event.unlink()
        os.mkfifo(event)
    elif control_kind == "symlink":
        event.unlink()
        forged = tmp_path / "forged-recovery-event.json"
        forged.write_text("{}", encoding="utf-8")
        event.symlink_to(forged)
    elif control_kind == "oversize":
        event.write_bytes(b"x" * (_journal_recovery._MAX_RECOVERY_EVENT_BYTES + 1))
    else:
        changed = _grow_file_during_bounded_read(
            event,
            monkeypatch,
            _journal_recovery,
        )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError):
            inspect_external_recovery(corpus, source, request, apply=True)
    if changed is not None:
        assert changed == [True]
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    assert (source / "new.txt").is_file()


@pytest.mark.parametrize("stage", ("prepare", "resume"))
@pytest.mark.parametrize(
    "swap_level",
    ("run", "durable_root", "transaction", "manifest"),
)
def test_external_recovery_rejects_post_discovery_path_swaps_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    swap_level: str,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    if stage == "resume":
        _prepare_external_recovery_event(state, monkeypatch)
    run = Path(corpus.spec_dir) / "runs" / request.run_id
    durable_root = run / "phase1_mutations"
    transaction = journal.journal_dir
    parked = tmp_path / f"parked-{stage}-{swap_level}"
    original_find = _journal_recovery._find_journal
    swapped = [False]

    def swap_after_discovery(*args, **kwargs):
        discovered = original_find(*args, **kwargs)
        if swap_level == "manifest":
            manifest = transaction / "manifest.json"
            manifest.rename(parked)
            shutil.copy2(parked, manifest)
        else:
            target = {
                "run": run,
                "durable_root": durable_root,
                "transaction": transaction,
            }[swap_level]
            target.rename(parked)
            shutil.copytree(parked, target)
        swapped[0] = True
        return discovered

    monkeypatch.setattr(_journal_recovery, "_find_journal", swap_after_discovery)
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert swapped == [True]
    assert parked.exists()
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    assert (source / "new.txt").is_file()


def test_external_recovery_rejects_event_swap_immediately_after_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    event_path = _prepare_external_recovery_event(state, monkeypatch)
    parked = tmp_path / "parked-loaded-recovery-event"
    original_load = _journal_recovery._load_event
    swapped = [False]

    def swap_after_load(journal):
        event = original_load(journal)
        event_path.rename(parked)
        shutil.copy2(parked, event_path)
        swapped[0] = True
        return event

    monkeypatch.setattr(_journal_recovery, "_load_event", swap_after_load)
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert swapped == [True]
    assert parked.is_file()
    assert event_path.is_file()
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    assert (source / "new.txt").is_file()


def test_external_recovery_rejects_transaction_swap_after_tombstone_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    parked = tmp_path / "parked-tombstone-prepared-transaction"
    replacement_sentinel = journal.journal_dir / "replacement-sentinel"
    original_write = _journal_recovery._write_event
    swapped = [False]

    def swap_after_tombstone_prepare(current, event):
        original_write(current, event)
        if event.get("stage") == "tombstone_prepared":
            current.journal_dir.rename(parked)
            shutil.copytree(parked, current.journal_dir)
            replacement_sentinel.write_text("replacement\n", encoding="utf-8")
            swapped[0] = True

    monkeypatch.setattr(
        _journal_recovery,
        "_write_event",
        swap_after_tombstone_prepare,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert swapped == [True]
    assert parked.is_dir()
    assert replacement_sentinel.read_text(encoding="utf-8") == "replacement\n"


def test_external_recovery_bound_cleanup_rejects_top_tombstone_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    parked = tmp_path / "parked-authoritative-tombstone"
    sentinel = [None]
    original = _journal_recovery._remove_bound_directory_tree

    def replace_before_bound_delete(
        parent_fd, name, directory_fd, expected, **kwargs
    ):
        if name == request.transaction_id and sentinel[0] is None:
            logical = Path(f"/proc/self/fd/{parent_fd}").resolve() / name
            logical.rename(parked)
            logical.mkdir()
            sentinel[0] = logical / "replacement-sentinel"
            sentinel[0].write_text("replacement\n", encoding="utf-8")
        return original(parent_fd, name, directory_fd, expected, **kwargs)

    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_directory_tree",
        replace_before_bound_delete,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="identity|target changed"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert parked.is_dir()
    assert isinstance(sentinel[0], Path)
    assert sentinel[0].read_text(encoding="utf-8") == "replacement\n"


@pytest.mark.parametrize("control_kind", ("fifo", "symlink", "oversize", "growth"))
def test_recovery_audit_bounded_read_failures_preserve_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_kind: str,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    tombstone, audit = _leave_completed_recovery_tombstone(state, monkeypatch)
    changed: list[bool] | None = None
    if control_kind == "fifo":
        audit.unlink()
        os.mkfifo(audit)
    elif control_kind == "symlink":
        audit.unlink()
        forged = tmp_path / "forged-recovery-audit.json"
        forged.write_text("{}", encoding="utf-8")
        audit.symlink_to(forged)
    elif control_kind == "oversize":
        audit.write_bytes(b"x" * (_journal_recovery._MAX_RECOVERY_EVENT_BYTES + 1))
    else:
        changed = _grow_file_during_bounded_read(
            audit,
            monkeypatch,
            _journal_recovery,
        )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError):
            reap_external_recovery_tombstones(corpus, source)
    if changed is not None:
        assert changed == [True]
    assert tombstone.is_dir()


def test_recovery_audit_swap_after_read_preserves_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    tombstone, audit = _leave_completed_recovery_tombstone(state, monkeypatch)
    parked = tmp_path / "parked-authoritative-recovery-audit"
    original = _journal_recovery._bound_recovery_audit
    swapped = [False]

    @contextmanager
    def swap_after_bound_read(audit_root_fd, name):
        with original(audit_root_fd, name) as bound:
            audit.rename(parked)
            shutil.copy2(parked, audit)
            swapped[0] = True
            yield bound

    monkeypatch.setattr(
        _journal_recovery,
        "_bound_recovery_audit",
        swap_after_bound_read,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="audit changed"):
            reap_external_recovery_tombstones(corpus, source)
    assert swapped == [True]
    assert tombstone.is_dir()
    assert parked.is_file()
    assert audit.is_file()


def test_event_to_audit_run_swap_fails_before_source_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    run = Path(corpus.spec_dir) / "runs" / request.run_id
    parked = tmp_path / "parked-event-authored-run"
    original = _journal_recovery.write_atomic_bytes
    swapped = [False]

    def swap_run_after_event(path, payload):
        original(path, payload)
        if Path(path).name == "external_recovery.json" and not swapped[0]:
            run.rename(parked)
            shutil.copytree(parked, run)
            swapped[0] = True

    monkeypatch.setattr(
        _journal_recovery,
        "write_atomic_bytes",
        swap_run_after_event,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert swapped == [True]
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    assert not (
        run / "journal_recovery" / f"{request.transaction_id}.json"
    ).exists()


def test_event_transition_rejects_unrelated_index_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    raw_index = journal.journal_dir / "index"
    original_index = raw_index.read_bytes()
    original = _journal_recovery.write_atomic_bytes
    mutated = [False]

    def mutate_index_after_event(path, payload):
        original(path, payload)
        if Path(path).name == "external_recovery.json" and not mutated[0]:
            raw_index.write_bytes(original_index + b"unauthorized")
            mutated[0] = True

    monkeypatch.setattr(
        _journal_recovery,
        "write_atomic_bytes",
        mutate_index_after_event,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="unauthorized paths"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert mutated == [True]
    assert raw_index.read_bytes() != original_index
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    assert (source / "new.txt").is_file()


def test_postimage_transition_rejects_unreferenced_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    sibling = journal.journal_dir / "external_recovery" / "unreferenced"
    original = _journal_recovery.write_atomic_bytes
    injected = [False]

    def inject_sibling_after_snapshot(path, payload):
        original(path, payload)
        if Path(path).name == "0" and not injected[0]:
            sibling.write_text("unreferenced\n", encoding="utf-8")
            injected[0] = True

    monkeypatch.setattr(
        _journal_recovery,
        "write_atomic_bytes",
        inject_sibling_after_snapshot,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(
            MutationJournalError,
            match="unauthorized paths|unreferenced paths",
        ):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert injected == [True]
    assert sibling.read_text(encoding="utf-8") == "unreferenced\n"
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"


def test_postimage_component_symlink_swap_cannot_escape_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    recovery = journal.journal_dir / "external_recovery"
    parked = tmp_path / "parked-bound-recovery-directory"
    outside = tmp_path / "outside-recovery-target"
    outside.mkdir()
    original = _journal_recovery._open_authority_directory_at
    swapped = [False]

    def swap_recovery_after_open(parent_fd, name, *, label):
        descriptor, opened = original(parent_fd, name, label=label)
        if name == "external_recovery" and not swapped[0]:
            recovery.rename(parked)
            recovery.symlink_to(outside, target_is_directory=True)
            swapped[0] = True
        return descriptor, opened

    monkeypatch.setattr(
        _journal_recovery,
        "_open_authority_directory_at",
        swap_recovery_after_open,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="path changed"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert swapped == [True]
    assert list(outside.iterdir()) == []
    assert (parked / "postimage" / "0").is_file()
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"


def test_resume_postimage_payload_stat_to_open_swap_fails_before_source_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    _prepare_external_recovery_event(state, monkeypatch)
    postimage = journal.journal_dir / "external_recovery" / "postimage"
    postimage_identity = postimage.stat()
    outside = tmp_path / "outside-postimage-payload"
    outside.write_text("outside\n", encoding="utf-8")
    original_open = _journal_recovery.os.open
    original_hash = _journal_recovery._hash_bound_regular_at
    swapped = [False]

    def swap_snapshot_after_stat(
        path,
        flags,
        mode=0o777,
        *,
        dir_fd=None,
    ):
        if path == "0" and dir_fd is not None and not swapped[0]:
            parent = os.fstat(dir_fd)
            if (parent.st_dev, parent.st_ino) == (
                postimage_identity.st_dev,
                postimage_identity.st_ino,
            ):
                os.unlink(path, dir_fd=dir_fd)
                os.symlink(outside, path, dir_fd=dir_fd)
                swapped[0] = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def hash_with_stat_to_open_swap(*args, **kwargs):
        _journal_recovery.os.open = swap_snapshot_after_stat
        try:
            return original_hash(*args, **kwargs)
        finally:
            _journal_recovery.os.open = original_open

    monkeypatch.setattr(
        _journal_recovery,
        "_hash_bound_regular_at",
        hash_with_stat_to_open_swap,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert swapped == [True]
    assert outside.read_text(encoding="utf-8") == "outside\n"
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    assert (source / "new.txt").is_file()


def test_resume_postimage_directory_swap_fails_before_source_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    _prepare_external_recovery_event(state, monkeypatch)
    postimage = journal.journal_dir / "external_recovery" / "postimage"
    parked = tmp_path / "parked-resume-postimage"
    outside = tmp_path / "outside-resume-postimage"
    outside.mkdir()
    original = _journal_recovery._open_authority_directory_at
    swapped = [False]

    def swap_postimage_after_open(parent_fd, name, *, label):
        descriptor, opened = original(parent_fd, name, label=label)
        if name == "postimage" and not swapped[0]:
            postimage.rename(parked)
            postimage.symlink_to(outside, target_is_directory=True)
            swapped[0] = True
        return descriptor, opened

    monkeypatch.setattr(
        _journal_recovery,
        "_open_authority_directory_at",
        swap_postimage_after_open,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(
            MutationJournalError,
            match="changed during use|special or aliased file",
        ):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert swapped == [True]
    assert list(outside.iterdir()) == []
    assert (parked / "0").is_file()
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    assert (source / "new.txt").is_file()


def test_checkout_root_swap_is_rejected_before_source_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    _prepare_external_recovery_event(state, monkeypatch)
    parked = tmp_path / "parked-authoritative-source"
    replacement_sentinel = source / "replacement-sentinel"
    original = _journal_recovery._restore_source_only
    swapped = [False]

    def swap_before_restore(journal):
        source.rename(parked)
        shutil.copytree(parked, source, symlinks=True)
        replacement_sentinel.write_text("replacement\n", encoding="utf-8")
        swapped[0] = True
        return original(journal)

    monkeypatch.setattr(
        _journal_recovery,
        "_restore_source_only",
        swap_before_restore,
    )
    with pytest.raises(MutationJournalError) as captured:
        with workspace_mutation_lock(source):
            inspect_external_recovery(corpus, source, request, apply=True)
    # A changed root can invalidate both the operation and its lexical lock
    # release.  The retained owner must remain reachable in that case, so the
    # public error is the typed cleanup envelope and the original authority
    # rejection is preserved as its cause.
    authority_error = captured.value.__cause__ or captured.value
    assert isinstance(authority_error, MutationJournalError)
    assert (
        "workspace authority root" in str(authority_error)
        or "external workspace root" in str(authority_error)
    )
    assert swapped == [True]
    assert replacement_sentinel.read_text(encoding="utf-8") == "replacement\n"
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
    assert (parked / "a.txt").read_text(encoding="utf-8") == "model postimage\n"


def test_registry_swap_between_family_bindings_deletes_no_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery

    state = _setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    family = _family(journal)
    assert len(family) > 1
    registry = Path(str(journal._manifest["registry_directory"]))
    marker_names = {
        Path(str(member._manifest["registry_marker"])).name
        for member in family
    }
    parked = tmp_path / "parked-authoritative-registry"
    original = _journal_recovery._open_authority_directory
    swapped = [False]

    def swap_after_first_registry_open(path, *, label):
        descriptor, opened = original(path, label=label)
        if label == "bound mutation journal registry" and not swapped[0]:
            registry.rename(parked)
            shutil.copytree(parked, registry)
            swapped[0] = True
        return descriptor, opened

    monkeypatch.setattr(
        _journal_recovery,
        "_open_authority_directory",
        swap_after_first_registry_open,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(MutationJournalError, match="between member bindings"):
            inspect_external_recovery(corpus, source, request, apply=True)
    assert swapped == [True]
    assert marker_names.issubset({path.name for path in parked.iterdir()})
    assert marker_names.issubset({path.name for path in registry.iterdir()})
    assert (source / "a.txt").read_text(encoding="utf-8") == "model postimage\n"
