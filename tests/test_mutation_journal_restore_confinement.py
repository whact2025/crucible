# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Adversarial confinement tests for descriptor-relative workspace restore."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest


_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent/mutation-restore-test",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}
_GIT_CONFIG = (
    "-c",
    "maintenance.auto=false",
    "-c",
    "gc.auto=0",
)


def _git(cwd: Path, *args: str) -> bytes:
    return subprocess.run(
        ["/usr/bin/git", *_GIT_CONFIG, *args],
        cwd=cwd,
        env=_GIT_ENV,
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def test_fixture_git_commands_cannot_detach_automatic_maintenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(
        argv: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout=b"stable\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _git(tmp_path, "status", "--porcelain") == b"stable\n"
    assert observed["argv"] == [
        "/usr/bin/git",
        "-c",
        "maintenance.auto=false",
        "-c",
        "gc.auto=0",
        "status",
        "--porcelain",
    ]
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["env"] == _GIT_ENV
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"


def _init_repository(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "restore@test.invalid")
    _git(repo, "config", "user.name", "restore test")
    target = repo / "dir" / "victim.txt"
    target.parent.mkdir()
    target.write_bytes(b"tracked baseline\n")
    nested_file = repo / "nest" / "keep.txt"
    nested_file.parent.mkdir()
    nested_file.write_bytes(b"tracked parent baseline\n")
    _git(repo, "add", "dir/victim.txt", "nest/keep.txt")
    _git(repo, "commit", "-qm", "baseline")


def test_restore_rejects_parent_symlink_swap_without_touching_external_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost parent binding must retain both outside bytes and the journal."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "restore" / "mutations",
        slice_id="restore-parent-swap",
        scope=("dir/victim.txt",),
        sequence=1,
    )

    target = repo / "dir" / "victim.txt"
    target.write_bytes(b"provider mutation\n")
    target_before = target.lstat()

    external = tmp_path / "external"
    external.mkdir()
    external_target = external / "victim.txt"
    external_target.write_bytes(b"outside authority\n")
    external_target.chmod(0o640)
    external_before = external_target.lstat()

    parked_parent = tmp_path / "parked-parent"
    original_remove = journals._remove_bound_workspace_entry
    swapped = False

    def swap_parent_then_remove(
        root_fd: int,
        relative: str,
        parent_fd: int,
        name: str,
        parent_chain: tuple[tuple[int, int], ...],
        **kwargs: bool,
    ) -> None:
        nonlocal swapped
        if not swapped and relative == "dir/victim.txt":
            (repo / "dir").rename(parked_parent)
            (repo / "dir").symlink_to(external, target_is_directory=True)
            swapped = True
        original_remove(
            root_fd,
            relative,
            parent_fd,
            name,
            parent_chain,
            **kwargs,
        )

    monkeypatch.setattr(
        journals,
        "_remove_bound_workspace_entry",
        swap_parent_then_remove,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert journal.active
    assert journal.journal_dir.is_dir()

    # The held old parent and the attacker-selected external parent are both
    # untouched: binding loss fails before the first unlink or replacement.
    parked_target = parked_parent / "victim.txt"
    parked_after = parked_target.lstat()
    assert parked_target.read_bytes() == b"provider mutation\n"
    assert (parked_after.st_dev, parked_after.st_ino) == (
        target_before.st_dev,
        target_before.st_ino,
    )
    external_after = external_target.lstat()
    assert external_target.read_bytes() == b"outside authority\n"
    assert (
        external_after.st_dev,
        external_after.st_ino,
        external_after.st_mode,
        external_after.st_size,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
        external_before.st_size,
    )
    assert os.path.samefile(repo / "dir", external)


def test_nested_marker_cleanup_rejects_parent_swap_without_external_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested-VCS cleanup uses the same held-root/no-follow delete boundary."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "marker" / "mutations",
        slice_id="nested-marker-parent-swap",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    marker = repo / "nest" / ".git"
    marker.mkdir()
    (marker / "provider-control").write_bytes(b"provider marker\n")

    external_parent = tmp_path / "external-nest"
    external_marker = external_parent / ".git"
    external_marker.mkdir(parents=True)
    external_target = external_marker / "outside-control"
    external_target.write_bytes(b"outside authority\n")
    external_before = external_target.lstat()

    parked_parent = tmp_path / "parked-nest"
    original_remove = journals._remove_bound_workspace_entry
    swapped = False

    def swap_parent_then_remove(
        root_fd: int,
        relative: str,
        parent_fd: int,
        name: str,
        parent_chain: tuple[tuple[int, int], ...],
        **kwargs: bool,
    ) -> None:
        nonlocal swapped
        if not swapped and relative == "nest/.git":
            (repo / "nest").rename(parked_parent)
            (repo / "nest").symlink_to(
                external_parent,
                target_is_directory=True,
            )
            swapped = True
        original_remove(
            root_fd,
            relative,
            parent_fd,
            name,
            parent_chain,
            **kwargs,
        )

    monkeypatch.setattr(
        journals,
        "_remove_bound_workspace_entry",
        swap_parent_then_remove,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert journal.active
    assert journal.journal_dir.is_dir()
    assert (parked_parent / ".git" / "provider-control").read_bytes() == (
        b"provider marker\n"
    )
    external_after = external_target.lstat()
    assert external_target.read_bytes() == b"outside authority\n"
    assert (
        external_after.st_dev,
        external_after.st_ino,
        external_after.st_mode,
        external_after.st_size,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
        external_before.st_size,
    )
    assert os.path.samefile(repo / "nest", external_parent)


def test_restore_file_install_never_overwrites_a_racing_external_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final file name is claimed with O_EXCL instead of rename-overwrite."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "install" / "mutations",
        slice_id="restore-install-race",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")

    external_target = tmp_path / "external-hardlink-target"
    external_target.write_bytes(b"outside authority\n")
    external_before = external_target.lstat()
    original_claim = journals._claim_bound_workspace_file
    inserted = False

    def insert_before_exclusive_claim(
        parent_fd: int,
        name: str,
    ) -> int:
        nonlocal inserted
        if not inserted and name == "victim.txt":
            os.link(external_target, name, dst_dir_fd=parent_fd)
            inserted = True
        return original_claim(parent_fd, name)

    monkeypatch.setattr(
        journals,
        "_claim_bound_workspace_file",
        insert_before_exclusive_claim,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert inserted
    assert journal.active
    external_after = external_target.lstat()
    assert external_target.read_bytes() == b"outside authority\n"
    assert (external_after.st_dev, external_after.st_ino) == (
        external_before.st_dev,
        external_before.st_ino,
    )
    installed = (repo / "dir" / "victim.txt").lstat()
    assert (installed.st_dev, installed.st_ino) == (
        external_before.st_dev,
        external_before.st_ino,
    )


def test_malformed_snapshot_digest_fails_before_blob_read_or_workspace_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Journal blob names are exact digests, never path or special-file input."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    target = repo / "dir" / "victim.txt"
    target.write_bytes(b"user preexisting dirt\n")
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "blob" / "mutations",
        slice_id="malformed-snapshot-blob",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    target.write_bytes(b"provider mutation\n")
    journal._manifest["entries"]["dir/victim.txt"]["sha256"] = (
        "../../outside-authority"
    )
    journal._write()

    original_read = journals._read_bound_regular_entry

    def reject_blob_read(*args, **kwargs):
        if len(args) > 1 and args[1] == "../../outside-authority":
            raise AssertionError("malformed digest reached a filesystem read")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(
        journals,
        "_read_bound_regular_entry",
        reject_blob_read,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert journal.active
    assert target.read_bytes() == b"provider mutation\n"


def test_directory_install_rejects_post_mkdir_external_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory metadata is applied only to the inode proven after mkdir."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    scope_file = repo / "scope" / "item.txt"
    scope_file.parent.mkdir()
    scope_file.write_bytes(b"directory baseline\n")
    _git(repo, "add", "scope/item.txt")
    _git(repo, "commit", "-qm", "directory baseline")
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "directory" / "mutations",
        slice_id="restore-directory-race",
        scope=("scope/",),
        sequence=1,
    )
    shutil.rmtree(repo / "scope")

    external_directory = tmp_path / "external-directory"
    external_directory.mkdir()
    external_target = external_directory / "outside.txt"
    external_target.write_bytes(b"outside authority\n")
    external_directory.chmod(0o750)
    external_before = external_directory.lstat()
    external_target_before = external_target.lstat()
    parked_created = tmp_path / "parked-created-directory"
    original_open_created = journals._open_authority_directory_at
    swapped = False
    moved_external_paths: list[Path] = []

    def swap_before_created_directory_open(
        parent_fd: int,
        name: str,
        *,
        label: str,
    ) -> tuple[int, os.stat_result]:
        nonlocal swapped
        if not swapped and name.startswith(".crucible-restore-dir-"):
            private_path = Path(
                journals._authority_descriptor_cwd(parent_fd)
            ) / name
            private_path.rename(parked_created)
            external_directory.rename(private_path)
            moved_external_paths.append(repo / name)
            swapped = True
        return original_open_created(
            parent_fd,
            name,
            label=label,
        )

    monkeypatch.setattr(
        journals,
        "_open_authority_directory_at",
        swap_before_created_directory_open,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert journal.active
    moved_external = moved_external_paths[0]
    moved_after = moved_external.lstat()
    moved_target = moved_external / "outside.txt"
    moved_target_after = moved_target.lstat()
    assert moved_target.read_bytes() == b"outside authority\n"
    assert (
        moved_after.st_dev,
        moved_after.st_ino,
        moved_after.st_mode,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
    )
    assert (
        moved_target_after.st_dev,
        moved_target_after.st_ino,
        moved_target_after.st_mode,
        moved_target_after.st_size,
    ) == (
        external_target_before.st_dev,
        external_target_before.st_ino,
        external_target_before.st_mode,
        external_target_before.st_size,
    )


def test_symlink_xattr_restore_fails_before_replacing_live_workspace_entry(
    tmp_path: Path,
) -> None:
    """Unsupported inode-bound symlink xattrs retain the pre-restore image."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    link = repo / "scope-link"
    link.symlink_to("dir/victim.txt")
    _git(repo, "add", "scope-link")
    _git(repo, "commit", "-qm", "symlink baseline")
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "symlink" / "mutations",
        slice_id="restore-symlink-xattrs",
        scope=("scope-link",),
        sequence=1,
    )
    journal._manifest["entries"]["scope-link"]["xattrs"] = {
        "user.crucible-test": "626173656c696e65",
    }
    journal._write()
    link.unlink()
    link.symlink_to("nest/keep.txt")

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert journal.active
    assert link.is_symlink()
    assert os.readlink(link) == "nest/keep.txt"


def test_symlink_xattr_install_rejects_before_creating_workspace_entry(
    tmp_path: Path,
) -> None:
    """The generic installer validates unsupported metadata before mutation."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    root_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
    try:
        root = os.fstat(root_fd)
        with pytest.raises(
            journals.MutationJournalError,
            match="symlink xattr restoration is unsupported",
        ):
            journals._install_bound_workspace_entry(
                root_fd,
                "unsupported-link",
                expected_parent_chain=((root.st_dev, root.st_ino),),
                kind="symlink",
                mode=0o777,
                xattrs={"user.crucible-test": "626173656c696e65"},
                target="target.txt",
            )
    finally:
        os.close(root_fd)

    assert not os.path.lexists(repo / "unsupported-link")


def test_parent_metadata_restore_rejects_external_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured parent identity is reproved before descriptor metadata writes."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "parent" / "mutations",
        slice_id="restore-parent-metadata-race",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")

    external_parent = tmp_path / "external-parent-metadata"
    external_parent.mkdir()
    external_target = external_parent / "outside.txt"
    external_target.write_bytes(b"outside authority\n")
    external_parent.chmod(0o700)
    external_before = external_parent.lstat()
    external_target_before = external_target.lstat()
    parked_parent = tmp_path / "parked-baseline-parent"
    original_open = journals._open_authority_directory_at
    swapped = False

    def swap_on_parent_metadata_reopen(
        parent_fd: int,
        name: str,
        *,
        label: str,
    ) -> tuple[int, os.stat_result]:
        nonlocal swapped
        if (
            not swapped
            and name == "dir"
            and "workspace directory metadata" in label
        ):
            (repo / "dir").rename(parked_parent)
            external_parent.rename(repo / "dir")
            swapped = True
        return original_open(parent_fd, name, label=label)

    monkeypatch.setattr(
        journals,
        "_open_authority_directory_at",
        swap_on_parent_metadata_reopen,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert journal.active
    moved_external = repo / "dir"
    moved_after = moved_external.lstat()
    moved_target = moved_external / "outside.txt"
    moved_target_after = moved_target.lstat()
    assert moved_target.read_bytes() == b"outside authority\n"
    assert (
        moved_after.st_dev,
        moved_after.st_ino,
        moved_after.st_mode,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
    )
    assert (
        moved_target_after.st_dev,
        moved_target_after.st_ino,
        moved_target_after.st_mode,
        moved_target_after.st_size,
    ) == (
        external_target_before.st_dev,
        external_target_before.st_ino,
        external_target_before.st_mode,
        external_target_before.st_size,
    )


def test_tracked_magic_filename_uses_literal_git_tree_lookup(tmp_path: Path) -> None:
    """Manifest paths are never reinterpreted as Git pathspec magic."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    magic = repo / ":!x"
    magic.write_bytes(b"literal baseline\n")
    _git(repo, "--literal-pathspecs", "add", "--", ":!x")
    _git(repo, "commit", "-qm", "literal pathspec baseline")
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "literal" / "mutations",
        slice_id="literal-git-pathspec",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    magic.write_bytes(b"provider mutation\n")

    assert journal.detected_scope_escapes() == (":!x",)
    journal.record_scope_escapes((":!x",))
    assert journal.restore(full=False, _defer_terminal_cleanup=True)
    assert magic.read_bytes() == b"literal baseline\n"


def test_bounded_git_reader_kills_output_overrun(tmp_path: Path) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    with pytest.raises(
        journals.MutationJournalError,
        match="output byte ceiling",
    ):
        journals._run_bound_git_bounded_stdout(
            repo,
            (
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 4097)",
            ),
            max_stdout_bytes=4096,
            label="bounded-reader-test",
            timeout_s=5,
        )


def test_bounded_git_reader_kills_stalled_child(tmp_path: Path) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    with pytest.raises(
        journals.MutationJournalError,
        match="exceeded its 0.1s timeout",
    ):
        journals._run_bound_git_bounded_stdout(
            repo,
            (sys.executable, "-c", "import time; time.sleep(30)"),
            max_stdout_bytes=1,
            label="bounded-reader-timeout-test",
            timeout_s=0.1,
        )


@pytest.mark.parametrize("provider_created", (False, True))
def test_absence_reproof_rejects_parent_swap_before_restore_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_created: bool,
) -> None:
    """Both initial and post-delete ENOENT require a final parent reproof."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    relative = "dir/missing.txt"
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "absence" / "mutations",
        slice_id=f"absence-reproof-{provider_created}",
        scope=(relative,),
        sequence=1,
    )
    target = repo / relative
    if provider_created:
        target.write_bytes(b"provider-created\n")

    external_parent = tmp_path / "external-absence-parent"
    external_parent.mkdir()
    external_target = external_parent / "outside.txt"
    external_target.write_bytes(b"outside authority\n")
    external_before = external_target.lstat()
    parked_parent = tmp_path / "parked-absence-parent"
    original_assert = journals._assert_settlement_parent_binding
    calls = 0
    swapped = False

    def swap_at_final_absence_reproof(
        root_fd: int,
        rendered: str,
        parent_fd: int,
        held_chain: tuple[tuple[int, int], ...],
        **kwargs: bool,
    ) -> None:
        nonlocal calls, swapped
        if rendered == relative:
            calls += 1
            if not provider_created and calls == 2 and not swapped:
                (repo / "dir").rename(parked_parent)
                (repo / "dir").symlink_to(
                    external_parent,
                    target_is_directory=True,
                )
                swapped = True
                original_assert(
                    root_fd,
                    rendered,
                    parent_fd,
                    held_chain,
                    **kwargs,
                )
                return
        original_assert(
            root_fd,
            rendered,
            parent_fd,
            held_chain,
            **kwargs,
        )
        if (
            provider_created
            and rendered == relative
            and not swapped
            and not target.exists()
        ):
            (repo / "dir").rename(parked_parent)
            (repo / "dir").symlink_to(
                external_parent,
                target_is_directory=True,
            )
            swapped = True

    monkeypatch.setattr(
        journals,
        "_assert_settlement_parent_binding",
        swap_at_final_absence_reproof,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert journal.active
    external_after = external_target.lstat()
    assert external_target.read_bytes() == b"outside authority\n"
    assert (
        external_after.st_dev,
        external_after.st_ino,
        external_after.st_mode,
        external_after.st_size,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
        external_before.st_size,
    )


def test_no_replace_directory_collision_cleans_private_staging(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    parent = tmp_path / "parent"
    parent.mkdir()
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    directory_fd: int | None = None
    try:
        private_name, directory_fd, expected = (
            journals._create_private_bound_workspace_directory(
                parent_fd,
                label="collision-cleanup-test",
            )
        )
        final = parent / "final"
        final.mkdir()
        final_before = final.lstat()
        with pytest.raises(
            journals.MutationJournalError,
            match="target appeared before publication",
        ):
            journals._publish_private_bound_workspace_directory(
                parent_fd,
                private_name,
                "final",
                directory_fd,
                expected,
                label="collision-cleanup-test",
            )
        assert not (parent / private_name).exists()
        final_after = final.lstat()
        assert (final_after.st_dev, final_after.st_ino) == (
            final_before.st_dev,
            final_before.st_ino,
        )
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def test_directory_collision_never_recursively_deletes_injected_external_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collision cleanup removes only staging proven to remain empty."""

    from crucible.workflows import _mutation_journal as journals

    parent = tmp_path / "parent"
    parent.mkdir()
    external = tmp_path / "outside-authority"
    external.write_bytes(b"outside authority\n")
    external.chmod(0o640)
    external_before = external.lstat()
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    directory_fd: int | None = None
    try:
        private_name, directory_fd, expected = (
            journals._create_private_bound_workspace_directory(
                parent_fd,
                label="injected-collision-test",
            )
        )
        original_publish = journals._rename_bound_noreplace
        injected = False

        def inject_hardlink_then_collide(
            directory_descriptor: int,
            source_name: str,
            destination_name: str,
            *,
            label: str,
        ) -> None:
            nonlocal injected
            assert directory_descriptor == parent_fd
            assert source_name == private_name
            assert destination_name == "final"
            os.link(external, "injected", dst_dir_fd=directory_fd)
            injected = True
            raise journals.MutationJournalError(
                f"workspace restore target appeared before publication: {label}"
            )

        monkeypatch.setattr(
            journals,
            "_rename_bound_noreplace",
            inject_hardlink_then_collide,
        )
        with pytest.raises(
            journals.MutationJournalError,
            match="private staging cleanup failed",
        ):
            journals._publish_private_bound_workspace_directory(
                parent_fd,
                private_name,
                "final",
                directory_fd,
                expected,
                label="injected-collision-test",
            )
        assert injected
        staging = parent / private_name
        injected_path = staging / "injected"
        assert staging.is_dir()
        assert injected_path.read_bytes() == b"outside authority\n"
        injected_after = injected_path.lstat()
        external_after = external.lstat()
        assert (
            injected_after.st_dev,
            injected_after.st_ino,
        ) == (
            external_before.st_dev,
            external_before.st_ino,
        )
        assert (
            external_after.st_dev,
            external_after.st_ino,
            external_after.st_mode,
            external_after.st_size,
        ) == (
            external_before.st_dev,
            external_before.st_ino,
            external_before.st_mode,
            external_before.st_size,
        )
        monkeypatch.setattr(
            journals,
            "_rename_bound_noreplace",
            original_publish,
        )
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def test_successful_directory_publication_retracts_an_injected_external_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The staging-inode watch covers the successful no-replace rename."""

    from crucible.workflows import _mutation_journal as journals

    parent = tmp_path / "parent"
    parent.mkdir()
    external = tmp_path / "outside-authority"
    external.write_bytes(b"outside authority\n")
    external.chmod(0o640)
    external_before = external.lstat()
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    directory_fd: int | None = None
    try:
        private_name, directory_fd, expected = (
            journals._create_private_bound_workspace_directory(
                parent_fd,
                label="successful-injection-test",
            )
        )
        original_rename = journals._rename_bound_noreplace
        injected = False

        def publish_then_inject_hardlink(
            directory_descriptor: int,
            source_name: str,
            destination_name: str,
            *,
            label: str,
        ) -> None:
            nonlocal injected
            original_rename(
                directory_descriptor,
                source_name,
                destination_name,
                label=label,
            )
            if source_name == private_name and destination_name == "final":
                os.link(external, "injected", dst_dir_fd=directory_fd)
                injected = True

        monkeypatch.setattr(
            journals,
            "_rename_bound_noreplace",
            publish_then_inject_hardlink,
        )
        with pytest.raises(
            journals.MutationJournalError,
            match="changed during publication",
        ):
            journals._publish_private_bound_workspace_directory(
                parent_fd,
                private_name,
                "final",
                directory_fd,
                expected,
                label="successful-injection-test",
            )
        assert injected
        assert not (parent / "final").exists()
        returned_staging = parent / private_name
        injected_path = returned_staging / "injected"
        assert returned_staging.is_dir()
        assert injected_path.read_bytes() == b"outside authority\n"
        injected_after = injected_path.lstat()
        external_after = external.lstat()
        assert (
            injected_after.st_dev,
            injected_after.st_ino,
        ) == (
            external_before.st_dev,
            external_before.st_ino,
        )
        assert (
            external_after.st_dev,
            external_after.st_ino,
            external_after.st_mode,
            external_after.st_size,
        ) == (
            external_before.st_dev,
            external_before.st_ino,
            external_before.st_mode,
            external_before.st_size,
        )
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def test_successful_directory_publication_retracts_a_swapped_staging_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement at the private spelling is never left published."""

    from crucible.workflows import _mutation_journal as journals

    parent = tmp_path / "parent"
    parent.mkdir()
    external = tmp_path / "outside-directory"
    external.mkdir()
    external_target = external / "outside.txt"
    external_target.write_bytes(b"outside authority\n")
    external.chmod(0o750)
    external_before = external.lstat()
    external_target_before = external_target.lstat()
    parked_original = tmp_path / "parked-original-staging"
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    directory_fd: int | None = None
    try:
        private_name, directory_fd, expected = (
            journals._create_private_bound_workspace_directory(
                parent_fd,
                label="successful-staging-swap-test",
            )
        )
        original_rename = journals._rename_bound_noreplace
        swapped = False

        def swap_staging_then_publish(
            directory_descriptor: int,
            source_name: str,
            destination_name: str,
            *,
            label: str,
        ) -> None:
            nonlocal swapped
            if source_name == private_name and destination_name == "final":
                (parent / private_name).rename(parked_original)
                external.rename(parent / private_name)
                swapped = True
            original_rename(
                directory_descriptor,
                source_name,
                destination_name,
                label=label,
            )

        monkeypatch.setattr(
            journals,
            "_rename_bound_noreplace",
            swap_staging_then_publish,
        )
        with pytest.raises(
            journals.MutationJournalError,
            match="published workspace restore directory changed",
        ):
            journals._publish_private_bound_workspace_directory(
                parent_fd,
                private_name,
                "final",
                directory_fd,
                expected,
                label="successful-staging-swap-test",
            )
        assert swapped
        assert not (parent / "final").exists()
        returned_external = parent / private_name
        returned_after = returned_external.lstat()
        returned_target = returned_external / "outside.txt"
        returned_target_after = returned_target.lstat()
        assert parked_original.is_dir()
        assert returned_target.read_bytes() == b"outside authority\n"
        assert (
            returned_after.st_dev,
            returned_after.st_ino,
            returned_after.st_mode,
        ) == (
            external_before.st_dev,
            external_before.st_ino,
            external_before.st_mode,
        )
        assert (
            returned_target_after.st_dev,
            returned_target_after.st_ino,
            returned_target_after.st_mode,
            returned_target_after.st_size,
        ) == (
            external_target_before.st_dev,
            external_target_before.st_ino,
            external_target_before.st_mode,
            external_target_before.st_size,
        )
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def test_restore_rejects_parent_swap_between_ensure_and_target_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The target-open chain must equal the identities returned by ensure."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "ensure" / "mutations",
        slice_id="restore-ensure-target-open-race",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    target = repo / "dir" / "victim.txt"
    target.write_bytes(b"provider mutation\n")
    target_before = target.lstat()

    external_parent = tmp_path / "external-ensure-parent"
    external_parent.mkdir()
    external_target = external_parent / "victim.txt"
    external_target.write_bytes(b"outside authority\n")
    external_target.chmod(0o640)
    external_before = external_target.lstat()
    parked_parent = tmp_path / "parked-ensure-parent"
    original_open = journals._open_settlement_parent_if_present
    swapped = False

    @contextmanager
    def swap_before_target_parent_open(
        root_fd: int,
        relative: str,
        *,
        allow_terminal_vcs_metadata: bool = False,
    ):
        nonlocal swapped
        if relative == "dir/victim.txt" and not swapped:
            (repo / "dir").rename(parked_parent)
            external_parent.rename(repo / "dir")
            swapped = True
        with original_open(
            root_fd,
            relative,
            allow_terminal_vcs_metadata=allow_terminal_vcs_metadata,
        ) as binding:
            yield binding

    monkeypatch.setattr(
        journals,
        "_open_settlement_parent_if_present",
        swap_before_target_parent_open,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert journal.active
    parked_target = parked_parent / "victim.txt"
    parked_after = parked_target.lstat()
    assert parked_target.read_bytes() == b"provider mutation\n"
    assert (parked_after.st_dev, parked_after.st_ino) == (
        target_before.st_dev,
        target_before.st_ino,
    )
    external_after = (repo / "dir" / "victim.txt").lstat()
    assert (repo / "dir" / "victim.txt").read_bytes() == b"outside authority\n"
    assert (
        external_after.st_dev,
        external_after.st_ino,
        external_after.st_mode,
        external_after.st_size,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
        external_before.st_size,
    )


def test_tracked_absence_rejects_parent_appearance_between_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing tracked subtree cannot appear between absence probes."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    donor = repo / "donor"
    donor.mkdir()
    (donor / "child.py").write_text("tracked donor\n", encoding="utf-8")
    _git(repo, "add", "donor/child.py")
    _git(repo, "commit", "-qm", "add donor")
    shutil.rmtree(donor)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "absence" / "mutations",
        slice_id="restore-tracked-absence-race",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    authority = journal._restore_directory_modes(repo)
    assert "donor" not in authority
    assert journal._manifest["tracked_path_metadata"]["donor/child.py"][
        "kind"
    ] == "missing"
    original_open = journals._open_settlement_parent_if_present
    target = "donor/child.py"
    target_calls = 0
    injected = False

    @contextmanager
    def create_parent_on_second_probe(
        root_fd: int,
        relative: str,
        *,
        allow_terminal_vcs_metadata: bool = False,
    ):
        nonlocal target_calls, injected
        if relative == target:
            target_calls += 1
            if target_calls == 2:
                donor.mkdir()
                (donor / "foreign.txt").write_text(
                    "foreign\n",
                    encoding="utf-8",
                )
                injected = True
        with original_open(
            root_fd,
            relative,
            allow_terminal_vcs_metadata=allow_terminal_vcs_metadata,
        ) as binding:
            yield binding

    monkeypatch.setattr(
        journals,
        "_open_settlement_parent_if_present",
        create_parent_on_second_probe,
    )

    with pytest.raises(
        journals.MutationJournalError,
        match=r"workspace absence-token parent appeared: 'donor/child\.py'",
    ):
        journal._restore_tracked_path_metadata(repo, authority)

    assert injected
    assert target_calls == 2
    assert (donor / "foreign.txt").read_text(encoding="utf-8") == "foreign\n"
    assert not (donor / "child.py").exists()
    assert journal.active


def test_absent_attribute_policy_rejects_leaf_appearance_between_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A racing policy leaf is preserved instead of mistaken for absence."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    donor = repo / "donor"
    donor.mkdir()
    policy = donor / ".gitattributes"
    policy.write_bytes(b"* text=auto eol=lf\n")
    _git(repo, "add", "donor/.gitattributes")
    _git(repo, "commit", "-qm", "add donor attribute policy")
    shutil.rmtree(donor)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "policy" / "mutations",
        slice_id="restore-absent-attribute-policy-race",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    donor.mkdir()
    donor_metadata = donor.stat()
    foreign = b"foreign policy bytes\n"
    original_stat = journals.os.stat
    injected = False

    def inject_policy_after_first_absence(
        path,
        *,
        dir_fd=None,
        follow_symlinks=True,
    ):
        nonlocal injected
        if (
            path == ".gitattributes"
            and dir_fd is not None
            and (
                os.fstat(dir_fd).st_dev,
                os.fstat(dir_fd).st_ino,
            )
            == (donor_metadata.st_dev, donor_metadata.st_ino)
            and not injected
        ):
            policy.write_bytes(foreign)
            injected = True
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                path,
            )
        return original_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(journals.os, "stat", inject_policy_after_first_absence)

    with pytest.raises(
        journals.MutationJournalError,
        match=(
            "Git attribute policy appeared during absence proof: "
            "donor/.gitattributes"
        ),
    ):
        journal._restore_attribute_policy(repo)

    assert injected
    assert policy.read_bytes() == foreign
    assert journal.active
    monkeypatch.undo()
    policy.unlink()
    assert journal.restore()
    assert not donor.exists()


def test_restore_rejects_parent_swap_between_directory_and_child_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every child install remains bound to its newly installed ancestors."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    scope_file = repo / "scope" / "item.txt"
    scope_file.parent.mkdir()
    scope_file.write_bytes(b"directory baseline\n")
    _git(repo, "add", "scope/item.txt")
    _git(repo, "commit", "-qm", "directory baseline")
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "child" / "mutations",
        slice_id="restore-directory-child-race",
        scope=("scope/",),
        sequence=1,
    )
    shutil.rmtree(repo / "scope")

    external_parent = tmp_path / "external-child-parent"
    external_parent.mkdir()
    external_target = external_parent / "outside.txt"
    external_target.write_bytes(b"outside authority\n")
    external_parent.chmod(0o750)
    external_before = external_parent.lstat()
    external_target_before = external_target.lstat()
    parked_installed = tmp_path / "parked-installed-scope"
    original_open = journals._open_settlement_parent_if_present
    swapped = False

    @contextmanager
    def swap_before_child_parent_open(
        root_fd: int,
        relative: str,
        *,
        allow_terminal_vcs_metadata: bool = False,
    ):
        nonlocal swapped
        if relative == "scope/item.txt" and not swapped:
            (repo / "scope").rename(parked_installed)
            external_parent.rename(repo / "scope")
            swapped = True
        with original_open(
            root_fd,
            relative,
            allow_terminal_vcs_metadata=allow_terminal_vcs_metadata,
        ) as binding:
            yield binding

    monkeypatch.setattr(
        journals,
        "_open_settlement_parent_if_present",
        swap_before_child_parent_open,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert journal.active
    assert parked_installed.is_dir()
    assert not (parked_installed / "item.txt").exists()
    moved_external = repo / "scope"
    moved_after = moved_external.lstat()
    moved_target = moved_external / "outside.txt"
    moved_target_after = moved_target.lstat()
    assert not (moved_external / "item.txt").exists()
    assert moved_target.read_bytes() == b"outside authority\n"
    assert (
        moved_after.st_dev,
        moved_after.st_ino,
        moved_after.st_mode,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
    )
    assert (
        moved_target_after.st_dev,
        moved_target_after.st_ino,
        moved_target_after.st_mode,
        moved_target_after.st_size,
    ) == (
        external_target_before.st_dev,
        external_target_before.st_ino,
        external_target_before.st_mode,
        external_target_before.st_size,
    )


def test_restore_rejects_shared_parent_swap_between_separate_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later root cannot bless a different shared-parent identity locally."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    first = repo / "dir" / "a.txt"
    second = repo / "dir" / "b.txt"
    first.write_bytes(b"first baseline\n")
    second.write_bytes(b"second baseline\n")
    _git(repo, "add", "dir/a.txt", "dir/b.txt")
    _git(repo, "commit", "-qm", "shared parent baseline")
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "roots" / "mutations",
        slice_id="restore-cross-root-parent-race",
        scope=("dir/a.txt", "dir/b.txt"),
        sequence=1,
    )
    first.write_bytes(b"first provider mutation\n")
    second.write_bytes(b"second provider mutation\n")

    external_parent = tmp_path / "external-cross-root-parent"
    external_parent.mkdir()
    external_first = external_parent / "a.txt"
    external_second = external_parent / "b.txt"
    external_sentinel = external_parent / "outside.txt"
    external_first.write_bytes(b"outside first\n")
    external_second.write_bytes(b"outside second\n")
    external_sentinel.write_bytes(b"outside authority\n")
    external_second.chmod(0o640)
    external_second_before = external_second.lstat()
    external_sentinel_before = external_sentinel.lstat()
    parked_parent = tmp_path / "parked-cross-root-parent"
    original_restore = journal._restore_snapshot_root
    restored_roots: list[str] = []
    swapped = False

    def swap_after_first_root(
        cwd: Path,
        relative: str,
        *,
        expected_parent_bindings=None,
    ):
        nonlocal swapped
        bindings = original_restore(
            cwd,
            relative,
            expected_parent_bindings=expected_parent_bindings,
        )
        restored_roots.append(relative)
        if not swapped:
            (repo / "dir").rename(parked_parent)
            external_parent.rename(repo / "dir")
            swapped = True
        return bindings

    monkeypatch.setattr(journal, "_restore_snapshot_root", swap_after_first_root)

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert len(restored_roots) == 1
    assert journal.active
    moved_external_second = repo / "dir" / "b.txt"
    moved_external_sentinel = repo / "dir" / "outside.txt"
    external_second_after = moved_external_second.lstat()
    external_sentinel_after = moved_external_sentinel.lstat()
    assert moved_external_second.read_bytes() == b"outside second\n"
    assert moved_external_sentinel.read_bytes() == b"outside authority\n"
    assert (
        external_second_after.st_dev,
        external_second_after.st_ino,
        external_second_after.st_mode,
        external_second_after.st_size,
    ) == (
        external_second_before.st_dev,
        external_second_before.st_ino,
        external_second_before.st_mode,
        external_second_before.st_size,
    )
    assert (
        external_sentinel_after.st_dev,
        external_sentinel_after.st_ino,
        external_sentinel_after.st_mode,
        external_sentinel_after.st_size,
    ) == (
        external_sentinel_before.st_dev,
        external_sentinel_before.st_ino,
        external_sentinel_before.st_mode,
        external_sentinel_before.st_size,
    )


def test_cross_root_parent_check_precedes_deeper_parent_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure cannot mkdir below a shared prefix whose identity was replaced."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    nested = repo / "dir" / "nested"
    nested.mkdir()
    first = nested / "a.txt"
    second = nested / "b.txt"
    first.write_bytes(b"first baseline\n")
    second.write_bytes(b"second baseline\n")
    _git(repo, "add", "dir/nested/a.txt", "dir/nested/b.txt")
    _git(repo, "commit", "-qm", "nested shared parent baseline")
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "nested-roots" / "mutations",
        slice_id="restore-cross-root-mkdir-race",
        scope=("dir/nested/a.txt", "dir/nested/b.txt"),
        sequence=1,
    )
    first.write_bytes(b"first provider mutation\n")
    second.write_bytes(b"second provider mutation\n")

    external_parent = tmp_path / "external-cross-root-mkdir-parent"
    external_parent.mkdir()
    external_sentinel = external_parent / "outside.txt"
    external_sentinel.write_bytes(b"outside authority\n")
    external_sentinel.chmod(0o640)
    external_parent_before = external_parent.lstat()
    external_sentinel_before = external_sentinel.lstat()
    parked_parent = tmp_path / "parked-cross-root-mkdir-parent"
    original_restore = journal._restore_snapshot_root
    restored_roots: list[str] = []
    swapped = False

    def swap_after_first_root(
        cwd: Path,
        relative: str,
        *,
        expected_parent_bindings=None,
    ):
        nonlocal swapped
        bindings = original_restore(
            cwd,
            relative,
            expected_parent_bindings=expected_parent_bindings,
        )
        restored_roots.append(relative)
        if not swapped:
            (repo / "dir").rename(parked_parent)
            external_parent.rename(repo / "dir")
            swapped = True
        return bindings

    monkeypatch.setattr(journal, "_restore_snapshot_root", swap_after_first_root)

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert len(restored_roots) == 1
    assert journal.active
    moved_external = repo / "dir"
    moved_external_after = moved_external.lstat()
    moved_sentinel = moved_external / "outside.txt"
    moved_sentinel_after = moved_sentinel.lstat()
    assert not (moved_external / "nested").exists()
    assert moved_sentinel.read_bytes() == b"outside authority\n"
    assert (
        moved_external_after.st_dev,
        moved_external_after.st_ino,
        moved_external_after.st_mode,
    ) == (
        external_parent_before.st_dev,
        external_parent_before.st_ino,
        external_parent_before.st_mode,
    )
    assert (
        moved_sentinel_after.st_dev,
        moved_sentinel_after.st_ino,
        moved_sentinel_after.st_mode,
        moved_sentinel_after.st_size,
    ) == (
        external_sentinel_before.st_dev,
        external_sentinel_before.st_ino,
        external_sentinel_before.st_mode,
        external_sentinel_before.st_size,
    )


def test_cross_root_absent_restore_never_deletes_a_swapped_external_victim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline absence remains constrained by an earlier root's parent."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    first = repo / "dir" / "a.txt"
    first.write_bytes(b"first baseline\n")
    _git(repo, "add", "dir/a.txt")
    _git(repo, "commit", "-qm", "absent sibling baseline")
    absent_relative = "dir/missing.txt"
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "absent-root" / "mutations",
        slice_id="restore-cross-root-absence-race",
        scope=("dir/a.txt", absent_relative),
        sequence=1,
    )
    first.write_bytes(b"first provider mutation\n")
    (repo / absent_relative).write_bytes(b"provider-created victim\n")

    external_parent = tmp_path / "external-cross-root-absence-parent"
    external_parent.mkdir()
    external_victim = external_parent / "missing.txt"
    external_victim.write_bytes(b"outside victim\n")
    external_victim.chmod(0o640)
    external_sentinel = external_parent / "outside.txt"
    external_sentinel.write_bytes(b"outside authority\n")
    external_victim_before = external_victim.lstat()
    external_sentinel_before = external_sentinel.lstat()
    parked_parent = tmp_path / "parked-cross-root-absence-parent"
    original_restore = journal._restore_snapshot_root
    restored_roots: list[str] = []
    swapped = False

    def swap_after_first_root(
        cwd: Path,
        relative: str,
        *,
        expected_parent_bindings=None,
    ):
        nonlocal swapped
        bindings = original_restore(
            cwd,
            relative,
            expected_parent_bindings=expected_parent_bindings,
        )
        restored_roots.append(relative)
        if not swapped:
            (repo / "dir").rename(parked_parent)
            external_parent.rename(repo / "dir")
            swapped = True
        return bindings

    monkeypatch.setattr(journal, "_restore_snapshot_root", swap_after_first_root)

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert restored_roots == ["dir/a.txt"]
    assert journal.active
    moved_external_victim = repo / absent_relative
    moved_external_sentinel = repo / "dir" / "outside.txt"
    external_victim_after = moved_external_victim.lstat()
    external_sentinel_after = moved_external_sentinel.lstat()
    assert moved_external_victim.read_bytes() == b"outside victim\n"
    assert moved_external_sentinel.read_bytes() == b"outside authority\n"
    assert (
        external_victim_after.st_dev,
        external_victim_after.st_ino,
        external_victim_after.st_mode,
        external_victim_after.st_size,
    ) == (
        external_victim_before.st_dev,
        external_victim_before.st_ino,
        external_victim_before.st_mode,
        external_victim_before.st_size,
    )
    assert (
        external_sentinel_after.st_dev,
        external_sentinel_after.st_ino,
        external_sentinel_after.st_mode,
        external_sentinel_after.st_size,
    ) == (
        external_sentinel_before.st_dev,
        external_sentinel_before.st_ino,
        external_sentinel_before.st_mode,
        external_sentinel_before.st_size,
    )


@pytest.mark.parametrize("provider_parent_missing", (False, True))
def test_absence_first_root_binds_parent_before_later_materialized_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_parent_missing: bool,
) -> None:
    """An absence-first root contributes identity or missing-token authority."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    materialized_relative = "dir/z.txt"
    materialized = repo / materialized_relative
    materialized.write_bytes(b"materialized baseline\n")
    _git(repo, "add", materialized_relative)
    _git(repo, "commit", "-qm", "absence-first baseline")
    absent_relative = "dir/a-missing.txt"
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "absence-first" / "mutations",
        slice_id=f"restore-absence-first-parent-race-{provider_parent_missing}",
        scope=(absent_relative, materialized_relative),
        sequence=1,
    )
    if provider_parent_missing:
        shutil.rmtree(repo / "dir")
    else:
        (repo / absent_relative).write_bytes(b"provider-created absence\n")
        materialized.write_bytes(b"materialized provider mutation\n")

    external_parent = tmp_path / "external-absence-first-parent"
    external_parent.mkdir()
    external_victim = external_parent / "z.txt"
    external_victim.write_bytes(b"outside victim\n")
    external_victim.chmod(0o640)
    external_sentinel = external_parent / "outside.txt"
    external_sentinel.write_bytes(b"outside authority\n")
    external_victim_before = external_victim.lstat()
    external_sentinel_before = external_sentinel.lstat()
    parked_parent = tmp_path / "parked-absence-first-parent"
    original_restore = journal._restore_snapshot_root
    restored_roots: list[str] = []
    swapped = False

    def swap_after_first_root(
        cwd: Path,
        relative: str,
        *,
        expected_parent_bindings=None,
    ):
        nonlocal swapped
        bindings = original_restore(
            cwd,
            relative,
            expected_parent_bindings=expected_parent_bindings,
        )
        restored_roots.append(relative)
        if not swapped:
            if (repo / "dir").exists():
                (repo / "dir").rename(parked_parent)
            external_parent.rename(repo / "dir")
            swapped = True
        return bindings

    monkeypatch.setattr(journal, "_restore_snapshot_root", swap_after_first_root)

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert restored_roots == [absent_relative]
    assert journal.active
    moved_external_victim = repo / materialized_relative
    moved_external_sentinel = repo / "dir" / "outside.txt"
    external_victim_after = moved_external_victim.lstat()
    external_sentinel_after = moved_external_sentinel.lstat()
    assert moved_external_victim.read_bytes() == b"outside victim\n"
    assert moved_external_sentinel.read_bytes() == b"outside authority\n"
    assert (
        external_victim_after.st_dev,
        external_victim_after.st_ino,
        external_victim_after.st_mode,
        external_victim_after.st_size,
    ) == (
        external_victim_before.st_dev,
        external_victim_before.st_ino,
        external_victim_before.st_mode,
        external_victim_before.st_size,
    )
    assert (
        external_sentinel_after.st_dev,
        external_sentinel_after.st_ino,
        external_sentinel_after.st_mode,
        external_sentinel_after.st_size,
    ) == (
        external_sentinel_before.st_dev,
        external_sentinel_before.st_ino,
        external_sentinel_before.st_mode,
        external_sentinel_before.st_size,
    )


def test_absence_token_allows_later_baseline_parent_reconstruction(
    tmp_path: Path,
) -> None:
    """A still-missing token may transition to its newly ensured identity."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    materialized_relative = "shared/z.txt"
    materialized = repo / materialized_relative
    materialized.parent.mkdir()
    materialized.write_bytes(b"materialized baseline\n")
    _git(repo, "add", materialized_relative)
    _git(repo, "commit", "-qm", "absence-token reconstruction baseline")
    absent_relative = "shared/a-missing.txt"
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "absence-token" / "mutations",
        slice_id="restore-absence-token-reconstruction",
        scope=(absent_relative, materialized_relative),
        sequence=1,
    )
    shutil.rmtree(repo / "shared")

    assert journal.restore(full=False, _defer_terminal_cleanup=True)
    assert journal.state == "rolled_back"
    assert not (repo / absent_relative).exists()
    assert materialized.read_bytes() == b"materialized baseline\n"


def test_nested_provider_created_absence_restores_baseline_topology(
    tmp_path: Path,
) -> None:
    """Observed nonbaseline parents bind races but receive no metadata."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    relative = "newdir/provider-created.txt"
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "nested-absence" / "mutations",
        slice_id="restore-nested-provider-created-absence",
        scope=(relative,),
        sequence=1,
    )
    target = repo / relative
    target.parent.mkdir()
    target.write_bytes(b"provider-created\n")

    assert journal.restore(full=False, _defer_terminal_cleanup=True)
    assert journal.state == "rolled_back"
    assert not target.exists()
    assert not target.parent.exists()


def test_directory_topology_cleanup_rejects_post_inventory_external_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty-addition cleanup carries the inventory identity into rmdir."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    relative = "newdir/provider-created.txt"
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "directory-swap" / "mutations",
        slice_id="restore-directory-post-inventory-swap",
        scope=(relative,),
        sequence=1,
    )
    target = repo / relative
    target.parent.mkdir()
    target.write_bytes(b"provider-created\n")
    provider_parent_before = target.parent.lstat()

    external = tmp_path / "external-directory-topology"
    external.mkdir()
    external_target = external / "outside.txt"
    external_target.write_bytes(b"outside authority\n")
    external.chmod(0o750)
    external_before = external.lstat()
    external_target_before = external_target.lstat()
    parked_provider = tmp_path / "parked-provider-directory"
    original_remove = journals._remove_bound_empty_workspace_directory
    swapped = False

    def swap_after_inventory_then_remove(
        root_fd: int,
        rendered: str,
        expected,
        authority,
    ) -> None:
        nonlocal swapped
        if not swapped and rendered == "newdir":
            (repo / "newdir").rename(parked_provider)
            external.rename(repo / "newdir")
            swapped = True
        original_remove(root_fd, rendered, expected, authority)

    monkeypatch.setattr(
        journals,
        "_remove_bound_empty_workspace_directory",
        swap_after_inventory_then_remove,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert journal.active
    parked_after = parked_provider.lstat()
    assert not any(parked_provider.iterdir())
    assert (parked_after.st_dev, parked_after.st_ino) == (
        provider_parent_before.st_dev,
        provider_parent_before.st_ino,
    )
    moved_external = repo / "newdir"
    moved_after = moved_external.lstat()
    moved_target = moved_external / "outside.txt"
    moved_target_after = moved_target.lstat()
    assert moved_target.read_bytes() == b"outside authority\n"
    assert (
        moved_after.st_dev,
        moved_after.st_ino,
        moved_after.st_mode,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
    )
    assert (
        moved_target_after.st_dev,
        moved_target_after.st_ino,
        moved_target_after.st_mode,
        moved_target_after.st_size,
    ) == (
        external_target_before.st_dev,
        external_target_before.st_ino,
        external_target_before.st_mode,
        external_target_before.st_size,
    )


def test_missing_baseline_directory_collision_preserves_external_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline directory creation publishes no-replace through held parent."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    baseline_empty = repo / "baseline-empty"
    baseline_empty.mkdir()
    baseline_empty.chmod(0o750)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "directory-create" / "mutations",
        slice_id="restore-directory-create-collision",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")
    baseline_empty.rmdir()

    external = tmp_path / "external-directory-create"
    external.mkdir()
    external_target = external / "outside.txt"
    external_target.write_bytes(b"outside authority\n")
    external.chmod(0o700)
    external_before = external.lstat()
    external_target_before = external_target.lstat()
    original_publish = journals._publish_private_bound_workspace_directory
    inserted = False

    def insert_before_baseline_publication(
        parent_fd: int,
        private_name: str,
        final_name: str,
        descriptor: int,
        expected: os.stat_result,
        *,
        label: str,
    ) -> None:
        nonlocal inserted
        if not inserted and final_name == "baseline-empty":
            external.rename(repo / "baseline-empty")
            inserted = True
        original_publish(
            parent_fd,
            private_name,
            final_name,
            descriptor,
            expected,
            label=label,
        )

    monkeypatch.setattr(
        journals,
        "_publish_private_bound_workspace_directory",
        insert_before_baseline_publication,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert inserted
    assert journal.active
    moved_external = repo / "baseline-empty"
    moved_after = moved_external.lstat()
    moved_target = moved_external / "outside.txt"
    moved_target_after = moved_target.lstat()
    assert moved_target.read_bytes() == b"outside authority\n"
    assert (
        moved_after.st_dev,
        moved_after.st_ino,
        moved_after.st_mode,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
    )
    assert (
        moved_target_after.st_dev,
        moved_target_after.st_ino,
        moved_target_after.st_mode,
        moved_target_after.st_size,
    ) == (
        external_target_before.st_dev,
        external_target_before.st_ino,
        external_target_before.st_mode,
        external_target_before.st_size,
    )


def test_directory_baseline_depth_bound_precedes_topology_removal(
    tmp_path: Path,
) -> None:
    """Malformed oversized authority fails before deleting an added directory."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "directory-bound" / "mutations",
        slice_id="restore-directory-manifest-bound",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")
    added = repo / "empty-provider-directory"
    added.mkdir()
    added_before = added.lstat()
    too_deep = "/".join(
        "d" for _ in range(journals._MUTATION_JOURNAL_MAX_DISCOVERY_DEPTH + 1)
    )
    journal._manifest["directory_modes"][too_deep] = 0o700
    journal._manifest["directory_xattrs"][too_deep] = {}
    journal._manifest["directory_identities"][too_deep] = {
        "device": 0,
        "inode": 0,
    }
    journal._write()

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert journal.active
    added_after = added.lstat()
    assert (added_after.st_dev, added_after.st_ino) == (
        added_before.st_dev,
        added_before.st_ino,
    )


@pytest.mark.parametrize("replacement_kind", ("symlink", "file"))
def test_absence_token_rejects_every_post_snapshot_parent_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    """A missing directory token means no symlink, file, or directory node."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    relative = "absent/missing.txt"
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "absence-node" / "mutations",
        slice_id=f"restore-absence-node-{replacement_kind}",
        scope=(relative,),
        sequence=1,
    )
    external = tmp_path / "external-absence-node"
    external.mkdir()
    external_target = external / "outside.txt"
    external_target.write_bytes(b"outside authority\n")
    external_before = external_target.lstat()
    original_restore = journal._restore_snapshot_root
    inserted = False

    def insert_after_absence_observation(
        cwd: Path,
        rendered: str,
        *,
        expected_parent_bindings=None,
    ):
        nonlocal inserted
        bindings = original_restore(
            cwd,
            rendered,
            expected_parent_bindings=expected_parent_bindings,
        )
        if not inserted:
            if replacement_kind == "symlink":
                (repo / "absent").symlink_to(
                    external,
                    target_is_directory=True,
                )
            else:
                (repo / "absent").write_bytes(b"outside-like parent\n")
            inserted = True
        return bindings

    monkeypatch.setattr(
        journal,
        "_restore_snapshot_root",
        insert_after_absence_observation,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert inserted
    assert journal.active
    external_after = external_target.lstat()
    assert external_target.read_bytes() == b"outside authority\n"
    assert (
        external_after.st_dev,
        external_after.st_ino,
        external_after.st_mode,
        external_after.st_size,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
        external_before.st_size,
    )
    if replacement_kind == "symlink":
        assert (repo / "absent").is_symlink()
        assert os.path.samefile(repo / "absent", external)
    else:
        assert (repo / "absent").read_bytes() == b"outside-like parent\n"


def test_directory_removal_failure_cleans_exact_private_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-quarantine failure leaves no workflow-owned empty reservation."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    relative = "newdir/provider-created.txt"
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "reservation" / "mutations",
        slice_id="restore-directory-reservation-cleanup",
        scope=(relative,),
        sequence=1,
    )
    target = repo / relative
    target.parent.mkdir()
    target.write_bytes(b"provider-created\n")
    parked = tmp_path / "parked-before-quarantine"
    original_rename = journals.os.rename
    injected = False

    def remove_source_before_quarantine(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ) -> None:
        nonlocal injected
        if (
            not injected
            and source == "newdir"
            and isinstance(destination, str)
            and destination.startswith(".crucible-cleanup-directory-")
        ):
            original_rename(
                source,
                parked,
                src_dir_fd=src_dir_fd,
            )
            injected = True
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(journals.os, "rename", remove_source_before_quarantine)

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert injected
    assert journal.active
    assert parked.is_dir()
    assert not any(parked.iterdir())
    assert not any(
        path.name.startswith(".crucible-cleanup-directory-")
        for path in repo.iterdir()
    )


def test_nonempty_directory_inspection_restores_quarantine_name(
    tmp_path: Path,
) -> None:
    """A multi-entry directory cannot be stranded under a quarantine name."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    target = repo / "newdir"
    target.mkdir(parents=True)
    (target / "a").write_text("a\n", encoding="utf-8")
    (target / "b").write_text("b\n", encoding="utf-8")
    target_before = target.stat()
    root_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority = journals._bound_workspace_directory_inventory(
            root_fd,
            excluded_roots=(),
            require_restore_control=True,
            logical_root=repo,
        )
        with pytest.raises(
            journals.MutationJournalError,
            match="directory addition is not empty",
        ):
            journals._remove_bound_empty_workspace_directory(
                root_fd,
                "newdir",
                authority["newdir"],
                authority,
            )
    finally:
        os.close(root_fd)

    assert (target / "a").read_text(encoding="utf-8") == "a\n"
    assert (target / "b").read_text(encoding="utf-8") == "b\n"
    target_after = target.stat()
    assert (target_after.st_dev, target_after.st_ino) == (
        target_before.st_dev,
        target_before.st_ino,
    )
    assert not any(
        path.name.startswith(".crucible-cleanup-directory-")
        for path in repo.iterdir()
    )


def test_directory_inspection_error_restores_quarantine_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An indeterminate nonempty probe restores the exact original name."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    target = repo / "newdir"
    target.mkdir(parents=True)
    (target / "payload").write_text("payload\n", encoding="utf-8")
    target_before = target.stat()
    original_scandir = journals.os.scandir
    injected = False

    class _FailingDirectoryEntries:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            raise OSError("injected directory inspection failure")

    def fail_target_inspection(directory):
        nonlocal injected
        if isinstance(directory, int):
            observed = os.fstat(directory)
            if (observed.st_dev, observed.st_ino) == (
                target_before.st_dev,
                target_before.st_ino,
            ):
                injected = True
                return _FailingDirectoryEntries()
        return original_scandir(directory)

    root_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority = journals._bound_workspace_directory_inventory(
            root_fd,
            excluded_roots=(),
            require_restore_control=True,
            logical_root=repo,
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(journals.os, "scandir", fail_target_inspection)
            with pytest.raises(
                journals.MutationJournalError,
                match="cannot inspect workspace directory addition 'newdir'",
            ):
                journals._remove_bound_empty_workspace_directory(
                    root_fd,
                    "newdir",
                    authority["newdir"],
                    authority,
                )
    finally:
        os.close(root_fd)

    assert injected
    assert (target / "payload").read_text(encoding="utf-8") == "payload\n"
    target_after = target.stat()
    assert (target_after.st_dev, target_after.st_ino) == (
        target_before.st_dev,
        target_before.st_ino,
    )
    assert not any(
        path.name.startswith(".crucible-cleanup-directory-")
        for path in repo.iterdir()
    )


def test_orphan_directory_baseline_fails_before_removing_live_addition(
    tmp_path: Path,
) -> None:
    """Baseline topology closure is validated before the first deletion."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "orphan" / "mutations",
        slice_id="restore-directory-orphan-baseline",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")
    added = repo / "empty-provider-directory"
    added.mkdir()
    added_before = added.lstat()
    journal._manifest["directory_modes"]["ghost/child"] = 0o700
    journal._manifest["directory_xattrs"]["ghost/child"] = {}
    journal._manifest["directory_identities"]["ghost/child"] = {
        "device": 0,
        "inode": 0,
    }
    journal._write()

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert journal.active
    added_after = added.lstat()
    assert (added_after.st_dev, added_after.st_ino) == (
        added_before.st_dev,
        added_before.st_ino,
    )


def test_prepublication_baseline_directory_failure_cleans_private_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty private baseline directory is retracted on metadata failure."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    baseline_empty = repo / "baseline-empty"
    baseline_empty.mkdir()
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "prepublish" / "mutations",
        slice_id="restore-directory-prepublication-cleanup",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")
    baseline_empty.rmdir()
    original_restore_xattrs = journals._restore_bound_descriptor_xattrs
    injected = False

    def fail_baseline_directory_metadata(
        descriptor: int,
        expected,
        *,
        label: str,
    ) -> None:
        nonlocal injected
        if not injected and label == "baseline-empty":
            injected = True
            raise journals.MutationJournalError(
                "injected baseline directory metadata failure"
            )
        original_restore_xattrs(descriptor, expected, label=label)

    monkeypatch.setattr(
        journals,
        "_restore_bound_descriptor_xattrs",
        fail_baseline_directory_metadata,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert injected
    assert journal.active
    assert not baseline_empty.exists()
    assert not any(
        path.name.startswith(".crucible-restore-dir-")
        or path.name.startswith(".crucible-cleanup-directory-")
        for path in repo.iterdir()
    )


@pytest.mark.parametrize(
    "failed_helper",
    ("_open_authority_directory_at", "_open_authority_path_directory_at"),
)
def test_missing_baseline_directory_open_failure_cleans_exact_private_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_helper: str,
) -> None:
    """Both first-bind frontiers retract only the just-created empty inode."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    baseline_empty = repo / "baseline-empty"
    baseline_empty.mkdir()
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "open" / "mutations",
        slice_id=f"restore-directory-{failed_helper}",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")
    baseline_empty.rmdir()
    original_open = getattr(journals, failed_helper)
    injected = False

    def fail_once(parent_fd: int, name: str, *, label: str):
        nonlocal injected
        if not injected and name.startswith(".crucible-restore-dir-"):
            injected = True
            raise journals.MutationJournalError(
                "injected private directory bind failure"
            )
        return original_open(parent_fd, name, label=label)

    monkeypatch.setattr(journals, failed_helper, fail_once)

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert injected
    assert journal.active
    assert not baseline_empty.exists()
    assert not any(
        path.name.startswith(".crucible-restore-dir-")
        or path.name.startswith(".crucible-cleanup-directory-")
        for path in repo.iterdir()
    )


def test_directory_removal_readable_reservation_open_failure_cleans_exact_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An O_PATH-bound quarantine is retracted if its readable open fails."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "open" / "mutations",
        slice_id="restore-directory-reservation-readable-open",
        scope=("newdir/provider-created.txt",),
        sequence=1,
    )
    target = repo / "newdir" / "provider-created.txt"
    target.parent.mkdir()
    target.write_bytes(b"provider-created\n")
    original_open = journals._open_authority_directory_at
    injected = False

    def fail_reservation_readable_open(
        parent_fd: int,
        name: str,
        *,
        label: str,
    ):
        nonlocal injected
        if (
            not injected
            and name.startswith(".crucible-cleanup-directory-")
        ):
            injected = True
            raise journals.MutationJournalError(
                "injected readable-open failure"
            )
        return original_open(parent_fd, name, label=label)

    monkeypatch.setattr(
        journals,
        "_open_authority_directory_at",
        fail_reservation_readable_open,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert injected
    assert journal.active
    assert (repo / "newdir").is_dir()
    assert not any(
        path.name.startswith(".crucible-cleanup-directory-")
        for path in repo.iterdir()
    )


def test_final_directory_inventory_rejects_recreated_removed_addition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name recreated after exact removal is caught before handoff."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "final" / "mutations",
        slice_id="restore-directory-final-addition-recreation",
        scope=("newdir/provider-created.txt",),
        sequence=1,
    )
    target = repo / "newdir" / "provider-created.txt"
    target.parent.mkdir()
    target.write_bytes(b"provider-created\n")
    original_remove = journals._remove_bound_empty_workspace_directory
    replacement_metadata: os.stat_result | None = None

    def remove_then_recreate(root_fd, relative, expected, authority) -> None:
        nonlocal replacement_metadata
        original_remove(root_fd, relative, expected, authority)
        if relative == "newdir":
            (repo / relative).mkdir()
            replacement_metadata = (repo / relative).lstat()

    monkeypatch.setattr(
        journals,
        "_remove_bound_empty_workspace_directory",
        remove_then_recreate,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert replacement_metadata is not None
    assert journal.active
    current = (repo / "newdir").lstat()
    assert (current.st_dev, current.st_ino) == (
        replacement_metadata.st_dev,
        replacement_metadata.st_ino,
    )


def test_final_directory_inventory_rejects_post_metadata_descendant_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching-mode external clone cannot replace a proved descendant."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "final" / "mutations",
        slice_id="restore-directory-final-descendant-swap",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")
    external = tmp_path / "matching-external-directory"
    external.mkdir(mode=stat.S_IMODE((repo / "nest").lstat().st_mode))
    external_before = external.lstat()
    parked = tmp_path / "parked-baseline-nest"
    original_restore = journals._restore_bound_workspace_directory_metadata
    swapped = False

    def restore_then_swap(root_fd, relative, expected, mode, xattrs, authority):
        nonlocal swapped
        original_restore(
            root_fd,
            relative,
            expected,
            mode,
            xattrs,
            authority,
        )
        if not swapped and relative == "nest":
            (repo / "nest").rename(parked)
            external.rename(repo / "nest")
            swapped = True

    monkeypatch.setattr(
        journals,
        "_restore_bound_workspace_directory_metadata",
        restore_then_swap,
    )

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert journal.active
    current = (repo / "nest").lstat()
    assert (current.st_dev, current.st_ino) == (
        external_before.st_dev,
        external_before.st_ino,
    )
    assert parked.is_dir()


def test_owner_access_preparation_chmods_only_held_directory_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chmod-boundary ABA cannot redirect preparation to an external dir."""

    from crucible.workflows import _mutation_journal as journals

    root = tmp_path / "root"
    root.mkdir()
    child = root / "provider-child"
    child.mkdir(mode=0o700)
    child.chmod(0o000)
    external = root / "external"
    external.mkdir(mode=0o755)
    external.chmod(0o755)
    external_before = external.lstat()
    parked = root / "parked-provider-child"
    original_chmod = journals.os.chmod
    injected = False

    def swap_around_chmod(path, mode, *args, **kwargs) -> None:
        nonlocal injected
        if not injected and mode == 0o700:
            child.rename(parked)
            external.rename(child)
            try:
                original_chmod(path, mode, *args, **kwargs)
            finally:
                child.rename(external)
                parked.rename(child)
            injected = True
            return
        original_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(journals.os, "chmod", swap_around_chmod)

    journals._make_real_directory_tree_owner_accessible(
        root,
        tree_label="provider tree",
    )

    assert injected
    assert stat.S_IMODE(child.lstat().st_mode) == 0o700
    external_after = external.lstat()
    assert stat.S_IMODE(external_after.st_mode) == stat.S_IMODE(
        external_before.st_mode
    )
    assert (external_after.st_dev, external_after.st_ino) == (
        external_before.st_dev,
        external_before.st_ino,
    )


def test_full_restore_access_preparation_never_chmods_swapped_external_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full pre-restore access pass changes only its O_PATH-held inode."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    nest = repo / "nest"
    baseline_mode = stat.S_IMODE(nest.lstat().st_mode)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "access" / "mutations",
        slice_id="restore-access-exact-inode",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")
    # Owner-write permits the deterministic cross-parent ABA while denying
    # traversal, forcing the writable preparation pass to grant owner access.
    nest.chmod(0o200)
    nest_before = nest.lstat()
    external = tmp_path / "external-restore-access"
    external.mkdir(mode=0o755)
    external.chmod(0o755)
    external_before = external.lstat()
    parked = tmp_path / "parked-restore-access"
    original_chmod = journals.os.chmod
    injected = False

    def swap_around_target_chmod(path, mode, *args, **kwargs) -> None:
        nonlocal injected
        try:
            target = os.stat(path)
        except OSError:
            target = None
        if (
            not injected
            and target is not None
            and (target.st_dev, target.st_ino)
            == (nest_before.st_dev, nest_before.st_ino)
        ):
            nest.rename(parked)
            external.rename(nest)
            try:
                original_chmod(path, mode, *args, **kwargs)
            finally:
                nest.rename(external)
                parked.rename(nest)
            injected = True
            return
        original_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(journals.os, "chmod", swap_around_target_chmod)

    assert journal.restore(full=False)
    assert injected
    assert not journal.journal_dir.exists()
    assert stat.S_IMODE(nest.lstat().st_mode) == baseline_mode
    external_after = external.lstat()
    assert stat.S_IMODE(external_after.st_mode) == stat.S_IMODE(
        external_before.st_mode
    )
    assert (external_after.st_dev, external_after.st_ino) == (
        external_before.st_dev,
        external_before.st_ino,
    )


def test_restore_rejects_static_baseline_directory_inode_substitution(
    tmp_path: Path,
) -> None:
    """Mode/xattr equality cannot authorize a different present directory."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    baseline_empty = repo / "baseline-empty"
    baseline_empty.mkdir()
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "identity" / "mutations",
        slice_id="restore-directory-static-substitution",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")
    parked = tmp_path / "parked-baseline-empty"
    baseline_empty.rename(parked)
    external = tmp_path / "external-baseline-empty"
    external.mkdir(mode=stat.S_IMODE(parked.lstat().st_mode))
    secret = external / "secret"
    secret.write_bytes(b"external secret\n")
    external_before = external.lstat()
    external.rename(baseline_empty)

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert journal.active
    assert journal.journal_dir.is_dir()
    current = baseline_empty.lstat()
    assert (current.st_dev, current.st_ino) == (
        external_before.st_dev,
        external_before.st_ino,
    )
    assert (baseline_empty / "secret").read_bytes() == b"external secret\n"
    assert parked.is_dir()


def test_terminal_directory_reproof_rejects_post_restore_authority_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory authority remains live through the terminal journal handoff."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    baseline_empty = repo / "baseline-empty"
    baseline_empty.mkdir()
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "handoff" / "mutations",
        slice_id="restore-directory-terminal-handoff",
        scope=("dir/victim.txt",),
        sequence=1,
    )
    (repo / "dir" / "victim.txt").write_bytes(b"provider mutation\n")
    external = tmp_path / "external-terminal-handoff"
    external.mkdir(mode=stat.S_IMODE(baseline_empty.lstat().st_mode))
    (external / "secret").write_bytes(b"external secret\n")
    external_before = external.lstat()
    parked = tmp_path / "parked-terminal-handoff"
    original_restore = journal._restore_directory_modes
    swapped = False

    def restore_then_swap(cwd: Path, *, observed_parent_bindings=None):
        nonlocal swapped
        authority = original_restore(
            cwd,
            observed_parent_bindings=observed_parent_bindings,
        )
        baseline_empty.rename(parked)
        external.rename(baseline_empty)
        swapped = True
        return authority

    monkeypatch.setattr(journal, "_restore_directory_modes", restore_then_swap)

    assert not journal.restore(full=False)
    assert swapped
    assert journal.active
    assert journal.journal_dir.is_dir()
    current = baseline_empty.lstat()
    assert (current.st_dev, current.st_ino) == (
        external_before.st_dev,
        external_before.st_ino,
    )
    assert (baseline_empty / "secret").read_bytes() == b"external secret\n"
    assert parked.is_dir()


def test_tracked_metadata_rejects_post_install_external_file_without_laundering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A swapped file cannot replace installed authority or rewrite manifest."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    relative = "dir/victim.txt"
    target = repo / relative
    baseline_payload = target.read_bytes()
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "tracked" / "mutations",
        slice_id="restore-tracked-post-install-swap",
        scope=(relative,),
        sequence=1,
    )
    baseline_manifest = copy.deepcopy(
        journal._manifest["tracked_path_metadata"][relative]
    )
    target.write_bytes(b"provider mutation\n")
    external = tmp_path / "external-tracked-file"
    external.write_bytes(baseline_payload)
    external.chmod(int(baseline_manifest["mode"]))
    external_before = external.lstat()
    parked = tmp_path / "parked-restored-file"
    original_restore = journal._restore_snapshot_root
    swapped = False

    def restore_then_swap(
        cwd: Path,
        restored_relative: str,
        *,
        expected_parent_bindings=None,
    ):
        nonlocal swapped
        bindings = original_restore(
            cwd,
            restored_relative,
            expected_parent_bindings=expected_parent_bindings,
        )
        if not swapped and restored_relative == relative:
            target.rename(parked)
            external.rename(target)
            swapped = True
        return bindings

    monkeypatch.setattr(journal, "_restore_snapshot_root", restore_then_swap)

    assert not journal.restore(full=False, _defer_terminal_cleanup=True)
    assert swapped
    assert journal.active
    current = target.lstat()
    assert (current.st_dev, current.st_ino) == (
        external_before.st_dev,
        external_before.st_ino,
    )
    assert target.read_bytes() == baseline_payload
    assert parked.read_bytes() == baseline_payload
    assert journal._manifest["tracked_path_metadata"][relative] == (
        baseline_manifest
    )


def test_shared_parent_metadata_is_deferred_until_all_roots_are_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """One restrictive shared parent cannot strand a later sibling root."""

    from crucible.workflows import _mutation_journal as journals

    repo = tmp_path / "repo"
    _init_repository(repo)
    shared = repo / "shared"
    shared.mkdir()
    first = shared / "first.txt"
    second = shared / "second.txt"
    first.write_bytes(b"first baseline\n")
    second.write_bytes(b"second baseline\n")
    shared.chmod(0o500)
    request.addfinalizer(
        lambda: shared.chmod(0o700) if shared.exists() else None
    )
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "parents" / "mutations",
        slice_id="restore-shared-parent-order",
        scope=("shared/first.txt", "shared/second.txt"),
        sequence=1,
    )
    shared.chmod(0o700)
    first.write_bytes(b"first provider mutation\n")
    second.write_bytes(b"second provider mutation\n")

    original_parent_restore = (
        journals._restore_bound_workspace_directory_metadata
    )
    restored_roots: list[str] = []
    parent_restore_observations: list[tuple[str, ...]] = []
    original_snapshot_restore = journal._restore_snapshot_root

    def record_root_restore(
        cwd: Path,
        relative: str,
        *,
        expected_parent_bindings=None,
    ):
        bindings = original_snapshot_restore(
            cwd,
            relative,
            expected_parent_bindings=expected_parent_bindings,
        )
        restored_roots.append(relative)
        return bindings

    def prove_parent_restore_is_last(
        root_fd: int,
        relative: str,
        expected,
        mode: int,
        xattrs,
        authority,
    ) -> None:
        if relative == "shared":
            parent_restore_observations.append(tuple(restored_roots))
        original_parent_restore(
            root_fd,
            relative,
            expected,
            mode,
            xattrs,
            authority,
        )

    monkeypatch.setattr(journal, "_restore_snapshot_root", record_root_restore)
    monkeypatch.setattr(
        journals,
        "_restore_bound_workspace_directory_metadata",
        prove_parent_restore_is_last,
    )

    assert journal.restore(full=False, _defer_terminal_cleanup=True)
    assert first.read_bytes() == b"first baseline\n"
    assert second.read_bytes() == b"second baseline\n"
    assert parent_restore_observations == [
        ("shared/first.txt", "shared/second.txt"),
    ]
    assert stat.S_IMODE(shared.lstat().st_mode) == 0o500
