# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Paths a development gate run creates in the checkout it verifies.

The clean development-verifier preflight runs the gate once on the untouched
candidate and baseline checkouts and removes what that run created.  A later
slice gate that leaves the same paths charges them to the slice: a new file
Git reports is a scope escape, a new directory inside the mutation journal's
directory authority is a changed topology.

The preflight therefore records exactly the paths a slice gate objects to,
the task summary shows them, and the development verifier removes them again
after each gate run in which they were new.  These tests pair every kind of
path with the slice gate's verdict, run the two generated shell scripts on a
real tree, and follow the record from the preflight to a cascade gate run.
"""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import subprocess
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from assay import Implementation
from assay.corpus.store import Corpus

from crucible.products.deepswe.runners import deepswe_dev_verify as dv
from crucible.workflows import _deepswe_baseline_preflight as baseline_preflight
from crucible.workflows import deepswe as wf
from crucible.workflows._build_verify import BuildVerifyOutcome, RunnerResult
from crucible.workflows._mutation_journal import MutationJournalError
from crucible.workflows._provider_settlement import (
    ProviderSettlementCleanupError,
)
from tests import test_workflows_tier_empty_diff_retry as tier_harness
from tests.test_workflows_deepswe import (
    _build_corpus,
    _patch_container_lifecycle,
    _preflight_checkout_image,
    _prepare_parent_run,
)
from tests.test_workflows_tier_empty_diff_retry import (
    _WriteTargetBackend,
    _contract_aware_build_verify_stub,
    _run,
    _successful_gate_outcome,
)


# ---------- what a gate run can leave behind ----------


_IGNORE_RULES = "build/\n*.log\n"
_COMMITTED_FILES = {".gitignore": _IGNORE_RULES, "pkg/mod.py": "value = 1\n"}
_SELF_IGNORING = (("{}/.gitignore", "*\n"), ("{}/CACHEDIR.TAG", "cache\n"))


@dataclass(frozen=True)
class _GateResidue:
    """One kind of path a passing gate run leaves in the checkout.

    ``recorded`` is the root the preflight records for it, ``None`` when a
    slice gate accepts the path and the preflight therefore records nothing.
    """

    files: tuple[tuple[str, str], ...] = ()
    directories: tuple[str, ...] = ()
    recorded: str | None = None

    def create(self, root: Path) -> None:
        for relative in self.directories:
            (root / relative).mkdir(parents=True)
        for relative, content in self.files:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def remove(self, root: Path) -> None:
        assert self.recorded is not None
        path = root / self.recorded
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    @property
    def records_a_file(self) -> bool:
        return any(relative == self.recorded for relative, _ in self.files)


def _self_ignoring(directory: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name.format(directory), content) for name, content in _SELF_IGNORING
    )


_RESIDUE = {
    "file": _GateResidue(
        files=(("gate-created.lock", "locked\n"),),
        recorded="gate-created.lock",
    ),
    "directory": _GateResidue(
        files=(("gate-report/index.html", "report\n"),),
        recorded="gate-report",
    ),
    "empty_directory": _GateResidue(
        directories=("gate-empty",),
        recorded="gate-empty",
    ),
    "file_below_tracked_directory": _GateResidue(
        files=(("pkg/gate-created.txt", "created\n"),),
        recorded="pkg/gate-created.txt",
    ),
    "nested_self_ignoring_directory": _GateResidue(
        files=_self_ignoring("pkg/.pytest_cache"),
        recorded="pkg/.pytest_cache",
    ),
    "nested_pattern_ignored_directory": _GateResidue(
        files=(("pkg/build/out.txt", "built\n"),),
        recorded="pkg/build",
    ),
    "nested_directory_holding_an_ignored_file": _GateResidue(
        files=(("pkg/logs/run.log", "log\n"),),
        recorded="pkg/logs",
    ),
    "top_level_self_ignoring_directory": _GateResidue(
        files=_self_ignoring(".pytest_cache"),
    ),
    "top_level_pattern_ignored_directory": _GateResidue(
        files=(("build/out.txt", "built\n"),),
    ),
    "top_level_directory_holding_an_ignored_directory": _GateResidue(
        files=(("gate-out/build/out.txt", "built\n"),),
    ),
    "top_level_directory_holding_an_ignored_file": _GateResidue(
        files=(("gate-logs/run.log", "log\n"),),
    ),
    "nested_ignored_file": _GateResidue(
        files=(("pkg/run.log", "log\n"),),
    ),
}
_RECORDED = tuple(
    sorted(
        residue.recorded
        for residue in _RESIDUE.values()
        if residue.recorded is not None
    )
)


# Untracked directories the image brought along, one of them empty.  Git
# lists each as one collapsed root, before and after a gate run adds to it.
_INHERITED_DIRECTORY = "inherited-dir"
_INHERITED_EMPTY_DIRECTORY = "inherited-empty"


def _inherit_untracked_directories(checkout: Path) -> None:
    (checkout / _INHERITED_DIRECTORY).mkdir()
    (checkout / _INHERITED_DIRECTORY / "old.txt").write_text(
        "old\n",
        encoding="utf-8",
    )
    (checkout / _INHERITED_EMPTY_DIRECTORY).mkdir()


def _exception_chain(error: BaseException) -> str:
    """Render an error with its causes and contexts, as a CI log shows it."""

    return "".join(traceback.format_exception(error))


# ---------- the preflight records what its gate run created ----------


def _init_checkout(path: Path, *, label: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "tracked.txt").write_text(f"{label}\n", encoding="utf-8")
    for relative, content in _COMMITTED_FILES.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Crucible Test",
            "-c",
            "user.email=crucible@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=path,
        check=True,
    )
    # Dirt the image brought along is not created by the gate.
    (path / "inherited-untracked.txt").write_text(
        f"{label}-untracked\n",
        encoding="utf-8",
    )


class _Preflight:
    """One preflight over two fresh checkouts with a stubbed gate run."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch,
        *,
        inherit: Callable[[Path], None] | None = None,
    ) -> None:
        self.candidate = tmp_path / "candidate"
        self.baseline = tmp_path / "baseline"
        _init_checkout(self.candidate, label="candidate")
        _init_checkout(self.baseline, label="baseline")
        self.workspaces = (self.candidate, self.baseline)
        if inherit is not None:
            for workspace in self.workspaces:
                inherit(workspace)
        self.image = {
            workspace: _preflight_checkout_image(workspace)
            for workspace in self.workspaces
        }
        self.receipts: list[
            baseline_preflight.DevelopmentPreflightRestorationReceipt
        ] = []
        self.gate_ran = False
        self._tmp_path = tmp_path
        self._monkeypatch = monkeypatch

    def run(self, gate) -> BuildVerifyOutcome:
        from crucible.workflows import _build_verify

        def _gate(*_args, **_kwargs):
            for workspace in self.workspaces:
                gate(workspace)
            self.gate_ran = True
            return BuildVerifyOutcome(
                "success",
                build=None,
                verify=None,
                dev_verify=None,
            )

        self._monkeypatch.setattr(_build_verify, "build_and_verify", _gate)
        return baseline_preflight.run_development_verifier_preflight(
            spec=None,
            candidate_workspace=self.candidate,
            baseline_workspace=self.baseline,
            log_dir=self._tmp_path / "logs",
            journal_root=(
                self._tmp_path
                / "task-corpus"
                / "runs"
                / "attempt"
                / "final_gate_mutations"
            ),
            extra_parameters={
                "deepswe_metadata": {
                    "language": "python",
                    "container_id": "candidate-container",
                    "baseline_container_id": "baseline-container",
                },
            },
            restoration_callback=self.receipts.append,
        )

    def assert_restored(self) -> None:
        for workspace in self.workspaces:
            assert _preflight_checkout_image(workspace) == self.image[workspace]

    def fail_listing_after(self, calls: int, error: BaseException) -> None:
        """Let the listing of gate residue succeed ``calls`` times, then raise.

        The preflight lists each checkout once before its gate run (the
        candidate first) and once after it.
        """

        listing = baseline_preflight._preflight_gate_residue
        listed: list[Path] = []

        def _listing(worktree: Path):
            listed.append(worktree)
            if len(listed) > calls:
                raise error
            return listing(worktree)

        self._monkeypatch.setattr(
            baseline_preflight,
            "_preflight_gate_residue",
            _listing,
        )


def test_preflight_records_the_paths_a_slice_gate_would_charge(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _Preflight(tmp_path, monkeypatch)

    def _gate(workspace: Path) -> None:
        for residue in _RESIDUE.values():
            residue.create(workspace)
        (workspace / "inherited-untracked.txt").write_text(
            "overwritten\n",
            encoding="utf-8",
        )
        # The file stays where the image put it; the gate run only stops
        # Git from tracking it, so it joins the untracked listing.
        subprocess.run(
            ["git", "rm", "--cached", "-q", "tracked.txt"],
            cwd=workspace,
            check=True,
        )

    outcome = preflight.run(_gate)

    assert outcome.classification == "success"
    assert len(preflight.receipts) == 1
    receipt = preflight.receipts[0]
    # Every kind a slice gate fails on, and none it accepts: no top-level
    # ignored directory, no ignored file, and no path the checkout had
    # before the gate ran, overwritten or newly untracked.
    assert [item.gate_created_paths for item in receipt.workspaces] == [
        _RECORDED,
        _RECORDED,
    ]
    payload = receipt.to_payload()
    assert [item["gate_created_paths"] for item in payload["workspaces"]] == [
        list(_RECORDED),
        list(_RECORDED),
    ]
    restored = baseline_preflight.DevelopmentPreflightRestorationReceipt
    assert restored.from_payload(json.loads(json.dumps(payload))) == receipt
    preflight.assert_restored()


def test_preflight_listing_failure_is_its_own_error_after_the_restore(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _Preflight(tmp_path, monkeypatch)
    preflight.fail_listing_after(
        2,
        MutationJournalError("git ls-files is unavailable"),
    )

    with pytest.raises(MutationJournalError) as caught:
        preflight.run(_RESIDUE["file"].create)

    message = str(caught.value)
    assert preflight.gate_ran
    assert "could not list the paths its gate run created" in message
    assert "did not exactly restore" not in message
    assert "git ls-files is unavailable" in str(caught.value.__cause__)
    # Both checkouts were restored; only the evidence is missing.
    assert preflight.receipts == []
    preflight.assert_restored()


def test_preflight_listing_interrupt_is_raised_after_the_restore(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _Preflight(tmp_path, monkeypatch)
    interrupt = KeyboardInterrupt("operator interrupt during the listing")
    preflight.fail_listing_after(2, interrupt)

    with pytest.raises(KeyboardInterrupt) as caught:
        preflight.run(_RESIDUE["file"].create)

    # A control-flow exception is never replaced by a preflight error, and
    # it does not stop the rollback of either checkout.
    assert caught.value is interrupt
    assert preflight.gate_ran
    assert preflight.receipts == []
    preflight.assert_restored()


def test_preflight_listing_failure_after_a_failed_gate_run_keeps_both_errors(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _Preflight(tmp_path, monkeypatch)
    preflight.fail_listing_after(
        2,
        MutationJournalError("git ls-files is unavailable"),
    )

    def _gate(workspace: Path) -> None:
        _RESIDUE["file"].create(workspace)
        if workspace == preflight.baseline:
            raise RuntimeError("gate runner crashed")

    with pytest.raises(MutationJournalError) as caught:
        preflight.run(_gate)

    # The missing evidence is the error that ends the preflight, as for a
    # receipt that cannot be built; the gate's own error stays in its chain.
    assert "could not list the paths its gate run created" in str(caught.value)
    assert "git ls-files is unavailable" in str(caught.value.__cause__)
    assert "RuntimeError: gate runner crashed" in _exception_chain(caught.value)
    assert not preflight.gate_ran
    assert preflight.receipts == []
    preflight.assert_restored()


def test_preflight_listing_failure_before_the_gate_run_stops_the_preflight(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _Preflight(tmp_path, monkeypatch)
    # The candidate is listed; the baseline listing fails with both
    # journals already open.
    preflight.fail_listing_after(
        1,
        MutationJournalError("git ls-files is unavailable"),
    )

    with pytest.raises(MutationJournalError) as caught:
        preflight.run(_RESIDUE["file"].create)

    # The listing error itself: no gate ran, so nothing is said about one.
    assert str(caught.value) == "git ls-files is unavailable"
    assert not preflight.gate_ran
    assert preflight.receipts == []
    preflight.assert_restored()


def test_preflight_does_not_record_what_changes_inside_an_inherited_path(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _Preflight(
        tmp_path,
        monkeypatch,
        inherit=_inherit_untracked_directories,
    )

    def _gate(workspace: Path) -> None:
        _RESIDUE["file"].create(workspace)
        for directory in (_INHERITED_DIRECTORY, _INHERITED_EMPTY_DIRECTORY):
            (workspace / directory / "gate-new.txt").write_text(
                "new\n",
                encoding="utf-8",
            )
        (workspace / _INHERITED_DIRECTORY / "old.txt").write_text(
            "overwritten\n",
            encoding="utf-8",
        )
        (workspace / "pkg" / "mod.py").write_text("value = 3\n", encoding="utf-8")

    outcome = preflight.run(_gate)

    assert outcome.classification == "success"
    # Git lists an untracked directory as one root however full it is, so
    # nothing added below one is new to the listing; a changed file names no
    # path to remove.  Only the new top-level file is recorded.
    assert [
        item.gate_created_paths for item in preflight.receipts[0].workspaces
    ] == [("gate-created.lock",), ("gate-created.lock",)]
    # All of it is restored all the same.
    preflight.assert_restored()


def test_preflight_records_a_name_that_is_not_valid_utf8_as_decoded(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _Preflight(tmp_path, monkeypatch)
    raw_name = b"gate-\xff.lock"
    decoded = os.fsdecode(raw_name)

    def _gate(workspace: Path) -> None:
        with open(os.path.join(os.fsencode(workspace), raw_name), "wb") as file:
            file.write(b"locked\n")

    outcome = preflight.run(_gate)

    assert outcome.classification == "success"
    receipt = preflight.receipts[0]
    assert [item.gate_created_paths for item in receipt.workspaces] == [
        (decoded,),
        (decoded,),
    ]
    # The receipt keeps the exact name; JSON carries its escaped surrogate.
    restored = baseline_preflight.DevelopmentPreflightRestorationReceipt
    assert restored.from_payload(
        json.loads(json.dumps(receipt.to_payload()))
    ) == receipt
    # A later gate run is never bound to a name it cannot quote exactly.
    assert dv.gate_created_paths_evidence((decoded,)) is None
    preflight.assert_restored()


# ---------- each kind of path, paired with the slice gate's verdict ----------


def _slice_classification(
    tmp_path: Path,
    monkeypatch,
    gate,
    *,
    inherit: Callable[[Path], None] | None = None,
) -> str:
    """Run one slice whose passing gate calls ``gate(repo)``.

    Returns the slice classification.  A journal or settlement error that
    ends the tier without one propagates with its whole chain.  ``inherit``
    leaves untracked paths in a repository that tracks only its target file.
    """

    from crucible.workflows import tier as tier_workflow

    def _gate(*_args, **_kwargs):
        gate(tmp_path / "repo")
        return _successful_gate_outcome()

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(_gate),
    )
    committed_files: dict[str, str] | None = dict(_COMMITTED_FILES)
    if inherit is not None:
        # The fixtures are committed with ``git add -A``, which would track
        # whatever the repository inherited before them.
        committed_files = None
        init_repo = tier_harness._init_repo

        def _init_repo(repo: Path) -> str:
            base = init_repo(repo)
            inherit(repo)
            return base

        monkeypatch.setattr(tier_harness, "_init_repo", _init_repo)
    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        _WriteTargetBackend(),
        skip_build_verify=False,
        committed_files=committed_files,
    )
    phase = summary["phase_1_slices"][0]
    # The implementer wrote only its declared file and the gate passed.
    assert phase["build_verify"] == "success"
    assert (rc == 0) == (phase["classification"] == "success")
    return phase["classification"]


def _retained_scope_escapes(tmp_path: Path) -> tuple[str, ...]:
    """The paths the tier charged to the slice it rolled back."""

    corpus = Corpus(tmp_path / "corpus")
    retained = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    return retained.scope_escapes


@pytest.mark.parametrize(
    "kind",
    [
        kind
        for kind, residue in _RESIDUE.items()
        if residue.recorded is not None and kind != "file"
    ],
)
def test_slice_gate_charges_every_recorded_kind_to_the_slice(
    tmp_path: Path,
    monkeypatch,
    kind: str,
):
    # ``file`` is paired below, through the real verifier.
    residue = _RESIDUE[kind]
    recorded = residue.recorded

    if residue.records_a_file:
        assert _slice_classification(
            tmp_path,
            monkeypatch,
            residue.create,
        ) == "hard_fail_scope_escape"
        assert _retained_scope_escapes(tmp_path) == (recorded,)
        return

    # For a new directory the tier does not classify the slice: its journal
    # refuses the changed topology, directly or while settling the slice.
    with pytest.raises(
        (MutationJournalError, ProviderSettlementCleanupError)
    ) as caught:
        _slice_classification(tmp_path, monkeypatch, residue.create)

    # The recorded root, and nothing else, is what the journal objects to.
    assert (
        "changed guarded worktree directory topology or modes: "
        f"[{recorded!r}]"
    ) in _exception_chain(caught.value)


@pytest.mark.parametrize(
    "directory",
    [_INHERITED_DIRECTORY, _INHERITED_EMPTY_DIRECTORY],
)
def test_slice_gate_charges_a_path_added_below_an_inherited_untracked_directory(
    tmp_path: Path,
    monkeypatch,
    directory: str,
):
    created = f"{directory}/gate-new.txt"

    def _gate(repo: Path) -> None:
        (repo / created).write_text("new\n", encoding="utf-8")

    # The preflight does not record this path (see the record test above),
    # so the verifier never removes it and the slice is still charged.
    assert _slice_classification(
        tmp_path,
        monkeypatch,
        _gate,
        inherit=_inherit_untracked_directories,
    ) == "hard_fail_scope_escape"
    assert _retained_scope_escapes(tmp_path) == (created,)


def test_slice_gate_accepts_every_kind_the_preflight_does_not_record(
    tmp_path: Path,
    monkeypatch,
):
    accepted = [
        residue for residue in _RESIDUE.values() if residue.recorded is None
    ]

    def _gate(repo: Path) -> None:
        for residue in accepted:
            residue.create(repo)

    assert len(accepted) == 5
    assert _slice_classification(tmp_path, monkeypatch, _gate) == "success"
    repo = tmp_path / "repo"
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == "value = 2\n"
    for residue in accepted:
        assert all((repo / relative).is_file() for relative, _ in residue.files)


def test_slice_gate_accepts_recorded_kinds_removed_within_the_gate_run(
    tmp_path: Path,
    monkeypatch,
):
    recorded = [
        residue for residue in _RESIDUE.values() if residue.recorded is not None
    ]

    def _gate(repo: Path) -> None:
        for residue in recorded:
            residue.create(repo)
        for residue in recorded:
            residue.remove(repo)

    # Created and removed inside the gate window, the same paths are
    # invisible to both checks: this is what the verifier's removal relies on.
    assert _slice_classification(tmp_path, monkeypatch, _gate) == "success"
    repo = tmp_path / "repo"
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == "value = 2\n"


# ---------- the two generated scripts, run by bash on a real tree ----------


def _bash(command: str) -> tuple[int, str, str]:
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def _tree(root: Path) -> dict[str, str]:
    """Every entry below ``root``: file bytes, link target or directory."""

    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries[relative] = f"link:{os.readlink(path)}"
        elif path.is_dir():
            entries[relative] = "directory"
        else:
            entries[relative] = path.read_text(encoding="utf-8")
    return entries


_ODD_NAMES = (
    "-leading-dash",
    "glob[*]?.txt",
    "it's \"quoted\".txt",
    "pkg/$(touch injected)`touch injected`",
    "with space.txt",
)


def _workspace(tmp_path: Path) -> Path:
    root = (tmp_path / "app").resolve()
    root.mkdir()
    (root / "keep.txt").write_text("keep\n", encoding="utf-8")
    (root / "globbed.txt").write_text("not a recorded path\n", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("value = 1\n", encoding="utf-8")
    return root


def test_generated_scripts_handle_every_kind_of_recorded_path(
    tmp_path: Path,
    monkeypatch,
):
    root = _workspace(tmp_path)
    bystanders = _tree(root)
    (root / "gate-created.lock").write_text("locked\n", encoding="utf-8")
    (root / "gate-empty").mkdir()
    (root / "gate-report" / "assets").mkdir(parents=True)
    (root / "gate-report" / "assets" / "app.js").write_text("js\n", encoding="utf-8")
    (root / "gate-report" / "index.html").write_text("report\n", encoding="utf-8")
    for name in _ODD_NAMES:
        (root / name).write_text("odd\n", encoding="utf-8")
    paths = tuple(
        sorted(
            (
                "gate-created.lock",
                "gate-empty",
                "gate-report",
                "never-created",
                "pkg/never-created",
                *_ODD_NAMES,
            )
        )
    )
    evidence = dv.gate_created_paths_evidence(paths)
    assert evidence is not None and tuple(evidence["paths"]) == paths
    expected = [
        "absent" if path.endswith("never-created") else "present"
        for path in paths
    ]
    before = _tree(root)

    probe = _bash(dv._gate_created_probe_script(paths, workspace_root=str(root)))

    # The probe prints one line per path and changes nothing.
    assert probe == (0, "".join(f"{state}\n" for state in expected), "")
    assert _tree(root) == before

    removal = _bash(
        dv._gate_created_removal_script(paths, workspace_root=str(root))
    )

    assert removal == (
        0,
        "".join(
            "removed\n" if state == "present" else "absent\n"
            for state in expected
        ),
        "",
    )
    # Exactly the recorded paths are gone: no glob matched ``globbed.txt``
    # and no name was evaluated by the shell.
    assert _tree(root) == bystanders
    assert not list(tmp_path.rglob("injected"))

    # The callers read the same protocol.
    monkeypatch.setattr(
        dv,
        "_run_docker_exec",
        lambda _container, command, **_kwargs: _bash(command),
    )
    assert dv._absent_gate_created_paths(
        "candidate",
        paths,
        workspace_root=str(root),
    ) == paths
    assert dv._remove_gate_created_paths(
        "candidate",
        paths,
        workspace_root=str(root),
    ) == 0


def test_removal_unlinks_a_recorded_symlink_and_keeps_its_target(
    tmp_path: Path,
):
    root = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside\n", encoding="utf-8")
    (root / "to-directory").symlink_to(root / "pkg")
    (root / "to-outside").symlink_to(outside)
    (root / "to-outside-file").symlink_to(outside / "secret.txt")
    (root / "dangling").symlink_to(root / "missing")
    paths = ("dangling", "to-directory", "to-outside", "to-outside-file")

    probe = _bash(dv._gate_created_probe_script(paths, workspace_root=str(root)))
    removal = _bash(
        dv._gate_created_removal_script(paths, workspace_root=str(root))
    )

    assert probe == (0, "present\n" * 4, "")
    assert removal == (0, "removed\n" * 4, "")
    assert _tree(root) == {
        "globbed.txt": "not a recorded path\n",
        "keep.txt": "keep\n",
        "pkg": "directory",
        "pkg/mod.py": "value = 1\n",
    }
    assert _tree(outside) == {"secret.txt": "outside\n"}


@pytest.mark.parametrize("target", ["outside", "inside"])
def test_removal_refuses_a_path_reached_through_a_symlinked_parent(
    tmp_path: Path,
    monkeypatch,
    target: str,
):
    root = _workspace(tmp_path)
    real = (tmp_path / "outside") if target == "outside" else (root / "real")
    real.mkdir()
    (real / "victim.txt").write_text("victim\n", encoding="utf-8")
    (real / "deep").mkdir()
    (real / "deep" / "victim.txt").write_text("victim\n", encoding="utf-8")
    (root / "linked").symlink_to(real)
    (root / "a-recorded-file").write_text("created\n", encoding="utf-8")
    # The first path alone would be removed; the second and third are
    # reached through the link, one directly and one a level further down.
    paths = ("a-recorded-file", "linked/deep/victim.txt", "linked/victim.txt")
    before = _tree(tmp_path)

    removal = _bash(
        dv._gate_created_removal_script(paths, workspace_root=str(root))
    )

    # Refused as a whole, before anything is removed.
    assert removal == (44, "", "")
    assert _tree(tmp_path) == before

    monkeypatch.setattr(
        dv,
        "_run_docker_exec",
        lambda _container, command, **_kwargs: _bash(command),
    )
    with pytest.raises(ValueError, match=r"removal failed: rc=44, lines=0/3"):
        dv._remove_gate_created_paths(
            "candidate",
            paths,
            workspace_root=str(root),
        )
    assert _tree(tmp_path) == before
    # The probe only reads: it reports what the link shows.
    assert _bash(
        dv._gate_created_probe_script(paths, workspace_root=str(root))
    ) == (0, "present\n" * 3, "")


def test_removal_does_not_resolve_the_parent_of_an_absent_path(
    tmp_path: Path,
):
    root = _workspace(tmp_path)
    paths = ("gone/build", "pkg/build")

    removal = _bash(
        dv._gate_created_removal_script(paths, workspace_root=str(root))
    )

    # ``gone`` does not exist at all: nothing to remove is not a failure.
    assert removal == (0, "absent\nabsent\n", "")


def test_largest_bindable_record_fits_one_shell_argument():
    # 64 names of 255 single quotes: each byte needs the longest quoting.
    paths = tuple(
        sorted(f"{index:02d}" + "'" * 253 for index in range(64))
    )
    assert sum(len(path) for path in paths) <= dv.MAX_GATE_CREATED_PATH_BYTES
    assert dv.gate_created_paths_evidence(paths) is not None
    many = tuple(f"p{index:03d}" for index in range(dv.MAX_GATE_CREATED_PATHS))
    assert dv.gate_created_paths_evidence(many) is not None

    for record in (paths, many):
        for script in (
            dv._gate_created_probe_script(record),
            dv._gate_created_removal_script(record),
        ):
            # MAX_ARG_STRLEN: one argument of an exec is at most 128 KiB.
            assert len(script.encode("utf-8")) < 131_072


# ---------- a slice gate through the real verifier ----------


def _bash_on(repo: Path):
    """The exec seam of an authoring container whose ``/app`` is ``repo``."""

    def _exec(_container_id, command, *, timeout, cleanup_timeout):
        del timeout, cleanup_timeout
        return _bash(command.replace("/app", str(repo)))

    return _exec


@pytest.mark.parametrize("recorded", [True, False])
def test_slice_gate_through_the_verifier_charges_only_unrecorded_gate_paths(
    tmp_path: Path,
    monkeypatch,
    recorded: bool,
):
    from crucible.workflows import tier as tier_workflow

    created = _RESIDUE["file"].recorded
    monkeypatch.setattr(
        dv,
        "run_contained_docker_exec",
        _bash_on(tmp_path / "repo"),
    )
    metadata: dict[str, object] = {
        "container_id": "candidate-container",
        "language": "python",
        "dev_verify_command": (
            f"cd /app && printf 'locked\\n' > {created} "
            "&& printf '1 passed in 0.01s\\n'"
        ),
    }
    if recorded:
        metadata[dv.GATE_CREATED_PATHS_KEY] = dv.gate_created_paths_evidence(
            (created,)
        )
    gate_runs: list[tuple[int, dict]] = []

    def _gate(*_args, **_kwargs):
        stdout = io.StringIO()
        rc = dv.main(
            stdin=io.StringIO(
                json.dumps(
                    {
                        "parameters": {
                            "role": "dev_verify",
                            "deepswe_metadata": metadata,
                        },
                    }
                )
            ),
            stdout=stdout,
        )
        gate_runs.append((rc, json.loads(stdout.getvalue())))
        return _successful_gate_outcome()

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(_gate),
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        _WriteTargetBackend(),
        skip_build_verify=False,
    )

    # The verifier ran the gate command once and it passed.
    assert [gate_rc for gate_rc, _payload in gate_runs] == [0]
    values = gate_runs[0][1]["values"]
    assert values["tests_passed"] == 1.0
    phase = summary["phase_1_slices"][0]
    assert phase["build_verify"] == "success"
    if recorded:
        # The verifier removed the path its own command created ...
        assert values["dev_verify_gate_created_paths_removed"] == 1.0
        assert not repo.joinpath(created).exists()
        # ... and the slice keeps its accepted write.
        assert rc == 0
        assert phase["classification"] == "success"
        assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
            "value = 2\n"
        )
        return
    # Without the record the same path is still charged to the slice.
    assert "dev_verify_gate_created_paths_removed" not in values
    assert rc == 1
    assert phase["classification"] == "hard_fail_scope_escape"
    assert _retained_scope_escapes(tmp_path) == (created,)
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == "value = 1\n"
    assert not repo.joinpath(created).exists()


# ---------- the record reaches the task summary and the cascade gate ----------


def _receipt(
    tmp_path: Path,
    candidate_paths: tuple[str, ...],
) -> baseline_preflight.DevelopmentPreflightRestorationReceipt:
    def _restoration(label: str, transaction: str, paths: tuple[str, ...]):
        root = tmp_path / "receipt" / label
        root.mkdir(parents=True)
        metadata = root.stat()
        return baseline_preflight.DevelopmentPreflightWorkspaceRestoration(
            label=label,
            workspace=str(root.resolve()),
            workspace_device=metadata.st_dev,
            workspace_inode=metadata.st_ino,
            namespace=(),
            split_namespaces=(),
            initialized_gitlinks=(),
            journal_transaction_id=transaction,
            journal_family=((transaction, None),),
            gate_created_paths=paths,
        )

    return baseline_preflight.DevelopmentPreflightRestorationReceipt(
        workspaces=(
            _restoration("candidate", "a" * 32, candidate_paths),
            _restoration("baseline", "b" * 32, ()),
        )
    )


def _run_cascade(tmp_path: Path, monkeypatch, receipt):
    """Run one task cascade whose preflight emits ``receipt`` (if any)."""

    from crucible.common.cost import CostAccumulator

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    _patch_container_lifecycle(monkeypatch, tmp_path, {})
    seen: dict[str, dict] = {}

    def _preflight(**kwargs):
        seen["preflight"] = copy.deepcopy(kwargs["extra_parameters"])
        if receipt is not None:
            kwargs["restoration_callback"](receipt)
        return BuildVerifyOutcome(
            "success",
            build=None,
            verify=None,
            dev_verify=RunnerResult(
                role="dev_verify",
                runner="crucible.products.deepswe.runners.deepswe_dev_verify",
                returncode=0,
                values={"returncode": 0.0},
                raw_output=b"{}",
                stderr=b"",
                duration_s=0.0,
            ),
        )

    def _architect(**kwargs):
        seen["architect"] = copy.deepcopy(kwargs["extra_parameters"])
        return 0

    monkeypatch.setattr(wf, "_run_dev_verify_preflight", _preflight)
    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)
    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "gate-created"),
        cost=CostAccumulator(),
    )
    return summary["dev_verify_preflight"], seen


def test_gate_created_paths_reach_the_task_summary_and_the_cascade_gate(
    tmp_path: Path,
    monkeypatch,
):
    paths = ("Cargo.lock", "coverage")

    preflight, seen = _run_cascade(
        tmp_path,
        monkeypatch,
        _receipt(tmp_path, paths),
    )

    # The preflight itself runs unbound; every later gate run is bound.
    assert dv.GATE_CREATED_PATHS_KEY not in seen["preflight"]["deepswe_metadata"]
    bound = seen["architect"]["deepswe_metadata"][dv.GATE_CREATED_PATHS_KEY]
    assert bound == {
        "schema_version": dv.GATE_CREATED_PATHS_SCHEMA,
        "paths": ["Cargo.lock", "coverage"],
    }
    assert dv._validated_gate_created_paths({dv.GATE_CREATED_PATHS_KEY: bound}) == (
        paths
    )
    assert preflight["gate_created_paths"] == ["Cargo.lock", "coverage"]
    assert preflight["gate_created_path_count"] == 2
    assert preflight["gate_created_paths_bound"] is True


@pytest.mark.parametrize("receipt_emitted", [True, False])
def test_no_gate_created_paths_leave_the_cascade_parameters_unchanged(
    tmp_path: Path,
    monkeypatch,
    receipt_emitted: bool,
):
    preflight, seen = _run_cascade(
        tmp_path,
        monkeypatch,
        _receipt(tmp_path, ()) if receipt_emitted else None,
    )

    assert seen["architect"] == seen["preflight"]
    assert preflight["gate_created_paths"] == []
    assert preflight["gate_created_path_count"] == 0
    assert preflight["gate_created_paths_bound"] is False


@pytest.mark.parametrize("case", ["not_utf8", "too_many"])
def test_unbindable_gate_created_paths_are_listed_but_not_bound(
    tmp_path: Path,
    monkeypatch,
    case: str,
):
    if case == "not_utf8":
        paths: tuple[str, ...] = ("Cargo.lock", os.fsdecode(b"gate-\xff.lock"))
        listed = ["Cargo.lock", "gate-\\udcff.lock"]
    else:
        paths = tuple(
            f"p{index:03d}" for index in range(dv.MAX_GATE_CREATED_PATHS + 1)
        )
        listed = list(paths[: dv.MAX_GATE_CREATED_PATHS])

    preflight, seen = _run_cascade(
        tmp_path,
        monkeypatch,
        _receipt(tmp_path, paths),
    )

    # The task continues as before: its gate runs are not bound, and the
    # summary says so with names that every JSON writer can encode.
    assert seen["architect"] == seen["preflight"]
    assert preflight["gate_created_paths"] == listed
    assert preflight["gate_created_path_count"] == len(paths)
    assert preflight["gate_created_paths_bound"] is False
    json.dumps(preflight, ensure_ascii=False).encode("utf-8")
