# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Git's rediscovery of a held checkout, answered from an unchanged witness.

``_assert_workspace_mutation_authority`` proves at every call that the thread
still holds the mutation lock on exactly the checkout it locked.  Its last
layer asks Git itself where the common directory is.  The three helpers that
only hand out a held descriptor pass ``reuse_discovery=True`` (a "bracket
call" below): there the answer may come from memory, and only while a witness
of what Git reads for the question is unchanged.  Every other call (a
"default call") asks Git.

A remembered proof of identity must never be stale, so most tests here are
refusal tests: an answer is remembered, the checkout is changed so that Git's
answer differs, and the bracket call must refuse exactly as the default call
does, at that call and with its text.  The remaining tests pin the gain (no
Git process for an unchanged checkout), the call sites, and the rule that on
a remembered answer the journal neither changes the checkout through a held
descriptor nor records a settlement state or a stage of a recovery.
"""

from __future__ import annotations

import ast
import functools
import json
import mmap
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import FrameType
from typing import Callable, Iterator

import pytest

from assay import Implementation
from assay.corpus.store import Corpus

from crucible.workflows import _mutation_journal as journals
from crucible.workflows import _reaper
from crucible.workflows._run import exec_ctx_now

from .test_workflows_tier_mutation_transaction import _spec

pytestmark = pytest.mark.skipif(
    os.name == "nt" or not os.path.isdir("/proc/self/fd"),
    reason="workspace authority needs POSIX descriptors and /proc/self/fd",
)

_REAL_GIT = shutil.which("git") or "git"
#: Longer than the time a change needs to settle before a witness is taken.
_SETTLE_S = 0.06
_DESCRIPTOR_PATH = re.compile(r"/(?:proc/self|dev)/fd/\d+")
_CHANGED = "workspace mutation authority changed after lock acquisition"
_UNREADABLE = "git rev-parse --path-format=absolute --git-common-dir failed"
_BRACKET_HELPERS = frozenset((
    "_bound_workspace_restore_root",
    "_git_admin_authority_batch",
    "_bound_git_admin_path",
))
_ASSERTION = "_assert_workspace_mutation_authority"
_RECORD_PROOF = "_assert_git_discovered_authority"
_WRITE_PROOF = "_ask_git_before_held_write"
_SETTLEMENT_STATES = frozenset((
    "scoped",
    "prepared_keep",
    "prepared_rollback",
    "committed",
    "rolled_back",
))
#: The stage states exact journal recovery records in a journal manifest.
_RECOVERY_STATES = frozenset((
    "external_recovery_prepared",
    "external_recovery_verified",
))
#: The stages of its event that a resumed recovery acts on.
_RECOVERY_STAGES = frozenset((
    "prepared",
    "source_restored",
    "verified",
    "tombstone_prepared",
))


# --------------------------------------------------------------- repositories

def _git(cwd: Path, *args: str, check: bool = True) -> str:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": "witness test",
        "GIT_AUTHOR_EMAIL": "witness@test.invalid",
        "GIT_COMMITTER_NAME": "witness test",
        "GIT_COMMITTER_EMAIL": "witness@test.invalid",
    })
    result = subprocess.run(
        [
            _REAL_GIT,
            "-c", "protocol.file.allow=always",
            "-c", "init.defaultBranch=main",
            "-c", "advice.addEmbeddedRepo=false",
            "-c", "maintenance.auto=false",
            "-c", "gc.auto=0",
            *args,
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed in {cwd}: {result.stderr}"
        )
    return result.stdout.strip()


def _init_repository(path: Path, tag: str = "base") -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    for name in ("a.py", "b.py"):
        (path / name).write_text(f"# {tag} {name}\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", tag)


def _layout_plain(tmp: Path) -> dict[str, Path]:
    repo = tmp / "repo"
    _init_repository(repo)
    return {"tmp": tmp, "root": repo, "git_dir": repo / ".git"}


def _layout_deep(tmp: Path) -> dict[str, Path]:
    repo = tmp / "g" / "p" / "repo"
    _init_repository(repo)
    return {
        "tmp": tmp, "root": repo, "git_dir": repo / ".git",
        "g": tmp / "g", "p": tmp / "g" / "p",
    }


def _layout_nested(tmp: Path) -> dict[str, Path]:
    parent = tmp / "parent"
    repo = parent / "repo"
    _init_repository(repo)
    return {"tmp": tmp, "root": repo, "git_dir": repo / ".git", "parent": parent}


def _layout_worktree(tmp: Path) -> dict[str, Path]:
    main = tmp / "main"
    _init_repository(main)
    _git(main, "worktree", "add", "-q", "-b", "lane", str(tmp / "wt"))
    return {
        "tmp": tmp, "root": tmp / "wt", "main": main,
        "git_dir": main / ".git" / "worktrees" / "wt",
        "common": main / ".git",
    }


def _layout_submodule(tmp: Path) -> dict[str, Path]:
    lib = tmp / "lib"
    _init_repository(lib, "lib")
    grand = tmp / "grand"
    superproject = grand / "super"
    _init_repository(superproject, "super")
    _git(superproject, "submodule", "add", "-q", f"file://{lib}", "sub")
    _git(superproject, "commit", "-qm", "add sub")
    return {
        "tmp": tmp, "root": superproject / "sub", "super": superproject,
        "grand": grand, "git_dir": superproject / ".git" / "modules" / "sub",
    }


def _layout_nested_submodule(tmp: Path) -> dict[str, Path]:
    lib = tmp / "lib"
    _init_repository(lib, "lib")
    middle = tmp / "middle-source"
    _init_repository(middle, "middle")
    _git(middle, "submodule", "add", "-q", f"file://{lib}", "sub")
    _git(middle, "commit", "-qm", "middle with sub")
    top = tmp / "top"
    _init_repository(top, "top")
    _git(top, "submodule", "add", "-q", f"file://{middle}", "mid")
    _git(top, "commit", "-qm", "top with mid")
    _git(top, "submodule", "update", "--init", "--recursive", "-q")
    root = top / "mid" / "sub"
    assert (root / ".git").is_file()
    return {"tmp": tmp, "root": root, "super": top / "mid", "top": top}


_LAYOUTS: dict[str, Callable[[Path], dict[str, Path]]] = {
    "plain": _layout_plain,
    "deep": _layout_deep,
    "nested": _layout_nested,
    "worktree": _layout_worktree,
    "submodule": _layout_submodule,
    "nested-submodule": _layout_nested_submodule,
}


# ------------------------------------------------------------ calls and counts

class _GitCalls:
    """Every Git command the journal module starts through ``_git``."""

    def __init__(self) -> None:
        self.args: list[tuple[str, ...]] = []

    def discoveries(self, since: int = 0) -> int:
        return sum(
            1 for args in self.args[since:]
            if "--show-superproject-working-tree" in args
        )


@pytest.fixture
def git_calls(monkeypatch: pytest.MonkeyPatch) -> _GitCalls:
    calls = _GitCalls()
    original = journals._git

    def counting(cwd: Path, *args: str, **kwargs: object) -> bytes:
        calls.args.append(args)
        return original(cwd, *args, **kwargs)

    monkeypatch.setattr(journals, "_git", counting)
    return calls


@dataclass(frozen=True)
class _Outcome:
    """What one call of the assertion did: its refusal text, its Git calls."""

    refusal: str | None
    git: int


def _observe(calls: _GitCalls, body: Callable[[], object]) -> _Outcome:
    start = len(calls.args)
    try:
        body()
    except journals.MutationJournalError as exc:
        refusal: str | None = _DESCRIPTOR_PATH.sub("<fd>", str(exc))
    else:
        refusal = None
    return _Outcome(refusal, len(calls.args) - start)


def _bracket(root: Path) -> object:
    """The assertion as the descriptor-handing helpers call it."""

    return journals._assert_workspace_mutation_authority(
        root,
        reuse_discovery=True,
    )


def _default(root: Path) -> object:
    """The assertion as every other call site calls it."""

    return journals._assert_workspace_mutation_authority(root)


def _remembered(authority: object) -> bool:
    """Whether the held authority carries a remembered answer (presence only)."""

    return getattr(authority, "_discovery_witness", None) is not None


def witness_unavailable_reason(root: Path) -> str | None:
    """Why no answer can be remembered for a checkout at ``root`` here.

    The gain exists only where the Git program lies beyond the reach of the
    user that holds the lock and nothing repository-like lies above the
    checkout.  A test that expects an answer from memory skips elsewhere.
    """

    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        return "the test runs as the superuser"
    if any(name.startswith(("LD_", "DYLD_")) for name in os.environ):
        return "the environment sets a variable of the program loader"
    program: str | None = None
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not os.path.isabs(directory):
            return "PATH has a relative entry before the Git program"
        candidate = os.path.join(directory, "git")
        if os.path.lexists(candidate):
            program = candidate
            break
        current = directory
        while True:
            try:
                entry = os.lstat(current)
            except FileNotFoundError:
                pass
            except OSError:
                return "a PATH entry before the Git program cannot be read"
            else:
                if not stat.S_ISDIR(entry.st_mode):
                    return "a PATH entry before the Git program is no real directory"
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    if program is None:
        return "PATH names no Git program"
    metadata = os.lstat(program)
    if not (stat.S_ISREG(metadata.st_mode) and os.access(program, os.X_OK)):
        return "the first entry named git on PATH is no program file"
    current = program
    while True:
        entry = os.lstat(current)
        if (
            stat.S_ISLNK(entry.st_mode)
            or entry.st_uid != 0
            or entry.st_mode & 0o022
        ):
            return "the first git on PATH is within reach of the test's user"
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    above = os.path.dirname(os.fspath(root))
    while True:
        for name in (".git", "HEAD"):
            if os.path.lexists(os.path.join(above, name)):
                return "an entry named .git or HEAD lies above the checkout"
        parent = os.path.dirname(above)
        if parent == above:
            break
        above = parent
    return None


#: A backward step of the wall clock beyond this defeats the settle time the
#: tests wait out.
_CLOCK_STEP_NS = 20_000_000
#: How long a test waits for a wall clock that was set back to catch up.
_SETTLE_LIMIT_S = 5.0


class _NotFromMemory(AssertionError):
    """Git was asked where the answer had to come from memory.

    The clocks are read here, when the requirement fails: what the test
    does afterwards, such as settling the journal it began, is not part
    of what ``steady_wall_clock`` looks at.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        point = _SETTLE_POINT[0]
        #: The wait before the requirement could not outlast a clock step.
        self.unsettled = point.unsettled
        #: How far the wall clock ran ahead of the monotonic one since
        #: that wait; negative when the wall clock was set back.
        self.drift_ns = (time.time_ns() - point.wall) - (
            time.monotonic_ns() - point.monotonic
        )


def from_memory(held: bool, detail: object = "") -> None:
    """Require that an answer came from memory, or a bound that rests on it.

    Only this requirement depends on the wall clock (``steady_wall_clock``).
    Every other assertion of a test, a refusal above all, is a plain
    ``assert`` and fails whatever the clock did.
    """

    if not held:
        raise _NotFromMemory(f"Git was asked where memory had to answer: {detail}")


class _SettlePoint:
    """The end of the last wait for the settle time, read on both clocks."""

    def __init__(self) -> None:
        self.wall = time.time_ns()
        self.monotonic = time.monotonic_ns()
        #: A witnessed change time was still too new when the wait ended.
        self.unsettled = False


_SETTLE_POINT = [_SettlePoint()]


def _newest_witnessed_change(root: Path) -> int:
    """The newest change time that must have settled for an answer to be kept.

    Found here without the code under test, for the checkout at ``root``:
    the control files and the two directories Git needs, which the witness
    holds, and what guards a Git question: the last directory on the way
    to each ``PATH`` directory ahead of the Git program, and of every
    directory on a held path the older of its own change time and that of
    the directory it lies in.
    """

    def changed(path: str) -> int | None:
        try:
            return os.lstat(path).st_ctime_ns
        except OSError:
            return None

    def way(path: str) -> list[str]:
        found = [path]
        while os.path.dirname(found[-1]) != found[-1]:
            found.append(os.path.dirname(found[-1]))
        return found[::-1]

    def exchanges(path: str) -> Iterator[int]:
        for directory in way(path)[1:]:
            pair = [
                time for time in (changed(os.path.dirname(directory)), changed(directory))
                if time is not None
            ]
            if pair:
                yield min(pair)

    dot_git = os.path.join(root, ".git")
    git_dir = dot_git
    strict = []
    if os.path.isfile(dot_git):
        strict.append(dot_git)
        with open(dot_git, encoding="utf-8") as handle:
            git_dir = handle.read().partition("gitdir:")[2].strip()
    common = git_dir
    try:
        with open(os.path.join(git_dir, "commondir"), encoding="utf-8") as handle:
            common = os.path.normpath(os.path.join(git_dir, handle.read().strip()))
    except OSError:
        pass
    strict.extend(
        os.path.join(git_dir, name)
        for name in ("HEAD", "commondir", "config.worktree")
    )
    strict.extend(os.path.join(common, name) for name in ("config", "objects", "refs"))
    times = [time for time in map(changed, strict) if time is not None]
    for held in (os.fspath(root), git_dir, common):
        times.extend(exchanges(held))
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not os.path.isabs(directory) or os.path.lexists(
            os.path.join(directory, "git")
        ):
            break
        existing = [path for path in way(directory) if changed(path) is not None]
        times.append(changed(existing[-1]) or 0)
        times.extend(exchanges(existing[-1]))
    return max(times, default=0)


def _settle(root: Path | None = None) -> None:
    """Wait out the settle time after the last change to a checkout.

    The end of the wait is where ``steady_wall_clock`` measures from.  With
    ``root`` the wait goes on until the wall clock has passed the newest
    witnessed change time of that checkout by the same margin, which takes
    longer only when the clock was set back before or during the wait.  If
    it has not after ``_SETTLE_LIMIT_S``, that is noted.
    """

    time.sleep(_SETTLE_S)
    behind = 0
    if root is not None:
        limit = time.monotonic() + _SETTLE_LIMIT_S
        while True:
            behind = (
                _newest_witnessed_change(root)
                + int(_SETTLE_S * 1_000_000_000)
                - time.time_ns()
            )
            if behind <= 0 or time.monotonic() >= limit:
                break
            time.sleep(min(behind / 1_000_000_000, 0.25))
    point = _SettlePoint()
    point.unsettled = behind > 0
    _SETTLE_POINT[0] = point


def steady_wall_clock(test: Callable[..., object]) -> Callable[..., object]:
    """Skip a failed ``from_memory`` when the wall clock was set back around it.

    A witnessed change time is compared with the wall clock.  When that clock
    is set back, as a virtual machine does when it resynchronises with its
    host, an entry changed a moment ago looks changed in the future: nothing
    is remembered until the clock has caught up, and Git is asked, as it must
    be.  An expectation of "no Git process" then says nothing about the code.

    Only a step that can have done that is excused: one between the last
    wait for the settle time (``_settle``) and the failed requirement, or
    one that wait could not outlast.  A step earlier in the test is waited
    out and excuses nothing, and no other failure is excused at all.  A skip
    with this reason means that a requirement that an answer came from
    memory was not examined in that run.
    """

    @functools.wraps(test)
    def steady(*args: object, **kwargs: object) -> object:
        _SETTLE_POINT[0] = _SettlePoint()
        try:
            return test(*args, **kwargs)
        except _NotFromMemory as failure:
            drift = failure.drift_ns
            if not (failure.unsettled or drift < -_CLOCK_STEP_NS):
                raise
            name = os.environ.get("PYTEST_CURRENT_TEST", test.__name__)
            step = (
                "and had not caught up when the wait for the checkout ended"
                if failure.unsettled
                else f"by {-drift // 1_000_000} ms after the checkout had settled"
            )
            pytest.skip(
                f"{name.rsplit(' (', 1)[0]}: the wall clock was set back {step}; "
                "whether an answer came from memory was not examined"
            )

    return steady


class _Clock:
    """The wall clock as the journal module reads it, set by a test.

    Whether a change lies inside the settle time is then decided by the
    change time itself and not by how fast the test ran.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._time = journals.time
        self.now_ns: int | None = None
        monkeypatch.setattr(journals, "time", self)

    def time_ns(self) -> int:
        return self._time.time_ns() if self.now_ns is None else self.now_ns

    def __getattr__(self, name: str) -> object:
        return getattr(self._time, name)


#: How often a pair of bracket calls is made before an answer that was not
#: kept counts against the code.
_MEMORY_ROUNDS = 4


def _expect_answer_from_memory(
    root: Path,
    calls: _GitCalls,
    *,
    body: Callable[[], object] | None = None,
) -> None:
    """Wait out the settle time; a repeated bracket call must not ask Git.

    An answer is not kept when directories above the checkout changed while
    Git was asked, which other tests that share the temporary directory
    can cause.  The pair of calls is therefore made again a few times
    before the requirement fails.
    """

    call = body or (lambda: _bracket(root))
    for _round in range(_MEMORY_ROUNDS):
        _settle(root)
        call()
        second = _observe(calls, call)
        assert second.refusal is None, second
        if second.git == 0:
            return
    reason = witness_unavailable_reason(root)
    if reason is not None:
        pytest.skip(f"no answer can be remembered here: {reason}")
    from_memory(False, f"a bracket call of an unchanged checkout: {second}")


def _expect_no_git(calls: _GitCalls, body: Callable[[], object]) -> None:
    """One call that passes and must be answered from memory."""

    outcome = _observe(calls, body)
    assert outcome.refusal is None, outcome
    from_memory(outcome.git == 0, outcome)


def _expect_git_asked_every_time(root: Path, calls: _GitCalls) -> None:
    """A shape without a witness: every bracket call starts Git's discovery."""

    _settle(root)
    for _ in range(2):
        before = len(calls.args)
        assert _observe(calls, lambda: _bracket(root)).refusal is None
        assert calls.discoveries(before) >= 1


# ---------------------------------------------------- states made under a lock
# Each state changes the checkout and returns the function that undoes it.

Undo = Callable[[], None]


def _pwrite(path: Path, payload: bytes, offset: int = 0) -> None:
    descriptor = os.open(path, os.O_WRONLY)
    try:
        os.pwrite(descriptor, payload, offset)
    finally:
        os.close(descriptor)


def _rewrite(path: Path, payload: bytes) -> Undo:
    before = path.read_bytes()
    path.write_bytes(payload)

    def undo() -> None:
        path.write_bytes(before)

    return undo


def _append(path: Path, payload: bytes) -> Undo:
    return _rewrite(path, path.read_bytes() + payload)


def _change_in_place(path: Path, offset: int, payload: bytes) -> Undo:
    before = path.read_bytes()[offset:offset + len(payload)]
    _pwrite(path, payload, offset)
    return lambda: _pwrite(path, before, offset)


def _change_through_mapping(path: Path, offset: int, payload: bytes) -> Undo:
    descriptor = os.open(path, os.O_RDWR)
    try:
        view = mmap.mmap(
            descriptor, 0, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(descriptor)
    before = bytes(view[offset:offset + len(payload)])
    view[offset:offset + len(payload)] = payload

    def undo() -> None:
        view[offset:offset + len(payload)] = before
        view.close()

    return undo


def _rename_away(path: Path) -> Undo:
    away = path.with_name(path.name + ".away")
    path.rename(away)

    def undo() -> None:
        away.rename(path)

    return undo


def _unsearchable(path: Path) -> Undo:
    mode = stat.S_IMODE(path.lstat().st_mode)
    path.chmod(0)
    return lambda: path.chmod(mode)


def _symlink_to_itself(path: Path) -> Undo:
    real = path.with_name(path.name + ".real")
    path.rename(real)
    os.symlink(real, path)

    def undo() -> None:
        path.unlink()
        real.rename(path)

    return undo


def _remove_repository(directory: Path) -> Undo:
    def undo() -> None:
        shutil.rmtree(directory / ".git")

    return undo


def _format_version_offset(config: Path) -> int:
    needle = b"repositoryformatversion = 0"
    return config.read_bytes().index(needle) + len(needle) - 1


def _gitlink_in_parent(ctx: dict[str, Path]) -> Undo:
    parent = ctx["parent"]
    _git(parent, "init", "-q")
    _git(parent, "add", "repo")
    assert _git(parent, "ls-files", "--stage").startswith("160000 ")
    return _remove_repository(parent)


def _gitlink_committed_in_parent(ctx: dict[str, Path]) -> Undo:
    parent = ctx["parent"]
    undo_repository = _gitlink_in_parent(ctx)
    (parent / "top.txt").write_text("top\n", encoding="utf-8")
    _git(parent, "add", "top.txt")
    _git(parent, "commit", "-qm", "the checkout as a gitlink")

    def undo() -> None:
        (parent / "top.txt").unlink()
        undo_repository()

    return undo


def _gitlink_two_levels_up(ctx: dict[str, Path]) -> Undo:
    _git(ctx["g"], "init", "-q")
    _git(ctx["g"], "add", "p/repo")
    assert _git(ctx["g"], "ls-files", "--stage").startswith("160000 ")
    return _remove_repository(ctx["g"])


def _head_removed_below_parent_repository(ctx: dict[str, Path]) -> Undo:
    _git(ctx["parent"], "init", "-q")
    undo_repository = _remove_repository(ctx["parent"])
    undo_head = _rename_away(ctx["git_dir"] / "HEAD")

    def undo() -> None:
        undo_head()
        undo_repository()

    return undo


def _outside_work_tree_removed(ctx: dict[str, Path]) -> Undo:
    """``core.worktree`` names an outside directory whose parent then goes."""

    config = ctx["git_dir"] / "config"
    before = config.read_bytes()
    outside = ctx["tmp"] / "outside" / "a" / "b"
    outside.mkdir(parents=True)
    _git(ctx["root"], "config", "core.worktree", str(outside))
    shutil.rmtree(ctx["tmp"] / "outside")

    def undo() -> None:
        config.write_bytes(before)

    return undo


def _section_line_include_unparseable(ctx: dict[str, Path]) -> Undo:
    """An include the line test of the preflight passes; its file is broken."""

    extra = ctx["tmp"] / "extra.config"
    extra.write_text("[user\nthis is not config\n", encoding="utf-8")
    undo_config = _append(
        ctx["git_dir"] / "config",
        f"[core] [include]\n\tpath = {extra}\n".encode(),
    )

    def undo() -> None:
        undo_config()
        extra.unlink()

    return undo


def _write_program(path: Path, *, differing: bool) -> None:
    """A ``git`` that passes everything on, or answers one question wrongly."""

    lines = ["#!/bin/sh"]
    if differing:
        lines += [
            'case " $* " in',
            '  *" --git-common-dir "*)',
            '    if [ -z "${GIT_DIR:-}" ]; then',
            "      echo /nonexistent/elsewhere/.git; exit 0",
            "    fi ;;",
            "esac",
        ]
    lines += [f'exec "{_REAL_GIT}" "$@"', ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _write_answering_program(path: Path, common: Path) -> None:
    """A ``git`` that answers the discovery as a readable checkout would."""

    lines = [
        "#!/bin/sh",
        'case " $* " in',
        '  *" --show-superproject-working-tree "*) exit 0 ;;',
        f'  *" --git-common-dir "*) echo "{common}"; exit 0 ;;',
        "esac",
        f'exec "{_REAL_GIT}" "$@"',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _differing_program_placed(ctx: dict[str, Path]) -> Undo:
    program = ctx["front"] / "git"
    _write_program(program, differing=True)
    return program.unlink


def _selected_program_replaced(ctx: dict[str, Path]) -> Undo:
    program = ctx["front"] / "git"
    fresh = ctx["front"] / "git.new"
    _write_program(fresh, differing=True)
    os.replace(fresh, program)

    def undo() -> None:
        _write_program(fresh, differing=False)
        os.replace(fresh, program)

    return undo


def _differing_program_in_next_directory(ctx: dict[str, Path]) -> Undo:
    program = ctx["later"] / "git"
    _write_program(program, differing=True)
    return program.unlink


def _front_directory(ctx: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty directory of the test first on ``PATH``."""

    front = ctx["tmp"] / "front-bin"
    front.mkdir()
    ctx["front"] = front
    monkeypatch.setenv("PATH", f"{front}{os.pathsep}{os.environ.get('PATH', '')}")


def _front_program(ctx: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """A pass-through ``git`` of the test first on ``PATH``."""

    _front_directory(ctx, monkeypatch)
    _write_program(ctx["front"] / "git", differing=False)


def _front_dud(ctx: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """An executable file named ``git`` that is no program, first on ``PATH``."""

    dud = ctx["tmp"] / "dud-bin"
    later = ctx["tmp"] / "later-bin"
    dud.mkdir()
    later.mkdir()
    (dud / "git").write_text("not a program\n", encoding="utf-8")
    (dud / "git").chmod(0o755)
    ctx["later"] = later
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((str(dud), str(later), os.environ.get("PATH", ""))),
    )


def _index_entry_mode_offset(index: Path, name: bytes) -> int:
    """Offset of the mode field of one entry of a version 2 or 3 index."""

    data = index.read_bytes()
    assert data[:4] == b"DIRC"
    version, count = struct.unpack(">II", data[4:12])
    assert version in (2, 3)
    offset = 12
    for _ in range(count):
        flags = struct.unpack(">H", data[offset + 60:offset + 62])[0]
        name_at = offset + 62 + (2 if version >= 3 and flags & 0x4000 else 0)
        length = flags & 0x0FFF
        if data[name_at:name_at + length] == name:
            return offset + 24
        offset += ((name_at - offset + length + 8) // 8) * 8
    raise AssertionError(f"no index entry {name!r}")


def _submodule_gitlink_removed(ctx: dict[str, Path]) -> Undo:
    superproject = ctx["super"]
    name = ctx["root"].name
    entry = _git(superproject, "ls-files", "--stage", "--", name).split()
    _git(superproject, "update-index", "--force-remove", name)

    def undo() -> None:
        _git(
            superproject, "update-index", "--add", "--cacheinfo",
            f"160000,{entry[1]},{name}",
        )

    return undo


def _superproject_made_submodule(ctx: dict[str, Path]) -> Undo:
    _git(ctx["grand"], "init", "-q")
    _git(ctx["grand"], "add", "super")
    return _remove_repository(ctx["grand"])


def _module_work_tree_changed(ctx: dict[str, Path]) -> Undo:
    config = ctx["git_dir"] / "config"
    before = config.read_bytes()
    elsewhere = ctx["tmp"] / "elsewhere"
    elsewhere.mkdir()
    _git(ctx["root"], "config", "core.worktree", str(elsewhere))

    def undo() -> None:
        config.write_bytes(before)
        elsewhere.rmdir()

    return undo


def _superproject_modules_moved(ctx: dict[str, Path]) -> Undo:
    modules = ctx["super"] / ".git" / "modules"
    away = ctx["tmp"] / "moved" / "modules"
    away.parent.mkdir()
    modules.rename(away)
    os.symlink(away, modules)

    def undo() -> None:
        modules.unlink()
        away.rename(modules)
        away.parent.rmdir()

    return undo


def _superproject_index_entry_changed(ctx: dict[str, Path]) -> Undo:
    index = ctx["super"] / ".git" / "index"
    offset = _index_entry_mode_offset(index, ctx["root"].name.encode())
    return _change_in_place(index, offset, struct.pack(">I", 0o100644))


@dataclass(frozen=True)
class _Refusal:
    """One state that only Git's rediscovery refuses."""

    layout: str
    state: Callable[[dict[str, Path]], Undo]
    text: str
    #: False where the shape of the checkout allows no remembered answer.
    remembers: bool = True
    before_lock: Callable[[dict[str, Path], pytest.MonkeyPatch], None] | None = None


_REFUSALS: dict[str, _Refusal] = {
    "gitlink-staged-in-parent-repository": _Refusal(
        "nested", _gitlink_in_parent, _CHANGED),
    "gitlink-staged-two-levels-up": _Refusal(
        "deep", _gitlink_two_levels_up, _CHANGED),
    "head-removed": _Refusal(
        "plain", lambda ctx: _rename_away(ctx["git_dir"] / "HEAD"), _UNREADABLE),
    "head-emptied": _Refusal(
        "plain", lambda ctx: _rewrite(ctx["git_dir"] / "HEAD", b""), _UNREADABLE),
    "head-changed-in-place": _Refusal(
        "plain",
        lambda ctx: _change_in_place(ctx["git_dir"] / "HEAD", 0, b"x"),
        _UNREADABLE),
    "head-changed-through-mapping": _Refusal(
        "plain",
        lambda ctx: _change_through_mapping(ctx["git_dir"] / "HEAD", 0, b"x"),
        _UNREADABLE),
    "head-removed-below-parent-repository": _Refusal(
        "nested", _head_removed_below_parent_repository, _CHANGED),
    "config-unparseable": _Refusal(
        "plain",
        lambda ctx: _rewrite(
            ctx["git_dir"] / "config", b"[core\n\tthis is not config\n"),
        _UNREADABLE),
    "config-format-99": _Refusal(
        "plain",
        lambda ctx: _append(
            ctx["git_dir"] / "config",
            b"[core]\n\trepositoryformatversion = 99\n"),
        _UNREADABLE),
    "config-changed-in-place": _Refusal(
        "plain",
        lambda ctx: _change_in_place(
            ctx["git_dir"] / "config",
            _format_version_offset(ctx["git_dir"] / "config"), b"9"),
        _UNREADABLE),
    "config-changed-through-mapping": _Refusal(
        "plain",
        lambda ctx: _change_through_mapping(
            ctx["git_dir"] / "config",
            _format_version_offset(ctx["git_dir"] / "config"), b"9"),
        _UNREADABLE),
    "objects-removed": _Refusal(
        "plain", lambda ctx: _rename_away(ctx["git_dir"] / "objects"), _UNREADABLE),
    "refs-removed": _Refusal(
        "plain", lambda ctx: _rename_away(ctx["git_dir"] / "refs"), _UNREADABLE),
    "objects-unsearchable": _Refusal(
        "plain", lambda ctx: _unsearchable(ctx["git_dir"] / "objects"), _UNREADABLE),
    "refs-unsearchable": _Refusal(
        "plain", lambda ctx: _unsearchable(ctx["git_dir"] / "refs"), _UNREADABLE),
    "grandparent-replaced-by-symlink-to-itself": _Refusal(
        "deep", lambda ctx: _symlink_to_itself(ctx["g"]), _CHANGED),
    "worktree-main-checkout-replaced-by-symlink-to-itself": _Refusal(
        "worktree", lambda ctx: _symlink_to_itself(ctx["main"]), _CHANGED),
    "outside-work-tree-removed": _Refusal(
        "plain", _outside_work_tree_removed, _UNREADABLE),
    "section-line-include-unparseable": _Refusal(
        "plain", _section_line_include_unparseable, _UNREADABLE),
    "differing-git-placed-earlier-on-path": _Refusal(
        "plain", _differing_program_placed, _CHANGED,
        before_lock=_front_directory),
    "selected-git-replaced-under-its-name": _Refusal(
        "plain", _selected_program_replaced, _CHANGED,
        remembers=False, before_lock=_front_program),
    "differing-git-behind-a-file-that-is-no-program": _Refusal(
        "plain", _differing_program_in_next_directory, _CHANGED,
        remembers=False, before_lock=_front_dud),
    "submodule-gitlink-removed-from-superproject-index": _Refusal(
        "submodule", _submodule_gitlink_removed, _CHANGED, remembers=False),
    "submodule-superproject-made-a-submodule": _Refusal(
        "submodule", _superproject_made_submodule, _CHANGED, remembers=False),
    "submodule-module-work-tree-changed": _Refusal(
        "submodule", _module_work_tree_changed, _CHANGED, remembers=False),
    "submodule-superproject-modules-moved": _Refusal(
        "submodule", _superproject_modules_moved, _UNREADABLE, remembers=False),
    "submodule-superproject-index-entry-changed-in-place": _Refusal(
        "submodule", _superproject_index_entry_changed, _CHANGED,
        remembers=False),
    "submodule-superproject-head-changed-in-place": _Refusal(
        "submodule",
        lambda ctx: _change_in_place(ctx["super"] / ".git" / "HEAD", 0, b"x"),
        _CHANGED, remembers=False),
    "nested-submodule-gitlink-removed-from-superproject-index": _Refusal(
        "nested-submodule", _submodule_gitlink_removed, _CHANGED,
        remembers=False),
}


# --------------------------------------------- refusals kept at the bracket call

@pytest.mark.parametrize("name", sorted(_REFUSALS))
@steady_wall_clock
def test_git_only_refusal_is_kept_at_the_bracket_call(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    case = _REFUSALS[name]
    ctx = _LAYOUTS[case.layout](tmp_path)
    root = ctx["root"]
    if case.before_lock is not None:
        case.before_lock(ctx, monkeypatch)
    with journals.workspace_mutation_lock(root):
        if case.remembers:
            _expect_answer_from_memory(root, git_calls)
        else:
            _expect_git_asked_every_time(root, git_calls)
        undo = case.state(ctx)
        try:
            refused = _observe(git_calls, lambda: _bracket(root))
            again = _observe(git_calls, lambda: _bracket(root))
            default = _observe(git_calls, lambda: _default(root))
        finally:
            undo()
        restored = _observe(git_calls, lambda: _bracket(root))

    assert refused.refusal is not None and case.text in refused.refusal, refused
    # The same call, the same text, the same Git processes as a call that
    # never reuses an answer; and the refusal repeats while the state lasts.
    assert refused == default
    assert again == default
    assert restored.refusal is None, restored
    if not case.remembers:
        assert restored.git >= 2


@pytest.mark.parametrize("helper", sorted(_BRACKET_HELPERS))
@steady_wall_clock
def test_git_only_refusal_is_kept_through_each_helper(
    helper: str,
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=root,
        journal_root=tmp_path / "corpus" / "runs" / "witness" / "mutations",
        slice_id="witness-helper",
        scope=("a.py",),
        sequence=1,
    )

    def through_helper() -> None:
        if helper == "_bound_workspace_restore_root":
            with journals._bound_workspace_restore_root(journal):
                pass
        elif helper == "_git_admin_authority_batch":
            with journal._git_admin_authority_batch():
                pass
        else:
            with journal._bound_git_admin_path(ctx["git_dir"] / "config"):
                pass

    with journals.workspace_mutation_lock(root):
        _expect_answer_from_memory(root, git_calls, body=through_helper)
        undo = _rename_away(ctx["git_dir"] / "HEAD")
        try:
            refused = _observe(git_calls, through_helper)
            default = _observe(git_calls, lambda: _default(root))
        finally:
            undo()
        restored = _observe(git_calls, through_helper)

    assert refused.refusal is not None and _UNREADABLE in refused.refusal
    assert refused == default
    assert restored.refusal is None
    assert journal.restore()


# ------------------------------------------- the preflight at the bracket call

def _include_in_config(ctx: dict[str, Path]) -> Undo:
    extra = ctx["tmp"] / "extra.config"
    extra.write_text("[user]\n\tname = included\n", encoding="utf-8")
    return _append(
        ctx.get("common", ctx["git_dir"]) / "config",
        f"[include]\n\tpath = {extra}\n".encode(),
    )


def _include_in_worktree_config(ctx: dict[str, Path]) -> Undo:
    extra = ctx["tmp"] / "extra.config"
    extra.write_text("[user]\n\tname = included\n", encoding="utf-8")
    target = ctx["git_dir"] / "config.worktree"
    target.write_bytes(f"[include]\n\tpath = {extra}\n".encode())
    return target.unlink


def _config_parked_as_directory(ctx: dict[str, Path]) -> Undo:
    config = ctx["git_dir"] / "config"
    undo_rename = _rename_away(config)
    config.mkdir()

    def undo() -> None:
        config.rmdir()
        undo_rename()

    return undo


_PREFLIGHT_STATES: dict[
    str, tuple[str, Callable[[dict[str, Path]], Undo]]
] = {
    "include-in-local-config": ("plain", _include_in_config),
    "include-in-worktree-common-config": ("worktree", _include_in_config),
    "include-in-config-worktree": ("worktree", _include_in_worktree_config),
    "config-missing": (
        "plain", lambda ctx: _rename_away(ctx["git_dir"] / "config")),
    "config-parked-as-directory": ("plain", _config_parked_as_directory),
}


@pytest.mark.parametrize("name", sorted(_PREFLIGHT_STATES))
@steady_wall_clock
def test_config_preflight_refuses_at_the_bracket_call_without_git(
    name: str,
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    layout, state = _PREFLIGHT_STATES[name]
    ctx = _LAYOUTS[layout](tmp_path)
    root = ctx["root"]
    with journals.workspace_mutation_lock(root):
        _expect_answer_from_memory(root, git_calls)
        undo = state(ctx)
        try:
            refused = _observe(git_calls, lambda: _bracket(root))
            default = _observe(git_calls, lambda: _default(root))
        finally:
            undo()
        restored = _observe(git_calls, lambda: _bracket(root))

    assert refused.refusal is not None
    assert refused == default
    assert refused.git == 0
    assert restored.refusal is None


# --------------------------------------------------- shapes without a witness

def _head_as_symlink(ctx: dict[str, Path], _monkeypatch: object) -> None:
    # The symbolic-link form of a symbolic ref, which Git still reads.
    head = ctx["git_dir"] / "HEAD"
    head.unlink()
    os.symlink("refs/heads/main", head)


def _empty_git_directory_above(ctx: dict[str, Path], _monkeypatch: object) -> None:
    (ctx["parent"] / ".git").mkdir()


def _head_file_above(ctx: dict[str, Path], _monkeypatch: object) -> None:
    (ctx["tmp"] / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")


def _config_with_include_word(ctx: dict[str, Path], _monkeypatch: object) -> None:
    _append(ctx["git_dir"] / "config", b"# include nothing\n")


def _config_with_worktree_word(ctx: dict[str, Path], _monkeypatch: object) -> None:
    _git(ctx["root"], "config", "branch.worktree-lane.remote", "origin")


def _objects_as_symlink(ctx: dict[str, Path], _monkeypatch: object) -> None:
    objects = ctx["git_dir"] / "objects"
    target = ctx["git_dir"] / "objects.target"
    objects.rename(target)
    os.symlink(target, objects)


def _oversized_config(ctx: dict[str, Path], _monkeypatch: object) -> None:
    config = ctx["git_dir"] / "config"
    padding = b"".join(b"# padding line %05d\n" % index for index in range(4096))
    config.write_bytes(config.read_bytes() + padding)
    assert config.stat().st_size > 64 * 1024


def _relative_path_entry(
    _ctx: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PATH", f"relative-bin{os.pathsep}{os.environ.get('PATH', '')}",
    )


def _config_with_word_in_upper_case(
    ctx: dict[str, Path],
    _monkeypatch: object,
) -> None:
    # Git matches the names of sections and keys without regard to case.
    _git(ctx["root"], "config", "branch.WorkTree-lane.remote", "origin")
    content = (ctx["git_dir"] / "config").read_bytes()
    assert b"worktree" not in content and b"worktree" in content.lower()


def _symlinked_path_entry(
    ctx: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symbolic link to an empty directory of the test first on ``PATH``."""

    target = ctx["tmp"] / "linked-bin"
    target.mkdir()
    link = ctx["tmp"] / "link-to-bin"
    os.symlink(target, link)
    monkeypatch.setenv("PATH", f"{link}{os.pathsep}{os.environ.get('PATH', '')}")


def _file_named_git_that_cannot_run(
    ctx: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``git`` without execute permission ahead of the program."""

    _front_directory(ctx, monkeypatch)
    (ctx["front"] / "git").write_bytes(b"not for running")
    (ctx["front"] / "git").chmod(0o644)


def _directory_named_git(
    ctx: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _front_directory(ctx, monkeypatch)
    (ctx["front"] / "git").mkdir()


def _loader_variable(
    ctx: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A variable that sends the loader of the Git program to a directory."""

    libraries = ctx["tmp"] / "libraries"
    libraries.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(libraries))


def _loader_variable_of_another_loader(
    ctx: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same for the loader of a platform that reads ``DYLD_`` variables."""

    libraries = ctx["tmp"] / "libraries"
    libraries.mkdir()
    monkeypatch.setenv("DYLD_LIBRARY_PATH", str(libraries))


def _git_file_through_a_symlink(ctx: dict[str, Path], _monkeypatch: object) -> None:
    """The ``.git`` file names the Git directory through a symbolic link.

    The link lies beside the Git directory.  Git and the journal's own
    rediscovery resolve it, so the checkout is admitted.
    """

    alias = ctx["git_dir"].with_name("alias")
    os.symlink(ctx["git_dir"], alias)
    (ctx["root"] / ".git").write_bytes(b"gitdir: " + os.fsencode(alias) + b"\n")


def _git_file_through_another_directory(
    ctx: dict[str, Path],
    _monkeypatch: object,
) -> None:
    """The ``.git`` file leaves the held paths and comes back with ``..``."""

    (ctx["tmp"] / "other").mkdir()
    (ctx["root"] / ".git").write_bytes(
        b"gitdir: " + os.fsencode(_way_through_another_directory(ctx)) + b"\n"
    )


def _way_through_another_directory(ctx: dict[str, Path]) -> str:
    return os.path.join(
        ctx["tmp"], "other", "..", "main", ".git", "worktrees", "wt",
    )


def _commondir_through_a_symlink(ctx: dict[str, Path], _monkeypatch: object) -> None:
    """``commondir`` names the common directory through a symbolic link."""

    alias = ctx["tmp"] / "common-alias"
    os.symlink(ctx["common"], alias)
    (ctx["git_dir"] / "commondir").write_bytes(os.fsencode(alias) + b"\n")


_UNWITNESSED_SHAPES: dict[
    str,
    tuple[str, Callable[[dict[str, Path], pytest.MonkeyPatch], None]],
] = {
    "empty-git-directory-above": ("nested", _empty_git_directory_above),
    "head-file-above": ("nested", _head_file_above),
    "head-a-symlink": ("plain", _head_as_symlink),
    "objects-a-symlink": ("plain", _objects_as_symlink),
    "control-file-over-64-kib": ("plain", _oversized_config),
    "config-names-an-include": ("plain", _config_with_include_word),
    "config-names-a-worktree": ("plain", _config_with_worktree_word),
    "config-names-a-worktree-in-upper-case": (
        "plain", _config_with_word_in_upper_case),
    "relative-path-entry-before-the-program": ("plain", _relative_path_entry),
    "pass-through-git-of-the-user-first-on-path": ("plain", _front_program),
    "path-entry-a-symlink-before-the-program": ("plain", _symlinked_path_entry),
    "file-named-git-that-cannot-run-before-the-program": (
        "plain", _file_named_git_that_cannot_run),
    "directory-named-git-before-the-program": ("plain", _directory_named_git),
    "loader-variable-in-the-environment": ("plain", _loader_variable),
    "loader-variable-of-another-loader-in-the-environment": (
        "plain", _loader_variable_of_another_loader),
    "git-file-leads-through-a-symlink": ("worktree", _git_file_through_a_symlink),
    "git-file-leads-through-another-directory-and-back": (
        "worktree", _git_file_through_another_directory),
    "commondir-leads-through-a-symlink": (
        "worktree", _commondir_through_a_symlink),
}


@pytest.mark.parametrize("name", sorted(_UNWITNESSED_SHAPES))
def test_shape_without_a_witness_asks_git_at_every_bracket_call(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    layout, shape = _UNWITNESSED_SHAPES[name]
    ctx = _LAYOUTS[layout](tmp_path)
    root = ctx["root"]
    shape(ctx, monkeypatch)
    with journals.workspace_mutation_lock(root) as authority:
        _settle()
        start = len(git_calls.args)
        first = _observe(git_calls, lambda: _bracket(root))
        second = _observe(git_calls, lambda: _bracket(root))
        assert first == _Outcome(None, 2)
        assert second == _Outcome(None, 2)
        assert git_calls.discoveries(start) == 2
        assert not _remembered(authority)


@steady_wall_clock
def test_superuser_gets_no_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A holder that could replace the Git program is never answered from memory.

    Asked of the witness itself: the lock's own ownership checks read the
    effective user too, so an assertion call cannot run with it replaced.
    """

    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    with journals.workspace_mutation_lock(root) as authority:
        _settle(root)
        with monkeypatch.context() as scoped:
            scoped.setattr(os, "geteuid", lambda: 0)
            assert not _witness_exists(authority)
        # That says something only where the same checkout has a witness for
        # another holder.
        _require_witness(authority, root)


@steady_wall_clock
def test_change_inside_the_settle_time_is_not_remembered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """A change is witnessed only once it is as old as the settle time.

    The clock of the journal module is set from the change time itself, so
    the speed of the machine decides nothing here.
    """

    settle = journals._DISCOVERY_WITNESS_SETTLE_NS
    assert settle == 20_000_000
    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    head = ctx["git_dir"] / "HEAD"
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        # The same bytes again: only the stamps of the file move.
        head.write_bytes(head.read_bytes())
        changed = os.lstat(head).st_ctime_ns
        clock = _Clock(monkeypatch)
        # One nanosecond short of the settle time: Git is asked at every
        # bracket call and nothing is remembered.
        clock.now_ns = changed + settle - 1
        start = len(git_calls.args)
        assert _observe(git_calls, lambda: _bracket(root)) == _Outcome(None, 2)
        assert _observe(git_calls, lambda: _bracket(root)) == _Outcome(None, 2)
        assert git_calls.discoveries(start) == 2
        assert not _remembered(authority)
        # At the settle time: one discovery, then none.
        clock.now_ns = changed + settle
        assert _observe(git_calls, lambda: _bracket(root)) == _Outcome(None, 2)
        from_memory(_remembered(authority))
        _expect_no_git(git_calls, lambda: _bracket(root))


@steady_wall_clock
def test_witnessed_change_costs_one_discovery_after_the_settle_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        _git(root, "checkout", "-q", "-b", "one")
        _settle(root)
        assert _observe(git_calls, lambda: _bracket(root)) == _Outcome(None, 2)
        _expect_no_git(git_calls, lambda: _bracket(root))
        # A default call asks Git anyway and leaves a fresh answer behind.
        _git(root, "checkout", "-q", "-b", "two")
        _settle(root)
        assert _observe(git_calls, lambda: _default(root)) == _Outcome(None, 2)
        from_memory(_remembered(authority))
        _expect_no_git(git_calls, lambda: _bracket(root))
        # Inside the settle time a default call cannot leave one behind: the
        # journal module's clock stands one nanosecond behind the change.
        _git(root, "checkout", "-q", "-b", "three")
        clock = _Clock(monkeypatch)
        clock.now_ns = os.lstat(ctx["git_dir"] / "HEAD").st_ctime_ns + 1
        assert _observe(git_calls, lambda: _default(root)) == _Outcome(None, 2)
        assert not _remembered(authority)


# ------------------------------------------------------------- nothing stale

@pytest.mark.parametrize("refusing_call", ["bracket", "default"])
@steady_wall_clock
def test_refusal_leaves_nothing_remembered(
    refusing_call: str,
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    ctx = _layout_nested(tmp_path)
    root = ctx["root"]
    call = _bracket if refusing_call == "bracket" else _default
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        # Undoing this state leaves every stamp the witness holds as it was,
        # so only an emptied memory makes the next bracket call ask Git.
        undo = _gitlink_in_parent(ctx)
        try:
            refused = _observe(git_calls, lambda: call(root))
        finally:
            undo()
        assert refused.refusal is not None and _CHANGED in refused.refusal
        assert not _remembered(authority)
        assert _observe(git_calls, lambda: _bracket(root)) == _Outcome(None, 2)

        undo = _rename_away(ctx["git_dir"] / "HEAD")
        try:
            refused = _observe(git_calls, lambda: call(root))
        finally:
            undo()
        assert refused.refusal is not None and _UNREADABLE in refused.refusal
        assert not _remembered(authority)
        assert _observe(git_calls, lambda: _bracket(root)).git == 2


def _good_only_while_git_is_asked_once(
    monkeypatch: pytest.MonkeyPatch,
    make_good: Callable[[], None],
    make_bad: Callable[[], None],
) -> dict[str, bool]:
    """The next Git question of the assertion sees the good entry, once.

    The wrapper stands for another process that changes an entry twice while
    the holder asks Git a single time.
    """

    original = journals._workspace_authority_common_dir
    shot = {"fired": False}

    def flipping(cwd: Path, *, root_fd: int | None = None) -> Path:
        if shot["fired"]:
            return original(cwd, root_fd=root_fd)
        shot["fired"] = True
        make_good()
        try:
            return original(cwd, root_fd=root_fd)
        finally:
            make_bad()

    monkeypatch.setattr(journals, "_workspace_authority_common_dir", flipping)
    return shot


def _flicker_in_place(
    path: Path,
    offset: int,
    payload: bytes,
) -> tuple[Undo, Undo]:
    good = path.read_bytes()[offset:offset + len(payload)]
    return (
        lambda: _pwrite(path, good, offset),
        lambda: _pwrite(path, payload, offset),
    )


def _flicker_renamed(path: Path) -> tuple[Undo, Undo]:
    away = path.with_name(path.name + ".away")

    def make_good() -> None:
        if not os.path.lexists(path):
            away.rename(path)

    def make_bad() -> None:
        path.rename(away)

    return make_good, make_bad


def _flicker_unsearchable(path: Path) -> tuple[Undo, Undo]:
    mode = stat.S_IMODE(path.lstat().st_mode)
    return (lambda: path.chmod(mode), lambda: path.chmod(0))


_FLICKERS: dict[str, Callable[[dict[str, Path]], tuple[Undo, Undo]]] = {
    "head-written-in-place": lambda ctx: _flicker_in_place(
        ctx["git_dir"] / "HEAD", 0, b"x"),
    "config-written-in-place": lambda ctx: _flicker_in_place(
        ctx["git_dir"] / "config",
        _format_version_offset(ctx["git_dir"] / "config"), b"9"),
    "head-renamed": lambda ctx: _flicker_renamed(ctx["git_dir"] / "HEAD"),
    "objects-renamed": lambda ctx: _flicker_renamed(ctx["git_dir"] / "objects"),
    "refs-renamed": lambda ctx: _flicker_renamed(ctx["git_dir"] / "refs"),
    "objects-unsearchable": lambda ctx: _flicker_unsearchable(
        ctx["git_dir"] / "objects"),
    "refs-unsearchable": lambda ctx: _flicker_unsearchable(
        ctx["git_dir"] / "refs"),
}


@pytest.mark.parametrize("waited", [True, False], ids=["settled", "at-once"])
@pytest.mark.parametrize("name", sorted(_FLICKERS))
@steady_wall_clock
def test_entry_good_only_during_the_git_question_is_not_remembered(
    name: str,
    waited: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    make_good, make_bad = _FLICKERS[name](ctx)
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        make_bad()
        try:
            if waited:
                _settle(root)
            shot = _good_only_while_git_is_asked_once(
                monkeypatch, make_good, make_bad,
            )
            during = _observe(git_calls, lambda: _bracket(root))
            assert shot["fired"]
            assert during == _Outcome(None, 2)
            assert not _remembered(authority)
            refused = _observe(git_calls, lambda: _bracket(root))
            default = _observe(git_calls, lambda: _default(root))
        finally:
            make_good()

    assert refused.refusal is not None and _UNREADABLE in refused.refusal
    assert refused == default


@pytest.mark.parametrize("hidden_by", ["renamed-away", "held-open-for-writing"])
@pytest.mark.parametrize("waited", [True, False], ids=["settled", "at-once"])
@steady_wall_clock
def test_program_hidden_during_the_git_question_is_not_remembered(
    hidden_by: str,
    waited: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    _front_directory(ctx, monkeypatch)
    hide, show = _program_hiding(ctx, hidden_by)
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        _write_program(ctx["front"] / "git", differing=True)
        try:
            if waited:
                _settle(root)
            shot = _good_only_while_git_is_asked_once(monkeypatch, hide, show)
            during = _observe(git_calls, lambda: _bracket(root))
            assert shot["fired"]
            assert during == _Outcome(None, 2)
            assert not _remembered(authority)
            refused = _observe(git_calls, lambda: _bracket(root))
            default = _observe(git_calls, lambda: _default(root))
        finally:
            (ctx["front"] / "git").unlink()

    assert refused.refusal is not None and _CHANGED in refused.refusal
    assert refused == default


def _program_hiding(ctx: dict[str, Path], hidden_by: str) -> tuple[Undo, Undo]:
    """Hide a differing ``git`` from one Git question, then show it again."""

    front = ctx["front"]
    if hidden_by == "renamed-away":
        away = front.with_name(front.name + ".away")

        def hide() -> None:
            front.rename(away)

        def show() -> None:
            away.rename(front)

        return hide, show
    # A child cannot execute a file that is open for writing; it runs the
    # next ``git`` on PATH instead, and no stamp of the file moves.
    held: list[int] = []
    return (
        lambda: held.append(os.open(front / "git", os.O_WRONLY)),
        lambda: os.close(held.pop()),
    )


def _mapped_first_byte(path: Path) -> tuple[mmap.mmap, bytes, tuple[int, int]]:
    """Map a file for writing and store once, so later stores move no stamp."""

    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
        view = mmap.mmap(
            descriptor, 0, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(descriptor)
    good = bytes(view[0:1])
    view[0:1] = good
    time.sleep(_SETTLE_S)
    return view, good, _stamp(path)


def _stamp(path: Path) -> tuple[int, int]:
    metadata = os.lstat(path)
    return (metadata.st_mtime_ns, metadata.st_ctime_ns)


@steady_wall_clock
def test_mapped_store_during_the_git_question_is_refused_by_the_next_default_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """A store through a shared mapping moves no stamp and equal bytes.

    No witness can tell such a store from none, so what is pinned is the
    next call that asks Git: it refuses and leaves nothing remembered.
    """

    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    head = ctx["git_dir"] / "HEAD"
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        view, good, stamp = _mapped_first_byte(head)
        try:
            view[0:1] = b"x"
            shot = _good_only_while_git_is_asked_once(
                monkeypatch,
                lambda: view.__setitem__(slice(0, 1), good),
                lambda: view.__setitem__(slice(0, 1), b"x"),
            )
            during = _observe(git_calls, lambda: _bracket(root))
            assert shot["fired"]
            assert during == _Outcome(None, 2)
            refused = _observe(git_calls, lambda: _default(root))
            moved = _stamp(head) != stamp
            assert refused.refusal is not None and _UNREADABLE in refused.refusal
            assert not _remembered(authority)
        finally:
            view[0:1] = good
            view.close()
    if moved:
        pytest.skip("the file system stamped a store through the mapping")


def _program_appearing(
    ctx: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    how: str,
) -> tuple[Undo, Undo]:
    """A way for a ``git`` to come to lie ahead of the program, and to go.

    The program answers the discovery as a readable checkout would.  Returns
    the two steps; ``PATH`` is set here, before the lock is taken.
    """

    tmp = ctx["tmp"]

    def first_on_path(directory: Path) -> None:
        monkeypatch.setenv(
            "PATH", f"{directory}{os.pathsep}{os.environ.get('PATH', '')}",
        )

    if how == "created-in-a-path-directory":
        front = tmp / "front-bin"
        front.mkdir()
        first_on_path(front)
        program = front / "git"
        return (
            lambda: _write_answering_program(program, ctx["git_dir"]),
            program.unlink,
        )
    if how == "missing-path-directory-created":
        missing = tmp / "not-there"
        first_on_path(missing / "bin")

        def create() -> None:
            (missing / "bin").mkdir(parents=True)
            _write_answering_program(missing / "bin" / "git", ctx["git_dir"])

        return create, lambda: shutil.rmtree(missing)
    # A prepared directory takes the place of one on the way to the PATH
    # directory, by plain renames; nothing is created or removed there.
    above = tmp / "front"
    (above / "bin").mkdir(parents=True)
    first_on_path(above / "bin")
    exchanged = above / "bin" if how == "path-directory-exchanged" else above
    prepared = exchanged.with_name(exchanged.name + ".prepared")
    parked = exchanged.with_name(exchanged.name + ".parked")
    holder = prepared if exchanged.name == "bin" else prepared / "bin"
    holder.mkdir(parents=True)
    _write_answering_program(holder / "git", ctx["git_dir"])

    def exchange() -> None:
        exchanged.rename(parked)
        prepared.rename(exchanged)

    def restore() -> None:
        exchanged.rename(prepared)
        parked.rename(exchanged)

    return exchange, restore


@pytest.mark.parametrize("how", [
    "created-in-a-path-directory",
    "missing-path-directory-created",
    "path-directory-exchanged",
    "directory-above-a-path-directory-exchanged",
])
@steady_wall_clock
def test_program_present_only_during_the_git_question_is_not_remembered(
    how: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """A ``git`` that lies ahead of the program only while Git is asked once.

    It answers for a checkout Git itself cannot read.  However it comes and
    goes, a directory on the way to it changes, so nothing is remembered and
    the next bracket call refuses as the default call does.
    """

    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    appear, vanish = _program_appearing(ctx, monkeypatch, how)
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        undo = _rewrite(ctx["git_dir"] / "HEAD", b"")
        try:
            _settle(root)
            unreadable = _observe(git_calls, lambda: _bracket(root))
            shot = _good_only_while_git_is_asked_once(monkeypatch, appear, vanish)
            during = _observe(git_calls, lambda: _bracket(root))
            assert shot["fired"]
            assert during == _Outcome(None, 2)
            assert not _remembered(authority)
            refused = _observe(git_calls, lambda: _bracket(root))
            default = _observe(git_calls, lambda: _default(root))
        finally:
            undo()

    assert refused.refusal is not None and _UNREADABLE in refused.refusal
    assert refused == default == unreadable


@steady_wall_clock
def test_program_gone_long_before_the_git_question_ends_is_not_remembered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """A ``git`` ahead of the program that goes well before the question ends.

    A Git question can outlast the settle time.  The directory the entry
    lay in has then settled again when the question ends, and only its
    change time, compared with the one before the question, shows that an
    entry came and went.  The clock of the journal module is set ahead
    when the entry goes, so the length of the question decides nothing.
    """

    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    appear, vanish = _program_appearing(
        ctx, monkeypatch, "created-in-a-path-directory",
    )
    clock = _Clock(monkeypatch)

    def vanish_long_ago() -> None:
        vanish()
        clock.now_ns = time.time_ns() + 60_000_000_000

    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        undo = _rewrite(ctx["git_dir"] / "HEAD", b"")
        try:
            _settle(root)
            unreadable = _observe(git_calls, lambda: _bracket(root))
            shot = _good_only_while_git_is_asked_once(
                monkeypatch, appear, vanish_long_ago,
            )
            during = _observe(git_calls, lambda: _bracket(root))
            clock.now_ns = None
            assert shot["fired"]
            assert during == _Outcome(None, 2)
            assert not _remembered(authority)
            refused = _observe(git_calls, lambda: _bracket(root))
            default = _observe(git_calls, lambda: _default(root))
        finally:
            clock.now_ns = None
            undo()

    assert refused.refusal is not None and _UNREADABLE in refused.refusal
    assert refused == default == unreadable


_EXCHANGED_DIRECTORIES: dict[str, tuple[str, Callable[[dict[str, Path]], Path]]] = {
    "git-directory-of-the-checkout": ("plain", lambda ctx: ctx["root"] / ".git"),
    "main-checkout-of-a-linked-worktree": ("worktree", lambda ctx: ctx["main"]),
}


@pytest.mark.parametrize("name", sorted(_EXCHANGED_DIRECTORIES))
@steady_wall_clock
def test_directory_exchanged_during_the_git_question_is_not_remembered(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """A directory on a held path exchanged for a readable copy, once.

    ``HEAD`` is unreadable at rest.  Plain renames put a copy made before
    that in the directory's place for the length of one Git question and the
    original back.  No entry Git reads has moved afterwards, but the
    change times of the exchanged directory and of the one it lies in
    have: nothing is remembered and the next bracket call refuses as the
    default call does.
    """

    layout, exchanged = _EXCHANGED_DIRECTORIES[name]
    ctx = _LAYOUTS[layout](tmp_path)
    root = ctx["root"]
    original = exchanged(ctx)
    copy = tmp_path / "exchange.copy"
    parked = tmp_path / "exchange.parked"

    def exchange() -> None:
        original.rename(parked)
        copy.rename(original)

    def restore() -> None:
        original.rename(copy)
        parked.rename(original)

    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        shutil.copytree(original, copy, symlinks=True)
        undo = _rewrite(ctx["git_dir"] / "HEAD", b"")
        try:
            _settle(root)
            unreadable = _observe(git_calls, lambda: _bracket(root))
            shot = _good_only_while_git_is_asked_once(monkeypatch, exchange, restore)
            during = _observe(git_calls, lambda: _bracket(root))
            assert shot["fired"]
            assert during == _Outcome(None, 2)
            assert not _remembered(authority)
            refused = _observe(git_calls, lambda: _bracket(root))
            default = _observe(git_calls, lambda: _default(root))
        finally:
            undo()

    assert refused.refusal is not None
    assert refused == default == unreadable


def _stand_in_git_directory(ctx: dict[str, Path], path: Path) -> None:
    """A directory Git takes for the Git directory of the linked worktree.

    Its ``HEAD`` is a copy made while the real one was readable, and its
    ``commondir`` names the real common directory.
    """

    path.mkdir(parents=True)
    for name in ("HEAD", "gitdir"):
        (path / name).write_bytes((ctx["git_dir"] / name).read_bytes())
    (path / "commondir").write_bytes(os.fsencode(ctx["common"]) + b"\n")


def _way_led_elsewhere(
    ctx: dict[str, Path],
    placement: str,
) -> tuple[bytes, Undo, Undo]:
    """A way to the Git directory that can be led elsewhere for a moment.

    Returns what the ``.git`` file is to name, and the two steps that lead
    that way to a stand-in Git directory and back to the real one.
    """

    tmp, git_dir = ctx["tmp"], ctx["git_dir"]
    if placement == "another-directory-and-back":
        # No link at rest: a real directory beside the checkouts is exchanged
        # for a link into a prepared tree.
        other = tmp / "other"
        other.mkdir()
        parked = tmp / "other.parked"
        prepared = tmp / "prepared"
        (prepared / "inside").mkdir(parents=True)
        _stand_in_git_directory(
            ctx, prepared / "main" / ".git" / "worktrees" / "wt",
        )

        def lead_away() -> None:
            other.rename(parked)
            os.symlink(prepared / "inside", other)

        def lead_back() -> None:
            other.unlink()
            parked.rename(other)

        return (
            os.fsencode(_way_through_another_directory(ctx)),
            lead_away,
            lead_back,
        )

    if placement == "symlink-beside-the-git-directory":
        links = git_dir.parent
    else:
        # Nothing below a held path is touched.  The config preflight reads
        # the way as text and so looks for a config two levels above the link.
        links = tmp / "elsewhere" / "worktrees"
        links.mkdir(parents=True)
        (tmp / "elsewhere" / "config").write_bytes(b"[core]\n")
    alias = links / "alias"
    stand_in = links / "stand-in"
    _stand_in_git_directory(ctx, stand_in)
    os.symlink(git_dir, alias)

    def retarget() -> None:
        alias.unlink()
        os.symlink(stand_in, alias)

    def put_back() -> None:
        alias.unlink()
        os.symlink(git_dir, alias)

    return os.fsencode(alias), retarget, put_back


@pytest.mark.parametrize("placement", [
    "symlink-beside-the-git-directory",
    "symlink-outside-every-held-path",
    "another-directory-and-back",
])
@steady_wall_clock
def test_way_led_elsewhere_during_the_git_question_is_not_remembered(
    placement: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """A ``.git`` file whose way to the Git directory leaves the held paths.

    The file is rewritten under the lock to go through a symbolic link, or
    through another directory and ``..``.  Git and the journal's own
    rediscovery resolve that way to the held directories, so the checkout
    stays admitted.  ``HEAD`` is unreadable at rest.  For the length of one
    Git question the way leads to a stand-in Git directory with a readable
    ``HEAD``; nothing of the checkout or its Git directories has to move
    for that.  Such a way has no witness: nothing is remembered and the
    next bracket call refuses as the default call does.
    """

    ctx = _layout_worktree(tmp_path)
    root = ctx["root"]
    way, lead_away, lead_back = _way_led_elsewhere(ctx, placement)
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        undo_way = _rewrite(root / ".git", b"gitdir: " + way + b"\n")
        try:
            _settle(root)
            admitted = _observe(git_calls, lambda: _bracket(root))
            undo_head = _rewrite(ctx["git_dir"] / "HEAD", b"")
            try:
                _settle(root)
                unreadable = _observe(git_calls, lambda: _bracket(root))
                shot = _good_only_while_git_is_asked_once(
                    monkeypatch, lead_away, lead_back,
                )
                during = _observe(git_calls, lambda: _bracket(root))
                assert shot["fired"]
                assert during == _Outcome(None, 2)
                assert not _remembered(authority)
                refused = _observe(git_calls, lambda: _bracket(root))
                default = _observe(git_calls, lambda: _default(root))
            finally:
                undo_head()
        finally:
            undo_way()

    assert admitted == _Outcome(None, 2)
    assert refused.refusal is not None
    assert refused == default == unreadable


@steady_wall_clock
def test_way_spelled_over_held_directories_keeps_its_witness(
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    """A relative way, as Git itself can write it, costs nothing.

    It leaves the checkout with ``..`` for the directory the checkout lies
    in and goes on over the directories of the held Git directory.
    """

    ctx = _layout_worktree(tmp_path)
    root = ctx["root"]
    relative = os.path.relpath(ctx["git_dir"], root)
    assert relative.startswith(".." + os.sep)
    with journals.workspace_mutation_lock(root):
        _expect_answer_from_memory(root, git_calls)
        undo = _rewrite(root / ".git", b"gitdir: " + os.fsencode(relative) + b"\n")
        try:
            _expect_answer_from_memory(root, git_calls)
        finally:
            undo()


_WAYS_NOT_OVER_HELD_DIRECTORIES: dict[str, Callable[[dict[str, Path]], str]] = {
    "ends-above-the-git-directory": lambda ctx: os.fspath(ctx["git_dir"].parent),
    "begins-with-two-slashes": lambda ctx: "/" + os.fspath(ctx["git_dir"]),
}


@pytest.mark.parametrize("way", sorted(_WAYS_NOT_OVER_HELD_DIRECTORIES))
@steady_wall_clock
def test_way_that_does_not_end_at_the_held_git_directory_has_no_witness(
    way: str,
    tmp_path: Path,
) -> None:
    """Asked of the witness itself.

    The assertion's own rediscovery refuses a ``.git`` file that names
    another directory before the witness is looked at, so an assertion
    call cannot show what the witness makes of such a file.
    """

    ctx = _layout_worktree(tmp_path)
    root = ctx["root"]
    named = _WAYS_NOT_OVER_HELD_DIRECTORIES[way](ctx)
    with journals.workspace_mutation_lock(root) as authority:
        _settle(root)
        _require_witness(authority, root)
        undo = _rewrite(root / ".git", b"gitdir: " + os.fsencode(named) + b"\n")
        try:
            _settle(root)
            assert not _witness_exists(authority)
        finally:
            undo()
        _settle(root)
        _require_witness(authority, root)


@steady_wall_clock
def test_held_path_that_is_not_normalized_has_no_witness(tmp_path: Path) -> None:
    """The witness reads a held path as text.

    Above a directory lies the directory that the text of its path names
    only when no component is ``..``.  The lock holds resolved paths; the
    witness does not rest on that.
    """

    ctx = _layout_nested(tmp_path)
    root = ctx["root"]
    (ctx["parent"] / "beside").mkdir()
    above = Path(os.path.realpath(ctx["parent"]))

    def respelled(path: Path) -> Path:
        return Path(os.path.join(above, "beside", "..", path.relative_to(above)))

    with journals.workspace_mutation_lock(root) as authority:
        _settle(root)
        _require_witness(authority, root)
        # The same four directories, each named through a neighbour and "..".
        other = replace(
            authority,  # type: ignore[type-var]
            root=respelled(authority.root),
            common_dir=respelled(authority.common_dir),
            git_common_dir=respelled(authority.git_common_dir),
            git_dir=respelled(authority.git_dir),
        )
        assert os.fspath(other.root) != os.fspath(authority.root)
        assert other.root.samefile(authority.root)
        assert other.git_dir.samefile(authority.git_dir)
        assert not _witness_exists(other)


def _libraries_of_the_git_program() -> list[str]:
    """Shared libraries the Git program loads besides the C library."""

    try:
        listed = subprocess.run(
            ["ldd", _REAL_GIT],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    names: list[str] = []
    for line in listed.splitlines():
        name = line.strip().split(" ")[0]
        if "=>" in line and not name.startswith(("libc.so", "linux-vdso", "ld-")):
            names.append(os.path.basename(name))
    return names


def test_library_the_loader_is_sent_to_is_refused_at_the_bracket_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """A loader variable names a directory of the holder's user.

    A file placed there under the name of a library of the Git program
    stops every Git process, and it stays.  The witness holds the variable
    and not the file, so with such a variable there is no witness: the
    bracket call refuses as the default call does, at every call.
    """

    names = _libraries_of_the_git_program()
    if not names:
        pytest.skip("the Git program loads no library that can be shadowed here")
    libraries = tmp_path / "libraries"
    libraries.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(libraries))
    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    placed = [libraries / name for name in names]
    with journals.workspace_mutation_lock(root):
        _settle(root)
        assert _observe(git_calls, lambda: _bracket(root)).refusal is None
        for path in placed:
            path.write_bytes(b"this is no library\n")
        try:
            _settle(root)
            refused = _observe(git_calls, lambda: _bracket(root))
            again = _observe(git_calls, lambda: _bracket(root))
            default = _observe(git_calls, lambda: _default(root))
        finally:
            for path in placed:
                path.unlink()
        restored = _observe(git_calls, lambda: _bracket(root))

    if default.refusal is None:
        pytest.skip("a file that is no library does not stop the Git program here")
    assert refused == default
    assert again == default
    assert _UNREADABLE in default.refusal
    assert restored.refusal is None


def _witness_exists(authority: object) -> bool:
    """Whether a witness can be taken now.

    The witness holds the process environment.  A test asserts on this
    answer and never on the witness itself, which a failing assertion
    would print.
    """

    return journals._authority_discovery_witness(authority) is not None  # type: ignore[arg-type]


def _require_witness(authority: object, root: Path) -> None:
    """Skip where the checkout allows no witness; a settled one must have one."""

    if not _witness_exists(authority):
        reason = witness_unavailable_reason(root)
        if reason is not None:
            pytest.skip(f"no answer can be remembered here: {reason}")
        from_memory(False, "a settled, unchanged checkout has no witness")


@steady_wall_clock
def test_byte_stored_through_an_established_mapping_is_refused(
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    """The bytes of a control file are part of the witness.

    A store through a shared mapping that already stored once moves no
    stamp.  The changed byte stays, so the bytes differ from the remembered
    ones: the bracket call asks Git and refuses as the default call does.
    """

    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    head = ctx["git_dir"] / "HEAD"
    with journals.workspace_mutation_lock(root):
        view, good, stamp = _mapped_first_byte(head)
        try:
            _expect_answer_from_memory(root, git_calls)
            view[0:1] = b"x"
            moved = _stamp(head) != stamp
            refused = _observe(git_calls, lambda: _bracket(root))
            default = _observe(git_calls, lambda: _default(root))
        finally:
            view[0:1] = good
            view.close()
    if moved:
        pytest.skip("the file system stamped a store through the mapping")
    assert refused.refusal is not None and _UNREADABLE in refused.refusal
    assert refused == default


@steady_wall_clock
def test_directory_changed_but_not_exchanged_costs_no_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """A directory on a held path changes whenever an entry of it does.

    The witness holds such a directory by identity, so entries created in
    the checkout and beside it between two calls cost nothing.  Change
    times count only while Git is asked.  An exchange of the directory
    moves two of them, its own and that of the directory it lies in; one
    that stands as it was excludes it.  So the answer to a question during
    which an entry was created at the top level of the checkout is kept.
    """

    ctx = _layout_nested(tmp_path)
    root = ctx["root"]
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        (root / "created-at-the-top-level").mkdir()
        (ctx["parent"] / "created-beside-the-checkout").mkdir()
        _expect_no_git(git_calls, lambda: _bracket(root))
        _settle(root)
        shot = _good_only_while_git_is_asked_once(
            monkeypatch,
            (root / "created-while-git-is-asked").mkdir,
            lambda: None,
        )
        assert _observe(git_calls, lambda: _default(root)) == _Outcome(None, 2)
        assert shot["fired"]
        from_memory(
            _remembered(authority),
            "one of the two change times of the checkout stood",
        )
        _expect_no_git(git_calls, lambda: _bracket(root))


@pytest.mark.parametrize("guarding", [
    "path-directory-ahead-of-the-program",
    "checkout-and-the-directory-it-lies-in",
])
@steady_wall_clock
def test_answer_is_not_kept_while_a_guarding_change_time_has_not_settled(
    guarding: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """A change time that guards a Git question must have settled too.

    A directory changed a moment before Git is asked, so a further change
    during the question might not move its change time again: a ``PATH``
    directory ahead of the Git program, in which an entry could be
    created; or the checkout and the directory it lies in, which leaves
    neither of the two times that would show an exchange.  The directories
    are held by identity, so the remembered answer is still reused.  But
    the answer to a question asked then is not kept, and once the time
    has settled it is: for an exchange the older of the two is enough.
    The clock of the journal module is set from the change time itself.
    """

    settle = journals._DISCOVERY_WITNESS_SETTLE_NS
    ctx = _layout_nested(tmp_path)
    root = ctx["root"]
    front = tmp_path / "front-bin"
    front.mkdir()
    monkeypatch.setenv(
        "PATH", f"{front}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        if guarding == "path-directory-ahead-of-the-program":
            (front / "created-ahead-of-the-program").mkdir()
            changed = os.lstat(front).st_ctime_ns
        else:
            (root / "created-at-the-top-level").mkdir()
            (ctx["parent"] / "created-beside-the-checkout").mkdir()
            changed = min(
                os.lstat(root).st_ctime_ns,
                os.lstat(ctx["parent"]).st_ctime_ns,
            )
        clock = _Clock(monkeypatch)
        clock.now_ns = changed + settle - 1
        _expect_no_git(git_calls, lambda: _bracket(root))
        assert _observe(git_calls, lambda: _default(root)) == _Outcome(None, 2)
        assert not _remembered(authority)
        clock.now_ns = changed + settle
        assert _observe(git_calls, lambda: _default(root)) == _Outcome(None, 2)
        from_memory(_remembered(authority))
        _expect_no_git(git_calls, lambda: _bracket(root))


@steady_wall_clock
def test_changed_environment_costs_one_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """The environment a Git child is given is part of the witness."""

    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    with journals.workspace_mutation_lock(root):
        _expect_answer_from_memory(root, git_calls)
        monkeypatch.setenv("CRUCIBLE_TEST_WITNESS_CHANGE", "set under the lock")
        assert _observe(git_calls, lambda: _bracket(root)) == _Outcome(None, 2)
        _expect_no_git(git_calls, lambda: _bracket(root))
        monkeypatch.delenv("CRUCIBLE_TEST_WITNESS_CHANGE")
        assert _observe(git_calls, lambda: _bracket(root)) == _Outcome(None, 2)


@steady_wall_clock
def test_answer_is_not_kept_when_git_names_another_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """An answer is kept only when Git named the held common directory."""

    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    elsewhere = tmp_path / "elsewhere" / ".git"
    with journals.workspace_mutation_lock(root) as authority:
        _settle(root)
        _require_witness(authority, root)
        original = journals._workspace_authority_common_dir
        pending = [elsewhere]

        def naming(cwd: Path, *, root_fd: int | None = None) -> Path:
            found = original(cwd, root_fd=root_fd)
            return pending.pop() if pending else found

        monkeypatch.setattr(journals, "_workspace_authority_common_dir", naming)
        refused = _observe(git_calls, lambda: _default(root))
        assert refused.refusal is not None and _CHANGED in refused.refusal
        assert not _remembered(authority)
        assert _observe(git_calls, lambda: _bracket(root)) == _Outcome(None, 2)


def test_proof_asks_git_for_every_held_checkout_of_the_common_directory(
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    """A journal binds through every held checkout of its Git common directory.

    The proof before a record asks Git for each of them, so it refuses for
    a held checkout Git cannot read although the journal's own is readable.
    """

    ctx = _layout_worktree(tmp_path)
    main, lane = ctx["main"], ctx["root"]
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=lane,
        journal_root=tmp_path / "corpus" / "runs" / "witness" / "mutations",
        slice_id="witness-proof",
        scope=("a.py",),
        sequence=1,
    )
    with journals.workspace_mutation_lock(main):
        with journals.workspace_mutation_lock(lane):
            assert len(journal._git_admin_workspace_authorities()) == 2
            start = len(git_calls.args)
            journal._assert_git_discovered_authority()
            assert git_calls.discoveries(start) == 2
            undo = _rename_away(main / ".git" / "HEAD")
            try:
                own = _observe(git_calls, lambda: _default(lane))
                refused = _observe(git_calls, journal._assert_git_discovered_authority)
            finally:
                undo()
    assert own.refusal is None
    assert refused.refusal is not None and _UNREADABLE in refused.refusal
    assert journal.restore()


@steady_wall_clock
def test_new_lock_on_the_same_checkout_starts_without_a_remembered_answer(
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    _settle(root)
    start = len(git_calls.args)
    with journals.workspace_mutation_lock(root) as first:
        acquisition = git_calls.discoveries(start)
        _expect_answer_from_memory(root, git_calls)
    with pytest.raises(journals.MutationJournalError, match="not held"):
        _bracket(root)
    start = len(git_calls.args)
    with journals.workspace_mutation_lock(root) as second:
        # Taking the lock again asks Git exactly as the first time: nothing
        # of the released authority answers for the new one.
        assert git_calls.discoveries(start) == acquisition
        assert second is not first


@steady_wall_clock
def test_two_held_roots_keep_separate_memories(
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    ctx = _layout_worktree(tmp_path)
    main, lane = ctx["main"], ctx["root"]
    lane_head = ctx["git_dir"] / "HEAD"
    with journals.workspace_mutation_lock(main) as first:
        _expect_answer_from_memory(main, git_calls)
        start = len(git_calls.args)
        with journals.workspace_mutation_lock(lane) as second:
            # A second root of the held common directory is a new authority:
            # its own acquisition asked Git.
            assert second is not first
            assert git_calls.discoveries(start) >= 1
            _expect_answer_from_memory(lane, git_calls)
            _expect_no_git(git_calls, lambda: _bracket(main))

            undo = _rename_away(lane_head)
            try:
                refused = _observe(git_calls, lambda: _bracket(lane))
                assert refused.refusal is not None
                assert _UNREADABLE in refused.refusal
                assert not _remembered(second)
                from_memory(_remembered(first))
                _expect_no_git(git_calls, lambda: _bracket(main))
            finally:
                undo()
            assert _observe(git_calls, lambda: _bracket(lane)).refusal is None


# ------------------------------------------------- the calls that always ask Git

@steady_wall_clock
def test_default_call_asks_git_whatever_is_remembered(
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    with journals.workspace_mutation_lock(root):
        _expect_answer_from_memory(root, git_calls)
        start = len(git_calls.args)
        assert _observe(git_calls, lambda: _default(root)) == _Outcome(None, 2)
        assert _observe(git_calls, lambda: _default(root)) == _Outcome(None, 2)
        assert git_calls.discoveries(start) == 2


def _workflow_sources() -> Iterator[tuple[Path, ast.Module]]:
    directory = Path(journals.__file__).resolve().parent
    for path in sorted(directory.glob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def _enclosing_functions(tree: ast.Module) -> dict[ast.AST, str]:
    owner: dict[ast.AST, str] = {}

    def visit(node: ast.AST, name: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = name
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = child.name
            owner[child] = inner
            visit(child, inner)

    visit(tree, "<module>")
    return owner


def test_exactly_the_descriptor_handing_helpers_reuse_the_discovery() -> None:
    reusing: list[str] = []
    asking: list[str] = []
    for path, tree in _workflow_sources():
        owner = _enclosing_functions(tree)
        called: set[ast.AST] = set()
        for node in ast.walk(tree):
            # A call under either spelling, the bare name or an attribute.
            if not (isinstance(node, ast.Call) and _called_name(node) == _ASSERTION):
                continue
            called.add(node.func)
            # Nothing may pass the keyword where this test cannot read it.
            assert not any(isinstance(item, ast.Starred) for item in node.args)
            assert all(keyword.arg is not None for keyword in node.keywords)
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            if "reuse_discovery" not in keywords:
                asking.append(f"{path.name}:{owner[node]}")
                continue
            value = keywords["reuse_discovery"]
            assert isinstance(value, ast.Constant) and value.value is True
            reusing.append(owner[node])
        # The assertion is called where it is named, never handed on.
        for node in ast.walk(tree):
            named = (
                node.id if isinstance(node, ast.Name)
                else node.attr if isinstance(node, ast.Attribute)
                else None
            )
            assert named != _ASSERTION or node in called, path.name

    assert sorted(reusing) == [
        "_bound_git_admin_path",
        "_bound_workspace_restore_root",
        "_bound_workspace_restore_root",
        "_git_admin_authority_batch",
        "_git_admin_authority_batch",
    ]
    # Every other call site asks Git at every call: the 41 places where a
    # journal operation begins, binds the registry, settles, recovers or
    # reaps, the proof that precedes each settlement record, and the proof
    # that precedes a write through a held descriptor.
    assert len(asking) == 43
    assert asking.count(f"_mutation_journal.py:{_RECORD_PROOF}") == 1
    assert asking.count(f"_mutation_journal.py:{_WRITE_PROOF}") == 1
    assert sum(1 for site in asking if site.startswith("_reaper.py:")) == 11
    assert sum(
        1 for site in asking if site.startswith("_journal_recovery.py:")
    ) == 4


# ------------------------------------------------------------------ the slot

@steady_wall_clock
def test_remembered_answer_is_not_compared_printed_or_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    marker = f"witness-marker-{os.getpid()}"
    monkeypatch.setenv("CRUCIBLE_TEST_WITNESS_MARKER", marker)
    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    slot = {
        field.name: field for field in fields(journals._WorkspaceMutationAuthority)
    }["_discovery_witness"]
    assert slot.compare is False and slot.repr is False and slot.init is False
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        from_memory(_remembered(authority))
        assert marker not in repr(authority)
        undo = _rename_away(ctx["git_dir"] / "HEAD")
        try:
            refused = _observe(git_calls, lambda: _bracket(root))
        finally:
            undo()
        assert refused.refusal is not None and marker not in refused.refusal


# ------------------------- the witness never answers where Git would not

_NAMES_GIT_LOOKS_FOR = (
    ".git", "HEAD", "objects", "refs", "config", "commondir", "gitdir",
    ".gitmodules", ".gitconfig", "packed-refs", "index", "modules", "worktrees",
)
State = tuple[str, Callable[[], Undo]]


def _git_directory_entries(ctx: dict[str, Path]) -> list[Path]:
    """Every entry of every Git directory the held checkout uses."""

    found: list[Path] = []
    for current, directories, files in os.walk(ctx.get("common", ctx["git_dir"])):
        for name in sorted((*directories, *files)):
            found.append(Path(current) / name)
    dot_git = ctx["root"] / ".git"
    if dot_git.is_file():
        found.append(dot_git)
    return found


def _first_byte_changed(path: Path) -> Undo:
    mode = stat.S_IMODE(path.lstat().st_mode)
    original = path.read_bytes()[:1]
    path.chmod(mode | 0o200)
    _pwrite(path, bytes([original[0] ^ 0x01]))

    def undo() -> None:
        _pwrite(path, original)
        path.chmod(mode)

    return undo


def _created(path: Path, kind: str) -> Undo:
    if kind == "file":
        path.write_text("ref: refs/heads/main\n", encoding="utf-8")
        return path.unlink
    path.mkdir()
    return path.rmdir


def _sweep_renamed_away(ctx: dict[str, Path]) -> Iterator[State]:
    for entry in _git_directory_entries(ctx):
        yield str(entry), lambda entry=entry: _rename_away(entry)


def _sweep_mode_000(ctx: dict[str, Path]) -> Iterator[State]:
    for entry in _git_directory_entries(ctx):
        yield str(entry), lambda entry=entry: _unsearchable(entry)


def _sweep_first_byte_changed(ctx: dict[str, Path]) -> Iterator[State]:
    for entry in _git_directory_entries(ctx):
        metadata = entry.lstat()
        if stat.S_ISREG(metadata.st_mode) and metadata.st_size:
            yield str(entry), lambda entry=entry: _first_byte_changed(entry)


def _sweep_names_created(ctx: dict[str, Path]) -> Iterator[State]:
    for directory in (ctx["root"], ctx["root"].parent):
        for name in _NAMES_GIT_LOOKS_FOR:
            path = directory / name
            if os.path.lexists(path):
                continue
            for kind in ("file", "directory"):
                yield (
                    f"{kind} {path}",
                    lambda path=path, kind=kind: _created(path, kind),
                )


_SWEEPS: dict[str, Callable[[dict[str, Path]], Iterator[State]]] = {
    "renamed-away": _sweep_renamed_away,
    "mode-000": _sweep_mode_000,
    "first-byte-changed": _sweep_first_byte_changed,
    "names-created": _sweep_names_created,
}


@pytest.mark.timeout(600)
@pytest.mark.parametrize("layout", ["plain", "worktree"])
@pytest.mark.parametrize("sweep", sorted(_SWEEPS))
@steady_wall_clock
def test_bracket_call_answers_as_the_default_call_for_every_changed_entry(
    sweep: str,
    layout: str,
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    """Each state starts from a remembered answer and is undone afterwards.

    The sweep covers entries of the checkout's Git directories renamed away,
    made inaccessible or changed in place, and names Git looks for created in
    and above the checkout.  It does not show what Git reads elsewhere.
    """

    ctx = _LAYOUTS[layout](tmp_path)
    root = ctx["root"]
    differing: list[tuple[str, _Outcome, _Outcome]] = []
    with journals.workspace_mutation_lock(root):
        states = list(_SWEEPS[sweep](ctx))
        assert states
        for label, make in states:
            _expect_answer_from_memory(root, git_calls)
            undo = make()
            try:
                bracket = _observe(git_calls, lambda: _bracket(root))
                default = _observe(git_calls, lambda: _default(root))
            finally:
                undo()
            if bracket.refusal != default.refusal:
                differing.append((label, bracket, default))
    assert not differing


# --------------------------------------------- one transaction, step by step

class _Transaction:
    """A one-file transaction on one checkout; each journal step alone."""

    def __init__(
        self,
        tmp: Path,
        prepare: Callable[[Path], None] | None = None,
    ) -> None:
        self.ctx = _layout_nested(tmp)
        self.root = self.ctx["root"]
        if prepare is not None:
            # What the checkout holds besides its two files, at the baseline.
            prepare(self.root)
        self.target = self.root / "a.py"
        self.baseline = self.target.read_bytes()
        self.written = b"# written inside the transaction\n"
        self.base = _git(self.root, "rev-parse", "HEAD")
        corpus_dir = tmp / "corpus"
        corpus_dir.mkdir()
        self.corpus = Corpus(corpus_dir)
        self.corpus.put_spec(_spec(self.base))
        self.journal: journals.WorkspaceMutationJournal | None = None
        self.implementation: Implementation | None = None
        self.attached = False
        self.dead = False

    def begin(self) -> None:
        self.journal = journals.WorkspaceMutationJournal.begin(
            cwd=self.root,
            journal_root=self.corpus.spec_dir / "runs" / "witness" / "mutations",
            slice_id="s0",
            scope=("a.py",),
            sequence=1,
            spec=self.corpus.latest_spec(),
        )

    def write(self) -> None:
        self.target.write_bytes(self.written)

    def write_escape(self) -> None:
        self.write()
        (self.root / "unscoped.py").write_text("# an escape\n", encoding="utf-8")

    def capture(self) -> None:
        assert self.journal is not None
        self.journal.prepare_git_capture()

    def validate(self) -> None:
        assert self.journal is not None
        candidate = Implementation(
            slice_id="s0",
            base_commit=self.base,
            result_commit=None,
            diff_ref=self.corpus.put_artifact(
                subprocess.run(
                    [_REAL_GIT, "diff", "--binary", "--", "a.py"],
                    cwd=self.root,
                    check=True,
                    capture_output=True,
                ).stdout,
                kind="diffs",
                name_hint="witness.diff",
            ),
            transcript_ref=self.corpus.put_artifact(
                b"[]",
                kind="transcripts",
                name_hint="witness.json",
            ),
            files_modified=("a.py",),
            scope_escapes=(),
            classification="success",
            execution=exec_ctx_now("test"),
        )
        self.implementation = self.journal.validate_implementation(candidate)

    def prepare(self) -> None:
        assert self.journal is not None and self.implementation is not None
        self.journal.prepare(self.implementation, keep=True)

    def attach(self) -> None:
        assert self.implementation is not None
        self.corpus.commit(implementations=(self.implementation,))
        self.attached = True

    def commit(self) -> None:
        assert self.journal is not None
        self.journal.commit(attached=True)

    def restore(self) -> bool:
        assert self.journal is not None
        return self.journal.restore()

    def metadata_guard(self) -> None:
        assert self.journal is not None
        self.journal.assert_repository_metadata_unchanged()

    def restore_paths(self) -> None:
        assert self.journal is not None
        self.journal.restore_paths_from_snapshot(["unscoped.py"])

    def record_escapes(self) -> None:
        assert self.journal is not None
        self.journal.record_scope_escapes(["unscoped.py"])

    def die(self) -> None:
        """Drop the process-local owners; the durable journal stays."""

        assert self.journal is not None
        for transaction_id in sorted(
            {self.journal.transaction_id} | self.journal._nested_transaction_ids()
        ):
            journals._unregister_live(transaction_id)
        self.dead = True

    def reap(self) -> object:
        return _reaper.reap_orphaned_dirt(self.corpus, self.root)

    def durable_state(self) -> str | None:
        assert self.journal is not None
        return _durable_state(self.journal)

    def settle(self) -> None:
        """Finish whatever a test left open, with the journal's own code."""

        journal = self.journal
        if journal is None:
            return
        if self.dead:
            if journal.journal_dir.exists():
                self.reap()
            return
        if journal.active:
            if journal.state == "prepared_keep":
                journal.settle_or_restore(self.corpus)
            else:
                journal.restore()
        elif journal.cleanup_pending:
            journal.retry_cleanup()
        self.die()


def _durable_state(journal: journals.WorkspaceMutationJournal) -> str | None:
    try:
        manifest = json.loads(
            (journal.journal_dir / journals._MANIFEST).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    state = manifest.get("state")
    return state if isinstance(state, str) else None


#: operation -> (steps before it, the step that is the operation, lock held)
_OPERATIONS: dict[str, tuple[tuple[str, ...], str, bool]] = {
    "metadata_guard": (("begin", "write"), "metadata_guard", True),
    "validate": (("begin", "write", "capture"), "validate", True),
    "prepare": (("begin", "write", "capture", "validate"), "prepare", True),
    # The caller holds no lock: the settlement capture takes one of its own.
    "prepare_unlocked": (
        ("begin", "write", "capture", "validate"),
        "prepare",
        False,
    ),
    "commit": (
        ("begin", "write", "capture", "validate", "prepare", "attach"),
        "commit",
        True,
    ),
    "restore": (("begin", "write"), "restore", True),
    "restore_paths": (("begin", "write_escape"), "restore_paths", True),
    "scope_escapes": (("begin", "write_escape"), "record_escapes", True),
    # A dead journal: the reaper takes the lock itself.
    "reap": (("begin", "write", "die"), "reap", False),
}


@contextmanager
def _operation(
    tmp: Path,
    name: str,
    prepare: Callable[[Path], None] | None = None,
) -> Iterator[tuple[_Transaction, Callable[[], object]]]:
    """Run the steps before one journal operation; yield the operation."""

    transaction = _Transaction(tmp, prepare)
    steps, final, locked = _OPERATIONS[name]
    with ExitStack() as stack:
        if locked:
            stack.enter_context(journals.recovery_workspace_mutation_lock(
                transaction.corpus,
                transaction.root,
            ))
            journals.recover_mutation_journals(transaction.corpus, transaction.root)
        for step in steps:
            getattr(transaction, step)()
        finished = False
        try:
            yield transaction, getattr(transaction, final)
            finished = True
        finally:
            try:
                transaction.settle()
            except Exception:
                if finished:
                    raise


def _run(operation: Callable[[], object]) -> tuple[object, str | None]:
    try:
        return operation(), None
    except journals.MutationJournalError as exc:
        return None, str(exc)


_CALL_KINDS = frozenset(("bracket", "default", "proof", "write"))


class _AssertionTrace:
    """What happens around the authority assertion while one operation runs.

    Events, in order: a call of the assertion by its kind (``bracket`` from
    a descriptor-handing helper, ``proof`` from the proof before a settlement
    record, ``write`` from the proof before a write through a held
    descriptor, ``default`` otherwise), ``asked`` when Git's discovery is
    started, ``hook`` when a write through a held descriptor is about to be
    made, ``record`` when a manifest write changes the durable state,
    ``stage`` when exact journal recovery writes its event, and ``guard``
    when the first metadata guard of a rollback has returned.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from crucible.workflows import _journal_recovery as recovery

        self.armed = False
        self.events: list[tuple[str, str]] = []
        #: Positions in ``events`` of the assertion calls that refused.
        self.refused: set[int] = set()
        #: Positions of the assertion calls during which Git was asked.
        self.asked: set[int] = set()
        self._running: list[int] = []
        self.on_call: Callable[[str, FrameType], None] | None = None
        #: The authority the thread holds at the call being observed.
        self.authority: object | None = None
        self._busy = False
        trace = self
        active = journals._active_workspace_mutation_authority
        discovery = journals._workspace_authority_common_dir
        write = journals.WorkspaceMutationJournal._write
        guard = journals.WorkspaceMutationJournal.assert_repository_metadata_unchanged
        proved = journals._assert_workspace_mutation_authority
        write_proof = getattr(journals, _WRITE_PROOF, None)
        write_event = recovery._write_event
        assertion = _ASSERTION

        def hooking(*through: object) -> None:
            handed_out = getattr(
                journals._LOCK_STATE, "handed_out_authorities", None,
            )
            if trace.armed and (
                handed_out or any(item is not None for item in through)
            ):
                trace.events.append(("hook", ""))
            assert write_proof is not None
            write_proof(*through)

        # The stand-in carries the assertion's own name, so that the caller
        # of the assertion is still found behind it.
        def _assert_workspace_mutation_authority(
            cwd: str | Path,
            *,
            reuse_discovery: bool = False,
        ) -> object:
            position = len(trace.events)
            trace._running.append(position)
            try:
                return proved(cwd, reuse_discovery=reuse_discovery)
            except journals.MutationJournalError:
                if trace.armed:
                    trace.refused.add(position)
                raise
            finally:
                trace._running.pop()

        def staging(
            journal: journals.WorkspaceMutationJournal,
            event: dict[str, object],
        ) -> None:
            if trace.armed:
                trace.events.append(("stage", str(event.get("stage"))))
            write_event(journal, event)

        def entered(*args: object, **kwargs: object) -> object:
            frame = sys._getframe(1)
            if trace.armed and not trace._busy and frame.f_code.co_name == assertion:
                caller = frame.f_back
                while caller is not None and caller.f_code.co_name == assertion:
                    caller = caller.f_back
                name = caller.f_code.co_name if caller is not None else ""
                if name in _BRACKET_HELPERS:
                    kind = "bracket"
                elif name == _RECORD_PROOF:
                    kind = "proof"
                elif name == _WRITE_PROOF:
                    kind = "write"
                else:
                    kind = "default"
                trace.events.append((kind, name))
                if trace.on_call is not None:
                    trace._busy = True
                    try:
                        trace.authority = active(*args, **kwargs)
                        trace.on_call(kind, frame)
                    finally:
                        trace._busy = False
            return active(*args, **kwargs)

        def asking(cwd: Path, *, root_fd: int | None = None) -> Path:
            if trace.armed:
                trace.events.append(("asked", ""))
                # A lock acquisition asks Git outside every call.
                trace.asked.update(trace._running[-1:])
            return discovery(cwd, root_fd=root_fd)

        def recording(journal: journals.WorkspaceMutationJournal) -> None:
            if trace.armed:
                state = str(journal._manifest.get("state"))
                if state != _durable_state(journal):
                    trace.events.append(("record", state))
            write(journal)

        def guarding(journal: journals.WorkspaceMutationJournal) -> None:
            in_rollback = sys._getframe(1).f_code.co_name == "_restore_bound"
            try:
                guard(journal)
            finally:
                if (
                    trace.armed
                    and in_rollback
                    and ("guard", "") not in trace.events
                ):
                    trace.events.append(("guard", ""))

        monkeypatch.setattr(
            journals,
            "_assert_workspace_mutation_authority",
            _assert_workspace_mutation_authority,
        )
        monkeypatch.setattr(recovery, "_write_event", staging)
        if write_proof is not None:
            monkeypatch.setattr(journals, _WRITE_PROOF, hooking)
        monkeypatch.setattr(journals, "_active_workspace_mutation_authority", entered)
        monkeypatch.setattr(journals, "_workspace_authority_common_dir", asking)
        monkeypatch.setattr(journals.WorkspaceMutationJournal, "_write", recording)
        monkeypatch.setattr(
            journals.WorkspaceMutationJournal,
            "assert_repository_metadata_unchanged",
            guarding,
        )

    def count(self, kind: str) -> int:
        return sum(1 for event in self.events if event[0] == kind)

    def records(self, since: int = 0) -> list[str]:
        return [
            detail for kind, detail in self.events[since:] if kind == "record"
        ]

    def records_on_remembered_answers(self) -> list[str]:
        """Settlement states and recovery stages recorded although the last
        call of the assertion did not ask Git."""

        stale: list[str] = []
        asked = True
        for kind, detail in self.events:
            if kind in _CALL_KINDS:
                asked = False
            elif kind == "asked":
                asked = True
            elif asked:
                continue
            elif kind == "record" and detail in _SETTLEMENT_STATES | _RECOVERY_STATES:
                stale.append(detail)
            elif kind == "stage" and detail in _RECOVERY_STAGES:
                stale.append(f"stage {detail}")
        return stale

    def first_call_after_the_guard_asked_git(self) -> bool:
        """Whether Git was asked before the write phase of a rollback."""

        position = self.events.index(("guard", ""))
        after = [
            kind for kind, _detail in self.events[position + 1:]
            if kind != "record"
        ]
        return len(after) >= 2 and after[0] != "asked" and after[1] == "asked"

    def writes_on_remembered_answers(self) -> list[int]:
        """Positions at which a write through a held descriptor was about to
        be made although the last call had not asked Git, and Git was not
        asked there either."""

        missed: list[int] = []
        asked = True
        for position, (kind, _detail) in enumerate(self.events):
            if kind in _CALL_KINDS:
                asked = False
            elif kind == "asked":
                asked = True
            elif kind == "hook" and not asked:
                proof = self.events[position + 1:position + 3]
                if [item[0] for item in proof] != ["write", "asked"]:
                    missed.append(position)
        return missed

    def watch_modes(self, monkeypatch: pytest.MonkeyPatch, below: Path) -> None:
        """Add a ``mode`` event for every mode call on an entry at or below
        ``below``, by descriptor or by path."""

        trace = self
        top = os.path.realpath(below)

        def watched(original: Callable[..., None]) -> Callable[..., None]:
            def setting(target: object, *args: object, **kwargs: object) -> None:
                if trace.armed:
                    if isinstance(target, int):
                        named = f"/proc/self/fd/{target}"
                    else:
                        named = os.fsdecode(target)  # type: ignore[arg-type]
                        if kwargs.get("dir_fd") is not None:
                            named = f"/proc/self/fd/{kwargs['dir_fd']}/{named}"
                    path = os.path.realpath(named)
                    if path == top or path.startswith(top + os.sep):
                        trace.events.append(("mode", path[len(top):] or "/"))
                original(target, *args, **kwargs)

            return setting

        monkeypatch.setattr(os, "fchmod", watched(os.fchmod))
        monkeypatch.setattr(os, "chmod", watched(os.chmod))

    def modes_on_remembered_answers(self) -> list[str]:
        """The entries whose mode was set although the last call of the
        assertion had not asked Git."""

        stale: list[str] = []
        asked = True
        for kind, detail in self.events:
            if kind in _CALL_KINDS:
                asked = False
            elif kind == "asked":
                asked = True
            elif kind == "mode" and not asked:
                stale.append(detail)
        return stale


class _StateInside:
    """Make a state at one bracket call inside a running operation."""

    def __init__(
        self,
        trace: _AssertionTrace,
        make: Callable[[], Undo],
        *,
        at: Callable[[FrameType], bool],
        undo_after: int | None = None,
    ) -> None:
        self.trace = trace
        self.make = make
        self.at = at
        self.undo_after = undo_after
        self.event: int | None = None
        self.remembered = False
        #: Position in the trace of the bracket call the state is made at.
        self.call: int | None = None
        self._calls = 0
        self._made_at = 0
        self._undo: Undo | None = None
        trace.on_call = self

    def __call__(self, kind: str, frame: FrameType) -> None:
        self._calls += 1
        if self.event is None:
            if kind == "bracket" and self.at(frame):
                held = self.trace.authority
                self.remembered = held is not None and _remembered(held)
                self.call = len(self.trace.events) - 1
                self._undo = self.make()
                self._made_at = self._calls
                self.event = len(self.trace.events)
        elif (
            self.undo_after is not None
            and self._calls >= self._made_at + self.undo_after
        ):
            self.finish()

    def finish(self) -> None:
        undo, self._undo = self._undo, None
        if undo is not None:
            undo()

    def refused_at_its_call(self) -> bool:
        """Whether the bracket call the state was made at refused itself."""

        return self.call is not None and self.call in self.trace.refused


def _expect_next_bracket_call_from_memory(inside: _StateInside) -> None:
    """A store Git saw only once was remembered and then reused.

    The call the store is made at asks Git, which sees the readable entry.
    If the next call of the assertion is a bracket call, it must have been
    answered from memory: otherwise the test did not meet the state it is
    about, and what refused later proves nothing about it.
    """

    assert inside.event is not None and inside.call is not None
    trace = inside.trace
    # Whatever the clock does, the stored byte differs from the witnessed one.
    assert inside.call in trace.asked, (
        "the bracket call the store was made at did not ask Git"
    )
    calls = [
        position for position, (kind, _detail) in enumerate(trace.events)
        if position >= inside.event and kind in _CALL_KINDS
    ]
    if not calls or trace.events[calls[0]][0] != "bracket":
        return
    from_memory(
        calls[0] not in trace.asked,
        "the bracket call behind the question that saw the store asked Git",
    )


def _expect_remembered_inside(inside: _StateInside, root: Path) -> None:
    """The state was made at a bracket call that had an answer to reuse."""

    if inside.remembered:
        return
    reason = witness_unavailable_reason(root)
    if reason is not None:
        pytest.skip(f"no answer can be remembered here: {reason}")
    from_memory(False, "the state was made at a bracket call with nothing remembered")


def _after_default_calls(
    trace: _AssertionTrace,
    count: int,
) -> Callable[[FrameType], bool]:
    """The first bracket call after ``count`` default calls of the operation."""

    return lambda _frame: trace.count("default") >= count


_TERMINAL_REPROOFS = frozenset((
    "_reprove_restored_workspace_directory_authority",
    "_reprove_restored_workspace_entry_authority",
    "_reprove_restored_tracked_path_authority",
))


_ROLLBACKS = frozenset(("_restore_bound", "restore_paths_from_snapshot"))


def _in_terminal_reproof(frame: FrameType | None) -> bool:
    """Whether a call belongs to the reproofs a rollback makes of what it
    restored: behind its last Git command, they start no process."""

    while frame is not None:
        outer = frame.f_back
        if (
            frame.f_code.co_name in _TERMINAL_REPROOFS
            and outer is not None
            and outer.f_code.co_name in _ROLLBACKS
        ):
            return True
        frame = outer
    return False


def _first_bracket_call_inside(*functions: str) -> Callable[[FrameType], bool]:
    """A bracket call made while every one of ``functions`` is running."""

    def at(frame: FrameType | None) -> bool:
        running = set()
        while frame is not None:
            running.add(frame.f_code.co_name)
            frame = frame.f_back
        return running.issuperset(functions)

    return at


def _nth_terminal_reproof(position: int) -> Callable[[FrameType], bool]:
    seen = [0]

    def at(frame: FrameType) -> bool:
        if not _in_terminal_reproof(frame):
            return False
        seen[0] += 1
        return seen[0] == position

    return at


def _head_moved_out(ctx: dict[str, Path]) -> Undo:
    head = ctx["git_dir"] / "HEAD"
    kept = ctx["tmp"] / "saved-HEAD"
    head.rename(kept)

    def undo() -> None:
        os.replace(kept, head)

    return undo


class _MappedHeadFlicker:
    """``HEAD`` stored through a shared mapping, in step with one Git question.

    The entry is bad, good for the length of the next Git question of the
    assertion, and bad again; after the first store no stamp moves.  It
    stands for another process of the same user that has the file mapped.
    """

    def __init__(self, head: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.head = head
        self.monkeypatch = monkeypatch
        self.moved = False

    def start(self) -> Undo:
        view, good, stamp = _mapped_first_byte(self.head)

        def store(value: bytes) -> None:
            view[0:1] = value

        store(b"x")
        _good_only_while_git_is_asked_once(
            self.monkeypatch, lambda: store(good), lambda: store(b"x"),
        )

        def stop() -> None:
            self.moved = _stamp(self.head) != stamp
            store(good)
            view.close()

        return stop


# ------------------------------------- a state made inside a journal operation

#: place -> (operation, default calls of the operation before the bracket call)
_INSIDE: dict[str, tuple[str, int]] = {
    "validate": ("validate", 2),
    "prepare": ("prepare", 5),
    "commit": ("commit", 9),
    "restore-before-its-writes": ("restore", 3),
    "restore-at-its-end": ("restore", 8),
    "restore-paths": ("restore_paths", 2),
    "reap": ("reap", 16),
}
_INSIDE_STATES: dict[str, Callable[[dict[str, Path]], Undo]] = {
    "gitlink-committed-in-parent-repository": _gitlink_committed_in_parent,
    "head-removed": _head_moved_out,
}


def _assert_operation_refused(
    place: str,
    transaction: _Transaction,
    state_before: str,
    result: object,
    refusal: str | None,
    recorded: list[str],
) -> None:
    """The operation ended as it does when Git itself refuses at that call."""

    operation = _INSIDE[place][0]
    assert transaction.journal is not None
    assert not _SETTLEMENT_STATES.intersection(recorded), recorded
    if operation == "reap":
        failures = [str(failure) for failure in getattr(result, "failures")]
        assert any("rollback failed; journal retained" in text for text in failures)
        assert transaction.durable_state() == "invoking"
        return
    assert transaction.journal.state == state_before
    if operation == "restore":
        assert result is False and refusal is None
        expected = (
            transaction.baseline if place == "restore-at-its-end"
            else transaction.written
        )
        assert transaction.target.read_bytes() == expected
        return
    assert refusal is not None
    assert transaction.target.read_bytes() == transaction.written


@pytest.mark.timeout(600)
@pytest.mark.parametrize("lasting", ["stays", "undone-three-calls-later"])
@pytest.mark.parametrize("state", sorted(_INSIDE_STATES))
@pytest.mark.parametrize("place", sorted(_INSIDE))
@steady_wall_clock
def test_state_made_inside_an_operation_is_refused_at_its_bracket_call(
    place: str,
    state: str,
    lasting: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation, defaults = _INSIDE[place]
    trace = _AssertionTrace(monkeypatch)
    with _operation(tmp_path, operation) as (transaction, run):
        assert transaction.journal is not None
        inside = _StateInside(
            trace,
            lambda: _INSIDE_STATES[state](transaction.ctx),
            at=_after_default_calls(trace, defaults),
            undo_after=None if lasting == "stays" else 3,
        )
        state_before = transaction.journal.state
        _settle(transaction.root)
        trace.armed = True
        try:
            result, refusal = _run(run)
        finally:
            trace.armed = False
            inside.finish()
        assert inside.event is not None
        # The bracket call that met the state refused itself, as the default
        # call does; a state that only a later Git command stops would not
        # show that.
        assert inside.refused_at_its_call()
        _assert_operation_refused(
            place, transaction, state_before, result, refusal,
            trace.records(inside.event),
        )
        _expect_remembered_inside(inside, transaction.root)
        # With the state undone the journal's own code settles what is left.
        if operation == "reap":
            report = transaction.reap()
            assert getattr(report, "failures") == ()
            assert not transaction.journal.journal_dir.exists()
        elif operation == "commit":
            assert transaction.journal.settle_or_restore(transaction.corpus)
        else:
            assert transaction.journal.restore()


@pytest.mark.timeout(300)
@pytest.mark.parametrize("hidden_by", ["renamed-away", "held-open-for-writing"])
@steady_wall_clock
def test_program_hidden_during_one_git_question_stops_a_kept_settlement(
    hidden_by: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A differing ``git`` that one Git question does not see is refused at
    the next bracket call: ``prepared_keep`` is not recorded."""

    ctx = {"tmp": tmp_path}
    _front_directory(ctx, monkeypatch)
    program = ctx["front"] / "git"
    hide, show = _program_hiding(ctx, hidden_by)

    def place() -> Undo:
        _write_program(program, differing=True)
        _good_only_while_git_is_asked_once(monkeypatch, hide, show)
        return program.unlink

    trace = _AssertionTrace(monkeypatch)
    with _operation(tmp_path, "prepare") as (transaction, run):
        assert transaction.journal is not None
        inside = _StateInside(
            trace, place, at=_after_default_calls(trace, _INSIDE["prepare"][1]),
        )
        _settle(transaction.root)
        trace.armed = True
        try:
            result, refusal = _run(run)
        finally:
            trace.armed = False
            inside.finish()
        assert inside.event is not None
        _assert_operation_refused(
            "prepare", transaction, "scoped", result, refusal,
            trace.records(inside.event),
        )
        _expect_remembered_inside(inside, transaction.root)
        assert transaction.journal.restore()


# --------------------------------------------------------- the bound on Git

class _ChildProcesses:
    """Every child process started while armed, and Git's discoveries."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.armed = False
        self.started = 0
        self.discoveries = 0
        counter = self
        original = subprocess.Popen.__init__

        def counting(
            popen: subprocess.Popen[bytes],
            args: object,
            *rest: object,
            **kwargs: object,
        ) -> None:
            if counter.armed:
                counter.started += 1
                if isinstance(args, (list, tuple)) and any(
                    str(part) == "--show-superproject-working-tree"
                    for part in args
                ):
                    counter.discoveries += 1
            original(popen, args, *rest, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(subprocess.Popen, "__init__", counting)


#: operation -> (most child processes, fewest discoveries, most discoveries)
_BOUNDS: dict[str, tuple[int, int, int | None]] = {
    "metadata_guard": (125, 2, 2),
    "validate": (165, 3, 3),
    "prepare": (310, 8, 8),
    "commit": (615, 11, 11),
    # The three below write to the checkout, and Git is asked before every
    # write behind an answer from memory.  A rollback sets the mode of each
    # directory it knows, each behind a descriptor handed out for it, so it
    # asks about once a directory; fewer discoveries than the floor would be
    # a mode set without a question.  It also rewrites HEAD: the bracket
    # calls inside the settle time behind that ask Git, so its count depends
    # on how many calls fit into that time.
    "restore_paths": (165, 5, 5),
    "restore": (570, 30, 55),
    "reap": (600, 40, None),
}


@pytest.mark.timeout(300)
@pytest.mark.parametrize("operation", sorted(_BOUNDS))
@steady_wall_clock
def test_journal_operation_starts_a_bounded_number_of_git_processes(
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    most, fewest, at_most = _BOUNDS[operation]
    children = _ChildProcesses(monkeypatch)
    with _operation(tmp_path, operation) as (transaction, run):
        reason = witness_unavailable_reason(transaction.root)
        if reason is not None:
            pytest.skip(f"no answer can be remembered here: {reason}")
        _settle(transaction.root)
        children.armed = True
        try:
            run()
        finally:
            children.armed = False
    measured = (
        f"{operation}: {children.started} child processes, "
        f"{children.discoveries} discoveries"
    )
    # Fewer would be a default call or a proof that did not ask Git.
    assert children.discoveries >= fewest, measured
    from_memory(children.started <= most, measured)
    if at_most is not None:
        from_memory(children.discoveries <= at_most, measured)


# ----------------------- no settlement state on a remembered answer

def _called_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    return function.id if isinstance(function, ast.Name) else ""


def _settlement_records() -> list[
    tuple[str, str, tuple[str, ...], tuple[str, ...]]
]:
    """(file, function, states, the last two calls before the record) of
    every assignment of a settlement or recovery state to a manifest."""

    found: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []
    for path, tree in _workflow_sources():
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = sorted(
                (
                    ((node.lineno, node.col_offset), node)
                    for node in ast.walk(function) if isinstance(node, ast.Call)
                ),
                key=lambda item: item[0],
            )
            for node in ast.walk(function):
                if not isinstance(node, ast.Assign):
                    continue
                target = node.targets[0]
                if not (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "state"
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "_manifest"
                ):
                    continue
                states = tuple(sorted(
                    constant.value for constant in ast.walk(node.value)
                    if isinstance(constant, ast.Constant)
                    and constant.value in _SETTLEMENT_STATES | _RECOVERY_STATES
                ))
                if not states:
                    continue
                before = [
                    _called_name(call) for position, call in calls
                    if position < (node.lineno, node.col_offset)
                ]
                found.append(
                    (path.name, function.name, states, tuple(before[-2:]))
                )
    return sorted(found)


def test_every_settlement_record_is_directly_preceded_by_the_git_proof() -> None:
    records = _settlement_records()
    journal, recovery = "_mutation_journal.py", "_journal_recovery.py"
    assert [record[:3] for record in records] == [
        (recovery, "_inspect_external_recovery", ("external_recovery_prepared",)),
        (recovery, "_inspect_external_recovery", ("external_recovery_verified",)),
        (journal, "_commit_prevalidated_lifecycle_family", ("committed",)),
        (journal, "_restore_bound", ("rolled_back",)),
        (journal, "_restore_nested_gitlink_guard", ("rolled_back",)),
        (journal, "commit", ("committed",)),
        (journal, "prepare", ("prepared_keep", "prepared_rollback")),
        (journal, "prepare_lifecycle_acceptances", ("prepared_keep",)),
        (journal, "record_implementation", ("scoped",)),
        (journal, "record_scope_escapes", ("scoped",)),
    ]
    for file_name, _function, _states, before in records:
        if file_name == journal:
            # In source order nothing is called between proof and record.
            assert before[-1] == _RECORD_PROOF
        else:
            # A recovery stage writes its event behind the proof, then the
            # state.
            assert before == (_RECORD_PROOF, "_write_event")


def _proof_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == _RECORD_PROOF
    )


def test_every_recovery_stage_is_directly_preceded_by_the_git_proof() -> None:
    """A resumed recovery acts on the stage of its event.

    The first stage is written with the state ``external_recovery_prepared``,
    which the test of the settlement records covers.  Each later stage is
    assigned directly behind the proof, and the cleanup begins with it.
    """

    from crucible.workflows import _journal_recovery as recovery

    tree = ast.parse(Path(recovery.__file__).read_text(encoding="utf-8"))
    staged: list[str] = []
    for block in ast.walk(tree):
        for body in (
            getattr(block, "body", None),
            getattr(block, "orelse", None),
            getattr(block, "finalbody", None),
        ):
            if not isinstance(body, list):
                continue
            for index, statement in enumerate(body):
                if not (
                    isinstance(statement, ast.Assign)
                    and isinstance(statement.targets[0], ast.Subscript)
                    and isinstance(statement.targets[0].value, ast.Name)
                    and statement.targets[0].value.id == "event"
                    and isinstance(statement.targets[0].slice, ast.Constant)
                    and statement.targets[0].slice.value == "stage"
                    and isinstance(statement.value, ast.Constant)
                    and statement.value.value in _RECOVERY_STAGES
                ):
                    continue
                staged.append(statement.value.value)
                assert index > 0 and _proof_call(body[index - 1]), statement.lineno
    assert sorted(staged) == [
        "source_restored", "tombstone_prepared", "tombstone_prepared", "verified",
    ]
    cleanup, = (
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_cleanup_verified"
    )
    assert _proof_call(cleanup.body[0])


def test_nested_journals_settle_only_behind_the_git_proof_of_their_parent() -> None:
    """A nested journal settles under the proof of its own checkout.

    The journal that drives it, a commit or a rollback of the checkout
    above, asks Git about its own checkout before the first of them does.
    """

    tree = ast.parse(Path(journals.__file__).read_text(encoding="utf-8"))
    proved: list[str] = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        settlements = [
            node.lineno for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("commit", "restore")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "nested"
        ]
        if not settlements:
            continue
        proofs = [
            node.lineno for node in ast.walk(function)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "nested_journals"
            and _proof_call(node.body[0])
        ]
        assert len(proofs) == 1 and proofs[0] < min(settlements), function.name
        proved.append(function.name)
    assert sorted(proved) == [
        "_restore_bound", "_restore_nested_gitlink_guard", "commit",
    ]


#: function -> calls of the proof before a write through a held descriptor
_WRITE_PROOFS: dict[str, int] = {
    "_remove_bound_regular_entry": 1,
    "_remove_bound_symlink_entry": 1,
    "_remove_bound_directory_tree": 2,
    "_make_real_directory_tree_owner_accessible": 5,
    "_restore_bound_descriptor_xattrs": 2,
    "_restore_bound_workspace_parent_metadata": 1,
    "_remove_bound_empty_workspace_directory": 1,
    "_restore_bound_workspace_directory_metadata": 2,
    "_remove_bound_workspace_entry": 2,
    "_publish_bound_regular_noreplace": 1,
    "_claim_bound_workspace_file": 1,
    "_create_private_bound_workspace_directory": 1,
    "_install_bound_workspace_entry": 1,
    "_restore_bound_git_admin_entry_metadata": 1,
    "_restore_bound_git_admin_tree": 2,
    "_restore_tracked_path_metadata": 2,
    "_restore_git_entry": 1,
}


def test_every_restoring_primitive_asks_git_before_it_writes() -> None:
    """The functions that change a checkout through a held descriptor.

    Each asks Git before its first change where the last discovery answer
    came from memory, a mode that is set whatever mode is in place
    included, and both helpers that hand out a held descriptor name the
    checkout it belongs to.
    """

    tree = ast.parse(Path(journals.__file__).read_text(encoding="utf-8"))
    found: dict[str, int] = {}
    naming: list[str] = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        calls = [
            _called_name(node) for node in ast.walk(function)
            if isinstance(node, ast.Call)
        ]
        asking = calls.count(_WRITE_PROOF)
        if asking and function.name in _WRITE_PROOFS:
            found[function.name] = asking
        naming.extend(
            function.name for name in calls if name == "_held_descriptor_handed_out"
        )
    assert found == _WRITE_PROOFS
    # ``_git_admin_authority_batch`` hands out no descriptor of its own: its
    # entries are reached through ``_bound_git_admin_path``.
    assert sorted(naming) == ["_bound_git_admin_path", "_bound_workspace_restore_root"]


def test_selective_restore_ends_its_writes_and_itself_with_the_git_proof() -> None:
    tree = ast.parse(Path(journals.__file__).read_text(encoding="utf-8"))
    restore, = (
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "restore_paths_from_snapshot"
    )
    # It returns only behind the proof ...
    assert _proof_call(restore.body[-1])
    # ... and consumes its durable latch, its last write, only behind it.
    latch_writes = [
        (block, index)
        for block in ast.walk(restore) if isinstance(block, ast.If)
        for index, statement in enumerate(block.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Subscript)
        and isinstance(statement.targets[0].slice, ast.Constant)
        and statement.targets[0].slice.value == "pre_capture_attribute_changes"
    ]
    assert len(latch_writes) == 1
    block, index = latch_writes[0]
    assert index > 0 and _proof_call(block.body[index - 1])


#: operation -> the settlement states it records
_RECORDS: dict[str, list[str]] = {
    "validate": ["scoped"],
    "prepare": ["prepared_keep"],
    "prepare_unlocked": ["prepared_keep"],
    "commit": ["committed"],
    "restore": ["rolled_back"],
    "reap": ["rolled_back"],
    "scope_escapes": ["scoped"],
}


@pytest.mark.timeout(300)
@pytest.mark.parametrize("operation", sorted(_RECORDS))
def test_settlement_state_is_recorded_only_after_git_was_asked(
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _AssertionTrace(monkeypatch)
    with _operation(tmp_path, operation) as (transaction, run):
        _settle(transaction.root)
        trace.armed = True
        try:
            if _OPERATIONS[operation][2]:
                # The operation begins behind a call that reused an answer
                # wherever one can be reused.
                _bracket(transaction.root)
            run()
        finally:
            trace.armed = False
    recorded = [
        state for state in trace.records() if state in _SETTLEMENT_STATES
    ]
    assert recorded == _RECORDS[operation]
    assert trace.records_on_remembered_answers() == []
    assert trace.writes_on_remembered_answers() == []
    if operation in ("restore", "reap"):
        # The first metadata guard of a rollback swallows a refusal; the
        # write phase behind it must not start on a remembered answer.
        assert trace.first_call_after_the_guard_asked_git()


def test_recovery_stage_is_recorded_only_after_git_was_asked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact journal recovery restores the source through held descriptors
    and records its stages in the manifest of the journal."""

    from .test_workflows_journal_recover import _apply, _setup

    recovery = _setup(tmp_path)
    trace = _AssertionTrace(monkeypatch)
    _settle()
    trace.armed = True
    try:
        _apply(recovery)
    finally:
        trace.armed = False
    recorded = [state for state in trace.records() if state in _RECOVERY_STATES]
    assert recorded == ["external_recovery_prepared", "external_recovery_verified"]
    stages = [detail for kind, detail in trace.events if kind == "stage"]
    assert _RECOVERY_STAGES.issubset(stages)
    assert trace.records_on_remembered_answers() == []
    assert trace.writes_on_remembered_answers() == []


@pytest.fixture(scope="module")
def terminal_reproof_calls(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, int]:
    """How many bracket calls a clean rollback makes in its reproofs."""

    counts: dict[str, int] = {}
    for operation in ("restore", "restore_paths"):
        counted = [0]

        def count(kind: str, frame: FrameType, counted: list[int] = counted) -> None:
            if kind == "bracket" and _in_terminal_reproof(frame):
                counted[0] += 1

        with pytest.MonkeyPatch.context() as monkeypatch:
            trace = _AssertionTrace(monkeypatch)
            trace.on_call = count
            with _operation(
                tmp_path_factory.mktemp(f"clean-{operation}"), operation,
            ) as (_transaction, run):
                trace.armed = True
                try:
                    run()
                finally:
                    trace.armed = False
        counts[operation] = counted[0]
    return counts


@pytest.mark.timeout(300)
@pytest.mark.parametrize("position", range(1, 8))
@steady_wall_clock
def test_rollback_is_not_recorded_on_a_store_git_saw_only_once(
    position: int,
    terminal_reproof_calls: dict[str, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reproofs behind a rollback's last Git command start no process.

    A store that moves no stamp, made there, is seen only by a call that
    asks Git: ``rolled_back`` must not be recorded in that state.  The last
    reproof is left out: a state that arises behind it arises behind every
    proof of the rollback.
    """

    if position >= terminal_reproof_calls["restore"]:
        pytest.skip(
            f"a clean rollback makes {terminal_reproof_calls['restore']} "
            "such reproofs"
        )
    trace = _AssertionTrace(monkeypatch)
    with _operation(tmp_path, "restore") as (transaction, run):
        assert transaction.journal is not None
        flicker = _MappedHeadFlicker(transaction.ctx["git_dir"] / "HEAD", monkeypatch)
        inside = _StateInside(
            trace, flicker.start, at=_nth_terminal_reproof(position),
        )
        _settle(transaction.root)
        trace.armed = True
        try:
            result, refusal = _run(run)
        finally:
            trace.armed = False
            inside.finish()
        assert inside.event is not None
        assert result is False and refusal is None
        assert transaction.journal.state == "invoking"
        assert "rolled_back" not in trace.records(inside.event)
        assert transaction.target.read_bytes() == transaction.baseline
        _expect_remembered_inside(inside, transaction.root)
        _expect_next_bracket_call_from_memory(inside)
        assert transaction.journal.restore()
    if flicker.moved:
        pytest.skip("the file system stamped a store through the mapping")


@pytest.mark.timeout(300)
@pytest.mark.parametrize("back", range(1, 6))
@steady_wall_clock
def test_selective_restore_does_not_return_on_a_store_git_saw_only_once(
    back: int,
    terminal_reproof_calls: dict[str, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selective restore ends with reproofs that start no process.

    The state is made ``back`` reproofs before the last one, which is left
    out as in the rollback above.
    """

    position = terminal_reproof_calls["restore_paths"] - back
    assert position >= 1
    trace = _AssertionTrace(monkeypatch)
    with _operation(tmp_path, "restore_paths") as (transaction, run):
        assert transaction.journal is not None
        flicker = _MappedHeadFlicker(transaction.ctx["git_dir"] / "HEAD", monkeypatch)
        inside = _StateInside(
            trace, flicker.start, at=_nth_terminal_reproof(position),
        )
        _settle(transaction.root)
        trace.armed = True
        try:
            result, refusal = _run(run)
        finally:
            trace.armed = False
            inside.finish()
        assert inside.event is not None
        _assert_operation_refused(
            "restore-paths", transaction, "invoking", result, refusal,
            trace.records(inside.event),
        )
        _expect_remembered_inside(inside, transaction.root)
        _expect_next_bracket_call_from_memory(inside)
        assert transaction.journal.restore()
    if flicker.moved:
        pytest.skip("the file system stamped a store through the mapping")


@pytest.mark.timeout(300)
def test_selective_restore_returns_only_after_git_was_asked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _AssertionTrace(monkeypatch)
    with _operation(tmp_path, "restore_paths") as (transaction, run):
        _settle(transaction.root)
        trace.armed = True
        try:
            run()
        finally:
            trace.armed = False
    calls = [
        index for index, (kind, _detail) in enumerate(trace.events)
        if kind in _CALL_KINDS
    ]
    assert calls
    assert ("asked", "") in trace.events[calls[-1]:]


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "place", ["validate", "prepare", "commit", "restore-before-its-writes"],
)
@steady_wall_clock
def test_no_settlement_state_is_recorded_on_a_store_git_saw_only_once(
    place: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation, defaults = _INSIDE[place]
    trace = _AssertionTrace(monkeypatch)
    with _operation(tmp_path, operation) as (transaction, run):
        assert transaction.journal is not None
        flicker = _MappedHeadFlicker(transaction.ctx["git_dir"] / "HEAD", monkeypatch)
        inside = _StateInside(
            trace, flicker.start, at=_after_default_calls(trace, defaults),
        )
        state_before = transaction.journal.state
        _settle(transaction.root)
        trace.armed = True
        try:
            result, refusal = _run(run)
        finally:
            trace.armed = False
            inside.finish()
        assert inside.event is not None
        _assert_operation_refused(
            place, transaction, state_before, result, refusal,
            trace.records(inside.event),
        )
        _expect_remembered_inside(inside, transaction.root)
        _expect_next_bracket_call_from_memory(inside)
        if operation == "commit":
            assert transaction.journal.settle_or_restore(transaction.corpus)
        else:
            assert transaction.journal.restore()
    if flicker.moved:
        pytest.skip("the file system stamped a store through the mapping")


# ------------------------------- no write to the checkout on a remembered answer

@steady_wall_clock
def test_write_through_a_held_descriptor_asks_git_after_an_answer_from_memory(
    tmp_path: Path,
    git_calls: _GitCalls,
) -> None:
    """A helper may hand out a held descriptor on a remembered answer.

    That is enough to read through it.  Before a write the proof is repeated
    with Git asked, once: where Git was asked since, nothing is repeated.
    """

    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=root,
        journal_root=tmp_path / "corpus" / "runs" / "witness" / "mutations",
        slice_id="witness-write",
        scope=("a.py",),
        sequence=1,
    )
    write_proof = journals._ask_git_before_held_write
    with journals.workspace_mutation_lock(root) as authority:
        _expect_answer_from_memory(root, git_calls)
        # No descriptor is handed out: nothing to prove, whatever is reused.
        assert _observe(git_calls, write_proof) == _Outcome(None, 0)
        with journals._bound_workspace_restore_root(journal):
            first = _observe(git_calls, write_proof)
            second = _observe(git_calls, write_proof)
        assert second == _Outcome(None, 0)
        assert first.refusal is None
        from_memory(first.git == 2, first)
        # A checkout named by the caller is proved like one handed out.
        _expect_no_git(git_calls, lambda: _bracket(root))
        assert _observe(git_calls, lambda: write_proof(authority)) == _Outcome(None, 2)
        assert _observe(git_calls, lambda: write_proof(authority)) == _Outcome(None, 0)
        # Git itself decides: a checkout it cannot read is refused here.
        _expect_no_git(git_calls, lambda: _bracket(root))
        head = ctx["git_dir"] / "HEAD"
        with journal._bound_git_admin_path(head):
            undo = _rename_away(head)
            try:
                refused = _observe(git_calls, write_proof)
            finally:
                undo()
        assert refused.refusal is not None and _UNREADABLE in refused.refusal
    assert journal.restore()


#: place -> (the function whose first bracket call the store is made at,
#: whether the scoped file had been put back before)
_ROLLBACK_WRITES: dict[str, tuple[str, bool]] = {
    "scoped-file": ("_restore_snapshot_root", False),
    "control-file": ("_restore_git_admin_topology", True),
}


@pytest.mark.timeout(300)
@pytest.mark.parametrize("place", sorted(_ROLLBACK_WRITES))
@steady_wall_clock
def test_rollback_writes_nothing_on_a_store_git_saw_only_once(
    place: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback restores through held descriptors, call after call.

    ``HEAD`` is stored through a shared mapping: unreadable, readable for the
    one Git question of the bracket call the store is made at, unreadable
    again, with no stamp moved.  The calls behind it reuse that answer.  The
    next write of the rollback, the scoped file's removal or the removal of
    the unreadable ``HEAD`` it is about to rewrite, asks Git first: the
    rollback stops there with the file and ``HEAD`` as they were.
    """

    function, file_restored = _ROLLBACK_WRITES[place]
    trace = _AssertionTrace(monkeypatch)
    with _operation(tmp_path, "restore") as (transaction, run):
        assert transaction.journal is not None
        head = transaction.ctx["git_dir"] / "HEAD"
        identity = os.lstat(head).st_ino
        flicker = _MappedHeadFlicker(head, monkeypatch)
        inside = _StateInside(
            trace, flicker.start, at=_first_bracket_call_inside(function),
        )
        _settle(transaction.root)
        trace.armed = True
        try:
            result, refusal = _run(run)
        finally:
            trace.armed = False
            left = (os.lstat(head).st_ino, head.read_bytes()[:1])
            inside.finish()
        assert inside.event is not None
        assert result is False and refusal is None
        assert transaction.journal.state == "invoking"
        assert "rolled_back" not in trace.records(inside.event)
        assert left == (identity, b"x")
        assert transaction.target.read_bytes() == (
            transaction.baseline if file_restored else transaction.written
        )
        _expect_remembered_inside(inside, transaction.root)
        _expect_next_bracket_call_from_memory(inside)
        assert transaction.journal.restore()
    if flicker.moved:
        pytest.skip("the file system stamped a store through the mapping")


def _plain_directory(root: Path) -> None:
    """Commit a directory with one file to the checkout."""

    directory = root / "pkg"
    directory.mkdir()
    (directory / "c.py").write_text("# base c.py\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "pkg")


def _closed_directory(root: Path) -> None:
    """The same directory, closed to writing at the baseline."""

    _plain_directory(root)
    (root / "pkg").chmod(0o555)


_DIRECTORIES: dict[str, Callable[[Path], None]] = {
    "plain": _plain_directory,
    "closed-to-writing": _closed_directory,
}


@pytest.mark.timeout(300)
@pytest.mark.parametrize("directory", sorted(_DIRECTORIES))
@pytest.mark.parametrize("operation", ["reap", "restore", "restore_paths"])
def test_no_mode_is_set_in_the_checkout_on_a_remembered_answer(
    operation: str,
    directory: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback sets the mode of every directory it knows, changed or not.

    Setting the mode a directory already has moves its change time, so it
    is a write through a held descriptor like any other: Git is asked
    before it wherever the last discovery answer came from memory.  The
    checkout has a directory below its root, whose mode a rollback sets
    before the root's.  Where that directory is closed to writing at the
    baseline, a rollback opens it and closes it again at its end: two
    mode calls that change the mode.
    """

    trace = _AssertionTrace(monkeypatch)
    with _operation(
        tmp_path, operation, _DIRECTORIES[directory],
    ) as (transaction, run):
        below = transaction.root / "pkg"
        baseline = stat.S_IMODE(os.lstat(below).st_mode)
        trace.watch_modes(monkeypatch, transaction.root)
        _settle(transaction.root)
        trace.armed = True
        try:
            run()
        finally:
            trace.armed = False
        after = stat.S_IMODE(os.lstat(below).st_mode)
    below.chmod(0o755)
    assert trace.count("mode")
    assert trace.modes_on_remembered_answers() == []
    assert after == baseline
    if operation != "restore_paths":
        assert ("mode", "/pkg") in trace.events


def _put_in_its_place(entry: Path, replacement: Path) -> Callable[[], None]:
    """Park ``entry`` beside itself and put ``replacement`` in its place."""

    def exchange() -> None:
        entry.rename(entry.with_name(entry.name + ".parked"))
        replacement.rename(entry)

    return exchange


@pytest.mark.parametrize("removed", ["file", "directory", "cleanup-directory"])
@steady_wall_clock
def test_entry_replaced_while_git_is_asked_before_its_removal_stays(
    removed: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_calls: _GitCalls,
) -> None:
    """A removal reads its entry again and compares it with what it read.

    The question before a removal behind an answer from memory lasts as
    long as Git's processes run.  It is asked ahead of that comparison,
    which stays next to the removal: an entry that another process puts in
    the place of the checked one while Git is asked is refused, not
    removed.
    """

    ctx = _layout_plain(tmp_path)
    root = ctx["root"]
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=root,
        journal_root=tmp_path / "corpus" / "runs" / "witness" / "mutations",
        slice_id="witness-removal",
        scope=("a.py",),
        sequence=1,
    )
    entry = root / "made-inside"
    replacement = root / "made-inside.replacement"
    for path in (entry, replacement):
        if removed == "file":
            path.write_bytes(b"made inside the transaction\n")
        else:
            path.mkdir()
    kept = os.lstat(replacement).st_ino
    exchange = _put_in_its_place(entry, replacement)
    with journals.workspace_mutation_lock(root):
        _expect_answer_from_memory(root, git_calls)
        with journals._bound_workspace_restore_root(journal) as root_fd:
            shot = _good_only_while_git_is_asked_once(
                monkeypatch, exchange, lambda: None,
            )
            if removed == "cleanup-directory":
                directory_fd = os.open(entry, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    refused = _observe(
                        git_calls,
                        lambda: journals._remove_bound_directory_tree(
                            root_fd,
                            entry.name,
                            directory_fd,
                            os.fstat(directory_fd),
                        ),
                    )
                finally:
                    os.close(directory_fd)
            else:
                with journals._open_settlement_parent_if_present(
                    root_fd, entry.name,
                ) as (parent_fd, name, chain, missing):
                    assert missing is None
                    refused = _observe(
                        git_calls,
                        lambda: journals._remove_bound_workspace_entry(
                            root_fd, entry.name, parent_fd, name, chain,
                        ),
                    )
            asked_before_the_removal = shot["fired"]
            # No later question of this test exchanges anything.
            shot["fired"] = True
    from_memory(
        asked_before_the_removal,
        "the removal did not follow an answer from memory",
    )
    assert refused.refusal is not None and "changed before" in refused.refusal
    assert refused.git == 2
    # What lies in the entry's place is the replacement, as it was put there.
    assert os.lstat(entry).st_ino == kept
    assert journal.restore()


@pytest.mark.timeout(300)
@steady_wall_clock
def test_nested_journal_is_not_rolled_back_on_a_store_git_saw_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback hands over to the rollbacks of its nested journals.

    A checkout with a submodule has a nested journal for it, which is rolled
    back under the proof of the submodule's own checkout.  The store is made
    in the outer checkout at the last bracket call before that hand-over:
    the outer rollback asks Git first and stops, and the nested journal is
    left as it was.
    """

    ctx = _layout_submodule(tmp_path)
    outer = ctx["super"]
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=outer,
        journal_root=tmp_path / "corpus" / "runs" / "witness" / "mutations",
        slice_id="s0",
        scope=("a.py",),
        sequence=1,
    )
    nested = journal._nested_journals_for_restore()
    assert nested
    (outer / "a.py").write_bytes(b"# written inside the transaction\n")
    trace = _AssertionTrace(monkeypatch)
    flicker = _MappedHeadFlicker(outer / ".git" / "HEAD", monkeypatch)
    inside = _StateInside(
        trace,
        flicker.start,
        at=_first_bracket_call_inside(
            "_bound_workspace_restore_root", "_prepare_restore_access",
        ),
    )
    with journals.workspace_mutation_lock(outer):
        _settle(outer)
        trace.armed = True
        try:
            result = journal.restore()
        finally:
            trace.armed = False
            inside.finish()
        assert inside.event is not None
        assert result is False
        assert trace.records(inside.event) == []
        assert {_durable_state(item) for item in nested} == {"invoking"}
        _expect_remembered_inside(inside, outer)
        _expect_next_bracket_call_from_memory(inside)
        assert journal.restore()
    if flicker.moved:
        pytest.skip("the file system stamped a store through the mapping")


@pytest.mark.timeout(300)
@steady_wall_clock
def test_recovery_writes_no_source_on_a_store_git_saw_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact journal recovery restores the source as a rollback does.

    The store is made at the first bracket call of that restore.  The
    recovery stops before its first source write and records no further
    stage; resumed with a readable checkout it completes.
    """

    from .test_workflows_journal_recover import _apply, _setup

    recovery = _setup(tmp_path)
    source = recovery["source"]
    assert isinstance(source, Path)
    files = [source / "a.txt", source / "new.txt"]
    before = [path.read_bytes() for path in files]
    trace = _AssertionTrace(monkeypatch)
    flicker = _MappedHeadFlicker(source / ".git" / "HEAD", monkeypatch)
    inside = _StateInside(
        trace,
        flicker.start,
        at=_first_bracket_call_inside(
            "_restore_source_only_bound", "_restore_snapshot_root",
        ),
    )
    _settle(source)
    trace.armed = True
    try:
        _result, refusal = _run(lambda: _apply(recovery))
    finally:
        trace.armed = False
        after = [path.read_bytes() if path.exists() else None for path in files]
        inside.finish()
    assert inside.event is not None
    assert refusal is not None
    assert after == before
    assert not [
        detail for kind, detail in trace.events[inside.event:] if kind == "stage"
    ]
    _expect_remembered_inside(inside, source)
    _expect_next_bracket_call_from_memory(inside)
    assert getattr(_apply(recovery), "status") == "recovered"
    if flicker.moved:
        pytest.skip("the file system stamped a store through the mapping")
