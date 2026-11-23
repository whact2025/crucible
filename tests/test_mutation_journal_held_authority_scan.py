# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""A dropped workspace lock owner released while another owner is at work.

A caller that drops an entered workspace lock without exiting it leaves the
owner registered in its thread, with its descriptors open, until a garbage
collection finalizes the lock generators.  That collection can begin at any
call inside a journal operation of a later, healthy owner of the same
thread.  Each regression here switches the automatic collector off and
places exactly one collection at such a call; each fail-closed pin states
which descriptor numbers the held-authority scan never recognizes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import errno
import gc
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import FrameType

import pytest

from crucible.workflows import _mutation_journal as journals


pytestmark = pytest.mark.skipif(
    os.name == "nt"
    or not hasattr(os, "O_PATH")
    or not os.path.isdir("/proc/self/fd"),
    reason=(
        "workspace authority descriptors require POSIX O_PATH/flock and "
        "/proc/self/fd"
    ),
)

_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent/held-authority-scan-test",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "maintenance.auto=false", "-c", "gc.auto=0", *args],
        cwd=cwd,
        env=_GIT_ENV,
        check=True,
        capture_output=True,
        timeout=30,
    )


def _open_descriptors() -> frozenset[int]:
    found = set()
    for name in os.listdir("/proc/self/fd"):
        try:
            os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            # The descriptor of this listing itself is closed again.
            continue
        found.add(int(name))
    return frozenset(found)


def _are_open(descriptors: Iterable[int]) -> list[bool]:
    table = _open_descriptors()
    return [descriptor in table for descriptor in descriptors]


def _directory_descriptors(
    authority: journals._WorkspaceMutationAuthority,
) -> tuple[int, int, int, int]:
    """The four descriptors the scan may recognize for one authority."""

    return (
        authority.root_descriptor,
        authority.common_descriptor,
        authority.git_common_descriptor,
        authority.git_dir_descriptor,
    )


def _held_descriptors() -> frozenset[int]:
    return frozenset(
        journals._HELD_AUTHORITY_DESCRIPTORS | journals._HELD_LOCK_DESCRIPTORS
    )


def _journal_state() -> tuple[frozenset[object], ...]:
    """What a lock chain publishes in this process and this thread."""

    return (
        frozenset(journals._HELD_AUTHORITY_DESCRIPTORS),
        frozenset(journals._HELD_LOCK_DESCRIPTORS),
        frozenset(
            os.fspath(key)
            for key in tuple(getattr(journals._LOCK_STATE, "authorities", {}))
        ),
        frozenset(
            os.fspath(coordination.binding.root)
            for coordination in tuple(getattr(
                journals._LOCK_STATE,
                "external_workspace_authorities",
                (),
            ))
        ),
        _open_descriptors(),
    )


class _Collection:
    """One garbage collection, placed by a hook inside a product call."""

    def __init__(self, released: Iterable[int] = ()) -> None:
        self._released = tuple(released)
        self.placed_at: str | None = None
        self.left_open: list[bool] = []

    def __call__(self, where: str) -> None:
        if self.placed_at is None:
            self.placed_at = where
            gc.collect()
            # Read here: a later open may take a released number again.
            self.left_open = _are_open(self._released)


class _Placement:
    """Profile hook that acts once, at the n-th call or return of the scan.

    ``sys.setprofile`` reports where a call starts and where it ends.  The
    hook counts the reports that code of the journal module causes while the
    scan is on the stack: the entries and exits of the journal's own
    functions and generators, its calls of C functions, and the entries and
    exits of the Python functions of other modules that it calls.  A sweep
    over n therefore reaches every such call the scan makes, also one that
    a later edit adds.  The interpreter has no report for the call of a
    class that is implemented in C, such as ``str``, so that is no placement.
    """

    def __init__(self, index: int, action: Callable[[], None]) -> None:
        self._index = index
        self._action = action
        self._seen = 0
        self.placed_at: str | None = None
        self.at_return_of_the_scan = False

    def __call__(self, frame: FrameType, event: str, argument: object) -> None:
        if self.placed_at is not None:
            return
        journal = vars(journals)
        scan = journals._held_authority_descriptor_path.__code__
        maker: FrameType | None = frame
        if event in ("call", "return") and frame.f_globals is not journal:
            # A function of another module: the frame that called it counts.
            maker = frame.f_back
        if maker is None or maker.f_globals is not journal:
            return
        while maker is not None and maker.f_code is not scan:
            maker = maker.f_back
        if maker is None:
            return
        self._seen += 1
        if self._seen != self._index:
            return
        if event in ("call", "return"):
            subject = frame.f_code.co_qualname
        else:
            subject = (
                f"{getattr(argument, '__qualname__', argument)} in "
                f"{frame.f_code.co_qualname}"
            )
        self.placed_at = f"{event} {subject}, line {frame.f_lineno}"
        self.at_return_of_the_scan = event == "return" and frame.f_code is scan
        self._action()


class _Holder(threading.Thread):
    """Another thread that acquires one workspace lock and holds it."""

    def __init__(self, repository: Path) -> None:
        super().__init__(daemon=True)
        self._repository = repository
        self._release = threading.Event()
        self.entered = threading.Event()
        self.descriptors: tuple[int, ...] = ()
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            with journals.workspace_mutation_lock(self._repository) as held:
                self.descriptors = _directory_descriptors(held)
                self.entered.set()
                self._release.wait(60)
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            self.error = exc
        finally:
            self.entered.set()

    def finish(self) -> None:
        """Release the lock and join before the test returns."""

        self._release.set()
        if self.ident is not None:
            self.join(60)
            assert not self.is_alive()
        assert self.error is None, self.error


class _Scene:
    """Repositories of one test and the descriptors it opened itself."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._owned: list[int] = []

    def repository(self, name: str) -> Path:
        repository = self._root / name
        repository.mkdir()
        _git(repository, "init", "-q", "-b", "main")
        _git(repository, "config", "user.email", "held-authority@test.invalid")
        _git(repository, "config", "user.name", "Held Authority Test")
        (repository / "tracked.txt").write_text("baseline\n", encoding="utf-8")
        _git(repository, "add", "tracked.txt")
        _git(repository, "commit", "-qm", "baseline")
        return repository

    def journal(self, checkout: Path) -> journals.WorkspaceMutationJournal:
        return journals.WorkspaceMutationJournal.begin(
            cwd=checkout,
            journal_root=self._root / "corpus" / "runs" / "scan" / "mutations",
            slice_id="held-authority-scan",
            scope=("tracked.txt",),
            sequence=1,
        )

    def drop_owner(self, name: str) -> journals._WorkspaceMutationAuthority:
        """Enter a workspace lock for a new repository and drop it unexited."""

        return self.drop_owner_of(self.repository(name))

    def drop_owner_of(
        self,
        repository: Path,
    ) -> journals._WorkspaceMutationAuthority:
        """Enter a workspace lock for ``repository`` and drop it unexited.

        The owner stays reachable only through a reference cycle, as the
        retained owner of a dropped workflow error does, so nothing releases
        it before a garbage collection.
        """

        owner = journals.workspace_mutation_lock(repository)
        authority = owner.__enter__()
        cycle: dict[str, object] = {"owner": owner}
        cycle["self"] = cycle
        return authority

    def unrelated_directory(self) -> int:
        directory = self._root / f"unrelated-{len(self._owned)}"
        directory.mkdir()
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        self._owned.append(descriptor)
        return descriptor

    def reuse(self, source: int, number: int) -> None:
        """Make the closed ``number`` name what ``source`` has open."""

        os.dup2(source, number, inheritable=False)
        self._owned.append(number)

    def close(self) -> None:
        for descriptor in self._owned:
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.fixture
def scene(tmp_path: Path) -> Iterator[_Scene]:
    """Collector off; afterwards the journal holds nothing it did not hold."""

    gc.collect()
    collector_was_enabled = gc.isenabled()
    gc.disable()
    before = _journal_state()
    current = _Scene(tmp_path)
    try:
        yield current
    finally:
        current.close()
        gc.collect()
        if collector_was_enabled:
            gc.enable()
    left = [
        sorted(after - incoming)
        for after, incoming in zip(_journal_state(), before)
    ]
    assert left == [[], [], [], [], []]


def _scan(path: Path) -> tuple[int, Path, Path] | None:
    return journals._held_authority_descriptor_path(path)


def _alias(descriptor: int) -> Path:
    return Path(f"/proc/self/fd/{descriptor}")


# ---------- a healthy owner while another owner of the thread is released ----------


def test_scan_survives_another_owner_released_between_snapshot_and_inspection(
    scene: _Scene,
) -> None:
    dropped = _directory_descriptors(scene.drop_owner("dropped"))
    collection = _Collection()

    class _FirstUse(type(Path())):
        """A path whose first use inside the scan places the collection."""

        def relative_to(self, other, *args, **kwargs):
            collection("relative_to")
            return super().relative_to(other, *args, **kwargs)

        @property
        def parts(self):
            collection("parts")
            return super().parts

    with journals.workspace_mutation_lock(scene.repository("live")) as live:
        assert _are_open(dropped) == [True] * 4

        binding = _scan(_FirstUse("/nonexistent/names-no-descriptor"))

        # The collection ran inside the scan and released the other owner.
        assert collection.placed_at is not None
        assert _are_open(dropped) == [False] * 4
        assert binding is None
        assert _are_open(_directory_descriptors(live)) == [True] * 4
        assert journals._assert_workspace_mutation_authority(live.root) is live


@pytest.mark.parametrize(
    "placement",
    ["registry_read", "inspection", "last_check"],
)
def test_scan_survives_another_owner_released_while_its_own_descriptor_is_checked(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
    placement: str,
) -> None:
    dropped = _directory_descriptors(scene.drop_owner("dropped"))
    collection = _Collection()
    with journals.workspace_mutation_lock(scene.repository("live")) as live:
        named = live.root_descriptor
        inspected = False
        fstat = os.fstat
        getpid = os.getpid

        def observed_fstat(descriptor, *args, **kwargs):
            nonlocal inspected
            result = fstat(descriptor, *args, **kwargs)
            if descriptor == named:
                inspected = True
                if placement == "inspection":
                    collection("after fstat of the named descriptor")
            return result

        def observed_getpid():
            if placement == "registry_read":
                collection("first getpid of the scan")
            return getpid()

        def observed_getattr(*args):
            # Shadows the builtin for the journal module only.
            if placement == "last_check" and inspected:
                collection("registry read of the last check")
            return getattr(*args)

        with monkeypatch.context() as patch:
            patch.setattr(os, "fstat", observed_fstat)
            patch.setattr(os, "getpid", observed_getpid)
            patch.setattr(journals, "getattr", observed_getattr, raising=False)
            binding = _scan(_alias(named) / "sub")

        assert collection.placed_at is not None
        assert _are_open(dropped) == [False] * 4
        assert binding == (named, _alias(named), Path("sub"))
        assert _are_open(_directory_descriptors(live)) == [True] * 4


def test_scan_inspects_only_the_descriptor_the_path_names(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped = _directory_descriptors(scene.drop_owner("dropped"))
    collection = _Collection()
    with journals.workspace_mutation_lock(scene.repository("live")) as live:
        inspected: list[int] = []
        fstat = os.fstat

        def observed_fstat(descriptor, *args, **kwargs):
            result = fstat(descriptor, *args, **kwargs)
            inspected.append(descriptor)
            if descriptor in dropped:
                collection("after fstat of another owner's descriptor")
            return result

        named = live.git_dir_descriptor
        with monkeypatch.context() as patch:
            patch.setattr(os, "fstat", observed_fstat)
            unnamed_binding = _scan(Path("/nonexistent/names-no-descriptor"))
            inspected_for_no_name = list(inspected)
            inspected.clear()
            named_binding = _scan(_alias(named) / "hooks")

        # No descriptor of another owner is looked at, so its release between
        # the two stats of one number cannot fail this owner.
        assert unnamed_binding is None
        assert inspected_for_no_name == []
        assert named_binding == (named, _alias(named), Path("hooks"))
        assert inspected == [named]
        assert collection.placed_at is None
        assert _are_open(dropped) == [True] * 4


def test_control_path_survives_another_owner_released_in_its_registry_loop(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped = _directory_descriptors(scene.drop_owner("dropped"))
    collection = _Collection()
    with journals.workspace_mutation_lock(scene.repository("live")) as live:
        scan = journals._held_authority_descriptor_path
        getpid = os.getpid
        scanned = False

        def observed_scan(path):
            nonlocal scanned
            try:
                return scan(path)
            finally:
                scanned = True

        def observed_getpid():
            if scanned:
                collection("first getpid after the scan returned")
            return getpid()

        with monkeypatch.context() as patch:
            patch.setattr(
                journals,
                "_held_authority_descriptor_path",
                observed_scan,
            )
            patch.setattr(os, "getpid", observed_getpid)
            logical = journals._lexical_absolute_control_path(
                Path("/"),
                f"/proc/self/fd/{live.git_dir_descriptor}/hooks",
            )

        assert collection.placed_at is not None
        assert _are_open(dropped) == [False] * 4
        assert logical == live.git_dir / "hooks"


@pytest.mark.parametrize(
    ("function", "table", "result"),
    [
        ("_external_file_lock", "held", "held_keys"),
        (
            "_canonical_workspace_mutation_lock",
            "held_before_acquire",
            "held_independent_keys",
        ),
    ],
    ids=["external_common_lock", "canonical_lock"],
)
def test_lock_acquisition_survives_another_owner_released_while_a_held_table_is_read(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
    function: str,
    table: str,
    result: str,
) -> None:
    dropped = _directory_descriptors(scene.drop_owner("dropped"))
    repository = scene.repository("live")
    collection = _Collection(dropped)
    getpid = os.getpid

    def observed_getpid():
        frame = sys._getframe(1)
        local = frame.f_locals
        if (
            frame.f_code.co_name == function
            and local.get("kind", "common") == "common"
            and table in local
            and result not in local
        ):
            # The table is bound and the list built from it is not: this
            # call filters one entry of the table.
            collection(f"getpid while {function} reads {table}")
        return getpid()

    with monkeypatch.context() as patch:
        patch.setattr(os, "getpid", observed_getpid)
        with journals.workspace_mutation_lock(repository) as live:
            assert collection.placed_at is not None
            assert collection.left_open == [False] * 4
            assert (
                journals._assert_workspace_mutation_authority(repository)
                is live
            )


@pytest.mark.parametrize("order", ["dropped_then_live", "live_then_dropped"])
def test_external_coordination_release_survives_another_owner_released_after_its_length_was_read(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
    order: str,
) -> None:
    repository = scene.repository("live")
    if order == "dropped_then_live":
        dropped = _directory_descriptors(scene.drop_owner("dropped"))
    owner = journals.workspace_mutation_lock(repository)
    owner.__enter__()
    if order == "live_then_dropped":
        dropped = _directory_descriptors(scene.drop_owner("dropped"))
    registered = len(journals._LOCK_STATE.external_workspace_authorities)
    collection = _Collection()

    def observed_len(value):
        # Shadows the builtin for the journal module only.
        length = len(value)
        if (
            sys._getframe(1).f_code.co_name
            == "_external_workspace_coordination"
        ):
            collection("after len() of the coordination list")
        return length

    with monkeypatch.context() as patch:
        patch.setattr(journals, "len", observed_len, raising=False)
        owner.__exit__(None, None, None)

    assert collection.placed_at is not None
    assert _are_open(dropped) == [False] * 4
    # Both entries are gone, and the checkout can be locked again.
    assert (
        len(journals._LOCK_STATE.external_workspace_authorities)
        == registered - 2
    )
    with journals.workspace_mutation_lock(repository) as again:
        assert journals._assert_workspace_mutation_authority(repository) is again


def test_external_coordination_lookup_survives_other_owners_released_during_the_lookup(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with journals.workspace_mutation_lock(scene.repository("live")) as live:
        # Registered after the live owner: a reversed walk meets them first.
        dropped = [
            *_directory_descriptors(scene.drop_owner("dropped-first")),
            *_directory_descriptors(scene.drop_owner("dropped-second")),
        ]
        collection = _Collection()
        getpid = os.getpid

        def observed_getpid():
            if (
                sys._getframe(1).f_code.co_name
                == "_active_external_workspace_coordination"
            ):
                collection("first getpid of the coordination lookup")
            return getpid()

        with monkeypatch.context() as patch:
            patch.setattr(os, "getpid", observed_getpid)
            proved = journals._assert_workspace_mutation_authority(live.root)

        assert collection.placed_at is not None
        assert _are_open(dropped) == [False] * 8
        assert proved is live


def test_git_admin_authority_survives_another_owner_released_while_the_registry_is_read(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = scene.repository("live")
    journal = scene.journal(repository)
    active = journals._active_workspace_mutation_authority
    getpid = os.getpid

    def read(reader: Callable[[], object]) -> tuple[object, _Collection]:
        collection = _Collection(
            _directory_descriptors(scene.drop_owner(f"dropped-{reader.__name__}"))
        )
        looked_up = False

        def observed_active(cwd):
            nonlocal looked_up
            try:
                return active(cwd)
            finally:
                looked_up = True

        def observed_getpid():
            if looked_up:
                collection("first getpid after the exact lookup")
            return getpid()

        with monkeypatch.context() as patch:
            patch.setattr(
                journals,
                "_active_workspace_mutation_authority",
                observed_active,
            )
            patch.setattr(os, "getpid", observed_getpid)
            return reader(), collection

    # No owner of this checkout in the thread: another repository's dropped
    # owner neither answers the question nor fails it.
    present, collection = read(journal._has_held_git_admin_authority)
    assert collection.placed_at is not None
    assert collection.left_open == [False] * 4
    assert present is False

    # The dropped owner is registered before the live one, so the registry
    # yields it first.
    dropped = scene.drop_owner("dropped-before-live")
    with journals.workspace_mutation_lock(repository) as live:
        listed, collection = read(journal._git_admin_workspace_authorities)
        assert collection.placed_at is not None
        assert collection.left_open == [False] * 4
        assert _are_open(_directory_descriptors(dropped)) == [False] * 4
        assert listed == (live,)

    assert journal.restore()


# ---------- what the scan never recognizes ----------


@pytest.mark.parametrize(
    ("placement", "reuse"),
    [
        ("inspection", "unrelated_directory"),
        ("inspection", "another_threads_authority"),
        ("last_check", "unrelated_directory"),
        ("last_check", "another_threads_authority"),
        ("last_check", "nothing"),
    ],
)
def test_scan_refuses_a_named_descriptor_released_and_reused_during_inspection(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
    placement: str,
    reuse: str,
) -> None:
    named = scene.drop_owner("dropped").root_descriptor
    unrelated = scene.unrelated_directory()
    holder = _Holder(scene.repository("other-thread"))
    main = threading.get_ident()
    events: list[str] = []
    inspected = False
    fstat = os.fstat

    def release_and_reuse(where: str) -> None:
        if events:
            return
        events.append(where)
        gc.collect()
        events.append(
            "closed" if named not in _open_descriptors() else "still open"
        )
        if reuse == "unrelated_directory":
            scene.reuse(unrelated, named)
        elif reuse == "another_threads_authority":
            holder.start()
            holder.entered.wait(60)

    def observed_fstat(descriptor, *args, **kwargs):
        nonlocal inspected
        if threading.get_ident() != main or descriptor != named:
            return fstat(descriptor, *args, **kwargs)
        if placement == "inspection":
            release_and_reuse("fstat of the named descriptor")
        result = fstat(descriptor, *args, **kwargs)
        inspected = True
        return result

    def observed_getattr(*args):
        # Shadows the builtin for the journal module only.
        if (
            threading.get_ident() == main
            and placement == "last_check"
            and inspected
        ):
            release_and_reuse("registry read of the last check")
        return getattr(*args)

    try:
        with journals.workspace_mutation_lock(scene.repository("live")):
            with monkeypatch.context() as patch:
                patch.setattr(os, "fstat", observed_fstat)
                patch.setattr(
                    journals,
                    "getattr",
                    observed_getattr,
                    raising=False,
                )
                with pytest.raises(
                    journals.MutationJournalError,
                    match="was released during inspection",
                ):
                    _scan(_alias(named) / "x")

            # The owner was released inside the scan, and by the time the
            # scan decided the number named something else.
            assert events[1:] == ["closed"]
            if reuse == "unrelated_directory":
                assert named in _open_descriptors()
                assert named not in journals._HELD_AUTHORITY_DESCRIPTORS
            elif reuse == "another_threads_authority":
                assert holder.error is None
                assert named in holder.descriptors
                assert named in journals._HELD_AUTHORITY_DESCRIPTORS
            else:
                assert named not in _open_descriptors()
    finally:
        holder.finish()


_SWEEP_LIMIT = 200


def _scan_with_one_placement(
    index: int,
    path: Path,
    action: Callable[[], None],
) -> tuple[_Placement, object]:
    """Run the scan with ``action`` at its ``index``-th call or return."""

    placement = _Placement(index, action)
    profile = sys.getprofile()
    sys.setprofile(placement)
    try:
        try:
            return placement, _scan(path)
        except journals.MutationJournalError as refusal:
            return placement, refusal
    finally:
        sys.setprofile(profile)


@pytest.mark.parametrize(
    "reuse",
    ["unrelated_directory", "another_threads_authority"],
)
def test_scan_never_binds_a_named_descriptor_released_and_reused_at_any_call_it_makes(
    scene: _Scene,
    reuse: str,
) -> None:
    """One release and reuse of the named number at every call of the scan.

    The cases above each place the release at one call that they name.  The
    sweep places it at every call and return in turn, so it does not depend
    on the form of the scan: a call that an edit puts between the scan's
    last look at the registry and its return is a placement too, and a
    release placed there is found.  Until the scan itself returns, a number
    that was released and names something else is "not held" or refused,
    never bound.
    """

    dropped_repository = scene.repository("dropped")
    other_repository = scene.repository("other-thread")
    unrelated = scene.unrelated_directory()
    answers: list[tuple[str, object]] = []
    returned_at: list[int] = []

    with journals.workspace_mutation_lock(scene.repository("live")):
        for index in range(1, _SWEEP_LIMIT + 1):
            named = scene.drop_owner_of(dropped_repository).root_descriptor
            binding = (named, _alias(named), Path("x"))
            holder = _Holder(other_repository)
            closed: list[bool] = []

            def release_and_reuse() -> None:
                gc.collect()
                closed.append(named not in _open_descriptors())
                if closed != [True]:
                    return
                if reuse == "unrelated_directory":
                    os.dup2(unrelated, named, inheritable=False)
                else:
                    holder.start()
                    holder.entered.wait(60)

            try:
                placement, answer = _scan_with_one_placement(
                    index,
                    _alias(named) / "x",
                    release_and_reuse,
                )
                if placement.placed_at is None:
                    # Past the last call: nothing was released, and the
                    # alias of the dropped owner binds.
                    assert answer == binding
                    break
                # The owner was released at the placement, and from then on
                # the number named something else.
                assert closed == [True], placement.placed_at
                if reuse == "unrelated_directory":
                    assert named in _open_descriptors()
                    assert named not in journals._HELD_AUTHORITY_DESCRIPTORS
                else:
                    assert holder.error is None
                    assert named in holder.descriptors
                    assert named in journals._HELD_AUTHORITY_DESCRIPTORS
                if placement.at_return_of_the_scan:
                    # The answer had been given when the owner was released.
                    assert answer == binding
                    returned_at.append(index)
                answers.append((placement.placed_at, answer))
            finally:
                holder.finish()
                if closed != [True]:
                    gc.collect()
                elif reuse == "unrelated_directory":
                    os.close(named)
        else:
            pytest.fail(
                f"the scan makes more than {_SWEEP_LIMIT} calls and returns"
            )

    # The sweep began at the entry of the scan, where the release precedes
    # everything the scan reads, and ended at the scan's own return.
    assert answers[0][1] is None, answers[0]
    assert returned_at == [len(answers)]
    bound = [
        f"{where}: {answer!r}"
        for where, answer in answers[:-1]
        if answer is not None
        and not isinstance(answer, journals.MutationJournalError)
    ]
    assert bound == []
    # A release at the last call before the return is refused by the scan's
    # last check: nothing else is left to notice it.
    where, answer = answers[-2]
    assert isinstance(answer, journals.MutationJournalError), where
    assert "was released during inspection" in str(answer), where


def test_scan_reports_a_named_descriptor_released_while_it_is_inspected(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named = scene.drop_owner("dropped").root_descriptor
    collection = _Collection()
    fstat = os.fstat

    def observed_fstat(descriptor, *args, **kwargs):
        result = fstat(descriptor, *args, **kwargs)
        if descriptor == named:
            collection("after fstat of the named descriptor")
        return result

    with journals.workspace_mutation_lock(scene.repository("live")):
        with monkeypatch.context() as patch:
            patch.setattr(os, "fstat", observed_fstat)
            # Not "an ordinary path": the caller must not go on with a dead
            # alias as a path prefix.
            with pytest.raises(
                journals.MutationJournalError,
                match="platform cannot bind Git authority discovery",
            ):
                _scan(_alias(named) / "x")
        assert collection.placed_at is not None
        assert named not in _open_descriptors()


def test_scan_reports_a_named_descriptor_it_cannot_inspect(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with journals.workspace_mutation_lock(scene.repository("live")) as live:
        named = live.root_descriptor
        fstat = os.fstat

        def failing_fstat(descriptor, *args, **kwargs):
            if descriptor == named:
                raise OSError(errno.EBADF, os.strerror(errno.EBADF))
            return fstat(descriptor, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "fstat", failing_fstat)
            with pytest.raises(
                journals.MutationJournalError,
                match="cannot inspect held workspace authority descriptor",
            ):
                _scan(_alias(named) / "x")


def test_scan_does_not_recognize_the_number_of_a_released_owner(
    scene: _Scene,
) -> None:
    unrelated = scene.unrelated_directory()
    with journals.workspace_mutation_lock(scene.repository("live")):
        with journals.workspace_mutation_lock(
            scene.repository("released")
        ) as gone:
            released = _directory_descriptors(gone)
            for number in released:
                assert _scan(_alias(number) / "x") == (
                    number,
                    _alias(number),
                    Path("x"),
                )

        assert _are_open(released) == [False] * 4
        for number in released:
            assert _scan(_alias(number) / "x") is None
        for number in released:
            scene.reuse(unrelated, number)
            assert _scan(_alias(number) / "x") is None


def test_scan_does_not_recognize_a_number_held_for_something_else(
    scene: _Scene,
) -> None:
    held_before = _held_descriptors()
    dropped = scene.drop_owner("dropped")
    unrelated = scene.unrelated_directory()
    with journals.workspace_mutation_lock(scene.repository("live")) as live:
        recognized = {
            *_directory_descriptors(live),
            *_directory_descriptors(dropped),
        }
        others = _held_descriptors() - held_before - recognized

        # Per lock chain: the pre-root pair, the three of each of the two
        # external lock namespaces, and the three lock files.
        assert len(others) == 2 * (2 + 3 + 3 + 3)
        assert {live.lock_descriptor, dropped.lock_descriptor} <= others
        for number in sorted(others | {unrelated}):
            assert _scan(_alias(number) / "x") is None, number


def test_scan_recognizes_only_the_spelling_the_platform_binds(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with journals.workspace_mutation_lock(scene.repository("live")) as live:
        root = live.root_descriptor
        proc = _alias(root)
        dev = Path(f"/dev/fd/{root}")

        assert _scan(proc) == (root, proc, Path("."))
        assert _scan(proc / "a" / "b") == (root, proc, Path("a/b"))
        for spelling in (
            f"/dev/fd/{root}",
            f"/dev/fd/{root}/a",
            f"/proc/self/fd/0{root}/a",
            f"/proc/self/fd/+{root}/a",
            f"/proc/self/fd/-{root}/a",
            f"/proc/self/fd/{root}x/a",
            f"/proc/self/fd/{root} /a",
            f"//proc/self/fd/{root}/a",
            f"/proc/{os.getpid()}/fd/{root}/a",
            f"/proc/thread-self/fd/{root}/a",
            f"proc/self/fd/{root}/a",
            "/proc/self/fd",
            "/dev/fd",
        ):
            assert _scan(Path(spelling)) is None, spelling

        stat = os.stat

        def stat_without_proc(path, *args, **kwargs):
            if isinstance(path, str) and path.startswith("/proc/self/fd/"):
                raise FileNotFoundError(errno.ENOENT, "no procfs", path)
            return stat(path, *args, **kwargs)

        # Where the kernel binds the other spelling the roles swap.
        with monkeypatch.context() as patch:
            patch.setattr(os, "stat", stat_without_proc)
            assert _scan(dev / "a") == (root, dev, Path("a"))
            assert _scan(proc / "a") is None


def test_scan_rejects_parent_traversal_below_a_held_alias(
    scene: _Scene,
) -> None:
    with journals.workspace_mutation_lock(scene.repository("live")) as live:
        root = live.root_descriptor
        for suffix in (f"../{live.git_dir_descriptor}", "a/../b", ".."):
            with pytest.raises(
                journals.MutationJournalError,
                match="held authority descriptor path escapes its root",
            ):
                _scan(_alias(root) / suffix)

        unheld = max(_open_descriptors()) + 1000
        assert _scan(_alias(unheld) / ".." / "x") is None


def test_scan_does_not_recognize_another_threads_authority(
    scene: _Scene,
) -> None:
    holder = _Holder(scene.repository("other-thread"))
    holder.start()
    try:
        assert holder.entered.wait(60)
        assert holder.error is None
        assert len(holder.descriptors) == 4
        assert set(holder.descriptors) <= journals._HELD_AUTHORITY_DESCRIPTORS

        for number in holder.descriptors:
            assert _scan(_alias(number) / "x") is None
        with journals.workspace_mutation_lock(scene.repository("live")):
            for number in holder.descriptors:
                assert _scan(_alias(number) / "x") is None
    finally:
        holder.finish()


# ---------- the release order the fork hook and the suite guard rely on ----------


def test_workspace_lock_release_drops_each_number_from_the_held_sets_before_closing_it(
    scene: _Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held_before = _held_descriptors()
    owner = journals.workspace_mutation_lock(scene.repository("live"))
    owner.__enter__()
    chain = _held_descriptors() - held_before
    closed: list[int] = []
    closed_while_held: list[int] = []
    close = os.close

    def observed_close(descriptor):
        if descriptor in _held_descriptors():
            closed_while_held.append(descriptor)
        closed.append(descriptor)
        return close(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "close", observed_close)
        owner.__exit__(None, None, None)

    # Twelve directory descriptors and three lock files per chain.
    assert len(chain) == 15
    assert chain <= set(closed)
    assert closed_while_held == []
