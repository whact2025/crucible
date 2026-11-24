# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Contracts for the suite guard against workspace authority left held."""

from __future__ import annotations

import ast
from collections.abc import Iterator
import gc
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from crucible.workflows import _mutation_journal as journals
from tests import conftest as shared_conftest


ROOT = Path(__file__).resolve().parents[1]
_GUARD = "_contain_workspace_authority"
_LEAKING_TEST = "tests/test_example.py::test_drops_an_owner"

_requires_workspace_locks = pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "O_PATH"),
    reason="workspace locks require POSIX O_PATH/flock",
)

_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent/workspace-authority-hygiene-test",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


def _repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=path,
        env=_GIT_ENV,
        check=True,
        capture_output=True,
        timeout=30,
    )
    return path


def _guard(nodeid: str = _LEAKING_TEST) -> Iterator[None]:
    """The guard as pytest runs it around one test, without pytest."""

    request = SimpleNamespace(node=SimpleNamespace(nodeid=nodeid))
    return shared_conftest._contain_workspace_authority.__wrapped__(request)


def _held() -> tuple[int, int, int]:
    return (
        len(journals._HELD_AUTHORITY_DESCRIPTORS),
        len(journals._HELD_LOCK_DESCRIPTORS),
        len(getattr(journals._LOCK_STATE, "authorities", {})),
    )


@pytest.fixture
def collector_off() -> Iterator[None]:
    """Only the guard decides when a dropped owner is collected."""

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


@pytest.fixture(autouse=True)
def _module_autouse_fixture() -> None:
    """Stands for any fixture a test module adds on its own."""


def _drop_owner(repository: Path) -> BaseException:
    """Enter a workspace lock and drop it unexited, owned by an error.

    The error stands for a workflow error that retains its lock owner; a
    reference cycle keeps both alive until a garbage collection.
    """

    owner = journals.workspace_mutation_lock(repository)
    owner.__enter__()
    error = RuntimeError("stands for an error that owns a lock chain")
    cycle: dict[str, object] = {"owner": owner, "error": error}
    cycle["self"] = cycle
    error.owner = cycle
    return error


def _passes(guard: Iterator[None]) -> None:
    with pytest.raises(StopIteration):
        next(guard)


@_requires_workspace_locks
def test_guard_charges_a_dropped_owner_and_releases_it_before_the_next_test(
    tmp_path: Path,
    collector_off: None,
) -> None:
    repository = _repository(tmp_path / "repo")
    incoming = _held()
    guard = _guard()
    next(guard)
    _drop_owner(repository)

    with pytest.raises(pytest.fail.Exception) as charged:
        next(guard)

    # The test is named, with what it left: the twelve directory and three
    # lock descriptors of one chain and the paths it is registered for.
    message = str(charged.value)
    assert message.startswith(
        f"{_LEAKING_TEST} ended with workspace authority that appeared "
        "during it: 12 authority descriptor(s) ["
    )
    assert "], 3 lock descriptor(s) [" in message
    assert "authority registered for [" in message
    assert repr(os.fspath(repository)) in message
    assert "retry_cleanup" in message
    held = _held()
    assert held[:2] == (incoming[0] + 12, incoming[1] + 3)
    assert held[2] > incoming[2]

    following = _guard("tests/test_example.py::test_after_the_leak")
    next(following)
    # Released in the setup of the next test, which then starts clean and
    # is not charged.
    assert _held() == incoming
    _passes(following)


@_requires_workspace_locks
def test_guard_releases_an_owner_kept_by_the_post_mortem_names(
    tmp_path: Path,
    collector_off: None,
) -> None:
    incoming = _held()
    guard = _guard()
    next(guard)
    # pytest keeps the exception of a failed test in these names until the
    # next test is called.
    error = _drop_owner(_repository(tmp_path / "repo"))
    sys.last_exc = error
    sys.last_value = error
    sys.last_type = type(error)
    sys.last_traceback = None
    del error

    with pytest.raises(pytest.fail.Exception, match="12 authority descriptor"):
        next(guard)
    gc.collect()
    assert _held() != incoming

    following = _guard("tests/test_example.py::test_after_the_failure")
    next(following)
    assert _held() == incoming
    assert not hasattr(sys, "last_exc")
    assert not hasattr(sys, "last_value")
    _passes(following)


@_requires_workspace_locks
def test_guard_does_not_charge_what_was_held_before_the_test(
    tmp_path: Path,
) -> None:
    with journals.workspace_mutation_lock(_repository(tmp_path / "repo")):
        guard = _guard()
        next(guard)
        with journals.workspace_mutation_lock(
            _repository(tmp_path / "nested")
        ):
            pass
        _passes(guard)


@_requires_workspace_locks
def test_guard_charges_an_owner_the_test_still_references_only_once(
    tmp_path: Path,
) -> None:
    incoming = _held()
    guard = _guard()
    next(guard)
    owner = journals.workspace_mutation_lock(_repository(tmp_path / "repo"))
    owner.__enter__()
    try:
        with pytest.raises(pytest.fail.Exception, match=_LEAKING_TEST):
            next(guard)

        # The collection of the next setup cannot release a referenced
        # owner; the tests after the leaking one are not charged with it.
        following = _guard("tests/test_example.py::test_after_the_leak")
        next(following)
        assert _held() != incoming
        _passes(following)
    finally:
        owner.__exit__(None, None, None)
    assert _held() == incoming


@_requires_workspace_locks
def test_guard_charges_a_lock_that_a_thread_of_the_test_still_holds(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "repo")
    incoming = _held()
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def hold() -> None:
        try:
            with journals.workspace_mutation_lock(repository):
                entered.set()
                release.wait(60)
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            errors.append(exc)
        finally:
            entered.set()

    guard = _guard()
    next(guard)
    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    try:
        assert entered.wait(60)
        assert errors == []
        # The descriptors are process-wide; the registration belongs to the
        # other thread and is not read.
        with pytest.raises(
            pytest.fail.Exception,
            match=r"12 authority descriptor\(s\) .* 3 lock descriptor\(s\) "
            r".* authority registered for \[\];",
        ):
            next(guard)
    finally:
        release.set()
        holder.join(60)
    assert not holder.is_alive()
    assert errors == []

    following = _guard("tests/test_example.py::test_after_the_join")
    next(following)
    assert _held() == incoming
    _passes(following)


def test_guard_surrounds_every_fixture_of_a_test(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set up first, the guard looks after every other fixture has ended."""

    names = list(request.fixturenames)

    assert _GUARD in names
    for later in ("_module_autouse_fixture", "tmp_path", "monkeypatch"):
        assert names.index(_GUARD) < names.index(later)


def test_only_the_shared_conftest_defines_the_guard() -> None:
    """A fixture of the same name in a test module would replace the guard."""

    shared = ROOT / "tests" / "conftest.py"
    definitions: list[str] = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            defined = (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == _GUARD
            )
            renamed = isinstance(node, ast.Call) and any(
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == _GUARD
                for keyword in node.keywords
            )
            if defined or renamed:
                definitions.append(path.relative_to(ROOT).as_posix())

    assert definitions == [shared.relative_to(ROOT).as_posix()]
