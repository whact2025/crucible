# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fixtures and hooks of the backend conformance matrix.

- the network guard is on for every test of this package, and a recorded
  violation fails the test at teardown;
- every test gets the scripted route environment, the matrix rate card and a
  private home directory;
- an item with a skip or xfail marker is refused at collection;
- an expected failure is a failure unless the item is the cell of a ledger
  entry, so a self-test cannot pass by reaching the comparator's verdict;
- the outcome of every cell is recorded, so that a matrix module can require
  that the seam canary of each of its columns passed in this process;
- when the run writes a JUnit file, the cell report of this process is
  written beside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from . import cells, ledger, runner

_PACKAGE = Path(__file__).resolve().parent
_KNOWN = ledger.load()


@pytest.fixture(autouse=True)
def _matrix_network_guard() -> Iterator[None]:
    yield from runner.network_guard()


@pytest.fixture(autouse=True)
def _matrix_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner.isolate_environment(monkeypatch, tmp_path)


@pytest.fixture(scope="session")
def matrix_preflight() -> dict[str, str | None]:
    """Each driver's preflight, once: a column that cannot run here is an error."""

    return runner.preflight_all()


def pytest_collection_modifyitems(config: pytest.Config, items: list[Any]) -> None:
    cells.refuse_skip_markers(items, _PACKAGE)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Iterator[None]:
    outcome: Any = yield
    runner.judge_report(item, outcome.get_result(), _KNOWN)


def pytest_sessionfinish(session: pytest.Session) -> None:
    runner.write_cell_report(session.config)
