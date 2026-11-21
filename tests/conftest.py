# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for Crucible tests.

This module contains only fixtures shared across the current test suite:

- ``_contain_process_trace_state`` (autouse): restores an exact incoming
  interpreter tracer and fails a test that leaks trace-state drift;
- ``locked_worker_launch`` (autouse): every interpreter of a locked worker
  environment (``backends/<name>_worker/.venv``) that a test starts, through
  any worker adapter, runs under the worker harness of the backend matrix,
  which installs the network guard before the worker's dependencies are
  imported. A worker that reaches beyond loopback fails the test that started
  it. A test that starts such an interpreter itself uses the fixture's
  ``run`` or ``popen`` in place of ``subprocess``;
- ``_contain_worker_adapter_network`` (autouse): the same network guard in
  the test process, for every test module whose file name names a worker
  adapter (``kimi``, ``deepseek``, ``codex``);
- ``require_kimi_worker``: the interpreter of the locked Kimi worker
  environment. A test that needs it takes this fixture instead of a skip
  marker: a missing environment is an error that names the install command.

"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

from .backend_matrix import netguard
from .backend_matrix.peers import launch

# The adapters that run a locked worker environment. A test module is named
# after the adapter it tests, so the rule is the file name, not a list.
_WORKER_ADAPTERS = ("kimi", "deepseek", "codex")
# The command of the CI step, in the shape of the DeepSeek worker's. Copy mode
# is a convention here: the Kimi pin takes any regular file and does not
# require it.
_KIMI_WORKER_INSTALL = (
    "uv sync --frozen --reinstall --link-mode copy --project backends/kimi_worker"
)
_MATRIX_PACKAGE = Path(netguard.__file__).resolve().parent


@pytest.fixture(autouse=True)
def _contain_process_trace_state():
    """Restore and reject a test-owned ``sys.settrace`` leak."""

    incoming_trace = sys.gettrace()
    try:
        yield
    finally:
        outgoing_trace = sys.gettrace()
        if outgoing_trace is not incoming_trace:
            if incoming_trace is None:
                sys.settrace(None)
            else:
                sys.settrace(incoming_trace)
            pytest.fail(
                "test leaked sys.settrace state; restore the incoming tracer",
                pytrace=False,
            )


def names_a_worker_adapter(path: Path) -> bool:
    """The file-name rule: a test module named after a worker adapter.

    The modules of the backend matrix are not selected here: its own fixture
    installs the guard for every test of the package.
    """

    return _MATRIX_PACKAGE not in path.resolve().parents and any(
        adapter in path.name for adapter in _WORKER_ADAPTERS
    )


@pytest.fixture(autouse=True)
def locked_worker_launch(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[launch.Interposer]:
    """Run every locked worker interpreter of the test under the network guard."""

    yield from launch.suite_guard(
        lambda: tmp_path_factory.mktemp("worker-harness"),
        lambda reason: pytest.fail(reason, pytrace=False),
    )


@pytest.fixture(autouse=True)
def _contain_worker_adapter_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail a worker-adapter test that reaches beyond loopback in this process."""

    if names_a_worker_adapter(Path(str(request.node.path))):
        yield from netguard.watch(lambda reason: pytest.fail(reason, pytrace=False))
    else:
        yield


@pytest.fixture()
def require_kimi_worker() -> Path:
    """The interpreter of the locked Kimi worker environment, or an error.

    Never a skip: a test that needs the environment and does not find it did
    not run, and the suite must say so.
    """

    return launch.require_environment(
        "kimi", _KIMI_WORKER_INSTALL, lambda reason: pytest.fail(reason, pytrace=False)
    )
