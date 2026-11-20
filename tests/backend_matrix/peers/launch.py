# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The launch interposer: a locked worker interpreter runs under the harness.

Two adapters start an interpreter of a locked worker environment
(``backends/<name>_worker/.venv``): the Kimi adapter through ``subprocess``
and the DeepSeek adapter through its supervisor function. Whatever a test
starts that way executes the dependencies of that environment in the test
job, so it runs under :mod:`.worker_harness`, which installs the network
guard before any of them is imported.

One decision, shared by every user. A launch is wrapped when its ``argv[0]``
is a locked worker interpreter: the path, made absolute *without resolving
the interpreter's own link*, lies inside a worker environment. (The link must
stay: a virtual environment's interpreter links to a base interpreter that
other environments share.) Every other launch is passed through untouched,
with the same arguments and the same callable. A locked interpreter whose
target is none of the three forms the harness runs (``-m``, ``-c``, a script)
is refused before a process starts.

Only the mode differs between users:

- the suite-wide fixture (:func:`suite_guard`, wired in ``tests/conftest.py``)
  asks for ``guard`` on every launch of every test;
- a worker driver of the matrix asks for ``redirect`` on the request launch of
  its worker module, so that the provider request is answered by the scripted
  provider, and for ``guard`` on every other launch;
- a test that starts a locked interpreter itself hands its argument list to
  :meth:`Interposer.run` or :meth:`Interposer.popen`, which apply the same
  rule.

Standard library only: ``tests/conftest.py`` imports this module for every
test of the suite.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from . import worker_harness

HARNESS = Path(os.path.abspath(worker_harness.__file__))
REPOSITORY = Path(os.path.abspath(__file__)).parents[3]
# Where a worker adapter starts its locked interpreter: (module, attribute).
# The DeepSeek adapter calls its supervisor function, looked up as a module
# global at each call; the Kimi adapter calls ``subprocess.Popen`` and
# ``subprocess.run`` through its own reference to the module.
SUPERVISOR_SEAMS: tuple[tuple[str, str], ...] = (
    ("crucible.backends.deepseek", "_run_supervised_command"),
)
SUBPROCESS_SEAMS: tuple[tuple[str, str], ...] = (("crucible.backends.kimi", "subprocess"),)
# Interpreter options that take no value and leave the target where it is.
_PLAIN_OPTIONS = frozenset("BEIOPRSbdqsuv")
_VALUE_OPTIONS = ("-W", "-X")


class LaunchRefused(RuntimeError):
    """A locked worker interpreter was asked to run what the harness cannot guard."""


def worker_environments(repository: Path | None = None) -> tuple[Path, ...]:
    """The environment of every worker project: ``backends/<name>_worker/.venv``."""

    backends = Path(os.path.realpath(repository or REPOSITORY)) / "backends"
    return tuple(
        sorted(
            Path(os.path.realpath(project / ".venv"))
            for project in backends.glob("*_worker")
            if (project / "pyproject.toml").is_file()
        )
    )


def is_locked_interpreter(
    argv0: str | os.PathLike[str], *, environments: Sequence[Path] | None = None
) -> bool:
    """Whether ``argv0`` is an interpreter inside a locked worker environment.

    The directory is resolved, the last component is not: the interpreter of
    a virtual environment is a link to a base interpreter outside it.
    """

    absolute = os.path.abspath(os.fspath(argv0))
    located = Path(os.path.realpath(os.path.dirname(absolute))) / os.path.basename(absolute)
    roots = worker_environments() if environments is None else environments
    return any(root in located.parents for root in roots)


def split(argv: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(interpreter options, target and its arguments)`` of a Python command line.

    The target starts at ``-m``, at ``-c`` or at a script path. Anything else
    (standard input, an interactive session, an option this list does not
    know) is refused: the harness could not run it as the interpreter would.
    """

    options: list[str] = []
    index = 1
    while index < len(argv):
        item = argv[index]
        if item in ("-m", "-c"):
            if index + 1 >= len(argv):
                raise LaunchRefused(f"{item} names no target: {list(argv)!r}")
            return tuple(options), tuple(argv[index:])
        if not item.startswith("-"):
            return tuple(options), tuple(argv[index:])
        if item[:2] in _VALUE_OPTIONS:
            if len(item) > 2:
                options.append(item)
            elif index + 1 < len(argv):
                options.extend(argv[index : index + 2])
                index += 1
            else:
                raise LaunchRefused(f"{item} has no value: {list(argv)!r}")
        elif len(item) > 1 and not item.startswith("--") and set(item[1:]) <= _PLAIN_OPTIONS:
            options.append(item)
        else:
            raise LaunchRefused(
                f"the harness cannot run a locked worker interpreter with {item!r} "
                f"(standard input, an interactive session or an unknown option): "
                f"{list(argv)!r}"
            )
        index += 1
    raise LaunchRefused(
        f"a locked worker interpreter without a target is an interactive session: "
        f"{list(argv)!r}"
    )


@dataclass(frozen=True)
class Redirect:
    """What a worker driver asks for on the request launch of its worker module."""

    module: str  # ``-m <module>`` with no argument is the request launch
    origin: str  # the route's production origin
    target: str  # the scripted provider on loopback


@dataclass(frozen=True)
class Journal:
    """What the harness wrote for one test: every launch and every violation."""

    directory: Path

    def _records(self, name: str) -> list[Mapping[str, Any]]:
        path = self.directory / name
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def launches(self) -> list[Mapping[str, Any]]:
        return self._records(worker_harness.LAUNCHES)

    def violations(self) -> list[str]:
        return [
            f"{record.get('source')} in worker process {record.get('pid')}: "
            f"{record.get('violation')}"
            for record in self._records(worker_harness.VIOLATIONS)
        ]

    def pids(self) -> tuple[int, ...]:
        return tuple(sorted({int(record["pid"]) for record in self.launches()}))


class Interposer:
    """The one decision at every launch seam, and the harness command line.

    ``journal`` is called for the journal of the test at the first launch
    that is wrapped, so a test that starts no worker costs nothing.
    """

    def __init__(
        self,
        journal: Callable[[], Journal],
        *,
        redirect: Redirect | None = None,
        environments: Sequence[Path] | None = None,
    ) -> None:
        self._open_journal = journal
        self._journal: Journal | None = None
        self._redirect = redirect
        self._environments = None if environments is None else tuple(environments)

    @property
    def journal(self) -> Journal | None:
        """The journal, once a launch was wrapped; ``None`` before."""

        return self._journal

    def violations(self) -> list[str]:
        return [] if self._journal is None else self._journal.violations()

    def decides(self, argv: Sequence[str]) -> bool:
        """Whether a launch of ``argv`` is wrapped: its ``argv[0]`` is a locked worker."""

        return bool(argv) and is_locked_interpreter(argv[0], environments=self._environments)

    def command(
        self, argv: Sequence[str], env: Mapping[str, str] | None
    ) -> tuple[tuple[str, ...], dict[str, str]] | None:
        """The harness command line and environment, or ``None``: pass through."""

        if not self.decides(argv):
            return None
        options, target = split(argv)
        if self._journal is None:
            self._journal = self._open_journal()
        settings = {"mode": "guard", "journal": str(self._journal.directory)}
        if self._redirect is not None and target == ("-m", self._redirect.module):
            settings.update(
                mode="redirect", origin=self._redirect.origin, target=self._redirect.target
            )
        environment = dict(os.environ if env is None else env)
        environment[worker_harness.SETTINGS_ENV] = json.dumps(settings, sort_keys=True)
        return (argv[0], *options, str(HARNESS), *target), environment

    # -- the supervisor seam (DeepSeek) -------------------------------------

    def supervised(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """``original(argv, *, env=..., ...)`` with the decision applied to ``argv``."""

        @functools.wraps(original)
        def run_supervised_command(argv: Sequence[str], *args: Any, **kwargs: Any) -> Any:
            if not self.decides(tuple(argv)):
                return original(argv, *args, **kwargs)
            if "env" not in kwargs:
                raise LaunchRefused(
                    "the supervisor seam no longer takes its environment by keyword; "
                    "the launch interposer cannot guard it"
                )
            wrapped = self.command(tuple(argv), kwargs["env"])
            assert wrapped is not None
            command, environment = wrapped
            return original(command, *args, **{**kwargs, "env": environment})

        run_supervised_command.matrix_interposer = self  # type: ignore[attr-defined]
        return run_supervised_command

    # -- the subprocess seam (Kimi, and tests that start a worker themselves) --

    def _subprocess(self, call: Callable[..., Any], args: Any, more: Any, named: Any) -> Any:
        if isinstance(args, (str, bytes, os.PathLike)) or named.get("shell"):
            return call(args, *more, **named)  # no argument list to decide on
        argv = tuple(os.fspath(item) for item in args)
        if named.get("executable") is not None and is_locked_interpreter(
            named["executable"], environments=self._environments
        ):
            raise LaunchRefused(
                "a locked worker interpreter given as executable= cannot be guarded; "
                "name it as argv[0]"
            )
        if not self.decides(argv):
            return call(args, *more, **named)
        if more:
            raise LaunchRefused(
                "a locked worker interpreter is started with positional arguments "
                "after the command; the launch interposer cannot place its environment"
            )
        wrapped = self.command(argv, named.get("env"))
        assert wrapped is not None
        command, environment = wrapped
        return call(list(command), **{**named, "env": environment})

    def run(self, args: Any, *more: Any, **named: Any) -> Any:
        """``subprocess.run`` under the decision; for a test's own launch too."""

        return self._subprocess(subprocess.run, args, more, named)

    def popen(self, args: Any, *more: Any, **named: Any) -> Any:
        """``subprocess.Popen`` under the decision; for a test's own launch too."""

        return self._subprocess(subprocess.Popen, args, more, named)


class SubprocessView:
    """``subprocess`` as one adapter module sees it: ``Popen`` and ``run`` decided.

    Every other name is the real module's. The real ``Popen`` and ``run`` are
    looked up at each call, so a test that replaced one of them still gets its
    own callable, with untouched arguments for every launch that is not a
    locked worker interpreter.
    """

    def __init__(self, real: Any, interposer: Interposer) -> None:
        self.matrix_real = real
        self.matrix_interposer = interposer

    def __getattr__(self, name: str) -> Any:
        return getattr(self.matrix_real, name)

    def Popen(self, args: Any, *more: Any, **named: Any) -> Any:  # noqa: N802
        return self.matrix_interposer._subprocess(self.matrix_real.Popen, args, more, named)

    def run(self, args: Any, *more: Any, **named: Any) -> Any:
        return self.matrix_interposer._subprocess(self.matrix_real.run, args, more, named)


@contextlib.contextmanager
def at_supervisor(module: Any, interposer: Interposer, name: str) -> Iterator[None]:
    """Interpose at a module's supervisor function for the block.

    An interposer that is already there is replaced, not stacked: a launch
    runs under one harness.
    """

    current = getattr(module, name)
    original = (
        current.__wrapped__
        if getattr(current, "matrix_interposer", None) is not None
        else current
    )
    setattr(module, name, interposer.supervised(original))
    try:
        yield
    finally:
        setattr(module, name, current)


@contextlib.contextmanager
def at_subprocess(module: Any, interposer: Interposer, name: str) -> Iterator[None]:
    """Interpose at ``subprocess`` as ``module`` sees it, for the block."""

    current = getattr(module, name)
    real = current.matrix_real if isinstance(current, SubprocessView) else current
    setattr(module, name, SubprocessView(real, interposer))
    try:
        yield
    finally:
        setattr(module, name, current)


def suite_guard(
    new_directory: Callable[[], Path], fail: Callable[[str], None]
) -> Iterator[Interposer]:
    """Fixture body: every locked worker a test starts runs in mode ``guard``.

    Interposes at the two launch seams for one test and yields the interposer,
    whose :meth:`Interposer.run` and :meth:`Interposer.popen` are the launcher
    for a test that starts a locked interpreter itself. At the end ``fail`` is
    called when a worker of the test reached beyond loopback, whether or not
    the worker or its adapter swallowed the exception.
    """

    interposer = Interposer(lambda: Journal(new_directory()))
    with contextlib.ExitStack() as seams:
        for module, name in SUPERVISOR_SEAMS:
            seams.enter_context(at_supervisor(importlib.import_module(module), interposer, name))
        for module, name in SUBPROCESS_SEAMS:
            seams.enter_context(at_subprocess(importlib.import_module(module), interposer, name))
        yield interposer
    violations = interposer.violations()
    if violations:
        fail(
            "a locked worker interpreter that the test started reached for the "
            "network (blocked by the matrix guard): " + "; ".join(violations)
        )
