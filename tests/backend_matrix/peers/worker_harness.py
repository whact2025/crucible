# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The worker harness: what a locked worker interpreter runs in front of its target.

A test never starts a locked worker interpreter on its target directly. The
launch interposer (:mod:`.launch`) puts this file in front of the target and
keeps the interpreter options of the original launch::

    <interpreter> <options> worker_harness.py -m <module> [arguments]
    <interpreter> <options> worker_harness.py -c <code> [arguments]
    <interpreter> <options> worker_harness.py <script> [arguments]

The harness then

1. installs the network guard of the matrix, before anything of the worker
   environment is imported: only loopback may be connected to or resolved;
2. in mode ``redirect`` also installs the origin redirect, so that a request
   for the route's production origin is answered by the scripted provider;
3. appends one record of the launch to the journal of the test that started
   it. In mode ``redirect`` that record is the mark that the redirect was in
   place before the target ran;
4. runs the target as ``__main__``, with ``sys.argv`` and the first entry of
   ``sys.path`` as the interpreter would have set them for the original
   command line.

A worker environment may also bundle a program of its own (the official
Codex SDK bundles the vendor's command-line runtime). Such a program opens its
sockets outside this interpreter, where the guard does not see them. So the
harness also refuses to start any program that lies inside the environment of
the interpreter: the attempt is a violation. A driver that needs the worker
to start a tool puts a scripted peer outside the environment in its place
(:mod:`.codex_worker`).

Every violation is appended to the journal as it happens, before the
exception is raised: a worker may swallow the exception, and a supervisor may
kill the worker. The test that started the worker reads the journal at its
end and fails on any entry.

Constraints, from the pins of the two worker adapters. The harness writes no
bytecode and nothing under a worker project, nothing to standard output and,
unless it cannot start at all, nothing to standard error. It takes its
settings from one environment variable and removes it, so the target sees
the environment and the argument list of the original launch.

Standard library only at import and free of package-relative imports: a
locked interpreter runs this file by path, in isolated mode. The redirect
(:mod:`..tap`) needs ``httpx``, which every worker environment that talks to
a provider through it has.
"""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import sys
import types
from typing import Any, Callable, Sequence

# The one variable the settings travel in: a JSON object with ``mode``,
# ``journal`` and, in mode ``redirect``, ``origin`` and ``target``.
SETTINGS_ENV = "CRUCIBLE_MATRIX_WORKER_HARNESS"
MODES = ("guard", "redirect")
# The two files of a journal directory.
LAUNCHES = "launches.jsonl"
VIOLATIONS = "violations.jsonl"
# A client that opens sockets below the ``socket`` module is not seen by the
# guard. This one is in a worker's lock; importing it is a violation.
UNGUARDED_EVENT_LOOPS = ("uvloop",)
FORMS = ("-m", "-c", "script")

_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HarnessError(Exception):
    """The harness was started in a way it cannot run a target under."""


class UnguardedProgram(Exception):
    """A worker tried to start a program of its own environment.

    Not an ``OSError``: a worker that catches a failed launch must not turn
    this one into an ordinary error of its own.
    """


def refuse_environment_programs(on_violation: Callable[[str], None]) -> None:
    """Make the start of a program inside this interpreter's environment a violation.

    The interpreter itself is a link to a base interpreter outside the
    environment and stays startable; whatever else the environment holds as
    an executable runs outside the network guard.
    """

    import shutil  # noqa: PLC0415 - nothing before the guard is installed
    import subprocess  # noqa: PLC0415

    # The environment as it is named and as it resolves: a file of it may be
    # a link into a package cache.
    roots = {os.path.abspath(sys.prefix) + os.sep, os.path.realpath(sys.prefix) + os.sep}
    interpreter = os.path.realpath(sys.executable)
    real = subprocess.Popen

    def of_the_environment(program: str, search: str | None) -> str | None:
        """The path of ``program`` when it is a program of this environment."""

        if os.sep not in program:
            program = shutil.which(program, path=search) or program
        named = os.path.abspath(program)
        resolved = os.path.realpath(named)
        inside = any(path.startswith(root) for path in (named, resolved) for root in roots)
        return named if inside and resolved != interpreter else None

    class Popen(real):  # type: ignore[misc,valid-type]
        def __init__(self, args: Any, *more: Any, **named: Any) -> None:
            program = named.get("executable")
            if program is None:
                program = args if isinstance(args, (str, bytes, os.PathLike)) else args[0]
            environment = named.get("env")
            search = (os.environ if environment is None else environment).get("PATH")
            located = None if named.get("shell") else of_the_environment(
                os.fsdecode(program), search
            )
            if located is not None:
                on_violation(
                    f"started {located}: a program of the worker environment opens "
                    "its sockets where the network guard does not see them"
                )
                raise UnguardedProgram(located)
            super().__init__(args, *more, **named)

    Popen.__name__ = real.__name__
    Popen.__qualname__ = real.__qualname__
    subprocess.Popen = Popen  # type: ignore[misc]


def append(directory: str, name: str, record: dict[str, Any]) -> None:
    """Append one record to a journal file; one write, so lines never interleave."""

    line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        os.path.join(directory, name), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
    )
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def split_target(arguments: Sequence[str]) -> tuple[str, str, list[str]]:
    """``(form, target, its arguments)`` of what follows the harness on the command line."""

    if not arguments:
        raise HarnessError("no target: an interactive session cannot run under the harness")
    head, rest = arguments[0], list(arguments[1:])
    if head in ("-m", "-c"):
        if not rest:
            raise HarnessError(f"{head} names no target")
        return head, rest[0], rest[1:]
    if head.startswith("-"):
        raise HarnessError(f"{head!r} is not a target the harness can run")
    return "script", head, rest


def _load(name: str, filename: str) -> Any:
    """Load a module of the matrix by path; the package itself is not importable here."""

    spec = importlib.util.spec_from_file_location(name, os.path.join(_PACKAGE, filename))
    if spec is None or spec.loader is None:
        raise HarnessError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _settings() -> dict[str, Any]:
    raw = os.environ.pop(SETTINGS_ENV, None)
    if raw is None:
        raise HarnessError(f"{SETTINGS_ENV} is not set: only the launch interposer starts me")
    try:
        settings = json.loads(raw)
    except ValueError as exc:
        raise HarnessError(f"{SETTINGS_ENV} is not JSON") from exc
    expected = {"mode", "journal"}
    if isinstance(settings, dict) and settings.get("mode") == "redirect":
        expected |= {"origin", "target"}
    if (
        not isinstance(settings, dict)
        or set(settings) != expected
        or settings["mode"] not in MODES
        or not all(isinstance(value, str) and value for value in settings.values())
    ):
        raise HarnessError(f"{SETTINGS_ENV} does not hold the settings of a launch")
    if not os.path.isdir(settings["journal"]):
        raise HarnessError("the journal directory of the launch does not exist")
    return settings


def _run(form: str, target: str, arguments: list[str]) -> None:
    """Run the target as ``__main__``, as the interpreter would have."""

    # With a safe path (``-P``, implied by ``-I``) the interpreter prepends
    # nothing; otherwise it prepended the directory of this file, where the
    # original command line would have put its own first entry.
    prepended = not sys.flags.safe_path
    if form == "-m":
        sys.argv = ["-m", *arguments]  # runpy puts the module's file first
        if prepended:
            sys.path[0] = os.getcwd()
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    elif form == "-c":
        sys.argv = ["-c", *arguments]
        if prepended:
            sys.path[0] = ""
        main = types.ModuleType("__main__")
        main.__dict__["__builtins__"] = __builtins__
        sys.modules["__main__"] = main
        exec(compile(target, "<string>", "exec"), main.__dict__)
    else:
        sys.argv = [target, *arguments]
        if prepended:
            sys.path[0] = os.path.dirname(os.path.realpath(target))
        runpy.run_path(target, run_name="__main__")


def main(arguments: Sequence[str]) -> None:
    # The fingerprint of a worker environment covers its bytecode files.
    launched_without_bytecode = bool(sys.flags.dont_write_bytecode)
    sys.dont_write_bytecode = True
    try:
        settings = _settings()
        form, target, rest = split_target(arguments)
    except HarnessError as exc:
        # Nothing ran and nothing may run unguarded.
        raise SystemExit(f"worker harness: {exc}") from None
    journal = settings["journal"]

    def violation(source: str) -> Callable[[str], None]:
        def record(entry: str) -> None:
            append(
                journal,
                VIOLATIONS,
                {"pid": os.getpid(), "source": source, "violation": entry},
            )

        return record

    try:
        _load("crucible_matrix_netguard", "netguard.py").install(violation("network guard"))
        refuse_environment_programs(violation("worker harness"))
        redirect = None
        if settings["mode"] == "redirect":
            redirect = {"origin": settings["origin"], "target": settings["target"]}
            _load("crucible_matrix_tap", "tap.py").install(
                redirect["origin"],
                redirect["target"],
                on_violation=violation("origin redirect"),
            )
    except BaseException as exc:  # noqa: BLE001 - nothing may run unguarded
        violation("worker harness")(f"could not install: {type(exc).__name__}: {exc}")
        raise SystemExit(f"worker harness: could not install ({type(exc).__name__})") from None
    append(
        journal,
        LAUNCHES,
        {
            "pid": os.getpid(),
            "parent": os.getppid(),
            "mode": settings["mode"],
            "form": form,
            # The code of a ``-c`` launch is the test's own and may be long.
            "target": "<code>" if form == "-c" else target,
            "arguments": rest,
            "executable": sys.executable,
            "isolated": bool(sys.flags.isolated),
            "no_bytecode": launched_without_bytecode,
            "guard": True,
            "redirect": redirect,
        },
    )
    try:
        _run(form, target, rest)
    finally:
        for name in UNGUARDED_EVENT_LOOPS:
            if name in sys.modules:
                violation("worker harness")(
                    f"{name} was imported: an event loop that opens sockets below "
                    "the socket module is not seen by the network guard"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
