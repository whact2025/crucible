# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The worker harness, the launch interposer and the two suite-wide fixtures.

No worker environment of the repository is needed here. The harness is run by
the test interpreter inside a *stand-in* worker environment, built per test
module under a temporary directory: a virtual environment whose interpreter
is a link to the test interpreter's base, as a locked environment's is, with
the test environment's packages on its path and a small worker module of its
own. The interposer is pointed at it through ``environments=``.

Shown here, before any worker environment is installed in CI:

- the harness runs a module, a ``-c`` script and a script path exactly as the
  interpreter would have: the same standard output, standard error, exit
  status, ``sys.argv``, first ``sys.path`` entry and standard input, with no
  setting of its own left in the environment;
- a worker that connects or resolves beyond loopback is recorded, whatever it
  does with the exception, and the test that started it fails;
- only an interpreter inside a worker environment is wrapped. Any other
  executable goes through either launch seam untouched: the same arguments,
  the same callable;
- a target the harness cannot run is refused before a process starts;
- in mode ``redirect`` a request for the route's production origin is
  answered on loopback inside the worker, and in mode ``guard`` it is a
  violation.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import json
import os
import subprocess
import sys
import sysconfig
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import pytest

from crucible.backends import deepseek as deepseek_adapter
from crucible.backends import kimi as kimi_adapter

from .. import conftest as suite_conftest
from . import columns as matrix_columns
from . import ledger, netguard, runner
from .cells import Cell
from .drivers import MatrixSetupError
from .drivers import deepseek as deepseek_driver
from .peers import launch, worker_harness

WORKER_MODULE = "matrix_standin_worker"
SUPERVISOR = "_run_supervised_command"
PRODUCTION = "https://provider.invalid"
UNREACHABLE = ["192.0.2.1", 9]

# What the stand-in worker does: it reads a JSON object of commands from its
# standard input, as a worker reads its request, and prints one JSON report.
_WORKER = '''
import json
import os
import signal
import socket
import sys

from matrix_standin_worker import helper


def main():
    text = sys.stdin.read()
    commands = json.loads(text) if text.strip() else {}
    report = {
        "name": __name__,
        "argv": sys.argv,
        "path0": sys.path[0] if sys.path else None,
        "stdin": text,
        "harness_variables": sorted(name for name in os.environ if "HARNESS" in name),
        "isolated": bool(sys.flags.isolated),
        "launched_without_bytecode": bool(sys.flags.dont_write_bytecode),
        "helper": helper.VALUE,
    }
    if "connect" in commands:
        try:
            socket.create_connection(tuple(commands["connect"]), timeout=1).close()
            report["connect"] = "connected"
        except BaseException as exc:  # a worker reports errors by type
            report["connect"] = type(exc).__name__
    if "resolve" in commands:
        try:
            socket.getaddrinfo(commands["resolve"], 443)
            report["resolve"] = "resolved"
        except BaseException as exc:
            report["resolve"] = type(exc).__name__
    if "fetch" in commands:
        import httpx

        try:
            with httpx.Client(trust_env=False) as client:
                response = client.get(commands["fetch"])
            report["fetch"] = [response.status_code, response.text]
        except BaseException as exc:
            report["fetch"] = type(exc).__name__
    if "import" in commands:
        __import__(commands["import"])
    if "stderr" in commands:
        sys.stderr.write(commands["stderr"])
    print(json.dumps(report, sort_keys=True))
    sys.stdout.flush()
    if commands.get("kill"):
        os.kill(os.getpid(), signal.SIGKILL)
    raise SystemExit(commands.get("exit", 0))


main()
'''


@dataclass(frozen=True)
class StandIn:
    """A stand-in worker environment and what can be run in it."""

    project: Path
    script: Path

    @property
    def environment(self) -> Path:
        return self.project / ".venv"

    @property
    def python(self) -> Path:
        return self.environment / "bin" / "python"

    @property
    def code(self) -> str:
        return self.script.read_text(encoding="utf-8")

    def command(self, form: str, options: Sequence[str] = (), *arguments: str) -> list[str]:
        target = {
            "-m": ["-m", WORKER_MODULE],
            "-c": ["-c", self.code],
            "script": [str(self.script)],
        }[form]
        return [str(self.python), *options, *target, *arguments]


def _build_standin(root: Path) -> StandIn:
    project = root / "backends" / "standin_worker"
    environment = project / ".venv"
    (environment / "bin").mkdir(parents=True)
    base = Path(sys.executable).resolve()
    # As in a locked environment: the interpreter is a link to a base
    # interpreter outside the environment.
    (environment / "bin" / "python").symlink_to(base)
    (environment / "pyvenv.cfg").write_text(
        f"home = {base.parent}\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = environment / "lib" / version / "site-packages"
    package = site / WORKER_MODULE
    package.mkdir(parents=True)
    # The redirect needs httpx, which the test environment has.
    (site / "test-environment.pth").write_text(
        sysconfig.get_paths()["purelib"] + "\n", encoding="utf-8"
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helper.py").write_text("VALUE = 'imported'\n", encoding="utf-8")
    (package / "__main__.py").write_text(_WORKER.lstrip(), encoding="utf-8")
    # A module by the name of an event loop the guard cannot see.
    (site / "uvloop.py").write_text("", encoding="utf-8")
    script = root / "scripts" / "worker_script.py"
    script.parent.mkdir()
    script.write_text(_WORKER.lstrip(), encoding="utf-8")
    return StandIn(project, script)


@pytest.fixture(scope="module")
def standin(tmp_path_factory: pytest.TempPathFactory) -> StandIn:
    return _build_standin(tmp_path_factory.mktemp("standin-repository"))


def _interposer(
    standin: StandIn, directory: Path, redirect: launch.Redirect | None = None
) -> launch.Interposer:
    def journal() -> launch.Journal:
        directory.mkdir()
        return launch.Journal(directory)

    return launch.Interposer(
        journal, redirect=redirect, environments=(Path(os.path.realpath(standin.environment)),)
    )


def _report(completed: subprocess.CompletedProcess[str]) -> Mapping[str, Any]:
    assert completed.stdout.count("\n") == 1, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def _run(
    interposer: launch.Interposer, command: Sequence[str], commands: Mapping[str, Any], **more: Any
) -> subprocess.CompletedProcess[str]:
    return interposer.run(
        list(command),
        input=json.dumps(commands),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={"PYTHONNOUSERSITE": "1"},
        **more,
    )


def _supervised(command: Sequence[str], commands: Mapping[str, Any], cwd: Path) -> Any:
    """A launch through the DeepSeek adapter's own supervisor seam."""

    executable = Path(command[0]).resolve(strict=True)
    return deepseek_adapter._run_supervised_command(
        tuple(command),
        stdin_bytes=json.dumps(commands).encode("utf-8"),
        cwd=cwd.resolve(strict=True),
        env={"PYTHONNOUSERSITE": "1"},
        timeout_seconds=60.0,
        max_stdout_bytes=1024 * 1024,
        max_stderr_bytes=64 * 1024,
        executable_path=executable,
        executable_sha256="sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest(),
    )


# -- the decision ---------------------------------------------------------------


def test_only_an_interpreter_inside_a_worker_environment_is_locked(
    standin: StandIn, tmp_path: Path
) -> None:
    environments = (Path(os.path.realpath(standin.environment)),)

    def locked(path: Path | str) -> bool:
        return launch.is_locked_interpreter(path, environments=environments)

    assert locked(standin.python)
    # The interpreter's own link is not followed: its target is the base
    # interpreter, which every environment of that Python version shares.
    assert standin.python.is_symlink() and not locked(standin.python.resolve())
    assert not locked(sys.executable) and not locked(Path(sys.executable).resolve())
    # However the directory is spelled: through a link, or not normalised.
    alias = tmp_path / "alias"
    alias.symlink_to(standin.project, target_is_directory=True)
    assert locked(alias / ".venv" / "bin" / "python")
    assert locked(standin.environment / "lib" / ".." / "bin" / "python")
    assert not locked(standin.project / "python") and not locked(standin.script)

    # The repository's own worker projects, whether or not they are installed.
    names = {path.parent.name for path in launch.worker_environments()}
    assert names == {
        path.name
        for path in (launch.REPOSITORY / "backends").glob("*_worker")
        if (path / "pyproject.toml").is_file()
    }
    assert {"deepseek_worker", "kimi_worker"} <= names
    for environment in launch.worker_environments():
        assert launch.is_locked_interpreter(environment / "bin" / "python")
    # The test interpreter belongs to no worker environment.
    assert not launch.is_locked_interpreter(sys.executable)
    assert not launch.is_locked_interpreter(Path(sys.executable).resolve())


def test_a_python_command_line_is_split_at_its_target_or_refused() -> None:
    python = "/x/.venv/bin/python"

    assert launch.split([python, "-B", "-I", "-m", "worker", "--probe"]) == (
        ("-B", "-I"),
        ("-m", "worker", "--probe"),
    )
    assert launch.split([python, "-c", "print(1)", "-m", "x"]) == (
        (),
        ("-c", "print(1)", "-m", "x"),
    )
    assert launch.split([python, "-BI", "/s.py", "-c"]) == (("-BI",), ("/s.py", "-c"))
    assert launch.split([python, "-X", "utf8", "-Wd", "-u", "s.py"]) == (
        ("-X", "utf8", "-Wd", "-u"),
        ("s.py",),
    )
    for refused in (
        [python],
        [python, "-B", "-I"],
        [python, "-"],
        [python, "-i", "s.py"],
        [python, "-m"],
        [python, "-I", "-c"],
        [python, "--version"],
        [python, "-X"],
        [python, "-Bc", "print(1)"],
    ):
        with pytest.raises(launch.LaunchRefused):
            launch.split(refused)


# -- the harness runs a target as the interpreter would -------------------------


@pytest.mark.parametrize("options", [(), ("-B", "-I")], ids=["plain", "isolated"])
@pytest.mark.parametrize("form", worker_harness.FORMS)
def test_the_harness_runs_a_target_exactly_as_the_interpreter_would(
    standin: StandIn, tmp_path: Path, form: str, options: tuple[str, ...]
) -> None:
    interposer = _interposer(standin, tmp_path / "journal")
    command = standin.command(form, options, "first", "--second")
    commands = {"stderr": "the worker's own error text\n", "exit": 7}
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    # The interpreter of the stand-in on its target, with no harness.
    direct = subprocess.run(
        command,
        input=json.dumps(commands),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={"PYTHONNOUSERSITE": "1"},
        cwd=cwd,
    )
    harnessed = _run(interposer, command, commands, cwd=cwd)

    assert (harnessed.returncode, harnessed.stderr) == (7, "the worker's own error text\n")
    assert harnessed.stdout == direct.stdout and harnessed.stderr == direct.stderr
    assert harnessed.returncode == direct.returncode
    report = _report(harnessed)
    assert report["name"] == "__main__" and report["helper"] == "imported"
    assert report["argv"][1:] == ["first", "--second"]
    assert report["stdin"] == json.dumps(commands)
    # No setting of the harness is left for the target to see.
    assert report["harness_variables"] == []
    assert report["isolated"] is bool(options)
    prepended = {"-m": str(cwd), "-c": "", "script": str(standin.script.parent)}
    if options:
        # Isolated: nothing is prepended, the first entry is the interpreter's own.
        assert report["path0"] not in prepended.values()
    else:
        assert report["path0"] == prepended[form]
    if form == "-m":
        module_file = Path(report["argv"][0])
        assert (module_file.parent.name, module_file.name) == (WORKER_MODULE, "__main__.py")
        assert standin.environment in module_file.parents
    else:
        assert report["argv"][0] == {"-c": "-c", "script": str(standin.script)}[form]

    # The launch was journalled before the target ran.
    assert interposer.journal is not None
    (record,) = interposer.journal.launches()
    assert record["mode"] == "guard" and record["guard"] is True and record["redirect"] is None
    assert record["form"] == form and record["arguments"] == ["first", "--second"]
    assert record["target"] == {
        "-m": WORKER_MODULE,
        "-c": "<code>",
        "script": str(standin.script),
    }[form]
    # The interpreter ran under the path it was started with, link and all.
    assert record["executable"] == str(standin.python)
    assert record["pid"] != os.getpid() and record["parent"] == os.getpid()
    assert record["isolated"] is bool(options) and record["no_bytecode"] is bool(options)
    assert interposer.violations() == []


def test_the_harness_writes_no_bytecode_into_the_worker_environment(tmp_path: Path) -> None:
    # An environment of its own: no other test has run an interpreter in it.
    standin = _build_standin(tmp_path / "repository")

    def caches() -> list[Path]:
        return sorted(standin.project.rglob("__pycache__"))

    assert caches() == []
    interposer = _interposer(standin, tmp_path / "journal")

    # No ``-B`` on the command line and nothing in the environment: left to
    # itself the interpreter would cache the worker's helper module.
    completed = interposer.run(
        standin.command("-m"),
        input="{}",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={},
    )

    assert _report(completed)["launched_without_bytecode"] is False
    assert caches() == []


# -- the guard inside the worker ------------------------------------------------


def test_a_worker_that_reaches_beyond_loopback_is_recorded_whatever_it_does(
    standin: StandIn, tmp_path: Path
) -> None:
    interposer = _interposer(standin, tmp_path / "journal")
    reach = {"connect": UNREACHABLE, "resolve": "provider.invalid"}

    for form in worker_harness.FORMS:
        completed = _run(interposer, standin.command(form, ("-B", "-I")), reach)
        # The worker swallowed both exceptions and ended cleanly.
        report = _report(completed)
        assert (completed.returncode, completed.stderr) == (0, "")
        assert report["connect"] == report["resolve"] == "NetworkGuardViolation"

    violations = interposer.violations()
    assert len(violations) == 6
    assert sum("create_connection -> ('192.0.2.1', 9)" in line for line in violations) == 3
    assert sum("getaddrinfo -> 'provider.invalid'" in line for line in violations) == 3
    assert all(line.startswith("network guard in worker process ") for line in violations)

    # A worker that is killed leaves the record too: it is written at once.
    killed = _interposer(standin, tmp_path / "killed")
    completed = _run(killed, standin.command("-m"), {"resolve": "provider.invalid", "kill": True})
    assert completed.returncode == -9
    assert len(killed.violations()) == 1

    # Loopback passes inside the worker.
    with _echo_server() as (target, _seen):
        quiet = _interposer(standin, tmp_path / "quiet")
        host, port = target.removeprefix("http://").split(":")
        completed = _run(quiet, standin.command("-m"), {"connect": [host, int(port)]})
    assert _report(completed)["connect"] == "connected" and quiet.violations() == []


def test_an_event_loop_the_guard_cannot_see_is_a_violation(
    standin: StandIn, tmp_path: Path
) -> None:
    interposer = _interposer(standin, tmp_path / "journal")

    completed = _run(interposer, standin.command("-m", ("-B", "-I")), {"import": "uvloop"})

    assert completed.returncode == 0
    (violation,) = interposer.violations()
    assert "uvloop was imported" in violation and violation.startswith("worker harness")
    assert worker_harness.UNGUARDED_EVENT_LOOPS == ("uvloop",)


def test_a_violation_in_a_worker_fails_the_test_that_started_it(
    standin: StandIn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stand-in is the only worker environment for this test, so that the
    # fixture body below is the suite's own, unchanged.
    environments = (Path(os.path.realpath(standin.environment)),)
    monkeypatch.setattr(launch, "worker_environments", lambda repository=None: environments)
    failures: list[str] = []
    directories = iter([tmp_path / "first", tmp_path / "second"])

    def new_directory() -> Path:
        directory = next(directories)
        directory.mkdir()
        return directory

    # A test that starts a worker which stays on loopback: nothing to report.
    guard = launch.suite_guard(new_directory, failures.append)
    next(guard)
    completed = _supervised(standin.command("-m", ("-B", "-I"), "--probe"), {}, tmp_path)
    assert completed.returncode == 0 and completed.stderr == b""
    assert list(guard) == [] and failures == []

    # One whose worker reaches out, through the adapter's own launch seam. The
    # worker swallows the exception and the launch looks clean.
    guard = launch.suite_guard(new_directory, failures.append)
    next(guard)
    completed = _supervised(
        standin.command("-m", ("-B", "-I")), {"resolve": "provider.invalid"}, tmp_path
    )
    assert completed.returncode == 0 and completed.stderr == b""
    assert json.loads(completed.stdout)["resolve"] == "NetworkGuardViolation"
    assert failures == []
    assert list(guard) == []
    (failure,) = failures
    assert "a locked worker interpreter that the test started reached for the network" in failure
    assert "getaddrinfo -> 'provider.invalid'" in failure


_INNER_CONFTEST = """
from crucible.tests.conftest import (  # noqa: F401 - the suite's own fixtures
    _contain_worker_adapter_network,
    locked_worker_launch,
)
"""
_INNER_ADAPTER_MODULE = """
import socket

from crucible.tests.backend_matrix import netguard


def test_a_blocked_call_the_code_under_test_swallows():
    assert netguard.installed()
    try:
        socket.getaddrinfo("provider.invalid", 443)
    except Exception:
        pass


def test_loopback_passes():
    assert netguard.installed()
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    socket.create_connection(server.getsockname(), timeout=5).close()
    server.close()
"""
_INNER_OTHER_MODULE = """
from crucible.backends import deepseek, kimi
from crucible.tests.backend_matrix import netguard
from crucible.tests.backend_matrix.peers import launch


def test_a_module_that_names_no_worker_adapter_is_not_guarded_in_process():
    assert not netguard.installed()


def test_the_two_launch_seams_are_interposed_for_every_test(locked_worker_launch):
    assert deepseek._run_supervised_command.matrix_interposer is locked_worker_launch
    assert isinstance(kimi.subprocess, launch.SubprocessView)
    assert kimi.subprocess.matrix_interposer is locked_worker_launch
"""


def test_the_suite_fixtures_fail_a_test_by_the_file_name_rule(tmp_path: Path) -> None:
    root = tmp_path / "inner"
    root.mkdir()
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    for name, source in (
        ("conftest.py", _INNER_CONFTEST),
        ("test_inner_deepseek_module.py", _INNER_ADAPTER_MODULE),
        ("test_inner_kimi_module.py", _INNER_ADAPTER_MODULE),
        ("test_inner_other_module.py", _INNER_OTHER_MODULE),
    ):
        (root / name).write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")

    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rE"],
        cwd=root,
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )

    assert run.returncode == 1, run.stdout + run.stderr
    assert "6 passed, 2 errors" in run.stdout
    assert run.stdout.count("blocked by the matrix guard") >= 2
    summary = run.stdout.split("short test summary")[-1]
    for module in ("test_inner_deepseek_module", "test_inner_kimi_module"):
        assert f"ERROR {module}.py::test_a_blocked_call_the_code_under_test_swallows" in summary
    assert "test_loopback_passes" not in summary and "test_inner_other_module" not in summary


def test_the_file_name_rule_selects_worker_adapter_modules_outside_the_matrix() -> None:
    tests = Path(suite_conftest.__file__).resolve().parent
    package = Path(__file__).resolve().parent

    assert suite_conftest.names_a_worker_adapter(tests / "test_backends_kimi_k3.py")
    assert suite_conftest.names_a_worker_adapter(tests / "test_workflows_bind_deepseek.py")
    assert not suite_conftest.names_a_worker_adapter(tests / "test_backends_claude.py")
    # The matrix guards every test of its own package with its own fixture.
    assert not suite_conftest.names_a_worker_adapter(package / "test_matrix_deepseek.py")
    assert netguard.installed()
    # By the rule, not by a list: whatever the suite holds today is selected.
    selected = [
        path
        for path in tests.rglob("test_*.py")
        if ".venv" not in path.parts and suite_conftest.names_a_worker_adapter(path)
    ]
    assert selected and all(package not in path.parents for path in selected)
    assert {name for path in selected for name in ("kimi", "deepseek") if name in path.name} == {
        "kimi",
        "deepseek",
    }


# -- every other executable passes through untouched -----------------------------


def test_any_other_executable_passes_through_either_seam_untouched(
    standin: StandIn, tmp_path: Path, locked_worker_launch: launch.Interposer
) -> None:
    interposer = _interposer(standin, tmp_path / "journal")
    calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def original(argv: Any, *args: Any, **kwargs: Any) -> str:
        calls.append((argv, args, kwargs))
        return "the original's own result"

    # The supervisor seam: the very same argument objects reach the original.
    argv = (str(Path(sys.executable).resolve()), "-I", str(standin.script), "argument")
    environment = {"PYTHONNOUSERSITE": "1"}
    wrapped = interposer.supervised(original)
    assert wrapped(argv, stdin_bytes=b"", env=environment) == "the original's own result"
    ((seen_argv, seen_args, seen_kwargs),) = calls
    assert seen_argv is argv and seen_args == () and seen_kwargs["env"] is environment
    assert set(seen_kwargs) == {"stdin_bytes", "env"}
    assert getattr(wrapped, "__wrapped__") is original  # noqa: B009 - set by functools.wraps

    # The subprocess seam: the callable of the moment, with the same arguments.
    calls.clear()
    view = launch.SubprocessView(subprocess, interposer)
    command = [sys.executable, "-c", "print('passed through')"]
    assert view.run(command, capture_output=True, text=True).stdout == "passed through\n"
    with view.Popen(command, stdout=subprocess.PIPE, text=True) as process:
        assert process.communicate()[0] == "passed through\n"
    assert view.PIPE is subprocess.PIPE and view.TimeoutExpired is subprocess.TimeoutExpired
    replaced = pytest.MonkeyPatch()
    try:
        replaced.setattr(subprocess, "run", original)
        replaced.setattr(subprocess, "Popen", original)
        assert view.run(command, check=False) == "the original's own result"
        assert view.Popen(command, bufsize=0) == "the original's own result"
        # A shell line and a string are no argument list to decide on.
        assert view.run(f"{standin.python} -m x", shell=True) == "the original's own result"
    finally:
        replaced.undo()
    assert [call[0] for call in calls] == [command, command, f"{standin.python} -m x"]
    assert calls[0][0] is command and calls[0][2] == {"check": False}
    assert calls[1][2] == {"bufsize": 0}

    # Nothing was journalled: no launch was a locked worker interpreter.
    assert interposer.journal is None and interposer.violations() == []

    # The suite's own fixture is in place for this very test, at both seams,
    # and lets the test interpreter through the real supervisor untouched.
    assert deepseek_adapter._run_supervised_command.matrix_interposer is locked_worker_launch
    assert isinstance(kimi_adapter.subprocess, launch.SubprocessView)
    plain = tmp_path / "plain_target.py"
    plain.write_text("import sys\nprint(sys.argv[1:])\nsys.exit(3)\n", encoding="utf-8")
    completed = _supervised(
        [str(Path(sys.executable).resolve()), "-I", str(plain), "kept"], {}, tmp_path
    )
    assert (completed.returncode, completed.stdout) == (3, b"['kept']\n")
    kimi_run = kimi_adapter.subprocess.run(command, capture_output=True, text=True, check=False)
    assert kimi_run.stdout == "passed through\n"
    assert locked_worker_launch.journal is None


def test_the_seams_are_restored_and_an_interposer_is_replaced_not_stacked(
    standin: StandIn, tmp_path: Path
) -> None:
    before_supervisor = deepseek_adapter._run_supervised_command
    before_subprocess = kimi_adapter.subprocess
    interposer = _interposer(standin, tmp_path / "journal")

    with (
        launch.at_supervisor(deepseek_adapter, interposer, SUPERVISOR),
        launch.at_subprocess(kimi_adapter, interposer, "subprocess"),
    ):
        current = deepseek_adapter._run_supervised_command
        assert current.matrix_interposer is interposer
        # One harness per launch: the suite's wrapper is not underneath.
        assert getattr(current.__wrapped__, "matrix_interposer", None) is None
        assert current.__wrapped__ is before_supervisor.__wrapped__
        assert kimi_adapter.subprocess.matrix_interposer is interposer
        assert kimi_adapter.subprocess.matrix_real is subprocess
        completed = _supervised(standin.command("-m", ("-B", "-I"), "--probe"), {}, tmp_path)
        assert completed.returncode == 0

    assert deepseek_adapter._run_supervised_command is before_supervisor
    assert kimi_adapter.subprocess is before_subprocess
    assert interposer.journal is not None and len(interposer.journal.launches()) == 1
    # The seams the suite fixture interposes at are the two the adapters have.
    assert ((deepseek_adapter.__name__, SUPERVISOR),) == launch.SUPERVISOR_SEAMS
    assert ((kimi_adapter.__name__, "subprocess"),) == launch.SUBPROCESS_SEAMS
    assert kimi_adapter.subprocess.matrix_real is subprocess


# -- a locked interpreter is wrapped at both seams --------------------------------


def test_a_locked_interpreter_is_wrapped_at_the_subprocess_seam(
    standin: StandIn, tmp_path: Path
) -> None:
    interposer = _interposer(standin, tmp_path / "journal")
    probe = standin.command("-m", ("-B", "-I"), "--probe")
    request = standin.command("-m", ("-B", "-I"))

    with launch.at_subprocess(kimi_adapter, interposer, "subprocess"):
        # The two launch shapes of the adapter that uses subprocess.
        completed = kimi_adapter.subprocess.run(
            probe, check=False, capture_output=True, env={"PYTHONNOUSERSITE": "1"}, timeout=60
        )
        process = kimi_adapter.subprocess.Popen(
            request,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PYTHONNOUSERSITE": "1"},
            start_new_session=True,
            bufsize=0,
        )
        stdout, stderr = process.communicate(b'{"resolve": "provider.invalid"}', timeout=60)

    assert completed.returncode == 0 and completed.stderr == b""
    assert json.loads(completed.stdout)["argv"][1:] == ["--probe"]
    assert process.returncode == 0 and stderr == b""
    assert json.loads(stdout)["resolve"] == "NetworkGuardViolation"
    assert interposer.journal is not None
    assert [(item["mode"], item["arguments"]) for item in interposer.journal.launches()] == [
        ("guard", ["--probe"]),
        ("guard", []),
    ]
    assert len(interposer.journal.pids()) == 2 and os.getpid() not in interposer.journal.pids()
    assert len(interposer.violations()) == 1

    # An environment the caller passes is kept, and one it inherits is too.
    with launch.at_subprocess(kimi_adapter, interposer, "subprocess"):
        inherited = kimi_adapter.subprocess.run(
            probe, check=False, capture_output=True, input=b"", timeout=60
        )
    assert json.loads(inherited.stdout)["harness_variables"] == []


def test_a_target_the_harness_cannot_run_is_refused_before_a_process_starts(
    standin: StandIn, tmp_path: Path
) -> None:
    interposer = _interposer(standin, tmp_path / "journal")
    started: list[Any] = []

    def original(argv: Any, **_kwargs: Any) -> None:
        started.append(argv)

    wrapped = interposer.supervised(original)
    for refused in (
        (str(standin.python),),
        (str(standin.python), "-B", "-I", "-"),
        (str(standin.python), "-i", str(standin.script)),
    ):
        with pytest.raises(launch.LaunchRefused):
            wrapped(refused, env={})
        with pytest.raises(launch.LaunchRefused):
            interposer.run(list(refused), capture_output=True)
    # A seam that changed shape cannot lose the guard quietly.
    with pytest.raises(launch.LaunchRefused, match="by keyword"):
        wrapped(tuple(standin.command("-m")))
    with pytest.raises(launch.LaunchRefused, match="positional"):
        interposer.run(standin.command("-m"), 0)
    with pytest.raises(launch.LaunchRefused, match="executable="):
        interposer.run(["python", "-m", "x"], executable=str(standin.python))
    assert started == [] and interposer.journal is None

    # The harness itself runs nothing without the settings of a launch.
    command = [str(standin.python), "-B", "-I", str(launch.HARNESS), "-m", WORKER_MODULE]
    for environment, reason in (
        ({}, "is not set"),
        ({worker_harness.SETTINGS_ENV: "not json"}, "is not JSON"),
        ({worker_harness.SETTINGS_ENV: json.dumps({"mode": "guard"})}, "does not hold"),
        (
            {
                worker_harness.SETTINGS_ENV: json.dumps(
                    {"mode": "guard", "journal": str(tmp_path / "absent")}
                )
            },
            "does not exist",
        ),
        (
            {
                worker_harness.SETTINGS_ENV: json.dumps(
                    {"mode": "open", "journal": str(tmp_path)}
                )
            },
            "does not hold",
        ),
    ):
        completed = subprocess.run(
            command, input="{}", capture_output=True, text=True, timeout=60, env=environment
        )
        assert completed.returncode == 1 and completed.stdout == ""
        assert completed.stderr.startswith("worker harness: ") and reason in completed.stderr
    for arguments, reason in (((), "no target"), (("-m",), "names no target"), (("-x",), "not a")):
        with pytest.raises(worker_harness.HarnessError, match=reason):
            worker_harness.split_target(arguments)


# -- a worker environment that is missing is an error ------------------------------


class _NotInstalled(deepseek_driver.DeepSeekDriver):
    """The DeepSeek driver on a machine without its worker environment."""

    def __init__(self, project: Path) -> None:
        self._project = project

    @property
    def interpreter(self) -> Path:
        return self._project / ".venv" / "bin" / "python"


def test_a_missing_worker_environment_is_an_error_not_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kind = matrix_columns.DEEPSEEK_KIND
    (column,) = (item for item in matrix_columns.real_columns() if item.kind == kind)
    missing = _NotInstalled(launch.REPOSITORY / "backends" / "absent_worker")

    with pytest.raises(MatrixSetupError) as refused:
        missing.preflight()
    reason = str(refused.value)
    assert "does not exist" in reason and deepseek_driver.INSTALL in reason
    assert deepseek_driver.INSTALL == (
        "uv sync --frozen --reinstall --link-mode copy --project backends/deepseek_worker"
    )

    # The session fixture keeps the reason, and every cell of the column fails
    # with it: a failure of the test item, never a skip and never a pass.
    monkeypatch.setattr(deepseek_driver.DeepSeekDriver, "interpreter", missing.interpreter)
    preflight = runner.preflight_all()
    kept = preflight[kind]
    assert kept is not None and deepseek_driver.INSTALL in kept
    assert preflight[matrix_columns.TOY_KIND] is None
    with pytest.raises(pytest.fail.Exception, match="column cannot be driven here") as failed:
        runner.run_cell(
            Cell(column, "S00", "raw"), tmp_path, preflight=preflight, known=ledger.load()
        )
    assert deepseek_driver.INSTALL in str(failed.value)
    assert failed.type is pytest.fail.Exception

    # An environment that is installed and cannot be pinned (a hard-linked
    # install, a drifted lock) is the same error, with the same command.
    present = _NotInstalled(tmp_path / "present_worker")
    present.interpreter.parent.mkdir(parents=True)
    present.interpreter.write_text("", encoding="utf-8")

    def unpinnable() -> None:
        raise deepseek_driver.adapter.DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker tree contains an unsupported entry"
        )

    monkeypatch.setattr(
        deepseek_driver.adapter, "capture_project_locked_deepseek_runtime_identity", unpinnable
    )
    with pytest.raises(MatrixSetupError, match="cannot be pinned") as drifted:
        present.preflight()
    assert deepseek_driver.INSTALL in str(drifted.value)
    assert "unsupported entry" in str(drifted.value)

    # An adapter that started an interpreter outside every worker environment
    # would run it past the launch interposer: that is refused too.
    monkeypatch.setattr(
        deepseek_driver.adapter,
        "capture_project_locked_deepseek_runtime_identity",
        lambda: SimpleNamespace(logical_worker_python=sys.executable),
    )
    with pytest.raises(MatrixSetupError, match="outside every worker environment"):
        present.preflight()

    # The session fixture asks every driver once, before the per-test fixture
    # has isolated the environment. A variable that no cell ever sees must not
    # decide the answer there. With a developer's override of the worker
    # interpreter set, which the adapter's capture function refuses, the
    # driver's own preflight fails for it; the preflight of the session does
    # not, because it runs under the cleared environment of the cells.
    override = "CRUCIBLE_DEEPSEEK_WORKER_PYTHON"
    machine = dict.fromkeys(matrix_columns.CLEARED_ENV, "a-value-from-the-machine")
    assert override in machine
    seen: list[set[str]] = []

    def capture() -> SimpleNamespace:
        seen.append(set(machine) & set(os.environ))
        if os.environ.get(override, "").strip():
            raise deepseek_driver.adapter.DeepSeekSDKQualificationError(
                "project-locked DeepSeek worker rejects ambient interpreter overrides"
            )
        return SimpleNamespace(logical_worker_python=sys.executable)

    monkeypatch.setattr(
        deepseek_driver.adapter, "capture_project_locked_deepseek_runtime_identity", capture
    )
    monkeypatch.setattr(deepseek_driver.DeepSeekDriver, "interpreter", present.interpreter)
    for name, value in machine.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(MatrixSetupError, match="rejects ambient interpreter overrides"):
        present.preflight()
    kept = runner.preflight_all()[kind]
    assert kept is not None and "outside every worker environment" in kept
    assert "ambient interpreter overrides" not in kept
    assert seen == [set(machine), set()]
    # The machine's variables are back once the preflight is over.
    assert {name: os.environ.get(name) for name in machine} == machine


# -- the redirect inside the worker ---------------------------------------------


@contextlib.contextmanager
def _echo_server() -> Iterator[tuple[str, list[dict[str, str]]]]:
    """A loopback HTTP server that records the Host header and path it saw."""

    seen: list[dict[str, str]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            seen.append({"host": self.headers.get("Host", ""), "path": self.path})
            self.send_response(200)
            self.send_header("Content-Length", "8")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"scripted")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_the_request_launch_is_redirected_and_every_other_launch_is_only_guarded(
    standin: StandIn, tmp_path: Path
) -> None:
    fetch = {"fetch": f"{PRODUCTION}/v1/answer?round=1"}
    request = standin.command("-m", ("-B", "-I"))
    probe = standin.command("-m", ("-B", "-I"), "--probe")

    with _echo_server() as (target, seen):
        interposer = _interposer(
            standin,
            tmp_path / "journal",
            launch.Redirect(module=WORKER_MODULE, origin=PRODUCTION, target=target),
        )
        with launch.at_supervisor(deepseek_adapter, interposer, SUPERVISOR):
            answered = _supervised(request, fetch, tmp_path)
            # The same request from any other launch of the interpreter.
            probed = _supervised(probe, fetch, tmp_path)
            scripted = _supervised(standin.command("script", ("-B", "-I")), fetch, tmp_path)
            # Another origin is refused by the redirect itself.
            foreign = _supervised(request, {"fetch": "https://other.invalid/x"}, tmp_path)

    # The worker asked for production and was answered by the scripted side.
    assert answered.returncode == 0 and answered.stderr == b""
    assert json.loads(answered.stdout)["fetch"] == [200, "scripted"]
    assert seen == [{"host": "provider.invalid", "path": "/v1/answer?round=1"}]
    # Outside the request launch the same request never left the worker.
    assert json.loads(probed.stdout)["fetch"] == "NetworkGuardViolation"
    assert json.loads(scripted.stdout)["fetch"] == "NetworkGuardViolation"
    assert json.loads(foreign.stdout)["fetch"] == "TapViolation"

    assert interposer.journal is not None
    launches = interposer.journal.launches()
    assert [item["mode"] for item in launches] == ["redirect", "guard", "guard", "redirect"]
    assert launches[0]["redirect"] == {"origin": PRODUCTION, "target": target}
    assert launches[1]["redirect"] is None
    violations = interposer.violations()
    assert len(violations) == 3
    assert sum(line.startswith("network guard in worker process") for line in violations) == 2
    assert sum(line.startswith("origin redirect in worker process") for line in violations) == 1
    assert any("other.invalid" in line for line in violations)
    # The test process itself reached nowhere.
    assert netguard.take_violations() == ()
