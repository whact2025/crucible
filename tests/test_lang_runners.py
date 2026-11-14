# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for the per-language runners + polyglot dispatcher.

Each language runner is invokable via ``python -m
crucible.common.runners.lang.<lang>``. We exercise them by calling
their ``main()`` with synthetic stdin/stdout streams that point at
shell commands inside a tmp project. No real cargo / cmake required.

Polyglot runner is exercised via subprocess (the real path workflows
take) and verifies short-circuit-on-first-failure + aggregated values.
"""

from __future__ import annotations

import ast
import io
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


# ----------------------------------------------------------------------
# Python language runner — exercise pytest summary parsing
# ----------------------------------------------------------------------


def test_python_verify_defaults_use_shared_bounded_failure_batch(tmp_path):
    from crucible.common.runners.lang import python as python_runner
    from crucible.common.runners.pytest_policy import (
        PYTEST_DEV_VERIFY_MAXFAIL_ARG,
    )

    assert python_runner._DEFAULT_TEST_COMMAND[-1] == (
        PYTEST_DEV_VERIFY_MAXFAIL_ARG
    )
    assert python_runner.verify_command(tmp_path) == (
        f"pytest {PYTEST_DEV_VERIFY_MAXFAIL_ARG}"
    )
    assert "-x" not in python_runner._DEFAULT_TEST_COMMAND


def test_embedded_pytest_observer_stays_python36_compatible():
    """The observer is compiled by the target repo's Python, not Crucible's."""

    from crucible.common.runners import pytest_observer

    source_path = Path(pytest_observer.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Python 3.6 cannot enable postponed annotations, evaluate PEP 585
    # built-in generics, or use PEP 604 unions. Keeping the embedded module
    # annotation-free avoids silently reintroducing any of those forms.
    assert "from __future__ import annotations" not in source
    assert not any(isinstance(node, ast.AnnAssign) for node in ast.walk(tree))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assert node.returns is None
        arguments = (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        assert all(argument.annotation is None for argument in arguments)

    # ``str.removeprefix`` starts at Python 3.9. The observer supports the
    # older testbed interpreters used by SWE-bench images.
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"removeprefix", "removesuffix"}
        for node in ast.walk(tree)
    )

    python36 = shutil.which("python3.6")
    if python36 is not None:
        completed = subprocess.run(
            [
                python36,
                "-c",
                "import sys; compile(open(sys.argv[1], 'rb').read(), "
                "sys.argv[1], 'exec')",
                str(source_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


_TINY_PYTEST_PROJECT = """\
import json, sys
# Fake pytest invocation: print a summary line and exit.
print("================ 3 passed, 1 failed, 2 skipped in 0.50s ================")
sys.exit(1)
"""


def test_python_runner_parses_pytest_summary(tmp_path: Path):
    """python.py runs the configured test_command and parses N passed/failed/skipped."""
    fake_pytest = tmp_path / "fake_pytest.py"
    fake_pytest.write_text(_TINY_PYTEST_PROJECT)

    cfg = {
        "bench_id": "t",
        "parameters": {
            "test_command": [sys.executable, str(fake_pytest)],
            "working_dir": str(tmp_path),
        },
        "env": {},
    }
    stdin = io.StringIO(json.dumps(cfg))
    stdout = io.StringIO()

    from crucible.common.runners.lang import python as py_runner
    rc = py_runner.main(stdin=stdin, stdout=stdout)
    assert rc == 1  # fake pytest exits non-zero

    payload = json.loads(stdout.getvalue())
    v = payload["values"]
    assert v["tests_passed"] == 3.0
    assert v["tests_failed"] == 1.0
    assert v["tests_skipped"] == 2.0
    assert v["python_tests_passed"] == 3.0
    assert v["returncode"] == 1.0


def test_python_runner_success_path(tmp_path: Path):
    """Runner exits 0 + sets tests_failed=0 when fake pytest succeeds."""
    good_pytest = tmp_path / "good_pytest.py"
    good_pytest.write_text(
        "import sys; print('5 passed in 0.10s'); sys.exit(0)"
    )

    cfg = {
        "bench_id": "t",
        "parameters": {
            "test_command": [sys.executable, str(good_pytest)],
            "working_dir": str(tmp_path),
        },
        "env": {},
    }
    stdin = io.StringIO(json.dumps(cfg))
    stdout = io.StringIO()

    from crucible.common.runners.lang import python as py_runner
    rc = py_runner.main(stdin=stdin, stdout=stdout)
    assert rc == 0
    v = json.loads(stdout.getvalue())["values"]
    assert v["tests_passed"] == 5.0
    # pytest "5 passed in 0.10s" carries no failed/skipped counts;
    # the runner only emits keys it parses.
    assert v.get("tests_failed", 0.0) == 0.0


def test_python_build_with_breach_envelope_skips_pytest_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The build half of a breach gate must not require a pytest command."""
    from crucible.common.runners.lang import python as py_runner

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="build complete\n",
            stderr="",
        )

    monkeypatch.setattr(py_runner.subprocess, "run", fake_run)
    stdout = io.StringIO()
    rc = py_runner.main(
        stdin=io.StringIO(
            json.dumps(
                {
                    "bench_id": "breach-build",
                    "parameters": {
                        "role": "build",
                        "runner_mode": "build",
                        "build_command": ["python", "-m", "build"],
                        "breach_observations": True,
                        "breach_attack_ids": ["attack"],
                    },
                    "env": {},
                }
            )
        ),
        stdout=stdout,
    )

    assert rc == 0
    assert calls == [["python", "-m", "build"]]
    payload = json.loads(stdout.getvalue())
    assert payload["runner_mode"] == "build"
    assert payload.get("test_observations", []) == []


def test_python_breach_observer_preserves_project_gate_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A Bash suite gate and its arguments remain byte-for-byte selected."""

    from crucible.common.runners.lang import python as py_runner

    monkeypatch.chdir(tmp_path)
    gate = ["bash", "internal/tools/run_gate.sh", "test", "--project-flag"]
    calls: list[list[str]] = []
    observer_prefixes: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        calls.append(list(command))
        if "import json,pytest,sys" in command[-1]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({
                    "python": sys.executable,
                    "pytest": pytest.__file__,
                }),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="1 passed in 0.01s\n",
            stderr="",
        )

    def fake_observations(*_args, python_prefix, **_kwargs):
        observer_prefixes.append(list(python_prefix))
        return [{
            "language": "python",
            "id": "tests/test_breach.py::test_attack",
            "status": "passed",
            "line": "tests/test_breach.py::test_attack",
        }]

    monkeypatch.setattr(py_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        py_runner,
        "_run_exact_pytest_observations",
        fake_observations,
    )
    stdout = io.StringIO()
    rc = py_runner.main(
        stdin=io.StringIO(json.dumps({
            "bench_id": "separate-observation-command",
            "parameters": {
                "test_command": gate,
                "working_dir": str(tmp_path),
                "per_language": {
                    "python": {
                        "breach_test_command": [
                            sys.executable,
                            "-m",
                            "pytest",
                        ],
                    },
                },
                "breach_observations": True,
                "breach_attack_ids": ["attack"],
                "breach_test_file": "tests/test_breach.py",
            },
            "env": {},
        })),
        stdout=stdout,
    )

    assert rc == 0
    assert calls[0][:3] == [sys.executable, "-B", "-I"]
    assert calls[1] == gate
    assert observer_prefixes == [[str(Path(sys.executable).absolute())]]
    assert json.loads(stdout.getvalue())["test_observations"][0]["status"] == (
        "passed"
    )


def test_python_breach_observer_rejects_non_pytest_gate_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy fallback never mines a pytest token from an arbitrary wrapper."""

    from crucible.common.runners.breach_observation import (
        BREACH_OBSERVATION_FAILURE_KEY,
    )
    from crucible.common.runners.lang import python as py_runner

    monkeypatch.setattr(
        py_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "project gate ran before observation binding failed"
        ),
    )
    stdout = io.StringIO()
    rc = py_runner.main(
        stdin=io.StringIO(json.dumps({
            "bench_id": "invalid-observation-binding",
            "parameters": {
                "test_command": ["bash", "gate.sh", "pytest"],
                "working_dir": str(tmp_path),
                "breach_observations": True,
                "breach_attack_ids": ["attack"],
                "breach_test_file": "tests/test_breach.py",
            },
            "env": {},
        })),
        stdout=stdout,
    )

    assert rc == 2
    payload = json.loads(stdout.getvalue())
    assert payload[BREACH_OBSERVATION_FAILURE_KEY]["kind"] == "configuration"
    assert payload["test_observations"] == []


def test_python_observation_binding_preserves_virtualenv_symlink_path() -> None:
    from crucible.common.runners.breach_observation import (
        resolve_python_observation_prefix,
    )

    selected = str(Path(sys.executable).absolute())
    assert resolve_python_observation_prefix(
        {"breach_test_command": [selected, "-m", "pytest"]},
        default_test_command=("pytest",),
        cwd=os.getcwd(),
    ) == [selected]
    if Path(selected).is_symlink():
        assert selected != str(Path(selected).resolve())


def test_python_observation_preflight_ignores_candidate_pytest_module(
    tmp_path: Path,
) -> None:
    from crucible.common.runners.breach_observation import (
        preflight_python_observation_channel,
    )

    marker = tmp_path / "candidate-pytest-imported"
    (tmp_path / "pytest.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    binding = preflight_python_observation_channel(
        {
            "breach_test_command": [
                str(Path(sys.executable).absolute()),
                "-m",
                "pytest",
            ],
        },
        default_test_command=("pytest",),
        cwd=tmp_path,
        candidate_root=tmp_path,
    )

    assert not marker.exists()
    assert not Path(binding.pytest_identity.lexical_path).is_relative_to(
        tmp_path,
    )


def test_python_observation_preflight_rejects_candidate_runtime(
    tmp_path: Path,
) -> None:
    from crucible.common.runners.breach_observation import (
        BreachObservationConfigurationError,
        preflight_python_observation_channel,
    )

    candidate_python = tmp_path / "python"
    candidate_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    candidate_python.chmod(0o755)

    with pytest.raises(
        BreachObservationConfigurationError,
        match="mutable candidate workspace",
    ):
        preflight_python_observation_channel(
            {
                "breach_test_command": [
                    str(candidate_python),
                    "-m",
                    "pytest",
                ],
            },
            default_test_command=("pytest",),
            cwd=tmp_path,
            candidate_root=tmp_path,
        )


def test_python_runner_rejects_runtime_outside_project_subdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trust root is the whole checkout, not only the project subdir."""

    from crucible.common.runners.breach_observation import (
        BREACH_OBSERVATION_FAILURE_KEY,
    )
    from crucible.common.runners.lang import python as py_runner

    project = tmp_path / "project"
    project.mkdir()
    candidate_python = tmp_path / "python"
    candidate_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    candidate_python.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        py_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "candidate runtime executed before trust-root rejection"
        ),
    )
    stdout = io.StringIO()

    assert py_runner.main(
        stdin=io.StringIO(json.dumps({
            "parameters": {
                "working_dir": str(tmp_path),
                "project_subdir": "project",
                "test_command": ["true"],
                "breach_test_command": [
                    str(candidate_python),
                    "-m",
                    "pytest",
                ],
                "breach_observations": True,
            },
            "env": {},
        })),
        stdout=stdout,
    ) == 2
    failure = json.loads(stdout.getvalue())[BREACH_OBSERVATION_FAILURE_KEY]
    assert failure["kind"] == "configuration"
    assert "mutable candidate workspace" in failure["detail"]


def test_python_observer_rejects_runtime_identity_change_after_gate(
    tmp_path: Path,
) -> None:
    from crucible.common.runners.breach_observation import (
        BreachObservationInfrastructureError,
        assert_python_observation_binding,
        preflight_python_observation_channel,
    )

    runtime = tmp_path.parent / f"{tmp_path.name}-trusted-runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "lib" / "python3.12").mkdir(parents=True)
    source_venv_root = Path(sys.executable).parent.parent
    (runtime / "pyvenv.cfg").write_text(
        (source_venv_root / "pyvenv.cfg").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    python_path = runtime / "bin" / "python"
    python_path.symlink_to(Path(sys.executable).resolve())
    site_packages = runtime / "lib" / "python3.12" / "site-packages"
    site_packages.symlink_to(source_venv_root / "lib/python3.12/site-packages")
    binding = preflight_python_observation_channel(
        {
            "breach_test_command": [
                str(python_path),
                "-m",
                "pytest",
            ],
        },
        default_test_command=("pytest",),
        cwd=tmp_path,
        candidate_root=tmp_path,
    )
    python_path.unlink()
    python_path.symlink_to("/bin/false")

    with pytest.raises(
        BreachObservationInfrastructureError,
        match="changed after preflight",
    ):
        assert_python_observation_binding(binding)


def test_python_runner_types_post_gate_runtime_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.common.runners.breach_observation import (
        BREACH_OBSERVATION_FAILURE_KEY,
    )
    from crucible.common.runners.lang import python as py_runner

    trusted = tmp_path.parent / f"{tmp_path.name}-bound-runtime"
    trusted.mkdir()
    python_path = trusted / "python"
    python_path.symlink_to(Path(sys.executable).resolve())
    pytest_path = trusted / "pytest_runtime.py"
    pytest_path.write_text("# bound pytest identity\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        calls.append(list(command))
        if "import json,pytest,sys" in command[-1]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({
                    "python": str(python_path),
                    "pytest": str(pytest_path),
                }),
                stderr="",
            )
        if command == ["bash", "project-gate.sh", "test"]:
            python_path.unlink()
            python_path.symlink_to("/bin/false")
            return subprocess.CompletedProcess(command, 0, "gate passed\n", "")
        pytest.fail(f"exact node executed after runtime replacement: {command!r}")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(py_runner.subprocess, "run", fake_run)
    stdout = io.StringIO()
    rc = py_runner.main(
        stdin=io.StringIO(json.dumps({
            "parameters": {
                "test_command": ["bash", "project-gate.sh", "test"],
                "breach_test_command": [
                    str(python_path),
                    "-m",
                    "pytest",
                ],
                "breach_observations": True,
                "breach_attack_ids": ["attack"],
                "breach_test_file": "tests/test_breach.py",
            },
            "env": {},
        })),
        stdout=stdout,
    )

    assert rc == 2
    assert len(calls) == 2
    failure = json.loads(stdout.getvalue())[BREACH_OBSERVATION_FAILURE_KEY]
    assert failure["kind"] == "infrastructure"
    assert "changed after preflight" in failure["detail"]


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["python", "-m", "pytest"],
        ["pytest"],
        ["python", "-m", "pytest", "-k", "attack"],
        ["bash", "-lc", "pytest"],
        ["pytest", "tests/test_breach.py"],
        ["python", "-m", "unittest"],
    ],
)
def test_static_breach_test_command_rejects_unsafe_shapes(command) -> None:
    from crucible.common.runners.parameters import validate_static

    with pytest.raises(ValueError, match="breach_test_command"):
        validate_static({
            "per_language": {
                "python": {"breach_test_command": command},
            },
        })


def test_static_breach_test_command_is_python_only() -> None:
    from crucible.common.runners.parameters import validate_static

    with pytest.raises(ValueError, match="per_language.python"):
        validate_static({
            "per_language": {
                "rust": {"breach_test_command": ["pytest"]},
            },
        })


@pytest.mark.parametrize(
    "parameters",
    [
        {"breach_test_command": ["pytest"]},
        {
            "per_slice": {
                "s0": {"breach_test_command": ["pytest"]},
            },
        },
        {"tool": {"breach_test_command": ["pytest"]}},
        {
            "per_language": {
                "python": {"nested": {"breach_test_command": ["pytest"]}},
            },
        },
        {
            "per_slice": {
                "s0": {"nested": {"breach_test_command": ["pytest"]}},
            },
        },
    ],
)
def test_static_breach_test_command_rejects_non_language_placement(
    parameters,
) -> None:
    from crucible.common.runners.parameters import validate_static

    with pytest.raises(ValueError, match="only under per_language.python"):
        validate_static(parameters)


def test_slice_local_breach_binding_deep_merges_python_parameters() -> None:
    from crucible.common.runners.parameters import for_language, for_slice

    base = {
        "per_language": {
            "python": {
                "test_args": ["-q"],
                "breach_test_command": [
                    str(Path(sys.executable).absolute()),
                    "-m",
                    "pytest",
                ],
            },
        },
        "per_slice": {
            "s0": {
                "per_language": {
                    "python": {
                        "breach_test_command": [
                            "/opt/slice/bin/python",
                            "-m",
                            "pytest",
                        ],
                    },
                },
            },
        },
    }

    selected = for_language(for_slice(base, "s0"), "python")
    assert selected["test_args"] == ["-q"]
    assert selected["breach_test_command"] == [
        "/opt/slice/bin/python",
        "-m",
        "pytest",
    ]


def test_python_runner_retains_bounded_attack_observation_before_long_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Structured attack evidence survives even when human output is truncated."""

    test_file = tmp_path / "test_observations.py"
    test_file.write_text(
        "def test_race_a():\n    assert True\n\n"
        "import pytest\n\n"
        "@pytest.mark.parametrize('value', range(180))\n"
        "def test_noise(value):\n    assert value >= 0\n",
        encoding="utf-8",
    )
    cfg = {
        "bench_id": "breach-observations",
        "parameters": {
            "test_command": [sys.executable, "-m", "pytest"],
            "working_dir": str(tmp_path),
            "breach_observations": True,
            "breach_attack_ids": ["race_a"],
            "breach_test_file": "test_observations.py",
        },
        "env": {},
    }
    stdout = io.StringIO()

    from crucible.common.runners.lang import python as py_runner

    monkeypatch.chdir(tmp_path)
    assert py_runner.main(
        stdin=io.StringIO(json.dumps(cfg)),
        stdout=stdout,
    ) == 0
    payload = json.loads(stdout.getvalue())
    assert "test_race_a PASSED" not in payload["raw_output"]
    assert len(payload["test_observations"]) == 1
    observation = payload["test_observations"][0]
    assert observation["language"] == "python"
    assert observation["id"] == "test_observations.py::test_race_a"
    assert observation["status"] == "passed"
    assert observation["line"] == "test_observations.py::test_race_a"


@pytest.mark.parametrize(
    ("source", "expected", "failure_phase"),
    [
        ("...\n", "runner_error", None),
        ("def test_attack():\n    assert True\n", "passed", None),
        ("def test_attack():\n    assert False\n", "failed", "call"),
        (
            "import pytest\n"
            "@pytest.mark.skip(reason='safety unexecutable')\n"
            "def test_attack():\n    assert False\n",
            "not_observed",
            None,
        ),
        (
            "import pytest\n"
            "def test_attack():\n    pytest.skip('not exercised')\n",
            "not_observed",
            None,
        ),
        (
            "import pytest\n"
            "@pytest.mark.xfail(reason='not exercised')\n"
            "def test_attack():\n    assert False\n",
            "not_observed",
            None,
        ),
        (
            "import pytest\n"
            "@pytest.fixture(autouse=True)\n"
            "def broken_setup():\n    raise RuntimeError('setup')\n"
            "def test_attack():\n    assert True\n",
            "runner_error",
            "setup",
        ),
        (
            "import pytest\n"
            "@pytest.fixture(autouse=True)\n"
            "def broken_teardown():\n"
            "    yield\n"
            "    raise RuntimeError('teardown')\n"
            "def test_attack():\n    assert True\n",
            "runner_error",
            "teardown",
        ),
        (
            "import os\n"
            "def test_attack():\n    os._exit(70)\n",
            "runner_error",
            None,
        ),
        (
            "import atexit, os\n"
            "def test_attack():\n"
            "    atexit.register(lambda: os._exit(0))\n"
            "    assert False\n",
            "runner_error",
            None,
        ),
    ],
)
def test_exact_pytest_observer_is_phase_aware_and_supervised(
    tmp_path: Path,
    source: str,
    expected: str,
    failure_phase: str | None,
):
    test_file = tmp_path / "test_exact.py"
    test_file.write_text(source, encoding="utf-8")

    from crucible.common.runners.lang.python import (
        _run_exact_pytest_observations,
    )

    observations = _run_exact_pytest_observations(
        [sys.executable, "-m", "pytest"],
        working_dir=str(tmp_path),
        env=os.environ.copy(),
        test_file="test_exact.py",
        attack_ids=["attack"],
    )
    assert [{key: observation[key] for key in (
        "language", "id", "status", "line",
    )} for observation in observations] == [{
        "language": "python",
        "id": "test_exact.py::test_attack",
        "status": expected,
        "line": "test_exact.py::test_attack",
    }]
    if failure_phase is None:
        assert "failure" not in observations[0]
    else:
        failure = observations[0]["failure"]
        assert failure["schema"] == "crucible.pytest.failure.v1"
        assert failure["phase"] == failure_phase
        assert failure["test"] == {
            "path": "test_exact.py",
            "function": "test_attack",
        }
        assert set(failure["frame"]) == {"path", "function", "line"}
        assert set(failure["exception"]) == {"type"}


def test_exact_pytest_observer_proves_direct_class_attribute(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_exact.py").write_text(
        "from enum import Enum\n"
        "class NameStyle(Enum):\n"
        "    LOWER_SNAKE = 'lower_snake'\n"
        "def test_attack():\n"
        "    assert NameStyle.SNAKE\n",
        encoding="utf-8",
    )

    from crucible.common.runners.lang.python import (
        _run_exact_pytest_observations,
    )

    observation = _run_exact_pytest_observations(
        [sys.executable, "-m", "pytest"],
        working_dir=str(tmp_path),
        env=os.environ.copy(),
        test_file="test_exact.py",
        attack_ids=["attack"],
    )[0]

    assert observation["status"] == "failed"
    assert observation["failure"] == {
        "schema": "crucible.pytest.failure.v1",
        "phase": "call",
        "test": {"path": "test_exact.py", "function": "test_attack"},
        "frame": {
            "path": "test_exact.py",
            "function": "test_attack",
            "line": 5,
        },
        "exception": {
            "type": "AttributeError",
            "subject_kind": "type",
            "subject_module": "test_exact",
            "subject_name": "NameStyle",
            "attribute": "SNAKE",
            "direct_source_reference": True,
        },
    }


def test_exact_pytest_observer_isolates_bootstrap_from_candidate_pytest(
    tmp_path: Path,
) -> None:
    """The trusted import is isolated while pytest still loads project tests."""

    (tmp_path / "pytest.py").write_text(
        "raise RuntimeError('candidate pytest shadow imported')\n",
        encoding="utf-8",
    )
    (tmp_path / "product.py").write_text(
        "def answer():\n    return 42\n",
        encoding="utf-8",
    )
    (tmp_path / "test_exact.py").write_text(
        "from product import answer\n"
        "def test_attack():\n"
        "    assert answer() == 42\n",
        encoding="utf-8",
    )

    from crucible.common.runners.lang.python import (
        _run_exact_pytest_observations,
    )

    observation = _run_exact_pytest_observations(
        python_prefix=[str(Path(sys.executable).absolute())],
        working_dir=str(tmp_path),
        env=os.environ.copy(),
        test_file="test_exact.py",
        attack_ids=["attack"],
    )[0]

    assert observation["status"] == "passed"


def test_exact_pytest_observer_never_writes_candidate_bytecode(
    tmp_path: Path,
) -> None:
    """Command-line isolation must not override the no-bytecode boundary."""

    (tmp_path / "product.py").write_text(
        "def answer():\n    return 42\n",
        encoding="utf-8",
    )
    (tmp_path / "test_exact.py").write_text(
        "from product import answer\n"
        "def test_attack():\n"
        "    assert answer() == 42\n",
        encoding="utf-8",
    )

    from crucible.common.runners.lang.python import (
        _run_exact_pytest_observations,
    )

    env = os.environ.copy()
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    observation = _run_exact_pytest_observations(
        python_prefix=[str(Path(sys.executable).absolute())],
        working_dir=str(tmp_path),
        env=env,
        test_file="test_exact.py",
        attack_ids=["attack"],
    )[0]

    assert observation["status"] == "passed"
    assert not list(tmp_path.rglob("*.pyc"))
    assert not list(tmp_path.rglob("__pycache__"))


def test_exact_pytest_observer_recovers_legacy_attribute_fields():
    """Python 3.6-3.9 lack ``AttributeError.obj`` and ``.name``."""

    from crucible.common.runners import pytest_observer

    class LegacyNameStyle:
        pass

    # A nested class has ``<locals>`` in its qualname, which the wire schema
    # deliberately rejects. Give this test fixture the ordinary identity a
    # module-level product class would expose.
    LegacyNameStyle.__qualname__ = "LegacyNameStyle"

    def test_attack():
        return LegacyNameStyle.SNAKE

    try:
        test_attack()
    except AttributeError as exc:
        frames = pytest_observer._traceback_frames(exc.__traceback__)
        # Reproduce the structured fields exposed before Python 3.10.
        exc.name = None
        exc.obj = None
        record = pytest_observer._direct_class_attribute(
            exc,
            frames,
            expected_path=str(Path(__file__).resolve()),
            expected_function="test_attack",
        )
    else:  # pragma: no cover - the missing member is the test premise
        raise AssertionError("missing class member unexpectedly resolved")

    assert record == {
        "subject_kind": "type",
        "subject_module": __name__,
        "subject_name": "LegacyNameStyle",
        "attribute": "SNAKE",
        "direct_source_reference": True,
    }


def test_exact_pytest_observer_keeps_terminal_product_frame_without_message(
    tmp_path: Path,
) -> None:
    (tmp_path / "product.py").write_text(
        "def reject():\n"
        "    raise ValueError('untrusted test input was incomplete')\n",
        encoding="utf-8",
    )
    (tmp_path / "test_exact.py").write_text(
        "from product import reject\n"
        "def test_attack():\n"
        "    reject()\n",
        encoding="utf-8",
    )

    from crucible.common.runners.lang.python import (
        _run_exact_pytest_observations,
    )

    failure = _run_exact_pytest_observations(
        [sys.executable, "-m", "pytest"],
        working_dir=str(tmp_path),
        env=os.environ.copy(),
        test_file="test_exact.py",
        attack_ids=["attack"],
    )[0]["failure"]

    assert failure == {
        "schema": "crucible.pytest.failure.v1",
        "phase": "call",
        "test": {"path": "test_exact.py", "function": "test_attack"},
        "frame": {"path": "product.py", "function": "reject", "line": 2},
        "exception": {"type": "ValueError"},
    }
    assert "untrusted test input" not in json.dumps(failure)


def test_exact_pytest_observer_excludes_instance_attribute_details(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_exact.py").write_text(
        "class Subject:\n"
        "    pass\n"
        "def test_attack():\n"
        "    Subject().MISSING\n",
        encoding="utf-8",
    )

    from crucible.common.runners.lang.python import (
        _run_exact_pytest_observations,
    )

    failure = _run_exact_pytest_observations(
        [sys.executable, "-m", "pytest"],
        working_dir=str(tmp_path),
        env=os.environ.copy(),
        test_file="test_exact.py",
        attack_ids=["attack"],
    )[0]["failure"]
    assert failure["exception"] == {"type": "AttributeError"}


def test_pytest_failure_validation_rejects_unbound_or_diagnostic_fields() -> None:
    from copy import deepcopy

    from crucible.common.runners.observations import (
        pytest_observer_failure,
        validate_pytest_failure,
    )

    failure = {
        "schema": "crucible.pytest.failure.v1",
        "phase": "call",
        "test": {"path": "test_exact.py", "function": "test_attack"},
        "frame": {"path": "product.py", "function": "reject", "line": 2},
        "exception": {"type": "ValueError"},
    }
    assert validate_pytest_failure(
        failure, expected_node_id="test_exact.py::test_attack",
    ) == failure
    canonical = json.dumps(
        failure, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
    )
    assert pytest_observer_failure(
        canonical, expected_node_id="test_exact.py::test_attack",
    ) == failure
    assert pytest_observer_failure(
        "", expected_node_id="test_exact.py::test_attack",
    ) is None

    malformed: list[dict[str, object]] = []
    wrong_test = deepcopy(failure)
    wrong_test["test"]["function"] = "test_other"
    malformed.append(wrong_test)
    with_message = deepcopy(failure)
    with_message["exception"]["message"] = "arbitrary diagnostic"
    malformed.append(with_message)
    with_source = deepcopy(failure)
    with_source["frame"]["source"] = "raise ValueError(secret)"
    malformed.append(with_source)
    unbounded = deepcopy(failure)
    unbounded["frame"]["path"] = "x" * 513
    malformed.append(unbounded)

    for candidate in malformed:
        with pytest.raises(ValueError, match="pytest failure metadata"):
            validate_pytest_failure(
                candidate, expected_node_id="test_exact.py::test_attack",
            )
    with pytest.raises(ValueError, match="canonical JSON"):
        pytest_observer_failure(
            canonical + "\n", expected_node_id="test_exact.py::test_attack",
        )


def test_pytest_failure_phase_must_match_observation_status() -> None:
    from crucible.common.runners.observations import filter_test_observations

    observation = {
        "language": "python",
        "id": "test_exact.py::test_attack",
        "status": "runner_error",
        "line": "test_exact.py::test_attack",
        "failure": {
            "schema": "crucible.pytest.failure.v1",
            "phase": "call",
            "test": {"path": "test_exact.py", "function": "test_attack"},
            "frame": {"path": "product.py", "function": "reject", "line": 2},
            "exception": {"type": "ValueError"},
        },
    }

    with pytest.raises(ValueError, match="phase contradicts"):
        filter_test_observations(
            [observation], ["attack"], language="python",
        )


def test_exact_pytest_observer_does_not_wait_for_spawned_stdio_holder(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "test_exact.py"
    test_file.write_text(
        "import subprocess, sys\n"
        "def test_attack():\n"
        "    subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(5)'])\n"
        "    assert True\n",
        encoding="utf-8",
    )

    from crucible.common.runners.lang.python import (
        _run_exact_pytest_observations,
    )

    started = time.monotonic()
    observations = _run_exact_pytest_observations(
        [sys.executable, "-m", "pytest"],
        working_dir=str(tmp_path),
        env=os.environ.copy(),
        test_file="test_exact.py",
        attack_ids=["attack"],
    )
    elapsed = time.monotonic() - started
    assert elapsed < 3
    assert observations[0]["status"] == "passed"


def test_canonical_attack_identity_rejects_cross_finding_alias():
    from crucible.common.runners.observations import canonical_attack_test_ids

    with pytest.raises(ValueError, match="collide after canonicalization"):
        canonical_attack_test_ids(["race-a", "race_a"], "python")


def test_direct_rust_runner_honors_nested_spec_parameters():
    from crucible.common.runners.lang.rust import _build_cmd

    parameters = {
        "role": "build",
        "per_language": {
            "rust": {
                "crate": "fabric-render-ovrtx",
                "cargo_features": ["ovrtx"],
                "cargo_test_args": ["--", "--test-threads=1"],
            },
        },
    }
    assert _build_cmd(parameters) == [
        "cargo",
        "build",
        "-p",
        "fabric-render-ovrtx",
        "--features",
        "ovrtx",
    ]
    parameters["role"] = "verify"
    assert _build_cmd(parameters) == [
        "cargo",
        "test",
        "-p",
        "fabric-render-ovrtx",
        "--features",
        "ovrtx",
        "--",
        "--test-threads=1",
    ]

    parameters["runner_mode"] = "bench"
    parameters["per_language"]["rust"]["cargo_bench_args"] = [
        "--bench",
        "runtime_update",
    ]
    assert _build_cmd(parameters) == [
        "cargo",
        "bench",
        "-p",
        "fabric-render-ovrtx",
        "--features",
        "ovrtx",
        "--bench",
        "runtime_update",
    ]


def test_direct_rust_runner_rejects_recursive_language_projection():
    from crucible.common.runners.lang import rust as rust_runner

    cfg = {
        "bench_id": "recursive",
        "parameters": {
            "per_language": {
                "rust": {
                    "crate": "expected",
                    "per_language": {
                        "rust": {
                            "crate": "unexpected",
                            "cargo_test_args": ["--test", "omitted_attack"],
                        },
                    },
                },
            },
        },
        "env": {},
    }
    stdout = io.StringIO()

    assert rust_runner.main(
        stdin=io.StringIO(json.dumps(cfg)), stdout=stdout,
    ) == 2
    assert "may not nest" in json.loads(stdout.getvalue())["raw_output"]


def test_polyglot_rejects_recursive_language_projection():
    from crucible.common.runners.polyglot import _per_language_params

    with pytest.raises(ValueError, match="may not nest"):
        _per_language_params(
            {
                "languages": ["rust"],
                "per_language": {
                    "rust": {
                        "per_language": {
                            "rust": {"crate": "unexpected"},
                        },
                    },
                },
            },
            "rust",
        )


# ----------------------------------------------------------------------
# Rust language runner — exercise cargo-output parsing
# ----------------------------------------------------------------------


def test_rust_runner_parses_cargo_summary_across_crates(tmp_path: Path):
    """Cargo emits one ``test result:`` line per crate; runner aggregates."""
    fake_cargo = tmp_path / "fake_cargo.py"
    fake_cargo.write_text(textwrap.dedent("""
        import sys
        print('   Compiling foo v0.1.0')
        print('test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured')
        print('test result: ok. 3 passed; 1 failed; 0 ignored; 0 measured')
        sys.exit(1)
    """).lstrip())

    cfg = {
        "bench_id": "t",
        "parameters": {
            # Override cargo with our fake by passing the test_args
            # path via _build_cmd... actually, since rust.py hardcodes
            # ``cargo``, we'd need a PATH-shim. Easier: shell into the
            # script directly via a sentinel.
        },
        "env": {},
    }
    # Approach: monkeypatch the cargo invocation by stubbing
    # subprocess.run inside the rust runner module.
    import importlib
    rust_runner = importlib.import_module("crucible.common.runners.lang.rust")

    class _FakeCompleted:
        def __init__(self, rc, out):
            self.returncode = rc
            self.stdout = out
            self.stderr = ""

    def fake_run(cmd, **kw):  # noqa: ARG001
        return _FakeCompleted(
            rc=1,
            out=(
                "   Compiling foo v0.1.0\n"
                "test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured\n"
                "test result: ok. 3 passed; 1 failed; 0 ignored; 0 measured\n"
            ),
        )

    orig_run = rust_runner.subprocess.run
    rust_runner.subprocess.run = fake_run
    try:
        stdin = io.StringIO(json.dumps(cfg))
        stdout = io.StringIO()
        rc = rust_runner.main(stdin=stdin, stdout=stdout)
    finally:
        rust_runner.subprocess.run = orig_run

    assert rc == 1
    v = json.loads(stdout.getvalue())["values"]
    # Workspace totals: 5+3 passed, 0+1 failed, 1+0 ignored.
    assert v["tests_passed"] == 8.0
    assert v["tests_failed"] == 1.0
    assert v["tests_ignored"] == 1.0
    assert v["rust_tests_passed"] == 8.0


def test_rust_build_with_breach_envelope_skips_libtest_discovery(monkeypatch):
    """The build half of a breach gate must not append libtest arguments."""
    from crucible.common.runners.lang import rust as rust_runner

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command, 0, stdout="Finished dev profile\n", stderr="",
        )

    monkeypatch.setattr(rust_runner.subprocess, "run", fake_run)
    stdout = io.StringIO()
    rc = rust_runner.main(
        stdin=io.StringIO(json.dumps({
            "bench_id": "breach-build",
            "parameters": {
                "role": "build",
                "runner_mode": "build",
                "breach_observations": True,
                "breach_attack_ids": ["attack"],
            },
            "env": {},
        })),
        stdout=stdout,
    )

    assert rc == 0
    assert calls == [["cargo", "build", "--workspace"]]
    payload = json.loads(stdout.getvalue())
    assert payload["runner_mode"] == "build"
    assert payload.get("test_observations", []) == []


def test_rust_runner_uses_crate_arg_when_set():
    """parameters.crate switches cargo to ``-p <crate>`` form."""
    from crucible.common.runners.lang.rust import _build_cmd
    assert _build_cmd({}) == ["cargo", "test", "--workspace"]
    assert _build_cmd({"role": "build"}) == ["cargo", "build", "--workspace"]
    assert _build_cmd({"crate": "foo"}) == ["cargo", "test", "-p", "foo"]
    assert _build_cmd({
        "crate": "foo",
        "test_args": ["--features", "vulkan-rt"],
    }) == ["cargo", "test", "-p", "foo", "--features", "vulkan-rt"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"crate": 7},
        {"crate": ""},
        {"cargo_features": "ovrtx"},
        {"cargo_features": ["ovrtx", 7]},
        {"cargo_test_args": "-- --test-threads=1"},
        {"cargo_bench_args": "--bench runtime_update"},
        {"test_args": "--features ovrtx"},
    ],
)
def test_rust_runner_rejects_malformed_selection(parameters):
    from crucible.common.runners.lang.rust import _build_cmd

    with pytest.raises(ValueError, match="Rust runner parameter"):
        _build_cmd(parameters)


def test_rust_convention_helpers_ignore_legacy_feature_environment(
    monkeypatch,
    tmp_path: Path,
):
    from crucible.common.runners.lang import rust as rust_runner

    monkeypatch.setenv("CRUCIBLE_CARGO_FEATURES", "--all-features")
    assert rust_runner.build_command(tmp_path) == "cargo build --workspace"
    assert rust_runner.verify_command(tmp_path) == "cargo test --workspace"


def test_rust_runner_applies_typed_features_to_build_and_verify():
    """Explicit features are shared; test selectors are verify-only."""
    from crucible.common.runners.lang.rust import _build_cmd

    parameters = {
        "crate": "fabric-render-ovrtx",
        "cargo_features": ["ovrtx", "trace"],
        "cargo_test_args": [
            "--test", "ovstage_protocol", "--",
            "--nocapture", "--test-threads=1",
        ],
    }
    assert _build_cmd({**parameters, "role": "build"}) == [
        "cargo", "build", "-p", "fabric-render-ovrtx",
        "--features", "ovrtx,trace",
    ]
    assert _build_cmd({**parameters, "role": "verify"}) == [
        "cargo", "test", "-p", "fabric-render-ovrtx",
        "--features", "ovrtx,trace",
        "--test", "ovstage_protocol", "--",
        "--nocapture", "--test-threads=1",
    ]


def test_rust_bench_uses_only_frozen_bench_arguments():
    from crucible.common.runners.lang.rust import _build_cmd

    parameters = {
        "role": "verify",
        "runner_mode": "bench",
        "crate": "fabric-render-ovrtx",
        "cargo_features": ["ovrtx", "trace"],
        "test_args": ["--all-targets"],
        "cargo_test_args": ["--test", "protocol", "--", "--nocapture"],
        "cargo_bench_args": ["--bench", "runtime_update", "--", "--sample-size", "10"],
    }
    assert _build_cmd(parameters) == [
        "cargo", "bench", "-p", "fabric-render-ovrtx",
        "--features", "ovrtx,trace",
        "--bench", "runtime_update", "--", "--sample-size", "10",
    ]


@pytest.mark.parametrize("language", ["python", "cpp"])
def test_non_benchmark_language_adapters_fail_closed_before_execution(
    language: str,
    monkeypatch: pytest.MonkeyPatch,
):
    module = __import__(
        f"crucible.common.runners.lang.{language}", fromlist=["main"]
    )

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("unsupported benchmark adapter executed a command")

    monkeypatch.setattr(module.subprocess, "run", unexpected_run)
    stdout = io.StringIO()
    rc = module.main(
        stdin=io.StringIO(json.dumps({
            "bench_id": "unsupported",
            "parameters": {
                "role": "verify",
                "runner_mode": "bench",
            },
            "env": {},
        })),
        stdout=stdout,
    )
    assert rc == 2
    assert "does not support" in json.loads(stdout.getvalue())["raw_output"]


@pytest.mark.parametrize(
    ("exact_output", "exact_rc", "expected"),
    [
        ("test test_attack ... ok\n", 0, "passed"),
        ("test test_attack ... FAILED\n", 101, "failed"),
        ("test test_attack ... ignored\n", 0, "not_observed"),
        (
            "test test_attack ... ignored, SAFETY: private DSO unavailable\n",
            0,
            "not_observed",
        ),
        ("test test_attack ... ignored arbitrary suffix\n", 0, "runner_error"),
        ("test test_attack ... ok, forged suffix\n", 0, "runner_error"),
        ("error: could not compile\n", 101, "runner_error"),
    ],
)
def test_exact_libtest_observer_requires_consistent_named_result(
    monkeypatch: pytest.MonkeyPatch,
    exact_output: str,
    exact_rc: int,
    expected: str,
):
    from crucible.common.runners.lang import rust as rust_runner

    def fake_run(command, **kwargs):  # noqa: ARG001
        if command[-2:] == ["--", "--list"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="test_attack: test\n", stderr="",
            )
        return subprocess.CompletedProcess(
            command, exact_rc, stdout=exact_output, stderr="",
        )

    monkeypatch.setattr(rust_runner.subprocess, "run", fake_run)
    observations = rust_runner._run_exact_libtest_observations(
        {
            "breach_attack_ids": ["attack"],
            "cargo_test_args": ["--", "--test-threads=1"],
        },
        working_dir=".",
        env=os.environ.copy(),
    )
    assert observations[0]["status"] == expected
    if expected != "runner_error":
        assert observations[0]["line"] == exact_output.rstrip()


def test_exact_libtest_observer_marks_missing_discovery_as_runner_error(
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.common.runners.lang import rust as rust_runner

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="unrelated_test: test\n", stderr="",
        )

    monkeypatch.setattr(rust_runner.subprocess, "run", fake_run)
    observations = rust_runner._run_exact_libtest_observations(
        {
            "breach_attack_ids": ["attack"],
            "cargo_test_args": ["--", "--test-threads=1"],
        },
        working_dir=".",
        env=os.environ.copy(),
    )

    assert len(calls) == 1
    assert observations == [{
        "language": "rust",
        "id": "test_attack",
        "status": "runner_error",
        "line": "test_attack (canonical libtest node absent from discovery)",
    }]


# ----------------------------------------------------------------------
# C++ language runner — exercise ctest-output parsing
# ----------------------------------------------------------------------


def test_cpp_runner_parses_ctest_summary(tmp_path: Path, monkeypatch):
    """cmake build + ctest, summarize ``X% tests passed, Y failed out of Z``."""
    cpp_runner = __import__(
        "crucible.common.runners.lang.cpp", fromlist=["main"]
    )

    class _FakeCompleted:
        def __init__(self, rc, out):
            self.returncode = rc
            self.stdout = out
            self.stderr = ""

    calls: list[list[str]] = []

    def fake_run(cmd, **kw):  # noqa: ARG001
        calls.append(list(cmd))
        if cmd[0] == "cmake":
            return _FakeCompleted(0, "cmake build ok\n")
        # ctest
        return _FakeCompleted(
            1,
            "Test #1: foo .......... Passed\n"
            "Test #2: bar .......... Failed\n"
            "75% tests passed, 1 tests failed out of 4\n",
        )

    monkeypatch.setattr(cpp_runner.subprocess, "run", fake_run)
    stdin = io.StringIO(json.dumps({
        "bench_id": "t",
        "parameters": {"build_dir": "build"},
        "env": {},
    }))
    stdout = io.StringIO()
    rc = cpp_runner.main(stdin=stdin, stdout=stdout)

    assert rc == 1  # ctest failed
    assert calls[0][:3] == ["cmake", "--build", "build"]
    assert calls[1][:3] == ["ctest", "--test-dir", "build"]
    v = json.loads(stdout.getvalue())["values"]
    assert v["tests_failed"] == 1.0
    assert v["tests_passed"] == 3.0  # 4 total - 1 failed
    assert v["cpp_build_returncode"] == 0.0
    assert v["cpp_ctest_returncode"] == 1.0


def test_cpp_runner_short_circuits_on_build_failure(monkeypatch):
    """When cmake fails, ctest is not attempted."""
    cpp_runner = __import__(
        "crucible.common.runners.lang.cpp", fromlist=["main"]
    )

    class _FakeCompleted:
        def __init__(self, rc, out):
            self.returncode = rc
            self.stdout = out
            self.stderr = ""

    calls = []

    def fake_run(cmd, **kw):  # noqa: ARG001
        calls.append(list(cmd))
        return _FakeCompleted(2, "cmake error\n")

    monkeypatch.setattr(cpp_runner.subprocess, "run", fake_run)
    stdin = io.StringIO(json.dumps({
        "bench_id": "t",
        "parameters": {"build_dir": "build"},
        "env": {},
    }))
    stdout = io.StringIO()
    rc = cpp_runner.main(stdin=stdin, stdout=stdout)
    assert rc == 2
    assert len(calls) == 1  # ctest skipped
    v = json.loads(stdout.getvalue())["values"]
    assert v["cpp_build_returncode"] == 2.0
    assert v["cpp_ctest_returncode"] == -1.0


# ----------------------------------------------------------------------
# Polyglot dispatcher
# ----------------------------------------------------------------------


def test_polyglot_dispatches_all_languages_when_all_succeed(tmp_path: Path):
    """Two stub languages both exit 0 → polyglot exits 0 and aggregates."""
    # Use subprocess to exercise the real ``python -m crucible.common.runners.polyglot``
    # path with two synthetic mini-runners hosted in a tmp module.
    cfg = {
        "bench_id": "poly_smoke",
        "parameters": {
            "languages": ["python"],  # known language
            "per_language": {
                "python": {
                    "test_command": [
                        sys.executable, "-c",
                        "import sys; print('1 passed in 0.01s'); sys.exit(0)",
                    ],
                },
            },
        },
        "env": {},
    }
    completed = subprocess.run(
        [sys.executable, "-m", "crucible.common.runners.polyglot"],
        input=json.dumps(cfg).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.decode("utf-8"))
    v = payload["values"]
    # Single-language aggregated value.
    assert v["tests_passed"] == 1.0
    assert v["returncode"] == 0.0


def test_polyglot_short_circuits_on_first_failure():
    """Polyglot stops after the first non-zero exit."""
    # python runner exits non-zero; second language (cpp) should NOT
    # be attempted. We detect this by passing a deliberately broken
    # cpp setup that would error LOUDLY if it ran — but here we
    # simply assert the returncode + that only one language is in the
    # raw_output.
    cfg = {
        "bench_id": "poly_fail",
        "parameters": {
            "languages": ["python", "rust"],
            "per_language": {
                "python": {
                    "test_command": [
                        sys.executable, "-c",
                        "import sys; print('boom'); sys.exit(7)",
                    ],
                },
                # rust would invoke cargo which isn't necessarily on
                # PATH; if reached, the test would still observe a
                # non-zero exit but raw_output wouldn't carry "rust".
            },
        },
        "env": {},
    }
    completed = subprocess.run(
        [sys.executable, "-m", "crucible.common.runners.polyglot"],
        input=json.dumps(cfg).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    # Python runner returned 7; polyglot propagates.
    assert completed.returncode == 7
    payload = json.loads(completed.stdout.decode("utf-8"))
    assert "=== python" in payload["raw_output"]
    assert "=== rust" not in payload["raw_output"]


def test_polyglot_types_malformed_breach_child_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.common.runners import polyglot
    from crucible.common.runners.breach_observation import (
        BREACH_OBSERVATION_FAILURE_KEY,
    )

    monkeypatch.setattr(
        polyglot.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            2,
            stdout=b"not-json",
            stderr=b"",
        ),
    )

    result = polyglot._run_language(  # noqa: SLF001
        "python",
        {"parameters": {"breach_observations": True}},
    )
    failure = result[-1]
    assert failure is not None
    assert failure["kind"] == "channel"
    assert BREACH_OBSERVATION_FAILURE_KEY == "breach_observation_failure"


def test_polyglot_keeps_normal_failed_exact_observation_repairable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.common.runners import polyglot

    payload = {
        "values": {"returncode": 1.0},
        "test_observations": [{
            "language": "python",
            "id": "tests/test_breach.py::test_attack",
            "status": "failed",
            "line": "tests/test_breach.py::test_attack",
        }],
    }
    monkeypatch.setattr(
        polyglot.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            1,
            stdout=json.dumps(payload).encode(),
            stderr=b"",
        ),
    )

    result = polyglot._run_language(  # noqa: SLF001
        "python",
        {"parameters": {"breach_observations": True}},
    )
    assert result[0] == 1
    assert result[3][0]["status"] == "failed"
    assert result[-1] is None


def test_polyglot_rejects_unknown_language():
    cfg = {
        "bench_id": "poly_bad",
        "parameters": {"languages": ["fortran"]},
        "env": {},
    }
    completed = subprocess.run(
        [sys.executable, "-m", "crucible.common.runners.polyglot"],
        input=json.dumps(cfg).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout.decode("utf-8"))
    assert "fortran" in payload["raw_output"]
    assert "Supported" in payload["raw_output"]


def test_polyglot_rejects_empty_language_list():
    cfg = {
        "bench_id": "poly_empty",
        "parameters": {"languages": []},
        "env": {},
    }
    completed = subprocess.run(
        [sys.executable, "-m", "crucible.common.runners.polyglot"],
        input=json.dumps(cfg).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2


# ----------------------------------------------------------------------
# LanguagesConstraint registration + _build_verify resolution
# ----------------------------------------------------------------------


def test_languages_constraint_registered_in_canonical_kinds():
    """Importing crucible.workflows transitively registers LanguagesConstraint."""
    import crucible.workflows  # noqa: F401 — side-effect import
    from assay.canonical.registry import CONSTRAINT_TYPES
    assert "languages" in CONSTRAINT_TYPES
    cls = CONSTRAINT_TYPES["languages"]
    instance = cls(languages=("python", "rust"))
    assert instance.kind == "languages"
    assert instance.languages == ("python", "rust")


def test_languages_constraint_loads_from_yaml():
    """Spec YAML with `kind: languages` constructs a LanguagesConstraint."""
    import crucible.workflows  # noqa: F401 — side-effect import
    from assay import Spec

    spec = Spec.from_yaml(textwrap.dedent("""
        schema_version: 1
        project_context:
          project: test_polyglot
          repo_url: https://example.invalid/.git
          base_commit: abc
          branch: main
          submodule_path: null
          runners: {}
        intent: x
        constraints:
          - kind: languages
            languages: [python, rust]
        done_criteria: []
    """).lstrip())

    assert len(spec.constraints) == 1
    c = spec.constraints[0]
    assert c.kind == "languages"
    # YAML loader keeps lists; ``_resolve_languages`` normalizes both
    # list and tuple forms. Assert via tuple() to handle either.
    assert tuple(c.languages) == ("python", "rust")


def test_yaml_spec_runner_parameters_reach_polyglot_as_mutable_lists():
    """Frozen canonical tuples must project into the JSON runner envelope."""
    import crucible.workflows  # noqa: F401 — side-effect import
    from assay import Spec
    from crucible.common.runners.lang.rust import _build_cmd
    from crucible.common.runners.polyglot import _per_language_params
    from crucible.workflows._build_verify import (
        _resolve_languages,
        runner_parameters,
    )

    spec = Spec.from_yaml(textwrap.dedent("""
        schema_version: 1
        project_context:
          project: frozen_runner_parameters
          repo_url: https://example.invalid/.git
          base_commit: abc
          branch: main
          submodule_path: null
          runners:
            build: crucible.common.runners.polyglot
            verify: crucible.common.runners.polyglot
          runner_parameters:
            per_language:
              rust:
                crate: fabric-render-ovrtx
                cargo_features: [ovrtx]
                cargo_test_args: ["--", "--test-threads=1"]
        intent: x
        constraints:
          - kind: languages
            languages: [rust]
        done_criteria: []
    """).lstrip())

    canonical_rust = spec.project_context.runner_parameters[
        "per_language"
    ]["rust"]
    assert tuple(canonical_rust["cargo_features"]) == ("ovrtx",)
    resolved = runner_parameters(
        spec,
        languages=_resolve_languages(spec),
    )
    assert resolved["per_language"]["rust"]["cargo_features"] == [
        "ovrtx",
    ]
    projected = _per_language_params(resolved, "rust")
    assert _build_cmd({**projected, "role": "build"}) == [
        "cargo", "build", "-p", "fabric-render-ovrtx",
        "--features", "ovrtx",
    ]


def test_build_verify_resolves_languages_from_constraint():
    """_resolve_languages picks LanguagesConstraint when set."""
    from assay import ProjectContext, Spec
    from crucible.constraints import LanguagesConstraint
    from crucible.workflows._build_verify import _resolve_languages

    spec = Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="t", repo_url="x", base_commit="0",
            branch=None, submodule_path=None, runners={},
        ),
        execution=None, intent="t",
        constraints=(LanguagesConstraint(languages=("python", "rust")),),
        done_criteria=(), plan_policy=None,
    )
    assert _resolve_languages(spec) == ("python", "rust")


def test_build_verify_slice_override_beats_spec_default():
    """ImplementationDirective.blast_radius.languages overrides Spec default."""
    from assay import ImplementationDirective, ProjectContext, Spec
    from crucible.constraints import LanguagesConstraint
    from crucible.workflows._build_verify import _resolve_languages

    spec = Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="t", repo_url="x", base_commit="0",
            branch=None, submodule_path=None, runners={},
        ),
        execution=None, intent="t",
        constraints=(LanguagesConstraint(languages=("python",)),),
        done_criteria=(), plan_policy=None,
    )
    directive = ImplementationDirective(
        slice_id="s0", intent="x",
        files_in_scope=(), creates_paths=(),
        verify_mode="tests",
        blast_radius={"languages": ["rust", "cpp"]},
    )
    # Slice override wins over the Spec-wide languages=python default.
    assert _resolve_languages(spec, directive) == ("rust", "cpp")


def test_build_verify_no_languages_returns_empty():
    from assay import ProjectContext, Spec
    from crucible.workflows._build_verify import _resolve_languages

    spec = Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="t", repo_url="x", base_commit="0",
            branch=None, submodule_path=None, runners={},
        ),
        execution=None, intent="t",
        constraints=(), done_criteria=(), plan_policy=None,
    )
    assert _resolve_languages(spec) == ()
