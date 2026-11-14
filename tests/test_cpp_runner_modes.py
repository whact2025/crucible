# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""CMake builds and CTest verification follow the selected workflow role."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.common.runners.lang import cpp


def _invoke(parameters: dict[str, object]) -> tuple[int, dict[str, object]]:
    output = io.StringIO()
    rc = cpp.main(
        stdin=io.StringIO(json.dumps({
            "parameters": parameters,
            "env": {"CPP_RUNNER_TEST_MARKER": "selected"},
        })),
        stdout=output,
    )
    return rc, json.loads(output.getvalue())


@pytest.mark.parametrize("runner_mode", (None, "build"))
def test_build_role_runs_only_cmake_with_selected_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner_mode: str | None,
) -> None:
    project = tmp_path / "native" / "cpp"
    project.mkdir(parents=True)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command[0] == "cmake", "a build request must never start CTest"
        return SimpleNamespace(returncode=0, stdout="built native target\n", stderr="")

    monkeypatch.setattr(cpp.subprocess, "run", run)
    rc, payload = _invoke({
        "role": "build",
        "runner_mode": runner_mode,
        "working_dir": str(tmp_path),
        "per_language": {"cpp": {
            "project_subdir": "native/cpp",
            "build_dir": "out/native",
            "cmake_target": "native_brain",
            "ctest_args": ["-R", "closed_loop"],
        }},
    })

    assert rc == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == ["cmake", "--build", "out/native", "--target", "native_brain"]
    assert kwargs["cwd"] == str(project)
    assert kwargs["env"]["CPP_RUNNER_TEST_MARKER"] == "selected"
    assert payload["runner_mode"] == "build"
    assert payload["values"]["cpp_build_returncode"] == 0.0
    assert payload["values"]["returncode"] == 0.0
    assert "cpp_ctest_returncode" not in payload["values"]
    assert "tests_passed" not in payload["values"]


@pytest.mark.parametrize("role", ("verify", "dev_verify"))
@pytest.mark.parametrize("skip_build", (False, True))
def test_verification_runs_selected_ctest_and_reports_its_failure(
    monkeypatch: pytest.MonkeyPatch, role: str, skip_build: bool,
) -> None:
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[0] == "cmake":
            return SimpleNamespace(returncode=0, stdout="built\n", stderr="")
        return SimpleNamespace(
            returncode=8,
            stdout="50% tests passed, 1 tests failed out of 2\n",
            stderr="",
        )

    monkeypatch.setattr(cpp.subprocess, "run", run)
    rc, payload = _invoke({
        "role": role,
        "runner_mode": "tests",
        "per_language": {"cpp": {
            "build_dir": "out/native",
            "ctest_args": ["-R", "closed_loop"],
            "skip_build": skip_build,
        }},
    })
    expected = [] if skip_build else [["cmake", "--build", "out/native"]]
    expected.append([
        "ctest", "--test-dir", "out/native", "--output-on-failure",
        "-R", "closed_loop",
    ])
    assert calls == expected
    assert rc == 8
    assert payload["runner_mode"] == "tests"
    assert payload["values"]["tests_passed"] == 1.0
    assert payload["values"]["tests_failed"] == 1.0
    assert payload["values"]["returncode"] == 8.0


@pytest.mark.parametrize("role, mode", (("build", "build"), ("verify", "tests")))
def test_cmake_failure_never_starts_ctest(
    monkeypatch: pytest.MonkeyPatch, role: str, mode: str,
) -> None:
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7, stdout="", stderr="compile failed\n")

    monkeypatch.setattr(cpp.subprocess, "run", run)
    rc, payload = _invoke({"role": role, "runner_mode": mode})
    assert calls == [["cmake", "--build", "build"]]
    assert rc == 7
    assert payload["runner_mode"] == mode
    assert payload["values"]["cpp_ctest_returncode"] == -1.0
    assert payload["values"]["returncode"] == 7.0


@pytest.mark.parametrize("parameters", (
    {"role": "build", "runner_mode": "build", "skip_build": True},
    {"role": "build", "runner_mode": "bench"},
    {"role": "verify", "runner_mode": "bench"},
    {"role": "dev_verify", "runner_mode": "bench"},
))
def test_unsupported_requests_fail_before_command_execution(
    monkeypatch: pytest.MonkeyPatch, parameters: dict[str, object],
) -> None:
    def run(*_args, **_kwargs):
        raise AssertionError("rejected request started a command")

    monkeypatch.setattr(cpp.subprocess, "run", run)
    rc, payload = _invoke(parameters)
    assert rc == 2
    assert payload["values"]["returncode"] == 2.0
