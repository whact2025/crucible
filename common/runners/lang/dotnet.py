# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Build one SDK target or verify one VSTest C# project and target framework.

Canonical parameters are project_subdir, project (build only), test_project,
and target_framework. Verification uses a private TRX and the selected SDK
project's evaluated TargetPath (MSBuild 17.8+). The supported output layout
distinguishes frameworks under the project directory. No benchmark or exact
breach observation protocol is provided by this adapter.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from crucible.common.runners.parameters import for_language, resolve_project_working_dir


_TRX = "{http://microsoft.com/schemas/VisualStudio/TeamTest/2010}"
_BUILD_SUFFIXES = {".csproj", ".sln", ".slnx"}


def build_command(repo_root: Path) -> str:
    return "dotnet build"


def verify_command(repo_root: Path) -> str:
    return "dotnet test"


def breach_test_path(repo_root: Path) -> Path:
    raise ValueError("The dotnet adapter has no exact breach-test observation provider")


def _target(root: Path, value: object, suffixes: set[str]) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty relative .NET target path is required")
    path = Path(value)
    windows = PureWindowsPath(value)
    if path.is_absolute() or windows.anchor or ".." in path.parts or ".." in windows.parts:
        raise ValueError(f".NET target must be a confined relative path: {value!r}")
    candidate = (root / path).resolve()
    if (not candidate.is_relative_to(root) or candidate.suffix.lower() not in suffixes
            or not candidate.is_file()):
        raise ValueError(f"Missing, unsupported or out-of-project .NET target: {value!r}")
    return candidate


def _build_target(root: Path, params: dict) -> Path:
    for key in ("project", "test_project"):
        if key in params:
            return _target(root, params[key], _BUILD_SUFFIXES if key == "project" else {".csproj"})
    candidates = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _BUILD_SUFFIXES]
    if len(candidates) != 1:
        raise ValueError("Select 'project' explicitly when the SDK root has zero or multiple targets")
    return _target(root, candidates[0].name, _BUILD_SUFFIXES)


def _run(argv, root, env):
    return subprocess.run(argv, cwd=str(root), env=env, capture_output=True,
                          encoding="utf-8", errors="replace", check=False)


def _assembly(project: Path, framework: str, root: Path, env: dict) -> Path:
    proc = _run(["dotnet", "msbuild", str(project), "-nologo", "-getProperty:TargetPath",
                 "-property:Configuration=Debug", f"-property:TargetFramework={framework}"], root, env)
    if proc.returncode:
        raise ValueError(f"MSBuild could not evaluate the test project's TargetPath: {proc.stderr[-1000:]}")
    output = Path(proc.stdout.strip())
    if not output.is_absolute():
        raise ValueError("MSBuild TargetPath must be a single absolute assembly path")
    output = output.resolve()
    if not output.is_relative_to(project.parent) or not output.is_file() or output.suffix.lower() != ".dll":
        raise ValueError("MSBuild TargetPath is not a project-local test assembly")
    if framework.lower() not in (part.lower() for part in output.relative_to(project.parent).parts[:-1]):
        raise ValueError("The test assembly output must distinguish the selected target framework")
    return output


def _report(report: Path, assembly: Path, start: datetime, end: datetime) -> tuple[dict, bool]:
    trx = ET.parse(report).getroot()
    if trx.tag != _TRX + "TestRun":
        raise ValueError("Expected a VSTest TRX TestRun")
    times = trx.find(_TRX + "Times")
    if times is None:
        raise ValueError("Missing TRX run times")
    first = datetime.fromisoformat(times.get("start", ""))
    last = datetime.fromisoformat(times.get("finish", ""))
    if first.tzinfo is None or last.tzinfo is None or not start <= first <= last <= end:
        raise ValueError("TRX times fall outside this test invocation")
    definitions = {}
    for definition in trx.findall(f"{_TRX}TestDefinitions/{_TRX}UnitTest"):
        test_id = definition.get("id")
        method = definition.find(_TRX + "TestMethod")
        source = Path(method.get("codeBase", "")) if method is not None else Path()
        if not test_id or test_id in definitions or not source.is_absolute() or source.resolve() != assembly:
            raise ValueError("TRX test definitions do not identify the expected project/framework assembly")
        definitions[test_id] = source
    results = trx.find(_TRX + "Results")
    summary = trx.find(_TRX + "ResultSummary")
    counters = summary.find(_TRX + "Counters") if summary is not None else None
    if results is None or counters is None or not len(results):
        raise ValueError("Missing TRX results or counters")
    executions = set()
    outcomes = Counter()
    for row in results:
        execution = row.get("executionId")
        outcome = row.get("outcome")
        if (row.tag != _TRX + "UnitTestResult" or row.find(_TRX + "InnerResults") is not None
                or row.get("testId") not in definitions or not execution or execution in executions
                or outcome not in {"Passed", "Failed", "NotExecuted"}):
            raise ValueError("Unsupported, duplicated or unattributable TRX result")
        executions.add(execution)
        outcomes[outcome] += 1
    passed, failed, skipped = (outcomes[k] for k in ("Passed", "Failed", "NotExecuted"))
    for key, expected in {"total": len(executions), "executed": passed + failed,
                          "passed": passed, "failed": failed}.items():
        if int(counters.get(key, "-1")) != expected:
            raise ValueError(f"TRX counter {key!r} disagrees with its observed test rows")
    if not passed + failed:
        raise ValueError("No .NET test executed for the selected project and target framework")
    clean = summary.get("outcome") in {"Passed", "Completed"} and not failed
    for key in ("error", "timeout", "aborted", "inconclusive", "passedButRunAborted",
                "notRunnable", "disconnected", "inProgress", "pending"):
        if int(counters.get(key, "0")):
            clean = False
    if any(info.get("outcome") in {"Error", "Failed", "Aborted", "Timeout"}
           for info in summary.findall(f"{_TRX}RunInfos/{_TRX}RunInfo")):
        clean = False
    return {"tests_passed": passed, "tests_failed": failed, "tests_skipped": skipped}, clean


def _test(root: Path, params: dict, env: dict, values: dict) -> tuple[int, str]:
    project = _target(root, params.get("test_project"), {".csproj"})
    framework = params.get("target_framework")
    if not isinstance(framework, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", framework):
        raise ValueError("Verification requires one explicit 'target_framework'")
    with tempfile.TemporaryDirectory(prefix="crucible-dotnet-") as work:
        directory = Path(work)
        start = datetime.now(timezone.utc)
        proc = _run(["dotnet", "test", str(project), "--framework", framework, "--configuration", "Debug",
                     "--logger", "trx", "--results-directory", work], root, env)
        end = datetime.now(timezone.utc)
        values["dotnet_test_returncode"] = float(proc.returncode)
        output = proc.stdout + proc.stderr
        reports = [p for p in directory.rglob("*") if p.suffix.lower() == ".trx"]
        if (len(reports) != 1 or reports[0].parent != directory or reports[0].is_symlink()
                or not reports[0].is_file()):
            return proc.returncode or 2, output + "\nExpected exactly one fresh VSTest TRX report"
        try:
            assembly = _assembly(project, framework, root, env)
            counts, clean = _report(reports[0], assembly, start, end)
        except (OSError, ValueError, OverflowError, ET.ParseError) as exc:
            return proc.returncode or 2, output + "\n" + str(exc)
        for key, value in counts.items():
            values[key] = values[f"dotnet_{key}"] = float(value)
        if not clean:
            return proc.returncode or 1, output + "\nThe VSTest run reported unsuccessful tests or run status"
        return proc.returncode, output


def main(stdin=None, stdout=None) -> int:
    start = time.monotonic()
    values = {}
    mode = None
    try:
        cfg = json.loads((stdin or sys.stdin).read())
        params = for_language(cfg.get("parameters", {}), "dotnet")
        role = params.get("role", "verify")
        requested = params.get("runner_mode")
        expected = "build" if role == "build" else "tests"
        if role not in {"build", "verify", "dev_verify"} or requested not in (None, expected):
            raise ValueError(f"Unsupported dotnet role={role!r}, runner_mode={requested!r}")
        if expected == "tests" and params.get("breach_observations"):
            raise ValueError("The dotnet runner does not support exact breach observations")
        root = resolve_project_working_dir(params, process_cwd=os.getcwd())
        env = {**os.environ, **(cfg.get("env") or {})}
        if role == "build":
            project = _build_target(root, params)
            proc = _run(["dotnet", "build", str(project)], root, env)
            rc, output = proc.returncode, proc.stdout + proc.stderr
            values["dotnet_build_returncode"] = float(rc)
        else:
            rc, output = _test(root, params, env, values)
        mode = expected
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        rc, output = 2, str(exc)
    values["returncode"] = values["dotnet_returncode"] = float(rc)
    values["duration_s"] = values["dotnet_duration_s"] = time.monotonic() - start
    payload = {"values": values, "raw_output": output[-4000:]}
    if mode:
        payload["runner_mode"] = mode
    stream = stdout or sys.stdout
    stream.write(json.dumps(payload))
    stream.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
