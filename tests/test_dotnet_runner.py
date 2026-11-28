# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""One .NET gate must prove the selected suite, even inside a polyglot run."""

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from crucible.common.runners.lang import dotnet


def invoke(params):
    output = io.StringIO()
    rc = dotnet.main(io.StringIO(json.dumps({"parameters": params})), output)
    payload = json.loads(output.getvalue())
    assert payload["values"]["returncode"] == rc
    return rc, payload


@pytest.fixture
def module(tmp_path):
    root = tmp_path / "service with spaces"
    project = root / "tests" / "Workers.csproj"
    assembly = project.parent / "bin" / "Debug" / "net8.0" / "ActualTestAssembly.dll"
    assembly.parent.mkdir(parents=True)
    assembly.touch()
    project.touch()
    solution = root / "Warehouse.slnx"
    solution.touch()
    params = {"working_dir": str(tmp_path), "per_language": {"dotnet": {
        "project_subdir": root.name, "project": solution.name,
        "test_project": "tests/Workers.csproj", "target_framework": "net8.0"}}}
    return root, project, assembly, params


def trx(assembly, outcomes, date=None):
    when = (date or datetime.now(timezone.utc)).isoformat()
    root = ET.Element("TestRun", {"xmlns": dotnet._TRX[1:-1]})
    ET.SubElement(root, "Times", {"start": when, "finish": when})
    definitions = ET.SubElement(root, "TestDefinitions")
    results = ET.SubElement(root, "Results")
    for index, outcome in enumerate(outcomes):
        test = ET.SubElement(definitions, "UnitTest", {"id": str(index), "storage": str(assembly)})
        ET.SubElement(test, "TestMethod", {"codeBase": str(assembly)})
        ET.SubElement(results, "UnitTestResult", {
            "testId": str(index), "executionId": "run-" + str(index), "outcome": outcome})
    passed, failed = outcomes.count("Passed"), outcomes.count("Failed")
    summary = ET.SubElement(root, "ResultSummary", {"outcome": "Failed" if failed else "Completed"})
    ET.SubElement(summary, "Counters", {
        "total": str(len(outcomes)), "passed": str(passed), "failed": str(failed),
        "executed": str(passed + failed)})
    return root


@pytest.mark.parametrize("role", ["verify", "dev_verify"])
def test_attributable_project_and_framework_with_private_results(module, monkeypatch, role):
    root, project, assembly, params = module
    commands = []
    directories = []

    def run(argv, **opts):
        commands.append(argv)
        assert opts["cwd"] == str(root)
        if argv[1] == "test":
            assert argv[:5] == ["dotnet", "test", str(project), "--framework", "net8.0"]
            assert argv[argv.index("--logger") + 1] == "trx"
            results = Path(argv[argv.index("--results-directory") + 1])
            assert list(results.iterdir()) == []
            directories.append(results)
            ET.ElementTree(trx(assembly, ["Passed", "NotExecuted"])).write(results / "run.trx")
            return SimpleNamespace(returncode=0, stdout="selected suite", stderr="")
        assert argv[1:3] == ["msbuild", str(project)]
        assert "-property:TargetFramework=net8.0" in argv
        return SimpleNamespace(returncode=0, stdout=str(assembly) + "\n", stderr="")

    monkeypatch.setattr(dotnet.subprocess, "run", run)
    rc, payload = invoke({**params, "role": role, "runner_mode": "tests"})
    assert rc == 0 and payload["runner_mode"] == "tests"
    assert payload["values"]["tests_passed"] == payload["values"]["dotnet_tests_passed"] == 1
    assert payload["values"]["tests_skipped"] == 1
    assert len(commands) == 2 and not directories[0].exists()


@pytest.mark.parametrize("problem", [
    "absent", "extra", "stale", "malformed", "wrong-root", "empty", "skipped", "failed",
    "sibling", "other-framework", "wrong-output-framework", "mismatched-counters",
    "unknown-result", "duplicate-execution", "unmatched-definition", "aborted", "run-error",
    "partially-aborted", "sdk-failed",
])
def test_success_requires_complete_matching_result_evidence(module, monkeypatch, problem):
    _, _, assembly, params = module
    # This same-named assembly in another project must never count for Workers.csproj.
    sibling = assembly.parents[4] / "sibling" / "bin" / "Debug" / "net8.0" / assembly.name
    sibling.parent.mkdir(parents=True)
    sibling.touch()
    alternate = assembly.parent.parent / "net9.0" / assembly.name
    alternate.parent.mkdir()
    alternate.touch()

    def run(argv, **opts):
        if argv[1] == "msbuild":
            output = alternate if problem == "wrong-output-framework" else assembly
            return SimpleNamespace(returncode=0, stdout=str(output), stderr="")
        directory = Path(argv[argv.index("--results-directory") + 1])
        source = {"sibling": sibling, "other-framework": alternate,
                  "wrong-output-framework": alternate}.get(problem, assembly)
        outcomes = {"empty": [], "skipped": ["NotExecuted"], "failed": ["Passed", "Failed"],
                    "unknown-result": ["Odd"]}.get(problem, ["Passed", "Passed"])
        old = datetime.now(timezone.utc) - timedelta(hours=1) if problem == "stale" else None
        doc = trx(source, outcomes, old)
        if problem == "wrong-root":
            doc.tag = "Other"
        if problem == "mismatched-counters":
            doc.find("ResultSummary/Counters").set("passed", "12")
        if problem == "duplicate-execution":
            doc.find("Results")[1].set("executionId", "run-0")
        if problem == "unmatched-definition":
            doc.find("Results")[0].set("testId", "not-in-this-suite")
        if problem == "aborted":
            doc.find("ResultSummary").set("outcome", "Aborted")
        if problem == "run-error":
            infos = ET.SubElement(doc.find("ResultSummary"), "RunInfos")
            ET.SubElement(infos, "RunInfo", {"outcome": "Error"})
        if problem == "partially-aborted":
            doc.find("ResultSummary/Counters").set("passedButRunAborted", "1")
        if problem != "absent":
            ET.ElementTree(doc).write(directory / "run.trx")
        if problem == "extra":
            ET.ElementTree(doc).write(directory / "second.trx")
        if problem == "malformed":
            (directory / "run.trx").write_text("<broken>")
        return SimpleNamespace(returncode=7 if problem == "sdk-failed" else 0,
                               stdout="100 tests passed", stderr="")

    monkeypatch.setattr(dotnet.subprocess, "run", run)
    rc, payload = invoke(params)
    assert rc != 0
    if problem == "failed":
        assert payload["values"]["tests_failed"] == 1
    if problem == "sdk-failed":
        assert rc == 7


@pytest.mark.parametrize("explicit", ["solution", "test_project", "root"])
def test_build_runs_sdk_and_preserves_exit(module, monkeypatch, explicit):
    root, project, _, params = module
    local = params["per_language"]["dotnet"]
    target = root / "Warehouse.slnx"
    if explicit != "solution":
        del local["project"]
    if explicit == "test_project":
        target = project
    if explicit == "root":
        del local["test_project"]

    def run(argv, **opts):
        assert argv == ["dotnet", "build", str(target)]
        assert opts["cwd"] == str(root)
        return SimpleNamespace(returncode=7, stdout="", stderr="SDK build failed")

    monkeypatch.setattr(dotnet.subprocess, "run", run)
    rc, payload = invoke({**params, "role": "build", "runner_mode": "build"})
    assert rc == payload["values"]["dotnet_build_returncode"] == 7
    assert "tests_passed" not in payload["values"]


@pytest.mark.parametrize("invalid", ["missing-project", "solution-as-suite", "missing-framework",
                                     "option-as-framework", "traversal", "absolute", "ambiguous",
                                     "bench", "exact", "invalid-role"])
def test_invalid_selection_never_invokes_sdk(module, monkeypatch, invalid):
    root, project, _, params = module
    local = params["per_language"]["dotnet"]
    if invalid == "missing-project":
        del local["test_project"]
    elif invalid == "solution-as-suite":
        local["test_project"] = "Warehouse.slnx"
    elif invalid == "missing-framework":
        del local["target_framework"]
    elif invalid == "option-as-framework":
        local["target_framework"] = "--help"
    elif invalid == "traversal":
        local["test_project"] = "../tests/Workers.csproj"
    elif invalid == "absolute":
        local["test_project"] = str(project)
    elif invalid == "ambiguous":
        (root / "Second.csproj").touch()
        del local["test_project"], local["project"]
        params["role"] = "build"
    elif invalid == "bench":
        params["runner_mode"] = "bench"
    elif invalid == "exact":
        params["breach_observations"] = True
    elif invalid == "invalid-role":
        params["role"] = "benchmark"
    monkeypatch.setattr(dotnet.subprocess, "run", lambda *a, **kw: pytest.fail("SDK invoked"))
    rc, payload = invoke(params)
    assert rc == 2 and "runner_mode" not in payload
