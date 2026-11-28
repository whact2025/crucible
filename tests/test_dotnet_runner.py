# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""One .NET gate must prove the selected suite, even inside a polyglot run."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from crucible.common.runners.lang import dotnet
from crucible.common.runners.parameters import for_slice


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


def declared_trx(assembly):
    doc = trx(assembly, ["Passed", "Passed"])
    entries = ET.SubElement(doc, "TestEntries")
    for definition, result in zip(doc.find("TestDefinitions"), doc.find("Results"), strict=True):
        ET.SubElement(definition, "Execution", {"id": result.get("executionId")})
        ET.SubElement(entries, "TestEntry", {"executionId": result.get("executionId"),
                                              "testId": result.get("testId")})
    return doc


def parse_trx(doc, assembly, tmp_path):
    report = tmp_path / "test.trx"
    ET.ElementTree(doc).write(report)
    now = datetime.now(timezone.utc)
    return dotnet._report(report, assembly, now - timedelta(minutes=1), now)


@pytest.mark.parametrize("path", ["Times", "Results", "ResultSummary", "ResultSummary/Counters",
                                  "TestDefinitions", "TestDefinitions/UnitTest/TestMethod"])
@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_trx_requires_exactly_one_section(module, tmp_path, path, change):
    _, _, assembly, _ = module
    doc = declared_trx(assembly)
    parent_path, _, name = path.rpartition("/")
    parent = doc.find(parent_path) if parent_path else doc
    element = parent.find(name)
    if change == "missing":
        parent.remove(element)
    else:
        duplicate = deepcopy(element)
        if name == "Results":
            duplicate[0].set("outcome", "Failed")
        if name == "ResultSummary":
            duplicate.set("outcome", "Failed")
        if name == "Counters":
            duplicate.set("failed", "1")
        parent.append(duplicate)
    with pytest.raises(ValueError, match=f"TRX {name}"):
        parse_trx(doc, assembly, tmp_path)


@pytest.mark.parametrize("change", ["extra-entry", "missing-entry", "entry-id", "entry-test", "nested-entry",
                                    "duplicate-entries", "definition-id", "missing-id",
                                    "duplicate-execution"])
def test_optional_trx_execution_bindings_must_agree(module, tmp_path, change):
    _, _, assembly, _ = module
    doc = declared_trx(assembly)
    entries = doc.find("TestEntries")
    definition = doc.find("TestDefinitions/UnitTest")
    if change == "extra-entry":
        entries.append(deepcopy(entries[0]))
    elif change == "missing-entry":
        entries.remove(entries[0])
    elif change == "entry-id":
        entries[0].set("executionId", "different-execution")
    elif change == "entry-test":
        entries[0].set("testId", entries[1].get("testId"))
    elif change == "nested-entry":
        entries[0].append(deepcopy(entries[1]))
    elif change == "duplicate-entries":
        doc.append(deepcopy(entries))
    elif change == "definition-id":
        definition.find("Execution").set("id", entries[1].get("executionId"))
    elif change == "missing-id":
        definition.find("Execution").attrib.clear()
    else:
        definition.append(deepcopy(definition.find("Execution")))
    with pytest.raises(ValueError):
        parse_trx(doc, assembly, tmp_path)


@pytest.mark.parametrize("shared_definition", [False, True])
def test_two_flat_data_rows_have_separate_executions(module, tmp_path, shared_definition):
    _, _, assembly, _ = module
    doc = declared_trx(assembly)
    if shared_definition:
        definitions = doc.find("TestDefinitions")
        definitions.remove(definitions[1])
        doc.find("Results")[1].set("testId", definitions[0].get("id"))
        doc.find("TestEntries")[1].set("testId", definitions[0].get("id"))
    counts, clean = parse_trx(doc, assembly, tmp_path)
    assert clean and counts["tests_passed"] == 2


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


@pytest.mark.parametrize("second_evidence", ["failed test", "first project's passing report"])
def test_each_claimed_project_and_framework_needs_its_own_passing_run(tmp_path, monkeypatch, second_evidence):
    # This is the same slice projection used by workflow runner_parameters.
    pairs = {"receive": ("Receiving", "net8.0"), "ship": ("Shipping", "net10.0")}
    per_slice = {}
    assemblies = {}
    for slice_id, (suite, framework) in pairs.items():
        project = tmp_path / suite / f"{suite}.Tests.csproj"
        assembly = project.parent / "bin" / "Debug" / framework / "Rules.dll"
        assembly.parent.mkdir(parents=True)
        assembly.touch()
        project.touch()
        assemblies[project] = assembly
        per_slice[slice_id] = {"per_language": {"dotnet": {
            "project_subdir": suite, "test_project": project.name, "target_framework": framework,
        }}}
    selections = {"per_slice": per_slice}
    observed = []
    first_assembly = assemblies[tmp_path / "Receiving" / "Receiving.Tests.csproj"]

    def sdk(argv, cwd, env):
        project = Path(argv[2])
        assembly = assemblies[project]
        assert cwd == project.parent
        if argv[1] == "msbuild":
            assert f"-property:TargetFramework={assembly.parent.name}" in argv
            return SimpleNamespace(returncode=0, stdout=str(assembly), stderr="")
        assert argv[1] == "test"
        framework = argv[argv.index("--framework") + 1]
        observed.append((project.name, framework))
        assert framework == assembly.parent.name
        outcome = "Passed"
        if project.parent.name == "Shipping":
            if second_evidence == "failed test":
                outcome = "Failed"
            else:
                assembly = first_assembly
        folder = Path(argv[argv.index("--results-directory") + 1])
        ET.ElementTree(trx(assembly, [outcome])).write(folder / "suite.trx")
        return SimpleNamespace(returncode=int(outcome == "Failed"), stdout="", stderr="")

    monkeypatch.setattr(dotnet, "_run", sdk)
    def verify(slice_id):
        return invoke({**for_slice(selections, slice_id), "working_dir": str(tmp_path)})

    assert verify(None)[0] != 0
    first_rc, first = verify("receive")
    second_rc, second = verify("ship")
    assert first_rc == 0 and first["values"]["dotnet_tests_passed"] == 1
    assert second_rc != 0 and second["values"].get("dotnet_tests_passed", 0) == 0
    if second_evidence == "failed test":
        assert second["values"]["dotnet_tests_failed"] == 1
    else:
        assert "expected project/framework assembly" in second["raw_output"]
    assert observed == [("Receiving.Tests.csproj", "net8.0"), ("Shipping.Tests.csproj", "net10.0")]


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
